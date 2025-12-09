"""HDF5 storage for pairwise comparison results.

This module provides storage and retrieval of comparison results by wrapping
io.py functions. Maintains single source of responsibility for HDF5 operations.

NOTE: This module delegates actual HDF5 I/O to neural_analysis.utils.io.
      Use io.py directly for general HDF5 operations.

Storage Schema
--------------
Comparisons stored via io.save_result_to_hdf5_dataset():

    /{comparison_name}/{result_key}/
        scalars: {metric, dataset_i, dataset_j, mode, value, ...metadata}
        arrays: {pair_indices, pair_values}  # For shape metrics

Features
--------
- **Delegated I/O**: Uses io.py (single source of responsibility)
- **Metadata tracking**: mode, metric, timestamps, sample counts
- **Query support**: Filter by metric, mode, dataset via pandas
- **Type safety**: Explicit value_type tracking
- **Auto-save/load**: General caching wrapper for any computation

Examples
--------
>>> import numpy as np
>>> from neural_analysis.metrics import compute_between_distances
>>> from neural_analysis.utils.comparison_store import (
...     save_comparison_result, load_comparison_result
... )
>>>
>>> # Compute and save
>>> data1, data2 = np.random.randn(100, 10), np.random.randn(80, 10)
>>> result = compute_between_distances(data1, data2, metric="euclidean")
>>>
>>> save_comparison_result(
...     filepath="results.h5",
...     comparison_name="exp001",
...     dataset_i="control",
...     dataset_j="treatment",
...     metric="euclidean",
...     mode="between",
...     value=result,
...     metadata={"n_samples_i": 100}
... )
>>>
>>> # Load
>>> loaded = load_comparison_result(
...     filepath="results.h5",
...     comparison_name="exp001",
...     dataset_i="control",
...     dataset_j="treatment",
...     metric="euclidean"
... )
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, TypeVar

import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd

from neural_analysis.utils.logging import get_logger

__all__ = [
    "save_comparison",
    "load_comparison",
    "query_comparisons",
    "try_load_cached_comparison",
    "save_comparison_result",
]

logger = get_logger(__name__)

# HDF5 storage configuration
COMPRESSION = "gzip"
COMPRESSION_LEVEL = 6
SHUFFLE_FILTER = True
CHUNK_SIZE_THRESHOLD = 10_000  # Elements threshold for chunking

# Type variable for return type
T = TypeVar("T")


def _infer_value_type(value: Any) -> str:
    """Infer value type for storage metadata.

    Parameters
    ----------
    value : Any
        Value to classify (float, ndarray, or dict)

    Returns
    -------
    str
        Value type: "scalar", "matrix", or "dict"
    """
    if isinstance(value, dict):
        return "dict"
    elif isinstance(value, np.ndarray):
        return "matrix"
    elif isinstance(value, (int, float, np.number)):
        return "scalar"
    elif isinstance(value, tuple):
        # For shape metrics: (distance, pairs_dict)
        return "scalar"  # Store distance as scalar, pairs separately if needed
    else:
        raise TypeError(
            f"Unsupported value type: {type(value)}. Expected float, ndarray, or dict"
        )


def _encode_dict_for_hdf5(d: dict[str, dict[str, float]]) -> npt.NDArray[np.void]:
    """Encode nested dict to structured array for HDF5 storage.

    Parameters
    ----------
    d : dict[str, dict[str, float]]
        Nested dictionary (e.g., {dataset_i: {dataset_j: distance}})

    Returns
    -------
    ndarray
        Structured array with fields: (key_i, key_j, value)
        String fields are stored as bytes (S100) for HDF5 compatibility
    """
    from neural_analysis.utils.io import _to_bytes_array
    
    records = []
    for key_i, inner_dict in d.items():
        for key_j, value in inner_dict.items():
            records.append((str(key_i), str(key_j), float(value)))

    # Use bytes dtype (S100) instead of Unicode (U100) for HDF5 compatibility
    dtype = [("key_i", "S100"), ("key_j", "S100"), ("value", "f8")]
    return np.array(records, dtype=dtype)


def _decode_dict_from_hdf5(arr: npt.NDArray[np.void]) -> dict[str, dict[str, float]]:
    """Decode structured array back to nested dict.

    Parameters
    ----------
    arr : ndarray
        Structured array from HDF5 (with bytes string fields)

    Returns
    -------
    dict[str, dict[str, float]]
        Nested dictionary
    """
    from neural_analysis.utils.io import _from_bytes_array
    
    result: dict[str, dict[str, float]] = {}
    for record in arr:
        # Handle both bytes (S) and Unicode (U) string types
        key_i_bytes = record["key_i"]
        key_j_bytes = record["key_j"]
        
        if isinstance(key_i_bytes, bytes):
            key_i = key_i_bytes.decode("utf-8").rstrip("\x00")
        else:
            key_i = str(key_i_bytes)
            
        if isinstance(key_j_bytes, bytes):
            key_j = key_j_bytes.decode("utf-8").rstrip("\x00")
        else:
            key_j = str(key_j_bytes)
            
        value = float(record["value"])

        if key_i not in result:
            result[key_i] = {}
        result[key_i][key_j] = value

    return result


def save_comparison(
    filepath: str | Path,
    metric: str,
    dataset_i: str,
    dataset_j: str,
    mode: str,
    value: float | npt.NDArray[np.floating] | dict[str, dict[str, float]],
    metadata: dict[str, Any] | None = None,
    overwrite: bool = False,
    use_cache: bool = True,
) -> None:
    """Save a comparison result to HDF5 (delegates to io.py backend).

    **RECOMMENDED**: Use io.save_result_to_hdf5_dataset() directly for new code.

    This function provides comparison-specific formatting, then delegates to
    io.save_result_to_hdf5_dataset() for actual HDF5 operations.
    Optionally uses Redis cache and SQL metadata indexing.

    Parameters
    ----------
    filepath : str or Path
        Path to HDF5 file (will be created if doesn't exist)
    metric : str
        Metric name (e.g., "euclidean", "wasserstein")
    dataset_i : str
        First dataset identifier
    dataset_j : str
        Second dataset identifier
    mode : {"within", "between", "all-pairs"}
        Comparison mode
    value : float, ndarray, or dict
        Comparison result (type determines storage format)
    metadata : dict, optional
        Additional metadata to store as attributes
    overwrite : bool, default=False
        If True, overwrite existing comparison; if False, raise error
    use_cache : bool, default=True
        Whether to cache in Redis (if available)

    Raises
    ------
    ValueError
        If comparison already exists and overwrite=False
    TypeError
        If value type is not supported

    Examples
    --------
    >>> save_comparison(
    ...     filepath="results.h5",
    ...     metric="euclidean",
    ...     dataset_i="exp1",
    ...     dataset_j="exp2",
    ...     mode="between",
    ...     value=42.5,
    ...     metadata={"n_samples_i": 100, "n_samples_j": 80}
    ... )
    """
    from neural_analysis.utils.io import save_result_to_hdf5_dataset

    # Infer value type
    value_type = _infer_value_type(value)

    # Invalidate cache if overwriting
    if overwrite and use_cache:
        try:
            from neural_analysis.utils.storage.manager import StorageManager

            storage_manager = StorageManager()
            cache_key = f"{metric}:{dataset_i}:{dataset_j}"
            storage_manager.invalidate_cache(f"*{cache_key}*")
        except Exception:
            # Cache unavailable, continue
            pass

    # Build hierarchical key: metric/dataset_i/dataset_j
    dataset_name = metric
    result_key = f"{dataset_i}___{dataset_j}"

    logger.info(
        f"Saving comparison: {dataset_name}/{result_key}, mode={mode}, type={value_type}"
    )

    # Check if comparison exists (if overwrite=False)
    if not overwrite:
        filepath_obj = Path(filepath)
        if filepath_obj.exists():
            try:
                with h5py.File(filepath_obj, "r") as f:
                    if dataset_name in f and result_key in f[dataset_name]:
                        raise ValueError(
                            f"Comparison already exists: {dataset_name}/{result_key}. "
                            "Set overwrite=True to replace."
                        )
            except (OSError, IOError):
                # File exists but is corrupted or not a valid HDF5 file
                # Continue to overwrite it
                pass

    # Prepare scalar_data and array_data for io.py backend
    scalar_data: dict[str, Any] = {
        "mode": mode,
        "metric": metric,
        "dataset_i": dataset_i,
        "dataset_j": dataset_j,
        "timestamp": datetime.now(UTC).isoformat(),
        "value_type": value_type,
    }

    if metadata is not None:
        scalar_data.update(metadata)

    array_data: dict[str, npt.NDArray[Any]] = {}

    if value_type == "scalar":
        scalar_data["value"] = float(value)  # type: ignore[arg-type]
    elif value_type == "matrix":
        array_data["value"] = np.asarray(value, dtype=np.float64)
    elif value_type == "dict":
        # Encode dict as structured array
        array_data["value"] = _encode_dict_for_hdf5(value)  # type: ignore[arg-type]
    else:
        raise TypeError(f"Unexpected value_type: {value_type}")

    # Delegate to io.py backend (with caching/indexing)
    save_result_to_hdf5_dataset(
        save_path=filepath,
        dataset_name=dataset_name,
        result_key=result_key,
        scalar_data=scalar_data,
        array_data=array_data,
        compression=COMPRESSION,
        use_cache=use_cache,
        use_sql_index=True,
    )

    logger.info(
        f"Successfully saved comparison to {filepath}:{dataset_name}/{result_key}"
    )


def load_comparison(
    filepath: str | Path,
    metric: str,
    dataset_i: str,
    dataset_j: str,
) -> float | npt.NDArray[np.floating] | dict[str, dict[str, float]]:
    """Load a specific comparison from HDF5 file (delegates to io.py backend).

    **RECOMMENDED**: Use io.load_results_from_hdf5_dataset() directly for new code.

    This function provides comparison-specific formatting, then delegates to
    io.load_results_from_hdf5_dataset() for actual HDF5 operations.

    Parameters
    ----------
    filepath : str or Path
        Path to HDF5 file
    metric : str
        Metric name
    dataset_i : str
        First dataset identifier
    dataset_j : str
        Second dataset identifier

    Returns
    -------
    float, ndarray, or dict
        Comparison result (type depends on value_type attribute)

    Raises
    ------
    FileNotFoundError
        If HDF5 file doesn't exist
    KeyError
        If comparison not found in file

    Examples
    --------
    >>> result = load_comparison(
    ...     filepath="results.h5",
    ...     metric="euclidean",
    ...     dataset_i="exp1",
    ...     dataset_j="exp2"
    ... )
    """
    from neural_analysis.utils.io import load_results_from_hdf5_dataset

    filepath_obj = Path(filepath)
    if not filepath_obj.exists():
        raise FileNotFoundError(f"HDF5 file not found: {filepath}")

    # Build hierarchical key: metric/dataset_i/dataset_j
    dataset_name = metric
    result_key = f"{dataset_i}___{dataset_j}"

    logger.info(f"Loading comparison from {filepath}:{dataset_name}/{result_key}")

    # Load using io.py backend
    results = load_results_from_hdf5_dataset(
        save_path=filepath,
        dataset_name=dataset_name,
        result_key=result_key,
    )

    if (
        not results
        or dataset_name not in results
        or result_key not in results[dataset_name]
    ):
        raise KeyError(f"Comparison not found: {filepath}:{dataset_name}/{result_key}")

    entry = results[dataset_name][result_key]

    # Reconstruct value based on value_type
    # load_results_from_hdf5_dataset returns {"attributes": {...}, "arrays": {...}}
    attrs = entry.get("attributes", {})
    arrays = entry.get("arrays", {})
    
    value_type = attrs.get("value_type", "scalar")
    if value_type == "scalar":
        return float(attrs["value"])
    elif value_type == "matrix":
        return np.asarray(arrays["value"], dtype=np.float64)
    elif value_type == "dict":
        return _decode_dict_from_hdf5(arrays["value"])
    else:
        raise TypeError(f"Unknown value_type: {value_type}")


def query_comparisons(
    filepath: str | Path,
    metric: str | None = None,
    mode: str | None = None,
    dataset_i: str | None = None,
    dataset_j: str | None = None,
) -> pd.DataFrame:
    """Query stored comparisons with filters.

    Parameters
    ----------
    filepath : str or Path
        Path to HDF5 file
    metric : str, optional
        Filter by metric name
    mode : str, optional
        Filter by mode ("within", "between", "all-pairs")
    dataset_i : str, optional
        Filter by first dataset name
    dataset_j : str, optional
        Filter by second dataset name

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns: comparison_name, metric, mode, dataset_i,
        dataset_j, value_type, timestamp, etc.

    Examples
    --------
    >>> # Find all Wasserstein comparisons
    >>> df = query_comparisons("results.h5", metric="wasserstein")
    >>>
    >>> # Find comparisons involving "control" dataset
    >>> df = query_comparisons("results.h5", dataset_i="control")
    """
    from neural_analysis.utils.io import load_results_from_hdf5_dataset

    filepath_obj = Path(filepath)
    if not filepath_obj.exists():
        return pd.DataFrame()

    # Load all results
    all_results = load_results_from_hdf5_dataset(save_path=filepath)

    rows = []
    for dataset_name, result_dict in all_results.items():
        for result_key, entry in result_dict.items():
            scalars = entry.get("scalars", {})
            row = {
                "comparison_name": dataset_name,
                "result_key": result_key,
                "metric": scalars.get("metric"),
                "mode": scalars.get("mode"),
                "dataset_i": scalars.get("dataset_i"),
                "dataset_j": scalars.get("dataset_j"),
                "value_type": scalars.get("value_type"),
                "timestamp": scalars.get("timestamp"),
            }
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Apply filters
    if metric is not None:
        df = df[df["metric"] == metric]
    if mode is not None:
        df = df[df["mode"] == mode]
    if dataset_i is not None:
        df = df[df["dataset_i"] == dataset_i]
    if dataset_j is not None:
        df = df[df["dataset_j"] == dataset_j]

    return df


def try_load_cached_comparison(
    save_path: str | Path,
    mode: str,
    metric: str,
    dataset_names: tuple[str, str] | None = None,
) -> Any | None:
    """Try to load a cached comparison result.

    This is a helper function for auto-save/load logic that attempts to load
    cached results based on mode and metric.

    Parameters
    ----------
    save_path : str or Path
        Path to HDF5 file
    mode : {"between", "all-pairs"}
        Comparison mode (within mode not supported)
    metric : str
        Metric name
    dataset_names : tuple[str, str], optional
        Dataset names for between mode. Required for mode="between".

    Returns
    -------
    Any or None
        Cached result if found, None otherwise

    Raises
    ------
    ValueError
        If mode="between" and dataset_names is None
    """
    save_path_obj = Path(save_path)
    if not save_path_obj.exists():
        return None

    try:
        if mode == "between":
            if dataset_names is None:
                raise ValueError(
                    "dataset_names required for save_path with mode='between'. "
                    "Provide tuple like ('control', 'treatment')"
                )
            dataset_i, dataset_j = dataset_names
            cached_result = load_comparison(save_path_obj, metric, dataset_i, dataset_j)
            logger.info(
                f"Successfully loaded cached result from {save_path}: "
                f"{metric}/{dataset_i}___{dataset_j}"
            )
            return cached_result
        elif mode == "all-pairs":
            # For all-pairs, we use a special dataset pair naming
            cached_result = load_comparison(
                save_path_obj, metric, "all_pairs", "all_pairs"
            )
            logger.info(f"Successfully loaded cached all-pairs result from {save_path}")
            return cached_result
        # Within mode doesn't support save_path (single dataset)
        return None
    except (FileNotFoundError, KeyError, OSError, IOError) as e:
        logger.info(f"Cache miss ({type(e).__name__}), will compute result: {e}")
        return None


def save_comparison_result(
    save_path: str | Path,
    mode: str,
    metric: str,
    result: Any,
    dataset_names: tuple[str, str] | None = None,
    metadata: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> None:
    """Save a comparison result to HDF5.

    This is a helper function for auto-save/load logic that saves results
    based on mode and metric.

    Parameters
    ----------
    save_path : str or Path
        Path to HDF5 file
    mode : {"between", "all-pairs"}
        Comparison mode (within mode not supported)
    metric : str
        Metric name
    result : Any
        Result to save (float, ndarray, or dict)
    dataset_names : tuple[str, str], optional
        Dataset names for between mode. Required for mode="between".
    metadata : dict, optional
        Additional metadata to store
    overwrite : bool, default=False
        If True, overwrite existing comparison

    Raises
    ------
    ValueError
        If mode="between" and dataset_names is None
    """
    if mode == "between":
        if dataset_names is None:
            raise ValueError(
                "dataset_names required for save_path with mode='between'. "
                "Provide tuple like ('control', 'treatment')"
            )
        dataset_i, dataset_j = dataset_names

        # Handle dict return from compute_between_distances
        save_value: float | npt.NDArray[np.floating] | dict[str, dict[str, float]]
        if isinstance(result, dict) and "value" in result:
            save_value = float(result["value"])  # type: ignore[assignment]
        elif isinstance(result, tuple):
            # For shape metrics: (distance, pairs_dict)
            save_value = float(result[0])  # type: ignore[assignment]
        else:
            save_value = result  # type: ignore[assignment]

        save_comparison(
            filepath=save_path,
            metric=metric,
            dataset_i=dataset_i,
            dataset_j=dataset_j,
            mode=mode,
            value=save_value,
            metadata=metadata,
            overwrite=overwrite,
        )
        logger.info(f"Saved between-mode result: {metric}/{dataset_i}___{dataset_j}")
    elif mode == "all-pairs":
        # Determine number of datasets for all-pairs
        n_datasets = len(result) if isinstance(result, dict) else 1
        save_val_all_pairs = result  # type: ignore[assignment]
        save_comparison(
            filepath=save_path,
            metric=metric,
            dataset_i="all_pairs",
            dataset_j="all_pairs",
            mode=mode,
            value=save_val_all_pairs,
            metadata={"n_datasets": n_datasets, **(metadata or {})},
            overwrite=overwrite,
        )
        logger.info(f"Saved all-pairs result: {metric}/all_pairs")
    # Within mode: no save (single dataset, less useful to cache)
