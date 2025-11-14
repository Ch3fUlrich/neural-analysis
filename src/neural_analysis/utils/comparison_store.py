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
from typing import Any

import h5py  # type: ignore[import-untyped]
import numpy as np
import numpy.typing as npt
import pandas as pd

from neural_analysis.utils.logging import get_logger

__all__ = [
    "save_comparison",
    "load_comparison",
    "query_comparisons",
]

logger = get_logger(__name__)

# HDF5 storage configuration
COMPRESSION = "gzip"
COMPRESSION_LEVEL = 6
SHUFFLE_FILTER = True
CHUNK_SIZE_THRESHOLD = 10_000  # Elements threshold for chunking


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
    else:
        raise TypeError(
            f"Unsupported value type: {type(value)}. Expected float, ndarray, or dict"
        )


def _encode_dict_for_hdf5(d: dict[str, dict[str, float]]) -> npt.NDArray[np.void]:
    """Encode nested dict as structured array for HDF5 storage.

    Parameters
    ----------
    d : dict[str, dict[str, float]]
        Nested dictionary with structure {dataset_i: {dataset_j: value}}

    Returns
    -------
    ndarray
        Structured array with dtype [("key_i", "S100"), ("key_j", "S100"),
        ("value", "f8")]
    """
    rows = [
        (key_i.encode("utf-8"), key_j.encode("utf-8"), value)
        for key_i, inner in d.items()
        for key_j, value in inner.items()
    ]
    dtype = np.dtype([("key_i", "S100"), ("key_j", "S100"), ("value", "f8")])
    return np.array(rows, dtype=dtype)


def _decode_dict_from_hdf5(arr: npt.NDArray[np.void]) -> dict[str, dict[str, float]]:
    """Decode structured array back to nested dict.

    Parameters
    ----------
    arr : ndarray
        Structured array from HDF5 with dtype containing key_i, key_j, value

    Returns
    -------
    dict[str, dict[str, float]]
        Nested dictionary {dataset_i: {dataset_j: value}}
    """
    result: dict[str, dict[str, float]] = {}
    for row in arr:
        key_i = row["key_i"].decode("utf-8")
        key_j = row["key_j"].decode("utf-8")
        value = float(row["value"])

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
) -> None:
    """Save a comparison result to HDF5 (delegates to io.py backend).

    **RECOMMENDED**: Use io.save_result_to_hdf5_dataset() directly for new code.

    This function provides comparison-specific formatting, then delegates to
    io.save_result_to_hdf5_dataset() for actual HDF5 operations.

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
            with h5py.File(filepath_obj, "r") as f:
                if dataset_name in f and result_key in f[dataset_name]:
                    raise ValueError(
                        f"Comparison already exists: {dataset_name}/{result_key}. "
                        "Set overwrite=True to replace."
                    )

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

    # Delegate to io.py backend
    save_result_to_hdf5_dataset(
        save_path=filepath,
        dataset_name=dataset_name,
        result_key=result_key,
        scalar_data=scalar_data,
        array_data=array_data,
        compression=COMPRESSION,
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
        raise KeyError(
            f"Comparison not found: {dataset_name}/{result_key}. "
            f"Check file contents with query_comparisons()."
        )

    # Extract the single result
    result_data = results[dataset_name][result_key]
    attrs = result_data.get("attributes", result_data.get("attrs", {}))
    arrays = result_data.get("arrays", {})
    value_type = attrs.get("value_type", "unknown")

    # Decode based on value_type
    result: float | npt.NDArray[np.floating] | dict[str, dict[str, float]]
    if value_type == "scalar":
        result = float(attrs["value"])
    elif value_type == "matrix":
        result = arrays["value"]
    elif value_type == "dict":
        result = _decode_dict_from_hdf5(arrays["value"])
    else:
        logger.warning(f"Unknown value_type '{value_type}', returning raw array")
        result = arrays.get("value", np.array([]))

    logger.info(f"Successfully loaded comparison (type={value_type})")
    return result


def query_comparisons(
    filepath: str | Path,
    metric: str | None = None,
    dataset_i: str | None = None,
    dataset_j: str | None = None,
    mode: str | None = None,
    load_values: bool = False,
) -> pd.DataFrame:
    """Query comparisons with optional filters (delegates to io.py backend).

    **RECOMMENDED**: Use io.load_results_from_hdf5_dataset() with filter_attrs
    for new code.

    This function provides comparison-specific query interface, then delegates to
    io.load_results_from_hdf5_dataset() for actual HDF5 operations.

    Parameters
    ----------
    filepath : str or Path
        Path to HDF5 file
    metric : str, optional
        Filter by metric name
    dataset_i : str, optional
        Filter by first dataset identifier
    dataset_j : str, optional
        Filter by second dataset identifier
    mode : str, optional
        Filter by mode ("within", "between", "all-pairs")
    load_values : bool, default=False
        If True, load comparison values into "value" column (memory intensive)

    Returns
    -------
    DataFrame
        Table with columns: metric, dataset_i, dataset_j, mode
        If load_values=True, also includes "value" column

    Examples
    --------
    >>> # Query all euclidean distance comparisons
    >>> df = query_comparisons(filepath="results.h5", metric="euclidean")
    >>>
    >>> # Query between-dataset comparisons
    >>> df = query_comparisons(filepath="results.h5", mode="between")
    >>>
    >>> # Query with loaded values
    >>> df = query_comparisons(
    ...     filepath="results.h5",
    ...     metric="wasserstein",
    ...     load_values=True
    ... )
    >>> df["value"].mean()  # Compute mean distance
    """
    from neural_analysis.utils.io import load_results_from_hdf5_dataset

    filepath_obj = Path(filepath)
    if not filepath_obj.exists():
        raise FileNotFoundError(f"HDF5 file not found: {filepath}")

    logger.info(f"Querying comparisons from {filepath}")

    # Build filter for io.py backend
    filter_attrs: dict[str, Any] = {}
    if mode is not None:
        filter_attrs["mode"] = mode
    if dataset_i is not None:
        filter_attrs["dataset_i"] = dataset_i
    if dataset_j is not None:
        filter_attrs["dataset_j"] = dataset_j

    # Load all results from specified metric (or all metrics if None)
    results = load_results_from_hdf5_dataset(
        save_path=filepath,
        dataset_name=metric,  # None = load all metrics
        result_key=None,  # Load all comparisons
        filter_attrs=filter_attrs if filter_attrs else None,
    )

    # Convert to DataFrame
    rows: list[dict[str, Any]] = []
    for dataset_name, result_dict in results.items():
        for _result_key, result_data in result_dict.items():
            attrs = result_data.get("attrs", {})
            arrays = result_data.get("arrays", {})

            row = {
                "metric": attrs.get("metric", dataset_name),
                "dataset_i": attrs.get("dataset_i", ""),
                "dataset_j": attrs.get("dataset_j", ""),
                "mode": attrs.get("mode", ""),
            }

            if load_values:
                value_type = attrs.get("value_type", "unknown")
                if value_type == "scalar":
                    row["value"] = attrs.get("value")
                elif value_type == "matrix":
                    row["value"] = arrays.get("value")
                elif value_type == "dict":
                    row["value"] = _decode_dict_from_hdf5(arrays["value"])
                else:
                    row["value"] = arrays.get("value")

            rows.append(row)

    df = pd.DataFrame(rows)
    logger.info(f"Query returned {len(df)} comparisons")

    return df
