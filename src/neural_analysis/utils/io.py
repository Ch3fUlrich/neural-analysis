"""Array and HDF5 I/O utilities for neural_analysis.

This module provides both lightweight NumPy I/O helpers (npy/npz) used by
tests and a robust HDF5 interface compatible with the legacy ``h5io`` usage
from the todo code while offering a more modular API under the hood.

Goals:
- Modular and flexible HDF5 saving/loading (arrays, DataFrames, metadata)
- Backward-compatible wrapper ``h5io(path, task=...)``
- Simple NumPy helpers: ``save_array``, ``load_array``, ``update_array``

Notes:
- HDF5 attributes (attrs) are stored at file-level. Non-scalar attrs are
    JSON-encoded strings to ensure broad compatibility.
- DataFrames are stored as a group with datasets: values, columns, index, and
    an ``index_is_numeric`` flag for roundtrip reconstruction.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    import pandas as pd

try:
    from .logging import get_logger, log_calls
except ImportError:
    # Fallback no-op decorator if logging module unavailable
    from collections.abc import Callable
    def log_calls(
        *, level: int = logging.DEBUG, timeit: bool = True
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            return func
        return decorator

    def get_logger(name: str | None = None) -> logging.Logger:
        return logging.getLogger(name)


# Module logger
logger = get_logger(__name__)

try:
    import pandas as pd  # noqa: PGH003

    HAS_PANDAS = True
except Exception:  # pragma: no cover - optional dependency
    HAS_PANDAS = False


Jsonable = (
    str | int | float | bool | None | Mapping[str, "Jsonable"] | Iterable["Jsonable"]
)
DatasetDict = dict[str, Any]


def _ensure_parent_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _to_bytes_array(values: Iterable[str]) -> npt.NDArray[np.bytes_]:
    # h5py prefers fixed-width ASCII for strings; we use utf-8 encoded bytes
    return np.array([str(v).encode("utf-8") for v in values], dtype="S")


def _from_bytes_array(values: npt.NDArray[np.bytes_]) -> list[str]:
    return [
        v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else str(v)
        for v in values.tolist()
    ]


def _save_dataframe(
    group: Any, df: pd.DataFrame, compression: str, compression_opts: int
) -> None:
    group.create_dataset(
        "index_is_numeric", data=np.array([df.index.inferred_type != "string"])
    )

    # Save index
    if df.index.inferred_type == "string":
        group.create_dataset("index", data=_to_bytes_array(df.index))
    else:
        group.create_dataset("index", data=np.asarray(df.index))

    # Store columns and per-column datasets to avoid object dtype
    columns_bytes = _to_bytes_array(df.columns)
    group.create_dataset("columns", data=columns_bytes)
    cols_grp = (
        group.create_group("columns_data")
        if "columns_data" not in group
        else group["columns_data"]
    )
    # Clear existing children if overwriting
    for key in list(cols_grp.keys()):
        del cols_grp[key]
    for col in df.columns:
        arr = df[col].to_numpy()
        if arr.dtype.kind in {"U", "O"}:
            arr = _to_bytes_array(arr)  # type: ignore[assignment]
        cols_grp.create_dataset(
            col, data=arr, compression=compression, compression_opts=compression_opts
        )


def _load_dataframe(group: Any) -> pd.DataFrame:
    columns = _from_bytes_array(group["columns"][...])
    index_is_numeric = (
        bool(group["index_is_numeric"][0]) if "index_is_numeric" in group else False
    )
    if index_is_numeric:
        index = group["index"][...]
    else:
        index = _from_bytes_array(group["index"][...])
    # Restore columns and per-column data
    cols_grp = group.get("columns_data", None)
    data_dict: dict[str, Any] = {}
    if cols_grp is not None:
        for col in columns:
            arr = cols_grp[col][...]
            if isinstance(arr, np.ndarray) and arr.dtype.kind == "S":
                data_dict[col] = [x.decode("utf-8") for x in arr]
            else:
                data_dict[col] = arr
        df = pd.DataFrame(data_dict, index=index)
    else:
        # Fallback: no columns_data, attempt to read a flat values dataset
        values = group["values"][...]
        if values.dtype == object:
            out = []
            for row in values:
                out.append(
                    [
                        x.decode("utf-8")
                        if isinstance(x, (bytes, bytearray, np.bytes_))
                        else x
                        for x in row
                    ]
                )
            values = np.array(out, dtype=object)
        df = pd.DataFrame(values, index=index, columns=columns)
    return df


# ============================================================================
# NumPy I/O (NPY/NPZ)
# ============================================================================


def _resolve_npy_npz_path(path: str | Path) -> Path:
    p = Path(path)
    if p.suffix in {".npy", ".npz"}:
        return p
    # prefer existing file if any
    for ext in (".npy", ".npz"):
        if p.with_suffix(ext).exists():
            return p.with_suffix(ext)
    # default to provided (caller decides extension)
    return p


@log_calls(level=logging.DEBUG)
def save_array(
    path: str | Path,
    data: npt.NDArray[Any] | Mapping[str, npt.NDArray[Any]],
    *,
    allow_overwrite: bool = True,
) -> Path:
    """Save a single array (.npy) or a dict of arrays (.npz).

    If no extension is provided, it is inferred from the input type.
    Returns the path to the saved file.
    """
    p = Path(path)

    if isinstance(data, Mapping):
        if p.suffix != ".npz":
            p = p.with_suffix(".npz")
        logger.info(f"Saving {len(data)} arrays to NPZ: {p}")
        _ensure_parent_dir(p)
        if p.exists() and not allow_overwrite:
            raise ValueError(f"Path already exists: {p}")
        np.savez(p, **(dict[str, Any]({k: np.asarray(v) for k, v in data.items()})))
        logger.info(f"Successfully saved {len(data)} arrays to {p}")
        return p
    else:
        arr = np.asarray(data)
        if p.suffix != ".npy":
            p = p.with_suffix(".npy")
        logger.info(f"Saving array with shape {arr.shape} to NPY: {p}")
        _ensure_parent_dir(p)
        if p.exists() and not allow_overwrite:
            raise ValueError(f"File already exists: {p}")
        np.save(p, arr)
        logger.info(f"Successfully saved array to {p}")
        return p


@log_calls(level=logging.DEBUG)
def load_array(
    path: str | Path,
) -> npt.NDArray[Any] | dict[str, npt.NDArray[Any]] | None:
    """Load an array or dict of arrays from .npy/.npz. Returns None if missing."""
    p = _resolve_npy_npz_path(path)

    if not p.exists():
        logger.warning(f"File not found: {p}")
        return None

    logger.info(f"Loading array(s) from {p}")

    if p.suffix == ".npz":
        with np.load(p, allow_pickle=False) as data:
            dct = {k: data[k] for k in data.files}
            logger.info(f"Loaded {len(dct)} arrays from NPZ: {p}")
            return dct
    else:
        arr: npt.NDArray[Any] = np.load(p, allow_pickle=False)
        logger.info(f"Loaded array with shape {arr.shape} from NPY: {p}")
        return arr


def update_array(path: str | Path, new_data: Mapping[str, npt.NDArray[Any]]) -> Path:
    """Update or create an .npz file by merging in new arrays."""
    p = Path(path)
    if p.suffix != ".npz":
        p = p.with_suffix(".npz")
    _ensure_parent_dir(p)
    merged: dict[str, npt.NDArray[Any]] = {}
    if p.exists():
        with np.load(p, allow_pickle=False) as data:
            merged.update({k: data[k] for k in data.files})
    merged.update({k: np.asarray(v) for k, v in new_data.items()})
    np.savez(p, **(dict[str, Any](merged)))
    return p


def _write_attrs(f: Any, attrs: Mapping[str, Jsonable] | None) -> None:
    if not attrs:
        return
    for k, v in attrs.items():
        try:
            if isinstance(v, (str, int, float, bool)) or v is None:
                f.attrs[k] = v
            else:
                f.attrs[k] = json.dumps(v)
        except Exception:
            f.attrs[k] = json.dumps(str(v))


@log_calls(level=logging.DEBUG)
def save_hdf5(
    path: str | Path,
    data: Any,
    *,
    labels: Any | None = None,
    attrs: Mapping[str, Jsonable] | None = None,
    mode: str = "w",
    compression: str = "gzip",
    compression_opts: int = 4,
) -> None:
    """Save a DataFrame or array with optional labels into an HDF5 file.

    This maintains backward compatibility with existing tests and wrappers.
    """
    _ensure_parent_dir(path)
    import h5py  # local import to avoid hard dependency

    is_dataframe = HAS_PANDAS and "DataFrame" in type(data).__name__
    data_type = "DataFrame" if is_dataframe else "array"

    logger.info(
        f"Saving {data_type} to HDF5: {path}, mode='{mode}', "
        f"compression='{compression}', has_labels={labels is not None}, "
        f"has_attrs={attrs is not None}"
    )

    with h5py.File(path, mode) as f:
        _write_attrs(f, attrs)

        if is_dataframe:
            grp = f.create_group("data") if "data" not in f else f["data"]
            for key in list(grp.keys()):
                del grp[key]
            _save_dataframe(grp, data, compression, compression_opts)
            logger.info(f"Saved DataFrame with shape {data.shape} to {path}")
        else:
            arr = np.asarray(data)
            if "data" in f:
                del f["data"]
            f.create_dataset(
                "data",
                data=arr,
                compression=compression,
                compression_opts=compression_opts,
            )
            logger.info(f"Saved array with shape {arr.shape} to {path}")

        if labels is not None:
            lbl = np.asarray(labels)
            if lbl.dtype.kind in {"U", "O"}:
                lbl = _to_bytes_array(lbl)
            if "labels" in f:
                del f["labels"]
            f.create_dataset("labels", data=lbl)
            logger.info(f"Saved labels with shape {lbl.shape} to {path}")


@log_calls(level=logging.DEBUG)
def load_hdf5(
    path: str | Path,
    *,
    filter_pairs: Iterable[tuple[Any, Any]] | None = None,
    return_attrs: bool = False,
) -> tuple[Any, Any] | tuple[tuple[Any, Any], dict[str, Jsonable]]:
    """Load previously saved HDF5 content.

    Returns a tuple (data, labels) for compatibility with existing tests.
    When ``return_attrs`` is True, returns ((data, labels), attrs_dict).

    If ``filter_pairs`` is provided and a DataFrame with columns
    ("item_i", "item_j") is loaded, rows will be filtered to keep only the
    provided item pairs.
    """
    p = Path(path)

    if not p.exists():
        logger.warning(f"HDF5 file not found: {p}")
        return (None, []) if not return_attrs else ((None, []), {})

    logger.info(
        f"Loading HDF5 from {p}, "
        f"filter_pairs={filter_pairs is not None}, "
        f"return_attrs={return_attrs}"
    )

    import h5py  # local import

    data_out: Any = None
    labels_out: Any = []
    attrs_out: dict[str, Jsonable] = {}

    with h5py.File(p, "r") as f:
        # attrs
        for k, v in f.attrs.items():
            if isinstance(v, (np.bytes_, bytes, bytearray)):
                txt = (
                    v.decode("utf-8")
                    if isinstance(v, (bytes, bytearray))
                    else v.tobytes().decode("utf-8")
                )
                try:
                    attrs_out[k] = json.loads(txt)
                except Exception:
                    attrs_out[k] = txt
            else:
                if hasattr(v, "item"):
                    try:
                        attrs_out[k] = v.item()
                    except Exception:
                        attrs_out[k] = v
                else:
                    attrs_out[k] = v

        if attrs_out:
            logger.debug(f"Loaded {len(attrs_out)} attributes from {p}")

        # data
        if "data" in f:
            obj = f["data"]
            if hasattr(obj, "keys"):
                # Group: likely a DataFrame stored via columns_data
                if HAS_PANDAS and all(k in obj for k in ("columns", "index")):
                    df = _load_dataframe(obj)
                    logger.info(f"Loaded DataFrame with shape {df.shape} from {p}")
                    if (
                        filter_pairs is not None
                        and HAS_PANDAS
                        and {"item_i", "item_j"}.issubset(df.columns)
                    ):
                        wanted = set(filter_pairs)
                        pairs = list(zip(df["item_i"].tolist(), df["item_j"].tolist()))
                        mask = np.array([pair in wanted for pair in pairs])
                        n_before = len(df)
                        df = df[mask].reset_index(drop=True)
                        n_pairs = (
                            len(list(filter_pairs))
                            if hasattr(filter_pairs, "__len__")
                            else "N"
                        )
                        logger.info(
                            f"Filtered DataFrame from {n_before} to {len(df)} rows "
                            f"using {n_pairs} pairs"
                        )
                    data_out = df
                else:
                    # Unsupported group layout -> load children as dict of arrays
                    child = {k: obj[k][...] for k in obj}
                    logger.debug(f"Loaded group as dict with {len(child)} entries")
                    data_out = child
            else:
                # Dataset: read array
                data_out = obj[...]
                logger.info(f"Loaded array with shape {data_out.shape} from {p}")

        # labels
        if "labels" in f:
            lbl = f["labels"][...]
            if isinstance(lbl, np.ndarray) and lbl.dtype.kind == "S":
                labels_out = [x.decode("utf-8") for x in lbl]
            else:
                labels_out = lbl
            logger.debug(f"Loaded labels with shape {np.asarray(labels_out).shape}")

    logger.info(f"Successfully loaded HDF5 from {p}")

    if return_attrs:
        return (data_out, labels_out), attrs_out
    return data_out, labels_out


def save_comparison_batch(
    result_rows: list[dict[str, Any]],
    df_results: pd.DataFrame | None,
    save_path: Path | None,
) -> pd.DataFrame:
    """Save batch of comparison results to HDF5.

    This function is a modular I/O helper for batch comparison operations.
    It concatenates new results with existing results and saves to HDF5.

    Parameters
    ----------
    result_rows : list[dict]
        List of dictionaries containing comparison results. Each dict should have
        keys like 'dataset_i', 'dataset_j', 'metric', 'value', etc.
    df_results : DataFrame or None
        Existing results DataFrame to append to. If None, creates new DataFrame.
    save_path : Path or None
        Path to save HDF5 file. If None, skips saving.

    Returns
    -------
    DataFrame
        Combined DataFrame with all results (existing + new).

    Notes
    -----
    This function assumes pandas is available (checked by HAS_PANDAS flag).
    The DataFrame is saved using save_hdf5() with mode='w' (overwrite).
    """
    if not HAS_PANDAS:  # pragma: no cover
        raise ImportError("pandas is required for save_comparison_batch")

    if not result_rows:
        return df_results if df_results is not None else pd.DataFrame()

    df_new = pd.DataFrame(result_rows)
    if df_results is not None:
        df_results = pd.concat([df_results, df_new], ignore_index=True)
    else:
        df_results = df_new

    if save_path:
        save_hdf5(save_path, df_results, mode="w")

    return df_results


def get_missing_comparisons(
    item_pairs: list[tuple[str, str]],
    metrics_dict: dict[str, dict[str, object]],
    df_results: pd.DataFrame | None,
) -> list[tuple[str, str, str]]:
    """Determine which comparisons need to be computed.

    This function compares requested (dataset_i, dataset_j, metric) triplets
    against existing results to identify missing computations. Used for
    incremental/resumable batch processing.

    Parameters
    ----------
    item_pairs : list[tuple[str, str]]
        List of (name_i, name_j) pairs to compare.
    metrics_dict : dict[str, dict[str, object]]
        Dictionary mapping metric names to their kwargs.
    df_results : DataFrame or None
        Existing results DataFrame. If None, all comparisons are missing.

    Returns
    -------
    list[tuple[str, str, str]]
        List of (name_i, name_j, metric_name) triplets that need computation.

    Notes
    -----
    This function checks if a (dataset_i, dataset_j, metric) combination already
    exists in df_results. If the DataFrame is None or the combination is not found,
    it's added to the missing list.
    """
    if not HAS_PANDAS:  # pragma: no cover
        raise ImportError("pandas is required for get_missing_comparisons")

    missing = []
    for name_i, name_j in item_pairs:
        for metric_name in metrics_dict:
            # Check if this combination exists in results
            if df_results is None or df_results.empty:
                missing.append((name_i, name_j, metric_name))
            else:
                mask = (
                    (df_results["dataset_i"] == name_i)
                    & (df_results["dataset_j"] == name_j)
                    & (df_results["metric"] == metric_name)
                )
                if not mask.any():
                    missing.append((name_i, name_j, metric_name))
    return missing


def h5io(
    path: str | Path,
    *,
    task: str,
    data: Any | None = None,
    labels: Any | None = None,
    attrs: Mapping[str, Jsonable] | None = None,
    mode: str = "w",
) -> tuple[Any, Any] | tuple[tuple[Any, Any], dict[str, Jsonable]] | None:
    """Compatibility wrapper replicating legacy `h5io` API.

    Examples
    --------
    Save:
        >>> h5io("/tmp/file.h5", task="save", data=df, labels=labels)
    Load:
        >>> data, labels = h5io("/tmp/file.h5", task="load")
    """
    task = task.lower()
    if task == "save":
        save_hdf5(path, data, labels=labels, attrs=attrs, mode=mode)
        return None
    elif task == "load":
        result: tuple[Any, Any] | tuple[tuple[Any, Any], dict[str, Jsonable]] = (
            load_hdf5(path)
        )
        return result
    else:  # pragma: no cover - defensive
        raise ValueError("task must be either 'save' or 'load'")


def save_result_to_hdf5_dataset(
    save_path: str | Path,
    dataset_name: str,
    result_key: str,
    scalar_data: dict[str, Any],
    array_data: dict[str, npt.NDArray[Any]],
    compression: str = "gzip",
) -> None:
    """Save analysis results to HDF5 file with hierarchical structure.

    Creates a hierarchical structure: dataset_name / result_key / {scalars, arrays}
    This is a generalized function for saving any analysis results.

    Parameters
    ----------
    save_path : str or Path
        Path to HDF5 file
    dataset_name : str
        Top-level dataset identifier (e.g., "session_001", "mouse_A")
    result_key : str
        Unique key for this result within the dataset
    scalar_data : dict
        Dictionary of scalar metadata (stored as HDF5 attributes)
    array_data : dict
        Dictionary of numpy arrays (stored as HDF5 datasets)
    compression : str, default='gzip'
        Compression algorithm for arrays

    Examples
    --------
    >>> save_result_to_hdf5_dataset(
    ...     "results.h5",
    ...     dataset_name="session_001",
    ...     result_key="analysis_bins10_neighbors15",
    ...     scalar_data={"metric_value": 0.75, "n_bins": 10},
    ...     array_data={"matrix": overlap_matrix, "shuffles": shuf_values}
    ... )
    """
    import h5py

    save_path = Path(save_path)
    _ensure_parent_dir(save_path)

    with h5py.File(save_path, "a") as f:
        # Create or get dataset group
        if dataset_name not in f:
            ds_group = f.create_group(dataset_name)
        else:
            ds_group = f[dataset_name]

        # Remove existing result if present (overwrite)
        if result_key in ds_group:
            del ds_group[result_key]

        # Create result group
        result_group = ds_group.create_group(result_key)

        # Save scalar metadata as attributes
        for key, value in scalar_data.items():
            if value is not None:
                try:
                    if isinstance(value, (str, int, float, bool)) or value is None:
                        result_group.attrs[key] = value
                    else:
                        result_group.attrs[key] = str(value)
                except Exception:
                    result_group.attrs[key] = str(value)

        # Save arrays as datasets
        for key, arr in array_data.items():
            result_group.create_dataset(
                key,
                data=arr,
                compression=compression,
            )


def load_results_from_hdf5_dataset(
    save_path: str | Path,
    dataset_name: str | None = None,
    result_key: str | None = None,
    filter_attrs: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Load analysis results from HDF5 file.

    Parameters
    ----------
    save_path : str or Path
        Path to HDF5 file
    dataset_name : str, optional
        Load only this dataset. If None, loads all datasets.
    result_key : str, optional
        Load only this specific result key
    filter_attrs : dict, optional
        Filter results by attribute values (e.g., {"n_bins": 10})

    Returns
    -------
    results : dict
        Nested dictionary: {dataset_name: {result_key: {attrs, arrays}}}

    Examples
    --------
    >>> # Load all results
    >>> results = load_results_from_hdf5_dataset("results.h5")
    >>>
    >>> # Load specific dataset
    >>> results = load_results_from_hdf5_dataset(
    ...     "results.h5", dataset_name="session_001"
    ... )
    >>>
    >>> # Filter by parameters
    >>> results = load_results_from_hdf5_dataset(
    ...     "results.h5",
    ...     dataset_name="session_001",
    ...     filter_attrs={"n_bins": 10, "n_neighbors": 15}
    ... )
    """
    import h5py

    save_path = Path(save_path)

    if not save_path.exists():
        logger.debug(f"File not found: {save_path}")
        return {}

    results: dict[str, dict[str, Any]] = {}

    try:
        with h5py.File(save_path, "r") as f:
            # Determine which datasets to load
            if dataset_name is not None:
                if dataset_name not in f:
                    logger.debug(f"Dataset {dataset_name} not found")
                    return {}
                dataset_names = [dataset_name]
            else:
                dataset_names = list(f.keys())

            for ds_name in dataset_names:
                ds_group = f[ds_name]
                results[ds_name] = {}

                # Determine which results to load
                if result_key is not None:
                    if result_key not in ds_group:
                        continue
                    result_keys = [result_key]
                else:
                    result_keys = list(ds_group.keys())

                for res_key in result_keys:
                    result_group = ds_group[res_key]

                    # Load attributes
                    attrs = dict(result_group.attrs)

                    # Apply filters
                    if filter_attrs is not None and not all(
                        attrs.get(k) == v for k, v in filter_attrs.items()
                    ):
                        continue

                    # Load arrays
                    arrays = {key: result_group[key][:] for key in result_group}

                    results[ds_name][res_key] = {
                        "attributes": attrs,
                        "arrays": arrays,
                    }

    except Exception as e:
        logger.error(f"Error loading results from {save_path}: {e}")
        return {}

    return results


def get_hdf5_dataset_names(save_path: str | Path) -> list[str]:
    """Get list of all top-level dataset names in HDF5 file.

    Parameters
    ----------
    save_path : str or Path
        Path to HDF5 file

    Returns
    -------
    dataset_names : list of str
        List of dataset names

    Examples
    --------
    >>> datasets = get_hdf5_dataset_names("results.h5")
    >>> print(f"Found {len(datasets)} datasets")
    """
    import h5py

    save_path = Path(save_path)

    if not save_path.exists():
        return []

    try:
        with h5py.File(save_path, "r") as f:
            return list(f.keys())
    except Exception as e:
        logger.error(f"Error reading dataset names from {save_path}: {e}")
        return []


def get_hdf5_result_summary(
    save_path: str | Path,
    dataset_name: str | None = None,
) -> pd.DataFrame:
    """Get summary DataFrame of all results in HDF5 file.

    Parameters
    ----------
    save_path : str or Path
        Path to HDF5 file
    dataset_name : str, optional
        Filter by specific dataset

    Returns
    -------
    summary : DataFrame
        Summary with one row per result, including all attributes

    Examples
    --------
    >>> summary = get_hdf5_result_summary("results.h5")
    >>> print(summary[['dataset_name', 'result_key', 'metric_value']])
    """
    import h5py

    if not HAS_PANDAS:
        raise ImportError("pandas is required for get_hdf5_result_summary")

    save_path = Path(save_path)

    if not save_path.exists():
        return pd.DataFrame()

    rows = []

    try:
        with h5py.File(save_path, "r") as f:
            dataset_names_list = (
                [dataset_name] if dataset_name and dataset_name in f else list(f.keys())
            )

            for ds_name in dataset_names_list:
                ds_group = f[ds_name]

                for res_key in ds_group:
                    result_group = ds_group[res_key]
                    attrs = dict(result_group.attrs)

                    row = {
                        "dataset_name": ds_name,
                        "result_key": res_key,
                        **attrs,
                    }
                    rows.append(row)

    except Exception as e:
        logger.error(f"Error reading summary from {save_path}: {e}")

    return pd.DataFrame(rows)


def load_distribution_comparisons(
    save_path: str | Path,
    comparison_name: str | None = None,
    dataset_i: str | None = None,
    dataset_j: str | None = None,
    metric: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Load distribution comparison results from HDF5.

    Parameters
    ----------
    save_path : str or Path
        Path to HDF5 file
    comparison_name : str, optional
        Filter by comparison group name
    dataset_i : str, optional
        Filter by first dataset name
    dataset_j : str, optional
        Filter by second dataset name
    metric : str, optional
        Filter by metric name

    Returns
    -------
    results : dict
        Nested dictionary: {comparison_name: {result_key: {scalars, arrays}}}

    Examples
    --------
    >>> # Load all comparisons
    >>> results = load_distribution_comparisons("output/comparisons.h5")
    >>>
    >>> # Load specific comparison group
    >>> results = load_distribution_comparisons(
    ...     "output/comparisons.h5",
    ...     comparison_name="session_001"
    ... )
    >>>
    >>> # Filter by datasets and metric
    >>> results = load_distribution_comparisons(
    ...     "output/comparisons.h5",
    ...     dataset_i="condition_A",
    ...     metric="wasserstein"
    ... )
    """
    # Build filter attributes
    filter_attrs = {}
    if dataset_i is not None:
        filter_attrs["dataset_i"] = dataset_i
    if dataset_j is not None:
        filter_attrs["dataset_j"] = dataset_j
    if metric is not None:
        filter_attrs["metric"] = metric

    # Load using generalized function
    return load_results_from_hdf5_dataset(
        save_path=save_path,
        dataset_name=comparison_name,
        filter_attrs=filter_attrs if filter_attrs else None,
    )


__all__ = [
    "save_array",
    "load_array",
    "update_array",
    "save_hdf5",
    "load_hdf5",
    "h5io",
    "save_comparison_batch",
    "get_missing_comparisons",
    "save_result_to_hdf5_dataset",
    "load_results_from_hdf5_dataset",
    "get_hdf5_dataset_names",
    "get_hdf5_result_summary",
]
