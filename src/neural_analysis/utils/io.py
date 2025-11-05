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

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple, Union

import json
import logging
import numpy as np

try:
    from .logging import log_calls, get_logger  # type: ignore
except ImportError:
    # Fallback no-op decorator if logging module unavailable
    def log_calls(**kwargs):  # type: ignore
        def decorator(func):  # type: ignore
            return func
        return decorator
    def get_logger(name: str):  # type: ignore
        return logging.getLogger(name)

# Module logger
logger = get_logger(__name__)

try:
    import pandas as pd  # Optional, used when available
    HAS_PANDAS = True
except Exception:  # pragma: no cover - optional dependency
    HAS_PANDAS = False


Jsonable = Union[str, int, float, bool, None, Mapping[str, "Jsonable"], Iterable["Jsonable"]]
DatasetDict = Dict[str, Any]


def _ensure_parent_dir(path: Union[str, Path]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _to_bytes_array(values: Iterable[str]) -> np.ndarray:
    # h5py prefers fixed-width ASCII for strings; we use utf-8 encoded bytes
    return np.array([str(v).encode("utf-8") for v in values], dtype="S")


def _from_bytes_array(values: np.ndarray) -> list[str]:
    return [v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else str(v) for v in values.tolist()]


def _save_dataframe(group, df: "pd.DataFrame", compression: str, compression_opts: int) -> None:  # type: ignore[name-defined]
    group.create_dataset("index_is_numeric", data=np.array([df.index.inferred_type != "string"]))

    # Save index
    if df.index.inferred_type == "string":
        group.create_dataset("index", data=_to_bytes_array(df.index))
    else:
        group.create_dataset("index", data=np.asarray(df.index))

    # Store columns and per-column datasets to avoid object dtype
    columns_bytes = _to_bytes_array(df.columns)
    group.create_dataset("columns", data=columns_bytes)
    cols_grp = group.create_group("columns_data") if "columns_data" not in group else group["columns_data"]
    # Clear existing children if overwriting
    for key in list(cols_grp.keys()):
        del cols_grp[key]
    for col in df.columns:
        arr = df[col].to_numpy()
        if arr.dtype.kind in {"U", "O"}:
            arr = _to_bytes_array(arr)
        cols_grp.create_dataset(col, data=arr, compression=compression, compression_opts=compression_opts)


def _load_dataframe(group) -> "pd.DataFrame":  # type: ignore[name-defined]
    columns = _from_bytes_array(group["columns"][...])
    index_is_numeric = bool(group["index_is_numeric"][0]) if "index_is_numeric" in group else False
    if index_is_numeric:
        index = group["index"][...]
    else:
        index = _from_bytes_array(group["index"][...])
    # Restore columns and per-column data
    cols_grp = group["columns_data"] if "columns_data" in group else None
    data_dict: Dict[str, Any] = {}
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
                out.append([
                    x.decode("utf-8") if isinstance(x, (bytes, bytearray, np.bytes_)) else x for x in row
                ])
            values = np.array(out, dtype=object)
        df = pd.DataFrame(values, index=index, columns=columns)
    return df  # type: ignore[name-defined]


# ============================================================================
# NumPy I/O (NPY/NPZ)
# ============================================================================

def _resolve_npy_npz_path(path: Union[str, Path]) -> Path:
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
    path: Union[str, Path],
    data: Union[np.ndarray, Mapping[str, np.ndarray]],
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
            raise ValueError(f"File already exists: {p}")
        np.savez(p, **{k: np.asarray(v) for k, v in data.items()})
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
def load_array(path: Union[str, Path]) -> Union[np.ndarray, Dict[str, np.ndarray], None]:
    """Load an array or dict of arrays from .npy/.npz. Returns None if missing."""
    p = _resolve_npy_npz_path(path)
    
    if not p.exists():
        logger.warning(f"File not found: {p}")
        return None
    
    logger.info(f"Loading array(s) from {p}")
    
    if p.suffix == ".npz":
        with np.load(p, allow_pickle=False) as data:
            result = {k: data[k] for k in data.files}
            logger.info(f"Loaded {len(result)} arrays from NPZ: {p}")
            return result
    else:
        arr = np.load(p, allow_pickle=False)
        logger.info(f"Loaded array with shape {arr.shape} from NPY: {p}")
        return arr


def update_array(path: Union[str, Path], new_data: Mapping[str, np.ndarray]) -> Path:
    """Update or create an .npz file by merging in new arrays."""
    p = Path(path)
    if p.suffix != ".npz":
        p = p.with_suffix(".npz")
    _ensure_parent_dir(p)
    merged: Dict[str, np.ndarray] = {}
    if p.exists():
        with np.load(p, allow_pickle=False) as data:
            merged.update({k: data[k] for k in data.files})
    merged.update({k: np.asarray(v) for k, v in new_data.items()})
    np.savez(p, **merged)
    return p


def _write_attrs(f, attrs: Mapping[str, Jsonable] | None) -> None:
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
    path: Union[str, Path],
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
    import h5py  # local import to avoid hard dependency at import time

    is_dataframe = HAS_PANDAS and "DataFrame" in type(data).__name__
    data_type = "DataFrame" if is_dataframe else "array"
    
    logger.info(
        f"Saving {data_type} to HDF5: {path}, mode='{mode}', "
        f"compression='{compression}', has_labels={labels is not None}, has_attrs={attrs is not None}"
    )

    with h5py.File(path, mode) as f:
        _write_attrs(f, attrs)

        if is_dataframe:
            grp = f.create_group("data") if "data" not in f else f["data"]
            for key in list(grp.keys()):
                del grp[key]
            _save_dataframe(grp, data, compression, compression_opts)  # type: ignore[arg-type]
            logger.info(f"Saved DataFrame with shape {data.shape} to {path}")  # type: ignore
        else:
            arr = np.asarray(data)
            if "data" in f:
                del f["data"]
            f.create_dataset("data", data=arr, compression=compression, compression_opts=compression_opts)
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
    path: Union[str, Path],
    *,
    filter_pairs: Iterable[Tuple[Any, Any]] | None = None,
    return_attrs: bool = False,
):
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

    logger.info(f"Loading HDF5 from {p}, filter_pairs={filter_pairs is not None}, return_attrs={return_attrs}")

    import h5py  # local import

    data_out: Any = None
    labels_out: Any = []
    attrs_out: Dict[str, Jsonable] = {}

    with h5py.File(p, "r") as f:
        # attrs
        for k, v in f.attrs.items():
            if isinstance(v, (np.bytes_, bytes, bytearray)):
                txt = v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else v.tobytes().decode("utf-8")
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
                if HAS_PANDAS and all(k in obj.keys() for k in ("columns", "index")):
                    df = _load_dataframe(obj)  # type: ignore[assignment]
                    logger.info(f"Loaded DataFrame with shape {df.shape} from {p}")  # type: ignore
                    if filter_pairs is not None and HAS_PANDAS and set(["item_i", "item_j"]).issubset(df.columns):
                        wanted = set(filter_pairs)
                        pairs = list(zip(df["item_i"].tolist(), df["item_j"].tolist()))
                        mask = np.array([pair in wanted for pair in pairs])
                        n_before = len(df)  # type: ignore
                        df = df[mask].reset_index(drop=True)  # type: ignore
                        logger.info(f"Filtered DataFrame from {n_before} to {len(df)} rows using {len(filter_pairs)} pairs")  # type: ignore
                    data_out = df
                else:
                    # Unsupported group layout -> load children as dict of arrays
                    child = {k: obj[k][...] for k in obj.keys()}
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


def h5io(
    path: Union[str, Path],
    *,
    task: str,
    data: Any | None = None,
    labels: Any | None = None,
    attrs: Mapping[str, Jsonable] | None = None,
    mode: str = "w",
) -> Union[Tuple[Any, Any], None]:
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
        return load_hdf5(path)
    else:  # pragma: no cover - defensive
        raise ValueError("task must be either 'save' or 'load'")


__all__ = [
    "save_array",
    "load_array",
    "update_array",
    "save_hdf5",
    "load_hdf5",
    "h5io",
]
