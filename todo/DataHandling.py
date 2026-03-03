"""DataHandling -- persistence & pipeline for the SERBRA analysis stack.

Responsibilities
----------------
* **Animal-info H5 store** -- single-file store for per-task metadata,
  full cell-analysis DataFrames, binarized neural activity, and rich
  Animal / Session / Task metadata dictionaries.
* **AnimalInfoStore** -- lazy-loading accessor class that opens the H5,
  loads only the lightweight index on init, and fetches cells /
  binarized / metadata / behavioral data on demand with in-memory
  caching.  Supports a ``release=True`` option on load methods to free
  RAM immediately after retrieval.
* **Batched H5 writing** -- :func:`append_batch_to_h5` and
  :func:`finalize_h5_index` support incremental construction of the H5
  file in RAM-friendly batches.
* **Disparity pipeline** -- merge *N* shape-similarity runs into one
  enhanced long-format DataFrame.

All public helpers use :func:`save_df` / :func:`load_df` for lightweight
CSV I/O and :class:`AnimalInfoStore` for the heavy H5 store.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import h5py
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Low-level CSV helpers (kept for backward compat & simple artefacts)
# ═══════════════════════════════════════════════════════════════════════════

def save_df(df: pd.DataFrame, filepath: Union[str, Path]) -> None:
    """Persist a DataFrame to CSV."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath)
    print(f"Saved dataframe to {filepath}")


def load_df(filepath: Union[str, Path]) -> pd.DataFrame:
    """Load a DataFrame from CSV."""
    filepath = Path(filepath)
    df = pd.read_csv(filepath, index_col=0)
    print(f"Loaded dataframe from {filepath}")
    return df


def load_or_save_single_df(
    filepath: Union[str, Path],
    df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Load a CSV if it exists, otherwise save *df* and return it."""
    filepath = Path(filepath)
    if filepath.exists():
        return load_df(filepath)
    save_df(df, filepath)
    return df


def save_or_load_pickle(
    filepath: Union[str, Path],
    data: Any = None,
) -> Any:
    """Save *data* to a pickle file, or load when *data* is ``None``."""
    import pickle

    filepath = Path(filepath)
    if data is not None:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as fh:
            pickle.dump(data, fh)
        print(f"Saved data to {filepath}")
        return data

    if filepath.exists():
        with open(filepath, "rb") as fh:
            loaded = pickle.load(fh)
        print(f"Loaded data from {filepath}")
        return loaded
    raise FileNotFoundError(f"Pickle file not found: {filepath}")


# ═══════════════════════════════════════════════════════════════════════════
# Animal-info H5 store  (single file, hierarchical keys via h5py + pickle)
# ═══════════════════════════════════════════════════════════════════════════
# Layout inside the H5 file:
#   /index                        - pickled pd.DataFrame (scalar metadata)
#   /cells/<task_key>/<ct>        - pickled pd.DataFrame (full cell data)
#   /binarized/<task_key>/<model> - np.ndarray (binarized neural activity)
#   /metadata/<task_key>          - pickled dict (full Animal/Session/Task meta)
#   /behavior/<task_key>/<source>  - np.ndarray (behavioural data, gzip compressed)
#       where <source> is one of: position, distance, stimulus,
#       velocity, acceleration, moving
#
# All DataFrames are serialised via ``pickle`` and stored as opaque byte
# blobs using ``h5py``.  This avoids pytables performance warnings for
# object-dtype columns (which contain variable-size numpy arrays) and
# guarantees a clean round-trip for any column type.
#
# Keys are sanitised so that HDF5-unsafe characters (``-``, space) become
# underscores while the original ``task_id`` is recoverable from the
# *index* DataFrame.

_H5_INDEX_KEY = "index"
_H5_CELLS_PREFIX = "cells"
_H5_BINARIZED_PREFIX = "binarized"
_H5_METADATA_PREFIX = "metadata"
_H5_BEHAVIOR_PREFIX = "behavior"

# All behavioural data sources recognised by the pipeline.
# Matches ``Datasets_Behavior.Sources`` in core/Datasets.py.
# behavioural data source names.  kept as a runtime tuple for iteration
# but also exposed as a Literal type for static typing.  deriving both from
# a single definition avoids drift.
BehaviorSource = Literal[
    "position",
    "distance",
    "stimulus",
    "velocity",
    "acceleration",
    "moving",
]

# runtime tuple used by pipelines and H5 keys.
BEHAVIOR_SOURCES: Tuple[BehaviorSource, ...] = tuple(BehaviorSource.__args__)  # type: ignore


def _sanitize_h5_key(raw: str) -> str:
    """Make a string safe to use as an HDF5 group/dataset name."""
    return raw.replace("-", "_").replace(" ", "_")


def _pickle_to_h5(
    group: h5py.Group,
    key: str,
    obj: Any,
) -> None:
    """Serialize *obj* via pickle and write as an opaque byte dataset."""
    import pickle

    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    group.create_dataset(key, data=np.void(data))


def _unpickle_from_h5(group: h5py.Group, key: str) -> Any:
    """Read an opaque byte dataset and deserialize via pickle."""
    import pickle

    raw = group[key][()].tobytes()
    return pickle.loads(raw)


def save_animal_info_h5(
    h5_path: Union[str, Path],
    index_df: pd.DataFrame,
    cell_dfs: Dict[str, pd.DataFrame],
    binarized_data: Optional[Dict[str, np.ndarray]] = None,
    task_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    *,
    retries: int = 3,
    retry_delay: float = 1.0,
) -> None:
    """Write the index, cell DataFrames, binarized arrays, and metadata
    into *one* HDF5 file.

    Uses a temp-file + atomic-move pattern so a crash never leaves a
    corrupt target file.

    Parameters
    ----------
    h5_path : Path
        Destination ``.h5`` file.
    index_df : DataFrame
        Flat per-task metadata (scalar columns only).
    cell_dfs : dict
        ``"<sanitised_task_id>/<cell_type>"`` -> full cell DataFrame.
    binarized_data : dict, optional
        ``"<sanitised_task_id>/<model_name>"`` -> 2-D numpy array.
    task_metadata : dict, optional
        ``"<sanitised_task_id>"`` -> dict with full Animal/Session/Task metadata.
    retries / retry_delay :
        Retry logic for file-system contention (e.g. antivirus locks).
    """
    h5_path = Path(h5_path).with_suffix(".h5")
    h5_path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp = tempfile.mkstemp(suffix=".h5", dir=h5_path.parent)
    os.close(fd)
    tmp = Path(tmp)

    for attempt in range(retries):
        try:
            with h5py.File(tmp, "w") as f:
                # -- index (pickled DataFrame) ----------------------------
                _pickle_to_h5(f, _H5_INDEX_KEY, index_df)

                # -- cell DataFrames (pickled) ----------------------------
                if cell_dfs:
                    cells_grp = f.create_group(_H5_CELLS_PREFIX)
                    for sub_key, cdf in cell_dfs.items():
                        _pickle_to_h5(cells_grp, sub_key, cdf)

                # -- binarized neural activity (raw numpy) ----------------
                if binarized_data:
                    bin_grp = f.create_group(_H5_BINARIZED_PREFIX)
                    for sub_key, arr in binarized_data.items():
                        bin_grp.create_dataset(
                            sub_key,
                            data=arr,
                            compression="gzip",
                            compression_opts=4,
                        )

                # -- full task metadata (pickled dicts) -------------------
                if task_metadata:
                    meta_grp = f.create_group(_H5_METADATA_PREFIX)
                    for sub_key, meta in task_metadata.items():
                        _pickle_to_h5(meta_grp, sub_key, meta)

            shutil.move(str(tmp), str(h5_path))
            n_cells = len(cell_dfs) if cell_dfs else 0
            n_bin = len(binarized_data) if binarized_data else 0
            print(
                f"Saved animal info H5 ({n_cells} cell DFs, "
                f"{n_bin} binarized arrays) -> {h5_path}"
            )
            return
        except (OSError, PermissionError) as exc:
            logger.warning(
                "Attempt %d/%d saving %s: %s",
                attempt + 1, retries, h5_path, exc,
            )
            if attempt == retries - 1:
                tmp.unlink(missing_ok=True)
                raise
            time.sleep(retry_delay)
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)


def append_batch_to_h5(
    h5_path: Union[str, Path],
    cell_dfs: Optional[Dict[str, pd.DataFrame]] = None,
    binarized_data: Optional[Dict[str, np.ndarray]] = None,
    task_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    behavior_data: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
    *,
    replace: bool = False,
) -> None:
    """Append one batch of extracted data to an existing (or new) H5 file.

    Opens the file in *append* mode (``"a"``) and writes only the heavy
    payload — cell DataFrames, binarized arrays, metadata dicts, and
    behavioural arrays.  The lightweight index is written separately by
    :func:`finalize_h5_index` after all batches are done.

    Parameters
    ----------
    h5_path : Path
        Destination file (created if absent).  Does **not** enforce a
        ``.h5`` suffix so it can target staging files.
    cell_dfs : dict, optional
        ``"<sanitised_task_id>/<cell_type>"`` -> cell DataFrame.
    binarized_data : dict, optional
        ``"<sanitised_task_id>/<model_name>"`` -> 2-D numpy array.
    task_metadata : dict, optional
        ``"<sanitised_task_id>"`` -> metadata dict.
    behavior_data : dict, optional
        ``"<sanitised_task_id>"`` -> ``{source_name: np.ndarray}``.
        Source names are from :data:`BEHAVIOR_SOURCES`
        (position, distance, stimulus, velocity, acceleration, moving).
    replace : bool
        If *True*, existing datasets at the same key are deleted before
        writing the new data.  When *False* (default), attempting to
        write a key that already exists raises an error.
    """
    h5_path = Path(h5_path)
    h5_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "a") as f:
        if cell_dfs:
            cells_grp = f.require_group(_H5_CELLS_PREFIX)
            for sub_key, cdf in cell_dfs.items():
                if replace and sub_key in cells_grp:
                    del cells_grp[sub_key]
                _pickle_to_h5(cells_grp, sub_key, cdf)

        if binarized_data:
            bin_grp = f.require_group(_H5_BINARIZED_PREFIX)
            for sub_key, arr in binarized_data.items():
                if replace and sub_key in bin_grp:
                    del bin_grp[sub_key]
                bin_grp.create_dataset(
                    sub_key,
                    data=arr,
                    compression="gzip",
                    compression_opts=4,
                )

        if task_metadata:
            meta_grp = f.require_group(_H5_METADATA_PREFIX)
            for sub_key, meta in task_metadata.items():
                if replace and sub_key in meta_grp:
                    del meta_grp[sub_key]
                _pickle_to_h5(meta_grp, sub_key, meta)

        if behavior_data:
            beh_grp = f.require_group(_H5_BEHAVIOR_PREFIX)
            for task_key, sources in behavior_data.items():
                task_grp = beh_grp.require_group(task_key)
                for source_name, arr in sources.items():
                    if replace and source_name in task_grp:
                        del task_grp[source_name]
                    task_grp.create_dataset(
                        source_name,
                        data=arr,
                        compression="gzip",
                        compression_opts=4,
                    )


def finalize_h5_index(
    h5_path: Union[str, Path],
    index_df: pd.DataFrame,
) -> None:
    """Write (or overwrite) the index DataFrame in the H5 file.

    Called once after all batches have been appended via
    :func:`append_batch_to_h5`.

    Raises ``ValueError`` if the index contains duplicate ``task_id``
    values.
    """
    h5_path = Path(h5_path)

    # ---- duplicate task_id safety check --------------------------------
    if "task_id" in index_df.columns:
        dups = index_df["task_id"][index_df["task_id"].duplicated()]
        if not dups.empty:
            raise ValueError(
                f"Duplicate task_id(s) in index: {sorted(dups.unique())}. "
                f"Each task_id must appear exactly once."
            )

    with h5py.File(h5_path, "a") as f:
        if _H5_INDEX_KEY in f:
            del f[_H5_INDEX_KEY]
        _pickle_to_h5(f, _H5_INDEX_KEY, index_df)


# ═══════════════════════════════════════════════════════════════════════════
# AnimalInfoStore -- lazy-loading class for the animal-info H5
# ═══════════════════════════════════════════════════════════════════════════


class AnimalInfoStore:
    """Lazy-loading accessor for a SERBRA animal-info HDF5 file.

    Designed for a **two-phase** workflow:

    1. **Fast open** -- only the lightweight *index* DataFrame (one row
       per task, scalar columns) is loaded when the store is created.
    2. **On-demand fetch** -- cell DataFrames, binarized activity arrays,
       and full metadata dicts are loaded per-task when requested, then
       cached in memory for instant re-use.

    Supports **filtering** by animal, date, or task to operate on
    subsets, and **bulk loading** with ``include`` / ``exclude``
    parameters to skip expensive data such as binarized arrays.

    Parameters
    ----------
    h5_path : str or Path
        Path to the ``.h5`` file produced by
        :meth:`Mother.build_animal_info`.

    Examples
    --------
    >>> store = AnimalInfoStore("animal_info.h5")
    >>> store.index.head()                       # fast -- already loaded
    >>> cells = store.load_cells("DON-007021_20211022_FS1", "all")
    >>> store.load_all(exclude=["binarized"])     # skip large arrays
    >>> store.load_all(animal_ids=["DON-007021"]) # one animal only
    """

    # Valid category names for include / exclude parameters
    CATEGORIES: FrozenSet[str] = frozenset(
        {"cells", "binarized", "metadata", "behavior"}
    )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, h5_path: Union[str, Path]) -> None:
        self._h5_path = Path(h5_path).with_suffix(".h5")
        if not self._h5_path.exists():
            raise FileNotFoundError(f"H5 store not found: {self._h5_path}")

        # Fast-load the index (typically < 20 ms even for large stores)
        with h5py.File(self._h5_path, "r") as f:
            self._index: pd.DataFrame = _unpickle_from_h5(f, _H5_INDEX_KEY)

        # Per-category in-memory caches (populated lazily)
        self._cells: Dict[Tuple[str, str], pd.DataFrame] = {}
        self._binarized: Dict[str, Dict[str, np.ndarray]] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._behavior: Dict[str, Dict[str, np.ndarray]] = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def h5_path(self) -> Path:
        """Absolute path to the backing HDF5 file."""
        return self._h5_path

    @property
    def index(self) -> pd.DataFrame:
        """Summary index DataFrame (one row per task, scalar columns)."""
        return self._index

    @property
    def task_ids(self) -> List[str]:
        """All ``task_id`` values in the index."""
        return self._index["task_id"].tolist()

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def filter_index(
        self,
        *,
        animal_ids: Optional[List[str]] = None,
        dates: Optional[List[str]] = None,
        tasks: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Return a filtered *copy* of the index.

        Parameters
        ----------
        animal_ids : list of str, optional
            Keep rows where ``animal_id`` is in this list.
        dates : list of str, optional
            Keep rows where ``date`` is in this list.
        tasks : list of str, optional
            Keep rows where ``task_name`` is in this list.

        Returns
        -------
        pd.DataFrame
        """
        df = self._index
        if animal_ids is not None:
            df = df[df["animal_id"].isin(animal_ids)]
        if dates is not None:
            df = df[df["date"].isin(dates)]
        if tasks is not None:
            df = df[df["task_name"].isin(tasks)]
        return df

    def get_task_ids(
        self,
        *,
        animal_ids: Optional[List[str]] = None,
        dates: Optional[List[str]] = None,
        tasks: Optional[List[str]] = None,
    ) -> List[str]:
        """Return ``task_id`` values matching the given filters.

        Parameters are identical to :meth:`filter_index`.
        """
        return self.filter_index(
            animal_ids=animal_ids, dates=dates, tasks=tasks,
        )["task_id"].tolist()

    # ------------------------------------------------------------------
    # Single-item loading (with transparent caching)
    # ------------------------------------------------------------------

    def load_cells(
        self,
        task_id: str,
        cell_type: str,
        *,
        release: bool = False,
    ) -> Optional[pd.DataFrame]:
        """Load the full cell DataFrame for one ``(task, cell_type)`` pair.

        The result is cached in memory -- subsequent calls for the same
        key return instantly without touching the H5 file (unless
        *release* was used on the previous call).

        Parameters
        ----------
        task_id : str
            The task identifier (unsanitised is fine).
        cell_type : str
            Cell category, e.g. ``"all"``, ``"place"``, ``"non-place"``.
        release : bool
            If *True*, the returned value is removed from the in-memory
            cache immediately after retrieval, freeing RAM.  The next
            call for the same key will re-read from the H5 file.

        Returns ``None`` when the requested key does not exist.
        """
        key = (task_id, cell_type)
        if key in self._cells:
            return self._cells.pop(key) if release else self._cells[key]

        # Not cached -- read from H5
        h5_key = (
            f"{_H5_CELLS_PREFIX}/{_sanitize_h5_key(task_id)}/"
            f"{_sanitize_h5_key(cell_type)}"
        )
        try:
            with h5py.File(self._h5_path, "r") as f:
                if h5_key not in f:
                    logger.debug("Key %s not found in %s", h5_key, self._h5_path)
                    return None
                df = _unpickle_from_h5(f, h5_key)
        except Exception:
            logger.warning(
                "Failed to load %s from %s", h5_key, self._h5_path,
                exc_info=True,
            )
            return None

        if not release:
            self._cells[key] = df
        return df

    def load_binarized(
        self,
        task_id: str,
        model_name: Optional[str] = None,
        *,
        release: bool = False,
    ) -> Optional[Union[np.ndarray, Dict[str, np.ndarray]]]:
        """Load binarized neural activity for one task.

        If *model_name* is given, returns a single ``ndarray``.
        Otherwise returns a ``{model_name: ndarray}`` dict for all
        models under this task.

        The full per-task dict is cached on first access.

        Parameters
        ----------
        task_id : str
            The task identifier (unsanitised is fine).
        model_name : str, optional
            Specific model to retrieve.
        release : bool
            If *True*, the cached entry for this *task_id* is removed
            from memory after the return value is prepared.
        """
        if task_id not in self._binarized:
            prefix = f"{_H5_BINARIZED_PREFIX}/{_sanitize_h5_key(task_id)}"
            try:
                with h5py.File(self._h5_path, "r") as f:
                    if prefix not in f:
                        return None
                    grp = f[prefix]
                    self._binarized[task_id] = {
                        name: np.array(ds) for name, ds in grp.items()
                    }
            except Exception:
                logger.warning(
                    "Failed to load binarized %s from %s",
                    task_id, self._h5_path, exc_info=True,
                )
                return None

        cached = self._binarized.get(task_id)
        if cached is None:
            return None

        if release:
            cached = self._binarized.pop(task_id)

        if model_name is not None:
            return cached.get(_sanitize_h5_key(model_name))
        return cached

    def load_metadata(
        self,
        task_id: str,
        *,
        release: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Load the full Animal / Session / Task metadata dict.

        The result is cached in memory on first access.

        Parameters
        ----------
        task_id : str
            The task identifier.
        release : bool
            If *True*, the cached entry is removed from memory after
            retrieval.
        """
        if task_id not in self._metadata:
            h5_key = f"{_H5_METADATA_PREFIX}/{_sanitize_h5_key(task_id)}"
            try:
                with h5py.File(self._h5_path, "r") as f:
                    if h5_key not in f:
                        return None
                    self._metadata[task_id] = _unpickle_from_h5(f, h5_key)
            except Exception:
                logger.warning(
                    "Failed to load metadata %s from %s",
                    h5_key, self._h5_path, exc_info=True,
                )
                return None

        if release:
            return self._metadata.pop(task_id, None)
        return self._metadata.get(task_id)

    # re-export the module-level Literal so callers can import the type
    # directly from the class if they prefer.  this avoids duplicating the
    # definition and keeps mypy/pyright happy.
    def load_behavior(
        self,
        task_id: str,
        source_name: Optional[BehaviorSource] = None,
        *,
        release: bool = False,
    ) -> Optional[Union[np.ndarray, Dict[BehaviorSource, np.ndarray]]]:
        """Load behavioural data for one task.

        If *source_name* defined in :data:`BEHAVIOR_SOURCES` is given
        (e.g. ``"position"``, ``"velocity"``), returns a single ``ndarray``.
        Otherwise returns a dict ``{source: ndarray}`` with all available
        sources for that task.

        The full per-task dict is cached on first access.

        Parameters
        ----------
        task_id : str
            The task identifier.
        source_name : str, optional
            One of the values listed in :data:`BEHAVIOR_SOURCES`.
            ``None`` returns all.
        release : bool
            If *True*, the cached entry is removed from memory after
            retrieval.
        """
        if task_id not in self._behavior:
            prefix = f"{_H5_BEHAVIOR_PREFIX}/{_sanitize_h5_key(task_id)}"
            try:
                with h5py.File(self._h5_path, "r") as f:
                    if prefix not in f:
                        return None
                    grp = f[prefix]
                    self._behavior[task_id] = {
                        name: np.array(ds) for name, ds in grp.items()
                    }
            except Exception:
                logger.warning(
                    "Failed to load behavior %s from %s",
                    prefix, self._h5_path, exc_info=True,
                )
                return None

        cached = self._behavior.get(task_id)
        if cached is None:
            return None

        if release:
            cached = self._behavior.pop(task_id)

        if source_name is not None:
            return cached.get(source_name)
        return cached

    # Backward-compat alias
    def load_position(
        self,
        task_id: str,
        *,
        release: bool = False,
    ) -> Optional[np.ndarray]:
        """Shorthand for ``load_behavior(task_id, "position", ...)``.

        .. deprecated:: Use :meth:`load_behavior` instead.
        """
        return self.load_behavior(task_id, "position", release=release)

    # ------------------------------------------------------------------
    # Bulk loading
    # ------------------------------------------------------------------

    def load_all(
        self,
        *,
        include: Optional[Sequence[str]] = None,
        exclude: Optional[Sequence[str]] = None,
        animal_ids: Optional[List[str]] = None,
        dates: Optional[List[str]] = None,
        tasks: Optional[List[str]] = None,
        cell_types: Sequence[str] = ("all", "place", "non-place"),
    ) -> "AnimalInfoStore":
        """Bulk-load data into the in-memory cache.

        Opens the HDF5 file **once** and reads all requested data in a
        single pass, which is substantially faster than many individual
        :meth:`load_cells` / :meth:`load_binarized` calls.

        Parameters
        ----------
        include : sequence of str, optional
            Data categories to load: ``"cells"``, ``"binarized"``,
            ``"metadata"``.  Loads **all** categories when *None*.
        exclude : sequence of str, optional
            Categories to **skip**.  E.g. ``exclude=["binarized"]``
            avoids loading the large activity arrays.
        animal_ids, dates, tasks :
            Filters forwarded to :meth:`get_task_ids` to select which
            tasks to load data for.
        cell_types : sequence of str
            Cell types to load (only relevant for the ``"cells"``
            category).

        Returns
        -------
        AnimalInfoStore
            *self* -- for method chaining.
        """
        categories = self._resolve_categories(include, exclude)
        target_ids = self.get_task_ids(
            animal_ids=animal_ids, dates=dates, tasks=tasks,
        )
        self._bulk_read(target_ids, categories, cell_types)
        return self

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def cache_info(self) -> Dict[str, int]:
        """Return the number of cached items per category."""
        return {
            "cells": len(self._cells),
            "binarized": len(self._binarized),
            "metadata": len(self._metadata),
            "behavior": len(self._behavior),
        }

    def clear_cache(self, category: Optional[str] = None) -> None:
        """Drop cached data to free memory.

        Parameters
        ----------
        category : str, optional
            ``"cells"``, ``"binarized"``, ``"metadata"``, or
            ``"behavior"``.  Clears **all** categories when *None*.
        """
        if category is None or category == "cells":
            self._cells.clear()
        if category is None or category == "binarized":
            self._binarized.clear()
        if category is None or category == "metadata":
            self._metadata.clear()
        if category is None or category == "behavior":
            self._behavior.clear()

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    def keys(self) -> List[str]:
        """List every HDF5 key in the backing file."""
        with h5py.File(self._h5_path, "r") as f:

            def _walk(grp: h5py.Group, prefix: str = "/") -> List[str]:
                result: List[str] = []
                for name in grp:
                    full = f"{prefix}{name}"
                    if isinstance(grp[name], h5py.Group):
                        result.extend(_walk(grp[name], f"{full}/"))
                    else:
                        result.append(full)
                return result

            return _walk(f)

    def summary(self) -> str:
        """Return a human-readable summary of store contents."""
        ci = self.cache_info()
        n_keys = len(self.keys())
        lines = [
            f"AnimalInfoStore: {self._h5_path}",
            f"  Index rows   : {len(self._index)}",
            f"  H5 keys      : {n_keys}",
            f"  Cached cells : {ci['cells']}",
            f"  Cached binar.: {ci['binarized']}",
            f"  Cached meta  : {ci['metadata']}",
            f"  Cached behav.: {ci['behavior']}",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        ci = self.cache_info()
        return (
            f"AnimalInfoStore({self._h5_path.name!r}, "
            f"tasks={len(self._index)}, "
            f"cached: cells={ci['cells']}, "
            f"binarized={ci['binarized']}, "
            f"metadata={ci['metadata']}, "
            f"behavior={ci['behavior']})"
        )

    def __len__(self) -> int:
        """Number of tasks (rows) in the index."""
        return len(self._index)

    # ------------------------------------------------------------------
    # Extending an existing H5 store
    # ------------------------------------------------------------------

    def extend(
        self,
        new_index_rows: pd.DataFrame,
        cell_dfs: Optional[Dict[str, pd.DataFrame]] = None,
        binarized_data: Optional[Dict[str, np.ndarray]] = None,
        task_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
        behavior_data: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
        *,
        replace: bool = False,
    ) -> None:
        """Extend the backing H5 file with new data without recreating it.

        Appends data for new tasks (or replaces existing data when
        *replace=True*) and merges *new_index_rows* into the stored
        index.  The in-memory index is refreshed automatically.

        Parameters
        ----------
        new_index_rows : DataFrame
            Rows to merge into the index.  Must contain a ``task_id``
            column.
        cell_dfs, binarized_data, task_metadata :
            Same format as :func:`append_batch_to_h5`.
        behavior_data : dict, optional
            ``"<sanitised_task_id>"`` -> ``{source_name: ndarray}``.
        replace : bool
            If *True*, existing datasets at colliding keys are
            overwritten.  When *False*, duplicate ``task_id`` entries
            raise a ``ValueError``.
        """
        # ---- duplicate safety ----------------------------------------
        if "task_id" not in new_index_rows.columns:
            raise ValueError("new_index_rows must contain a 'task_id' column")

        incoming_ids = set(new_index_rows["task_id"])
        existing_ids = set(self._index["task_id"])
        duplicates = incoming_ids & existing_ids

        if duplicates and not replace:
            raise ValueError(
                f"Duplicate task_id(s) already in the H5 store: "
                f"{sorted(duplicates)}.  Use replace=True to overwrite."
            )

        # ---- append payload to H5 ------------------------------------
        append_batch_to_h5(
            self._h5_path,
            cell_dfs=cell_dfs,
            binarized_data=binarized_data,
            task_metadata=task_metadata,
            behavior_data=behavior_data,
            replace=replace,
        )

        # ---- merge index ---------------------------------------------
        if duplicates:
            # Remove old rows for tasks that will be replaced
            self._index = self._index[
                ~self._index["task_id"].isin(duplicates)
            ]
        merged_index = pd.concat(
            [self._index, new_index_rows], ignore_index=True,
        )
        finalize_h5_index(self._h5_path, merged_index)
        self._index = merged_index

        # ---- invalidate caches for replaced tasks --------------------
        if duplicates:
            for tid in duplicates:
                self._metadata.pop(tid, None)
                self._binarized.pop(tid, None)
                self._behavior.pop(tid, None)
                # cells are keyed (task_id, cell_type) — remove all
                for key in [k for k in self._cells if k[0] == tid]:
                    del self._cells[key]

    # ------------------------------------------------------------------
    # Metadata enrichment (disparity workflow)
    # ------------------------------------------------------------------

    def enhance_disparity_df(
        self,
        long_df: pd.DataFrame,
        output_dir: Union[str, Path],
        output_filename: str,
        *,
        regenerate: bool = False,
        task_groups: Optional[Dict[str, Dict[str, List[str]]]] = None,
        group_by: str = "condition",
        compare_by: str = "task",
    ) -> pd.DataFrame:
        """Add condition, group, colour, and cell-count metadata to a disparity DF.

        This function enriches the disparity dataframe with metadata from the
        AnimalInfoStore's index. It applies `improve_animals_df` to generate
        group and color information based on task groupings.

        Parameters
        ----------
        long_df : DataFrame
            Merged long-format disparity dataframe with columns:
            animal_id_i/j, task_name_i/j, date_i/j, and disparity columns.
        output_dir : Path
            Directory where the enhanced CSV is cached.
        output_filename : str
            Name of the enhanced CSV file.
        regenerate : bool
            Force recomputation even if a cached CSV exists.
        task_groups : dict, optional
            Task grouping structure for `improve_animals_df`.
            If None, auto-grouping is applied.
        group_by : str
            Column for primary grouping (default: "condition").
        compare_by : str
            Column for subgroup comparison (default: "task").

        Returns
        -------
        pd.DataFrame
            Enhanced disparity DataFrame with added columns:
            - condition_i/j: experimental conditions
            - group_i/j: group identifiers
            - color_i/j: RGBA colors 
            - place_cell_count_i/j: place cell counts
            - non_place_cell_count_i/j: non-place cell counts
            - combined_color: averaged color for the pair
        """
        from temporary import add_conditions_and_colors
        from Classes import Mother

        output_dir = Path(output_dir)
        output_path = output_dir / output_filename
        if output_path.exists() and not regenerate:
            return load_df(output_path)

        index_df = self._index.copy()

        # Apply improve_animals_df to add group and color columns.
        # improve_animals_df requires: animal_id, task, task_name,
        # task_number, date, <group_by>, <compare_by>.
        # The AnimalInfoStore index may lack 'task', 'task_number' -- derive them.
        try:
            temp_df = index_df.copy()

            # Derive 'task' column (alias of task_name) if missing
            if "task" not in temp_df.columns and "task_name" in temp_df.columns:
                temp_df["task"] = temp_df["task_name"]

            # Derive 'task_number' from task_name (e.g. 'FS1' -> 1)
            if "task_number" not in temp_df.columns and "task_name" in temp_df.columns:
                temp_df["task_number"] = (
                    temp_df["task_name"]
                    .str.extract(r'(\d+)$', expand=False)
                    .astype(float)
                )

            # Final check for required columns
            required_cols = ["animal_id", "task", "task_name", "task_number", "date", group_by]
            missing = [c for c in required_cols if c not in temp_df.columns]
            if missing:
                print(
                    f"Warning: Cannot run improve_animals_df, missing columns: {missing}\n"
                    f"  Available: {list(temp_df.columns)}\n"
                    f"  Proceeding without group/color generation."
                )
            else:
                improved_df = Mother.improve_animals_df(
                    df=temp_df,
                    task_groups=task_groups,
                    group_by=group_by,
                    compare_by=compare_by,
                )

                # Copy back derived columns to index_df
                for col in ["group_key", "group_name", "color",
                            "task", "task_number", "group_task_number"]:
                    if col in improved_df.columns:
                        index_df[col] = improved_df[col].values

        except Exception as e:
            import traceback
            print(f"Warning: Failed to apply improve_animals_df: {e}")
            traceback.print_exc()
            print("Proceeding without group/color information")

        # Build lookup helpers keyed by (animal_id, task_name)
        def _build_task_data_proxy(
            idx_df: pd.DataFrame,
            cell_count_col: str,
        ) -> pd.DataFrame:
            """Create proxy dataframe for add_conditions_and_colors.
            
            Parameters
            ----------
            idx_df : DataFrame
                Source index dataframe
            cell_count_col : str
                Column name to use for cell_count
            """
            # Check for required base columns with flexible naming
            # Support both "animal_id"/"animal" and "task_name"/"task"
            animal_col = "animal_id" if "animal_id" in idx_df.columns else "animal"
            task_col = "task_name" if "task_name" in idx_df.columns else "task"
            
            # Verify required columns exist
            required_cols = [animal_col, task_col, "condition"]
            missing = [c for c in required_cols if c not in idx_df.columns]
            if missing:
                raise ValueError(
                    f"Index DataFrame is missing required columns: {missing}\n"
                    f"Available columns: {list(idx_df.columns)}"
                )
            
            # Select base columns
            base_cols = [animal_col, task_col, "condition"]
            
            # Add optional columns if they exist
            optional_cols = ["group_key", "group_name", "color"]
            available_optional = [c for c in optional_cols if c in idx_df.columns]
            
            proxy = idx_df[base_cols + available_optional].copy()
            
            # Rename to expected format (normalize column names)
            proxy.rename(
                columns={animal_col: "animal", task_col: "task"},
                inplace=True,
            )
            
            # Add cell count if available
            if cell_count_col in idx_df.columns:
                proxy["cell_count"] = idx_df[cell_count_col]
            
            return proxy

        # Create proxies for place and non-place cells
        place_proxy = _build_task_data_proxy(index_df, "place_cell_count")
        notplace_proxy = _build_task_data_proxy(index_df, "non_place_cell_count")

        # Enrich with metadata
        enhanced = add_conditions_and_colors(
            long_df,
            place_proxy,
            notplace_proxy,
            add_cell_counts=True,
        )

        save_df(enhanced, output_path)
        return enhanced

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_categories(
        include: Optional[Sequence[str]],
        exclude: Optional[Sequence[str]],
    ) -> Set[str]:
        """Compute the final set of data categories to operate on."""
        cats = set(include or AnimalInfoStore.CATEGORIES)
        invalid = cats - AnimalInfoStore.CATEGORIES
        if invalid:
            raise ValueError(
                f"Unknown categories: {invalid}.  "
                f"Valid: {sorted(AnimalInfoStore.CATEGORIES)}"
            )
        if exclude:
            cats -= set(exclude)
        return cats

    def _bulk_read(
        self,
        task_ids: List[str],
        categories: Set[str],
        cell_types: Sequence[str],
    ) -> None:
        """Single-pass read of the H5 file for the requested data.

        Opens the file once and iterates over the target tasks,
        populating the in-memory caches for items not already loaded.
        """
        with h5py.File(self._h5_path, "r") as f:
            for tid in task_ids:
                skey = _sanitize_h5_key(tid)

                # -- cells ------------------------------------------------
                if "cells" in categories:
                    for ct in cell_types:
                        cache_key = (tid, ct)
                        if cache_key in self._cells:
                            continue
                        h5_key = (
                            f"{_H5_CELLS_PREFIX}/{skey}/"
                            f"{_sanitize_h5_key(ct)}"
                        )
                        if h5_key in f:
                            self._cells[cache_key] = _unpickle_from_h5(
                                f, h5_key,
                            )

                # -- binarized --------------------------------------------
                if "binarized" in categories and tid not in self._binarized:
                    prefix = f"{_H5_BINARIZED_PREFIX}/{skey}"
                    if prefix in f:
                        grp = f[prefix]
                        self._binarized[tid] = {
                            name: np.array(ds) for name, ds in grp.items()
                        }

                # -- metadata ---------------------------------------------
                if "metadata" in categories and tid not in self._metadata:
                    h5_key = f"{_H5_METADATA_PREFIX}/{skey}"
                    if h5_key in f:
                        self._metadata[tid] = _unpickle_from_h5(f, h5_key)

                # -- behavior ---------------------------------------------
                if "behavior" in categories and tid not in self._behavior:
                    prefix = f"{_H5_BEHAVIOR_PREFIX}/{skey}"
                    if prefix in f:
                        grp = f[prefix]
                        self._behavior[tid] = {
                            name: np.array(ds) for name, ds in grp.items()
                        }


# ═══════════════════════════════════════════════════════════════════════════
# Disparity helpers (pivot, H5 loading, merge, enhance)
# ═══════════════════════════════════════════════════════════════════════════

def pivot_disparity_matrix(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Convert a long-format disparity DF to a symmetric pivot table."""
    d_matrix = df_raw.pivot(index="item_i", columns="item_j", values="disparity")
    return d_matrix.combine_first(d_matrix.T)


def load_h5_and_pivot(
    h5_path: Union[str, Path],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load a shape-similarity H5 and return ``(pivot_matrix, raw_df)``."""
    from Helper import h5io

    d_raw, _ = h5io(h5_path, task="load", labels=None, item_pairs=None)
    return pivot_disparity_matrix(d_raw), d_raw


def _load_run_config(
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, str]]:
    """Return shape-similarity run definitions from *config* or hard-coded defaults.

    Each run maps to ``{"cell_type": str, "h5_stem": str}``.
    The ``h5_stem`` is the filename prefix for the disparity H5 file
    (e.g., "soft-matching_shape_similarity_FLUORESCENCE_activity_...").

    Note: The legacy ``task_data_csv`` field in config is ignored.
    """
    if config and config.get("shape_similarity_runs"):
        return config["shape_similarity_runs"]

    # Hard-coded defaults when no config provided
    return {
        "place": {
            "cell_type": "place",
            "h5_stem": (
                "soft-matching_shape_similarity_FLUORESCENCE_activity_"
                "spacetuned_to_positionPLACE_cellsvel_0.00-0"
            ),
        },
        "non_place": {
            "cell_type": "non-place",
            "h5_stem": (
                "soft-matching_shape_similarity_FLUORESCENCE_activity_"
                "spacetuned_to_positionNON-PLACE_cellsvel_0.00-0"
            ),
        },
    }


def _load_h5_raw(output_dir: Path, h5_stem: str) -> pd.DataFrame:
    """Load the raw (long-format) disparity DF from an H5 file.

    Parameters
    ----------
    output_dir : Path
        Directory containing the H5 files (typically ``<root>/output``).
    h5_stem : str
        Filename stem without the ``.h5`` extension.

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns: ``item_i``, ``item_j``, ``disparity``
        where ``disparity`` is a plain float.
    """
    h5_path = output_dir / f"{h5_stem}.h5"
    if not h5_path.exists():
        raise FileNotFoundError(
            f"Disparity H5 file not found: {h5_path}\n"
            f"Ensure shape_similarity() has been run with the correct parameters."
        )
    _matrix, d_raw = load_h5_and_pivot(h5_path)

    # Clean disparity values: H5 may store (float, {}) tuples from
    # soft-matching -- extract the scalar float.
    if "disparity" in d_raw.columns:
        d_raw["disparity"] = d_raw["disparity"].apply(
            lambda x: x[0] if isinstance(x, tuple) else x
        )
    return d_raw


def _merge_raw_disparities(
    raw_dfs: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Merge N raw disparity DataFrames into one long-format DF.

    Creates fully-descriptive column names for each run's disparity values
    plus backward-compatible short aliases for the first two runs.

    **Column naming convention:**
    
    - **Full names**: ``<label>_disparity`` where label comes from config
      (e.g., ``place_disparity``, ``non_place_disparity``, ``all_disparity``)
    - **Short aliases** (backward compatibility): ``pc_disparity`` and 
      ``npc_disparity`` for the first two runs (typically place and non-place)
    
    Both naming schemes provide the same data - use whichever is clearer
    in your context. The full names are more explicit; the short aliases
    are convenient for interactive work and match legacy code.

    Parameters
    ----------
    raw_dfs : dict
        ``{label: DataFrame}`` mapping run names to raw disparity dataframes.
        Each DataFrame must have columns: ``item_i``, ``item_j``, ``disparity``.

    Returns
    -------
    pd.DataFrame
        Merged long-format dataframe with columns:
        - ``animal_id_i/j``, ``date_i/j``, ``task_name_i/j``
        - ``<label>_disparity`` for each run (full descriptive names)
        - ``pc_disparity``, ``npc_disparity`` (short aliases, if applicable)
    """
    from temporary import split_item_name

    labels = list(raw_dfs.keys())
    base_label = labels[0]
    base_raw = raw_dfs[base_label].copy()

    # Expand item identifiers into animal / date / task columns
    for suffix in ("i", "j"):
        col = f"item_{suffix}"
        parts = base_raw[col].apply(split_item_name)
        base_raw[f"animal_id_{suffix}"] = parts.apply(lambda p: p[0])
        base_raw[f"date_{suffix}"] = parts.apply(lambda p: p[1])
        base_raw[f"task_name_{suffix}"] = parts.apply(lambda p: p[2])

    base_raw.rename(columns={"disparity": f"{base_label}_disparity"}, inplace=True)

    for label in labels[1:]:
        other = raw_dfs[label][["item_i", "item_j", "disparity"]].copy()
        other.rename(columns={"disparity": f"{label}_disparity"}, inplace=True)
        base_raw = base_raw.merge(other, on=["item_i", "item_j"], how="outer")

    base_raw.drop(columns=["item_i", "item_j"], errors="ignore", inplace=True)

    # Add backward-compatible short aliases for first two runs
    # pc_disparity -> place cells, npc_disparity -> non-place cells
    _COMPAT = {0: "pc_disparity", 1: "npc_disparity"}
    for idx, label in enumerate(labels):
        alias = _COMPAT.get(idx)
        run_col = f"{label}_disparity"
        if alias and run_col in base_raw.columns and alias not in base_raw.columns:
            base_raw[alias] = base_raw[run_col]

    return base_raw


# ═══════════════════════════════════════════════════════════════════════════
# Public API -- main disparity pipeline
# ═══════════════════════════════════════════════════════════════════════════


def build_enhanced_disparity_df(
    animal_root_dir: Union[str, Path, None] = None,
    output_filename: str = "better_d_df_with_cell_numbers_enhanced.csv",
    *,
    config: Optional[Dict[str, Any]] = None,
    animal_info_h5_path: Optional[Union[str, Path]] = None,
    regenerate: Optional[Literal["enhanced", "all"]] = None,
    task_groups: Optional[Dict[str, Dict[str, List[str]]]] = None,
    group_by: str = "condition",
    compare_by: str = "task",
) -> pd.DataFrame:
    """Build one enhanced long-format disparity DataFrame from multiple H5 runs.

    **Workflow**:

    1. Load disparity H5 files (one per cell type: place, non-place, all, etc.)
       from ``<root>/output/`` using the naming convention:
       ``soft-matching_shape_similarity_FLUORESCENCE_activity_spacetuned_to_position<CELLTYPE>_cellsvel_0.00-0.h5``

    2. Merge disparities into a single long-format DataFrame.

    3. Enhance with metadata (conditions, groups, colors, cell counts) from
       ``animal_info.h5`` via :class:`AnimalInfoStore`. Groups and colors
       are generated using :func:`Classes.Mother.improve_animals_df`.

    **Changed in v2**: No longer uses ``run_inputs`` parameter or task-data CSVs.
    All data is loaded directly from H5 files.

    Parameters
    ----------
    animal_root_dir : Path, optional
        Experiment root directory (e.g., ``F:\\Experiments\\Nathalie\\OpenFieldDynamics_3``).
        Inferred from ``config["animal_root_dir"]`` when omitted.
    output_filename : str, default="better_d_df_with_cell_numbers_enhanced.csv"
        Name of the final enhanced CSV in ``<root>/output``.
    config : dict, optional
        Resolved experiment config from :func:`Helper.load_experiment_config`.
        Must contain ``shape_similarity_runs`` defining H5 file stems.
    animal_info_h5_path : Path, optional
        Path to the animal-info H5 file. Defaults to ``<root>/output/animal_info.h5``.
    regenerate : {"enhanced", "all"} or None
        - ``"enhanced"``: Re-run metadata enrichment only
        - ``"all"``: Reload all H5 files and re-merge from scratch
        - ``None``: Use cached enhanced CSV if available
    task_groups : dict, optional
        Task grouping structure for ``improve_animals_df``.
        Outer keys match ``group_by`` values, inner dicts map group names
        to lists of ``compare_by`` values. If None, auto-grouping is applied.
    group_by : str, default="condition"
        Column for primary grouping (passed to ``improve_animals_df``).
    compare_by : str, default="task"
        Column for subgroup comparison (passed to ``improve_animals_df``).

    Returns
    -------
    pd.DataFrame
        Enhanced long-format disparity DataFrame with columns:
        - ``animal_id_i/j``, ``date_i/j``, ``task_name_i/j``
        - ``<label>_disparity`` for each run (e.g., ``place_disparity``, ``non_place_disparity``)
        - Backward-compatible aliases: ``pc_disparity``, ``npc_disparity``
        - Metadata: ``condition_i/j``, ``group_i/j``, ``color_i/j``, 
          ``place_cell_count_i/j``, ``non_place_cell_count_i/j``

    Raises
    ------
    FileNotFoundError
        If required H5 files (disparities or animal_info) are not found.

    Examples
    --------
    >>> from Helper import load_experiment_config
    >>> config = load_experiment_config()
    >>> enhanced_df = build_enhanced_disparity_df(config=config)
    >>> enhanced_df.head()
    >>> enhanced_df.head()
    """
    # --- resolve paths ----------------------------------------------------
    if config and animal_root_dir is None:
        animal_root_dir = config["animal_root_dir"]
    if animal_root_dir is None:
        raise ValueError(
            "animal_root_dir must be provided either directly or via config."
        )
    animal_root_dir = Path(animal_root_dir)
    output_dir = animal_root_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    if animal_info_h5_path is None:
        animal_info_h5_path = output_dir / "animal_info.h5"
    animal_info_h5_path = Path(animal_info_h5_path)

    if not animal_info_h5_path.exists():
        raise FileNotFoundError(
            f"animal_info.h5 not found: {animal_info_h5_path}\n"
            f"Run Mother.build_animal_info() first to generate this file."
        )

    regen_all = regenerate == "all"
    regen_enhanced = regenerate in ("enhanced", "all")

    # --- fast path: cached enhanced CSV -----------------------------------
    output_path = output_dir / output_filename
    if output_path.exists() and not regen_enhanced:
        print(f"Loading cached enhanced DF from {output_path}")
        return load_df(output_path)

    # --- load run definitions ---------------------------------------------
    runs = _load_run_config(config)
    print(f"Loading {len(runs)} disparity runs: {list(runs.keys())}")

    # --- step 1: load + merge raw H5 disparities -------------------------
    if regen_all:
        print("Reloading all H5 disparity files...")
        raw_dfs: Dict[str, pd.DataFrame] = {}
        for label, run_def in runs.items():
            print(f"  Loading '{label}' from {run_def['h5_stem']}.h5")
            raw_dfs[label] = _load_h5_raw(output_dir, run_def["h5_stem"])
        long_df = _merge_raw_disparities(raw_dfs)
    else:
        # No intermediate cache - always load fresh from H5
        raw_dfs: Dict[str, pd.DataFrame] = {}
        for label, run_def in runs.items():
            raw_dfs[label] = _load_h5_raw(output_dir, run_def["h5_stem"])
        long_df = _merge_raw_disparities(raw_dfs)

    # --- step 2: enhance with metadata from animal-info H5 ---------------
    print(f"Enhancing with metadata from {animal_info_h5_path.name}...")
    store = AnimalInfoStore(animal_info_h5_path)
    enhanced_df = store.enhance_disparity_df(
        long_df,
        output_dir,
        output_filename,
        regenerate=regen_enhanced,
        task_groups=task_groups,
        group_by=group_by,
        compare_by=compare_by,
    )

    print(f"✓ Enhanced DF ready: {enhanced_df.shape[0]} rows, {enhanced_df.shape[1]} columns")
    return enhanced_df