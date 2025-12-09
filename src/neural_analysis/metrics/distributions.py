"""Distribution comparison utilities for neural data analysis.

This module provides functions for comparing probability distributions using
various statistical metrics. It includes both pairwise comparisons and
group-based comparisons with optional outlier filtering.

All distance computations delegate to the distance module to avoid code duplication.

Shape Similarity:
This module also includes shape distance functions for comparing neural population
activity matrices. These treat each population as a distribution in feature space
and compute distances using Procrustes alignment, one-to-one matching, or soft
optimal transport matching.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable, Mapping, Sequence, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    TypedDict,
    TypeVar,
    cast,
    Callable,
    Dict,
    List,
    Sequence,
    Tuple,
    Union,
)


import numpy as np
import numpy.typing as npt
from scipy.linalg import orthogonal_procrustes
from scipy.optimize import linear_sum_assignment
from scipy.spatial import procrustes
from scipy.spatial.distance import cdist
from scipy.spatial import procrustes

if TYPE_CHECKING:
    import pandas as pd
else:
    import pandas as pd  # noqa: PGH003

# I/O functions imported locally where needed to avoid circular dependencies

from .pairwise_metrics import compute_pairwise_matrix, pairwise_distance

try:
    import ot  # Python Optimal Transport

    OT_AVAILABLE = True
except ImportError:
    OT_AVAILABLE = False

from neural_analysis.utils.subsampling import run_with_subsampling

try:
    from neural_analysis.utils.logging import get_logger, log_calls
except ImportError:
    if TYPE_CHECKING:
        from collections.abc import Callable

    def log_calls(
        *, level: int = logging.DEBUG, timeit: bool = True
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            return func

        return decorator

    def get_logger(name: str | None = None) -> logging.Logger:
        return logging.getLogger(name or "neural_analysis")


# Module logger
logger = get_logger(__name__)

__all__ = [
    "wasserstein_distance_multi",
    "kolmogorov_smirnov_distance",
    "jensen_shannon_divergence",
    "distribution_distance",
    "shape_distance",
    "pairwise_distribution_comparison_batch",
    "batch_comparison",
]

SHAPE_METRICS = {"procrustes", "one-to-one", "soft-matching"}
DEFAULT_COMPARISON_SAVE_PATH = Path("./output/distribution_comparisons.h5")

T = TypeVar("T")


def _progress_iterable(
    iterable: Iterable[T],
    *,
    enable: bool,
    desc: str | None = None,
) -> Iterable[T]:
    """Optionally wrap iterable with tqdm progress bar."""
    if not enable:
        return iterable
    try:
        from tqdm.auto import tqdm as tqdm_impl
    except Exception:  # pragma: no cover - optional dependency
        return iterable
    return cast(Iterable[T], tqdm_impl(iterable, desc=desc))


def _normalize_metrics_input(
    metrics: Sequence[str] | Mapping[str, Mapping[str, Any]],
    common_kwargs: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Normalize metrics input to dict[metric_name, kwargs]."""
    common_kwargs = dict(common_kwargs or {})
    if isinstance(metrics, Mapping):
        return {
            metric: {**common_kwargs, **(metric_kwargs or {})}
            for metric, metric_kwargs in metrics.items()
        }
    if not metrics:
        raise ValueError("metrics must contain at least one metric name")
    return {metric: dict(common_kwargs) for metric in metrics}


def _result_key(metric: str, dataset_i: str, dataset_j: str) -> str:
    """Generate stable result key for HDF5 storage."""
    return f"{metric}__{dataset_i}__{dataset_j}"


def _cache_key(save_path: Path, comparison_name: str, result_key: str) -> str:
    """Generate cache key for Redis storage."""
    return f"pairwise::{save_path}::{comparison_name}::{result_key}"


def _serialize_pairs(
    pairs: dict[tuple[int, int], float] | None,
) -> dict[str, npt.NDArray[Any]]:
    """Serialize pairs dict to numpy arrays for HDF5 storage."""
    if not pairs:
        return {}
    pair_indices = np.array(list(pairs.keys()), dtype=np.int64)
    pair_values = np.array(list(pairs.values()), dtype=np.float64)
    return {
        "pair_indices": pair_indices,
        "pair_values": pair_values,
    }


def _deserialize_pairs(
    arrays: Mapping[str, Any] | None,
) -> dict[tuple[int, int], float] | None:
    """Deserialize stored pair arrays back into dictionary."""
    if not arrays:
        return None
    if "pair_indices" not in arrays or "pair_values" not in arrays:
        return None
    indices = arrays["pair_indices"]
    values = arrays["pair_values"]
    pairs: dict[tuple[int, int], float] = {}
    for idx, value in zip(indices, values, strict=False):
        i, j = int(idx[0]), int(idx[1])
        pairs[(i, j)] = float(value)
    return pairs


def _function_accepts_argument(func: Callable[..., Any], name: str) -> bool:
    """Check if callable accepts a keyword argument."""
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):  # pragma: no cover - builtins
        return False
    for param in signature.parameters.values():
        if param.kind == param.VAR_KEYWORD:
            return True
        if param.name == name and param.kind in (
            param.POSITIONAL_OR_KEYWORD,
            param.KEYWORD_ONLY,
        ):
            return True
    return False


def _compute_metric_result(
    points_i: npt.ArrayLike,
    points_j: npt.ArrayLike,
    metric: str,
    *,
    metric_kwargs: Mapping[str, Any],
) -> tuple[float, dict[tuple[int, int], float] | None, str]:
    """Compute metric result and normalize output."""
    result = compute_pairwise_matrix(
        points_i,
        points_j,
        metric=metric,
        parallel=True,
        **metric_kwargs,
    )

    if metric in SHAPE_METRICS:
        if not isinstance(result, tuple):
            raise TypeError(
                f"Expected tuple return value for shape metric '{metric}', got {type(result)!r}"
            )
        value = float(result[0])
        pairs = {(int(i), int(j)): float(val) for (i, j), val in result[1].items()}
        return value, pairs, "shape"

    if isinstance(result, np.ndarray):
        value = float(np.mean(result))
        return value, None, "matrix"

    if isinstance(result, tuple):
        value = float(result[0])
        pairs = result[1] if isinstance(result[1], dict) else None
        return value, pairs, "tuple"

    # Fallback: scalar result
    return float(result), None, "scalar"


def _row_from_saved_entry(
    result_key: str,
    entry: Mapping[str, Any],
    comparison_name: str,
) -> dict[str, Any] | None:
    """Convert stored HDF5 entry to dataframe row."""
    attrs = entry.get("attributes") or entry.get("attrs") or {}
    if not attrs or "value" not in attrs:
        return None
    row: dict[str, Any] = {
        "comparison_name": attrs.get("comparison_name", comparison_name),
        "dataset_i": attrs.get("dataset_i"),
        "dataset_j": attrs.get("dataset_j"),
        "metric": attrs.get("metric"),
        "value": attrs.get("value"),
        "value_type": attrs.get("value_type"),
        "n_samples_i": attrs.get("n_samples_i"),
        "n_samples_j": attrs.get("n_samples_j"),
        "n_features": attrs.get("n_features"),
        "timestamp": attrs.get("timestamp"),
        "mode": attrs.get("mode", "between"),
        "result_key": result_key,
    }
    arrays = entry.get("arrays")
    pairs = _deserialize_pairs(arrays)
    if pairs:
        row["pairs"] = pairs
        row["pair_count"] = len(pairs)
    return row


def _split_result_value(
    result: Any,
) -> tuple[float, Any]:
    """Normalize result from arbitrary comparison function.

    Handles both 2-element tuples (value, metadata) and 3-element tuples
    (value, pairs, metadata) from shape_distance functions.
    """
    if isinstance(result, tuple):
        if len(result) == 3:
            # shape_distance returns (distance, pairs, metadata)
            # Return distance as value, and (pairs, metadata) as metadata
            return float(result[0]), {"pairs": result[1], "metadata": result[2]}
        elif len(result) == 2:
            # Standard (value, metadata) tuple
            return float(result[0]), result[1]
        else:
            # Single element tuple or other
            return float(result[0]), result[1:] if len(result) > 1 else None
    return float(result), None


def _prepare_datasets(
    data: Mapping[str, npt.ArrayLike],
) -> dict[str, npt.NDArray[np.float64]]:
    """Convert mapping of dataset names to float64 ndarrays."""
    if not data:
        raise ValueError("data must contain at least one dataset")
    prepared: dict[str, npt.NDArray[np.float64]] = {}
    for name, values in data.items():
        arr = np.asarray(values)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        prepared[str(name)] = np.asarray(arr, dtype=np.float64)
    return prepared


# ============================================================================
# Type Definitions for kwargs
# ============================================================================


class SoftMatchingKwargs(TypedDict, total=False):
    """Kwargs for shape_distance_soft_matching."""

    approx: bool
    reg: float


class OneToOneKwargs(TypedDict, total=False):
    """Kwargs for shape_distance_one_to_one."""

    pass  # This method only uses metric, which is passed explicitly


class ProcrustesKwargs(TypedDict, total=False):
    """Kwargs for shape_distance_procrustes."""

    pass  # This method only uses default parameters


# Union type for all shape method kwargs
ShapeMethodKwargs = SoftMatchingKwargs | OneToOneKwargs | ProcrustesKwargs


# ============================================================================
# Distribution-Level Distance Metrics
# ============================================================================


def wasserstein_distance_multi(
    points1: npt.ArrayLike,
    points2: npt.ArrayLike,
) -> float:
    """Compute sum of Wasserstein distances over all features.

    Parameters
    ----------
    points1, points2 : array-like
        Point distributions (n_samples, n_features).

    Returns
    -------
    float
        Sum of Wasserstein distances across all dimensions.
        Returns NaN if either distribution is empty.

    Examples
    --------
    >>> p1 = np.random.randn(100, 3)
    >>> p2 = np.random.randn(100, 3) + 1.0
    >>> dist = wasserstein_distance_multi(p1, p2)
    """
    from scipy.stats import wasserstein_distance

    p1 = np.asarray(points1)
    p2 = np.asarray(points2)

    if p1.ndim == 1:
        p1 = p1.reshape(-1, 1)
    if p2.ndim == 1:
        p2 = p2.reshape(-1, 1)

    # Check for empty distributions
    if p1.shape[0] == 0 or p2.shape[0] == 0:
        logger.warning("Empty distribution detected, returning NaN")
        return np.nan

    # Check for dimension mismatch
    if p1.shape[1] != p2.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: points1 has {p1.shape[1]} features, "
            f"points2 has {p2.shape[1]} features"
        )

    distances = []
    for i in range(p1.shape[1]):
        dist = wasserstein_distance(p1[:, i], p2[:, i])
        # Check for inf or nan values
        if not np.isfinite(dist):
            logger.warning(
                f"Wasserstein distance returned non-finite value ({dist}) for feature {i}. "
                f"Replacing with np.nan. This may indicate numerical issues or disjoint distributions."
            )
            dist = np.nan
        distances.append(dist)

    total_distance = float(np.sum(distances))

    # Final check to ensure result is finite
    if not np.isfinite(total_distance):
        logger.warning(
            f"Total Wasserstein distance is non-finite ({total_distance}). "
            "Replacing with np.nan."
        )
        total_distance = np.nan

    return total_distance


@log_calls(level=logging.DEBUG)
def kolmogorov_smirnov_distance(
    points1: npt.ArrayLike,
    points2: npt.ArrayLike,
) -> float:
    """Compute maximum Kolmogorov-Smirnov statistic over all features.

    Parameters
    ----------
    points1, points2 : array-like
        Point distributions (n_samples, n_features).

    Returns
    -------
    float
        Maximum K-S statistic across all dimensions.
        Returns NaN if either distribution is empty.

    Examples
    --------
    >>> p1 = np.random.randn(100, 3)
    >>> p2 = np.random.randn(100, 3) + 1.0
    >>> dist = kolmogorov_smirnov_distance(p1, p2)
    """
    from scipy.stats import ks_2samp

    p1 = np.asarray(points1)
    p2 = np.asarray(points2)

    if p1.ndim == 1:
        p1 = p1.reshape(-1, 1)
    if p2.ndim == 1:
        p2 = p2.reshape(-1, 1)

    # Check for empty distributions
    if p1.shape[0] == 0 or p2.shape[0] == 0:
        logger.warning("Empty distribution detected, returning NaN")
        return np.nan

    # Check for dimension mismatch
    if p1.shape[1] != p2.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: points1 has {p1.shape[1]} features, "
            f"points2 has {p2.shape[1]} features"
        )

    ks_stats = [ks_2samp(p1[:, i], p2[:, i]).statistic for i in range(p1.shape[1])]
    max_ks = float(np.max(ks_stats))

    # KS statistic ranges from 0 to 1, where:
    # - 0: identical distributions
    # - 1: perfect separation (no overlap) in at least one dimension
    # A value of 1.0 is valid and indicates complete separation in at least one feature
    if max_ks == 1.0:
        logger.debug(
            f"KS distance = 1.0 indicates perfect separation in at least one feature. "
            f"Individual feature KS stats: {[f'{s:.4f}' for s in ks_stats]}"
        )

    return max_ks


@log_calls(level=logging.DEBUG)
def jensen_shannon_divergence(
    points1: npt.ArrayLike,
    points2: npt.ArrayLike,
    bins: int = 50,
) -> float:
    """Compute Jensen-Shannon divergence between point distributions.

    Parameters
    ----------
    points1, points2 : array-like
        Point distributions (n_samples, n_features).
    bins : int, default=50
        Number of bins for histogram computation.

    Returns
    -------
    float
        Jensen-Shannon divergence in [0, 1].
        Returns NaN if either distribution is empty.

    Examples
    --------
    >>> p1 = np.random.randn(100, 3)
    >>> p2 = np.random.randn(100, 3) + 1.0
    >>> div = jensen_shannon_divergence(p1, p2)
    """
    from scipy.stats import entropy

    p1 = np.asarray(points1)
    p2 = np.asarray(points2)

    if p1.ndim == 1:
        p1 = p1.reshape(-1, 1)
    if p2.ndim == 1:
        p2 = p2.reshape(-1, 1)

    # Check for empty distributions
    if p1.shape[0] == 0 or p2.shape[0] == 0:
        logger.warning("Empty distribution detected, returning NaN")
        return np.nan

    # Check for dimension mismatch
    if p1.shape[1] != p2.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: points1 has {p1.shape[1]} features, "
            f"points2 has {p2.shape[1]} features"
        )

    # Determine common bin edges
    all_data = np.vstack([p1, p2])
    ranges = [
        (all_data[:, i].min(), all_data[:, i].max()) for i in range(all_data.shape[1])
    ]

    # Adaptive binning for high-dimensional data to prevent memory explosion
    # For D dimensions with b bins, we need b^D total bins
    # Limit to ~10^6 bins maximum (e.g., 10^2 bins for 3D, 10 bins for 6D)
    n_dims = p1.shape[1]
    if n_dims > 3:
        # Reduce bins for high-D data: bins = max(3, original_bins^(3/D))
        adaptive_bins = max(3, int(bins ** (3.0 / n_dims)))
        if adaptive_bins < bins:
            logger.info(
                f"Reducing bins from {bins} to {adaptive_bins} for {n_dims}D data "
                f"(total bins: {adaptive_bins}^{n_dims} = {adaptive_bins**n_dims:,})"
            )
            bins = adaptive_bins

    # Compute multi-dimensional histograms
    hist1, _ = np.histogramdd(p1, bins=bins, range=ranges)
    hist2, _ = np.histogramdd(p2, bins=bins, range=ranges)

    # Flatten and normalize
    hist1 = hist1.ravel() + 1e-10  # avoid zeros
    hist2 = hist2.ravel() + 1e-10
    hist1 /= hist1.sum()
    hist2 /= hist2.sum()

    # Compute JS divergence
    m = 0.5 * (hist1 + hist2)
    js_div = 0.5 * (entropy(hist1, m) + entropy(hist2, m))
    return float(js_div)


def _compute_summary_statistics(
    dists: npt.NDArray[np.floating],
    summary: Literal["mean", "std", "median", "all"] = "mean",
) -> float | dict[str, float]:
    """Compute summary statistics from distance array.

    Parameters
    ----------
    dists : ndarray
        Array of distance values.
    summary : {"mean", "std", "median", "all"}, default="mean"
        Summary statistic to return.

    Returns
    -------
    float or dict
        Summary statistic(s).
    """
    match summary:
        case "mean":
            return float(np.mean(dists))
        case "std":
            return float(np.std(dists))
        case "median":
            return float(np.median(dists))
        case "all":
            return {
                "mean": float(np.mean(dists)),
                "std": float(np.std(dists)),
                "median": float(np.median(dists)),
                "min": float(np.min(dists)),
                "max": float(np.max(dists)),
            }
        case _:
            raise ValueError(
                f"Unknown summary '{summary}'. Choose from: mean, std, median, all."
            )


@log_calls(level=logging.DEBUG)
def distribution_distance(
    points1: npt.ArrayLike,
    points2: npt.ArrayLike | None = None,
    mode: Literal["within", "between"] = "between",
    metric: Literal[
        "euclidean",
        "manhattan",
        "mahalanobis",
        "cosine",
        "wasserstein",
        "kolmogorov-smirnov",
        "jensen-shannon",
        "procrustes",
        "one-to-one",
        "soft-matching",
    ] = "euclidean",
    parallel: bool = True,
    summary: Literal["mean", "std", "median", "all"] = "mean",
    **metric_kwargs: Any,  # Accept Any for flexible kwargs
) -> float | dict[str, float] | tuple[float, dict[tuple[int, int], float]]:
    """Compute pairwise distances within or between distributions.

    This unified function replaces the separate `within_distribution_distance`
    and `between_distribution_distance` functions, reducing code duplication.
    Also supports shape-based comparison methods.

    Parameters
    ----------
    points1 : array-like
        First point distribution, shape (n_samples, n_features).
    points2 : array-like, optional
        Second point distribution, shape (n_samples, n_features).
        Required if mode="between", ignored if mode="within".
    mode : {"within", "between"}, default="between"
        Whether to compute distances within a single distribution or between
        two distributions. Note: Shape metrics only work with mode="between".
    metric : str, default="euclidean"
        Distance metric to use. Supported metrics:

        **Point-wise metrics** (work with both "within" and "between"):
        - "euclidean": Euclidean distance
        - "manhattan": Manhattan (L1) distance
        - "cosine": Cosine similarity
        - "mahalanobis": Mahalanobis distance

        **Distribution-level metrics** (only "between" mode):
        - "wasserstein": Wasserstein distance (Earth Mover's Distance)
        - "kolmogorov-smirnov": Kolmogorov-Smirnov statistic
        - "jensen-shannon": Jensen-Shannon divergence

        **Shape metrics** (only "between" mode):
        - "procrustes": Procrustes distance after optimal alignment
        - "one-to-one": Bipartite matching distance
        - "soft-matching": Soft assignment via optimal transport
    parallel : bool, default=True
        Use parallel numba implementation if available.
    summary : {"mean", "std", "median", "all"}, default="mean"
        Summary statistic to return. If "all", returns dict with all statistics.
        Ignored for shape metrics, which always return a single distance.
    **metric_kwargs
        Additional keyword arguments for the distance function.
        For shape metrics: method-specific parameters like reg, approx, etc.

    Returns
    -------
    float or dict or tuple
        For point-wise/distribution metrics:
            - float: Single summary statistic (if summary != "all")
            - dict: All statistics (if summary == "all")
        For shape metrics:
            - tuple: (distance: float, pairs: dict[tuple[int, int], float])

    Raises
    ------
    ValueError
        If shape metrics are used with mode="within".

    Examples
    --------
    >>> # Within-distribution distances
    >>> points = np.random.randn(100, 10)
    >>> mean_dist = distribution_distance(points, mode="within")
    >>> isinstance(mean_dist, float)
    True

    >>> # Between-distribution distances
    >>> p1 = np.random.randn(100, 10)
    >>> p2 = np.random.randn(100, 10) + 1.0
    >>> mean_dist = distribution_distance(p1, p2, mode="between")
    >>> isinstance(mean_dist, float)
    True

    >>> # Get all statistics
    >>> stats = distribution_distance(p1, p2, mode="between", summary="all")
    >>> "mean" in stats and "std" in stats
    True

    >>> # Shape-based comparison
    >>> dist, pairs = distribution_distance(p1, p2, mode="between",
    ...                                     metric="procrustes")
    >>> isinstance(dist, float) and isinstance(pairs, dict)
    True
    """
    # Define shape metrics and distribution metrics
    shape_metrics = {"procrustes", "one-to-one", "soft-matching"}
    distribution_metrics = {"wasserstein", "kolmogorov-smirnov", "jensen-shannon"}

    # Validate metric/mode combination
    if mode == "within" and metric in shape_metrics:
        raise ValueError(
            f"Shape metric '{metric}' cannot be used with mode='within'. "
            f"Shape metrics only work in 'between' mode."
        )

    if mode == "within" and metric in distribution_metrics:
        raise ValueError(
            f"Distribution metric '{metric}' cannot be used with mode='within'. "
            f"Distribution metrics compare entire distributions, not point pairs. "
            f"Use mode='between' or a point-wise metric like 'euclidean' or 'cosine'."
        )

    points1_arr = np.asarray(points1)

    # Handle mode-specific logic
    match mode:
        case "within":
            if points1_arr.shape[0] < 2:
                logger.warning("Less than 2 samples, returning zero distances")
                return (
                    {"mean": 0.0, "std": 0.0, "median": 0.0}
                    if summary == "all"
                    else 0.0
                )

            logger.info(
                f"Computing within-distribution distances: "
                f"n_samples={points1_arr.shape[0]}, "
                f"n_features={points1_arr.shape[1]}, "
                f"metric='{metric}', summary='{summary}'"
            )

            # Compute pairwise distances within single distribution
            dists_matrix = pairwise_distance(
                points1_arr,
                points1_arr,
                metric=metric,
                parallel=parallel,
                **metric_kwargs,
            )

            # Extract upper triangle (excluding diagonal)
            mask = np.triu(np.ones_like(dists_matrix, dtype=bool), k=1)
            dists = dists_matrix[mask]

        case "between":
            if points2 is None:
                raise ValueError("points2 is required when mode='between'")

            points2_arr = np.asarray(points2)

            logger.info(
                f"Computing between-distribution distances: "
                f"p1.shape={points1_arr.shape}, "
                f"p2.shape={points2_arr.shape}, "
                f"metric='{metric}', summary='{summary}'"
            )

            # For shape metrics, use shape_distance
            if metric in shape_metrics:
                method_typed: Literal["procrustes", "one-to-one", "soft-matching"] = (
                    metric  # type: ignore[assignment]
                )
                result_shape = shape_distance(
                    points1_arr.astype(np.float64),
                    points2_arr.astype(np.float64),
                    method=method_typed,
                    return_pairs=True,
                    **metric_kwargs,
                )
                assert isinstance(result_shape, tuple)
                dist, pairs = result_shape
                logger.info(f"Shape distance computed: {dist:.6f}")
                return (dist, pairs)

            # For distribution-level metrics, compute directly
            if metric in {"wasserstein", "kolmogorov-smirnov", "jensen-shannon"}:
                # Use local functions (now in this module)
                if metric == "wasserstein":
                    dist = wasserstein_distance_multi(
                        points1_arr, points2_arr, **metric_kwargs
                    )
                elif metric == "kolmogorov-smirnov":
                    dist = kolmogorov_smirnov_distance(
                        points1_arr, points2_arr, **metric_kwargs
                    )
                else:  # jensen-shannon
                    dist = jensen_shannon_divergence(
                        points1_arr, points2_arr, **metric_kwargs
                    )

                logger.info(f"Distribution-level distance: {dist:.6f}")
                return (
                    dist
                    if summary != "all"
                    else {"mean": dist, "std": 0.0, "median": dist}
                )

            # For point-wise metrics, compute pairwise and summarize
            dists_matrix = pairwise_distance(
                points1_arr,
                points2_arr,
                metric=metric,
                parallel=parallel,
                **metric_kwargs,
            )

            # Flatten all pairwise distances
            dists = dists_matrix.ravel()

        case _:
            raise ValueError(f"Unknown mode '{mode}'. Choose 'within' or 'between'.")

    # Compute summary statistics using helper function
    result = _compute_summary_statistics(dists, summary)

    logger.info(f"{mode.capitalize()}-distribution distance computed: {result}")
    return result


# ============================================================================
# Public API Functions
# ============================================================================
# Note: compare_distributions and compare_distribution_groups have been removed.
# Use compare_datasets from neural_analysis.metrics.pairwise_metrics instead.
#
# Migration guide:
# - compare_distributions(p1, p2, metric="wasserstein")
#   → compare_datasets(p1, p2, mode="between", metric="wasserstein")
# - compare_distribution_groups(groups, compare_type="between", metric="wasserstein")
#   → compare_datasets(groups, mode="all-pairs", metric="wasserstein")  # for scalar metrics
#   → Use compare_datasets in a loop for point-to-point metrics


def pairwise_distribution_comparison_batch(
    data: Mapping[str, npt.ArrayLike],
    metrics: Sequence[str] | Mapping[str, Mapping[str, Any]],
    *,
    comparison_name: str = "default",
    save_path: str | Path | None = None,
    regenerate: bool = False,
    store_pairs: bool = True,
    progress: bool = False,
    use_cache: bool = True,
    use_sql_index: bool = True,
    **common_metric_kwargs: Any,
) -> pd.DataFrame:
    """Compute all-pairs distribution comparisons with caching and persistence.

    Parameters
    ----------
    data : Mapping[str, array-like]
        Mapping of dataset name -> samples (n_samples, n_features)
    metrics : sequence or mapping
        - Sequence of metric names (e.g., ["wasserstein", "procrustes"])
        - Mapping of metric name -> kwargs (e.g., {"wasserstein": {"summary": "median"}})
    comparison_name : str, default="default"
        Logical group name used for HDF5 storage/querying
    save_path : str or Path, optional
        HDF5 file path for caching results. Defaults to
        "./output/distribution_comparisons.h5"
    regenerate : bool, default=False
        If True, recompute even if cached results exist
    store_pairs : bool, default=True
        Whether to store pair correspondences for shape metrics
    progress : bool, default=False
        Display tqdm progress bar when available
    use_cache : bool, default=True
        Use Redis cache when available
    use_sql_index : bool, default=True
        Index metadata in DuckDB for fast queries
    **common_metric_kwargs
        Additional kwargs applied to every metric (overridden by per-metric kwargs)

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns:
        ['comparison_name', 'dataset_i', 'dataset_j', 'metric', 'value',
         'value_type', 'n_samples_i', 'n_samples_j', 'n_features',
         'timestamp', 'mode', 'pair_count', 'pairs']
    """
    from neural_analysis.utils.io import (
        load_results_from_hdf5_dataset,
        save_result_to_hdf5_dataset,
    )

    datasets = _prepare_datasets(data)
    metrics_dict = _normalize_metrics_input(metrics, common_metric_kwargs)

    if save_path is None:
        save_path = DEFAULT_COMPARISON_SAVE_PATH
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    storage_manager = None
    if use_cache or use_sql_index:
        try:
            from neural_analysis.utils.storage.manager import StorageManager

            storage_manager = StorageManager()
        except Exception:
            storage_manager = None

    existing_rows: dict[str, dict[str, Any]] = {}
    if save_path.exists() and not regenerate:
        loaded = load_results_from_hdf5_dataset(
            save_path=save_path,
            dataset_name=comparison_name,
            use_sql_query=use_sql_index,
        )
        comparison_results = loaded.get(comparison_name, {})
        for result_key, entry in comparison_results.items():
            saved_row = _row_from_saved_entry(result_key, entry, comparison_name)
            if not saved_row:
                continue
            metric_name = saved_row.get("metric")
            if metric_name not in metrics_dict:
                continue
            existing_rows[result_key] = saved_row
            if use_cache and storage_manager:
                cache_key = _cache_key(save_path, comparison_name, result_key)
                storage_manager.cache_set(cache_key, saved_row)

    tasks = [
        (metric_name, dataset_i, dataset_j)
        for metric_name in metrics_dict
        for dataset_i in datasets
        for dataset_j in datasets
    ]

    if not tasks:
        return pd.DataFrame()

    task_iter = _progress_iterable(
        tasks,
        enable=progress,
        desc=f"Pairwise comparisons for '{comparison_name}'",
    )

    rows: list[dict[str, Any]] = []
    for metric_name, dataset_i, dataset_j in task_iter:
        metric_kwargs = metrics_dict[metric_name]
        result_key = _result_key(metric_name, dataset_i, dataset_j)
        cache_key = _cache_key(save_path, comparison_name, result_key)

        if not regenerate:
            cached_row = None
            if use_cache and storage_manager:
                cached_row = storage_manager.cache_get(cache_key)
            if cached_row is not None:
                rows.append(cached_row)
                continue
            if result_key in existing_rows:
                existing_row = existing_rows[result_key]
                rows.append(existing_row)
                if use_cache and storage_manager:
                    storage_manager.cache_set(cache_key, existing_row)
                continue

        value, pairs, value_type = _compute_metric_result(
            datasets[dataset_i],
            datasets[dataset_j],
            metric_name,
            metric_kwargs=metric_kwargs,
        )

        timestamp = datetime.now(UTC).isoformat()
        row_data: dict[str, Any] = {
            "comparison_name": comparison_name,
            "dataset_i": dataset_i,
            "dataset_j": dataset_j,
            "metric": metric_name,
            "value": value,
            "value_type": value_type,
            "n_samples_i": int(datasets[dataset_i].shape[0]),
            "n_samples_j": int(datasets[dataset_j].shape[0]),
            "n_features": int(datasets[dataset_i].shape[1]),
            "timestamp": timestamp,
            "mode": "between",
        }
        if pairs:
            row_data["pair_count"] = len(pairs)
            if store_pairs:
                row_data["pairs"] = pairs

        rows.append(row_data)

        scalar_payload = {
            "comparison_name": comparison_name,
            "dataset_i": dataset_i,
            "dataset_j": dataset_j,
            "metric": metric_name,
            "value": value,
            "value_type": value_type,
            "n_samples_i": row_data["n_samples_i"],
            "n_samples_j": row_data["n_samples_j"],
            "n_features": row_data["n_features"],
            "timestamp": timestamp,
            "mode": "between",
            "result_key": result_key,
        }
        array_payload = _serialize_pairs(pairs) if store_pairs else {}

        save_result_to_hdf5_dataset(
            save_path=save_path,
            dataset_name=comparison_name,
            result_key=result_key,
            scalar_data=scalar_payload,
            array_data=array_payload,
            use_cache=use_cache,
            use_sql_index=use_sql_index,
            storage_manager=storage_manager,
        )

        if use_cache and storage_manager:
            storage_manager.cache_set(cache_key, row_data)
        existing_rows[result_key] = row_data

    df = pd.DataFrame(rows)
    if "pairs" not in df.columns:
        df["pairs"] = None
    df = df.sort_values(["metric", "dataset_i", "dataset_j"]).reset_index(drop=True)
    return df


def batch_comparison(
    datasets: Mapping[str, npt.ArrayLike],
    comparison_fn: Callable[..., Any],
    *,
    include_self: bool = True,
    symmetric: bool = False,
    progress: bool = False,
    **comparison_kwargs: Any,
) -> pd.DataFrame:
    """Generic batch-comparison utility for arbitrary comparison functions.

    Parameters
    ----------
    datasets : Mapping[str, array-like]
        Mapping of dataset names to arrays
    comparison_fn : callable
        Function accepting (dataset_i, dataset_j, **kwargs) returning scalar or tuple
    include_self : bool, default=True
        Include diagonal comparisons (dataset against itself)
    symmetric : bool, default=False
        If True, only compute upper triangle (i <= j)
    progress : bool, default=False
        Display tqdm progress bar when available
    **comparison_kwargs
        Additional kwargs forwarded to comparison_fn

    Returns
    -------
    pandas.DataFrame
        Columns: ['dataset_1', 'dataset_2', 'distance', 'metadata']
    """
    dataset_items = list(datasets.items())
    if not dataset_items:
        return pd.DataFrame(columns=["dataset_1", "dataset_2", "distance"])

    accepts_dataset_i = _function_accepts_argument(comparison_fn, "dataset_i")
    accepts_dataset_j = _function_accepts_argument(comparison_fn, "dataset_j")

    tasks: list[tuple[str, str, npt.ArrayLike, npt.ArrayLike]] = []
    for idx_i, (name_i, data_i) in enumerate(dataset_items):
        for idx_j, (name_j, data_j) in enumerate(dataset_items):
            if not include_self and name_i == name_j:
                continue
            if symmetric and idx_j < idx_i:
                continue
            tasks.append((name_i, name_j, data_i, data_j))

    task_iter = _progress_iterable(tasks, enable=progress, desc="Batch comparisons")
    rows: list[dict[str, Any]] = []
    for name_i, name_j, data_i, data_j in task_iter:
        call_kwargs = dict(comparison_kwargs)
        if accepts_dataset_i:
            call_kwargs.setdefault("dataset_i", name_i)
        if accepts_dataset_j:
            call_kwargs.setdefault("dataset_j", name_j)

        result = comparison_fn(data_i, data_j, **call_kwargs)
        value, metadata = _split_result_value(result)
        row: dict[str, Any] = {
            "dataset_1": name_i,
            "dataset_2": name_j,
            "distance": value,
        }
        if metadata is not None:
            row["metadata"] = metadata
        rows.append(row)

    df = pd.DataFrame(rows)
    return df.reset_index(drop=True)


# ============================================================================
# Shape Distance Functions
# ============================================================================


def modify_matrix(
    mtx: npt.NDArray[np.floating[Any]],
    whiten: bool = True,
    normalize: bool = True,
    scale_variance: bool = True,
    unit_length_per_column: bool = False,
) -> npt.NDArray[np.floating[Any]]:
    """Preprocess matrix for shape comparison.

    Optionally whitens and/or normalizes to unit Frobenius norm.
    Supports both standard whitening (center + scale to unit variance) and
    Procrustes-style normalization (center only, then normalize to unit Frobenius norm).
    Also supports unit-length normalization per column for shape distance metrics.

    Parameters
    ----------
    mtx : ndarray of shape (n_samples, n_features)
        The matrix to preprocess.
    whiten : bool, default=True
        If True, center the matrix (remove mean per column).
        If scale_variance=True, also scale each column to unit variance.
        If unit_length_per_column=True, normalize each column to unit length instead.
    normalize : bool, default=True
        If True, scale the entire matrix to have Frobenius norm = 1.
    scale_variance : bool, default=True
        If True and whiten=True and unit_length_per_column=False, scale each column
        to unit variance (standard whitening).
        If False and whiten=True and unit_length_per_column=False, only center
        (Procrustes-style normalization).
        Ignored if unit_length_per_column=True.
    unit_length_per_column : bool, default=False
        If True and whiten=True, normalize each column to unit length (L2 norm = 1)
        instead of unit variance. This is used for shape distance metrics where
        ||x_i - y_j||^2 = 2(1 - ρ(x_i, y_j)) for mean-zero unit vectors.

    Returns
    -------
    ndarray
        The preprocessed matrix.

    Notes
    -----
    Automatic subsampling:
    - When matrices have different numbers of neurons (rows) and the method requires
      equal sizes (procrustes, one-to-one), subsampling is automatically applied.
    - The subsample size is set to the minimum number of neurons across both matrices.
    - Soft-matching can handle different neuron counts natively, so no automatic
      subsampling is applied.
    - Explicit subsampling parameters override automatic subsampling.

    Biological interpretation:
    - This function measures how similar the neural "code" or representation is
      between two populations. A low distance means the populations encode information
      in similar ways (e.g., similar tuning curves, similar response patterns).
    - Useful for comparing: different brain regions, before/after learning, different
      experimental conditions, or different animals.
    - The three methods differ in how they handle neuron identity:
      * Procrustes: Assumes neurons are in the same order (best for same recording session)
      * One-to-one: Finds best matching between neurons (best for shuffled or unknown identities)
      * Soft-matching: Allows partial matches (best for different-sized populations)

    Examples
    --------
    >>> import numpy as np
    >>> mtx = np.random.randn(50, 10)
    >>> # Standard whitening (center + unit variance per column)
    >>> mtx_white = modify_matrix(mtx, whiten=True, scale_variance=True)
    >>> # Procrustes-style (center only, then normalize)
    >>> mtx_proc = modify_matrix(mtx, whiten=True, scale_variance=False, normalize=True)
    >>> # Shape distance preprocessing (center + unit-length per column)
    >>> mtx_shape = modify_matrix(mtx, whiten=True, unit_length_per_column=True, normalize=False)
    """
    out = mtx.copy().astype(np.float64)

    if whiten:
        # Center around origin (remove mean per column)
        means = out.mean(axis=0, keepdims=True)
        out = out - means

        if unit_length_per_column:
            # Normalize each column to unit length (L2 norm = 1)
            # For mean-zero unit vectors: ||x_i - y_j||^2 = 2(1 - ρ(x_i, y_j))
            column_norms = np.linalg.norm(out, axis=0, keepdims=True)
            column_norms[column_norms == 0] = 1.0  # Avoid division by zero
            out = out / column_norms
        elif scale_variance:
            # Scale each column to unit variance (standard whitening)
            stds = out.std(axis=0, keepdims=True, ddof=1)
            stds[stds == 0] = 1.0  # Avoid division by zero
            out = out / stds
        # If scale_variance=False and unit_length_per_column=False, only centering is done

    if normalize:
        # Scale to unit Frobenius norm
        norm = np.linalg.norm(out, ord="fro")
        if norm > 0:
            out = out / norm

    return out


def align_mtx(
    mtx1: npt.NDArray[np.floating],
    mtx2: npt.NDArray[np.floating],
    rotate: bool = True,
    scale: bool = True,
    whiten: bool = True,
    norm: bool = True,
    scale_variance: bool = True,
) -> npt.NDArray[np.floating]:
    """Align mtx2 to mtx1 using Procrustes analysis.

    Args:
        mtx1: Reference matrix of shape (n, m).
        mtx2: Matrix to align, same shape as mtx1.
        rotate: If True, apply optimal rotation.
        scale: If True, apply optimal scaling. Note: For shape similarity
            comparisons with normalized matrices, this should be False to
            preserve the normalized shape comparison. Scaling is only
            appropriate for general Procrustes analysis where scale
            differences are meaningful.
        whiten: If True, center columns to zero mean before alignment.
        norm: If True, normalize by Frobenius norm before alignment.
        scale_variance: If True and whiten=True, scale each column to unit variance.
            If False and whiten=True, only center (Procrustes-style). Default True
            for backward compatibility.

    Returns:
        Aligned mtx2.

    Raises:
        ValueError: If matrices have different shapes or are not 2D.
    """
    if mtx1.shape != mtx2.shape:
        raise ValueError("Input matrices must have the same shape")
    if mtx1.ndim != 2:
        raise ValueError("Input matrices must be two-dimensional")

    mtx1 = modify_matrix(
        mtx1, whiten=whiten, normalize=norm, scale_variance=scale_variance
    )
    mtx2 = modify_matrix(
        mtx2, whiten=whiten, normalize=norm, scale_variance=scale_variance
    )

    # Find optimal orthogonal transformation (rotation/reflection)
    # R transforms mtx1 to mtx2: mtx1 @ R ≈ mtx2
    # So to align mtx2 to mtx1, we use: mtx2 @ R.T
    # The scale 's' is the sum of singular values (a similarity measure)
    # and also equals the optimal scaling factor for general Procrustes
    if rotate or scale:
        r, s = orthogonal_procrustes(mtx1, mtx2)
        if rotate:
            mtx2 = np.dot(mtx2, r.T)
        if scale:
            mtx2 *= s

    return mtx2


def shape_distance_procrustes(
    mtx1: npt.NDArray[np.floating[Any]],
    mtx2: npt.NDArray[np.floating[Any]],
    return_pairs: bool = True,
) -> tuple[float, dict[tuple[int, int], float] | None]:
    """
    Orthogonal Procrustes shape distance: d_O(X, Y) = min_{Q in O_N} ||X - Q Y||_F.

    Computes the minimal Frobenius norm after optimal rotation/reflection Q in neuron
    space (rows=neurons N, columns=conditions M). Assumes FIXED point correspondence
    (row i in X ↔ row i in Y), optimizing only global orthogonal transform. Theoretically
    d_O ≤ d_P (one-to-one) since permutations Π_N ⊂ orthogonals O_N; smallest distance
    when fixed corr good (same neuron order + rotation drift, e.g., sessions).

    Preprocessing (modify_matrix): column-center (translations), unit Frobenius (scale),
    no whitening/var scaling (matches scipy procrustes). Raw ||diff||_F ~ sqrt(N M); divide
    by sqrt(N) for RMS per-neuron to compare with one-to-one/soft.

    When to use:
    - Best: Known neuron IDs/order (identity perm optimal), manifolds rotated (e.g., HD rings).
    - Avoid: Shuffled neurons (d_O > d_P; use one-to-one/soft).

    Parameters
    ----------
    mtx1, mtx2 : ndarray of shape (N, M)
        Neural activity: rows=neurons, columns=conditions/stimuli. Same shape required.
    return_pairs : bool, default=True
        If True, return per-neuron distances under aligned Q (always (i,i) pairs).

    Returns
    -------
    distance : float
        Raw Frobenius disparity ||X_aligned - Y||_F (unnormalized total; norm / sqrt(N) for per-neuron RMS).
        Normalized by sqrt(N) for per-neuron RMS to make it comparable with one-to-one/soft matching.
    pairs : dict[tuple[int,int], float] or None
        {(i,i): row_dist_i} for aligned neurons.

    Raises
    ------
    ValueError : Different shapes.

    Notes
    -----
    SVD solution: Q = U V^T from svd(X @ Y^T). Exact, O(N^3).
    Theoretical: d_O ≤ d_P ≤ d_T (nested sets O ⊃ Π ⊃ T).

    Examples
    --------
    >>> mtx1 = np.random.randn(50, 10)
    >>> mtx2_rot = Q @ mtx1  # Simulated rotation (Q ortho)
    >>> dist, pairs = shape_distance_procrustes(mtx1, mtx2_rot)
    >>> print(f"d_O={dist:.3f} ≈0; RMS={dist/np.sqrt(50):.3f}")  # Tiny
    """
    if mtx1.shape != mtx2.shape:
        raise ValueError("Procrustes distance requires matrices with the same shape.")

    X = modify_matrix(
        mtx1,
        whiten=True,
        normalize=True,
        scale_variance=False,
    )
    Y = modify_matrix(
        mtx2,
        whiten=True,
        normalize=True,
        scale_variance=False,
    )

    # Orthogonal Procrustes in neuron space (rows are neurons, columns conditions)
    C = X @ Y.T
    from scipy.linalg import svd

    U, _, Vt = svd(C, full_matrices=False)
    R = U @ Vt  # (N, N)

    Y_aligned = R @ Y
    diff = X - Y_aligned

    distance = float(np.linalg.norm(diff, ord="fro"))

    if not return_pairs:
        return distance, None

    row_dists = np.linalg.norm(diff, axis=1)
    pairs = {(int(i), int(i)): float(d) for i, d in enumerate(row_dists)}

    return distance, pairs


def shape_distance_one_to_one(
    mtx1: npt.NDArray[np.float64],
    mtx2: npt.NDArray[np.float64],
    metric: str = "sqeuclidean",
) -> tuple[float, dict[tuple[int, int], float]]:
    """
    One-to-one (permutation) shape distance: d_P(X, Y) = min_{Π in Π_N} ||X - Π Y||_F.

    Optimal hard bijective matching via Hungarian assignment on neuron tuning costs.
    Permutation-invariant (finds best row remapping); d_P ≥ d_O (Π_N ⊂ O_N), ≤ d_T (hard).

    Normalization: RMS per-neuron sqrt( (1/N) sum_matched ||x_i - y_πi||^2 ) for scale match with others.
    Preprocessing: Same as Procrustes (center + unit Frobenius).

    When to use:
    - Unknown/shuffled neuron IDs, same N (e.g., random sorting).
    - Beats Procrustes if shuffle > rotation; loses to soft if unequal N needed.

    Parameters
    ----------
    mtx1 : ndarray of shape (n_samples, n_features)
        First matrix representing neural population activity.
    mtx2 : ndarray of shape (n_samples, n_features)
        Second matrix to compare with mtx1. Must have same number of samples.
    metric : str, default='sqeuclidean'
        Distance metric for computing transport costs. Passed to cdist.
        Common options: 'sqeuclidean', 'euclidean', 'cosine', 'correlation'.

    Returns
    -------
    distance : float
        Optimal transport cost with hard assignment (sum of transport plan * cost matrix).
        Uses squared distances when metric='sqeuclidean' for comparability with Procrustes.
        Lower values indicate more similar shapes. Normalized by sqrt(N) for per-neuron RMS
        to make it comparable with Procrustes and soft-matching.
    pairs : dict[tuple[int, int], float]
        Dictionary mapping optimal point assignments (i, j) -> distance,
        where point i from mtx1 is matched to point j from mtx2.
        Each i and j appears exactly once (bijective matching).
        Values represent the distance between matched points.

    Notes
    -----
    This method uses optimal transport (Earth Mover's Distance) with uniform distributions
    and equal sizes, which naturally enforces one-to-one matching (hard assignment).
    Unlike soft-matching, each point can only be matched to one other point.

    Matrices are centered (zero mean per column) and normalized to unit Frobenius norm
    before comparison, matching the normalization used by scipy.spatial.procrustes.

    Requires the `pot` package: pip install pot

    Examples
    --------
    >>> mtx1 = np.random.randn(50, 10)
    >>> mtx2 = np.random.randn(50, 10)
    >>> dist, pairs = shape_distance_one_to_one(mtx1, mtx2)
    >>> print(f"One-to-one distance: {dist:.3f}")
    >>> print(f"Number of matched pairs: {len(pairs)}")
    """
    # Preprocess matrices: center + unit Frobenius norm (matching Procrustes normalization)
    # This ensures distances are on the same scale as Procrustes
    X = modify_matrix(
        mtx1,
        whiten=True,
        normalize=True,
        scale_variance=False,
    )
    Y = modify_matrix(
        mtx2,
        whiten=True,
        normalize=True,
        scale_variance=False,
    )

    if X.shape != Y.shape:
        raise ValueError("One-to-one distance requires matrices with the same shape.")
    N, _ = X.shape

    # Cost between neurons (rows)
    cost = cdist(X, Y, metric=metric)  # (N, N)
    row_ind, col_ind = linear_sum_assignment(cost)

    if metric == "sqeuclidean":
        # cost_ij = ||x_i - y_j||^2
        # linear_sum_assignment gives sum of costs for matched pairs
        # For comparability with optimal transport (which uses uniform distributions),
        # we need to normalize by N to get the average, then take sqrt to match soft-matching scale
        total_sq = float(cost[row_ind, col_ind].sum())
        # Normalize by N (like optimal transport with uniform distributions)
        avg_sq = total_sq / N
        distance = float(np.sqrt(max(avg_sq, 0.0)))
        per_pair_dist = np.sqrt(cost[row_ind, col_ind])
    else:
        # cost_ij is a distance; square each and sum for squared distance
        per_pair_dist = cost[row_ind, col_ind]
        total_sq = float(np.sum(per_pair_dist**2))
        # Normalize by N (like optimal transport with uniform distributions)
        avg_sq = total_sq / N
        distance = float(np.sqrt(max(avg_sq, 0.0)))

    pairs: dict[tuple[int, int], float] = {
        (int(i), int(j)): float(d)
        for i, j, d in zip(row_ind, col_ind, per_pair_dist, strict=False)
    }

    return distance, pairs


def shape_distance_soft_matching(
    mtx1: npt.NDArray[np.float64],
    mtx2: npt.NDArray[np.float64],
    metric: str = "sqeuclidean",
    approx: bool = False,
    reg: float = 0.1,
    threshold: float = 1e-9,
) -> tuple[float, dict[tuple[int, int], float]]:
    """
    Soft-matching (OT/Wasserstein) distance: d_T(X,Y) = min_T sum T_ij C_ij, T in transport polytope.

    Fractional neuron assignment (uniform Dirac masses); handles unequal N! Relaxed perms:
    d_T ≤ d_P (T ⊃ Π_N/N), smoothest/lowest. Exact EMD (approx=False) ≈ d_P equal N; Sinkhorn
    < due to reg (violates strict ≤ d_P).

    Normalization: sqrt(<T,C>) (RMS-like under uniform). Preprocess: Procrustes-match.

    When to use:
    - Unequal N, unknown IDs, smooth/differentiable metric (e.g., cross-region/animals).
    - Loosest: Always ≤ others; fractional good for pop codes.


    Reference:
    https://proceedings.mlr.press/v243/khosla24a/khosla24a.pdf

    Parameters
    ----------
    mtx1 : ndarray of shape (n_samples1, n_features)
        First matrix representing neural population activity.
    mtx2 : ndarray of shape (n_samples2, n_features)
        Second matrix to compare with mtx1. Can have different number of samples.
    metric : str, default='sqeuclidean'
        Distance metric for computing transport costs. Passed to cdist.
        Common options: 'sqeuclidean', 'euclidean', 'cosine', 'correlation'.
        Use 'sqeuclidean' for comparability with one-to-one and Procrustes.
    approx : bool, default=False
        If True, use Sinkhorn algorithm (faster, approximate).
        If False, use exact EMD algorithm (slower, exact).
        Note: With exact EMD and equal-sized distributions, soft-matching may equal
        one-to-one (both give optimal hard assignment). Sinkhorn approximation may
        violate the theoretical ordering property soft-matching ≤ one-to-one ≤ procrustes.
    reg : float, default=0.1
        Entropic regularization parameter for Sinkhorn algorithm.
        Higher values lead to more uniform (diffuse) transport plans.
        Only used when approx=True.

    Returns
    -------
    distance : float
        Square root of the optimal transport cost sqrt(W2^2 - Wasserstein-like distance).
        Lower values indicate more similar point distributions.
    pairs : dict[tuple[int, int], float]
        Dictionary mapping point pairs (i, j) to transport probabilities.
        Unlike hard matching, multiple pairs can involve the
        same point i or j (soft assignment).
        Values represent the fraction of mass transported from point i to point j.

    Raises
    ------
    ImportError
        If the POT (Python Optimal Transport) library is not installed.

    Notes
    -----
    Unlike hard one-to-one matching, optimal transport allows fractional
    assignment of mass between points, providing a smoother distance metric.
    This is particularly useful when point clouds have different sizes or when
    you want a continuous, differentiable distance measure.

    Matrices are centered (zero mean per column) and normalized to unit Frobenius norm
    before comparison, matching the normalization used by scipy.spatial.procrustes.
    Points are treated as uniform distributions (equal mass at each point).

    Requires the `pot` package: pip install pot

    Examples
    --------
    >>> mtx1 = np.random.randn(50, 10)
    >>> mtx2 = np.random.randn(60, 10)  # Different size OK
    >>> dist, pairs = shape_distance_soft_matching(mtx1, mtx2, approx=True)
    >>> print(f"Wasserstein distance: {dist:.3f}")
    >>> print(f"Number of significant transport pairs: {len(pairs)}")
    >>> # Check transport probabilities sum to ~1
    >>> print(f"Total transport mass: {sum(pairs.values()):.3f}")
    """
    if not OT_AVAILABLE:
        msg = "soft-matching requires the 'pot' library. Install with: pip install pot"
        raise ImportError(msg)

    X = modify_matrix(
        mtx1,
        whiten=True,
        normalize=True,
        scale_variance=False,
    )
    Y = modify_matrix(
        mtx2,
        whiten=True,
        normalize=True,
        scale_variance=False,
    )

    # Cost between neurons (rows)
    C = cdist(X, Y, metric=metric)  # (n1, n2)

    n1, n2 = X.shape[0], Y.shape[0]
    a = np.full(n1, 1.0 / n1, dtype=np.float64)
    b = np.full(n2, 1.0 / n2, dtype=np.float64)

    if approx:
        T = ot.sinkhorn(a, b, C, reg)
    else:
        T = ot.emd(a, b, C)

    ot_cost = float(np.sum(T * C))
    distance = float(np.sqrt(max(ot_cost, 0.0)))

    i_idx, j_idx = np.where(T > threshold)
    pairs: dict[tuple[int, int], float] = {
        (int(i), int(j)): float(T[i, j]) for i, j in zip(i_idx, j_idx, strict=False)
    }

    return distance, pairs


def shape_distance(
    mtx1: npt.NDArray[np.float64],
    mtx2: npt.NDArray[np.float64],
    method: Literal["procrustes", "one-to-one", "soft-matching"] = "procrustes",
    metric: str = "sqeuclidean",
    subsamples: Sequence[int] | None = None,
    subsample_axes: Sequence[int] | None = None,
    repeats: int = 10,
    seed: int | None = None,
    plot: bool = False,
    **method_kwargs: Any,
) -> tuple[
    Union[float, npt.NDArray[np.float64]],
    Union[Dict[Tuple[int, int], float], List[Dict[Tuple[int, int], float]]],
    Dict[str, Any],
]:
    """
    Compute a shape distance between two neural population activity matrices,
    optionally using repeated random subsampling.

    This function measures how similar the "shape" or structure of neural activity
    patterns are between two populations. In biology, this helps answer questions like:
    "Do these two brain regions encode information in the same way?" or "Has the neural
    representation changed after learning or between different conditions?"

    The function automatically handles cases where the two populations have different
    numbers of neurons by using subsampling when needed. For methods that require
    equal-sized populations (procrustes, one-to-one), subsampling is automatically
    applied if the matrices have different numbers of neurons.

    Matrix format:
    - Rows (axis 0) = neurons: Each row represents one neuron's activity pattern
    - Columns (axis 1) = features/conditions: Each column represents a feature dimension
      (e.g., different stimuli, time points, or task conditions)

    This function provides a unified interface for multiple shape comparison
    methods and an optional subsampling scheme for robustness and fair
    comparison when matrices differ in size. When no subsampling is requested,
    the selected method is applied once to the full matrices. When subsampling
    is enabled (either automatically or manually), the function repeatedly draws
    random subsets along specified axes, evaluates the distance on each subset,
    and returns all per-run distances together with detailed indexing metadata.

    Parameters
    ----------
    mtx1 : ndarray of shape (n_neurons1, n_features)
        First matrix representing neural population activity.
        Rows = neurons, Columns = features/conditions (e.g., stimuli, time points).
    mtx2 : ndarray of shape (n_neurons2, n_features)
        Second matrix to compare with `mtx1`.
        Rows = neurons, Columns = features/conditions (must match mtx1).
        Can have a different number of neurons (rows) than mtx1.
    method : {'procrustes', 'one-to-one', 'soft-matching'}, default='procrustes'
        Shape comparison method:
        - 'procrustes': Optimal orthogonal alignment (rotation/reflection).
            Preserves point correspondence, best suited for aligned data with
            fixed neuron identity.
        - 'one-to-one': Optimal hard assignment (Hungarian/OT-like).
            Permutation-invariant, finds a bijective matching between points.
        - 'soft-matching': Optimal transport with soft assignment.
            Allows fractional mass between points, handles different point
            cloud sizes and yields a smoother distance.
    metric : str, default='sqeuclidean'
        Distance metric used for 'one-to-one' and 'soft-matching'. Ignored
        for 'procrustes'. Use 'sqeuclidean' to obtain squared distances that
        are comparable to the Procrustes disparity.
    subsamples : sequence of int or None, default=None
        Subsample sizes along each axis listed in `subsample_axes`. If None,
        subsampling is automatically applied when matrices have different numbers
        of neurons (for methods that require equal sizes). If explicitly provided,
        overrides automatic subsampling. For typical neuron subsampling, use `[0]` (rows).
    subsample_axes : sequence of int or None, default=None
        Axes along which to perform random subsampling. Must have the same
        length as `subsamples` when explicitly provided. For automatic subsampling
        (when matrices differ in neuron count), defaults to `[0]` (rows/neurons).
    repeats : int, default=10
        Number of independent subsampling runs when subsampling is enabled
        (either automatically or manually). Ignored when no subsampling is needed
        (in which case a single run is performed).
    seed : int or None, default=None
        Seed for the NumPy random number generator used for subsampling.
        Set for reproducible subsampling; leave as None for non-deterministic
        behavior.
    **method_kwargs
        Additional keyword arguments passed to the method-specific
        implementation. For example:
        - For 'soft-matching': approx (bool), reg (float)

    Returns
    -------
    distance : float or ndarray of shape (repeats,)
        If subsampling is disabled (`subsamples` or `subsample_axes` is None),
        returns a single scalar distance. If subsampling is enabled, returns a
        1D array containing the distance from each subsampling run.
        Lower values indicate more similar shapes.
    pairs : dict or list of dict
        Point correspondence information. For single-run mode (no subsampling),
        this is a single dictionary:
        - 'procrustes': {(i, i): distance_i} aligned point distances.
        - 'one-to-one': {(i, j): distance_ij} optimal bijective matches.
        - 'soft-matching': {(i, j): probability_ij} transport probabilities.
        For subsampling mode, returns a list of such dictionaries, one per run.
    metadata : dict
        Dictionary with bookkeeping information about the computation, e.g.:
        - 'method': selected method name.
        - 'metric': distance metric used.
        - 'mtx1_shape', 'mtx2_shape': original input shapes.
        - 'runs': number of runs performed.
        - 'indices' (only when subsampling is enabled): list of per-run
          index mappings describing which rows were selected from each
          matrix in each subsampling iteration.

    Raises
    ------
    ValueError
        If an unknown method is specified, if inputs are not two-dimensional,
        if matrices have different numbers of features (columns), or if
        subsampling parameters are inconsistent (e.g. mismatched length
        of `subsamples` and `subsample_axes`).

    Notes
    -----
    Automatic subsampling:
    - When matrices have different numbers of neurons (rows) and the method requires
      equal sizes (procrustes, one-to-one), subsampling is automatically applied.
    - The subsample size is set to the minimum number of neurons across both matrices.
    - Soft-matching can handle different neuron counts natively, so no automatic
      subsampling is applied.
    - Explicit subsampling parameters override automatic subsampling.

    Biological interpretation:
    - This function measures how similar the neural "code" or representation is
      between two populations. A low distance means the populations encode information
      in similar ways (e.g., similar tuning curves, similar response patterns).
    - Useful for comparing: different brain regions, before/after learning, different
      experimental conditions, or different animals.
    - The three methods differ in how they handle neuron identity:
      * Procrustes: Assumes neurons are in the same order (best for same recording session)
      * One-to-one: Finds best matching between neurons (best for shuffled or unknown identities)
      * Soft-matching: Allows partial matches (best for different-sized populations)

    Examples
    --------
    >>> rng = np.random.default_rng(42)
    >>> mtx1 = rng.standard_normal((50, 10))
    >>> mtx2 = rng.standard_normal((50, 10))
    >>>
    >>> # Procrustes alignment on full matrices (single run)
    >>> dist, pairs, meta = shape_distance(mtx1, mtx2, method="procrustes")
    >>>
    >>> # One-to-one matching with neuron subsampling
    >>> dist_runs, pairs_runs, meta = shape_distance(
    ...     mtx1, mtx2,
    ...     method="one-to-one",
    ...     metric="sqeuclidean",
    ...     subsamples=[30],
    ...     subsample_axes=[0],
    ...     repeats=5,
    ...     seed=0,
    ... )
    >>>
    >>> # Soft optimal transport between unequal-size matrices, no subsampling
    >>> mtx3 = rng.standard_normal((60, 10))
    >>> dist, pairs, meta = shape_distance(
    ...     mtx1, mtx3,
    ...     method="soft-matching",
    ...     approx=True,
    ...     reg=0.05,
    ... )
    """
    if mtx1.ndim != 2 or mtx2.ndim != 2:
        raise ValueError("Input matrices must be two-dimensional")

    def core_compute(
        a: npt.NDArray[np.float64],
        b: npt.NDArray[np.float64],
    ) -> tuple[float, Dict[Tuple[int, int], float]]:
        match method:
            case "procrustes":
                return shape_distance_procrustes(a, b)
            case "one-to-one":
                return shape_distance_one_to_one(a, b, metric=metric)
            case "soft-matching":
                return shape_distance_soft_matching(
                    a, b, metric=metric, **method_kwargs
                )
            case _:
                raise ValueError(
                    f"Unknown method '{method}'. "
                    "Choose 'procrustes', 'one-to-one', or 'soft-matching'."
                )

    meta: Dict[str, Any] = {
        "method": method,
        "metric": metric,
        "mtx1_shape": mtx1.shape,
        "mtx2_shape": mtx2.shape,
    }

    # Check if matrices have different numbers of neurons (rows)
    n_neurons1, n_features1 = mtx1.shape
    n_neurons2, n_features2 = mtx2.shape

    if n_features1 != n_features2:
        raise ValueError(
            f"Matrices must have the same number of features (columns). "
            f"Got {n_features1} and {n_features2}."
        )

    # Determine if automatic subsampling is needed
    # Procrustes and one-to-one require equal neuron counts
    # Soft-matching can handle different neuron counts natively
    needs_subsampling = False
    if method in ("procrustes", "one-to-one") and n_neurons1 != n_neurons2:
        needs_subsampling = True
        if subsamples is None or subsample_axes is None:
            # Automatically set up subsampling to the smaller neuron count
            min_neurons = min(n_neurons1, n_neurons2)
            subsamples = [min_neurons]
            subsample_axes = [0]  # Subsample along rows (neurons)
            meta["auto_subsampling"] = True
            meta["subsample_size"] = min_neurons
        else:
            meta["auto_subsampling"] = False
    else:
        meta["auto_subsampling"] = False

    # No subsampling → single run, scalar distance
    if (subsamples is None or subsample_axes is None) and not needs_subsampling:
        dist, pairs = core_compute(mtx1, mtx2)
        meta["runs"] = 1
        return dist, pairs, meta

    # With subsampling → run_with_subsampling on the distance-only wrapper
    if len(subsamples) != len(subsample_axes):
        raise ValueError("subsamples and subsample_axes must have the same length")

    def distance_only(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> float:
        d, _p = core_compute(a, b)
        return d

    values, meta_sub = run_with_subsampling(
        func=distance_only,
        arrays=(mtx1, mtx2),
        subsamples=subsamples,
        subsample_axes=subsample_axes,
        repeats=repeats,
        seed=seed,
    )

    # If you want pairs for subsampled runs as well, you can recompute on each
    pairs_list: List[Dict[Tuple[int, int], float]] = []
    for per_array_indexers in meta_sub["indices"]:
        idx1 = [per_array_indexers[0].get(ax, slice(None)) for ax in range(mtx1.ndim)]
        idx2 = [per_array_indexers[1].get(ax, slice(None)) for ax in range(mtx2.ndim)]
        sub_mtx1 = mtx1[tuple(idx1)]
        sub_mtx2 = mtx2[tuple(idx2)]
        _d, p = core_compute(sub_mtx1, sub_mtx2)
        pairs_list.append(p)

    meta["runs"] = values.shape[0]
    meta["indices"] = meta_sub["indices"]

    return values, pairs_list, meta


def _comparison_results_to_dataframe(
    results: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Convert comparison results from HDF5 format to DataFrame.

    Parameters
    ----------
    results : dict[str, Any]
        Dictionary with result_key -> {scalars, arrays} structure

    Returns
    -------
    DataFrame
        Long-format DataFrame with comparison results
    """
    rows = []
    for _result_key, result_data in results.items():
        # Extract scalar attributes
        row = {}
        if "attributes" in result_data:
            for key, value in result_data["attributes"].items():
                row[key] = value

        # Reconstruct pairs dict if present
        if "arrays" in result_data:
            arrays = result_data["arrays"]
            if "pair_indices" in arrays and "pair_values" in arrays:
                pair_indices = arrays["pair_indices"]
                pair_values = arrays["pair_values"]
                # Convert to dict with string keys for DataFrame compatibility
                pairs_dict = {
                    f"{idx[0]},{idx[1]}": val
                    for idx, val in zip(pair_indices, pair_values)
                }
                row["pairs"] = pairs_dict
            else:
                row["pairs"] = None
        else:
            row["pairs"] = None

        rows.append(row)

    return pd.DataFrame(rows)
