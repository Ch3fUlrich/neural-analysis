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
from typing import TYPE_CHECKING, Any, Literal, TypedDict, TypeVar, cast

import numpy as np
import numpy.typing as npt
from scipy.linalg import orthogonal_procrustes
from scipy.optimize import linear_sum_assignment
from scipy.spatial import procrustes
from scipy.spatial.distance import cdist

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
    "compare_distributions",
    "compare_distribution_groups",
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


def _deserialize_pairs(arrays: Mapping[str, Any] | None) -> dict[tuple[int, int], float] | None:
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
        pairs = {
            (int(i), int(j)): float(val)
            for (i, j), val in result[1].items()
        }
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
    """Normalize result from arbitrary comparison function."""
    if isinstance(result, tuple):
        return float(result[0]), result[1]
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


@log_calls(level=logging.DEBUG)
def compare_distributions(
    points1: npt.ArrayLike,
    points2: npt.ArrayLike,
    metric: Literal[
        "wasserstein",
        "kolmogorov-smirnov",
        "jensen-shannon",
        "euclidean",
        "manhattan",
        "mahalanobis",
        "cosine",
        "procrustes",
        "one-to-one",
        "soft-matching",
    ] = "wasserstein",
    dataset_i: str | None = None,
    dataset_j: str | None = None,
    comparison_name: str | None = None,
    save_path: str | Path | None = None,
    **metric_kwargs: object,
) -> float | tuple[float, dict[tuple[int, int], float]]:
    """Compare two point distributions using a specified metric.

    .. deprecated:: 1.5.0
        Use :func:`~neural_analysis.metrics.pairwise_metrics.compare_datasets` with
        ``mode='between'`` instead. This function will be removed in version 2.0.0.

    **RECOMMENDED**: Use compare_datasets() for new code, which provides a unified
    API with better type safety and more flexible return types.

    This function wraps compute_pairwise_matrix() for between-mode comparisons
    and optionally saves results to HDF5.

    Parameters
    ----------
    points1, points2 : array-like
        Point distributions to compare. Shape: (n_samples, n_features).
    metric : str, default="wasserstein"
        Distance metric to use:

        **Distribution metrics:**
        - "wasserstein": Wasserstein distance (Earth Mover's Distance)
        - "kolmogorov-smirnov": K-S statistic (max over dimensions)
        - "jensen-shannon": Jensen-Shannon divergence (histogram-based)
        - "euclidean": Euclidean distance between distribution centers
        - "mahalanobis": Mahalanobis distance between distributions
        - "cosine": Cosine similarity of mean vectors

        **Shape metrics:**
        - "procrustes": Procrustes distance after optimal alignment
        - "one-to-one": Bipartite matching distance
        - "soft-matching": Soft assignment via optimal transport
    dataset_i : str, optional
        Name of first dataset. Required if save_path is provided.
    dataset_j : str, optional
        Name of second dataset. Required if save_path is provided.
    comparison_name : str, optional
        Name for comparison group. If None and save_path is provided,
        defaults to "default". Used as top-level group in HDF5.
    save_path : str or Path, optional
        Path to HDF5 file for saving results. If None, results are not saved.
        If provided, dataset_i and dataset_j must also be specified.
    **metric_kwargs
        Additional keyword arguments passed to the distance function.
        For shape metrics: reg, approx, etc.

    Returns
    -------
    float or tuple
        For distribution metrics:
            float: Distance or similarity value. Lower is more similar for
            distance metrics, higher is more similar for cosine (range [0, 1]).
        For shape metrics:
            tuple: (distance: float, pairs: dict[tuple[int, int], float])

    See Also
    --------
    distribution_distance : More flexible function with within/between modes
    shape_distance : Direct access to shape comparison methods

    Notes
    -----
    - Wasserstein: measures "work" to transform one distribution to another.
    - K-S: maximum CDF difference; sensitive to shape/location differences.
    - Jensen-Shannon: symmetric KL divergence variant; bounded in [0, 1].
    - Euclidean: simple center-to-center distance.
    - Mahalanobis: accounts for covariance structure.
    - Cosine: direction similarity; invariant to scale.
    - Procrustes: optimal orthogonal alignment distance.
    - One-to-one: optimal bijective matching distance.
    - Soft-matching: optimal transport with fractional assignments.

    Examples
    --------
    >>> import numpy as np
    >>> p1 = np.random.randn(100, 3)
    >>> p2 = np.random.randn(100, 3) + 1.0

    >>> # Distribution comparison
    >>> dist = compare_distributions(p1, p2, metric="wasserstein")
    >>> print(f"Wasserstein distance: {dist:.3f}")

    >>> # Shape comparison
    >>> dist, pairs = compare_distributions(p1, p2, metric="procrustes")
    >>> print(f"Procrustes distance: {dist:.3f}")

    >>> # Save single comparison to file
    >>> dist = compare_distributions(
    ...     p1, p2,
    ...     metric="wasserstein",
    ...     dataset_i="condition_A",
    ...     dataset_j="condition_B",
    ...     comparison_name="experiment_001",
    ...     save_path="output/comparisons.h5"
    ... )
    """
    # Validate save parameters
    if save_path is not None:
        if dataset_i is None or dataset_j is None:
            raise ValueError(
                "dataset_i and dataset_j must be provided when save_path is specified"
            )
        if comparison_name is None:
            comparison_name = "default"

    # Use unified pairwise computation system
    # This reduces code duplication and provides single source of responsibility
    result = compute_pairwise_matrix(
        points1,
        points2,
        metric=metric,
        parallel=True,
        **metric_kwargs,
    )

    # Handle different return types from unified pairwise system
    shape_metrics = {"procrustes", "one-to-one", "soft-matching"}
    if metric in shape_metrics:
        # Shape metrics return tuple[float, dict[tuple[int, int], float]]
        assert isinstance(result, tuple), f"Expected tuple for shape metric {metric}"
        value, pairs_dict = result
    else:
        # Distribution/distance metrics return float or matrix
        if isinstance(result, np.ndarray):
            # Point-to-point metrics return matrix; take mean as summary
            value = float(np.mean(result))
            pairs_dict = None
        elif isinstance(result, tuple):
            # Some metrics may return tuple even for non-shape metrics
            value = float(result[0])
            pairs_dict = result[1]
        else:
            # Distribution metrics return scalar
            value = float(result)
            pairs_dict = None

    # NOTE: save_path parameter deprecated - save functionality moved to
    # compare_datasets() with auto-save/load in Phase 4
    if save_path is not None:
        logger.warning(
            "save_path parameter in compare_distributions() is deprecated. "
            "Use compare_datasets() with save_path for auto-save/load functionality."
        )

    # Return result in original format
    if metric in shape_metrics:
        return (value, pairs_dict)  # type: ignore[return-value]
    else:
        return value


def compare_distribution_groups(
    group_vectors: dict[str | tuple[str, ...], npt.NDArray[np.floating]],
    compare_type: Literal["inside", "between"] = "between",
    metric: Literal[
        "euclidean",
        "manhattan",
        "cosine",
        "mahalanobis",
        "wasserstein",
        "kolmogorov_smirnov",
        "ks",
        "jensen_shannon",
        "js",
        "procrustes",
        "one-to-one",
        "soft-matching",
    ] = "wasserstein",
    **metric_kwargs: Any,
) -> (
    dict[str, npt.NDArray[np.floating]]
    | dict[str | tuple[str, ...], npt.NDArray[np.floating]]
):
    """Compare distributions within or between groups (legacy wrapper).

    **RECOMMENDED**: Use compare_datasets() for new code. This function is
    maintained for backward compatibility but has limitations compared to
    the unified Phase 3 API.

    Wraps Phase 3 API functions (compute_within_distances, compute_between_distances)
    to provide group-level comparisons with legacy return format.

    Parameters
    ----------
    group_vectors : dict[str, Any]
        Dictionary mapping group identifiers to point arrays (n_samples, n_features).
    compare_type : {"inside", "between"}, default="between"
        - "inside": Compare each group to itself (self-similarity).
            Only works with point-to-point metrics (euclidean, manhattan, etc.).
        - "between": Compare each group to all others.
    metric : str, default="wasserstein"
        Distance metric to use. Supported metrics:

        **Point-to-point metrics** (work with "inside" mode):
        - "euclidean": Euclidean distance
        - "manhattan": Manhattan (L1) distance
        - "cosine": Cosine similarity
        - "mahalanobis": Mahalanobis distance

        **Distribution metrics** (only "between" mode):
        - "wasserstein": Wasserstein distance (Earth Mover's Distance)
        - "kolmogorov_smirnov" or "ks": Kolmogorov-Smirnov statistic
        - "jensen_shannon" or "js": Jensen-Shannon divergence

        **Shape metrics** (only "between" mode):
        - "procrustes": Procrustes distance after optimal alignment
        - "one-to-one": Bipartite matching distance
        - "soft-matching": Soft assignment via optimal transport
    **metric_kwargs : dict[str, Any], optional
        Additional keyword arguments passed to the metric function.
        For shape metrics: reg, whiten, normalize, metric, approx, etc.

    Returns
    -------
    dict
        If compare_type == "inside":
            Returns {"mean": ndarray, "std": ndarray} with shape (n_groups,).
        If compare_type == "between":
            Returns {group_name: ndarray} where each array has shape (n_groups,)
            containing distances from that group to all others.

    Raises
    ------
    ValueError
        If an invalid metric-mode combination is used.

    Notes
    -----
    **Migration to Phase 3 API**:

    For new code, consider using the Phase 3 API directly:

    >>> # Instead of compare_distribution_groups with compare_type="between"
    >>> from neural_analysis.metrics import compute_all_pairs
    >>> results = compute_all_pairs(group_vectors, metric="wasserstein")
    >>> # results is dict[str, dict[str, float]]

    >>> # For single within-group comparison
    >>> from neural_analysis.metrics import compute_within_distances
    >>> mean_dist = compute_within_distances(data, metric="euclidean")

    Examples
    --------
    >>> # Distribution comparison between groups
    >>> groups = {
    ...     "A": np.random.randn(50, 3),
    ...     "B": np.random.randn(50, 3) + 1.0,
    ...     "C": np.random.randn(50, 3) + 2.0,
    ... }
    >>> similarities = compare_distribution_groups(
    ...     groups, compare_type="between", metric="wasserstein"
    ... )
    >>> similarities["A"]  # distances from A to all groups

    >>> # Shape comparison between groups
    >>> similarities = compare_distribution_groups(
    ...     groups, compare_type="between", metric="procrustes"
    ... )

    >>> # Within-group variability (only point-to-point metrics)
    >>> within_stats = compare_distribution_groups(
    ...     groups, compare_type="inside", metric="euclidean"
    ... )
    >>> within_stats["mean"]  # mean within-group distances
    """
    # Import Phase 3 functions
    from neural_analysis.metrics.pairwise_metrics import (
        compute_between_distances,
        compute_within_distances,
    )

    n_groups = len(group_vectors)

    logger.info(
        f"Comparing {n_groups} groups with "
        f"compare_type='{compare_type}', metric='{metric}'"
    )

    # Use match/case for cleaner dispatch
    match compare_type:
        case "inside":
            # Within-group variability using Phase 3 API
            # compute_within_distances validates that metric is point-to-point
            means = np.zeros(n_groups)
            stds = np.zeros(n_groups)

            for idx, (name, points) in enumerate(group_vectors.items()):
                if len(points) < 2:
                    logger.debug(f"Group '{name}' has <2 points, skipping")
                    means[idx] = 0.0
                    stds[idx] = 0.0
                    continue

                # Get full distance matrix and compute statistics
                try:
                    dist_matrix = compute_within_distances(
                        points,
                        metric=metric,
                        return_matrix=True,
                        **metric_kwargs,
                    )
                    # Extract upper triangle (excluding diagonal)
                    mask = np.triu(np.ones_like(dist_matrix, dtype=bool), k=1)
                    dists = dist_matrix[mask]
                    means[idx] = float(np.mean(dists))
                    stds[idx] = float(np.std(dists))
                except ValueError as e:
                    # Re-raise with more context
                    raise ValueError(
                        f"Metric '{metric}' cannot be used with "
                        f"compare_type='inside'. Only point-to-point metrics "
                        f"(euclidean, manhattan, cosine, mahalanobis) are allowed "
                        f"for within-group comparisons."
                    ) from e

            logger.info(f"Within-group statistics computed: mean={means}, std={stds}")
            return {"mean": means, "std": stds}  # type: ignore[return-value]

        case "between":
            # Between-group distances using Phase 3 API
            similarities = {}
            for _i, (name_i, group_i) in enumerate(group_vectors.items()):
                dists_to_all = np.zeros(n_groups)
                for j, (_name_j, group_j) in enumerate(group_vectors.items()):
                    # Use Phase 3 API - handles all metric types
                    dist = compute_between_distances(
                        group_i,
                        group_j,
                        metric=metric,
                        return_matrix=False,  # Get mean distance
                        **metric_kwargs,
                    )
                    # Extract value from dict
                    dist_value = dist["value"] if isinstance(dist, dict) else dist
                    dists_to_all[j] = float(dist_value)

                similarities[name_i] = dists_to_all
                logger.debug(f"Group '{name_i}' distances to all: {dists_to_all}")

            logger.info(f"Between-group distances computed for {n_groups} groups")
            return similarities  # type: ignore[return-value]

        case _:
            raise ValueError(
                f"Unknown compare_type '{compare_type}'. Choose 'inside' or 'between'."
            )


# ============================================================================
# Batch Comparison Utilities
# ============================================================================


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
            datasets[dataset_i], datasets[dataset_j], metric_name, metric_kwargs=metric_kwargs
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
) -> npt.NDArray[np.floating[Any]]:
    """Preprocess matrix for shape comparison.

    Optionally whitens and/or normalizes to unit Frobenius norm.

    Parameters
    ----------
    mtx : ndarray of shape (n_samples, n_features)
        The matrix to preprocess.
    whiten : bool, default=True
        If True, center and scale each feature (column) to unit variance.
    normalize : bool, default=True
        If True, scale the entire matrix to have Frobenius norm = 1.

    Returns
    -------
    ndarray
        The preprocessed matrix.

    Notes
    -----
    - Whitening standardizes features to zero mean and unit variance
    - Normalization scales the overall matrix magnitude
    - Both operations preserve shape structure while removing scale effects
    """
    out = mtx.copy().astype(np.float64)

    if whiten:
        # Center and scale each column
        means = out.mean(axis=0, keepdims=True)
        stds = out.std(axis=0, keepdims=True, ddof=1)
        stds[stds == 0] = 1.0  # Avoid division by zero
        out = (out - means) / stds

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

    Returns:
        Aligned mtx2.

    Raises:
        ValueError: If matrices have different shapes or are not 2D.
    """
    if mtx1.shape != mtx2.shape:
        raise ValueError("Input matrices must have the same shape")
    if mtx1.ndim != 2:
        raise ValueError("Input matrices must be two-dimensional")

    mtx1 = modify_matrix(mtx1, whiten=whiten, normalize=norm)
    mtx2 = modify_matrix(mtx2, whiten=whiten, normalize=norm)

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
    mtx1: npt.NDArray[np.float64],
    mtx2: npt.NDArray[np.float64],
) -> tuple[float, dict[tuple[int, int], float]]:
    """Compute shape distance using Procrustes alignment.

    This method aligns the two matrices using Procrustes analysis and returns
    the residual disparity after optimal rotation/reflection, along with
    point-to-point correspondence information.

    Parameters
    ----------
    mtx1 : ndarray of shape (n_samples, n_features)
        First matrix representing neural population activity.
    mtx2 : ndarray of shape (n_samples, n_features)
        Second matrix to compare with mtx1. Must have same shape as mtx1.

    Returns
    -------
    distance : float
        Procrustes disparity (sum of squared Euclidean distances after
        optimal alignment). Lower values indicate more similar shapes.
    pairs : dict[tuple[int, int], float]
        Dictionary mapping point index pairs (i, i) to their post-alignment
        distances. Since Procrustes preserves point correspondence, each
        point i in mtx1 is aligned to point i in mtx2.

    Notes
    -----
    The Procrustes method finds the optimal orthogonal transformation
    (rotation + reflection) that minimizes the distance between matrices.
    The scipy.spatial.procrustes function automatically standardizes both
    matrices (zero mean, unit variance per column, unit Frobenius norm).

    Examples
    --------
    >>> mtx1 = np.random.randn(50, 10)
    >>> mtx2 = np.random.randn(50, 10)
    >>> dist, pairs = shape_distance_procrustes(mtx1, mtx2)
    >>> print(f"Procrustes distance: {dist:.3f}")
    >>> print(f"Number of aligned pairs: {len(pairs)}")
    """
    m1, m2, disparity = procrustes(mtx1, mtx2)

    # Compute distance between pairs
    distances = np.linalg.norm(m1 - m2, axis=1)
    # Create pairs dictionary with point indices and their post-alignment squared distances
    pairs = {(i, i): float(distance) for i, distance in enumerate(distances)}

    return float(disparity), pairs


def shape_distance_one_to_one(
    mtx1: npt.NDArray[np.float64],
    mtx2: npt.NDArray[np.float64],
    metric: str = "euclidean",
) -> tuple[float, dict[tuple[int, int], float]]:
    """Compute shape distance using optimal one-to-one point matching via optimal transport.

    Uses optimal transport with hard assignment constraint (one-to-one matching) to find
    the optimal bijective matching between points. Similar to soft-matching but enforces
    that each point is matched to exactly one other point (hard assignment).

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
        Lower values indicate more similar shapes.
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

    Matrices are normalized to unit Frobenius norm before comparison (consistent with
    procrustes and soft-matching methods).

    Requires the `pot` package: pip install pot

    Examples
    --------
    >>> mtx1 = np.random.randn(50, 10)
    >>> mtx2 = np.random.randn(50, 10)
    >>> dist, pairs = shape_distance_one_to_one(mtx1, mtx2)
    >>> print(f"One-to-one distance: {dist:.3f}")
    >>> print(f"Number of matched pairs: {len(pairs)}")
    """
    if not OT_AVAILABLE:
        msg = "one-to-one requires the 'pot' library. Install with: pip install pot"
        raise ImportError(msg)

    # Normalize matrices to unit Frobenius norm (consistent with procrustes and soft-matching)
    m1 = modify_matrix(mtx1, whiten=False, normalize=True)
    m2 = modify_matrix(mtx2, whiten=False, normalize=True)

    # Compute cost matrix
    cost_matrix = cdist(m1, m2, metric=metric)

    # Uniform distributions (equal mass at each point)
    a = np.ones(m1.shape[0]) / m1.shape[0]
    b = np.ones(m2.shape[0]) / m2.shape[0]

    # Compute optimal transport plan with hard assignment (EMD gives one-to-one matching)
    transport_plan = ot.emd(a, b, cost_matrix)

    # Compute distance: sum of transport plan * cost matrix
    # For sqeuclidean metric, this gives sum of squared distances (comparable to Procrustes)
    distance = np.sum(transport_plan * cost_matrix)

    # Extract pairs with non-zero transport (hard assignment: exactly one match per point)
    i_indices, j_indices = np.where(transport_plan > 0)
    pairs = {
        (int(i), int(j)): float(cost_matrix[i, j])
        for i, j in zip(i_indices, j_indices, strict=False)
    }
    return float(distance), pairs

    return float(distance), pairs


def shape_distance_soft_matching(
    mtx1: npt.NDArray[np.float64],
    mtx2: npt.NDArray[np.float64],
    metric: str = "euclidean",
    approx: bool = False,
    reg: float = 0.1,
) -> tuple[float, dict[tuple[int, int], float]]:
    """Compute shape distance using soft optimal transport matching.

    Uses optimal transport to compute a soft matching between point distributions,
    allowing fractional assignment of mass. Can use exact (Earth Mover's Distance)
    or approximate (Sinkhorn) algorithms.

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
    approx : bool, default=False
        If True, use Sinkhorn algorithm (faster, approximate).
        If False, use exact EMD algorithm (slower, exact).
    reg : float, default=0.1
        Entropic regularization parameter for Sinkhorn algorithm.
        Higher values lead to more uniform (diffuse) transport plans.
        Only used when approx=True.

    Returns
    -------
    distance : float
        Square root of the optimal transport cost (Wasserstein-like distance).
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

    Matrices are normalized to unit Frobenius norm before comparison.
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

    m1 = modify_matrix(mtx1, whiten=False, normalize=True)
    m2 = modify_matrix(mtx2, whiten=False, normalize=True)

    # Compute cost matrix
    cost_matrix = cdist(m1, m2, metric=metric)

    # Uniform distributions
    a = np.ones(m1.shape[0]) / m1.shape[0]
    b = np.ones(m2.shape[0]) / m2.shape[0]

    # Compute optimal transport plan
    if approx:
        transport_plan = ot.sinkhorn(a, b, cost_matrix, reg)
        distance = np.sqrt(np.sum(transport_plan * cost_matrix))
    else:
        transport_plan = ot.emd(a, b, cost_matrix)
        distance = np.sqrt(np.sum(transport_plan * cost_matrix))

    threshold = 1e-9
    i_idx, j_idx = np.where(transport_plan > threshold)
    pairs = {
        (int(i), int(j)): float(transport_plan[i, j])
        for i, j in zip(i_idx, j_idx, strict=False)
    }
    return float(distance), pairs


def shape_distance(
    mtx1: npt.NDArray[np.float64],
    mtx2: npt.NDArray[np.float64],
    method: Literal["procrustes", "one-to-one", "soft-matching"] = "procrustes",
    metric: str = "euclidean",
    return_pairs: bool = False,
    **method_kwargs: Any,  # Accept Any for now, validated at runtime
) -> float | tuple[float, dict[tuple[int, int], float]]:
    """Compute shape distance between two matrices.

    Unified interface for multiple shape comparison methods. Delegates to
    specific method functions.

    Parameters
    ----------
    mtx1 : ndarray of shape (n_samples, n_features)
        First matrix representing neural population activity.
    mtx2 : ndarray of shape (n_samples, n_features)
        Second matrix to compare with mtx1.
    method : {'procrustes', 'one-to-one', 'soft-matching'}, default='procrustes'
        Shape comparison method:
        - 'procrustes': Optimal orthogonal alignment (rotation/reflection).
            Preserves point correspondence, best for aligned data.
        - 'one-to-one': Optimal hard assignment (Hungarian algorithm).
            Permutation-invariant, finds best bijective matching.
        - 'soft-matching': Optimal transport with soft assignment.
            Allows fractional matching, handles different point cloud sizes.
    metric : str, default='sqeuclidean'
        Distance metric for 'one-to-one' and 'soft-matching' methods.
        Ignored for 'procrustes' method.
    **method_kwargs
        Additional keyword arguments passed to the specific method:
        - For 'soft-matching': approx (bool), reg (float)

    return_pairs : bool, default=False
        If True, return both distance and pair information.
        If False, return only the distance value.

    Returns
    -------
    distance : float
        Shape distance between the two matrices. Lower values indicate
        more similar shapes. Scale depends on the method used.
        Only returned if return_pairs=False.
    (distance, pairs) : tuple[float, dict]
        If return_pairs=True, returns both distance and point correspondence:
        - For 'procrustes': {(i, i): distance} - aligned point distances
        - For 'one-to-one': {(i, j): distance} - optimal matching pairs
        - For 'soft-matching': {(i, j): probability} - transport probabilities

    Raises
    ------
    ValueError
        If an unknown method is specified.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.RandomState(42)
    >>> mtx1 = rng.randn(50, 10)
    >>> mtx2 = rng.randn(50, 10)

    >>> # Procrustes alignment
    >>> dist, pairs = shape_distance(mtx1, mtx2, method='procrustes')
    >>> print(f"Procrustes distance: {dist:.3f}")

    >>> # Optimal matching
    >>> dist, pairs = shape_distance(mtx1, mtx2, method='one-to-one',
    ...                               metric='euclidean')

    >>> # Soft optimal transport (can handle different sizes)
    >>> mtx3 = rng.randn(60, 10)
    >>> dist, pairs = shape_distance(mtx1, mtx3, method='soft-matching',
    ...                               approx=True, reg=0.05)
    """
    match method:
        case "procrustes":
            dist, pairs = shape_distance_procrustes(mtx1, mtx2)
        case "one-to-one":
            dist, pairs = shape_distance_one_to_one(mtx1, mtx2, metric=metric)
        case "soft-matching":
            dist, pairs = shape_distance_soft_matching(
                mtx1, mtx2, metric=metric, **method_kwargs
            )
        case _:
            raise ValueError(
                f"Unknown method '{method}'. "
                "Choose 'procrustes', 'one-to-one', or 'soft-matching'."
            )

    if return_pairs:
        return dist, pairs
    return dist


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
