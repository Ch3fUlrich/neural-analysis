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

import logging
from typing import TYPE_CHECKING, Any, Literal, TypedDict

import numpy as np
import numpy.typing as npt
from scipy.linalg import orthogonal_procrustes
from scipy.optimize import linear_sum_assignment
from scipy.spatial import procrustes
from scipy.spatial.distance import cdist

if TYPE_CHECKING:
    from pathlib import Path

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
]


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

    distances = [wasserstein_distance(p1[:, i], p2[:, i]) for i in range(p1.shape[1])]
    return float(np.sum(distances))


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
    return float(np.max(ks_stats))


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

    # Extract pairs after alignment - compute distances between aligned points
    # m1 is standardized mtx1, m2 is mtx2 transformed to best match m1
    distances = np.linalg.norm(m1 - m2, axis=1)

    # Create pairs dictionary with point indices and their post-alignment distances
    pairs = {(i, i): float(distances[i]) for i in range(len(distances))}

    return float(disparity), pairs


def shape_distance_one_to_one(
    mtx1: npt.NDArray[np.float64],
    mtx2: npt.NDArray[np.float64],
    metric: str = "sqeuclidean",
) -> tuple[float, dict[tuple[int, int], float]]:
    """Compute shape distance using optimal one-to-one point matching after
    Procrustes alignment.

    First applies Procrustes transformation (centering, scaling, and optimal
    rotation), then uses the Hungarian algorithm to find the optimal permutation
    of points that minimizes total distance. This ensures the mathematical
    property that one-to-one distance ≤ Procrustes distance, since we search
    over a larger space:
    {all permutations} × {optimal rotation} ⊇ {identity permutation} ×
    {optimal rotation}.

    Parameters
    ----------
    mtx1 : ndarray of shape (n_samples, n_features)
        First matrix representing neural population activity.
    mtx2 : ndarray of shape (n_samples, n_features)
        Second matrix to compare with mtx1. Must have same number of samples.
    metric : str, default='sqeuclidean'
        Distance metric for computing pairwise costs after rotation. Passed to
        scipy.spatial.distance.cdist. Common options:
        - 'sqeuclidean': Squared Euclidean distance (default, comparable to Procrustes)
        - 'euclidean': Euclidean distance
        - 'cosine': Cosine distance
        - 'correlation': Correlation distance

    Returns
    -------
    distance : float
        Sum of assigned distances under optimal matching after Procrustes rotation.
        When metric='sqeuclidean', this is the sum of squared distances, directly
        comparable to Procrustes disparity. Lower values indicate more similar shapes.
    pairs : dict[tuple[int, int], float]
        Dictionary mapping optimal point assignments (i, j) -> distance,
        where point i from standardized mtx1 is matched to point j from rotated mtx2.
        Each i and j appears exactly once (bijective matching).

    Notes
    -----
    Unlike pure Procrustes (which fixes point correspondence as i→i) or pure optimal
    matching (which allows permutation but no rotation), this method combines both:
    1. Applies Procrustes standardization (center, scale, rotate)
    2. Finds optimal permutation on the already-aligned matrices

    This guarantees: one-to-one distance ≤ Procrustes distance

    The Hungarian algorithm solves the linear sum assignment problem in O(n³) time.

    Examples
    --------
    >>> mtx1 = np.random.randn(50, 10)
    >>> mtx2 = np.random.randn(50, 10)
    >>> dist, pairs = shape_distance_one_to_one(mtx1, mtx2)
    >>> dist_proc, _ = shape_distance_procrustes(mtx1, mtx2)
    >>> assert dist <= dist_proc  # Property always holds
    >>> print(f"Optimal matching distance: {dist:.3f}")
    >>> print(f"First 5 matches: {list(pairs.items())[:5]}")
    """
    # Apply Procrustes preprocessing and rotation to make one-to-one ≤ procrustes
    # This ensures we're searching over a superset: {all perms} × {opt rotation}
    # vs Procrustes: {identity perm} × {opt rotation}
    m1_std, m2_rotated, _ = procrustes(mtx1, mtx2)

    # Now find optimal permutation on the already-rotated matrices
    # m1_std is standardized mtx1, m2_rotated is mtx2 after optimal rotation
    cost_matrix = cdist(m1_std, m2_rotated, metric=metric)

    # Find optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Get individual distances for each assignment
    assigned_distances = cost_matrix[row_ind, col_ind]

    # Sum of assigned distances
    # NOTE: Since we use sqeuclidean metric by default, this is already
    # sum of squared distances, directly comparable to Procrustes disparity.
    # Do NOT take square root to maintain mathematical property:
    # one-to-one distance ≤ Procrustes distance
    total_distance = assigned_distances.sum()

    # Create pairs dictionary with individual distances
    pairs = {
        (int(id1), int(id2)): float(dist)
        for id1, id2, dist in zip(row_ind, col_ind, assigned_distances)
    }
    return float(total_distance), pairs


def shape_distance_soft_matching(
    mtx1: npt.NDArray[np.float64],
    mtx2: npt.NDArray[np.float64],
    metric: str = "sqeuclidean",
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
        # For approximate method, still need the transport plan for pairs
        transport_plan = ot.sinkhorn(a, b, cost_matrix, reg)
        distance = np.sqrt(np.sum(transport_plan * cost_matrix))
    else:
        # Exact optimal transport
        transport_plan = ot.emd(a, b, cost_matrix)
        distance = np.sqrt(np.sum(transport_plan * cost_matrix))

    # Extract pairs with non-zero transport probability
    i_indices, j_indices = np.where(transport_plan > 0)
    pairs = {
        (int(i), int(j)): float(transport_plan[i, j])
        for i, j in zip(i_indices, j_indices)
    }

    return float(distance), pairs


def shape_distance(
    mtx1: npt.NDArray[np.float64],
    mtx2: npt.NDArray[np.float64],
    method: Literal["procrustes", "one-to-one", "soft-matching"] = "procrustes",
    metric: str = "sqeuclidean",
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
