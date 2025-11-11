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
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypedDict

import numpy as np
import numpy.typing as npt
from scipy.linalg import orthogonal_procrustes  # type: ignore[import-untyped]
from scipy.optimize import linear_sum_assignment  # type: ignore[import-untyped]
from scipy.spatial import procrustes  # type: ignore[import-untyped]
from scipy.spatial.distance import cdist  # type: ignore[import-untyped]
from tqdm.auto import tqdm

if TYPE_CHECKING:
    import pandas as pd
else:
    import pandas as pd  # noqa: PGH003

# I/O functions imported locally where needed to avoid circular dependencies

from .distance import pairwise_distance

try:
    import ot  # type: ignore[import-untyped]  # Python Optimal Transport

    OT_AVAILABLE = True
except ImportError:
    OT_AVAILABLE = False

try:
    from neural_analysis.utils.logging import get_logger, log_calls
except ImportError:

    def log_calls(**kwargs):  # type: ignore[no-untyped-def,no-redef]
        def decorator(func):  # type: ignore[no-untyped-def]
            return func

        return decorator

    def get_logger(name: str) -> logging.Logger:  # type: ignore[no-redef]
        return logging.getLogger(name)


# Module logger
logger = get_logger(__name__)

__all__ = [
    "compare_distributions",
    "compare_distribution_groups",
    "wasserstein_distance_multi",
    "kolmogorov_smirnov_distance",
    "jensen_shannon_divergence",
    "distribution_distance",
    "pairwise_distribution_comparison_batch",
    "shape_distance",
    "save_distribution_comparison",
    "load_distribution_comparisons",
    "get_comparison_summary",
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

    Examples
    --------
    >>> p1 = np.random.randn(100, 3)
    >>> p2 = np.random.randn(100, 3) + 1.0
    >>> dist = wasserstein_distance_multi(p1, p2)
    """
    from scipy.stats import wasserstein_distance  # type: ignore[import-untyped]

    p1 = np.asarray(points1)
    p2 = np.asarray(points2)

    if p1.ndim == 1:
        p1 = p1.reshape(-1, 1)
    if p2.ndim == 1:
        p2 = p2.reshape(-1, 1)

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

    # Determine common bin edges
    all_data = np.vstack([p1, p2])
    ranges = [
        (all_data[:, i].min(), all_data[:, i].max()) for i in range(all_data.shape[1])
    ]

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
    # Define shape metrics
    shape_metrics = {"procrustes", "one-to-one", "soft-matching"}

    # Validate metric/mode combination
    if mode == "within" and metric in shape_metrics:
        raise ValueError(
            f"Shape metric '{metric}' cannot be used with mode='within'. "
            f"Shape metrics only work in 'between' mode."
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
                dist, pairs = shape_distance(
                    points1_arr.astype(np.float64),
                    points2_arr.astype(np.float64),
                    method=method_typed,
                    **metric_kwargs,
                )
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

    This is a convenience function that wraps distribution_distance with
    mode="between" and returns a single distance value. For more control
    (e.g., getting multiple statistics), use distribution_distance directly.

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
    
    # Delegate to distribution_distance for the actual computation
    # This reduces code duplication
    result = distribution_distance(
        points1,
        points2,
        mode="between",
        metric=metric,
        parallel=True,
        summary="mean",
        **metric_kwargs,
    )

    # distribution_distance returns float for distribution metrics
    # and tuple for shape metrics
    shape_metrics = {"procrustes", "one-to-one", "soft-matching"}
    if metric in shape_metrics:
        # Result is tuple[float, dict[tuple[int, int], float]]
        # Mypy can't narrow union types based on runtime checks,
        # so we assert the type here
        assert isinstance(result, tuple), f"Expected tuple for shape metric {metric}"
        value, pairs_dict = result
    else:
        # For distribution metrics, result should be float
        assert not isinstance(result, tuple), f"Expected float for metric {metric}"
        value = float(result)
        pairs_dict = None
    
    # Save if path provided
    if save_path is not None:
        # Convert points to arrays for metadata
        p1 = np.asarray(points1)
        p2 = np.asarray(points2)
        
        # Build metadata
        metadata = {
            "n_samples_i": int(p1.shape[0]),
            "n_samples_j": int(p2.shape[0]),
        }
        if p1.ndim > 1:
            metadata["n_features_i"] = int(p1.shape[1])
        if p2.ndim > 1:
            metadata["n_features_j"] = int(p2.shape[1])
        
        # Save comparison
        save_distribution_comparison(
            save_path=save_path,
            comparison_name=comparison_name,
            dataset_i=dataset_i,  # type: ignore[arg-type]
            dataset_j=dataset_j,  # type: ignore[arg-type]
            metric=metric,
            value=value,
            pairs=pairs_dict,
            metadata=metadata,
        )
    
    # Return result in original format
    if metric in shape_metrics:
        return (value, pairs_dict)  # type: ignore[return-value]
    else:
        return value


def compare_distribution_groups(
    group_vectors: dict[str | tuple[str, ...], npt.NDArray[np.floating]],
    compare_type: Literal["inside", "between"] = "between",
    metric: str = "wasserstein",
    **metric_kwargs: Any,  # Accept Any for flexible kwargs
) -> (
    dict[str, npt.NDArray[np.floating]]
    | dict[str | tuple[str, ...], npt.NDArray[np.floating]]
):
    """Compare distributions within or between groups.

    Parameters
    ----------
    group_vectors : dict
        Dictionary mapping group identifiers to point arrays (n_samples, n_features).
    compare_type : {"inside", "between"}, default="between"
        - "inside": Compare each group to itself (self-similarity).
            Only works with distribution metrics, not shape metrics.
        - "between": Compare each group to all others.
    metric : str, default="wasserstein"
        Distance metric to use. Supported metrics:

        **Distribution metrics** (work with both "inside" and "between"):
        - "wasserstein": Wasserstein distance (Earth Mover's Distance)
        - "kolmogorov_smirnov" or "ks": Kolmogorov-Smirnov statistic
        - "jensen_shannon" or "js": Jensen-Shannon divergence
        - "euclidean": Euclidean distance between distributions
        - "mahalanobis": Mahalanobis distance
        - "cosine": Cosine distance

        **Shape metrics** (only work with "between" mode):
        - "procrustes": Procrustes distance after optimal alignment
        - "one-to-one": Bipartite matching distance
        - "soft-matching": Soft assignment via optimal transport
    **metric_kwargs : dict, optional
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
        If a shape metric is used with compare_type="inside".

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

    >>> # Within-group variability (only distribution metrics)
    >>> within_stats = compare_distribution_groups(
    ...     groups, compare_type="inside", metric="wasserstein"
    ... )
    >>> within_stats["mean"]  # mean within-group distances
    """
    # Define shape metrics
    shape_metrics = {"procrustes", "one-to-one", "soft-matching"}

    # Validate metric/mode combination
    if compare_type == "inside" and metric in shape_metrics:
        raise ValueError(
            f"Shape metric '{metric}' cannot be used with compare_type='inside'. "
            f"Shape metrics only work in 'between' mode. "
            f"Use a distribution metric instead: "
            f"wasserstein, kolmogorov_smirnov, jensen_shannon, etc."
        )
    n_groups = len(group_vectors)

    logger.info(
        f"Comparing {n_groups} groups with "
        f"compare_type='{compare_type}', metric='{metric}'"
    )

    # Use match/case for cleaner dispatch
    match compare_type:
        case "inside":
            # Within-group variability using new distance.py functions
            means = np.zeros(n_groups)
            stds = np.zeros(n_groups)

            for idx, (name, points) in enumerate(group_vectors.items()):
                if len(points) < 2:
                    logger.debug(f"Group '{name}' has <2 points, skipping")
                    means[idx] = 0.0
                    stds[idx] = 0.0
                    continue

                # Use unified distribution_distance from distance.py
                stats = distribution_distance(
                    points,
                    mode="within",
                    metric=metric,
                    summary="all",
                )
                means[idx] = stats["mean"]
                stds[idx] = stats["std"]

            logger.info(f"Within-group statistics computed: mean={means}, std={stds}")
            return {"mean": means, "std": stds}  # type: ignore[return-value]

        case "between":
            # Between-group distances
            similarities = {}
            for _i, (name_i, group_i) in enumerate(group_vectors.items()):
                dists_to_all = np.zeros(n_groups)
                for j, (_name_j, group_j) in enumerate(group_vectors.items()):
                    # Use distribution_distance which now handles both distribution
                    # and shape metrics
                    result = distribution_distance(
                        group_i,
                        group_j,
                        mode="between",
                        metric=metric,
                        summary="mean",
                        **metric_kwargs,
                    )

                    # Extract distance value (handle both float and tuple returns)
                    # Shape metrics: (distance, pairs), distribution: float
                    dist = result[0] if isinstance(result, tuple) else float(result)

                    dists_to_all[j] = dist
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
    """Compute shape distance using optimal one-to-one point matching.

    Uses the Hungarian algorithm to find the optimal assignment of points
    between the two matrices that minimizes total distance. Unlike Procrustes,
    this method does not assume point correspondence and finds the best
    permutation-invariant matching.

    Parameters
    ----------
    mtx1 : ndarray of shape (n_samples, n_features)
        First matrix representing neural population activity.
    mtx2 : ndarray of shape (n_samples, n_features)
        Second matrix to compare with mtx1. Must have same number of samples.
    metric : str, default='sqeuclidean'
        Distance metric for computing pairwise costs. Passed to
        scipy.spatial.distance.cdist. Common options:
        - 'sqeuclidean': Squared Euclidean distance
        - 'euclidean': Euclidean distance
        - 'cosine': Cosine distance
        - 'correlation': Correlation distance

    Returns
    -------
    distance : float
        Square root of the sum of assigned distances under optimal matching.
        Lower values indicate more similar point cloud shapes.
    pairs : dict[tuple[int, int], float]
        Dictionary mapping optimal point assignments (i, j) -> distance,
        where point i from mtx1 is matched to point j from mtx2.
        Each i and j appears exactly once (bijective matching).

    Notes
    -----
    The Hungarian algorithm solves the linear sum assignment problem,
    finding the best bijection between point sets. This is appropriate
    when point correspondence is unknown and you want to find the optimal
    permutation-invariant alignment.

    Matrices are normalized to unit Frobenius norm before comparison to
    ensure scale-invariant shape comparison.

    Examples
    --------
    >>> mtx1 = np.random.randn(50, 10)
    >>> mtx2 = np.random.randn(50, 10)
    >>> dist, pairs = shape_distance_one_to_one(mtx1, mtx2, metric='euclidean')
    >>> print(f"Optimal matching distance: {dist:.3f}")
    >>> print(f"First 5 matches: {list(pairs.items())[:5]}")
    """
    m1 = modify_matrix(mtx1, whiten=False, normalize=True)
    m2 = modify_matrix(mtx2, whiten=False, normalize=True)

    # Compute pairwise distances
    cost_matrix = cdist(m1, m2, metric=metric)

    # Find optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Get individual distances for each assignment
    assigned_distances = cost_matrix[row_ind, col_ind]

    # Sum of assigned distances
    total_distance = assigned_distances.sum()
    total_distance = np.sqrt(total_distance)

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

    pairs = {
        (int(i), int(j)): float(matching_prob)
        for i, j, matching_prob in zip(
            *np.where(transport_plan),
            transport_plan,
        )
    }

    return float(distance), pairs


def shape_distance(
    mtx1: npt.NDArray[np.float64],
    mtx2: npt.NDArray[np.float64],
    method: Literal["procrustes", "one-to-one", "soft-matching"] = "procrustes",
    metric: str = "euclidean",
    **method_kwargs: Any,  # Accept Any for now, validated at runtime
) -> tuple[float, dict[tuple[int, int], float]]:
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
    metric : str, default='euclidean'
        Distance metric for 'one-to-one' and 'soft-matching' methods.
        Ignored for 'procrustes' method.
    **method_kwargs
        Additional keyword arguments passed to the specific method:
        - For 'soft-matching': approx (bool), reg (float)

    Returns
    -------
    distance : float
        Shape distance between the two matrices. Lower values indicate
        more similar shapes. Scale depends on the method used.
    pairs : dict[tuple[int, int], float]
        Point correspondence information:
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
            return shape_distance_procrustes(mtx1, mtx2)
        case "one-to-one":
            return shape_distance_one_to_one(mtx1, mtx2, metric=metric)
        case "soft-matching":
            return shape_distance_soft_matching(
                mtx1, mtx2, metric=metric, **method_kwargs
            )
        case _:
            raise ValueError(
                f"Unknown method '{method}'. "
                "Choose 'procrustes', 'one-to-one', or 'soft-matching'."
            )


def pairwise_distribution_comparison_batch(
    data: dict[str, npt.NDArray[np.floating]],
    metrics: list[str] | dict[str, dict[str, object]],
    comparison_name: str = "default",
    save_path: str | Path | None = None,
    regenerate: bool = False,
    progress_desc: str = "Computing pairwise comparisons",
    batch_size: int = 1000,
) -> pd.DataFrame:
    """Compute pairwise distribution comparisons for multiple metrics.

    This function provides a general framework for batch computation of
    distribution comparisons using any metric. Results are stored in HDF5
    format with incremental computation support.

    Parameters
    ----------
    data : dict of str to ndarray
        Dictionary mapping dataset names to data arrays.
        Arrays can be 1D (samples) or 2D (samples × features).
    metrics : list of str or dict of str to dict
        Either a list of metric names or a dict mapping metric names to
        their keyword arguments. Supported metrics:
        - 'wasserstein', 'ks', 'js', 'euclidean', 'mahalanobis', 'cosine'
        - Shape distances: 'procrustes', 'one-to-one', 'soft-matching'
    comparison_name : str, default='default'
        Name for this comparison group (e.g., "experiment_001", "session_A")
    save_path : str or Path, optional
        Path to HDF5 file for caching results. If None, defaults to
        './output/distribution_comparisons.h5'
    regenerate : bool, default=False
        If True, recompute all pairs even if cached.
    progress_desc : str, default='Computing pairwise comparisons'
        Description for progress bar.

    Returns
    -------
    DataFrame
        Long-format DataFrame with columns:
        - dataset_i: first dataset name
        - dataset_j: second dataset name
        - metric: name of the distance/similarity metric
        - value: computed distance/similarity value
        - n_samples_i: number of samples in dataset_i
        - n_samples_j: number of samples in dataset_j
        - n_features_i: number of features in dataset_i (NaN for 1D)
        - n_features_j: number of features in dataset_j (NaN for 1D)
        - pairs: dict or None, point-to-point correspondence for shape metrics

    Notes
    -----
    For shape metrics (procrustes, one-to-one, soft-matching), the 'pairs' column
    contains a dictionary mapping point pairs to their distances/probabilities.
    Keys are formatted as "i,j" strings (for JSON compatibility), values are floats.
    For distribution metrics, the 'pairs' column is None.

    Example of accessing pairs data:
        >>> df = pairwise_distribution_comparison_batch(data, ['procrustes'])
        >>> pairs_dict = df.loc[0, 'pairs']  # Get pairs for first row
        >>> # Parse pair indices: "0,0" -> (0, 0)
        >>> parsed_pairs = {tuple(map(int, k.split(','))): v
        ...                 for k, v in pairs_dict.items()}

    Examples
    --------
    >>> # Shape distance comparisons
    >>> data = {
    ...     "session_001": np.random.rand(50, 20),
    ...     "session_002": np.random.rand(60, 20),
    ... }
    >>> metrics = {
    ...     'procrustes': {},
    ...     'one-to-one': {'metric': 'euclidean'},
    ...     'soft-matching': {'reg': 0.05}
    ... }
    >>> df = pairwise_distribution_comparison_batch(
    ...     data, metrics, save_path='comparisons.h5'
    ... )

    >>> # Distribution comparisons
    >>> data_1d = {
    ...     "condition_A": np.random.randn(1000),
    ...     "condition_B": np.random.randn(1000),
    ... }
    >>> metrics = ['wasserstein', 'ks', 'js']
    >>> df = pairwise_distribution_comparison_batch(data_1d, metrics)
    """
    # Set default save path if not provided
    if save_path is None:
        save_path = Path("./output/distribution_comparisons.h5")
    else:
        save_path = Path(save_path).with_suffix(".h5")
    
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Normalize metrics to dict format
    metrics_dict = {m: {} for m in metrics} if isinstance(metrics, list) else metrics

    # Generate all dataset pairs
    labels = list(data.keys())
    item_pairs = [
        (l1, l2) for i, l1 in enumerate(labels) for j, l2 in enumerate(labels) if i < j
    ]

    # Load existing results if available
    existing_results = {}
    if save_path.exists() and not regenerate:
        try:
            loaded_data = load_distribution_comparisons(
                save_path=save_path,
                comparison_name=comparison_name,
            )
            if comparison_name in loaded_data:
                existing_results = loaded_data[comparison_name]
        except Exception as e:
            logger.warning(f"Could not load existing results: {e}")
    
    # Determine which computations are missing
    missing_computations = []
    for name_i, name_j in item_pairs:
        for metric_name in metrics_dict:
            result_key = f"{name_i}_vs_{name_j}_{metric_name}"
            if regenerate or result_key not in existing_results:
                missing_computations.append((name_i, name_j, metric_name))

    # Early exit if all cached
    total_comparisons = len(item_pairs) * len(metrics_dict)
    if not missing_computations:
        logger.info(f"All {total_comparisons} computations cached.")
        # Convert existing results to DataFrame
        return _comparison_results_to_dataframe(existing_results)
    elif existing_results:
        n_cached = len(existing_results)
        logger.info(
            f"Found {n_cached} cached. Computing {len(missing_computations)} missing."
        )
    else:
        logger.info(f"Computing all {len(missing_computations)} pairs × metrics.")

    # Compute missing comparisons
    for idx, (name_i, name_j, metric_name) in enumerate(
        tqdm(missing_computations, desc=progress_desc), start=1
    ):
        data_i = data[name_i]
        data_j = data[name_j]
        metric_kwargs = metrics_dict[metric_name]

        # Use compare_distributions which now handles both types
        try:
            result = compare_distributions(
                data_i, data_j, metric=metric_name, **metric_kwargs
            )

            # Handle both distribution and shape metric returns
            pairs_dict = None
            if isinstance(result, tuple):
                # Shape metrics return (distance, pairs)
                value, pairs = result
                pairs_dict = pairs  # Keep as dict with tuple keys
            else:
                # Distribution metrics return float
                value = float(result)

        except Exception as e:
            logger.warning(f"Error for {name_i} vs {name_j} ({metric_name}): {e}")
            value = np.nan
            pairs_dict = None

        # Record metadata
        n_features_i = int(data_i.shape[1]) if data_i.ndim > 1 else None
        n_features_j = int(data_j.shape[1]) if data_j.ndim > 1 else None
        
        metadata = {
            "n_samples_i": int(data_i.shape[0]),
            "n_samples_j": int(data_j.shape[0]),
        }
        if n_features_i is not None:
            metadata["n_features_i"] = n_features_i
        if n_features_j is not None:
            metadata["n_features_j"] = n_features_j

        # Save both directions (symmetric)
        save_distribution_comparison(
            save_path=save_path,
            comparison_name=comparison_name,
            dataset_i=name_i,
            dataset_j=name_j,
            metric=metric_name,
            value=value,
            pairs=pairs_dict,
            metadata=metadata,
        )
        
        # Also save reverse direction
        save_distribution_comparison(
            save_path=save_path,
            comparison_name=comparison_name,
            dataset_i=name_j,
            dataset_j=name_i,
            metric=metric_name,
            value=value,
            pairs=pairs_dict,
            metadata={
                "n_samples_i": int(data_j.shape[0]),
                "n_samples_j": int(data_i.shape[0]),
                "n_features_i": n_features_j,
                "n_features_j": n_features_i,
            },
        )
        
        # Log progress periodically
        if idx % batch_size == 0:
            n_total = len(missing_computations)
            logger.info(f"Computed {idx}/{n_total} comparisons")

    # Load all results and convert to DataFrame
    all_results = load_distribution_comparisons(
        save_path=save_path,
        comparison_name=comparison_name,
    )
    
    if comparison_name not in all_results or not all_results[comparison_name]:
        raise ValueError("No comparison data computed or loaded.")
    
    # Convert to DataFrame and remove duplicates
    df_final = _comparison_results_to_dataframe(all_results[comparison_name])
    df_final = df_final.drop_duplicates(
        subset=["dataset_i", "dataset_j", "metric"], keep="first"
    )
    df_final = df_final.sort_values(["metric", "dataset_i", "dataset_j"])

    return df_final


def _comparison_results_to_dataframe(
    results: dict[str, dict[str, Any]]
) -> pd.DataFrame:
    """Convert comparison results from HDF5 format to DataFrame.
    
    Parameters
    ----------
    results : dict
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


# ============================================================================
# Distribution Comparison I/O Functions
# ============================================================================


def save_distribution_comparison(
    save_path: str | Path,
    comparison_name: str,
    dataset_i: str,
    dataset_j: str,
    metric: str,
    value: float,
    pairs: dict[tuple[int, int], float] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save a single distribution comparison result to HDF5.

    Uses hierarchical structure: comparison_name / result_key / {scalars, arrays}
    where result_key is formatted as "{dataset_i}_vs_{dataset_j}_{metric}"

    Parameters
    ----------
    save_path : str or Path
        Path to HDF5 file
    comparison_name : str
        Top-level comparison group name (e.g., "session_comparisons", "experiment_001")
    dataset_i : str
        Name of first dataset
    dataset_j : str
        Name of second dataset
    metric : str
        Distance/similarity metric used
    value : float
        Computed comparison value
    pairs : dict, optional
        For shape metrics: point-to-point correspondence mapping
    metadata : dict, optional
        Additional metadata to store (n_samples, n_features, etc.)

    Examples
    --------
    >>> save_distribution_comparison(
    ...     "output/comparisons.h5",
    ...     comparison_name="session_001",
    ...     dataset_i="condition_A",
    ...     dataset_j="condition_B",
    ...     metric="wasserstein",
    ...     value=0.523,
    ...     metadata={"n_samples_i": 100, "n_samples_j": 120}
    ... )
    """
    from neural_analysis.utils.io import save_result_to_hdf5_dataset

    # Create unique result key
    result_key = f"{dataset_i}_vs_{dataset_j}_{metric}"

    # Prepare scalar data
    scalar_data = {
        "dataset_i": dataset_i,
        "dataset_j": dataset_j,
        "metric": metric,
        "value": float(value),
    }

    # Add metadata if provided
    if metadata is not None:
        scalar_data.update(metadata)

    # Prepare array data
    array_data = {}
    if pairs is not None:
        # Convert pairs dict to arrays for HDF5 storage
        pair_indices = np.array(list(pairs.keys()), dtype=np.int64)
        pair_values = np.array(list(pairs.values()), dtype=np.float64)
        array_data["pair_indices"] = pair_indices
        array_data["pair_values"] = pair_values

    # Use generalized saving function
    save_result_to_hdf5_dataset(
        save_path=save_path,
        dataset_name=comparison_name,
        result_key=result_key,
        scalar_data=scalar_data,
        array_data=array_data,
    )


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
    from neural_analysis.utils.io import load_results_from_hdf5_dataset

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


def get_comparison_summary(
    save_path: str | Path,
    comparison_name: str | None = None,
) -> pd.DataFrame:
    """Get summary DataFrame of all comparison results.

    Parameters
    ----------
    save_path : str or Path
        Path to HDF5 file
    comparison_name : str, optional
        Filter by comparison group name

    Returns
    -------
    DataFrame
        Summary with columns: comparison_name, result_key, dataset_i, dataset_j,
        metric, value, and any additional metadata

    Examples
    --------
    >>> df = get_comparison_summary("output/comparisons.h5")
    >>> print(df[['dataset_i', 'dataset_j', 'metric', 'value']])
    """
    from neural_analysis.utils.io import get_hdf5_result_summary

    return get_hdf5_result_summary(
        save_path=save_path,
        dataset_name=comparison_name,
    )
