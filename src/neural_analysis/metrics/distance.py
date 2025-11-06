"""Distance metrics for neural data analysis.

This module provides comprehensive distance and similarity measures with support
for pairwise calculations, both within and between distributions. All functions
leverage optimized implementations from scipy, numpy, and scikit-learn where
available, with optional numba acceleration for large-scale computations.

The module supports a plugin-based system for computing pairwise distances using
different metrics through a unified interface.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import cdist, cosine
from scipy.stats import entropy, ks_2samp, wasserstein_distance

if TYPE_CHECKING:
    from collections.abc import Callable

try:
    from neural_analysis.utils.logging import get_logger, log_calls  # type: ignore
except ImportError:
    def log_calls(**kwargs):  # type: ignore
        def decorator(func):  # type: ignore
            return func
        return decorator
    def get_logger(name: str):  # type: ignore
        return logging.getLogger(name)

# Module logger
logger = get_logger(__name__)

# Optional acceleration with numba for large pairwise computations
try:  # pragma: no cover - optional dependency
    from numba import njit, prange  # type: ignore

    NUMBA_AVAILABLE = True

    @njit(parallel=True, fastmath=True)
    def _pairwise_euclidean_numba(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Numba-accelerated pairwise Euclidean distance."""
        n_x, n_feat = X.shape
        n_y, _ = Y.shape
        out = np.empty((n_x, n_y), dtype=np.float64)
        for i in prange(n_x):
            for j in range(n_y):
                s = 0.0
                for k in range(n_feat):
                    d = X[i, k] - Y[j, k]
                    s += d * d
                out[i, j] = np.sqrt(s)
        return out

    @njit(parallel=True, fastmath=True)
    def _pairwise_cosine_numba(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Numba-accelerated pairwise cosine similarity."""
        n_x, n_feat = X.shape
        n_y, _ = Y.shape
        out = np.empty((n_x, n_y), dtype=np.float64)
        for i in prange(n_x):
            norm_x = 0.0
            for k in range(n_feat):
                norm_x += X[i, k] ** 2
            norm_x = np.sqrt(norm_x)
            
            for j in range(n_y):
                dot_prod = 0.0
                norm_y = 0.0
                for k in range(n_feat):
                    dot_prod += X[i, k] * Y[j, k]
                    norm_y += Y[j, k] ** 2
                norm_y = np.sqrt(norm_y)
                
                if norm_x > 0 and norm_y > 0:
                    out[i, j] = dot_prod / (norm_x * norm_y)
                else:
                    out[i, j] = 0.0
        return out

    @njit(parallel=True, fastmath=True)
    def _pairwise_manhattan_numba(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Numba-accelerated pairwise Manhattan distance."""
        n_x, n_feat = X.shape
        n_y, _ = Y.shape
        out = np.empty((n_x, n_y), dtype=np.float64)
        for i in prange(n_x):
            for j in range(n_y):
                s = 0.0
                for k in range(n_feat):
                    s += abs(X[i, k] - Y[j, k])
                out[i, j] = s
        return out

except Exception:  # pragma: no cover - optional dependency
    NUMBA_AVAILABLE = False

__all__ = [
    "euclidean_distance",
    "manhattan_distance",
    "mahalanobis_distance",
    "cosine_similarity",
    "wasserstein_distance_multi",
    "kolmogorov_smirnov_distance",
    "jensen_shannon_divergence",
    "pairwise_distance",
    "distribution_distance",
]


# =============================================================================
# Helper Functions
# =============================================================================

def _compute_summary_statistics(
    dists: np.ndarray,
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


# =============================================================================
# Point-to-Point Distance Metrics
# =============================================================================

@log_calls(level=logging.DEBUG)
def euclidean_distance(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    axis: int | None = None,
    parallel: bool = True,
) -> float | np.ndarray:
    """Compute Euclidean distance between vectors or point clouds.

    Parameters
    ----------
    x, y : array-like
        Input arrays. Can be 1D vectors or 2D point clouds (samples × features).
    axis : int, optional
        Axis along which to compute distance for multi-dimensional inputs.
        If None and inputs are 2D, computes pairwise distances.
    parallel : bool, default=True
        Use numba parallel implementation if available (for 2D pairwise).

    Returns
    -------
    float or ndarray
        Euclidean distance(s). Scalar for 1D inputs, array for 2D.

    Examples
    --------
    >>> x = np.array([0, 0])
    >>> y = np.array([3, 4])
    >>> euclidean_distance(x, y)
    5.0

    >>> # Pairwise distances
    >>> X = np.random.randn(100, 10)
    >>> Y = np.random.randn(50, 10)
    >>> dists = euclidean_distance(X, Y)
    >>> dists.shape
    (100, 50)
    """
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)

    # Handle 1D case
    if x_arr.ndim == 1 and y_arr.ndim == 1:
        return float(np.linalg.norm(x_arr - y_arr))

    # Handle 2D case
    if x_arr.ndim == 2 and y_arr.ndim == 2:
        if axis is None:
            # Pairwise distances, return matrix
            if parallel and NUMBA_AVAILABLE:
                return _pairwise_euclidean_numba(
                    x_arr.astype(np.float64), y_arr.astype(np.float64)
                )
            else:
                return cdist(x_arr, y_arr, metric="euclidean")
        else:
            # Distance along specified axis
            return np.linalg.norm(x_arr - y_arr, axis=axis)

    # Fallback for any other shape
    return np.linalg.norm(x_arr - y_arr, axis=axis)


@log_calls(level=logging.DEBUG)
def manhattan_distance(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    axis: int | None = None,
    parallel: bool = True,
) -> float | np.ndarray:
    """Compute Manhattan (L1) distance between vectors or point clouds.

    Parameters
    ----------
    x, y : array-like
        Input arrays. Can be 1D vectors or 2D point clouds (samples × features).
    axis : int, optional
        Axis along which to compute distance for multi-dimensional inputs.
        If None and inputs are 2D, computes pairwise distances.
    parallel : bool, default=True
        Use numba parallel implementation if available (for 2D pairwise).

    Returns
    -------
    float or ndarray
        Manhattan distance(s). Scalar for 1D inputs, array for 2D.

    Examples
    --------
    >>> x = np.array([0, 0])
    >>> y = np.array([3, 4])
    >>> manhattan_distance(x, y)
    7.0
    """
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)

    # Handle 1D case
    if x_arr.ndim == 1 and y_arr.ndim == 1:
        return float(np.sum(np.abs(x_arr - y_arr)))

    # Handle 2D case
    if x_arr.ndim == 2 and y_arr.ndim == 2:
        if axis is None:
            # Pairwise distances
            if parallel and NUMBA_AVAILABLE:
                return _pairwise_manhattan_numba(
                    x_arr.astype(np.float64), y_arr.astype(np.float64)
                )
            else:
                return cdist(x_arr, y_arr, metric="cityblock")
        else:
            # Distance along specified axis
            return np.sum(np.abs(x_arr - y_arr), axis=axis)

    # Fallback
    return np.sum(np.abs(x_arr - y_arr), axis=axis)


@log_calls(level=logging.DEBUG)
def mahalanobis_distance(
    x: npt.ArrayLike,
    mean: npt.ArrayLike,
    cov: npt.ArrayLike | None = None,
    inv_cov: npt.ArrayLike | None = None,
) -> float | np.ndarray:
    """Compute Mahalanobis distance from point(s) to a distribution.

    Parameters
    ----------
    x : array-like
        Point or points (n_samples, n_features) to measure distance from.
    mean : array-like
        Mean of the reference distribution (n_features,).
    cov : array-like, optional
        Covariance matrix (n_features, n_features). If not provided, `inv_cov`
        must be supplied.
    inv_cov : array-like, optional
        Inverse covariance matrix. If not provided, computed from `cov`.

    Returns
    -------
    float or ndarray
        Mahalanobis distance(s). Scalar if x is 1D, array if x is 2D.

    Examples
    --------
    >>> mean = np.array([0, 0])
    >>> cov = np.eye(2)
    >>> x = np.array([1, 1])
    >>> mahalanobis_distance(x, mean, cov)
    1.414...
    """
    x_arr = np.asarray(x)
    mean_arr = np.asarray(mean)

    # Compute inverse covariance if not provided
    if inv_cov is None:
        if cov is None:
            raise ValueError("Either cov or inv_cov must be provided")
        cov_arr = np.asarray(cov)
        inv_cov_arr = np.linalg.inv(cov_arr)
    else:
        inv_cov_arr = np.asarray(inv_cov)

    # Center the data
    diff = x_arr - mean_arr

    # Handle 1D input
    if diff.ndim == 1:
        dist_sq = diff @ inv_cov_arr @ diff
        return float(np.sqrt(dist_sq))

    # Handle 2D input (multiple points)
    dist_sq = np.sum(diff @ inv_cov_arr * diff, axis=1)
    return np.sqrt(dist_sq)


@log_calls(level=logging.DEBUG)
def cosine_similarity(
    v1: npt.ArrayLike,
    v2: npt.ArrayLike,
    pairwise: bool = False,
    parallel: bool = True,
) -> float | np.ndarray:
    """Compute cosine similarity between vectors.

    Parameters
    ----------
    v1, v2 : array-like
        Input vectors. If 1D, computes single similarity. If 2D and pairwise=True,
        computes pairwise similarity matrix.
    pairwise : bool, default=False
        If True and inputs are 2D, compute pairwise similarities.
    parallel : bool, default=True
        Use numba parallel implementation if available (for 2D pairwise).

    Returns
    -------
    float or ndarray
        Cosine similarity in [-1, 1]. 1 = identical direction, -1 = opposite,
        0 = orthogonal.

    Examples
    --------
    >>> v1 = np.array([1, 0, 0])
    >>> v2 = np.array([1, 0, 0])
    >>> cosine_similarity(v1, v2)
    1.0

    >>> v1 = np.array([1, 0])
    >>> v2 = np.array([0, 1])
    >>> cosine_similarity(v1, v2)
    0.0
    """
    v1_arr = np.asarray(v1)
    v2_arr = np.asarray(v2)

    # Handle 1D case
    if v1_arr.ndim == 1:
        v1_arr = v1_arr.flatten()
        v2_arr = v2_arr.flatten()
        # scipy.spatial.distance.cosine returns 1 - similarity
        return float(1.0 - cosine(v1_arr, v2_arr))

    # Handle 2D pairwise case
    if v1_arr.ndim == 2 and v2_arr.ndim == 2 and pairwise:
        if parallel and NUMBA_AVAILABLE:
            return _pairwise_cosine_numba(
                v1_arr.astype(np.float64), v2_arr.astype(np.float64)
            )
        else:
            # Use sklearn's cosine_similarity
            from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
            return sklearn_cosine(v1_arr, v2_arr)

    # Default: flatten and compute single similarity
    return float(1.0 - cosine(v1_arr.flatten(), v2_arr.flatten()))


# =============================================================================
# Distribution-Level Distance Metrics
# =============================================================================

@log_calls(level=logging.DEBUG)
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
    p1 = np.asarray(points1)
    p2 = np.asarray(points2)

    if p1.ndim == 1:
        p1 = p1.reshape(-1, 1)
    if p2.ndim == 1:
        p2 = p2.reshape(-1, 1)

    distances = [
        wasserstein_distance(p1[:, i], p2[:, i]) for i in range(p1.shape[1])
    ]
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
    p1 = np.asarray(points1)
    p2 = np.asarray(points2)

    if p1.ndim == 1:
        p1 = p1.reshape(-1, 1)
    if p2.ndim == 1:
        p2 = p2.reshape(-1, 1)

    # Determine common bin edges
    all_data = np.vstack([p1, p2])
    ranges = [(all_data[:, i].min(), all_data[:, i].max()) for i in range(all_data.shape[1])]

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


# =============================================================================
# Pairwise Distance System (Plugin-based)
# =============================================================================

DistanceMetric = Literal[
    "euclidean",
    "manhattan",
    "mahalanobis",
    "cosine",
    "wasserstein",
    "kolmogorov-smirnov",
    "jensen-shannon",
]


def _get_distance_function(metric: DistanceMetric) -> Callable:
    """Get the distance function for a given metric name.
    
    This is the plugin registry for distance metrics.
    """
    match metric:
        case "euclidean":
            return euclidean_distance
        case "manhattan":
            return manhattan_distance
        case "mahalanobis":
            return mahalanobis_distance
        case "cosine":
            return cosine_similarity
        case "wasserstein":
            return wasserstein_distance_multi
        case "kolmogorov-smirnov":
            return kolmogorov_smirnov_distance
        case "jensen-shannon":
            return jensen_shannon_divergence
        case _:
            raise ValueError(
                f"Unknown metric '{metric}'. Choose from: euclidean, manhattan, "
                "mahalanobis, cosine, wasserstein, kolmogorov-smirnov, jensen-shannon"
            )


@log_calls(level=logging.DEBUG)
def pairwise_distance(
    X: npt.ArrayLike,
    Y: npt.ArrayLike | None = None,
    metric: DistanceMetric = "euclidean",
    parallel: bool = True,
    **metric_kwargs,
) -> np.ndarray:
    """Compute pairwise distances between samples using specified metric.

    Parameters
    ----------
    X : array-like
        First set of samples, shape (n_samples_X, n_features).
    Y : array-like, optional
        Second set of samples, shape (n_samples_Y, n_features).
        If None, computes pairwise distances within X.
    metric : str, default="euclidean"
        Distance metric to use. Options:
        - "euclidean": Euclidean distance
        - "manhattan": Manhattan (L1) distance
        - "cosine": Cosine similarity
        - "mahalanobis": Mahalanobis distance (requires mean/cov in metric_kwargs)
    parallel : bool, default=True
        Use parallel numba implementation if available.
    **metric_kwargs
        Additional keyword arguments for the distance function.

    Returns
    -------
    ndarray
        Pairwise distance matrix, shape (n_samples_X, n_samples_Y) or
        (n_samples_X, n_samples_X) if Y is None.

    Examples
    --------
    >>> X = np.random.randn(100, 10)
    >>> Y = np.random.randn(50, 10)
    >>> dists = pairwise_distance(X, Y, metric="euclidean")
    >>> dists.shape
    (100, 50)
    
    >>> # Within-sample distances
    >>> dists = pairwise_distance(X, metric="cosine")
    >>> dists.shape
    (100, 100)
    """
    X_arr = np.asarray(X)
    Y_arr = np.asarray(Y) if Y is not None else X_arr

    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(1, -1)
    if Y_arr.ndim == 1:
        Y_arr = Y_arr.reshape(1, -1)

    logger.info(
        f"Computing pairwise distances: X.shape={X_arr.shape}, Y.shape={Y_arr.shape}, "
        f"metric='{metric}', parallel={parallel}"
    )

    # For metrics that have native pairwise support - use match/case
    match metric:
        case "euclidean":
            result = euclidean_distance(X_arr, Y_arr, parallel=parallel)
        case "manhattan":
            result = manhattan_distance(X_arr, Y_arr, parallel=parallel)
        case "cosine":
            result = cosine_similarity(X_arr, Y_arr, pairwise=True, parallel=parallel)
        case "mahalanobis" | "wasserstein" | "kolmogorov-smirnov" | "jensen-shannon":
            # For other metrics, compute pairwise manually
            n_x = X_arr.shape[0]
            n_y = Y_arr.shape[0]
            dists = np.zeros((n_x, n_y))

            dist_func = _get_distance_function(metric)

            for i in range(n_x):
                for j in range(n_y):
                    if metric == "mahalanobis":
                        # Mahalanobis requires special handling
                        mean = metric_kwargs.get("mean", np.mean(Y_arr, axis=0))
                        cov = metric_kwargs.get("cov", np.cov(Y_arr, rowvar=False))
                        dists[i, j] = dist_func(X_arr[i], mean, cov=cov)
                    else:
                        dists[i, j] = dist_func(X_arr[i:i+1], Y_arr[j:j+1])

            result = dists
        case _:
            raise ValueError(f"Unknown metric '{metric}'")

    logger.info(f"Pairwise distance matrix computed: shape={result.shape}")
    return result


@log_calls(level=logging.DEBUG)
def distribution_distance(
    points1: npt.ArrayLike,
    points2: npt.ArrayLike | None = None,
    mode: Literal["within", "between"] = "between",
    metric: DistanceMetric = "euclidean",
    parallel: bool = True,
    summary: Literal["mean", "std", "median", "all"] = "mean",
    **metric_kwargs,
) -> float | dict[str, float]:
    """Compute pairwise distances within or between distributions.

    This unified function replaces the separate `within_distribution_distance`
    and `between_distribution_distance` functions, reducing code duplication.

    Parameters
    ----------
    points1 : array-like
        First point distribution, shape (n_samples, n_features).
    points2 : array-like, optional
        Second point distribution, shape (n_samples, n_features).
        Required if mode="between", ignored if mode="within".
    mode : {"within", "between"}, default="between"
        Whether to compute distances within a single distribution or between
        two distributions.
    metric : str, default="euclidean"
        Distance metric to use.
    parallel : bool, default=True
        Use parallel numba implementation if available.
    summary : {"mean", "std", "median", "all"}, default="mean"
        Summary statistic to return. If "all", returns dict with all statistics.
    **metric_kwargs
        Additional keyword arguments for the distance function.

    Returns
    -------
    float or dict
        Summary statistic(s) of distribution distances.

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
    """
    points1_arr = np.asarray(points1)

    # Handle mode-specific logic
    match mode:
        case "within":
            if points1_arr.shape[0] < 2:
                logger.warning("Less than 2 samples, returning zero distances")
                return {"mean": 0.0, "std": 0.0, "median": 0.0} if summary == "all" else 0.0

            logger.info(
                f"Computing within-distribution distances: n_samples={points1_arr.shape[0]}, "
                f"n_features={points1_arr.shape[1]}, metric='{metric}', summary='{summary}'"
            )

            # Compute pairwise distances within single distribution
            dists_matrix = pairwise_distance(
                points1_arr, points1_arr, metric=metric, parallel=parallel, **metric_kwargs
            )

            # Extract upper triangle (excluding diagonal)
            mask = np.triu(np.ones_like(dists_matrix, dtype=bool), k=1)
            dists = dists_matrix[mask]

        case "between":
            if points2 is None:
                raise ValueError("points2 is required when mode='between'")

            points2_arr = np.asarray(points2)

            logger.info(
                f"Computing between-distribution distances: p1.shape={points1_arr.shape}, "
                f"p2.shape={points2_arr.shape}, metric='{metric}', summary='{summary}'"
            )

            # For distribution-level metrics, compute directly
            if metric in {"wasserstein", "kolmogorov-smirnov", "jensen-shannon"}:
                dist_func = _get_distance_function(metric)
                dist = dist_func(points1_arr, points2_arr, **metric_kwargs)
                logger.info(f"Distribution-level distance: {dist:.6f}")
                return dist if summary != "all" else {"mean": dist, "std": 0.0, "median": dist}

            # For point-wise metrics, compute pairwise and summarize
            dists_matrix = pairwise_distance(
                points1_arr, points2_arr, metric=metric, parallel=parallel, **metric_kwargs
            )

            # Flatten all pairwise distances
            dists = dists_matrix.ravel()

        case _:
            raise ValueError(f"Unknown mode '{mode}'. Choose 'within' or 'between'.")

    # Compute summary statistics using helper function
    result = _compute_summary_statistics(dists, summary)

    logger.info(f"{mode.capitalize()}-distribution distance computed: {result}")
    return result
