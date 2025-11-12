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


if TYPE_CHECKING:
    from collections.abc import Callable

try:
    from neural_analysis.utils.logging import get_logger, log_calls
except ImportError:

    def log_calls(**kwargs: Any):  # type: ignore[no-untyped-def,misc]
        def decorator(func):  # type: ignore[no-untyped-def]
            return func

        return decorator

    def get_logger(name: str) -> logging.Logger:  # type: ignore[misc]
        return logging.getLogger(name)


# Module logger
logger = get_logger(__name__)

# Optional acceleration with numba for large pairwise computations
try:  # pragma: no cover - optional dependency
    from numba import njit, prange

    NUMBA_AVAILABLE = True

    @njit(parallel=True, fastmath=True)  # type: ignore[misc]
    def _pairwise_euclidean_numba(
        x_arr: npt.NDArray[np.float64], y_arr: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Numba-accelerated pairwise Euclidean distance."""
        n_x, n_feat = x_arr.shape
        n_y, _ = y_arr.shape
        out = np.empty((n_x, n_y), dtype=np.float64)
        for i in prange(n_x):
            for j in range(n_y):
                s = 0.0
                for k in range(n_feat):
                    d = x_arr[i, k] - y_arr[j, k]
                    s += d * d
                out[i, j] = np.sqrt(s)
        return out

    @njit(parallel=True, fastmath=True)  # type: ignore[misc]
    def _pairwise_cosine_numba(
        x_arr: npt.NDArray[np.float64], y_arr: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Numba-accelerated pairwise cosine similarity."""
        n_x, n_feat = x_arr.shape
        n_y, _ = y_arr.shape
        out = np.empty((n_x, n_y), dtype=np.float64)
        for i in prange(n_x):
            norm_x = 0.0
            for k in range(n_feat):
                norm_x += x_arr[i, k] ** 2
            norm_x = np.sqrt(norm_x)

            for j in range(n_y):
                dot_prod = 0.0
                norm_y = 0.0
                for k in range(n_feat):
                    dot_prod += x_arr[i, k] * y_arr[j, k]
                    norm_y += y_arr[j, k] ** 2
                norm_y = np.sqrt(norm_y)

                if norm_x > 0 and norm_y > 0:
                    out[i, j] = dot_prod / (norm_x * norm_y)
                else:
                    out[i, j] = 0.0
        return out

    @njit(parallel=True, fastmath=True)  # type: ignore[misc]
    def _pairwise_manhattan_numba(
        x_arr: npt.NDArray[np.float64], y_arr: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Numba-accelerated pairwise Manhattan distance."""
        n_x, n_feat = x_arr.shape
        n_y, _ = y_arr.shape
        out = np.empty((n_x, n_y), dtype=np.float64)
        for i in prange(n_x):
            for j in range(n_y):
                s = 0.0
                for k in range(n_feat):
                    s += abs(x_arr[i, k] - y_arr[j, k])
                out[i, j] = s
        return out

except Exception:  # pragma: no cover - optional dependency
    NUMBA_AVAILABLE = False

__all__ = [
    "euclidean_distance",
    "manhattan_distance",
    "mahalanobis_distance",
    "cosine_similarity",
    "pairwise_distance",
]
# =============================================================================
# Point-to-Point Distance Metrics
# =============================================================================


@log_calls(level=logging.DEBUG)
def euclidean_distance(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    axis: int | None = None,
    parallel: bool = True,
) -> float | npt.NDArray[np.floating]:
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
                result: npt.NDArray[np.floating] = _pairwise_euclidean_numba(
                    x_arr.astype(np.float64), y_arr.astype(np.float64)
                )
                return result
            else:
                result_cdist: npt.NDArray[np.floating] = cdist(
                    x_arr, y_arr, metric="euclidean"
                )
                return result_cdist
        else:
            # Distance along specified axis
            return np.linalg.norm(x_arr - y_arr, axis=axis)  # type: ignore[no-any-return]

    # Fallback for any other shape
    return np.linalg.norm(x_arr - y_arr, axis=axis)  # type: ignore[return-value]


@log_calls(level=logging.DEBUG)
def manhattan_distance(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    axis: int | None = None,
    parallel: bool = True,
) -> float | npt.NDArray[np.floating]:
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
                return _pairwise_manhattan_numba(  # type: ignore[no-any-return]
                    x_arr.astype(np.float64), y_arr.astype(np.float64)
                )
            else:
                return cdist(x_arr, y_arr, metric="cityblock")  # type: ignore[no-any-return]
        else:
            # Distance along specified axis
            return np.sum(np.abs(x_arr - y_arr), axis=axis)  # type: ignore[no-any-return]

    # Fallback
    return np.sum(np.abs(x_arr - y_arr), axis=axis)  # type: ignore[no-any-return]


@log_calls(level=logging.DEBUG)
def mahalanobis_distance(
    x: npt.ArrayLike,
    mean: npt.ArrayLike,
    cov: npt.ArrayLike | None = None,
    inv_cov: npt.ArrayLike | None = None,
) -> float | npt.NDArray[np.floating]:
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
    return np.sqrt(dist_sq)  # type: ignore[no-any-return]


@log_calls(level=logging.DEBUG)
def cosine_similarity(
    v1: npt.ArrayLike,
    v2: npt.ArrayLike,
    pairwise: bool = False,
    parallel: bool = True,
) -> float | npt.NDArray[np.floating]:
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
            return _pairwise_cosine_numba(  # type: ignore[no-any-return]
                v1_arr.astype(np.float64), v2_arr.astype(np.float64)
            )
        else:
            # Use sklearn's cosine_similarity
            from sklearn.metrics.pairwise import (  # type: ignore[import-untyped]  # noqa: E402
                cosine_similarity as sklearn_cosine,
            )

            return sklearn_cosine(v1_arr, v2_arr)  # type: ignore[no-any-return]

    # Default: flatten and compute single similarity
    return float(1.0 - cosine(v1_arr.flatten(), v2_arr.flatten()))


# =============================================================================
# Pairwise Distance System (Plugin-based)
# =============================================================================

DistanceMetric = Literal[
    "euclidean",
    "manhattan",
    "mahalanobis",
    "cosine",
]


def _get_distance_function(metric: DistanceMetric) -> Callable[..., float]:
    """Get the distance function for a given metric name.

    This is the plugin registry for distance metrics.

    Note: Distribution-level metrics (wasserstein, kolmogorov-smirnov,
    jensen-shannon) have been moved to the distributions module.
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
        case _:
            raise ValueError(
                f"Unknown metric '{metric}'. Choose from: euclidean, manhattan, "
                "mahalanobis, cosine"
            )


@log_calls(level=logging.DEBUG)
def pairwise_distance(
    x: npt.ArrayLike,
    y: npt.ArrayLike | None = None,
    metric: DistanceMetric = "euclidean",
    parallel: bool = True,
    **metric_kwargs: object,
) -> npt.NDArray[np.floating]:
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
    x_arr = np.asarray(x)
    y_arr = np.asarray(y) if y is not None else x_arr

    if x_arr.ndim == 1:
        x_arr = x_arr.reshape(1, -1)
    if y_arr.ndim == 1:
        y_arr = y_arr.reshape(1, -1)

    logger.info(
        f"Computing pairwise distances: x.shape={x_arr.shape}, y.shape={y_arr.shape}, "
        f"metric='{metric}', parallel={parallel}"
    )

    # For metrics that have native pairwise support - use match/case
    match metric:
        case "euclidean":
            result = euclidean_distance(x_arr, y_arr, parallel=parallel)
        case "manhattan":
            result = manhattan_distance(x_arr, y_arr, parallel=parallel)
        case "cosine":
            result = cosine_similarity(x_arr, y_arr, pairwise=True, parallel=parallel)
        case "mahalanobis":
            # Mahalanobis requires special handling with covariance matrix
            n_x = x_arr.shape[0]
            n_y = y_arr.shape[0]
            dists = np.zeros((n_x, n_y))

            dist_func = _get_distance_function(metric)

            for i in range(n_x):
                for j in range(n_y):
                    # Mahalanobis requires special handling
                    mean = metric_kwargs.get("mean", np.mean(y_arr, axis=0))
                    cov = metric_kwargs.get("cov", np.cov(y_arr, rowvar=False))
                    dists[i, j] = dist_func(x_arr[i], mean, cov=cov)

            result = dists
        case _:
            raise ValueError(f"Unknown metric '{metric}'")

    logger.info(f"Pairwise distance matrix computed: shape={result.shape}")
    return result  # type: ignore[no-any-return]
