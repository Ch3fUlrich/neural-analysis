"""Distance metrics for neural data analysis.

This module provides standard distance and similarity measures used throughout
neural analysis workflows. All functions accept NumPy arrays and return scalars
or arrays depending on input dimensionality.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import cdist, cosine

# Optional acceleration with numba for large pairwise computations
try:  # pragma: no cover - optional dependency
    from numba import njit, prange  # type: ignore

    NUMBA_AVAILABLE = True

    @njit(parallel=True, fastmath=True)
    def _pairwise_euclidean_numba(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
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
except Exception:  # pragma: no cover - optional dependency
    NUMBA_AVAILABLE = False

__all__ = ["euclidean_distance", "mahalanobis_distance", "cosine_similarity"]


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

    >>> # Point cloud distance (mean nearest neighbor)
    >>> X = np.random.randn(100, 10)
    >>> Y = np.random.randn(100, 10)
    >>> dist = euclidean_distance(X, Y)
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
                return _pairwise_euclidean_numba(x_arr.astype(np.float64), y_arr.astype(np.float64))
            else:
                return cdist(x_arr, y_arr, metric="euclidean")
        else:
            # Distance along specified axis
            return np.linalg.norm(x_arr - y_arr, axis=axis)

    # Fallback for any other shape
    return np.linalg.norm(x_arr - y_arr, axis=axis)


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
    # diff: (n_samples, n_features)
    # inv_cov: (n_features, n_features)
    dist_sq = np.sum(diff @ inv_cov_arr * diff, axis=1)
    return np.sqrt(dist_sq)


def cosine_similarity(
    v1: npt.ArrayLike,
    v2: npt.ArrayLike,
) -> float:
    """Compute cosine similarity between two vectors.

    Parameters
    ----------
    v1, v2 : array-like
        Input vectors (must be 1D and same length).

    Returns
    -------
    float
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
    v1_arr = np.asarray(v1).flatten()
    v2_arr = np.asarray(v2).flatten()

    # scipy.spatial.distance.cosine returns 1 - similarity
    return float(1.0 - cosine(v1_arr, v2_arr))
