"""Unified pairwise metrics: distances, similarities, correlations, autocorrelation

This module consolidates functionality from the original `distance.py`
and `similarity.py` modules into a single place. It exposes point-to-point
distance functions, pairwise distance dispatch, correlation and similarity
matrices, and spatial autocorrelation helpers.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import cdist, cosine
from scipy.stats import kendalltau, spearmanr

try:
    from neural_analysis.utils.logging import get_logger, log_calls
except Exception:

    def log_calls(**kwargs: Any):  # type: ignore[no-untyped-def,misc]
        def decorator(func):  # type: ignore[no-untyped-def]
            return func

        return decorator

    def get_logger(name: str) -> logging.Logger:  # type: ignore[misc]
        return logging.getLogger(name)


logger = get_logger(__name__)

# Optional numba support
try:
    import numba

    NUMBA_AVAILABLE = True


    class BetweenResult(TypedDict):
        """TypedDict for wrapping scalar between-mode result.

        Example: {"value": 0.1234, "metric": "euclidean"}
        """

        value: float
        metric: str
except Exception:
    numba = None
    NUMBA_AVAILABLE = False

# ============================================================================
# Metric Category Constants
# ============================================================================

# Point-to-point metrics: compute pairwise distances between all sample pairs
# Used in: within-dataset (mode="within"), between-dataset (mode="between")
# Returns: distance matrix (n_samples_x, n_samples_y)
POINT_TO_POINT_METRICS = frozenset(
    {
        "euclidean",
        "manhattan",
        "cosine",
        "mahalanobis",
    }
)

# Distribution-level metrics: treat datasets as distributions, return scalar
# Used in: between-dataset (mode="between"), all-pairs (mode="all-pairs")
# Returns: float distance value
DISTRIBUTION_METRICS = frozenset(
    {
        "wasserstein",
        "kolmogorov-smirnov",
        "jensen-shannon",
    }
)

# Shape metrics: compare neural population activity matrices as shapes
# Used in: between-dataset (mode="between"), all-pairs (mode="all-pairs")
# Returns: float distance value (+ optional pairs dict)
SHAPE_METRICS = frozenset(
    {
        "procrustes",
        "one-to-one",
        "soft-matching",
    }
)

# All scalar-returning metrics (for all-pairs mode)
SCALAR_METRICS = DISTRIBUTION_METRICS | SHAPE_METRICS

# All supported metrics
ALL_METRICS = POINT_TO_POINT_METRICS | DISTRIBUTION_METRICS | SHAPE_METRICS

__all__ = [
    # Metric constants
    "POINT_TO_POINT_METRICS",
    "DISTRIBUTION_METRICS",
    "SHAPE_METRICS",
    "SCALAR_METRICS",
    "ALL_METRICS",
    # Type aliases
    "DistanceMetric",
    "DistributionMetric",
    "ShapeMetric",
    "AnyMetric",
    "ComparisonMode",
    # Distances
    "euclidean_distance",
    "manhattan_distance",
    "mahalanobis_distance",
    "cosine_similarity",
    "pairwise_distance",
    # Phase 3 explicit comparison functions
    "compute_within_distances",
    "compute_between_distances",
    "compute_all_pairs",
    # Unified orchestration API (recommended)
    "compare_datasets",
    # Unified pairwise computation (internal dispatcher)
    "compute_pairwise_matrix",
    # Correlations / similarities
    "correlation",
    "correlation_matrix",
    "cosine_similarity_matrix",
    "angular_similarity_matrix",
    "similarity_matrix",
    # Spatial
    "spatial_autocorrelation",
]


# ---------------------------------------------------------------------------
# Numba-accelerated pairwise helpers (copied from distance.py semantics)
# ---------------------------------------------------------------------------
try:
    from numba import njit, prange

    @njit(parallel=True, fastmath=True)  # type: ignore[misc]
    def _pairwise_euclidean_numba(
        x_arr: npt.NDArray[np.float64], y_arr: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
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
except Exception:
    # numba functions not available; fall back to numpy/scipy
    pass


# ---------------------------------------------------------------------------
# Point-to-point distances (from distance.py)
# ---------------------------------------------------------------------------


@log_calls(level=logging.DEBUG)
def euclidean_distance(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    axis: int | None = None,
    parallel: bool = True,
) -> float | npt.NDArray[np.floating]:
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)

    if x_arr.ndim == 1 and y_arr.ndim == 1:
        return float(np.linalg.norm(x_arr - y_arr))

    if x_arr.ndim == 2 and y_arr.ndim == 2:
        if axis is None:
            if parallel and NUMBA_AVAILABLE:
                return cast(
                    "npt.NDArray[np.floating]",
                    _pairwise_euclidean_numba(
                        x_arr.astype(np.float64), y_arr.astype(np.float64)
                    ),
                )
            else:
                return cast(
                    "npt.NDArray[np.floating]", cdist(x_arr, y_arr, metric="euclidean")
                )
        else:
            result_norm = np.linalg.norm(x_arr - y_arr, axis=axis)
            return cast("float | npt.NDArray[np.floating]", result_norm)

    result_norm = np.linalg.norm(x_arr - y_arr, axis=axis)
    return cast("float | npt.NDArray[np.floating]", result_norm)


@log_calls(level=logging.DEBUG)
def manhattan_distance(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    axis: int | None = None,
    parallel: bool = True,
) -> float | npt.NDArray[np.floating]:
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)

    if x_arr.ndim == 1 and y_arr.ndim == 1:
        return float(np.sum(np.abs(x_arr - y_arr)))

    if x_arr.ndim == 2 and y_arr.ndim == 2:
        if axis is None:
            if parallel and NUMBA_AVAILABLE:
                return cast(
                    "npt.NDArray[np.floating]",
                    _pairwise_manhattan_numba(
                        x_arr.astype(np.float64), y_arr.astype(np.float64)
                    ),
                )
            else:
                return cast(
                    "npt.NDArray[np.floating]", cdist(x_arr, y_arr, metric="cityblock")
                )
        else:
            result_sum = np.sum(np.abs(x_arr - y_arr), axis=axis)
            return cast("float | npt.NDArray[np.floating]", result_sum)

    result_sum = np.sum(np.abs(x_arr - y_arr), axis=axis)
    return cast("float | npt.NDArray[np.floating]", result_sum)


@log_calls(level=logging.DEBUG)
def mahalanobis_distance(
    x: npt.ArrayLike,
    mean: npt.ArrayLike,
    cov: npt.ArrayLike | None = None,
    inv_cov: npt.ArrayLike | None = None,
) -> float | npt.NDArray[np.floating]:
    x_arr = np.asarray(x)
    mean_arr = np.asarray(mean)

    if inv_cov is None:
        if cov is None:
            raise ValueError("Either cov or inv_cov must be provided")
        cov_arr = np.asarray(cov)
        inv_cov_arr = np.linalg.inv(cov_arr)
    else:
        inv_cov_arr = np.asarray(inv_cov)

    diff = x_arr - mean_arr
    if diff.ndim == 1:
        dist_sq = diff @ inv_cov_arr @ diff
        return float(np.sqrt(dist_sq))

    dist_sq = np.sum(diff @ inv_cov_arr * diff, axis=1)
    return cast("npt.NDArray[np.floating]", np.sqrt(dist_sq))


@log_calls(level=logging.DEBUG)
def cosine_similarity(
    v1: npt.ArrayLike,
    v2: npt.ArrayLike,
    pairwise: bool = False,
    parallel: bool = True,
) -> float | npt.NDArray[np.floating]:
    v1_arr = np.asarray(v1)
    v2_arr = np.asarray(v2)

    if v1_arr.ndim == 1:
        return float(1.0 - cosine(v1_arr, v2_arr))

    if v1_arr.ndim == 2 and v2_arr.ndim == 2 and pairwise:
        if parallel and NUMBA_AVAILABLE:
            return cast(
                "npt.NDArray[np.floating]",
                _pairwise_cosine_numba(
                    v1_arr.astype(np.float64), v2_arr.astype(np.float64)
                ),
            )
        else:
            from sklearn.metrics.pairwise import (
                cosine_similarity as sklearn_cosine,
            )

            result_cos = sklearn_cosine(v1_arr, v2_arr)
            return cast("npt.NDArray[np.floating]", result_cos)

    return float(1.0 - cosine(v1_arr.flatten(), v2_arr.flatten()))


# ---------------------------------------------------------------------------
# Pairwise dispatch (plugin-like)
# ---------------------------------------------------------------------------

# Type aliases for improved API clarity and type safety
DistanceMetric = Literal["euclidean", "manhattan", "mahalanobis", "cosine"]
DistributionMetric = Literal["wasserstein", "kolmogorov-smirnov", "jensen-shannon"]
ShapeMetric = Literal["procrustes", "one-to-one", "soft-matching"]
AnyMetric = Literal[
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
]
ComparisonMode = Literal["within", "between", "all-pairs"]


def _get_distance_function(metric: DistanceMetric) -> Any:
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
            raise ValueError(f"Unknown metric '{metric}'")


@log_calls(level=logging.DEBUG)
def pairwise_distance(
    x: npt.ArrayLike,
    y: npt.ArrayLike | None = None,
    metric: DistanceMetric = "euclidean",
    parallel: bool = True,
    **metric_kwargs: object,
) -> npt.NDArray[np.floating]:
    """Compute pairwise distances using point-to-point metrics.

    This function computes distances between all pairs of samples from two
    datasets using various distance metrics.

    Parameters
    ----------
    x : array-like, shape (n_samples_x, n_features)
        First dataset
    y : array-like, shape (n_samples_y, n_features), optional
        Second dataset. If None, computes pairwise within x.
    metric : {"euclidean", "manhattan", "cosine", "mahalanobis"}
        Distance metric to use
    parallel : bool, default=True
        Use parallel computation if numba is available
    **metric_kwargs
        Additional arguments for the metric (e.g., mean/cov for mahalanobis)

    Returns
    -------
    ndarray, shape (n_samples_x, n_samples_y)
        Pairwise distance matrix

    See Also
    --------
    compute_pairwise_matrix : Unified interface for all pairwise metrics
    """
    x_arr, y_arr = _validate_pairwise_inputs(x, y)

    logger.info(
        f"Computing pairwise distances: x.shape={x_arr.shape}, y.shape={y_arr.shape}, "
        f"metric='{metric}', parallel={parallel}"
    )

    match metric:
        case "euclidean":
            result = euclidean_distance(x_arr, y_arr, parallel=parallel)
        case "manhattan":
            result = manhattan_distance(x_arr, y_arr, parallel=parallel)
        case "cosine":
            result = cosine_similarity(x_arr, y_arr, pairwise=True, parallel=parallel)
        case "mahalanobis":
            n_x = x_arr.shape[0]
            n_y = y_arr.shape[0]
            dists = np.zeros((n_x, n_y))
            dist_func = _get_distance_function(metric)
            for i in range(n_x):
                for j in range(n_y):
                    mean = metric_kwargs.get("mean", np.mean(y_arr, axis=0))
                    cov = metric_kwargs.get("cov", np.cov(y_arr, rowvar=False))
                    dists[i, j] = dist_func(x_arr[i], mean, cov=cov)
            result = dists
        case _:
            raise ValueError(f"Unknown metric '{metric}'")

    logger.info(f"Pairwise distance matrix computed: shape={result.shape}")
    return cast("npt.NDArray[np.floating]", result)


# ---------------------------------------------------------------------------
# Unified Pairwise Computation System
# ---------------------------------------------------------------------------


def _validate_pairwise_inputs(
    x: npt.ArrayLike, y: npt.ArrayLike | None
) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """Validate and normalize inputs for pairwise computations.

    Parameters
    ----------
    x : array-like
        First dataset
    y : array-like or None
        Second dataset. If None, uses x.

    Returns
    -------
    tuple of ndarray
        Validated (x_arr, y_arr) arrays with shape (n_samples, n_features)

    Raises
    ------
    ValueError
        If x and y have different feature dimensions
    """
    x_arr = np.asarray(x)
    y_arr = np.asarray(y) if y is not None else x_arr

    # Ensure 2D arrays
    if x_arr.ndim == 1:
        x_arr = x_arr.reshape(1, -1)
    if y_arr.ndim == 1:
        y_arr = y_arr.reshape(1, -1)

    # Validate feature dimension match
    if y is not None and x_arr.shape[1] != y_arr.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: x has {x_arr.shape[1]} features, "
            f"y has {y_arr.shape[1]} features. All datasets must have the same "
            "number of features."
        )

    return x_arr, y_arr


def compute_pairwise_matrix(
    x: npt.ArrayLike,
    y: npt.ArrayLike | None = None,
    metric: str = "euclidean",
    parallel: bool = True,
    **metric_kwargs: Any,
) -> npt.NDArray[np.floating] | tuple[float, dict[tuple[int, int], float]]:
    """Unified pairwise computation for any metric.

    This is a generic interface that dispatches to specialized functions based
    on the metric name. Supports both distance metrics (euclidean, manhattan, etc.)
    and distribution metrics (wasserstein, ks, js).

    Parameters
    ----------
    x : array-like, shape (n_samples_x, n_features)
        First dataset
    y : array-like, shape (n_samples_y, n_features), optional
        Second dataset. If None, computes pairwise within x.
    metric : str, default="euclidean"
        Metric to compute. Supported metrics:

        **Point-to-point distances** (return matrix):
        - "euclidean": Euclidean distance
        - "manhattan": Manhattan (L1) distance
        - "cosine": Cosine similarity
        - "mahalanobis": Mahalanobis distance (requires mean/cov kwargs)

        **Distribution-level metrics** (single value or tuple):
        - "wasserstein": Wasserstein distance (Earth Mover's Distance)
        - "kolmogorov-smirnov", "ks": Kolmogorov-Smirnov test
        - "jensen-shannon", "js": Jensen-Shannon divergence

        **Shape metrics** (return tuple with pairs):
        - "procrustes": Procrustes distance after alignment
        - "one-to-one": Bipartite matching distance
        - "soft-matching": Soft optimal transport distance
    parallel : bool, default=True
        Use parallel computation if available (for point-to-point metrics)
    **metric_kwargs
        Additional arguments passed to the metric function

    Returns
    -------
    ndarray or tuple
        For point-to-point metrics: (n_samples_x, n_samples_y) distance matrix
        For distribution metrics: float distance value
        For shape metrics: tuple of (distance, pairs_dict)

    Examples
    --------
    >>> x = np.random.randn(100, 10)
    >>> y = np.random.randn(50, 10)
    >>>
    >>> # Point-to-point distance matrix
    >>> dists = compute_pairwise_matrix(x, y, metric="euclidean")
    >>> dists.shape
    (100, 50)
    >>>
    >>> # Distribution-level comparison
    >>> dist = compute_pairwise_matrix(x, y, metric="wasserstein")
    >>> type(dist)
    <class 'float'>
    >>>
    >>> # Shape distance with pairs
    >>> dist, pairs = compute_pairwise_matrix(x, y, metric="procrustes")

    Notes
    -----
    This function provides a single source of responsibility for all pairwise
    computations, eliminating duplication between distance and distribution
    comparison logic.
    """
    x_arr, y_arr = _validate_pairwise_inputs(x, y)

    logger.debug(
        f"Computing pairwise {metric}: x.shape={x_arr.shape}, "
        f"y.shape={y_arr.shape}, parallel={parallel}"
    )

    # Normalize metric aliases
    metric_normalized = metric.lower().replace("_", "-").replace(" ", "-")
    metric_map = {
        "ks": "kolmogorov-smirnov",
        "js": "jensen-shannon",
    }
    metric_normalized = metric_map.get(metric_normalized, metric_normalized)

    # Dispatch to appropriate function based on metric type
    # Point-to-point distance metrics (return matrix)
    if metric_normalized in ["euclidean", "manhattan", "cosine", "mahalanobis"]:
        return pairwise_distance(  # type: ignore[no-any-return]
            x_arr,
            y_arr,
            metric=metric_normalized,
            parallel=parallel,
            **metric_kwargs,
        )

    # Distribution-level metrics (return scalar)
    elif metric_normalized in ["wasserstein", "kolmogorov-smirnov", "jensen-shannon"]:
        # Import here to avoid circular dependency
        from .distributions import (
            jensen_shannon_divergence,
            kolmogorov_smirnov_distance,
            wasserstein_distance_multi,
        )

        metric_func_map: dict[str, Any] = {
            "wasserstein": wasserstein_distance_multi,
            "kolmogorov-smirnov": kolmogorov_smirnov_distance,
            "jensen-shannon": jensen_shannon_divergence,
        }
        func = metric_func_map[metric_normalized]
        return func(x_arr, y_arr, **metric_kwargs)  # type: ignore[no-any-return]

    # Shape metrics (return tuple)
    elif metric_normalized in ["procrustes", "one-to-one", "soft-matching"]:
        # Import here to avoid circular dependency
        from typing import cast

        from .distributions import shape_distance


        method_typed = cast(
            "Literal['procrustes', 'one-to-one', 'soft-matching']",
            metric_normalized,
        )

        return cast(
            "tuple[float, dict[tuple[int, int], float]]",
            shape_distance(
            x_arr.astype(np.float64),
            y_arr.astype(np.float64),
            method=method_typed,
            return_pairs=True,
            **metric_kwargs,
        )
        )

    else:
        raise ValueError(
            f"Unknown metric '{metric}'. Supported: euclidean, manhattan, cosine, "
            "mahalanobis, wasserstein, kolmogorov-smirnov, jensen-shannon, "
            "procrustes, one-to-one, soft-matching"
        )


# ============================================================================
# New Explicit Comparison Functions (Mode-Based API)
# ============================================================================


def _validate_metric_mode(metric: str, mode: str) -> str:
    """Validate metric-mode compatibility and normalize metric name.

    Parameters
    ----------
    metric : str
        Metric name (may include aliases like "ks", "js")
    mode : {"within", "between", "all-pairs"}
        Comparison mode

    Returns
    -------
    str
        Normalized metric name

    Raises
    ------
    ValueError
        If metric-mode combination is invalid
    """
    # Normalize metric name
    metric_normalized = metric.lower().replace("_", "-").replace(" ", "-")
    metric_map = {
        "ks": "kolmogorov-smirnov",
        "js": "jensen-shannon",
    }
    metric_normalized = metric_map.get(metric_normalized, metric_normalized)

    # Check if metric exists
    if metric_normalized not in ALL_METRICS:
        raise ValueError(
            f"Unknown metric '{metric}'. Supported metrics: "
            f"{', '.join(sorted(ALL_METRICS))}"
        )

    # Validate mode-metric compatibility
    match mode:
        case "within":
            # Only point-to-point metrics allowed (they compute n×n matrices)
            if metric_normalized not in POINT_TO_POINT_METRICS:
                raise ValueError(
                    f"Metric '{metric}' cannot be used with mode='within'. "
                    f"Within-dataset comparisons require point-to-point metrics that "
                    f"compute pairwise distances between all samples. "
                    f"Allowed metrics for mode='within': "
                    f"{', '.join(sorted(POINT_TO_POINT_METRICS))}"
                )
        case "between":
            # All metrics allowed
            pass
        case "all-pairs":
            # Only scalar-returning metrics allowed (for compact dict output)
            if metric_normalized not in SCALAR_METRICS:
                raise ValueError(
                    f"Metric '{metric}' cannot be used with mode='all-pairs'. "
                    f"All-pairs mode requires metrics that return scalar values "
                    f"(not matrices) to enable compact "
                    f"dict[str, dict[str, float]] output. "
                    f"Allowed metrics for mode='all-pairs': "
                    f"{', '.join(sorted(SCALAR_METRICS))}"
                )
        case _:
            raise ValueError(
                f"Unknown mode '{mode}'. "
                f"Supported modes: 'within', 'between', 'all-pairs'"
            )

    return metric_normalized


@log_calls(level=logging.DEBUG)
def compute_within_distances(
    data: npt.ArrayLike,
    metric: str = "euclidean",
    return_matrix: bool = False,
    parallel: bool = True,
    **metric_kwargs: Any,
) -> float | npt.NDArray[np.floating]:
    """Compute pairwise distances within a single dataset.

    This function computes distances between all pairs of samples within a
    dataset using point-to-point metrics. Results can be returned as either
    a summary statistic (mean) or the full symmetric distance matrix.

    Parameters
    ----------
    data : array-like, shape (n_samples, n_features)
        Dataset to analyze
    metric : str, default="euclidean"
        Distance metric. Only point-to-point metrics allowed:
        - "euclidean": Euclidean distance
        - "manhattan": Manhattan (L1) distance
        - "cosine": Cosine similarity
        - "mahalanobis": Mahalanobis distance (requires cov/inv_cov kwargs)
    return_matrix : bool, default=False
        If True, return full (n_samples, n_samples) symmetric matrix.
        If False, return mean distance as summary statistic.
    parallel : bool, default=True
        Use parallel computation if available (numba)
    **metric_kwargs
        Additional arguments for the metric function

    Returns
    -------
    float or ndarray
        If return_matrix=False: float (mean pairwise distance)
        If return_matrix=True: ndarray with shape (n_samples, n_samples)

    Raises
    ------
    ValueError
        If metric is not a point-to-point metric

    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.randn(100, 10)
    >>>
    >>> # Get mean within-dataset distance
    >>> mean_dist = compute_within_distances(data, metric="euclidean")
    >>> isinstance(mean_dist, float)
    True
    >>>
    >>> # Get full distance matrix
    >>> dist_matrix = compute_within_distances(
    ...     data, metric="euclidean", return_matrix=True
    ... )
    >>> dist_matrix.shape
    (100, 100)
    >>>
    >>> # Matrix is symmetric
    >>> np.allclose(dist_matrix, dist_matrix.T)
    True
    """
    # Validate metric-mode compatibility
    metric_normalized = _validate_metric_mode(metric, "within")

    data_arr = np.asarray(data)
    if data_arr.ndim == 1:
        data_arr = data_arr.reshape(-1, 1)

    if data_arr.shape[0] < 2:
        logger.warning("Less than 2 samples, returning zero distance")
        if return_matrix:
            return np.zeros((data_arr.shape[0], data_arr.shape[0]))
        return 0.0

    logger.info(
        f"Computing within-dataset distances: n_samples={data_arr.shape[0]}, "
        f"n_features={data_arr.shape[1]}, metric='{metric_normalized}', "
        f"return_matrix={return_matrix}"
    )

    # Compute full pairwise distance matrix
    dist_matrix = pairwise_distance(
        data_arr,
        data_arr,
        metric=metric_normalized,
        parallel=parallel,
            **metric_kwargs,
    )

    if return_matrix:
        logger.info(f"Returning full distance matrix: shape={dist_matrix.shape}")
        return cast("npt.NDArray[np.floating]", dist_matrix)
    else:
        # Extract upper triangle (excluding diagonal) and compute mean
        mask = np.triu(np.ones_like(dist_matrix, dtype=bool), k=1)
        dists = dist_matrix[mask]
        mean_dist = float(np.mean(dists))
        logger.info(f"Mean within-dataset distance: {mean_dist:.6f}")
        return mean_dist


@log_calls(level=logging.DEBUG)
def compute_between_distances(
    data1: npt.ArrayLike,
    data2: npt.ArrayLike,
    metric: str = "euclidean",
    return_matrix: bool = False,
    parallel: bool = True,
    **metric_kwargs: Any,
) -> float | npt.NDArray[np.floating]:
    """Compute distances between two datasets.

    This function compares two datasets using various metrics. For point-to-point
    metrics, can return either a summary statistic or the full distance matrix.
    For distribution/shape metrics, always returns a scalar distance value.

    Parameters
    ----------
    data1 : array-like, shape (n_samples1, n_features)
        First dataset
    data2 : array-like, shape (n_samples2, n_features)
        Second dataset
    metric : str, default="euclidean"
        Distance metric. Supported metrics:

        **Point-to-point metrics** (can return matrix):
        - "euclidean": Euclidean distance
        - "manhattan": Manhattan (L1) distance
        - "cosine": Cosine similarity
        - "mahalanobis": Mahalanobis distance

        **Distribution metrics** (always return scalar):
        - "wasserstein": Wasserstein distance (Earth Mover's Distance)
        - "kolmogorov-smirnov": Kolmogorov-Smirnov statistic
        - "jensen-shannon": Jensen-Shannon divergence

        **Shape metrics** (always return scalar):
        - "procrustes": Procrustes distance after optimal alignment
        - "one-to-one": Bipartite matching distance
        - "soft-matching": Soft assignment via optimal transport
    return_matrix : bool, default=False
        If True and using point-to-point metric, return full (n1, n2) matrix.
        If False or using distribution/shape metric, return scalar mean/distance.
        Ignored for distribution/shape metrics (always return scalar).
    parallel : bool, default=True
        Use parallel computation if available
    **metric_kwargs
        Additional arguments for the metric function

    Returns
    -------
    float or ndarray
        Point-to-point + return_matrix=False: float (mean distance)
        Point-to-point + return_matrix=True: ndarray (n_samples1, n_samples2)
        Distribution/shape metrics: float (distance value)

    Examples
    --------
    >>> import numpy as np
    >>> data1 = np.random.randn(100, 10)
    >>> data2 = np.random.randn(80, 10)
    >>>
    >>> # Mean between-dataset distance
    >>> mean_dist = compute_between_distances(data1, data2, metric="euclidean")
    >>> isinstance(mean_dist, float)
    True
    >>>
    >>> # Full distance matrix
    >>> dist_matrix = compute_between_distances(
    ...     data1, data2, metric="euclidean", return_matrix=True
    ... )
    >>> dist_matrix.shape
    (100, 80)
    >>>
    >>> # Distribution comparison (always scalar)
    >>> dist = compute_between_distances(data1, data2, metric="wasserstein")
    >>> isinstance(dist, float)
    True
    """
    # Validate metric-mode compatibility
    metric_normalized = _validate_metric_mode(metric, "between")

    data1_arr = np.asarray(data1)
    data2_arr = np.asarray(data2)

    if data1_arr.ndim == 1:
        data1_arr = data1_arr.reshape(-1, 1)
    if data2_arr.ndim == 1:
        data2_arr = data2_arr.reshape(-1, 1)

    logger.info(
        f"Computing between-dataset distances: data1.shape={data1_arr.shape}, "
        f"data2.shape={data2_arr.shape}, metric='{metric_normalized}', "
        f"return_matrix={return_matrix}"
    )

    # Use unified dispatcher
    result = compute_pairwise_matrix(
        data1_arr,
        data2_arr,
        metric=metric_normalized,
        parallel=parallel,
        **metric_kwargs,
    )

    # Handle different return types and return standard floats/arrays
    if metric_normalized in POINT_TO_POINT_METRICS:
        # Point-to-point metrics return matrix
        assert isinstance(result, np.ndarray), (
            f"Expected ndarray for {metric_normalized}"
        )

        if return_matrix:
            logger.info(f"Returning full distance matrix: shape={result.shape}")
            return result
        else:
            mean_dist = float(np.mean(result))
            logger.info(f"Mean between-dataset distance: {mean_dist:.6f}")
            return mean_dist

    elif metric_normalized in DISTRIBUTION_METRICS:
        # Distribution metrics return scalar
        assert isinstance(result, (int, float, np.number)), (
            f"Expected scalar for {metric_normalized}"
        )
        dist = float(result)
        logger.info(f"Distribution distance: {dist:.6f}")
        return dist

    elif metric_normalized in SHAPE_METRICS:
        # Shape metrics return tuple (distance, pairs)
        assert isinstance(result, tuple), f"Expected tuple for {metric_normalized}"
        dist, _pairs = result
        logger.info(f"Shape distance: {dist:.6f}")
        return float(dist)

    else:
        raise ValueError(f"Unexpected metric type: {metric_normalized}")


@log_calls(level=logging.DEBUG)
def compute_all_pairs(
    datasets: dict[str, npt.NDArray[np.floating]],
    metric: str = "wasserstein",
    parallel: bool = True,
    show_progress: bool = True,
    **metric_kwargs: Any,
) -> dict[str, dict[str, float]]:
    """Compute pairwise comparisons for all dataset pairs (within + between).

    This function computes a full n×n comparison matrix where n is the number
    of datasets. Diagonal elements represent within-dataset comparisons, and
    off-diagonal elements represent between-dataset comparisons. Only scalar-
    returning metrics (distribution and shape metrics) are supported.

    Parameters
    ----------
    datasets : dict[str, ndarray]
        Dictionary mapping dataset names to data arrays (n_samples, n_features)
    metric : str, default="wasserstein"
        Distance metric. Only scalar-returning metrics allowed:

        **Distribution metrics:**
        - "wasserstein": Wasserstein distance (Earth Mover's Distance)
        - "kolmogorov-smirnov": Kolmogorov-Smirnov statistic
        - "jensen-shannon": Jensen-Shannon divergence

        **Shape metrics:**
        - "procrustes": Procrustes distance after optimal alignment
        - "one-to-one": Bipartite matching distance
        - "soft-matching": Soft assignment via optimal transport
    parallel : bool, default=True
        Use parallel computation if available (currently single-threaded
        to avoid HDF5 parallel write issues)
    show_progress : bool, default=True
        Show progress bar during computation
    **metric_kwargs
        Additional arguments for the metric function

    Returns
    -------
    dict[str, dict[str, float]]
        Nested dictionary with structure {dataset_i: {dataset_j: distance}}.
        Includes diagonal (within-dataset) and off-diagonal (between-dataset)
        comparisons. Result is symmetric: result[i][j] == result[j][i].

    Raises
    ------
    ValueError
        If metric is not a scalar-returning metric

    Examples
    --------
    >>> import numpy as np
    >>> datasets = {
    ...     "A": np.random.randn(50, 10),
    ...     "B": np.random.randn(60, 10),
    ...     "C": np.random.randn(55, 10),
    ... }
    >>>
    >>> # Compute all pairwise comparisons
    >>> results = compute_all_pairs(datasets, metric="wasserstein")
    >>>
    >>> # Access specific comparisons
    >>> results["A"]["B"]  # Distance from A to B
    >>> results["A"]["A"]  # Within-dataset distance for A
    >>>
    >>> # Verify symmetry
    >>> results["A"]["B"] == results["B"]["A"]
    True
    >>>
    >>> # Number of comparisons (n datasets → n² comparisons)
    >>> len(datasets) ** 2 == sum(len(v) for v in results.values())
    True
    """
    # Validate metric-mode compatibility
    metric_normalized = _validate_metric_mode(metric, "all-pairs")

    n_datasets = len(datasets)
    n_comparisons = n_datasets**2

    # Procrustes requires all datasets to have same number of samples
    if metric_normalized == "procrustes":
        sample_counts = [data.shape[0] for data in datasets.values()]
        if len(set(sample_counts)) > 1:
            counts_dict = dict(zip(datasets.keys(), sample_counts))
            raise ValueError(
                f"Procrustes distance requires all datasets to have the same number "
                f"of samples. Got sample counts: {counts_dict}. "
                "Consider using 'one-to-one' or 'soft-matching' for datasets with "
                "different sample counts."
            )

    logger.info(
        f"Computing all-pairs comparisons: n_datasets={n_datasets}, "
        f"n_comparisons={n_comparisons}, metric='{metric_normalized}'"
    )

    # Import tqdm for progress tracking
    try:
        from tqdm.auto import tqdm

        has_tqdm = True
    except ImportError:
        has_tqdm = False
        logger.warning("tqdm not available, progress bar disabled")

    # Initialize result structure
    results: dict[str, dict[str, float]] = {name: {} for name in datasets}

    # Generate all dataset pairs (including self-comparisons for diagonal)
    pairs = [(name_i, name_j) for name_i in datasets for name_j in datasets]

    # Wrap iterator with progress bar if requested
    iterator = (
        tqdm(pairs, desc=f"Computing {metric_normalized} distances")
        if show_progress and has_tqdm
        else pairs
    )

    # Compute all comparisons
    for name_i, name_j in iterator:
        data_i = datasets[name_i]
        data_j = datasets[name_j]

        # For shape metrics, require same sample count for within-dataset
        if name_i == name_j and metric_normalized in SHAPE_METRICS:
            # Within-dataset comparison with shape metric
            # Shape metrics need two separate point clouds, so we split dataset
            n_samples = data_i.shape[0]
            if n_samples < 4:
                # Too few samples to split meaningfully
                logger.debug(
                    f"Dataset '{name_i}' has <4 samples, setting self-distance to 0"
                )
                results[name_i][name_j] = 0.0
                continue

            # Split into two equal-sized halves for self-comparison
            # Procrustes requires identical shapes, so both halves must have same count
            mid = n_samples // 2
            data_i_half1 = data_i[:mid]
            data_i_half2 = data_i[mid : 2 * mid]  # Take mid:2*mid to ensure equal size

            result = compute_pairwise_matrix(
                data_i_half1,
                data_i_half2,
                metric=metric_normalized,
                parallel=parallel,
                **metric_kwargs,
            )
            # Shape metrics return tuple
            dist = result[0] if isinstance(result, tuple) else float(result)
        else:
            # Between-dataset or within-dataset with distribution metric
            result = compute_pairwise_matrix(
                data_i,
                data_j,
                metric=metric_normalized,
                parallel=parallel,
                **metric_kwargs,
            )
            # Extract scalar from tuple if shape metric
            dist = result[0] if isinstance(result, tuple) else float(result)

        results[name_i][name_j] = dist

    logger.info(f"Completed {n_comparisons} comparisons for {n_datasets} datasets")

    return results


# ============================================================================
# Unified Orchestration API
# ============================================================================


@log_calls(level=logging.DEBUG, timeit=True)
def compare_datasets(
    data: npt.ArrayLike | dict[str, npt.ArrayLike],
    data2: npt.ArrayLike | None = None,
    *,
    mode: ComparisonMode | None = None,
    metric: AnyMetric = "euclidean",
    return_matrix: bool = False,
    save_path: str | Path | None = None,
    regenerate: bool = False,
    dataset_names: tuple[str, str] | None = None,
    **metric_kwargs: Any,
) -> float | npt.NDArray[np.floating] | dict[str, dict[str, float]] | dict[str, float] | BetweenResult | tuple[float, dict[tuple[int, int], float]]:
    """Unified API for dataset comparisons (orchestrates within/between/all-pairs).

    This function provides a single entry point for all comparison modes,
    automatically routing to the appropriate specialized function based on
    input structure and mode parameter.

    **Routing Logic**:
    - Single dataset (data) + mode="within" → compute_within_distances()
    - Two datasets (data, data2) + mode="between" → compute_between_distances()
    - Dict of datasets + mode="all-pairs" → compute_all_pairs()
    - Auto-detection if mode=None (inferred from inputs)

    Parameters
    ----------
    data : array-like or dict[str, array-like]
        Primary dataset or collection of datasets:
        - array-like (n_samples, n_features): Single dataset
        - dict: Multiple datasets for all-pairs comparison
    data2 : array-like, optional
        Second dataset for between-mode comparison.
        Shape: (n_samples_2, n_features). Must have same n_features as data.
    mode : {"within", "between", "all-pairs"}, optional
        Comparison mode. If None, auto-detected:
        - If data2 provided: "between"
        - If data is dict: "all-pairs"
        - Otherwise: "within"
    metric : str, default="euclidean"
        Distance/similarity metric. Valid values:
        **Point-to-point** (within, between modes):
          "euclidean", "manhattan", "cosine", "mahalanobis"
        **Distribution** (between, all-pairs modes):
          "wasserstein", "kolmogorov-smirnov", "jensen-shannon"
        **Shape** (between, all-pairs modes):
          "procrustes", "one-to-one", "soft-matching"
    return_matrix : bool, default=False
        For point-to-point metrics in between mode:
        - False: return mean distance (scalar)
        - True: return full distance matrix (n_samples_1, n_samples_2)
        Ignored for other modes.
    save_path : str or Path, optional
        Path to HDF5 file for automatic result caching.
        If provided and file exists with matching comparison:
        - regenerate=False: Load cached result (instant)
        - regenerate=True: Recompute and save (overwrite)
        If file doesn't exist or comparison not found: Compute and save.
    regenerate : bool, default=False
        Force recomputation even if cached result exists.
        Only used when save_path is provided.
        Set to True to:
        - Update results after data/metric changes
        - Recompute with different metric_kwargs
        - Force cache refresh
    dataset_names : tuple[str, str], optional
        Names for datasets (used in save_path storage).
        Required for mode="between" with save_path.
        Format: (dataset_i_name, dataset_j_name)
        Example: ("control", "treatment")
    **metric_kwargs
        Additional metric-specific arguments:
        - mahalanobis: mean, cov
        - shape metrics: reg, whiten, normalize

    Returns
    -------
    float, ndarray, or dict
        Return type depends on mode and metric:
        **mode="within"**:
          - float: mean pairwise distance
          - ndarray: full distance matrix if return_matrix=True
        **mode="between"**:
          - float: scalar distance (default for all metrics)
          - ndarray: distance matrix if return_matrix=True (point-to-point only)
          - tuple: (distance, pairs_dict) for shape metrics (procrustes, etc.)
        **mode="all-pairs"**:
          - dict[str, dict[str, float]]: {dataset_i: {dataset_j: distance}}

    Raises
    ------
    ValueError
        - If mode conflicts with input structure (e.g., mode="all-pairs"
          but data is array)
        - If metric incompatible with mode (e.g., "wasserstein" with
          mode="within")
    TypeError
        - If data types don't match expected structure

    Examples
    --------
    **Within-dataset comparison (self-similarity)**:

    >>> import numpy as np
    >>> data = np.random.randn(100, 10)
    >>>
    >>> # Mean within-dataset distance
    >>> dist = compare_datasets(data, mode="within", metric="euclidean")
    >>> print(f"Mean distance: {dist:.3f}")
    >>>
    >>> # Full distance matrix
    >>> matrix = compare_datasets(
    ...     data, mode="within", metric="euclidean", return_matrix=True
    ... )
    >>> print(f"Matrix shape: {matrix.shape}")  # (100, 100)

    **Between two datasets**:

    >>> data1 = np.random.randn(80, 10)
    >>> data2 = np.random.randn(120, 10) + 0.5  # Shifted distribution
    >>>
    >>> # Auto-detect mode (inferred as "between")
    >>> dist = compare_datasets(data1, data2, metric="wasserstein")
    >>> print(f"Wasserstein distance: {dist:.4f}")
    >>>
    >>> # Explicit mode
    >>> dist = compare_datasets(data1, data2, mode="between", metric="euclidean")
    >>>
    >>> # Get distance matrix instead of mean
    >>> matrix = compare_datasets(
    ...     data1, data2, mode="between", metric="euclidean", return_matrix=True
    ... )
    >>> print(f"Matrix shape: {matrix.shape}")  # (80, 120)

    **All-pairs comparison (multiple datasets)**:

    >>> datasets = {
    ...     "control": np.random.randn(50, 8),
    ...     "treatment_A": np.random.randn(50, 8) + 0.3,
    ...     "treatment_B": np.random.randn(50, 8) + 0.7,
    ... }
    >>>
    >>> # Compare all pairs with Wasserstein
    >>> results = compare_datasets(datasets, mode="all-pairs", metric="wasserstein")
    >>> print(results["control"]["treatment_A"])  # Distance control → treatment_A
    >>> # Distance treatment_A → control (symmetric)
    >>> print(results["treatment_A"]["control"]) 

    **Shape comparison**:

    >>> # Procrustes distance (requires equal sample counts)
    >>> data1 = np.random.randn(50, 8)
    >>> data2 = np.random.randn(50, 8)
    >>> dist = compare_datasets(data1, data2, metric="procrustes")
    >>> print(f"Procrustes distance: {dist:.4f}")

    Notes
    -----
    **Mode Selection Guidelines**:
    - Use **within** for: intra-group variability, cluster tightness, self-similarity
    - Use **between** for: two-sample comparisons, condition A vs B, pre/post treatment
    - Use **all-pairs** for: distance matrices, hierarchical clustering,
      group comparisons

    **Metric Selection**:
    - **Euclidean/Manhattan**: General-purpose, interpretable
    - **Cosine**: Direction similarity, scale-invariant
    - **Wasserstein**: Distribution shape, robust to outliers
    - **Procrustes**: Shape after optimal alignment

    **Performance**:
    - within/between modes: O(n²) for point-to-point metrics
    - all-pairs mode: O(k²) comparisons for k datasets
    - Parallel computation used where available (numba for distance matrices)

    See Also
    --------
    compute_within_distances : Direct access to within-mode
    compute_between_distances : Direct access to between-mode
    compute_all_pairs : Direct access to all-pairs mode
    compare_distribution_groups : Legacy API for group comparisons
    """
    # Auto-detect mode if not specified
    if mode is None:
        if isinstance(data, dict):
            mode = "all-pairs"
        elif data2 is not None:
            mode = "between"
        else:
            mode = "within"
        logger.debug(f"Auto-detected mode: '{mode}'")

    # Validate mode-input consistency
    if mode == "all-pairs" and not isinstance(data, dict):
        raise TypeError(
            f"mode='all-pairs' requires data to be dict[str, array], got {type(data)}"
        )
    if mode in {"within", "between"} and isinstance(data, dict):
        raise TypeError(
            f"mode='{mode}' requires data to be array-like, got dict. "
            "Use mode='all-pairs' for dict inputs."
        )
    if mode == "within" and data2 is not None:
        raise ValueError(
            "mode='within' does not accept data2 parameter. "
            "Use mode='between' to compare two datasets."
        )

    # Auto-save/load logic
    if save_path is not None:
        save_path_obj = Path(save_path)
        
        # Try to load cached result if regenerate=False
        if not regenerate and save_path_obj.exists():
            logger.info(f"Attempting to load cached result from {save_path}")
            try:
                from neural_analysis.utils.comparison_store import load_comparison
                
                # Determine dataset names for loading
                if mode == "between":
                    if dataset_names is None:
                        raise ValueError(
                            "dataset_names required for save_path with mode='between'. "
                            "Provide tuple like ('control', 'treatment')"
                        )
                    dataset_i, dataset_j = dataset_names
                    cached_result = load_comparison(
                        save_path_obj, metric, dataset_i, dataset_j
                    )
                    logger.info(
                        f"Successfully loaded cached result from {save_path}:"
                        f"{metric}/{dataset_i}___{dataset_j}"
                    )
                    return cached_result
                elif mode == "all-pairs":
                    # For all-pairs, we use a special dataset pair naming
                    cached_result = load_comparison(
                        save_path_obj, metric, "all_pairs", "all_pairs"
                    )
                    logger.info(
                        f"Successfully loaded cached all-pairs result from {save_path}"
                    )
                    return cached_result
                # Within mode doesn't support save_path (single dataset)
                    
            except (FileNotFoundError, KeyError) as e:
                logger.info(
                    f"Cache miss ({type(e).__name__}), computing result: {e}"
                )
                # Fall through to computation
        elif regenerate:
            logger.info(
                "regenerate=True, forcing recomputation (will overwrite cache)"
            )

    # Route to appropriate function
    logger.info(
        f"Routing comparison: mode='{mode}', metric='{metric}', "
        f"return_matrix={return_matrix}"
    )

    # Compute result
    result: float | npt.NDArray[np.floating] | dict[str, dict[str, float]] | dict[str, float] | BetweenResult | tuple[float, dict[tuple[int, int], float]]
    
    if mode == "within":
        result = compute_within_distances(
            data,
            metric=metric,
            return_matrix=return_matrix,
            **metric_kwargs,
        )
    elif mode == "between":
        if data2 is None:
            raise ValueError(
                "mode='between' requires data2 parameter. "
                "Provide second dataset or use mode='within'."
            )
        raw_between_result = compute_between_distances(
            data,
            data2,
            metric=metric,
            return_matrix=return_matrix,
            **metric_kwargs,
        )
        result = cast("float | npt.NDArray[np.floating] | dict[str, dict[str, float]] | dict[str, float] | tuple[float, dict[tuple[int, int], float]]",raw_between_result)
        # For between-mode with scalar result, return legacy dict for compatibility
        if not return_matrix and isinstance(result, (int, float, np.floating)):
            # Wrap scalar result into dict for compatibility
            result = cast(
                "BetweenResult",
                {
                    "value": float(result),
                    "metric": str(metric),
                },
            )
    elif mode == "all-pairs":
        if return_matrix:
            logger.warning(
                "return_matrix=True ignored for mode='all-pairs' (always returns dict)"
            )
        raw_all_pairs_result = compute_all_pairs(
            data,
            metric=metric,
            show_progress=metric_kwargs.pop("show_progress", True),
            **metric_kwargs,
        )
        result = cast(
            "float | npt.NDArray[np.floating] | dict[str, dict[str, float]] | dict[str, float] | tuple[float, dict[tuple[int, int], float]]",
            raw_all_pairs_result,
        )
    else:
        raise ValueError(
            f"Unknown mode '{mode}'. Must be 'within', 'between', or 'all-pairs'."
        )
    
    # Auto-save result if save_path provided
    if save_path is not None:
        # Validate dataset_names before saving (for between mode)
        if mode == "between" and dataset_names is None:
            raise ValueError(
                "dataset_names required for save_path with mode='between'. "
                "Provide tuple like ('control', 'treatment')"
            )
        
        logger.info(f"Saving result to {save_path}")
        from neural_analysis.utils.comparison_store import save_comparison
        
        try:
            if mode == "between":
                if dataset_names is None:
                    raise ValueError("dataset_names required for between mode")
                dataset_i, dataset_j = dataset_names
                
                # Handle dict return from compute_between_distances
                save_val_typed: float | npt.NDArray[np.floating] | dict[str, dict[str, float]]
                if isinstance(result, dict) and "value" in result:
                    save_val_typed = float(cast("float", result["value"]))
                else:
                    save_val_typed = cast(
                        "float | npt.NDArray[np.floating] | dict[str, dict[str, float]]",
                        result,
                    )
                # For all-pairs, ensure result is a dict[str, dict[str, float]]
                save_val_all_pairs = cast(
                    "dict[str, dict[str, float]]",
                    result,
                )
                save_comparison(
                    filepath=save_path,
                    metric=metric,
                    dataset_i=dataset_i,
                    dataset_j=dataset_j,
                    mode=mode,
                    value=save_val_typed,
                    metadata={
                        "return_matrix": return_matrix,
                        **metric_kwargs,
                    },
                    overwrite=regenerate,
                )
                logger.info(
                    f"Saved between-mode result: {metric}/{dataset_i}___{dataset_j}"
                )
            elif mode == "all-pairs":
                # Determine number of datasets for all-pairs
                n_datasets = len(result) if isinstance(result, dict) else 1
                save_comparison(
                    filepath=save_path,
                    metric=metric,
                    dataset_i="all_pairs",
                    dataset_j="all_pairs",
                    mode=mode,
                    value=save_val_all_pairs,
                    metadata={"n_datasets": n_datasets, **metric_kwargs},
                    overwrite=regenerate,
                )
                logger.info(f"Saved all-pairs result: {metric}/all_pairs")
            # Within mode: no save (single dataset, less useful to cache)
        except Exception as e:
            logger.warning(f"Failed to save result: {e}")
            # Continue anyway, return computed result
    
    return result


# ---------------------------------------------------------------------------
# Correlation and similarity functions (from similarity.py)
# ---------------------------------------------------------------------------


def correlation(
    data: npt.ArrayLike,
    method: Literal["pearson", "spearman", "kendall"] = "pearson",
    mode: Literal["matrix", "pairwise"] = "matrix",
    parallel: bool = False,
) -> npt.NDArray[Any]:
    data_arr = np.asarray(data)
    if mode == "matrix":
        if parallel and NUMBA_AVAILABLE:
            return _correlation_matrix_parallel(data_arr, method)
        else:
            return correlation_matrix(data_arr, method=method)
    elif mode == "pairwise":
        n_features = data_arr.shape[1]
        pairwise_corrs = np.zeros(n_features - 1)
        for i in range(n_features - 1):
            corr_matrix = correlation_matrix(data_arr[:, [i, i + 1]], method=method)
            pairwise_corrs[i] = corr_matrix[0, 1]
        return pairwise_corrs
    else:
        raise ValueError(f"Unknown mode '{mode}'")


def correlation_matrix(
    data: npt.ArrayLike,
    method: Literal["pearson", "spearman", "kendall"] = "pearson",
) -> npt.NDArray[Any]:
    data_arr = np.asarray(data)
    if data_arr.ndim != 2:
        raise ValueError(f"data must be 2D, got shape {data_arr.shape}")

    logger.info(
        f"Computing {method} correlation matrix for {data_arr.shape[1]} features"
    )

    match method:
        case "pearson":
            corr_matrix = np.corrcoef(data_arr.T)
        case "spearman":
            corr_matrix, _ = spearmanr(data_arr, axis=0)
            if data_arr.shape[1] == 2 and np.ndim(corr_matrix) == 0:
                corr_matrix = np.array([[1.0, corr_matrix], [corr_matrix, 1.0]])
        case "kendall":
            n_features = data_arr.shape[1]
            corr_matrix = np.eye(n_features)
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    tau, _ = kendalltau(data_arr[:, i], data_arr[:, j])
                    corr_matrix[i, j] = tau
                    corr_matrix[j, i] = tau
        case _:
            raise ValueError(f"Unknown method '{method}'")

    return corr_matrix


def cosine_similarity_matrix(
    data: npt.ArrayLike,
    centered: bool = False,
) -> npt.NDArray[Any]:
    data_arr = np.asarray(data)
    if data_arr.ndim != 2:
        raise ValueError(f"data must be 2D, got shape {data_arr.shape}")

    if centered:
        data_arr = data_arr - data_arr.mean(axis=0, keepdims=True)

    norms = np.linalg.norm(data_arr, axis=0, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    data_normalized = data_arr / norms
    similarity_matrix: npt.NDArray[Any] = cast(
        "npt.NDArray[Any]", data_normalized.T @ data_normalized
    )
    return similarity_matrix


def angular_similarity_matrix(data: npt.ArrayLike) -> npt.NDArray[Any]:
    data_arr = np.asarray(data)
    if data_arr.ndim != 2:
        raise ValueError(f"data must be 2D, got shape {data_arr.shape}")
    cosine_sim = cosine_similarity_matrix(data_arr, centered=False)
    cosine_sim = np.clip(cosine_sim, -1.0, 1.0)
    angular_distance = np.arccos(cosine_sim)
    angular_similarity: npt.NDArray[Any] = cast(
        "npt.NDArray[Any]", 1.0 - (angular_distance / np.pi)
    )
    return angular_similarity


def similarity_matrix(
    data: npt.ArrayLike,
    method: Literal["pearson", "spearman", "kendall", "cosine", "angular"] = "pearson",
    centered: bool = False,
    parallel: bool = False,
    plot: bool = False,
    plot_config: dict[str, Any] | None = None,
) -> npt.NDArray[Any]:
    data_arr = np.asarray(data)
    if data_arr.ndim != 2:
        raise ValueError(f"data must be 2D, got shape {data_arr.shape}")

    logger.info(
        f"Computing {method} similarity matrix for {data_arr.shape[1]} features "
        f"(parallel={parallel})"
    )

    match method:
        case "pearson" | "spearman" | "kendall":
            if parallel:
                similarity = _correlation_matrix_parallel(data_arr, method)
            else:
                similarity = correlation_matrix(data_arr, method=method)
        case "cosine":
            if parallel:
                similarity = _cosine_similarity_matrix_parallel(data_arr, centered)
            else:
                similarity = cosine_similarity_matrix(data_arr, centered=centered)
        case "angular":
            if parallel:
                similarity = _angular_similarity_matrix_parallel(data_arr)
            else:
                similarity = angular_similarity_matrix(data_arr)
        case _:
            raise ValueError(f"Unknown method '{method}'")

    if plot:
        _plot_similarity_matrix(similarity, method, plot_config)

    return similarity


def _plot_similarity_matrix(
    similarity: npt.NDArray[Any], method: str, plot_config: dict[str, Any] | None
) -> None:
    try:
        from neural_analysis.plotting import PlotConfig, plot_heatmap
    except Exception:
        logger.warning("Plotting unavailable")
        return

    config_dict = {
        "title": f"{method.capitalize()} Similarity Matrix",
        "figsize": (8, 7),
        "cmap": "RdBu_r" if method in ["pearson", "spearman", "kendall"] else "viridis",
    }
    if plot_config is not None:
        config_dict.update(plot_config)

    config = PlotConfig(**config_dict)  # type: ignore[arg-type]
    plot_heatmap(
        similarity,
        config=config,
        show_values=similarity.shape[0] <= 15,
        colorbar=True,
        colorbar_label="Similarity",
    )


# ---------------------------------------------------------------------------
# Parallel correlation/similarity helpers and numba-accelerated paths
# ---------------------------------------------------------------------------


def _correlation_matrix_parallel(
    data: npt.NDArray[Any], method: str
) -> npt.NDArray[Any]:
    if method == "pearson":
        return np.corrcoef(data.T)
    elif method == "spearman":
        if NUMBA_AVAILABLE:
            return _spearman_numba(data)
        else:
            logger.debug("Numba not available, using scipy for Spearman")
            return correlation_matrix(data, method="spearman")
    elif method == "kendall":
        if NUMBA_AVAILABLE:
            return _kendall_numba(data)
        else:
            logger.debug("Numba not available, using scipy for Kendall")
            return correlation_matrix(data, method="kendall")
    else:
        raise ValueError(f"Unknown method: {method}")


def _cosine_similarity_matrix_parallel(
    data: npt.NDArray[Any], centered: bool
) -> npt.NDArray[Any]:
    if centered:
        data = data - data.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(data, axis=0, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    data_normalized = data / norms
    result: npt.NDArray[Any] = cast(
        "npt.NDArray[Any]", data_normalized.T @ data_normalized
    )
    return result


def _angular_similarity_matrix_parallel(data: npt.NDArray[Any]) -> npt.NDArray[Any]:
    cosine_sim = _cosine_similarity_matrix_parallel(data, centered=False)
    cosine_sim = np.clip(cosine_sim, -1.0, 1.0)
    angular_distance = np.arccos(cosine_sim)
    result: npt.NDArray[Any] = cast(
        "npt.NDArray[Any]", 1.0 - (angular_distance / np.pi)
    )
    return result


# Numba accelerated ranking and kendall/spearman if available (from similarity.py)
if NUMBA_AVAILABLE:

    @numba.jit(nopython=True, parallel=True, cache=True)  # type: ignore[misc]
    def _rank_data_numba(data: npt.NDArray[Any]) -> npt.NDArray[Any]:
        n_samples, n_features = data.shape
        ranks = np.empty_like(data)
        for j in numba.prange(n_features):
            sorted_idx = np.argsort(data[:, j])
            for i, idx in enumerate(sorted_idx):
                ranks[idx, j] = i + 1
        return ranks

    @numba.jit(nopython=True, parallel=True, cache=True)  # type: ignore[misc]
    def _kendall_tau_pairwise(x: npt.NDArray[Any], y: npt.NDArray[Any]) -> float:
        n = len(x)
        concordant = 0
        discordant = 0
        for i in range(n):
            for j in range(i + 1, n):
                sign_x = np.sign(x[j] - x[i])
                sign_y = np.sign(y[j] - y[i])
                prod = sign_x * sign_y
                if prod > 0:
                    concordant += 1
                elif prod < 0:
                    discordant += 1
        total_pairs = n * (n - 1) / 2
        return (concordant - discordant) / total_pairs

    def _spearman_numba(data: npt.NDArray[Any]) -> npt.NDArray[Any]:
        ranks = _rank_data_numba(data)
        return np.corrcoef(ranks.T)

    def _kendall_numba(data: npt.NDArray[Any]) -> npt.NDArray[Any]:
        n_features = data.shape[1]
        corr_matrix = np.eye(n_features)
        for i in numba.prange(n_features):
            for j in range(i + 1, n_features):
                tau = _kendall_tau_pairwise(data[:, i], data[:, j])
                corr_matrix[i, j] = tau
                corr_matrix[j, i] = tau
        return corr_matrix
else:

    def _spearman_numba(data: npt.NDArray[Any]) -> npt.NDArray[Any]:
        corr_matrix, _ = spearmanr(data, axis=0)
        if data.shape[1] == 2 and np.ndim(corr_matrix) == 0:
            corr_matrix = np.array([[1.0, corr_matrix], [corr_matrix, 1.0]])
        return cast("npt.NDArray[Any]", corr_matrix)

    def _kendall_numba(data: npt.NDArray[Any]) -> npt.NDArray[Any]:
        return correlation_matrix(data, method="kendall")


# ---------------------------------------------------------------------------
# Spatial autocorrelation functions (copied from similarity.py)
# ---------------------------------------------------------------------------


def spatial_autocorrelation(
    activity: npt.ArrayLike,
    positions: npt.ArrayLike,
    arena_size: float | tuple[float, ...],
    n_bins: int | None = None,
    n_cells_to_average: int = 10,
    method: Literal["fft", "direct"] = "fft",
) -> tuple[npt.NDArray[np.float64], list[npt.NDArray[np.float64]]]:
    activity_arr = np.asarray(activity)
    positions_arr = np.asarray(positions)
    if positions_arr.ndim == 1:
        n_dims = 1
        positions_arr = positions_arr.reshape(-1, 1)
    else:
        n_dims = positions_arr.shape[1]

    if n_bins is None:
        n_bins = {1: 50, 2: 40, 3: 20}[n_dims]

    logger.info(f"Computing {n_dims}D spatial autocorrelation with {n_bins} bins")

    if n_dims == 1:
        return _compute_1d_autocorrelation(
            activity_arr,
            positions_arr,
            arena_size,
            n_bins=n_bins,
            n_cells_to_average=n_cells_to_average,
            method=method,
        )
    elif n_dims == 2:
        return _compute_2d_autocorrelation(
            activity_arr,
            positions_arr,
            arena_size,
            n_bins=n_bins,
            n_cells_to_average=n_cells_to_average,
            method=method,
        )
    elif n_dims == 3:
        return _compute_3d_autocorrelation(
            activity_arr,
            positions_arr,
            arena_size,
            n_bins=n_bins,
            n_cells_to_average=n_cells_to_average,
            method=method,
        )
    else:
        raise ValueError(f"Unsupported dimensionality: {n_dims}")


def _compute_1d_autocorrelation(
    activity: npt.NDArray[np.float64],
    positions: npt.NDArray[np.float64],
    arena_size: float | tuple[float, ...],
    n_bins: int = 50,
    n_cells_to_average: int = 10,
    method: Literal["fft", "direct"] = "fft",
) -> tuple[npt.NDArray[np.float64], list[npt.NDArray[np.float64]]]:
    from neural_analysis.plotting.synthetic_plots import _compute_spatial_bins_1d

    x_max = arena_size[0] if isinstance(arena_size, tuple) else arena_size

    n_cells = min(n_cells_to_average, activity.shape[1])
    autocorr_sum: npt.NDArray[np.float64] | None = None

    for cell_idx in range(n_cells):
        _, firing_rates = _compute_spatial_bins_1d(
            positions, activity, arena_size, n_bins=n_bins, cell_idx=cell_idx
        )
        firing_centered = firing_rates - np.mean(firing_rates)
        if method == "fft":
            fft_1d = np.fft.fft(firing_centered, n=2 * n_bins)
            power_spectrum = np.abs(fft_1d) ** 2
            autocorr = np.fft.ifft(power_spectrum).real[:n_bins]
            autocorr = np.fft.fftshift(autocorr)
        else:
            autocorr = np.correlate(firing_centered, firing_centered, mode="same")

        if autocorr_sum is None:
            autocorr_sum = autocorr.copy()
        else:
            autocorr_sum += autocorr

    if autocorr_sum is not None:
        autocorr_avg = autocorr_sum / n_cells
    else:
        autocorr_avg = np.zeros(n_bins)

    center_idx = n_bins // 2
    if autocorr_avg[center_idx] != 0:
        autocorr_normalized = autocorr_avg / autocorr_avg[center_idx]
    else:
        autocorr_normalized = autocorr_avg

    lags = np.linspace(-x_max, x_max, n_bins)
    return autocorr_normalized, [cast("npt.NDArray[np.float64]", lags)]


def _compute_2d_autocorrelation(
    activity: npt.NDArray[np.float64],
    positions: npt.NDArray[np.float64],
    arena_size: float | tuple[float, ...],
    n_bins: int = 40,
    n_cells_to_average: int = 10,
    method: Literal["fft", "direct"] = "fft",
) -> tuple[npt.NDArray[np.float64], list[npt.NDArray[np.float64]]]:
    from neural_analysis.plotting.synthetic_plots import _compute_spatial_bins_2d

    if isinstance(arena_size, tuple):
        x_max, y_max = arena_size
    else:
        x_max = y_max = arena_size

    n_cells = min(n_cells_to_average, activity.shape[1])
    autocorr_sum: npt.NDArray[np.float64] | None = None

    for cell_idx in range(n_cells):
        _, _, firing_map = _compute_spatial_bins_2d(
            positions,
            activity,
            cast("tuple[float, ...]", arena_size),
            n_bins=n_bins,
            cell_idx=cell_idx,
        )
        firing_map_centered = firing_map - np.nanmean(firing_map)
        firing_map_centered = np.nan_to_num(firing_map_centered, nan=0.0)

        if method == "fft":
            fft_2d = np.fft.fft2(firing_map_centered)
            power_spectrum = np.abs(fft_2d) ** 2
            autocorr = np.fft.ifft2(power_spectrum).real
            autocorr = np.fft.fftshift(autocorr)
        else:
            from scipy.signal import correlate2d

            autocorr = correlate2d(
                firing_map_centered, firing_map_centered, mode="same"
            )

        if autocorr_sum is None:
            autocorr_sum = autocorr.copy()
        else:
            autocorr_sum += autocorr

    if autocorr_sum is not None:
        autocorr_avg = autocorr_sum / n_cells
    else:
        autocorr_avg = np.zeros((n_bins, n_bins))

    center_idx = (autocorr_avg.shape[0] // 2, autocorr_avg.shape[1] // 2)
    if autocorr_avg[center_idx] != 0:
        autocorr_normalized = autocorr_avg / autocorr_avg[center_idx]
    else:
        autocorr_normalized = autocorr_avg

    x_lags = np.linspace(-x_max, x_max, n_bins)
    y_lags = np.linspace(-y_max, y_max, n_bins)
    return autocorr_normalized, [
        cast("npt.NDArray[np.float64]", x_lags),
        cast("npt.NDArray[np.float64]", y_lags),
    ]


def _compute_3d_autocorrelation(
    activity: npt.NDArray[np.float64],
    positions: npt.NDArray[np.float64],
    arena_size: float | tuple[float, ...],
    n_bins: int = 20,
    n_cells_to_average: int = 10,
    method: Literal["fft", "direct"] = "fft",
) -> tuple[npt.NDArray[np.float64], list[npt.NDArray[np.float64]]]:
    from neural_analysis.plotting.synthetic_plots import _compute_spatial_bins_3d

    if isinstance(arena_size, tuple):
        x_max, y_max, z_max = arena_size
    else:
        x_max = y_max = z_max = arena_size

    n_cells = min(n_cells_to_average, activity.shape[1])
    autocorr_sum: npt.NDArray[np.float64] | None = None

    for cell_idx in range(n_cells):
        _, _, _, firing_volume = _compute_spatial_bins_3d(
            positions,
            activity,
            cast("tuple[float, ...]", arena_size),
            n_bins=n_bins,
            cell_idx=cell_idx,
        )
        firing_volume_centered = firing_volume - np.nanmean(firing_volume)
        firing_volume_centered = np.nan_to_num(firing_volume_centered, nan=0.0)

        if method == "fft":
            fft_3d = np.fft.fftn(firing_volume_centered)
            power_spectrum = np.abs(fft_3d) ** 2
            autocorr = np.fft.ifftn(power_spectrum).real
            autocorr = np.fft.fftshift(autocorr)
        else:
            from scipy.signal import correlate

            autocorr = correlate(
                firing_volume_centered, firing_volume_centered, mode="same"
            )

        if autocorr_sum is None:
            autocorr_sum = autocorr.copy()
        else:
            autocorr_sum += autocorr

    if autocorr_sum is not None:
        autocorr_avg = autocorr_sum / n_cells
    else:
        autocorr_avg = np.zeros((n_bins, n_bins, n_bins))

    center_idx = tuple(s // 2 for s in autocorr_avg.shape)
    if autocorr_avg[center_idx] != 0:
        autocorr_normalized = autocorr_avg / autocorr_avg[center_idx]
    else:
        autocorr_normalized = autocorr_avg

    x_lags = np.linspace(-x_max, x_max, n_bins)
    y_lags = np.linspace(-y_max, y_max, n_bins)
    z_lags = np.linspace(-z_max, z_max, n_bins)

    return autocorr_normalized, [
        cast("npt.NDArray[np.float64]", x_lags),
        cast("npt.NDArray[np.float64]", y_lags),
        cast("npt.NDArray[np.float64]", z_lags),
    ]
