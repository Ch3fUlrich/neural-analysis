"""Unified pairwise metrics: distances, similarities, correlations, autocorrelation

Facade module that re-exports from pairwise_core (non-numba functions)
and pairwise_numba (numba-accelerated functions).
"""

from __future__ import annotations

# Re-export all public and private names from both sub-modules so that
# existing imports like ``from neural_analysis.metrics.pairwise_metrics import X``
# continue to work unchanged.
from .pairwise_core import (  # type: ignore[attr-defined]  # noqa: F401
    ALL_METRICS,
    DISTRIBUTION_METRICS,
    POINT_TO_POINT_METRICS,
    SCALAR_METRICS,
    SHAPE_METRICS,
    AnyMetric,
    BetweenResult,
    ComparisonMode,
    DistanceMetric,
    DistributionMetric,
    ShapeMetric,
    _angular_similarity_matrix_parallel,
    _compute_1d_autocorrelation,
    _compute_2d_autocorrelation,
    _compute_3d_autocorrelation,
    _cosine_similarity_matrix_parallel,
    _get_distance_function,
    _plot_similarity_matrix,
    _validate_metric_mode,
    _validate_pairwise_inputs,
    angular_similarity_matrix,
    compare_datasets,
    compute_all_pairs,
    compute_between_distances,
    compute_pairwise_matrix,
    compute_within_distances,
    correlation,
    correlation_matrix,
    cosine_similarity,
    cosine_similarity_matrix,
    euclidean_distance,
    get_logger,
    log_calls,
    logger,
    mahalanobis_distance,
    manhattan_distance,
    pairwise_distance,
    similarity_matrix,
    spatial_autocorrelation,
)
from .pairwise_numba import (  # noqa: F401
    NUMBA_AVAILABLE,
    _correlation_matrix_parallel,
    _kendall_numba,
    _spearman_numba,
)

if NUMBA_AVAILABLE:
    from .pairwise_numba import (  # noqa: F401
        _pairwise_cosine_numba,
        _pairwise_euclidean_numba,
        _pairwise_manhattan_numba,
    )

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
