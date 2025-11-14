"""Metrics subpackage for neural_analysis.

This package provides distance metrics, distribution comparison tools,
similarity measures, shape similarity, and outlier detection utilities
for neural data analysis.
"""

from typing import Any

__all__ = [
    # Distance metrics (Phase 1-2 API)
    "euclidean_distance",
    "mahalanobis_distance",
    "cosine_similarity",
    "pairwise_distance",
    "distribution_distance",
    # Phase 3: New explicit comparison functions
    "compute_within_distances",
    "compute_between_distances",
    "compute_all_pairs",
    # Metric category constants (Phase 3)
    "POINT_TO_POINT_METRICS",
    "DISTRIBUTION_METRICS",
    "SHAPE_METRICS",
    "SCALAR_METRICS",
    "ALL_METRICS",
    # Distribution comparison
    "compare_distributions",
    "compare_distribution_groups",
    "pairwise_distribution_comparison_batch",
    "batch_comparison",
    # Similarity measures
    "correlation_matrix",
    "cosine_similarity_matrix",
    "angular_similarity_matrix",
    "similarity_matrix",
    # Shape similarity (now in distributions.py)
    "shape_distance",
    "shape_distance_procrustes",
    "shape_distance_one_to_one",
    "shape_distance_soft_matching",
    "modify_matrix",
    "align_matrix",
    # Outlier detection
    "filter_outlier",
]


def __getattr__(name: str) -> Any:
    """Lazy import for metrics modules."""
    import importlib

    if name in (
        "euclidean_distance",
        "mahalanobis_distance",
        "cosine_similarity",
        "pairwise_distance",
        "distribution_distance",
        # Phase 3: New functions
        "compute_within_distances",
        "compute_between_distances",
        "compute_all_pairs",
        # Phase 3: Metric categories
        "POINT_TO_POINT_METRICS",
        "DISTRIBUTION_METRICS",
        "SHAPE_METRICS",
        "SCALAR_METRICS",
        "ALL_METRICS",
    ):
        mod = importlib.import_module("neural_analysis.metrics.pairwise_metrics")
        return getattr(mod, name)
    if name in (
        "compare_distributions",
        "compare_distribution_groups",
        "batch_comparison",
    ):
        mod = importlib.import_module("neural_analysis.metrics.distributions")
        return getattr(mod, name)
    if name in (
        "correlation_matrix",
        "cosine_similarity_matrix",
        "angular_similarity_matrix",
        "similarity_matrix",
    ):
        mod = importlib.import_module("neural_analysis.metrics.pairwise_metrics")
        return getattr(mod, name)
    if name in (
        "shape_distance",
        "pairwise_distribution_comparison_batch",
        "modify_matrix",
        "align_matrix",
        "shape_distance_procrustes",
        "shape_distance_one_to_one",
        "shape_distance_soft_matching",
    ):
        mod = importlib.import_module("neural_analysis.metrics.distributions")
        return getattr(mod, name)
    if name == "filter_outlier":
        mod = importlib.import_module("neural_analysis.metrics.outliers")
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
