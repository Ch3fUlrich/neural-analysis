"""Metrics subpackage for neural_analysis.

This package provides distance metrics, distribution comparison tools,
similarity measures, shape similarity, and outlier detection utilities
for neural data analysis.
"""

__all__ = [
    # Distance metrics
    "euclidean_distance",
    "mahalanobis_distance",
    "cosine_similarity",
    "pairwise_distance",
    "distribution_distance",
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


def __getattr__(name: str):
    """Lazy import for metrics modules."""
    import importlib

    if name in (
        "euclidean_distance",
        "mahalanobis_distance",
        "cosine_similarity",
        "pairwise_distance",
        "distribution_distance",
    ):
        mod = importlib.import_module("neural_analysis.metrics.distance")
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
        mod = importlib.import_module("neural_analysis.metrics.similarity")
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
