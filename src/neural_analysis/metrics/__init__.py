"""Metrics subpackage for neural_analysis.

This package provides distance metrics, distribution comparison tools,
similarity measures, and outlier detection utilities for neural data analysis.
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
    # Similarity measures
    "correlation_matrix",
    "cosine_similarity_matrix",
    "angular_similarity_matrix",
    "similarity_matrix",
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
    if name in ("compare_distributions", "compare_distribution_groups"):
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
    if name == "filter_outlier":
        mod = importlib.import_module("neural_analysis.metrics.outliers")
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
