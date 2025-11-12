"""Machine learning and decoding methods for neural analysis.

This module provides functions for decoding neural activity patterns,
including cross-validated decoding and ensemble predictions.
"""

from .decoding import (
    compare_highd_lowd_decoding,
    cross_validated_knn_decoder,
    evaluate_decoder,
    knn_decoder,
    population_vector_decoder,
)

__all__ = [
    "compare_highd_lowd_decoding",
    "cross_validated_knn_decoder",
    "evaluate_decoder",
    "knn_decoder",
    "population_vector_decoder",
]
