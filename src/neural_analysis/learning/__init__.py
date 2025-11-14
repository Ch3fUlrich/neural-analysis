"""Machine learning and decoding methods for neural analysis.

This module provides functions for decoding neural activity patterns,
including cross-validated decoding and ensemble predictions, as well as
classification and clustering methods for cell type identification.
"""

from .classification import (
    SupervisedMethod,
    UnsupervisedMethod,
    classify_cells,
    cluster_cells,
    compare_classifiers,
    compare_clusterers,
    evaluate_classifier,
    evaluate_clustering,
    extract_cell_features,
    fit_clusterer,
    train_classifier,
)
from .decoding import (
    compare_highd_lowd_decoding,
    cross_validated_knn_decoder,
    evaluate_decoder,
    knn_decoder,
    population_vector_decoder,
)

__all__ = [
    # Decoding
    "compare_highd_lowd_decoding",
    "cross_validated_knn_decoder",
    "evaluate_decoder",
    "knn_decoder",
    "population_vector_decoder",
    # Classification
    "SupervisedMethod",
    "UnsupervisedMethod",
    "classify_cells",
    "cluster_cells",
    "compare_classifiers",
    "compare_clusterers",
    "evaluate_classifier",
    "evaluate_clustering",
    "extract_cell_features",
    "fit_clusterer",
    "train_classifier",
]
