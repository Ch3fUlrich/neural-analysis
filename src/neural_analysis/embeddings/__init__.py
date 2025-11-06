"""
Dimensionality reduction and embedding methods.

This module provides functions for computing various dimensionality reduction
embeddings (PCA, UMAP, TSNE, MDS, etc.). Visualization is handled through the
PlotGrid system (neural_analysis.plotting).
"""

from .dimensionality_reduction import (
    compute_embedding,
    compute_multiple_embeddings,
    pca_explained_variance,
)

__all__ = [
    "compute_embedding",
    "compute_multiple_embeddings",
    "pca_explained_variance",
]
