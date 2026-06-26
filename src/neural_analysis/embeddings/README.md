# Embeddings

This module is responsible for dimensionality reduction techniques.

## Purpose

Neural data is often high-dimensional. Embedding methods project this data into lower-dimensional spaces (typically 2D or 3D) while preserving specific structures (local neighborhoods, global pairwise distances, or topological features), facilitating easier visualization and downstream processing.

## What can be analyzed

- **Dimensionality Reduction**: Apply standard algorithms like PCA, t-SNE, UMAP, MDS, Isomap, LLE, and Spectral Embedding to neural firing rate matrices.
- **Visualization Helpers**: Tools specifically designed to assist in plotting these lower-dimensional representations.
- **Manifold Structure**: Analyze the intrinsic geometry of neural activity. For example, determine if place cells naturally form a 2D sheet or if head direction cells form a ring topology in the latent space.
