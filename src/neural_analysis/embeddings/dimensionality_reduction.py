"""
Dimensionality reduction and embedding methods for neural data analysis.

This module provides functions for computing various dimensionality reduction
embeddings commonly used in neuroscience, including PCA, UMAP, t-SNE, MDS,
Isomap, LLE, and Spectral Embedding.

The module supports:
- Consistent API across all methods
- 2D and 3D embeddings
- Integration with visualization (PlotGrid system)
- Connection to distribution comparison metrics

Functions are organized by:
- Core embedding computation
- Multi-method comparison
- Variance analysis (PCA)
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import numpy.typing as npt
from sklearn.decomposition import PCA
from sklearn.manifold import (
    MDS,
    TSNE,
    Isomap,
    LocallyLinearEmbedding,
    SpectralEmbedding,
)

# Optional UMAP support
try:
    from umap.umap_ import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    UMAP = None

logger = logging.getLogger(__name__)

__all__ = [
    "compute_embedding",
    "compute_multiple_embeddings",
    "pca_explained_variance",
]


EmbeddingMethod = Literal[
    "pca",
    "umap",
    "tsne",
    "mds",
    "isomap",
    "lle",
    "spectral",
]


def compute_embedding(
    data: npt.ArrayLike,
    method: EmbeddingMethod = "pca",
    n_components: int = 2,
    metric: str = "euclidean",
    n_neighbors: int = 15,
    random_state: int | None = 42,
    **kwargs
) -> np.ndarray:
    """
    Compute dimensionality reduction embedding using specified method.
    
    This function provides a unified interface for various dimensionality
    reduction techniques commonly used in neural data analysis. All methods
    support both 2D and 3D embeddings.
    
    Parameters
    ----------
    data : array-like, shape (n_samples, n_features)
        Input data matrix. Can be: (1) Feature matrix: (n_samples, n_features) 
        for methods in feature space, or (2) Distance matrix: (n_samples, 
        n_samples) for metric='precomputed' (MDS, Isomap, spectral).
    method : {"pca", "umap", "tsne", "mds", "isomap", "lle", "spectral"}, default="pca"
        Dimensionality reduction method. Options: "pca" (Principal Component 
        Analysis, linear, fast), "umap" (Uniform Manifold Approximation and 
        Projection, non-linear, preserves local + global), "tsne" (t-Distributed 
        Stochastic Neighbor Embedding, non-linear, preserves local), "mds" 
        (Multidimensional Scaling, preserves pairwise distances), "isomap" 
        (Isometric Mapping, non-linear, geodesic distances), "lle" (Locally 
        Linear Embedding, non-linear, preserves local geometry), "spectral" 
        (Spectral Embedding, graph-based, non-linear).
    n_components : int, default=2
        Number of dimensions for embedding (typically 2 or 3 for visualization).
    metric : str, default="euclidean"
        Distance metric for methods that use distances. Options: "euclidean" 
        (Euclidean distance), "cosine" (cosine distance), "precomputed" (use 
        data as precomputed distance matrix, MDS and Isomap only). Ignored 
        for PCA which works in feature space.
    n_neighbors : int, default=15
        Number of neighbors for methods that use local neighborhood (UMAP, 
        Isomap, LLE, Spectral). Controls tradeoff between local and global 
        structure. Typical range: 5-50.
    random_state : int, optional, default=42
        Random seed for reproducibility (used by UMAP, t-SNE, Spectral). 
        Set to None for non-deterministic results.
    **kwargs
        Additional method-specific parameters. For PCA: svd_solver 
        ("auto"|"full"|"arpack"|"randomized"), whiten (bool). For UMAP: 
        min_dist (float, default=0.1), spread (float, default=1.0). For 
        t-SNE: perplexity (float, default=30.0), learning_rate (float, 
        default=200.0), max_iter (int, default=1000). For MDS: dissimilarity 
        ("euclidean"|"precomputed"), n_init (int, default=1). For Isomap: 
        path_method ("auto"|"FW"|"D"). For LLE: reg (float, default=0.001), 
        eigen_solver ("auto"|"arpack"|"dense"). For Spectral: affinity 
        ("nearest_neighbors"|"rbf"|"precomputed"), gamma (float, optional).
    
    Returns
    -------
    embedding : ndarray, shape (n_samples, n_components)
        Low-dimensional embedding of the input data
    
    Raises
    ------
    ValueError
        If method is not recognized, n_components is invalid, or data
        shape is incompatible
    ImportError
        If UMAP is requested but not installed
    
    Examples
    --------
    >>> import numpy as np
    >>> from neural_analysis.embeddings import compute_embedding
    >>> 
    >>> # Generate high-dimensional neural data
    >>> rng = np.random.default_rng(42)
    >>> neural_data = rng.normal(0, 1, (200, 50))  # 200 samples, 50 neurons
    >>> 
    >>> # Compute PCA (fast, linear)
    >>> pca_2d = compute_embedding(neural_data, method='pca', n_components=2)
    >>> print(f"PCA shape: {pca_2d.shape}")  # (200, 2)
    >>> 
    >>> # Compute UMAP (slower, non-linear, preserves structure)
    >>> umap_2d = compute_embedding(neural_data, method='umap', n_components=2,
    ...                             n_neighbors=15, min_dist=0.1)
    >>> 
    >>> # Compute t-SNE (good for visualization)
    >>> tsne_2d = compute_embedding(neural_data, method='tsne', n_components=2,
    ...                             perplexity=30.0)
    >>> 
    >>> # Use precomputed distance matrix with MDS
    >>> from scipy.spatial.distance import pdist, squareform
    >>> distances = squareform(pdist(neural_data, metric='euclidean'))
    >>> mds_2d = compute_embedding(distances, method='mds', n_components=2,
    ...                           metric='precomputed')
    
    Notes
    -----
    **Method Selection Guide**:
    
    - **PCA**: Use for linear relationships, quick exploratory analysis, or
      when you need to explain variance. Fast and deterministic.
    
    - **UMAP**: Best general-purpose non-linear method. Preserves both local
      and global structure. Good for clustering visualization. Faster than
      t-SNE.
    
    - **t-SNE**: Excellent for visualizing clusters and local structure. Can
      distort global distances. Slower than UMAP.
    
    - **MDS**: Preserves pairwise distances well. Good when you have a
      distance matrix. Can be slow for large datasets.
    
    - **Isomap**: Good for data on a non-linear manifold. Uses geodesic
      distances. Sensitive to neighborhood size.
    
    - **LLE**: Preserves local linear relationships. Fast but can be sensitive
      to noise.
    
    - **Spectral**: Based on graph Laplacian. Good for data with clear
      cluster structure.
    
    **For neural data**:
    - Population activity analysis: PCA → UMAP → t-SNE (in order of speed)
    - Trajectory analysis: PCA or Isomap (preserve continuity)
    - Cluster identification: UMAP or Spectral
    - Distance matrix: MDS or Isomap with metric='precomputed'
    
    **Computational Complexity**:
    - Fast: PCA, Spectral
    - Medium: UMAP, Isomap, LLE
    - Slow: t-SNE, MDS (for large n)
    
    See Also
    --------
    compute_multiple_embeddings : Compare multiple methods side-by-side
    pca_explained_variance : Analyze variance explained by PCA components
    """
    data = np.asarray(data)
    
    # Validate inputs
    if data.ndim != 2:
        raise ValueError(f"Data must be 2D, got shape {data.shape}")
    
    if n_components < 1:
        raise ValueError(f"n_components must be >= 1, got {n_components}")
    
    if n_components > data.shape[0]:
        raise ValueError(
            f"n_components ({n_components}) cannot exceed n_samples ({data.shape[0]})"
        )
    
    method = method.lower()
    
    # PCA - works in feature space
    if method == "pca":
        model = PCA(n_components=n_components, random_state=random_state, **kwargs)
        embedding = model.fit_transform(data)
        logger.info(f"PCA: Explained variance ratio: {model.explained_variance_ratio_}")
    
    # UMAP - requires separate installation
    elif method == "umap":
        if not UMAP_AVAILABLE:
            raise ImportError(
                "UMAP is not installed. Install with: pip install umap-learn"
            )
        model = UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            metric=metric,
            random_state=random_state,
            **kwargs
        )
        embedding = model.fit_transform(data)
    
    # t-SNE - stochastic, good for visualization
    elif method == "tsne":
        perplexity = kwargs.pop("perplexity", 30.0)
        learning_rate = kwargs.pop("learning_rate", 200.0)
        max_iter = kwargs.pop("max_iter", 1000)
        # Also support old parameter name for backwards compatibility
        max_iter = kwargs.pop("n_iter", max_iter)
        
        model = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            learning_rate=learning_rate,
            max_iter=max_iter,
            metric=metric,
            random_state=random_state,
            **kwargs
        )
        embedding = model.fit_transform(data)
    
    # MDS - preserves pairwise distances
    elif method == "mds":
        dissimilarity = "precomputed" if metric == "precomputed" else "euclidean"
        # Set n_init=1 to avoid FutureWarning (will be default in sklearn 1.9)
        n_init = kwargs.pop("n_init", 1)
        model = MDS(
            n_components=n_components,
            dissimilarity=dissimilarity,
            n_init=n_init,
            random_state=random_state,
            **kwargs
        )
        embedding = model.fit_transform(data)
    
    # Isomap - geodesic distances
    elif method == "isomap":
        model = Isomap(
            n_components=n_components,
            n_neighbors=n_neighbors,
            metric=metric,
            **kwargs
        )
        embedding = model.fit_transform(data)
    
    # LLE - local linear relationships
    elif method == "lle":
        model = LocallyLinearEmbedding(
            n_components=n_components,
            n_neighbors=n_neighbors,
            random_state=random_state,
            **kwargs
        )
        embedding = model.fit_transform(data)
    
    # Spectral - graph-based
    elif method == "spectral":
        affinity = kwargs.pop("affinity", "nearest_neighbors")
        model = SpectralEmbedding(
            n_components=n_components,
            n_neighbors=n_neighbors,
            affinity=affinity,
            random_state=random_state,
            **kwargs
        )
        embedding = model.fit_transform(data)
    
    else:
        raise ValueError(
            f"Unknown method '{method}'. Choose from: "
            f"{', '.join(['pca', 'umap', 'tsne', 'mds', 'isomap', 'lle', 'spectral'])}"
        )
    
    logger.info(
        f"Computed {method.upper()} embedding: "
        f"{data.shape} → {embedding.shape}"
    )
    return embedding


def compute_multiple_embeddings(
    data: npt.ArrayLike,
    methods: list[EmbeddingMethod] = None,
    n_components: int = 2,
    metric: str = "euclidean",
    n_neighbors: int = 15,
    random_state: int | None = 42,
    **kwargs
) -> dict[str, np.ndarray]:
    """
    Compute multiple embeddings for comparison.
    
    This function computes several dimensionality reduction methods on the same
    data, useful for comparing different approaches and choosing the best one
    for your analysis.
    
    Parameters
    ----------
    data : array-like, shape (n_samples, n_features)
        Input data matrix
    methods : list of str, optional
        List of methods to compute. If None, defaults to ["pca", "umap", "tsne"].
        Available: ["pca", "umap", "tsne", "mds", "isomap", "lle", "spectral"]
    n_components : int, default=2
        Number of dimensions for all embeddings
    metric : str, default="euclidean"
        Distance metric for methods that use distances
    n_neighbors : int, default=15
        Number of neighbors for neighborhood-based methods
    random_state : int, optional, default=42
        Random seed for reproducibility
    **kwargs
        Additional parameters passed to compute_embedding()
    
    Returns
    -------
    embeddings : dict
        Dictionary mapping method names to embedding arrays:
        {"pca": array(...), "umap": array(...), ...}
    
    Examples
    --------
    >>> from neural_analysis.embeddings import compute_multiple_embeddings
    >>> import numpy as np
    >>> 
    >>> # Generate data
    >>> rng = np.random.default_rng(42)
    >>> data = rng.normal(0, 1, (200, 50))
    >>> 
    >>> # Compute multiple embeddings
    >>> embeddings = compute_multiple_embeddings(
    ...     data,
    ...     methods=["pca", "umap", "tsne"],
    ...     n_components=2
    ... )
    >>> 
    >>> # Access individual embeddings
    >>> pca_embedding = embeddings["pca"]
    >>> umap_embedding = embeddings["umap"]
    >>> 
    >>> # Use with plotting (see plot_multiple_embeddings in visualization.py)
    >>> from neural_analysis.embeddings import plot_multiple_embeddings
    >>> fig = plot_multiple_embeddings(embeddings)
    
    See Also
    --------
    compute_embedding : Compute single embedding
    plot_multiple_embeddings : Visualize multiple embeddings in grid
    """
    if methods is None:
        methods = ["pca", "umap", "tsne"] if UMAP_AVAILABLE else ["pca", "tsne"]
    
    embeddings = {}
    for method in methods:
        try:
            embedding = compute_embedding(
                data=data,
                method=method,
                n_components=n_components,
                metric=metric,
                n_neighbors=n_neighbors,
                random_state=random_state,
                **kwargs
            )
            embeddings[method] = embedding
        except ImportError as e:
            logger.warning(f"Skipping {method}: {e}")
        except Exception as e:
            logger.error(f"Failed to compute {method} embedding: {e}")
    
    return embeddings


def pca_explained_variance(
    data: npt.ArrayLike,
    n_components: int | None = None,
    cumulative: bool = True
) -> dict[str, np.ndarray]:
    """
    Compute explained variance for PCA components.
    
    This function helps determine how many PCA components are needed to
    capture most of the variance in the data. Useful for dimensionality
    reduction decisions.
    
    Parameters
    ----------
    data : array-like, shape (n_samples, n_features)
        Input data matrix
    n_components : int, optional
        Number of components to compute. If None, computes all components
        (min(n_samples, n_features))
    cumulative : bool, default=True
        Whether to return cumulative variance in addition to individual variance
    
    Returns
    -------
    variance_info : dict
        Dictionary containing: "explained_variance_ratio" (fraction of variance 
        explained by each component), "explained_variance" (absolute variance 
        explained by each component), "cumulative_variance_ratio" (cumulative sum 
        of explained variance, if cumulative=True), "n_components_90" (number of 
        components needed for 90% variance, if cumulative=True), "n_components_95" 
        (number of components for 95% variance, if cumulative=True), 
        "n_components_99" (number of components for 99% variance, if cumulative=True).
    
    Examples
    --------
    >>> from neural_analysis.embeddings import pca_explained_variance
    >>> import numpy as np
    >>> 
    >>> # Generate data
    >>> rng = np.random.default_rng(42)
    >>> data = rng.normal(0, 1, (200, 50))
    >>> 
    >>> # Compute variance
    >>> variance_info = pca_explained_variance(data)
    >>> 
    >>> # How many components for 90% variance?
    >>> print(f"Components for 90% variance: {variance_info['n_components_90']}")
    >>> 
    >>> # Plot scree plot
    >>> from neural_analysis.embeddings import plot_pca_variance
    >>> fig = plot_pca_variance(variance_info)
    
    See Also
    --------
    compute_embedding : Compute PCA embedding
    plot_pca_variance : Visualize explained variance (scree plot)
    """
    data = np.asarray(data)
    
    if n_components is None:
        n_components = min(data.shape)
    
    pca = PCA(n_components=n_components)
    pca.fit(data)
    
    result = {
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "explained_variance": pca.explained_variance_,
    }
    
    if cumulative:
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        result["cumulative_variance_ratio"] = cumsum
        
        # Find components needed for common thresholds
        for threshold in [0.90, 0.95, 0.99]:
            n_comp = np.argmax(cumsum >= threshold) + 1
            key = f"n_components_{int(threshold*100)}"
            result[key] = n_comp
    
    logger.info(
        f"PCA: {n_components} components explain "
        f"{pca.explained_variance_ratio_.sum():.2%} of variance"
    )
    
    return result
