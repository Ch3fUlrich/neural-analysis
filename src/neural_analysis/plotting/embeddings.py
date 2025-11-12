"""Embedding visualization functions for dimensionality reduction results.

This module provides functions for visualizing low-dimensional embeddings from
techniques like PCA, t-SNE, UMAP, etc. It supports both 2D and 3D visualizations
with optional convex hulls for group separation.

Examples:
    Basic 2D embedding visualization:
        >>> embedding = np.random.randn(100, 2)
        >>> labels = np.random.randint(0, 5, 100)
        >>> fig = plot_embedding_2d(embedding, labels, title="PCA Embedding")

    3D embedding with convex hulls:
        >>> embedding_3d = np.random.randn(100, 3)
        >>> fig = plot_embedding_3d(embedding_3d, labels, show_hulls=True)

    Automatic dimension selection:
        >>> embedding_nd = np.random.randn(100, 2)  # Can be 2D or 3D
        >>> fig = plot_embedding(embedding_nd, labels)  # Auto-selects plot type
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from scipy.spatial import ConvexHull

from .backend import BackendType
from .core import PlotConfig
from .grid_config import GridLayoutConfig, PlotGrid, PlotSpec

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go


def plot_embedding(
    embedding: npt.NDArray[np.floating],
    labels: npt.NDArray[np.floating[Any]] | None = None,
    *,
    colors: npt.NDArray[np.floating[Any]] | None = None,
    backend: BackendType = BackendType.MATPLOTLIB,
    config: PlotConfig | None = None,
    title: str = "Embedding",
    show_hulls: bool = False,
    **kwargs: Any,
) -> plt.Figure | go.Figure:
    """Plot embeddings with automatic 2D/3D detection.

    Automatically selects 2D or 3D plotting based on embedding dimensions.
    For higher dimensional embeddings, uses PCA to reduce to 2D.

    Args:
        embedding: Embedding coordinates, shape (n_samples, n_dims).
            Can be 2D or 3D. Higher dimensions will be reduced via PCA.
        labels: Optional labels for coloring points, shape (n_samples,) or
            (n_samples, n_label_dims). Used for continuous or discrete coloring.
        colors: Optional explicit RGBA colors, shape (n_samples, 4).
            If provided, overrides label-based coloring.
        backend: Plotting backend to use ("matplotlib" or "plotly").
        config: Optional plot configuration for styling.
        title: Plot title.
        show_hulls: If True, draws convex hulls around groups (requires discrete labels).
        **kwargs: Additional parameters passed to scatter plot.

    Returns:
        Backend-specific plot object (matplotlib Figure or plotly Figure).

    Raises:
        ValueError: If embedding has invalid shape or incompatible with projection.

    Examples:
        >>> embedding = np.random.randn(100, 2)
        >>> labels = np.random.randint(0, 5, 100)
        >>> fig = plot_embedding(embedding, labels, title="Auto Embedding")
    """
    n_samples, n_dims = embedding.shape

    # Validate dimensions
    if n_dims < 2:
        raise ValueError(f"Embedding must have at least 2 dimensions, got {n_dims}")

    # For >3 dimensions, reduce to 2D using PCA
    if n_dims > 3:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2)
        embedding = pca.fit_transform(embedding)
        n_dims = 2
        print(f"Reduced {n_samples} samples from {n_dims}D to 2D using PCA")

    # Route to appropriate function
    if n_dims == 2:
        return plot_embedding_2d(
            embedding=embedding,
            labels=labels,
            colors=colors,
            backend=backend,
            config=config,
            title=title,
            show_hulls=show_hulls, **kwargs)
    else:  # n_dims == 3
        return plot_embedding_3d(
            embedding=embedding,
            labels=labels,
            colors=colors,
            backend=backend,
            config=config,
            title=title,
            show_hulls=show_hulls, **kwargs)


def plot_embedding_2d(
    embedding: npt.NDArray[np.floating],
    labels: npt.NDArray[np.floating[Any]] | None = None,
    *,
    colors: npt.NDArray[np.floating[Any]] | None = None,
    backend: BackendType = BackendType.MATPLOTLIB,
    config: PlotConfig | None = None,
    title: str = "2D Embedding",
    show_hulls: bool = False,
    hull_alpha: float = 0.2,
    **kwargs: Any,
) -> plt.Figure | go.Figure:
    """Plot 2D embeddings with optional convex hulls.

    Creates a scatter plot of 2D embedding coordinates with optional
    convex hull visualization for group separation.

    Args:
        embedding: 2D embedding coordinates, shape (n_samples, 2).
        labels: Optional labels for coloring points, shape (n_samples,).
            Can be discrete (integers) or continuous (floats).
        colors: Optional explicit RGBA colors, shape (n_samples, 4).
            If provided, overrides label-based coloring.
        backend: Plotting backend to use ("matplotlib" or "plotly").
        config: Optional plot configuration for styling.
        title: Plot title.
        show_hulls: If True, draws convex hulls around groups defined by
            discrete labels. Requires labels to be provided.
        hull_alpha: Transparency of convex hull fills (0-1).
        **kwargs: Additional scatter plot parameters (e.g., markersize, alpha).

    Returns:
        Backend-specific plot object (matplotlib Figure or plotly Figure).

    Raises:
        ValueError: If embedding is not 2D or hulls requested without labels.

    Examples:
        >>> embedding = np.random.randn(100, 2)
        >>> labels = np.random.randint(0, 5, 100)
        >>> fig = plot_embedding_2d(embedding, labels, show_hulls=True)

        With continuous labels:
        >>> continuous_labels = np.random.rand(100)
        >>> fig = plot_embedding_2d(embedding, continuous_labels)
    """
    if embedding.shape[1] != 2:
        raise ValueError(
            f"Embedding must be 2D for plot_embedding_2d, got shape {embedding.shape}"
        )

    if show_hulls and labels is None:
        raise ValueError("show_hulls requires labels to be provided")

    # Create plot spec
    spec = PlotSpec(
        plot_type="scatter_2d",
        title=title,
        x_data=embedding[:, 0],
        y_data=embedding[:, 1],
        labels=labels,
        colors=colors,
        show_hulls=show_hulls,
        hull_alpha=hull_alpha if show_hulls else None,
        marker_size=kwargs.pop("markersize", kwargs.pop("s", 20)),
        alpha=kwargs.pop("alpha", 0.6), **kwargs)

    # Create plot config if not provided
    if config is None:
        config = PlotConfig(
            title=title,
            xlabel="Dimension 1",
            ylabel="Dimension 2",
            show_grid=False,
            show_legend=labels is not None,
        )

    # Create and plot grid
    grid = PlotGrid(
        specs=[spec],
        layout=GridLayoutConfig(n_rows=1, n_cols=1),
        config=config,
        backend=backend,
    )

    return grid.plot()


def plot_embedding_3d(
    embedding: npt.NDArray[np.floating],
    labels: npt.NDArray[np.floating[Any]] | None = None,
    *,
    colors: npt.NDArray[np.floating[Any]] | None = None,
    backend: BackendType = BackendType.MATPLOTLIB,
    config: PlotConfig | None = None,
    title: str = "3D Embedding",
    show_hulls: bool = False,
    hull_alpha: float = 0.2,
    **kwargs: Any,
) -> plt.Figure | go.Figure:
    """Plot 3D embeddings with optional convex hulls.

    Creates a 3D scatter plot of embedding coordinates with optional
    convex hull visualization for group separation.

    Args:
        embedding: 3D embedding coordinates, shape (n_samples, 3).
        labels: Optional labels for coloring points, shape (n_samples,).
            Can be discrete (integers) or continuous (floats).
        colors: Optional explicit RGBA colors, shape (n_samples, 4).
            If provided, overrides label-based coloring.
        backend: Plotting backend to use ("matplotlib" or "plotly").
        config: Optional plot configuration for styling.
        title: Plot title.
        show_hulls: If True, draws 3D convex hulls around groups defined by
            discrete labels. Requires labels to be provided.
        hull_alpha: Transparency of convex hull faces (0-1).
        **kwargs: Additional scatter plot parameters (e.g., markersize, alpha).

    Returns:
        Backend-specific plot object (matplotlib Figure or plotly Figure).

    Raises:
        ValueError: If embedding is not 3D or hulls requested without labels.

    Examples:
        >>> embedding = np.random.randn(100, 3)
        >>> labels = np.random.randint(0, 5, 100)
        >>> fig = plot_embedding_3d(embedding, labels, show_hulls=True)

        Interactive plotly visualization:
        >>> fig = plot_embedding_3d(
        ...     embedding, labels,
        ...     backend=BackendType.PLOTLY,
        ...     show_hulls=True
        ... )
    """
    if embedding.shape[1] != 3:
        raise ValueError(
            f"Embedding must be 3D for plot_embedding_3d, got shape {embedding.shape}"
        )

    if show_hulls and labels is None:
        raise ValueError("show_hulls requires labels to be provided")

    # Create plot spec
    spec = PlotSpec(
        plot_type="scatter_3d",
        title=title,
        x_data=embedding[:, 0],
        y_data=embedding[:, 1],
        z_data=embedding[:, 2],
        labels=labels,
        colors=colors,
        show_hulls=show_hulls,
        hull_alpha=hull_alpha if show_hulls else None,
        marker_size=kwargs.pop("markersize", kwargs.pop("s", 20)),
        alpha=kwargs.pop("alpha", 0.6), **kwargs)

    # Create plot config if not provided
    if config is None:
        config = PlotConfig(
            title=title,
            xlabel="Dimension 1",
            ylabel="Dimension 2",
            zlabel="Dimension 3",
            show_grid=False,
            show_legend=labels is not None,
        )

    # Create and plot grid
    grid = PlotGrid(
        specs=[spec],
        layout=GridLayoutConfig(n_rows=1, n_cols=1),
        config=config,
        backend=backend,
    )

    return grid.plot()


def compute_convex_hull(
    points: npt.NDArray[np.floating],
) -> ConvexHull | None:
    """Compute convex hull for a set of points.

    Args:
        points: Point coordinates, shape (n_points, n_dims).
            Must have at least n_dims + 1 points.

    Returns:
        scipy.spatial.ConvexHull object, or None if too few points.

    Examples:
        >>> points = np.random.randn(10, 2)
        >>> hull = compute_convex_hull(points)
        >>> if hull is not None:
        ...     vertices = points[hull.vertices]
    """
    n_points, n_dims = points.shape

    # Need at least n_dims + 1 points for a hull
    if n_points < n_dims + 1:
        return None

    try:
        hull = ConvexHull(points)
        return hull
    except Exception:
        # Hull computation can fail for degenerate cases
        return None


def group_points_by_labels(
    points: npt.NDArray[np.floating],
    labels: npt.NDArray[np.floating[Any]],
) -> dict[Any, npt.NDArray[np.floating[Any]]]:
    """Group points by their labels.

    Args:
        points: Point coordinates, shape (n_points, n_dims).
        labels: Point labels, shape (n_points,).

    Returns:
        Dictionary mapping each unique label to its corresponding points.

    Examples:
        >>> points = np.random.randn(100, 2)
        >>> labels = np.random.randint(0, 5, 100)
        >>> groups = group_points_by_labels(points, labels)
        >>> for label, group_points in groups.items():
        ...     print(f"Label {label}: {len(group_points)} points")
    """
    groups = {}
    unique_labels = np.unique(labels)

    for label in unique_labels:
        mask = labels == label
        groups[label] = points[mask]

    return groups
