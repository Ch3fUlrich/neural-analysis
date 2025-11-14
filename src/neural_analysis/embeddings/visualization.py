"""
Visualization functions for embeddings using the PlotGrid system.

This module provides plotting functions for dimensionality reduction embeddings,
integrated with the neural_analysis PlotGrid architecture for consistent styling
and flexible customization.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt

from neural_analysis.plotting.core import PlotConfig
from neural_analysis.plotting.grid_config import (
    GridLayoutConfig,
    PlotGrid,
    PlotSpec,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

__all__ = [
    "plot_multiple_embeddings",
    "plot_pca_variance",
]


def plot_multiple_embeddings(
    embeddings: dict[str, npt.NDArray[np.floating]],
    labels: npt.ArrayLike | None = None,
    colors: list[str] | str | None = None,
    title: str = "Dimensionality Reduction Comparison",
    figsize: tuple[int, int] = (16, 10),
    point_size: int = 20,
    alpha: float = 0.7,
    show_legend: bool = True,
    backend: Literal["matplotlib", "plotly"] = "matplotlib",
    **kwargs: Any,
) -> Figure:
    """
    Plot multiple embeddings side-by-side for comparison.

    This function creates a grid of subplots showing different dimensionality
    reduction methods applied to the same data. Uses the PlotGrid system for
    consistent styling and flexibility.

    Parameters
    ----------
    embeddings : dict[str, Any]
        Dictionary mapping method names to embedding arrays.
        Each array should have shape (n_samples, n_components).
        Example: {"pca": pca_embedding, "umap": umap_embedding}
    labels : array-like, optional, shape (n_samples,)
        Category labels for coloring points. If provided, points will be
        colored by category. If None, all points use the same color.
    colors : list[Any] of str or str, optional
        Colors for categories. Can be:
        - List of color names/hex codes (one per unique label)
        - Single color name/hex code (for all points if labels=None)
        - None (uses default color cycle)
    title : str, default="Dimensionality Reduction Comparison"
        Overall title for the figure
    figsize : tuple of int, default=(16, 10)
        Figure size in inches (width, height)
    point_size : int, default=20
        Size of scatter points
    alpha : float, default=0.7
        Transparency of points (0=transparent, 1=opaque)
    show_legend : bool, default=True
        Whether to show legend (only relevant if labels are provided)
    backend : {"matplotlib", "plotly"}, default="matplotlib"
        Plotting backend to use
    **kwargs
        Additional keyword arguments passed to PlotGrid configuration

    Returns
    -------
    fig : Figure
        Matplotlib Figure object (if backend="matplotlib")
        or Plotly Figure (if backend="plotly")

    Examples
    --------
    >>> from neural_analysis.embeddings import (
    ...     compute_multiple_embeddings,
    ...     plot_multiple_embeddings
    ... )
    >>> import numpy as np
    >>>
    >>> # Generate synthetic data with 3 clusters
    >>> rng = np.random.default_rng(42)
    >>> data = np.vstack([
    ...     rng.normal(0, 1, (100, 50)),
    ...     rng.normal(3, 1, (100, 50)),
    ...     rng.normal(-3, 1, (100, 50))
    ... ])
    >>> labels = np.repeat([0, 1, 2], 100)
    >>>
    >>> # Compute embeddings
    >>> embeddings = compute_multiple_embeddings(
    ...     data,
    ...     methods=["pca", "umap", "tsne"],
    ...     n_components=2
    ... )
    >>>
    >>> # Plot comparison
    >>> fig = plot_multiple_embeddings(
    ...     embeddings,
    ...     labels=labels,
    ...     colors=["steelblue", "coral", "mediumseagreen"],
    ...     title="Neural Population Activity Embeddings"
    ... )

    Notes
    -----
    This function automatically arranges subplots in a grid based on the number
    of embeddings provided. The layout is optimized for readability:
    - 1-2 embeddings: 1 row
    - 3-4 embeddings: 2 rows
    - 5-6 embeddings: 2 rows
    - 7+ embeddings: 3 rows

    Each subplot shows the embedding with the method name as the title.
    If labels are provided, points are colored by category and a legend is shown.

    See Also
    --------
    compute_multiple_embeddings : Compute multiple embeddings at once
    plot_pca_variance : Plot explained variance for PCA
    """
    if not embeddings:
        raise ValueError("No embeddings provided")

    # Determine grid layout
    n_embeddings = len(embeddings)
    if n_embeddings <= 2:
        rows, cols = 1, n_embeddings
    elif n_embeddings <= 4:
        rows, cols = 2, 2
    elif n_embeddings <= 6:
        rows, cols = 2, 3
    else:
        rows, cols = 3, (n_embeddings + 2) // 3

    # Prepare labels and colors
    if labels is not None:
        labels = np.asarray(labels)
        unique_labels = np.unique(labels)
        n_categories = len(unique_labels)

        # Default colors if not provided
        if colors is None:
            from matplotlib import cm

            cmap = cm.get_cmap("tab10")
            colors_list: list[str] = [
                f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
                for r, g, b, _ in (cmap(i) for i in range(n_categories))
            ]
            colors = colors_list
        elif isinstance(colors, str):
            colors = [colors] * n_categories
    else:
        # No labels - single color
        if colors is None:
            colors = "steelblue"
        labels = None
        unique_labels = None

    # Create PlotSpecs for each embedding
    plot_specs = []
    for idx, (method_name, embedding) in enumerate(embeddings.items()):
        embedding = np.asarray(embedding)

        if embedding.ndim != 2:
            raise ValueError(
                f"Embedding '{method_name}' must be 2D, got shape {embedding.shape}"
            )

        n_samples, n_dims = embedding.shape

        if n_dims not in [2, 3]:
            logger.warning(
                f"Embedding '{method_name}' has {n_dims} dimensions. "
                "Only 2D and 3D embeddings are supported for visualization."
            )
            continue

        # Create scatter plot(s) for this embedding
        if labels is None:
            # Single scatter plot (no categories)
            plot_specs.append(
                PlotSpec(
                    data=embedding,
                    plot_type="scatter",
                    subplot_position=idx,
                    title=method_name.upper(),
                    color=colors if isinstance(colors, str) else colors[0],
                    alpha=alpha,
                    kwargs={
                        "size": point_size,
                        "x_label": "Component 1" if n_dims >= 1 else None,
                        "y_label": "Component 2" if n_dims >= 2 else None,
                        "z_label": "Component 3" if n_dims == 3 else None,
                    },
                )
            )
        else:
            # Multiple scatter plots (one per category)
            for cat_idx, label_value in enumerate(unique_labels):
                mask = labels == label_value
                cat_embedding = embedding[mask]

                cat_color = colors[cat_idx] if isinstance(colors, list) else colors
                plot_specs.append(
                    PlotSpec(
                        data=cat_embedding,
                        plot_type="scatter",
                        subplot_position=idx,
                        title=method_name.upper(),
                        color=cat_color,
                        alpha=alpha,
                        label=f"Category {label_value}",
                        kwargs={
                            "size": point_size,
                            "x_label": "Component 1" if n_dims >= 1 else None,
                            "y_label": "Component 2" if n_dims >= 2 else None,
                            "z_label": "Component 3" if n_dims == 3 else None,
                        },
                    )
                )

    if not plot_specs:
        raise ValueError("No valid embeddings to plot")

    # Create PlotGrid
    grid = PlotGrid(
        plot_specs=plot_specs,
        config=PlotConfig(
            title=title,
            figsize=figsize,
            show=True,
            legend=show_legend and labels is not None,
            **kwargs,
        ),
        layout=GridLayoutConfig(rows=rows, cols=cols),
        backend=backend,
    )

    logger.info(f"Created embedding comparison plot with {n_embeddings} methods")
    return grid.plot()  # type: ignore[no-any-return]


def plot_pca_variance(
    variance_info: dict[str, npt.NDArray[np.floating]],
    cumulative: bool = True,
    n_components_to_show: int | None = None,
    threshold_lines: list[float] | None = None,
    figsize: tuple[int, int] = (12, 5),
    colors: list[str] | None = None,
    backend: Literal["matplotlib", "plotly"] = "matplotlib",
    **kwargs: Any,
) -> Figure:
    """
    Plot PCA explained variance (scree plot).

    This function creates a visualization of how much variance is explained by
    each PCA component. Useful for determining how many components to keep.

    Parameters
    ----------
    variance_info : dict[str, Any]
        Dictionary returned by pca_explained_variance(), containing:
        - "explained_variance_ratio": Individual component variance
        - "cumulative_variance_ratio": Cumulative variance (if available)
    cumulative : bool, default=True
        Whether to show cumulative variance in addition to individual variance.
        If True, creates two subplots side-by-side.
    n_components_to_show : int, optional
        Number of components to show. If None, shows all components.
    threshold_lines : list[Any] of float, optional
        Variance thresholds to mark with horizontal lines (e.g., [0.9, 0.95]).
        Only applies to cumulative plot.
    figsize : tuple of int, default=(12, 5)
        Figure size in inches (width, height)
    colors : list[Any] of str, optional
        Colors for [individual_variance, cumulative_variance] plots.
        Defaults to ["steelblue", "coral"]
    backend : {"matplotlib", "plotly"}, default="matplotlib"
        Plotting backend to use
    **kwargs
        Additional keyword arguments passed to PlotGrid configuration

    Returns
    -------
    fig : Figure
        Matplotlib Figure object (if backend="matplotlib")
        or Plotly Figure (if backend="plotly")

    Examples
    --------
    >>> from neural_analysis.embeddings import pca_explained_variance, plot_pca_variance
    >>> import numpy as np
    >>>
    >>> # Generate data
    >>> rng = np.random.default_rng(42)
    >>> data = rng.normal(0, 1, (200, 50))
    >>>
    >>> # Compute variance info
    >>> variance_info = pca_explained_variance(data)
    >>>
    >>> # Plot scree plot
    >>> fig = plot_pca_variance(
    ...     variance_info,
    ...     cumulative=True,
    ...     threshold_lines=[0.9, 0.95, 0.99]
    ... )
    >>>
    >>> # Check how many components needed for 90% variance
    >>> print(f"Components for 90%: {variance_info['n_components_90']}")

    Notes
    -----
    The scree plot helps determine the "elbow point" where adding more components
    provides diminishing returns. Common heuristics:
    - Keep components until cumulative variance reaches 90-95%
    - Look for "elbow" in individual variance plot
    - Keep components with eigenvalues > 1 (Kaiser criterion)

    See Also
    --------
    pca_explained_variance : Compute variance information
    compute_embedding : Compute PCA embedding
    """
    if colors is None:
        colors = ["steelblue", "coral"]

    if threshold_lines is None:
        threshold_lines = [0.90, 0.95] if cumulative else []

    # Get variance data
    var_ratio = variance_info["explained_variance_ratio"]
    n_components = len(var_ratio)

    if n_components_to_show is not None:
        n_components = min(n_components, n_components_to_show)
        var_ratio = var_ratio[:n_components]

    component_indices = np.arange(1, n_components + 1)

    # Create plot specs
    plot_specs = []

    # Individual variance bar plot
    plot_specs.append(
        PlotSpec(
            data=var_ratio,
            plot_type="bar",
            subplot_position=0,
            title="Individual Component Variance",
            color=colors[0],
            alpha=0.7,
            kwargs={
                "x": component_indices,
                "x_label": "Principal Component",
                "y_label": "Explained Variance Ratio",
                "show_values": False,
                "grid": {"axis": "y", "alpha": 0.3},
            },
        )
    )

    # Cumulative variance line plot
    if cumulative and "cumulative_variance_ratio" in variance_info:
        cumvar_ratio = variance_info["cumulative_variance_ratio"][:n_components]

        plot_specs.append(
            PlotSpec(
                data=cumvar_ratio,
                plot_type="line",
                subplot_position=1,
                title="Cumulative Variance Explained",
                color=colors[1],
                alpha=0.8,
                kwargs={
                    "x": component_indices,
                    "marker": "o",
                    "markersize": 4,
                    "linewidth": 2,
                    "x_label": "Principal Component",
                    "y_label": "Cumulative Variance Ratio",
                    "grid": {"axis": "both", "alpha": 0.3},
                },
            )
        )

        # Add threshold lines as horizontal reference lines
        # Note: PlotGrid doesn't have built-in support for axhline yet,
        # so we'll document this for future enhancement
        # For now, thresholds are visual guides users can add manually

    # Create layout
    n_plots = 2 if cumulative and "cumulative_variance_ratio" in variance_info else 1
    rows, cols = (1, n_plots)

    # Create PlotGrid
    grid = PlotGrid(
        plot_specs=plot_specs,
        config=PlotConfig(
            title="PCA Explained Variance",
            figsize=figsize,
            show=True,
            legend=False,
            **kwargs,
        ),
        layout=GridLayoutConfig(rows=rows, cols=cols),
        backend=backend,
    )

    # Create the plot
    fig = grid.plot()

    # Add threshold lines manually if matplotlib backend
    if backend == "matplotlib" and threshold_lines and cumulative and n_plots == 2:
        # Access the cumulative variance subplot (second subplot)
        ax = fig.axes[1] if len(fig.axes) > 1 else fig.axes[0]
        for threshold in threshold_lines:
            ax.axhline(
                threshold,
                color="gray",
                linestyle="--",
                linewidth=1,
                alpha=0.5,
                label=f"{threshold:.0%} variance",
            )
        ax.legend()

    logger.info(
        f"Created PCA variance plot: "
        f"{n_components} components, {var_ratio.sum():.2%} total variance"
    )

    return fig  # type: ignore[no-any-return]
