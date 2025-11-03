"""
Two-dimensional plotting functions for neural data visualization.

This module provides functions for creating 2D scatter plots, trajectory visualizations,
KDE density plots, and grouped scatter plots with optional convex hulls.

Functions support both matplotlib and plotly backends for flexibility between
static publication-quality figures and interactive exploratory visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Literal
from scipy.spatial import ConvexHull
from scipy.stats import gaussian_kde

from .backend import BackendType, get_backend
from .core import (
    PlotConfig,
    resolve_colormap,
    apply_layout_matplotlib,
    apply_layout_plotly,
    get_default_categorical_colors,
    finalize_plot_matplotlib,
    finalize_plot_plotly,
)

# Optional imports
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def plot_scatter_2d(
    x: np.ndarray,
    y: np.ndarray,
    config: PlotConfig | None = None,
    colors: np.ndarray | str | None = None,
    sizes: np.ndarray | float = 20,
    alpha: float = 0.7,
    cmap: str = "viridis",
    colorbar: bool = False,
    colorbar_label: str | None = None,
    backend: Literal["matplotlib", "plotly"] | None = None,
) -> plt.Axes | go.Figure:
    """
    Create a 2D scatter plot.
    
    Args:
        x: X coordinates (1D array)
        y: Y coordinates (1D array)
        config: Plot configuration (title, labels, etc.)
        colors: Color values (array for colormap, string for single color,
            or None for default)
        sizes: Point sizes (array or single value)
        alpha: Transparency level (0-1)
        cmap: Colormap name when colors is an array
        colorbar: Whether to show colorbar (matplotlib only, when colors
            is array)
        colorbar_label: Label for colorbar
        backend: Backend to use ('matplotlib' or 'plotly')
        
    Returns:
        Matplotlib Axes or Plotly Figure object
        
    Example:
        >>> x = np.random.randn(100)
        >>> y = np.random.randn(100)
        >>> plot_scatter_2d(x, y, PlotConfig(title="Random Points"))
    """
    # Input validation
    if len(x) != len(y):
        raise ValueError(f"x and y must have same length, got {len(x)} and {len(y)}")
    
    # Default config
    if config is None:
        config = PlotConfig()
    
    # Determine backend
    backend_type = get_backend() if backend is None else BackendType(backend)
    
    if backend_type == BackendType.MATPLOTLIB:
        return _plot_scatter_2d_matplotlib(
            x, y, config, colors, sizes, alpha, cmap, colorbar, colorbar_label
        )
    else:
        if not PLOTLY_AVAILABLE:
            raise ValueError("Plotly backend requested but plotly is not installed")
        return _plot_scatter_2d_plotly(
            x, y, config, colors, sizes, alpha, cmap, colorbar_label
        )


def _plot_scatter_2d_matplotlib(
    x: np.ndarray,
    y: np.ndarray,
    config: PlotConfig,
    colors: np.ndarray | str | None,
    sizes: np.ndarray | float,
    alpha: float,
    cmap: str,
    colorbar: bool,
    colorbar_label: str | None,
) -> plt.Axes:
    """Matplotlib implementation of 2D scatter plot."""
    fig, ax = plt.subplots(figsize=config.figsize)
    
    # Determine color parameter and whether to use colormap
    if colors is None:
        color_param = 'C0'  # Default matplotlib color
        use_cmap = False
    elif isinstance(colors, str):
        color_param = colors
        use_cmap = False
    else:
        color_param = colors
        use_cmap = True
        
    # Create scatter (only pass cmap if using array colors)
    if use_cmap:
        mpl_cmap = resolve_colormap(cmap, BackendType.MATPLOTLIB)
        scatter = ax.scatter(x, y, c=color_param, s=sizes, alpha=alpha, cmap=mpl_cmap)
    else:
        scatter = ax.scatter(x, y, c=color_param, s=sizes, alpha=alpha)
    
    # Add colorbar if requested and colors is an array
    if colorbar and use_cmap:
        cbar = plt.colorbar(scatter, ax=ax)
        if colorbar_label:
            cbar.set_label(colorbar_label)
    
    # Apply configuration consistently
    apply_layout_matplotlib(ax, config)
    
    finalize_plot_matplotlib(config)
    
    return ax


def _plot_scatter_2d_plotly(
    x: np.ndarray,
    y: np.ndarray,
    config: PlotConfig,
    colors: np.ndarray | str | None,
    sizes: np.ndarray | float,
    alpha: float,
    cmap: str,
    colorbar_label: str | None,
) -> go.Figure:
    """Plotly implementation of 2D scatter plot."""
    # Prepare marker dict
    marker_dict = {
        'size': sizes if isinstance(sizes, (int, float)) else sizes.tolist(),
        'opacity': alpha,
    }
    
    # Handle colors
    if colors is not None:
        if isinstance(colors, str):
            marker_dict['color'] = colors
        else:
            marker_dict['color'] = colors.tolist()
            marker_dict['colorscale'] = resolve_colormap(cmap, BackendType.PLOTLY)
            if colorbar_label:
                marker_dict['colorbar'] = {'title': colorbar_label}
                
    # Create figure
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='markers',
        marker=marker_dict,
    ))
    
    # Apply common layout (titles, labels, limits, grid, size)
    apply_layout_plotly(fig, config)
    
    finalize_plot_plotly(fig, config)
    
    return fig


def plot_trajectory_2d(
    x: np.ndarray,
    y: np.ndarray,
    config: PlotConfig | None = None,
    color_by_time: bool = True,
    cmap: str = "viridis",
    linewidth: float = 1.0,
    show_points: bool = True,
    point_size: float = 10,
    alpha: float = 0.7,
    backend: Literal["matplotlib", "plotly"] | None = None,
) -> plt.Axes | go.Figure:
    """
    Plot a 2D trajectory with line connecting points.
    
    Args:
        x: X coordinates  
        y: Y coordinates
        config: Plot configuration
        color_by_time: Color line segments by time progression
        cmap: Colormap for time coloring
        linewidth: Width of trajectory line
        show_points: Whether to show scatter points
        point_size: Size of scatter points
        alpha: Transparency
        backend: Backend to use
        
    Returns:
        Matplotlib Axes or Plotly Figure
        
    Example:
        >>> t = np.linspace(0, 4*np.pi, 100)
        >>> x = np.sin(t)
        >>> y = np.cos(t)
        >>> plot_trajectory_2d(x, y, PlotConfig(title="Circular Trajectory"))
    """
    # Input validation
    if len(x) != len(y):
        raise ValueError(f"x and y must have same length, got {len(x)} and {len(y)}")
    
    if config is None:
        config = PlotConfig()
    
    backend_type = get_backend() if backend is None else BackendType(backend)
    
    if backend_type == BackendType.MATPLOTLIB:
        return _plot_trajectory_2d_matplotlib(
            x, y, config, color_by_time, cmap, linewidth, show_points, point_size, alpha
        )
    else:
        if not PLOTLY_AVAILABLE:
            raise ValueError("Plotly backend requested but plotly is not installed")
        return _plot_trajectory_2d_plotly(
            x, y, config, color_by_time, cmap, linewidth, show_points, point_size, alpha
        )


def _plot_trajectory_2d_matplotlib(
    x: np.ndarray,
    y: np.ndarray,
    config: PlotConfig,
    color_by_time: bool,
    cmap: str,
    linewidth: float,
    show_points: bool,
    point_size: float,
    alpha: float,
) -> plt.Axes:
    """Matplotlib implementation of 2D trajectory plot."""
    fig, ax = plt.subplots(figsize=config.figsize)
    
    if color_by_time:
        # Create line collection with color gradient
        from matplotlib.collections import LineCollection
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        norm = plt.Normalize(0, len(x))
        lc = LineCollection(
            segments,
            cmap=resolve_colormap(cmap, BackendType.MATPLOTLIB),
            norm=norm,
            linewidth=linewidth,
            alpha=alpha,
        )
        lc.set_array(np.arange(len(x)))
        ax.add_collection(lc)
        
        # Add colorbar
        plt.colorbar(lc, ax=ax, label='Time')
    else:
        # Simple line plot
        ax.plot(x, y, linewidth=linewidth, alpha=alpha)
    
    # Add scatter points if requested
    if show_points:
        if color_by_time:
            colors = np.arange(len(x))
            mpl_cmap = resolve_colormap(cmap, BackendType.MATPLOTLIB)
            ax.scatter(x, y, c=colors, s=point_size, cmap=mpl_cmap, alpha=alpha, zorder=3)
        else:
            ax.scatter(x, y, s=point_size, alpha=alpha, zorder=3)
    
    # Apply configuration
    apply_layout_matplotlib(ax, config)
    ax.set_aspect('equal', adjustable='box')
    
    finalize_plot_matplotlib(config)
    
    return ax


def _plot_trajectory_2d_plotly(
    x: np.ndarray,
    y: np.ndarray,
    config: PlotConfig,
    color_by_time: bool,
    cmap: str,
    linewidth: float,
    show_points: bool,
    point_size: float,
    alpha: float,
) -> go.Figure:
    """Plotly implementation of 2D trajectory plot."""
    fig = go.Figure()
    
    # Prepare marker and line properties
    marker_dict = {'size': point_size, 'opacity': alpha}
    if color_by_time:
        marker_dict['color'] = np.arange(len(x)).tolist()
        marker_dict['colorscale'] = resolve_colormap(cmap, BackendType.PLOTLY)
    
    mode = 'lines+markers' if show_points else 'lines'
    
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode=mode,
        line=dict(width=linewidth),
        marker=marker_dict if show_points else {},
        opacity=alpha,
    ))
    
    # Apply common layout and set equal aspect
    apply_layout_plotly(fig, config)
    fig.update_layout(yaxis=dict(scaleanchor='x'))
    
    finalize_plot_plotly(fig, config)
    
    return fig


def plot_grouped_scatter_2d(
    group_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    config: PlotConfig | None = None,
    show_hulls: bool = False,
    alpha: float = 0.6,
    point_size: float = 20,
    colors: List[str] | None = None,
    backend: Literal["matplotlib", "plotly"] | None = None,
) -> plt.Axes | go.Figure:
    """
    Plot grouped 2D scatter data with optional convex hulls.
    
    Args:
        group_data: Dictionary mapping group names to (x, y) tuples
        config: Plot configuration
        show_hulls: Whether to show convex hulls around groups
        alpha: Transparency level
        point_size: Size of scatter points
        colors: List of colors for groups (auto-generated if None)
        backend: Backend to use
        
    Returns:
        Matplotlib Axes or Plotly Figure
        
    Example:
        >>> groups = {
        ...     'A': (np.random.randn(50), np.random.randn(50)),
        ...     'B': (np.random.randn(50) + 3, np.random.randn(50) + 3),
        ... }
        >>> plot_grouped_scatter_2d(groups, PlotConfig(title="Grouped Data"))
    """
    if not group_data:
        raise ValueError("group_data cannot be empty")
    
    # Validate all groups have matching x,y lengths
    for name, (x, y) in group_data.items():
        if len(x) != len(y):
            raise ValueError(f"Group '{name}': x and y must have same length")
    
    if config is None:
        config = PlotConfig()
    
    backend_type = get_backend() if backend is None else BackendType(backend)
    
    if backend_type == BackendType.MATPLOTLIB:
        return _plot_grouped_scatter_2d_matplotlib(
            group_data, config, show_hulls, alpha, point_size, colors
        )
    else:
        if not PLOTLY_AVAILABLE:
            raise ValueError("Plotly backend requested but plotly is not installed")
        return _plot_grouped_scatter_2d_plotly(
            group_data, config, show_hulls, alpha, point_size, colors
        )


def _plot_grouped_scatter_2d_matplotlib(
    group_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    config: PlotConfig,
    show_hulls: bool,
    alpha: float,
    point_size: float,
    colors: List[str] | None,
) -> plt.Axes:
    """Matplotlib implementation of grouped scatter plot."""
    fig, ax = plt.subplots(figsize=config.figsize)
    
    # Generate colors if not provided (use shared palette)
    if colors is None:
        colors = get_default_categorical_colors(len(group_data))
    
    # Plot each group
    for idx, (name, (x, y)) in enumerate(group_data.items()):
        color = colors[idx % len(colors)]
        ax.scatter(x, y, c=[color], s=point_size, alpha=alpha, label=name)
        
        # Add convex hull if requested and enough points
        if show_hulls and len(x) >= 3:
            try:
                points = np.column_stack([x, y])
                hull = ConvexHull(points)
                # Plot hull
                for simplex in hull.simplices:
                    ax.plot(points[simplex, 0], points[simplex, 1], 
                           color=color, alpha=0.3, linewidth=1)
            except Exception:
                # Skip hull if computation fails (e.g., collinear points)
                pass
    
    # Apply configuration consistently
    apply_layout_matplotlib(ax, config)
    ax.legend()
    
    finalize_plot_matplotlib(config)
    
    return ax


def _plot_grouped_scatter_2d_plotly(
    group_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    config: PlotConfig,
    show_hulls: bool,
    alpha: float,
    point_size: float,
    colors: List[str] | None,
) -> go.Figure:
    """Plotly implementation of grouped scatter plot."""
    fig = go.Figure()
    
    # Generate colors if not provided (use shared palette)
    if colors is None:
        colors = get_default_categorical_colors(len(group_data))
    
    # Plot each group
    for idx, (name, (x, y)) in enumerate(group_data.items()):
        color = colors[idx % len(colors)]
        
        # Add scatter trace
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='markers',
            name=name,
            marker=dict(size=point_size, color=color, opacity=alpha),
        ))
        
        # Add convex hull if requested
        if show_hulls and len(x) >= 3:
            try:
                points = np.column_stack([x, y])
                hull = ConvexHull(points)
                # Get hull vertices
                hull_points = points[hull.vertices]
                # Close the hull
                hull_x = np.append(hull_points[:, 0], hull_points[0, 0])
                hull_y = np.append(hull_points[:, 1], hull_points[0, 1])
                
                fig.add_trace(go.Scatter(
                    x=hull_x,
                    y=hull_y,
                    mode='lines',
                    line=dict(color=color, width=1),
                    showlegend=False,
                    opacity=0.3,
                ))
            except Exception:
                pass
    
    # Apply configuration consistently
    apply_layout_plotly(fig, config)
    
    finalize_plot_plotly(fig, config)
    
    return fig


def plot_kde_2d(
    x: np.ndarray,
    y: np.ndarray,
    config: PlotConfig | None = None,
    n_levels: int = 10,
    cmap: str = "Blues",
    fill: bool = True,
    alpha: float = 0.6,
    show_points: bool = False,
    point_size: float = 5,
    backend: Literal["matplotlib", "plotly"] | None = None,
) -> plt.Axes | go.Figure:
    """
    Create a 2D KDE (kernel density estimation) plot.
    
    Args:
        x: X coordinates
        y: Y coordinates
        config: Plot configuration
        n_levels: Number of contour levels
        cmap: Colormap name
        fill: Whether to fill contours
        alpha: Transparency level
        show_points: Whether to show underlying scatter points
        point_size: Size of scatter points if shown
        backend: Backend to use
        
    Returns:
        Matplotlib Axes or Plotly Figure
        
    Example:
        >>> x = np.random.randn(500)
        >>> y = np.random.randn(500)
        >>> plot_kde_2d(x, y, PlotConfig(title="2D Density"))
    """
    if len(x) != len(y):
        raise ValueError(f"x and y must have same length, got {len(x)} and {len(y)}")
    
    if len(x) < 2:
        raise ValueError("Need at least 2 points for KDE")
    
    if config is None:
        config = PlotConfig()
    
    backend_type = get_backend() if backend is None else BackendType(backend)
    
    if backend_type == BackendType.MATPLOTLIB:
        return _plot_kde_2d_matplotlib(
            x, y, config, n_levels, cmap, fill, alpha, show_points, point_size
        )
    else:
        if not PLOTLY_AVAILABLE:
            raise ValueError("Plotly backend requested but plotly is not installed")
        return _plot_kde_2d_plotly(
            x, y, config, n_levels, cmap, fill, alpha, show_points, point_size
        )


def _plot_kde_2d_matplotlib(
    x: np.ndarray,
    y: np.ndarray,
    config: PlotConfig,
    n_levels: int,
    cmap: str,
    fill: bool,
    alpha: float,
    show_points: bool,
    point_size: float,
) -> plt.Axes:
    """Matplotlib implementation of 2D KDE plot."""
    fig, ax = plt.subplots(figsize=config.figsize)
    
    # Calculate KDE
    values = np.vstack([x, y])
    kernel = gaussian_kde(values)
    
    # Create grid for evaluation
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    xi = np.linspace(x_min - 0.1 * x_range, x_max + 0.1 * x_range, 100)
    yi = np.linspace(y_min - 0.1 * y_range, y_max + 0.1 * y_range, 100)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    
    # Evaluate KDE on grid
    zi = kernel(np.vstack([xi_grid.flatten(), yi_grid.flatten()]))
    zi = zi.reshape(xi_grid.shape)
    
    # Plot contours
    mpl_cmap = resolve_colormap(cmap, BackendType.MATPLOTLIB)
    if fill:
        ax.contourf(xi_grid, yi_grid, zi, levels=n_levels, cmap=mpl_cmap, alpha=alpha)
    else:
        ax.contour(xi_grid, yi_grid, zi, levels=n_levels, cmap=mpl_cmap, alpha=alpha)
    
    # Show points if requested
    if show_points:
        ax.scatter(x, y, c='black', s=point_size, alpha=0.3, zorder=3)
    
    # Apply configuration consistently
    apply_layout_matplotlib(ax, config)
    
    finalize_plot_matplotlib(config)
    
    return ax


def _plot_kde_2d_plotly(
    x: np.ndarray,
    y: np.ndarray,
    config: PlotConfig,
    n_levels: int,
    cmap: str,
    fill: bool,
    alpha: float,
    show_points: bool,
    point_size: float,
) -> go.Figure:
    """Plotly implementation of 2D KDE plot."""
    # Calculate KDE
    values = np.vstack([x, y])
    kernel = gaussian_kde(values)
    
    # Create grid
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    xi = np.linspace(x_min - 0.1 * x_range, x_max + 0.1 * x_range, 100)
    yi = np.linspace(y_min - 0.1 * y_range, y_max + 0.1 * y_range, 100)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    
    # Evaluate KDE
    zi = kernel(np.vstack([xi_grid.flatten(), yi_grid.flatten()]))
    zi = zi.reshape(xi_grid.shape)
    
    fig = go.Figure()
    
    # Add contour
    fig.add_trace(go.Contour(
        x=xi,
        y=yi,
        z=zi,
        colorscale=resolve_colormap(cmap, BackendType.PLOTLY),
        ncontours=n_levels,
        opacity=alpha,
        contours=dict(
            coloring='fill' if fill else 'lines',
        ),
    ))
    
    # Show points if requested
    if show_points:
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='markers',
            marker=dict(size=point_size, color='black', opacity=0.3),
            showlegend=False,
        ))
    
    # Apply configuration consistently
    apply_layout_plotly(fig, config)
    
    finalize_plot_plotly(fig, config)
    
    return fig
