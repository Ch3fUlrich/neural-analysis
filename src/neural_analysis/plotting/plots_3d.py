"""
Three-dimensional plotting functions for neural data visualization.

This module provides functions for creating 3D visualizations including:
- 3D scatter plots with color mapping
- 3D trajectory/line plots
- 3D embeddings with optional convex hulls

Functions support both matplotlib and plotly backends for flexibility between
static publication-quality figures and interactive exploratory visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Literal
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from .backend import BackendType, get_backend
from .core import (
    PlotConfig,
    resolve_colormap,
    apply_layout_matplotlib,
    apply_layout_plotly_3d,
    create_rgba_labels,
    finalize_plot_matplotlib,
    finalize_plot_plotly,
)

# Optional imports
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

__all__ = [
    "plot_scatter_3d",
    "plot_trajectory_3d",
]


def plot_scatter_3d(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    config: PlotConfig | None = None,
    colors: np.ndarray | str | None = None,
    sizes: np.ndarray | float = 20,
    alpha: float = 0.6,
    cmap: str = "viridis",
    colorbar: bool = False,
    colorbar_label: str | None = None,
    marker: str = "o",
    backend: Literal["matplotlib", "plotly"] | None = None,
) -> plt.Axes | go.Figure:
    """
    Create a 3D scatter plot.
    
    Args:
        x: X coordinates (1D array)
        y: Y coordinates (1D array)
        z: Z coordinates (1D array)
        config: Plot configuration (title, labels, etc.)
        colors: Color values (array for colormap, string for single color,
            or None for default)
        sizes: Point sizes (array or single value)
        alpha: Transparency level (0-1)
        cmap: Colormap name when colors is an array
        colorbar: Whether to show colorbar (matplotlib only, when colors is array)
        colorbar_label: Label for colorbar
        marker: Marker style
        backend: Backend to use ('matplotlib' or 'plotly')
        
    Returns:
        Matplotlib Axes or Plotly Figure object
        
    Example:
        >>> x = np.random.randn(100)
        >>> y = np.random.randn(100)
        >>> z = np.random.randn(100)
        >>> plot_scatter_3d(x, y, z, PlotConfig(title="3D Points"))
    """
    # Input validation
    if not (len(x) == len(y) == len(z)):
        raise ValueError(
            f"x, y, z must have same length, got {len(x)}, {len(y)}, {len(z)}"
        )
    
    # Default config
    if config is None:
        config = PlotConfig()
    
    # Determine backend
    backend_type = get_backend() if backend is None else BackendType(backend)
    
    if backend_type == BackendType.MATPLOTLIB:
        return _plot_scatter_3d_matplotlib(
            x, y, z, config, colors, sizes, alpha, cmap, colorbar, colorbar_label, marker
        )
    else:
        if not PLOTLY_AVAILABLE:
            raise ValueError("Plotly backend requested but plotly is not installed")
        return _plot_scatter_3d_plotly(
            x, y, z, config, colors, sizes, alpha, cmap, colorbar_label, marker
        )


def _plot_scatter_3d_matplotlib(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    config: PlotConfig,
    colors: np.ndarray | str | None,
    sizes: np.ndarray | float,
    alpha: float,
    cmap: str,
    colorbar: bool,
    colorbar_label: str | None,
    marker: str,
) -> plt.Axes:
    """Matplotlib implementation of 3D scatter plot."""
    fig = plt.figure(figsize=config.figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Determine color parameter and whether to use colormap
    if colors is None:
        color_param = 'C0'
        use_cmap = False
    elif isinstance(colors, str):
        color_param = colors
        use_cmap = False
    else:
        color_param = colors
        use_cmap = True
    
    # Create scatter
    if use_cmap:
        mpl_cmap = resolve_colormap(cmap, BackendType.MATPLOTLIB)
        scatter = ax.scatter(
            x, y, z, c=color_param, s=sizes, alpha=alpha, cmap=mpl_cmap, marker=marker
        )
    else:
        scatter = ax.scatter(
            x, y, z, c=color_param, s=sizes, alpha=alpha, marker=marker
        )
    
    # Add colorbar if requested and colors is an array
    if colorbar and use_cmap:
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.1)
        if colorbar_label:
            cbar.set_label(colorbar_label)
    
    # Apply layout configuration
    apply_layout_matplotlib(ax, config)
    
    # Set 3D-specific background styling
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    
    finalize_plot_matplotlib(config)
    
    return ax


def _plot_scatter_3d_plotly(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    config: PlotConfig,
    colors: np.ndarray | str | None,
    sizes: np.ndarray | float,
    alpha: float,
    cmap: str,
    colorbar_label: str | None,
    marker: str,
) -> go.Figure:
    """Plotly implementation of 3D scatter plot."""
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
    
    fig.add_trace(go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=marker_dict,
    ))
    
    # Apply layout
    apply_layout_plotly_3d(fig, config)
    
    finalize_plot_plotly(fig, config)
    
    return fig


def plot_trajectory_3d(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    config: PlotConfig | None = None,
    color_by_time: bool = True,
    cmap: str = "viridis",
    linewidth: float = 2.0,
    show_points: bool = True,
    point_size: float = 10,
    alpha: float = 0.7,
    backend: Literal["matplotlib", "plotly"] | None = None,
) -> plt.Axes | go.Figure:
    """
    Plot a 3D trajectory with line connecting points.
    
    Args:
        x: X coordinates
        y: Y coordinates
        z: Z coordinates
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
        >>> z = t / 4
        >>> plot_trajectory_3d(x, y, z, PlotConfig(title="3D Trajectory"))
    """
    # Input validation
    if not (len(x) == len(y) == len(z)):
        raise ValueError(
            f"x, y, z must have same length, got {len(x)}, {len(y)}, {len(z)}"
        )
    
    if config is None:
        config = PlotConfig()
    
    backend_type = get_backend() if backend is None else BackendType(backend)
    
    if backend_type == BackendType.MATPLOTLIB:
        return _plot_trajectory_3d_matplotlib(
            x, y, z, config, color_by_time, cmap, linewidth, show_points, point_size, alpha
        )
    else:
        if not PLOTLY_AVAILABLE:
            raise ValueError("Plotly backend requested but plotly is not installed")
        return _plot_trajectory_3d_plotly(
            x, y, z, config, color_by_time, cmap, linewidth, show_points, point_size, alpha
        )


def _plot_trajectory_3d_matplotlib(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    config: PlotConfig,
    color_by_time: bool,
    cmap: str,
    linewidth: float,
    show_points: bool,
    point_size: float,
    alpha: float,
) -> plt.Axes:
    """Matplotlib implementation of 3D trajectory plot."""
    fig = plt.figure(figsize=config.figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    if color_by_time:
        # Create line with color gradient
        from mpl_toolkits.mplot3d.art3d import Line3DCollection
        
        points = np.array([x, y, z]).T.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # Create colors for each segment
        colors = np.linspace(0, 1, len(x))
        mpl_cmap = resolve_colormap(cmap, BackendType.MATPLOTLIB)
        
        lc = Line3DCollection(segments, cmap=mpl_cmap, linewidths=linewidth, alpha=alpha)
        lc.set_array(colors)
        ax.add_collection3d(lc)
        
        # Add colorbar
        plt.colorbar(lc, ax=ax, label='Time', shrink=0.8, pad=0.1)
    else:
        # Simple line plot
        ax.plot(x, y, z, linewidth=linewidth, alpha=alpha)
    
    # Add scatter points if requested
    if show_points:
        if color_by_time:
            colors_scatter = np.arange(len(x))
            mpl_cmap = resolve_colormap(cmap, BackendType.MATPLOTLIB)
            ax.scatter(x, y, z, c=colors_scatter, s=point_size, cmap=mpl_cmap, alpha=alpha)
        else:
            ax.scatter(x, y, z, s=point_size, alpha=alpha)
    
    # Apply layout configuration
    apply_layout_matplotlib(ax, config)
    
    # Set 3D-specific background styling
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    
    finalize_plot_matplotlib(config)
    
    return ax


def _plot_trajectory_3d_plotly(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    config: PlotConfig,
    color_by_time: bool,
    cmap: str,
    linewidth: float,
    show_points: bool,
    point_size: float,
    alpha: float,
) -> go.Figure:
    """Plotly implementation of 3D trajectory plot."""
    fig = go.Figure()
    
    # Prepare marker and line properties
    marker_dict = {'size': point_size, 'opacity': alpha}
    line_dict = {'width': linewidth}
    
    if color_by_time:
        colors = np.arange(len(x))
        marker_dict['color'] = colors.tolist()
        marker_dict['colorscale'] = resolve_colormap(cmap, BackendType.PLOTLY)
        marker_dict['showscale'] = True
        marker_dict['colorbar'] = {'title': 'Time'}
        line_dict['color'] = colors.tolist()
        line_dict['colorscale'] = resolve_colormap(cmap, BackendType.PLOTLY)
    
    mode = 'lines+markers' if show_points else 'lines'
    
    fig.add_trace(go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode=mode,
        line=line_dict,
        marker=marker_dict if show_points else {},
        opacity=alpha,
    ))
    
    # Apply layout
    apply_layout_plotly_3d(fig, config)
    
    finalize_plot_plotly(fig, config)
    
    return fig
