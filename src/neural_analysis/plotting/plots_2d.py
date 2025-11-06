"""Two-dimensional plotting functions for neural data visualization.

This module provides functions for creating 2D scatter plots, trajectory visualizations,
KDE density plots, and grouped scatter plots with optional convex hulls.
Functions support both matplotlib and plotly backends for flexibility between
static publication-quality figures and interactive exploratory visualizations.

All functions in this module use the PlotGrid system internally for consistent
behavior and code reuse.
"""

from typing import Dict, List, Tuple, Optional, Literal
import numpy as np
import matplotlib.pyplot as plt

from .backend import BackendType, get_backend
from .core import PlotConfig
from .grid_config import PlotGrid, PlotSpec

# Optional imports
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

__all__ = [
    "plot_scatter_2d",
    "plot_trajectory_2d",
    "plot_grouped_scatter_2d",
    "plot_kde_2d",
]

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
) -> "plt.Axes | go.Figure":
    """
    Create a 2D scatter plot.
    
    This is a convenience function for creating simple scatter plots.
    For multi-panel layouts, use PlotGrid with PlotSpec directly.
    
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
    # Validate backend if provided
    if backend is not None and backend not in ("matplotlib", "plotly"):
        raise ValueError(
            f"Invalid backend '{backend}'. Must be 'matplotlib' or 'plotly'"
        )
    
    # Input validation
    if len(x) != len(y):
        raise ValueError(f"x and y must have same length, got {len(x)} and {len(y)}")
    
    # Default config
    if config is None:
        config = PlotConfig()
    
    # Prepare data as (n, 2) array for PlotGrid
    data = np.column_stack([x, y])
    
    # Create PlotSpec for scatter plot
    spec = PlotSpec(
        data=data,
        plot_type='scatter',
        color=colors if isinstance(colors, str) else None,
        colors=colors if isinstance(colors, np.ndarray) else None,
        marker_size=sizes if isinstance(sizes, (int, float)) else None,
        sizes=sizes if isinstance(sizes, np.ndarray) else None,
        alpha=alpha,
        cmap=cmap,
        colorbar=colorbar,
        colorbar_label=colorbar_label,
    )
    
    # Create PlotGrid and plot
    grid = PlotGrid(
        plot_specs=[spec],
        config=config,
        backend=backend,
    )
    
    return grid.plot()


def plot_trajectory_2d(
    x: np.ndarray,
    y: np.ndarray,
    config: PlotConfig | None = None,
    color_by: Literal["time"] | None = "time",
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
        color_by: Coloring strategy. Use "time" for time progression, or None for solid color
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
    
    # Create PlotSpec for trajectory plot
    spec = PlotSpec(
        data={'x': x, 'y': y},
        plot_type='trajectory',
        color_by=color_by,
        cmap=cmap,
        line_width=linewidth,
        show_points=show_points,
        marker_size=point_size,
        alpha=alpha,
        equal_aspect=True,
    )
    
    # Create PlotGrid and plot
    grid = PlotGrid(
        plot_specs=[spec],
        config=config,
        backend=backend,
    )
    
    return grid.plot()


def plot_grouped_scatter_2d(
    group_data: dict[str, tuple[np.ndarray, np.ndarray]],
    config: PlotConfig | None = None,
    show_hulls: bool = True,
    hull_alpha: float = 0.2,
    point_size: float = 20,
    point_alpha: float = 0.7,
    colors: list[str] | None = None,
    backend: Literal["matplotlib", "plotly"] | None = None,
) -> plt.Axes | go.Figure:
    """
    Plot grouped scatter data with optional convex hulls.
    
    Args:
        group_data: Dictionary mapping group names to (x, y) tuples
        config: Plot configuration
        show_hulls: Whether to show convex hulls around groups
        hull_alpha: Transparency for hull fill
        point_size: Size of scatter points
        point_alpha: Transparency for points
        colors: Optional list of colors for groups
        backend: Backend to use
    
    Returns:
        Matplotlib Axes or Plotly Figure
        
    Example:
        >>> groups = {
        ...     'Group A': (np.random.randn(50), np.random.randn(50)),
        ...     'Group B': (np.random.randn(50) + 2, np.random.randn(50) + 2),
        ... }
        >>> plot_grouped_scatter_2d(groups, PlotConfig(title="Grouped Data"))
    """
    if config is None:
        config = PlotConfig()
    
    # Validate input
    if not group_data:
        raise ValueError("group_data cannot be empty")
    
    for name, (x, y) in group_data.items():
        if len(x) != len(y):
            raise ValueError(f"x and y must have same length for group '{name}'")
    
    # Create PlotSpec for grouped scatter plot
    spec = PlotSpec(
        data=group_data,
        plot_type='grouped_scatter',
        show_hulls=show_hulls,
        hull_alpha=hull_alpha,
        marker_size=point_size,
        alpha=point_alpha,
        colors=colors,
        equal_aspect=True,
    )
    
    # Create PlotGrid and plot
    grid = PlotGrid(
        plot_specs=[spec],
        config=config,
        backend=backend,
    )
    
    return grid.plot()


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
    bandwidth: float | None = None,
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
        bandwidth: KDE bandwidth (bw_method). If None, uses Scott's rule
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
    
    # Create PlotSpec for KDE plot
    spec = PlotSpec(
        data={'x': x, 'y': y},
        plot_type='kde',
        n_levels=n_levels,
        cmap=cmap,
        fill=fill,
        alpha=alpha,
        show_points=show_points,
        marker_size=point_size,
        bandwidth=bandwidth,
    )
    
    # Create PlotGrid and plot
    grid = PlotGrid(
        plot_specs=[spec],
        config=config,
        backend=backend,
    )
    
    return grid.plot()
