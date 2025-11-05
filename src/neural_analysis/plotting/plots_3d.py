"""
Three-dimensional plotting functions for neural data visualization.

This module provides convenience functions for creating 3D visualizations including:
- 3D scatter plots with color mapping
- 3D trajectory/line plots
- 3D embeddings with optional convex hulls

All functions in this module use the PlotGrid system internally to ensure
consistent behavior and eliminate code duplication.

Functions support both matplotlib and plotly backends for flexibility between
static publication-quality figures and interactive exploratory visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Literal
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from .backend import BackendType
from .core import PlotConfig
from .grid_config import PlotGrid, PlotSpec

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
    
    if config is None:
        config = PlotConfig()
    
    # Prepare data as (n, 3) array
    data = np.column_stack([x, y, z])
    
    # Create PlotSpec
    spec = PlotSpec(
        data=data,
        plot_type='scatter3d',
        color=colors if isinstance(colors, str) else None,
        colors=colors if isinstance(colors, np.ndarray) else None,
        marker=marker,
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


def plot_trajectory_3d(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    config: PlotConfig | None = None,
    color_by: Literal["time"] | None = "time",
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
    
    # Prepare data as (n, 3) array or dict
    data = {'x': x, 'y': y, 'z': z}
    
    # Create PlotSpec
    spec = PlotSpec(
        data=data,
        plot_type='trajectory3d',
        color_by=color_by,
        cmap=cmap,
        line_width=linewidth,
        show_points=show_points,
        marker_size=point_size,
        alpha=alpha,
    )
    
    # Create PlotGrid and plot
    grid = PlotGrid(
        plot_specs=[spec],
        config=config,
        backend=backend,
    )
    
    return grid.plot()
