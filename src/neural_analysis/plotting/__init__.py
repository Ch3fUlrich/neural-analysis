"""
Neural Analysis Plotting Package

A modular, well-tested plotting library for neural data analysis.
Supports both matplotlib and plotly backends for flexible, publication-quality
and interactive visualizations.

Main modules:
- backend: Backend selection and management
- core: Core utilities, colors, and configuration
- plots_1d: One-dimensional plotting functions
- plots_2d: Two-dimensional plotting functions  
- plots_3d: Three-dimensional plotting functions
- embeddings: Embedding visualizations (2D/3D)
- heatmaps: Heatmap visualizations
- statistical: Statistical plots (histograms, KDE, distributions)
- neural: Neural-specific plots (rasters, traces)
- animations: Animation functions

Example usage:
    >>> from neural_analysis.plotting import set_backend, plot_line
    >>> set_backend('matplotlib')
    >>> plot_line(data, title="My Plot", xlabel="Time", ylabel="Value")
"""

from .backend import BackendType, get_backend, set_backend
from .core import PlotConfig
from .grid_config import (
    ColorScheme,
    GridLayoutConfig,
    PlotGrid,
    PlotSpec,
    add_trace_to_subplot,
    create_subplot_grid,
    plot_comparison_grid,
    plot_grouped_comparison,
)
from .heatmaps import (
    plot_heatmap,
)
from .plots_1d import (
    plot_boolean_states,
    plot_line,
    plot_multiple_lines,
)
from .plots_2d import (
    plot_grouped_scatter_2d,
    plot_kde_2d,
    plot_scatter_2d,
    plot_trajectory_2d,
)
from .plots_3d import (
    plot_scatter_3d,
    plot_trajectory_3d,
)
from .statistical_plots import (
    plot_bar,
    plot_box,
    plot_comparison_distributions,
    plot_grouped_distributions,
    plot_violin,
)

__version__ = "0.1.0"
__all__ = [
    "BackendType",
    "set_backend", 
    "get_backend",
    "PlotConfig",
    "plot_line",
    "plot_multiple_lines",
    "plot_boolean_states",
    "plot_scatter_2d",
    "plot_trajectory_2d",
    "plot_grouped_scatter_2d",
    "plot_kde_2d",
    "plot_scatter_3d",
    "plot_trajectory_3d",
    "plot_heatmap",
    "create_subplot_grid",
    "add_trace_to_subplot",
    "PlotSpec",
    "GridLayoutConfig",
    "ColorScheme",
    "PlotGrid",
    "plot_comparison_grid",
    "plot_grouped_comparison",
    "plot_bar",
    "plot_violin",
    "plot_box",
    "plot_grouped_distributions",
    "plot_comparison_distributions",
]
