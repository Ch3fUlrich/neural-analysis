"""
Core configuration dataclasses for the PlotGrid system.

This module provides the metadata-driven configuration types used by PlotGrid:
- PlotSpec: specification for a single plot element
- GridLayoutConfig: grid layout and subplot arrangement
- ColorScheme: color scheme for grouped plots
- PlotType: literal type for supported plot types

Dispatch logic and the PlotGrid class itself live in grid_dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd


def _convert_data_to_array(
    data: npt.NDArray[np.floating[Any]] | pd.DataFrame | dict[str, Any],
) -> npt.NDArray[np.floating[Any]]:
    """Convert data to numpy array format expected by render functions."""
    if isinstance(data, dict):
        # Extract x, y from dict
        if "x" in data and "y" in data:
            return np.column_stack([data["x"], data["y"]])
        else:
            raise ValueError("Dict data must have 'x' and 'y' keys")
    elif hasattr(data, "values"):  # DataFrame
        return data.values
    else:
        return np.asarray(data)


PlotType = Literal[
    "scatter",
    "line",
    "histogram",
    "heatmap",
    "scatter3d",
    "violin",
    "box",
    "bar",
    "trajectory",
    "trajectory3d",
    "kde",
    "grouped_scatter",
    "convex_hull",
    "boolean_states",
    "ellipse",
    "heatmap_walls",
]


@dataclass
class PlotSpec:
    """
    Specification for a single plot element.

    This defines what data to plot and how to style it.

    Parameters
    ----------
    data : np.ndarray or pd.DataFrame or dict
        Data to plot. Format depends on plot_type:
        - scatter: (n_samples, 2) or (n_samples, 3) array, or separate x,y arrays
        - line: (n_samples,) or (n_samples, n_lines) array
        - trajectory: (n_samples, 2) or (n_samples, 3) array with time-based coloring
        - histogram/kde: (n_samples,) array
        - heatmap: (n_rows, n_cols) array
        - violin/box: (n_samples,) array or list of arrays
        - grouped_scatter: dict mapping group names to (x, y) tuples
        - convex_hull: (n_samples, 2) array for boundary computation
    plot_type : PlotType
        Type of plot: 'scatter', 'line', 'trajectory', 'trajectory3d', 'histogram',
        'kde', 'heatmap', 'scatter3d', 'violin', 'box', 'bar', 'grouped_scatter', 'convex_hull'
    subplot_position : int, optional
        Which subplot this trace belongs to (0-indexed).
        If None, each spec gets its own subplot.
        Multiple specs with the same subplot_position are overlaid.
    title : str, optional
        Title for this subplot (only used if first spec for this position)
    label : str, optional
        Legend label for this trace
    color : str, optional
        Color for this trace (name, hex, or rgb string)
    marker : str, optional
        Marker style for scatter plots. For matplotlib: 'o', 's', '^', 'D', etc.
        For plotly: 'circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', etc.
    marker_size : float, optional
        Size of markers (for scatter plots)
    line_width : float, optional
        Width of lines (for line plots)
    linestyle : str, optional
        Line style. For matplotlib: '-', '--', '-.', ':', etc.
        For plotly: 'solid', 'dash', 'dot', 'dashdot', etc.
    error_y : np.ndarray, optional
        Error bar values for y-axis (for line plots with error bands)
    alpha : float, optional
        Transparency (0-1)

    # Advanced features for specialized plots
    color_by : Literal["time"] | None, optional
        Coloring strategy for plots. Currently supports:
        - "time": Color trajectory segments by time progression
        - None: Use default coloring (default)
        Future support planned for: "speed", "direction", etc.
    show_points : bool, optional
        For trajectory plots: show scatter points along trajectory
    cmap : str, optional
        Colormap name (e.g., 'viridis', 'plasma', 'Blues')
    colorbar : bool, optional
        Whether to show colorbar for color-mapped plots
    colorbar_label : str, optional
        Label for colorbar
    force_colorbar : bool, optional
        If True, always show colorbar even if same cmap+label combination exists.
        Overrides automatic deduplication. Default: False (auto-deduplicate).
    colors : np.ndarray or list, optional
        Array of color values or list of colors for grouped data
    sizes : np.ndarray or float, optional
        Array of sizes or single size value
    show_hulls : bool, optional
        For grouped_scatter: show convex hulls around groups
    hull_alpha : float, optional
        For grouped_scatter: transparency for hull fill (0-1)
    fill : bool, optional
        For kde plots: fill contours
    n_levels : int, optional
        For kde plots: number of contour levels
    bandwidth : float, optional
        For kde plots: KDE bandwidth parameter
    equal_aspect : bool, optional
        Whether to use equal aspect ratio

    # Reference lines and annotations (for line plots)
    vlines : list of dict, optional
        Vertical reference lines. Each dict should contain:
        - 'x': float - x-coordinate for the line
        - 'color': str, optional - line color (default: 'black')
        - 'linestyle': str, optional - line style (default: '--')
        - 'linewidth': float, optional - line width (default: 1.5)
        - 'label': str, optional - legend label
        - 'alpha': float, optional - transparency (default: 0.7)
    hlines : list of dict, optional
        Horizontal reference lines. Each dict should contain:
        - 'y': float - y-coordinate for the line
        - 'color': str, optional - line color (default: 'black')
        - 'linestyle': str, optional - line style (default: '--')
        - 'linewidth': float, optional - line width (default: 1.5)
        - 'label': str, optional - legend label
        - 'alpha': float, optional - transparency (default: 0.7)
    annotations : list of dict, optional
        Text annotations. Each dict should contain:
        - 'text': str - annotation text
        - 'xy': tuple - (x, y) point to annotate
        - 'xytext': tuple, optional - (x, y) position for text
        - 'fontsize': float, optional - font size (default: 10)
        - 'bbox': dict, optional - bounding box properties
        - 'arrowprops': dict, optional - arrow properties

    kwargs : dict, optional
        Additional plot-specific arguments passed to underlying renderers
    """

    data: npt.NDArray[np.floating[Any]] | pd.DataFrame | dict[str, Any]
    plot_type: PlotType
    subplot_position: int | None = None
    title: str | None = None
    label: str | None = None
    color: str | None = None
    marker: str | None = None
    marker_size: float | None = None
    line_width: float | None = None
    linestyle: str | None = None
    error_y: npt.NDArray[np.floating[Any]] | None = None
    alpha: float = 0.7

    # Advanced features
    color_by: Literal["time"] | None = None
    show_points: bool = False
    cmap: str | None = None
    colorbar: bool = False
    colorbar_label: str | None = None
    force_colorbar: bool = False
    colors: npt.NDArray[np.floating[Any]] | list[Any] | None = None
    sizes: npt.NDArray[np.floating[Any]] | float | None = None
    show_hulls: bool = False
    hull_alpha: float | None = None
    fill: bool = True
    n_levels: int = 10
    bandwidth: float | None = None
    equal_aspect: bool = False

    # Reference lines for line plots
    vlines: list[dict[str, Any]] | None = (
        None  # Vertical reference lines: [{'x': value, 'color': 'red', 'linestyle': '--', 'linewidth': 2, 'label': 'label'}]
    )
    hlines: list[dict[str, Any]] | None = (
        None  # Horizontal reference lines: [{'y': value, 'color': 'blue', 'linestyle': ':', 'linewidth': 1, 'label': 'label'}]
    )
    annotations: list[dict[str, Any]] | None = (
        None  # Annotations: [{'text': 'label', 'xy': (x, y), 'xytext': (x, y), 'fontsize': 10, 'bbox': {...}, 'arrowprops': {...}}]
    )

    # Boolean states parameters
    true_color: str | None = None
    false_color: str | None = None
    true_label: str | None = None
    false_label: str | None = None

    # Ellipse plot parameters (for plot_type='ellipse')
    ellipse_widths: npt.NDArray[np.floating[Any]] | None = None
    ellipse_heights: npt.NDArray[np.floating[Any]] | None = None
    ellipse_angles: npt.NDArray[np.floating[Any]] | None = None

    kwargs: dict[str, Any] = field(default_factory=dict)

    # Internal attribute for legend handles (set dynamically during plotting)
    _legend_handle: Any = field(default=None, init=False, repr=False)


@dataclass
class GridLayoutConfig:
    """
    Configuration for grid layout and subplot arrangement.

    Parameters
    ----------
    rows : int, optional
        Number of rows. If None, auto-calculated from n_plots
    cols : int, optional
        Number of columns. If None, auto-calculated from n_plots
    subplot_titles : list of str, optional
        Title for each subplot position
    shared_xaxes : bool or str, default=False
        Share x-axes: True, False, 'all', 'rows', 'columns'
    shared_yaxes : bool or str, default=False
        Share y-axes: True, False, 'all', 'rows', 'columns'
    vertical_spacing : float, optional
        Vertical space between subplots (0-1)
    horizontal_spacing : float, optional
        Horizontal space between subplots (0-1)
    group_by : str, optional
        Column name in DataFrame to group plots by (auto-arrange in grid)
    width_ratios : list of float, optional
        Relative widths of columns. Length must equal cols.
        For uneven grids: e.g., [1, 2, 2] makes first column half width of others.
    height_ratios : list of float, optional
        Relative heights of rows. Length must equal rows.
        For uneven grids: e.g., [1, 2] makes first row half height of second.
    """

    rows: int | None = None
    cols: int | None = None
    subplot_titles: list[str] | None = None
    shared_xaxes: bool | str = False
    shared_yaxes: bool | str = False
    vertical_spacing: float | None = None
    horizontal_spacing: float | None = None
    group_by: str | None = None
    width_ratios: list[float] | None = None
    height_ratios: list[float] | None = None

    def auto_size_grid(self, n_plots: int) -> tuple[int, int]:
        """
        Automatically determine grid size from number of plots.

        Parameters
        ----------
        n_plots : int
            Number of plots to arrange

        Returns
        -------
        rows : int
            Number of rows
        cols : int
            Number of columns
        """
        if self.rows is not None and self.cols is not None:
            return self.rows, self.cols

        if self.rows is not None:
            cols = int(np.ceil(n_plots / self.rows))
            return self.rows, cols

        if self.cols is not None:
            rows = int(np.ceil(n_plots / self.cols))
            return rows, self.cols

        # Auto-determine square-ish grid
        cols = int(np.ceil(np.sqrt(n_plots)))
        rows = int(np.ceil(n_plots / cols))
        return rows, cols


@dataclass
class ColorScheme:
    """
    Color scheme for grouped plots.

    Parameters
    ----------
    palette : str or list of str
        Color palette name ('viridis', 'plasma', 'tab10') or list of colors
    group_colors : dict, optional
        Explicit color mapping: {group_name: color}
    alpha : float, default=0.7
        Default transparency for all colors
    """

    palette: str | list[str] = "tab10"
    group_colors: dict[str, str] | None = None
    alpha: float = 0.7

    def get_colors(self, groups: Sequence[str]) -> dict[str, str]:
        """
        Get color mapping for a list of groups.

        Parameters
        ----------
        groups : sequence of str
            Group names

        Returns
        -------
        dict
            Mapping from group name to color
        """
        if self.group_colors is not None:
            return self.group_colors

        # Use matplotlib or plotly color schemes
        if isinstance(self.palette, list):
            colors = self.palette
        else:
            # Map common palette names to colors
            color_palettes = {
                "tab10": [
                    "#1f77b4",
                    "#ff7f0e",
                    "#2ca02c",
                    "#d62728",
                    "#9467bd",
                    "#8c564b",
                    "#e377c2",
                    "#7f7f7f",
                    "#bcbd22",
                    "#17becf",
                ],
                "viridis": ["#440154", "#31688e", "#35b779", "#fde724"],
                "plasma": ["#0d0887", "#7e03a8", "#cc4778", "#f89540", "#f0f921"],
                "Set1": [
                    "#e41a1c",
                    "#377eb8",
                    "#4daf4a",
                    "#984ea3",
                    "#ff7f00",
                    "#ffff33",
                    "#a65628",
                    "#f781bf",
                    "#999999",
                ],
            }
            colors = color_palettes.get(self.palette, color_palettes["tab10"])

        # Cycle colors if more groups than colors
        n_colors = len(colors)
        return {group: colors[i % n_colors] for i, group in enumerate(groups)}


# Backward compatibility: re-export dispatch symbols from grid_dispatch
from neural_analysis.plotting.grid_dispatch import (  # noqa: E402
    PlotGrid as PlotGrid,
)
from neural_analysis.plotting.grid_dispatch import (  # noqa: E402
    add_trace_to_subplot as add_trace_to_subplot,
)
from neural_analysis.plotting.grid_dispatch import (  # noqa: E402
    create_subplot_grid as create_subplot_grid,
)
from neural_analysis.plotting.grid_dispatch import (  # noqa: E402
    plot_comparison_grid as plot_comparison_grid,
)
from neural_analysis.plotting.grid_dispatch import (  # noqa: E402
    plot_grouped_comparison as plot_grouped_comparison,
)
