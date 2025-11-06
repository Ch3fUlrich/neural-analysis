"""
Flexible grid configuration system for multi-panel plotting.

This module provides a metadata-driven approach to creating complex
multi-panel plots with minimal code. Instead of manually managing subplot
positions and styling, users provide a structured configuration that
specifies what to plot, where, and how.

The PlotGrid system is the unified plotting interface that supports:
- Basic plots: scatter, line, histogram, heatmap, bar, violin, box
- Advanced 2D plots: trajectory (with time-gradient coloring), kde (contour density), 
  grouped_scatter (with convex hulls)
- 3D plots: scatter3d, trajectory3d (with time-gradient coloring)
- Specialized plots: convex_hull (boundary rendering)

All convenience functions in plots_2d.py and plots_3d.py can be replicated
using PlotGrid with appropriate PlotSpec configurations. The PlotGrid system
provides a consistent interface across all plot types and backends (matplotlib/plotly).

Example Usage:
    >>> # Simple trajectory with time coloring
    >>> spec = PlotSpec(
    ...     data={'x': x_data, 'y': y_data},
    ...     plot_type='trajectory',
    ...     color_by="time",
    ...     show_points=True,
    ...     cmap='viridis'
    ... )
    >>> grid = PlotGrid(plot_specs=[spec])
    >>> fig = grid.plot()
    
    >>> # KDE density plot
    >>> spec = PlotSpec(
    ...     data={'x': x_data, 'y': y_data},
    ...     plot_type='kde',
    ...     fill=True,
    ...     n_levels=15,
    ...     show_points=True
    ... )
    >>> grid = PlotGrid(plot_specs=[spec])
    >>> fig = grid.plot()
    
    >>> # Grouped scatter with convex hulls
    >>> spec = PlotSpec(
    ...     data={'Group A': (x1, y1), 'Group B': (x2, y2)},
    ...     plot_type='grouped_scatter',
    ...     show_hulls=True
    ... )
    >>> grid = PlotGrid(plot_specs=[spec])
    >>> fig = grid.plot()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence
import numpy as np
import numpy.typing as npt
import pandas as pd

import matplotlib.pyplot as plt

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None
    make_subplots = None

from .core import PlotConfig, get_default_categorical_colors, resolve_colormap
from .backend import BackendType, get_backend
from . import renderers
from .renderers import extract_xy_from_data, extract_xyz_from_data
from neural_analysis.utils.trajectories import compute_colors
from neural_analysis.utils.geometry import compute_kde_2d, compute_convex_hull
from neural_analysis.plotting.renderers import (
    render_trajectory_matplotlib,
    render_kde_matplotlib,
    render_convex_hull_matplotlib,
    render_trajectory_plotly,
    render_trajectory3d_matplotlib,
    render_trajectory3d_plotly,
    render_kde_plotly,
    render_convex_hull_plotly,
)


PlotType = Literal['scatter', 'line', 'histogram', 'heatmap', 'scatter3d', 'violin', 'box', 'bar', 
                   'trajectory', 'trajectory3d', 'kde', 'grouped_scatter', 'convex_hull', 'boolean_states']

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
    data: np.ndarray | pd.DataFrame | dict
    plot_type: PlotType
    subplot_position: int | None = None
    title: str | None = None
    label: str | None = None
    color: str | None = None
    marker: str | None = None
    marker_size: float | None = None
    line_width: float | None = None
    linestyle: str | None = None
    error_y: npt.NDArray | None = None
    alpha: float = 0.7
    
    # Advanced features
    color_by: Literal["time"] | None = None
    show_points: bool = False
    cmap: str | None = None
    colorbar: bool = False
    colorbar_label: str | None = None
    colors: np.ndarray | list | None = None
    sizes: np.ndarray | float | None = None
    show_hulls: bool = False
    hull_alpha: float | None = None
    fill: bool = True
    n_levels: int = 10
    bandwidth: float | None = None
    equal_aspect: bool = False
    
    # Reference lines for line plots
    vlines: list[dict] | None = None  # Vertical reference lines: [{'x': value, 'color': 'red', 'linestyle': '--', 'linewidth': 2, 'label': 'label'}]
    hlines: list[dict] | None = None  # Horizontal reference lines: [{'y': value, 'color': 'blue', 'linestyle': ':', 'linewidth': 1, 'label': 'label'}]
    annotations: list[dict] | None = None  # Annotations: [{'text': 'label', 'xy': (x, y), 'xytext': (x, y), 'fontsize': 10, 'bbox': {...}, 'arrowprops': {...}}]
    
    # Boolean states parameters
    true_color: str | None = None
    false_color: str | None = None
    true_label: str | None = None
    false_label: str | None = None
    
    kwargs: dict[str, Any] = field(default_factory=dict)


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
    """
    rows: int | None = None
    cols: int | None = None
    subplot_titles: list[str] | None = None
    shared_xaxes: bool | str = False
    shared_yaxes: bool | str = False
    vertical_spacing: float | None = None
    horizontal_spacing: float | None = None
    group_by: str | None = None
    
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
    palette: str | list[str] = 'tab10'
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
                'tab10': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                         '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
                'viridis': ['#440154', '#31688e', '#35b779', '#fde724'],
                'plasma': ['#0d0887', '#7e03a8', '#cc4778', '#f89540', '#f0f921'],
                'Set1': ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
                        '#ffff33', '#a65628', '#f781bf', '#999999'],
            }
            colors = color_palettes.get(self.palette, color_palettes['tab10'])
        
        # Cycle colors if more groups than colors
        n_colors = len(colors)
        return {group: colors[i % n_colors] for i, group in enumerate(groups)}


class PlotGrid:
    """
    Flexible grid-based plotting system with metadata-driven configuration.
    
    This class provides a high-level interface for creating complex multi-panel
    plots. Instead of manually managing subplot positions and styling, you
    provide a structured configuration via a DataFrame or list of PlotSpec objects.
    
    The system supports:
    - Automatic grid layout based on data grouping
    - Flexible plot types (scatter, line, histogram, heatmap, 3D)
    - Color schemes applied to groups/categories
    - Single entry point for all plotting needs
    
    Parameters
    ----------
    config : PlotConfig, optional
        Overall plot configuration (size, title, etc.)
    layout : GridLayoutConfig, optional
        Grid layout configuration
    color_scheme : ColorScheme, optional
        Color scheme for grouped data
    backend : {'matplotlib', 'plotly'}, optional
        Plotting backend to use
        
    Examples
    --------
    >>> # Create from DataFrame
    >>> df = pd.DataFrame({
    ...     'data': [data1, data2, data3, data4],
    ...     'plot_type': ['scatter', 'scatter', 'line', 'histogram'],
    ...     'title': ['A', 'B', 'C', 'D'],
    ...     'group': ['control', 'treatment', 'control', 'treatment']
    ... })
    >>> grid = PlotGrid.from_dataframe(df, group_by='group')
    >>> fig = grid.plot()
    
    >>> # Create from list of PlotSpecs
    >>> specs = [
    ...     PlotSpec(data=data1, plot_type='scatter', title='Scatter 1'),
    ...     PlotSpec(data=data2, plot_type='line', title='Line 1', color='red'),
    ... ]
    >>> grid = PlotGrid(plot_specs=specs)
    >>> fig = grid.plot()
    """
    
    def __init__(
        self,
        plot_specs: list[PlotSpec] | None = None,
        config: PlotConfig | None = None,
        layout: GridLayoutConfig | None = None,
        color_scheme: ColorScheme | None = None,
        backend: Literal['matplotlib', 'plotly'] | None = None,
    ):
        self.plot_specs = plot_specs or []
        self.config = config or PlotConfig()
        self.layout = layout or GridLayoutConfig()
        self.color_scheme = color_scheme or ColorScheme()
        self.backend = backend
        
    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        data_col: str = 'data',
        plot_type_col: str = 'plot_type',
        title_col: str | None = 'title',
        label_col: str | None = 'label',
        color_col: str | None = 'color',
        group_by: str | None = None,
        **kwargs,
    ) -> PlotGrid:
        """
        Create PlotGrid from a pandas DataFrame.
        
        The DataFrame should have columns specifying what data to plot,
        what type of plot, and styling information.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with plot specifications
        data_col : str, default='data'
            Column name containing data arrays
        plot_type_col : str, default='plot_type'
            Column name specifying plot type
        title_col : str, optional
            Column name for subplot titles
        label_col : str, optional
            Column name for legend labels
        color_col : str, optional
            Column name for colors
        group_by : str, optional
            Column name to group plots by
        **kwargs
            Additional arguments passed to PlotGrid constructor
            
        Returns
        -------
        PlotGrid
            Configured plot grid
            
        Examples
        --------
        >>> df = pd.DataFrame({
        ...     'data': [arr1, arr2, arr3],
        ...     'plot_type': ['scatter', 'line', 'histogram'],
        ...     'title': ['Plot A', 'Plot B', 'Plot C'],
        ...     'group': ['control', 'treatment', 'control']
        ... })
        >>> grid = PlotGrid.from_dataframe(df, group_by='group')
        """
        plot_specs = []
        
        for idx, row in df.iterrows():
            spec = PlotSpec(
                data=row[data_col],
                plot_type=row[plot_type_col],
                title=row[title_col] if title_col and title_col in row else None,
                label=row[label_col] if label_col and label_col in row else None,
                color=row[color_col] if color_col and color_col in row else None,
            )
            plot_specs.append(spec)
        
        # Auto-configure layout if group_by specified
        if group_by and group_by in df.columns:
            layout = GridLayoutConfig(group_by=group_by)
            groups = df[group_by].unique()
            
            # Auto-assign colors by group
            color_scheme = kwargs.pop('color_scheme', ColorScheme())
            group_colors = color_scheme.get_colors(groups)
            
            # Apply colors to specs if not already specified
            for spec, (_, row) in zip(plot_specs, df.iterrows()):
                if spec.color is None and group_by in row:
                    spec.color = group_colors[row[group_by]]
            
            kwargs['layout'] = layout
            kwargs['color_scheme'] = color_scheme
        
        return cls(plot_specs=plot_specs, **kwargs)
    
    @classmethod
    def from_dict(
        cls,
        data_dict: dict[str, np.ndarray],
        plot_type: PlotType = 'scatter',
        **kwargs,
    ) -> PlotGrid:
        """
        Create PlotGrid from a dictionary of {label: data}.
        
        Parameters
        ----------
        data_dict : dict
            Dictionary mapping labels to data arrays
        plot_type : PlotType, default='scatter'
            Type of plot for all data ('scatter', 'line', 'histogram', etc.)
        **kwargs
            Additional arguments passed to PlotGrid constructor
            
        Returns
        -------
        PlotGrid
            Configured plot grid
            
        Examples
        --------
        >>> data = {'Control': arr1, 'Treatment': arr2, 'Test': arr3}
        >>> grid = PlotGrid.from_dict(data, plot_type='histogram')
        """
        plot_specs = [
            PlotSpec(data=data, plot_type=plot_type, title=label, label=label)
            for label, data in data_dict.items()
        ]
        return cls(plot_specs=plot_specs, **kwargs)
    
    def add_plot(
        self,
        data: np.ndarray | pd.DataFrame,
        plot_type: PlotType,
        **kwargs,
    ) -> None:
        """
        Add a plot to the grid.
        
        Parameters
        ----------
        data : array or DataFrame
            Data to plot
        plot_type : PlotType
            Type of plot ('scatter', 'line', 'histogram', 'heatmap', 
            'scatter3d', 'violin', 'box', 'bar')
        **kwargs
            Additional PlotSpec parameters
        """
        spec = PlotSpec(data=data, plot_type=plot_type, **kwargs)
        self.plot_specs.append(spec)
    
    def _convert_linestyle_to_plotly(self, linestyle: str) -> str:
        """Convert matplotlib linestyle to plotly dash style."""
        style_map = {
            '-': 'solid',
            '--': 'dash',
            '-.': 'dashdot',
            ':': 'dot',
            'solid': 'solid',
            'dashed': 'dash',
            'dashdot': 'dashdot',
            'dotted': 'dot',
        }
        return style_map.get(linestyle, 'dash')
    
    def plot(self) -> Any:
        """
        Generate the plot grid.
        
        This is the main entry point that creates the figure with all
        subplots arranged according to the configuration.
        
        Supports multiple traces per subplot when specs have the same
        subplot_position value.
        
        Returns
        -------
        matplotlib.figure.Figure or plotly.graph_objects.Figure
            The generated figure
        """
        # Group specs by subplot position
        if any(spec.subplot_position is not None for spec in self.plot_specs):
            # Group by explicit positions
            grouped_specs = {}
            for spec in self.plot_specs:
                pos = spec.subplot_position if spec.subplot_position is not None else len(grouped_specs)
                if pos not in grouped_specs:
                    grouped_specs[pos] = []
                grouped_specs[pos].append(spec)
            
            # Sort by position
            subplot_groups = [grouped_specs[i] for i in sorted(grouped_specs.keys())]
        else:
            # Each spec gets its own subplot
            subplot_groups = [[spec] for spec in self.plot_specs]
        
        # Auto-size grid based on number of subplot positions
        n_subplots = len(subplot_groups)
        rows, cols = self.layout.auto_size_grid(n_subplots)
        
        # Get subplot titles (use first spec's title for each group)
        if self.layout.subplot_titles is None:
            subplot_titles = [group[0].title for group in subplot_groups if group[0].title]
            if not subplot_titles:
                subplot_titles = None
        else:
            subplot_titles = self.layout.subplot_titles
        
        # Create the grid - determine backend string for later comparison
        backend_enum = get_backend() if self.backend is None else self.backend
        # Convert to string value if it's an enum
        if hasattr(backend_enum, 'value'):
            backend_str = backend_enum.value
        else:
            backend_str = backend_enum
        
        # Check if any specs are 3D to set projection
        needs_3d = any(
            spec.plot_type in ('scatter3d', 'trajectory3d')
            for group in subplot_groups 
            for spec in group
        )
        
        result = create_subplot_grid(
            rows=rows,
            cols=cols,
            config=self.config,
            subplot_titles=subplot_titles,
            shared_xaxes=self.layout.shared_xaxes,
            shared_yaxes=self.layout.shared_yaxes,
            backend=backend_str,
            projection='3d' if needs_3d else None,
            specs=[[{'type': 'scene'}] * cols] * rows if needs_3d and backend_str == 'plotly' else None,
        )
        
        if backend_str == 'matplotlib':
            fig, axes = result
            # Flatten axes for easier indexing
            axes_flat = [ax for row in axes for ax in row] if isinstance(axes[0], list) else axes
            
            # Track which labels have been shown per subplot and which axes have titles
            legend_tracker = {}
            axes_with_titles = set()
            
            # Plot each group of specs
            for i, spec_group in enumerate(subplot_groups):
                if i >= len(axes_flat):
                    break
                ax = axes_flat[i]
                legend_tracker[i] = set()
                
                # Assign positions to violin/box plots within this subplot
                position = 1
                for spec in spec_group:
                    if spec.plot_type in ['violin', 'box'] and 'position' not in spec.kwargs:
                        spec.kwargs['position'] = position
                        position += 1
                    
                    if spec.title:
                        axes_with_titles.add(i)
                    self._plot_spec_matplotlib(spec, ax, legend_tracker[i])
                
                # Create legend from stored handles if we have any
                handles = []
                labels = []
                for spec in spec_group:
                    if hasattr(spec, '_legend_handle'):
                        handles.append(spec._legend_handle)
                        labels.append(spec.label)
                if handles:
                    ax.legend(handles, labels)
                
                # Set x-axis labels for violin/box plots
                violin_box_specs = [s for s in spec_group if s.plot_type in ['violin', 'box']]
                if violin_box_specs:
                    positions = [s.kwargs.get('position', 1) for s in violin_box_specs]
                    labels_list = [s.label for s in violin_box_specs]
                    ax.set_xticks(positions)
                    ax.set_xticklabels(labels_list)
            
            # Apply PlotConfig settings to axes after plotting
            if self.config:
                for i, ax in enumerate(axes_flat):
                    # Only set title if PlotConfig has one, this is a single subplot, 
                    # and no spec has already set a title
                    if self.config.title and n_subplots == 1 and i not in axes_with_titles:
                        ax.set_title(self.config.title)
                    if self.config.xlabel:
                        ax.set_xlabel(self.config.xlabel)
                    if self.config.ylabel:
                        ax.set_ylabel(self.config.ylabel)
                    if self.config.xlim:
                        ax.set_xlim(self.config.xlim)
                    if self.config.ylim:
                        ax.set_ylim(self.config.ylim)
                    if self.config.grid:
                        ax.grid(self.config.grid)
            
            # For single subplot, return just the axes; for multiple return (fig, axes)
            if n_subplots == 1:
                return axes_flat[0]
            return fig, axes
        else:
            fig = result
            # Track which labels have been shown per subplot
            legend_tracker = {}
            
            # Plot each group of specs
            for i, spec_group in enumerate(subplot_groups):
                row = (i // cols) + 1
                col = (i % cols) + 1
                legend_tracker[i] = set()
                
                for spec in spec_group:
                    trace = self._plot_spec_plotly(spec, legend_tracker[i])
                    if trace is not None:
                        add_trace_to_subplot(fig, trace, row=row, col=col)
                        
                        # Add reference lines and annotations if present
                        # These are added as shapes/annotations to the figure
                        if hasattr(trace, '_hlines') and trace._hlines:
                            for hline in trace._hlines:
                                y_val = hline['y']
                                line_color = hline.get('color', 'black')
                                line_dash = self._convert_linestyle_to_plotly(hline.get('linestyle', '--'))
                                line_width = hline.get('linewidth', 1.5)
                                line_opacity = hline.get('alpha', 0.7)
                                
                                # Determine xref based on subplot position
                                xref = f'x{i+1}' if i > 0 else 'x'
                                yref = f'y{i+1}' if i > 0 else 'y'
                                
                                fig.add_shape(
                                    type='line',
                                    x0=0, x1=1,
                                    y0=y_val, y1=y_val,
                                    xref=f'{xref} domain',
                                    yref=yref,
                                    line=dict(
                                        color=line_color,
                                        dash=line_dash,
                                        width=line_width,
                                    ),
                                    opacity=line_opacity,
                                )
                                
                                # Add label to legend if provided
                                if hline.get('label'):
                                    fig.add_trace(
                                        go.Scatter(
                                            x=[None], y=[None],
                                            mode='lines',
                                            line=dict(color=line_color, dash=line_dash, width=line_width),
                                            name=hline['label'],
                                            showlegend=True,
                                        ),
                                        row=row, col=col
                                    )
                        
                        if hasattr(trace, '_vlines') and trace._vlines:
                            for vline in trace._vlines:
                                x_val = vline['x']
                                line_color = vline.get('color', 'black')
                                line_dash = self._convert_linestyle_to_plotly(vline.get('linestyle', '--'))
                                line_width = vline.get('linewidth', 1.5)
                                line_opacity = vline.get('alpha', 0.7)
                                
                                # Determine xref based on subplot position
                                xref = f'x{i+1}' if i > 0 else 'x'
                                yref = f'y{i+1}' if i > 0 else 'y'
                                
                                fig.add_shape(
                                    type='line',
                                    x0=x_val, x1=x_val,
                                    y0=0, y1=1,
                                    xref=xref,
                                    yref=f'{yref} domain',
                                    line=dict(
                                        color=line_color,
                                        dash=line_dash,
                                        width=line_width,
                                    ),
                                    opacity=line_opacity,
                                )
                                
                                # Add label to legend if provided
                                if vline.get('label'):
                                    fig.add_trace(
                                        go.Scatter(
                                            x=[None], y=[None],
                                            mode='lines',
                                            line=dict(color=line_color, dash=line_dash, width=line_width),
                                            name=vline['label'],
                                            showlegend=True,
                                        ),
                                        row=row, col=col
                                    )
                        
                        if hasattr(trace, '_annotations') and trace._annotations:
                            for annot in trace._annotations:
                                text = annot['text']
                                xy = annot['xy']
                                xytext = annot.get('xytext', xy)
                                fontsize = annot.get('fontsize', 10)
                                
                                # Determine xref/yref based on subplot position
                                xref = f'x{i+1}' if i > 0 else 'x'
                                yref = f'y{i+1}' if i > 0 else 'y'
                                
                                # Convert bbox to plotly style
                                bgcolor = 'rgba(255, 255, 255, 0.8)'
                                bordercolor = 'black'
                                if 'bbox' in annot and annot['bbox']:
                                    bbox = annot['bbox']
                                    if 'facecolor' in bbox:
                                        # Convert matplotlib color to rgba
                                        fc = bbox['facecolor']
                                        alpha = bbox.get('alpha', 0.7)
                                        if fc == 'yellow':
                                            bgcolor = f'rgba(255, 255, 0, {alpha})'
                                        elif fc == 'lightyellow':
                                            bgcolor = f'rgba(255, 255, 224, {alpha})'
                                        # Add more color mappings as needed
                                
                                fig.add_annotation(
                                    x=xytext[0],
                                    y=xytext[1],
                                    text=text,
                                    xref=xref,
                                    yref=yref,
                                    showarrow=True if 'arrowprops' in annot else False,
                                    arrowhead=2,
                                    arrowsize=1,
                                    arrowwidth=2,
                                    arrowcolor=annot.get('arrowprops', {}).get('color', 'black') if 'arrowprops' in annot else 'black',
                                    ax=xy[0] if 'arrowprops' in annot else xytext[0],
                                    ay=xy[1] if 'arrowprops' in annot else xytext[1],
                                    axref=xref,
                                    ayref=yref,
                                    font=dict(size=fontsize),
                                    bgcolor=bgcolor,
                                    bordercolor=bordercolor,
                                    borderwidth=1,
                                    borderpad=4,
                                )
            return fig
    
    def _plot_spec_matplotlib(self, spec: PlotSpec, ax, legend_tracker: set):
        """Plot a PlotSpec using matplotlib with renderer functions."""
        if ax is None:
            ax = plt.gca()
        
        # Determine if we should show this label
        show_label = spec.label and spec.label not in legend_tracker
        if show_label and spec.label:
            legend_tracker.add(spec.label)
        label_to_use = spec.label if show_label else None
        
        if spec.plot_type == 'scatter':
            scatter = renderers.render_scatter_matplotlib(
                ax=ax,
                data=spec.data,
                color=spec.color,
                colors=spec.colors,
                cmap=spec.cmap,
                marker=spec.marker or 'o',
                marker_size=spec.marker_size,
                alpha=spec.alpha,
                label=label_to_use,
                **spec.kwargs
            )
            # Add colorbar if requested and we have color-mapped data
            if spec.colorbar and scatter is not None and spec.colors is not None:
                # Get the figure from the axes
                fig = ax.get_figure()
                if fig is not None:
                    cbar = fig.colorbar(scatter, ax=ax)
                    if spec.colorbar_label:
                        cbar.set_label(spec.colorbar_label)
        
        elif spec.plot_type == 'scatter3d':
            # 3D scatter plot - ax must be a 3D axis
            scatter = renderers.render_scatter_matplotlib(
                ax=ax,
                data=spec.data,
                color=spec.color,
                colors=spec.colors,
                cmap=spec.cmap,
                marker=spec.marker or 'o',
                marker_size=spec.marker_size,
                alpha=spec.alpha,
                label=label_to_use,
                **spec.kwargs
            )
            # Add colorbar if requested and we have color-mapped data
            if spec.colorbar and scatter is not None and spec.colors is not None:
                # Get the figure from the axes
                fig = ax.get_figure()
                if fig is not None:
                    cbar = fig.colorbar(scatter, ax=ax)
                    if spec.colorbar_label:
                        cbar.set_label(spec.colorbar_label)
        
        elif spec.plot_type == 'line':
            # Pop custom parameters that shouldn't be passed to matplotlib
            x_label = spec.kwargs.pop('x_label', None)
            y_label = spec.kwargs.pop('y_label', None)
            grid_config = spec.kwargs.pop('grid', None)
            
            renderers.render_line_matplotlib(
                ax=ax,
                data=spec.data,
                color=spec.color,
                line_width=spec.line_width or 1.5,
                linestyle=spec.linestyle or '-',
                marker=spec.marker,
                marker_size=spec.marker_size,
                error_y=spec.error_y,
                alpha=spec.alpha,
                label=label_to_use,
                show_values=spec.kwargs.pop('show_values', False),
                value_format=spec.kwargs.pop('value_format', '.3f'),
                x_labels=spec.kwargs.pop('x_labels', None),
                **spec.kwargs
            )
            
            # Add vertical reference lines
            if spec.vlines:
                for vline in spec.vlines:
                    x_val = vline['x']
                    vline_color = vline.get('color', 'black')
                    vline_style = vline.get('linestyle', '--')
                    vline_width = vline.get('linewidth', 1.5)
                    vline_alpha = vline.get('alpha', 0.7)
                    vline_label = vline.get('label', None)
                    ax.axvline(x=x_val, color=vline_color, linestyle=vline_style, 
                              linewidth=vline_width, alpha=vline_alpha, label=vline_label)
            
            # Add horizontal reference lines
            if spec.hlines:
                for hline in spec.hlines:
                    y_val = hline['y']
                    hline_color = hline.get('color', 'black')
                    hline_style = hline.get('linestyle', '--')
                    hline_width = hline.get('linewidth', 1.5)
                    hline_alpha = hline.get('alpha', 0.7)
                    hline_label = hline.get('label', None)
                    ax.axhline(y=y_val, color=hline_color, linestyle=hline_style, 
                              linewidth=hline_width, alpha=hline_alpha, label=hline_label)
            
            # Add annotations
            if spec.annotations:
                for annot in spec.annotations:
                    text = annot['text']
                    xy = annot['xy']
                    xytext = annot.get('xytext', None)
                    fontsize = annot.get('fontsize', 10)
                    bbox = annot.get('bbox', None)
                    arrowprops = annot.get('arrowprops', None)
                    
                    ax.annotate(text, xy=xy, xytext=xytext, fontsize=fontsize,
                               bbox=bbox, arrowprops=arrowprops)
            
            # Apply custom settings
            if x_label:
                ax.set_xlabel(x_label)
            if y_label:
                ax.set_ylabel(y_label)
            if grid_config:
                if isinstance(grid_config, dict):
                    ax.grid(True, **grid_config)
                else:
                    ax.grid(grid_config)
        
        elif spec.plot_type == 'histogram':
            renderers.render_histogram_matplotlib(
                ax=ax,
                data=spec.data,
                color=spec.color,
                alpha=spec.alpha,
                bins=spec.kwargs.pop('bins', 30),
                label=label_to_use,
                **spec.kwargs
            )
        
        elif spec.plot_type == 'heatmap':
            renderers.render_heatmap_matplotlib(
                ax=ax,
                data=spec.data,
                cmap=spec.kwargs.pop('cmap', 'viridis'),
                colorbar_label=spec.colorbar_label,
                alpha=spec.alpha,
                **spec.kwargs
            )
        
        elif spec.plot_type == 'violin':
            # Enhanced violin plot with box and points
            # Remove plotly-specific parameters
            spec.kwargs.pop('meanline', None)
            result = renderers.render_violin_matplotlib(
                ax=ax,
                data=spec.data,
                position=spec.kwargs.pop('position', 1),
                color=spec.color,
                alpha=spec.alpha,
                showmeans=spec.kwargs.pop('showmeans', True),
                showmedians=spec.kwargs.pop('showmedians', True),
                showbox=spec.kwargs.pop('showbox', True),
                showpoints=spec.kwargs.pop('showpoints', True),
                label=label_to_use,
                **spec.kwargs
            )
            # Store legend handle for later
            if 'legend_handle' in result:
                spec._legend_handle = result['legend_handle']
        
        elif spec.plot_type == 'bar':
            # Pop custom parameters that shouldn't be passed to matplotlib
            x_label = spec.kwargs.pop('x_label', None)
            y_label = spec.kwargs.pop('y_label', None)
            grid_config = spec.kwargs.pop('grid', None)
            set_xticks = spec.kwargs.pop('set_xticks', None)
            set_xticklabels = spec.kwargs.pop('set_xticklabels', None)
            
            renderers.render_bar_matplotlib(
                ax=ax,
                data=spec.data,
                x=spec.kwargs.pop('x', None),
                color=spec.color,
                colors=spec.kwargs.pop('colors', None),
                alpha=spec.alpha,
                label=label_to_use,
                orientation=spec.kwargs.pop('orientation', 'v'),
                error_y=spec.kwargs.pop('error_y', None),
                error_x=spec.kwargs.pop('error_x', None),
                show_values=spec.kwargs.pop('show_values', False),
                value_format=spec.kwargs.pop('value_format', '.3f'),
                x_labels=spec.kwargs.pop('x_labels', None),
                **spec.kwargs
            )
            
            # Apply custom settings
            if x_label:
                ax.set_xlabel(x_label)
            if y_label:
                ax.set_ylabel(y_label)
            if grid_config:
                if isinstance(grid_config, dict):
                    ax.grid(True, **grid_config)
                else:
                    ax.grid(grid_config)
            if set_xticks is not None and set_xticklabels is not None:
                ax.set_xticks(set_xticks)
                ax.set_xticklabels(set_xticklabels)
        
        elif spec.plot_type == 'box':
            result = renderers.render_box_matplotlib(
                ax=ax,
                data=spec.data,
                position=spec.kwargs.pop('position', 1),
                color=spec.color,
                alpha=spec.alpha,
                label=label_to_use,
                notch=spec.kwargs.pop('notch', False),
                showpoints=spec.kwargs.pop('showpoints', True),
                **spec.kwargs
            )
            # Store legend handle for later
            if 'legend_handle' in result:
                spec._legend_handle = result['legend_handle']
        
        elif spec.plot_type == 'trajectory':
            # 2D trajectory with time-based coloring
            # Extract x, y from data
            x, y = extract_xy_from_data(spec.data)
            
            # Compute colors if requested
            colors = compute_colors(len(x), color_by=spec.color_by) if spec.color_by else None
            
            # Render trajectory using renderer (segments calculated internally)
            render_trajectory_matplotlib(
                ax=ax,
                x=x,
                y=y,
                colors=colors,
                cmap=spec.cmap or 'viridis',
                linewidth=spec.line_width or 1.0,
                alpha=spec.alpha,
                show_points=spec.show_points,
                point_color=spec.color or "black",
                point_size=spec.marker_size or 10,
                colorbar=spec.colorbar,
                colorbar_label=spec.colorbar_label or 'Time' if spec.color_by else None,
                label=label_to_use,
            )
            
            if spec.equal_aspect:
                ax.set_aspect('equal', adjustable='box')
        
        elif spec.plot_type == 'trajectory3d':
            # 3D trajectory with time-based coloring
            # Extract x, y, z from data
            x, y, z = extract_xyz_from_data(spec.data)
            
            # Compute colors if requested
            colors = compute_colors(len(x), color_by=spec.color_by) if spec.color_by else None
            
            # Render trajectory using renderer (segments calculated internally)
            render_trajectory3d_matplotlib(
                ax=ax,
                x=x,
                y=y,
                z=z,
                colors=colors,
                cmap=spec.cmap or 'viridis',
                linewidth=spec.line_width or 2.0,
                alpha=spec.alpha,
                show_points=spec.show_points,
                point_size=spec.marker_size or 10,
                colorbar=spec.colorbar,
                colorbar_label=spec.colorbar_label or 'Time' if spec.color_by else None,
                label=label_to_use,
            )
        
        elif spec.plot_type == 'kde':
            # 2D KDE contour plot
            # Extract x, y from data
            x, y = extract_xy_from_data(spec.data)
            
            # Calculate KDE using utility function
            xi, yi, zi = compute_kde_2d(
                x, y, 
                bandwidth=spec.bandwidth,
                grid_size=100,
                expand_fraction=0.1
            )
            
            # Render using renderer
            render_kde_matplotlib(
                ax=ax,
                xi=xi,
                yi=yi,
                zi=zi,
                fill=spec.fill,
                n_levels=spec.n_levels,
                cmap=spec.cmap or 'Blues',
                alpha=spec.alpha,
                colorbar=spec.colorbar,
                colorbar_label=spec.colorbar_label,
                label=label_to_use,
            )
            
            # Show points if requested
            if spec.show_points:
                ax.scatter(x, y, c='black', s=spec.marker_size or 5, 
                          alpha=0.3, zorder=3)
        
        elif spec.plot_type == 'grouped_scatter':
            # Grouped scatter with optional convex hulls
            if not isinstance(spec.data, dict):
                raise ValueError("grouped_scatter data must be dict mapping group names to (x,y) tuples")
            
            colors = spec.colors or get_default_categorical_colors(len(spec.data))
            
            for idx, (name, (x, y)) in enumerate(spec.data.items()):
                color = colors[idx % len(colors)]
                ax.scatter(x, y, s=spec.marker_size or 20, alpha=spec.alpha, 
                          label=name, color=color)
                
                # Add convex hull if requested
                if spec.show_hulls and len(x) >= 3:
                    result = compute_convex_hull(x, y)
                    if result is not None:
                        hull_x, hull_y = result
                        render_convex_hull_matplotlib(
                            ax=ax,
                            hull_x=hull_x,
                            hull_y=hull_y,
                            color=color,
                            linewidth=1,
                            alpha=spec.hull_alpha or 0.2,
                            fill=True,
                            fill_alpha=spec.hull_alpha or 0.2,
                        )
        
        elif spec.plot_type == 'convex_hull':
            # Just draw convex hull boundary
            x, y = extract_xy_from_data(spec.data)
            
            if len(x) >= 3:
                hull_x, hull_y = compute_convex_hull(x, y)
                if hull_x is not None and hull_y is not None:
                    render_convex_hull_matplotlib(
                        ax=ax,
                        hull_x=hull_x,
                        hull_y=hull_y,
                        color=spec.color or 'blue',
                        linewidth=spec.line_width or 1,
                        alpha=spec.alpha,
                        fill=spec.fill,
                        fill_alpha=0.2,
                        label=label_to_use,
                    )
        
        elif spec.plot_type == 'boolean_states':
            # Render boolean states as filled regions
            x, states = extract_xy_from_data(spec.data)
            states = states.astype(bool)
            
            renderers.render_boolean_states_matplotlib(
                ax=ax,
                x=x,
                states=states,
                true_color=spec.true_color or '#2ca02c',
                false_color=spec.false_color or '#d62728',
                true_label=spec.true_label or 'True',
                false_label=spec.false_label or 'False',
                alpha=spec.alpha,
            )
        
        if spec.title:
            ax.set_title(spec.title)
        
        # Apply per-subplot settings from kwargs if provided
        if 'x_label' in spec.kwargs:
            ax.set_xlabel(spec.kwargs['x_label'])
        if 'y_label' in spec.kwargs:
            ax.set_ylabel(spec.kwargs['y_label'])
        if 'grid' in spec.kwargs:
            grid_val = spec.kwargs['grid']
            if isinstance(grid_val, bool):
                ax.grid(grid_val, alpha=0.3)
            elif isinstance(grid_val, dict):
                ax.grid(**grid_val)
        
        # Note: Legend is now handled in the plotting loop for violin/box plots
        # For other plot types that use standard matplotlib labels, legend is still needed
        # but we skip it for violin/box since we handle those separately
        if legend_tracker is not None and spec.plot_type not in ['violin', 'box']:
            # Check if there are actually any legend entries before calling legend()
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend()
    
    def _plot_spec_plotly(self, spec: PlotSpec, legend_tracker: set):
        """Plot a PlotSpec using plotly with renderer functions (returns trace)."""
        if not PLOTLY_AVAILABLE:
            raise ValueError("Plotly backend requested but plotly is not installed")
        
        # Determine if we should show this label in legend
        show_legend = spec.label and spec.label not in legend_tracker
        if show_legend and spec.label:
            legend_tracker.add(spec.label)
        
        if spec.plot_type == 'scatter':
            return renderers.render_scatter_plotly(
                data=spec.data,
                color=spec.color,
                colors=spec.colors,
                cmap=spec.cmap,
                marker=spec.marker or 'circle',
                marker_size=spec.marker_size,
                sizes=spec.sizes,
                alpha=spec.alpha,
                label=spec.label,
                showlegend=show_legend,
                colorbar=spec.colorbar,
                colorbar_label=spec.colorbar_label,
                **spec.kwargs
            )
        
        elif spec.plot_type == 'scatter3d':
            return renderers.render_scatter3d_plotly(
                data=spec.data,
                color=spec.color,
                colors=spec.colors,
                cmap=spec.cmap,
                marker_size=spec.marker_size,
                sizes=spec.sizes,
                alpha=spec.alpha,
                label=spec.label,
                showlegend=show_legend,
                colorbar=spec.colorbar,
                colorbar_label=spec.colorbar_label,
                **spec.kwargs
            )
        
        elif spec.plot_type == 'line':
            trace = renderers.render_line_plotly(
                data=spec.data,
                color=spec.color,
                line_width=spec.line_width,
                linestyle=spec.linestyle,
                error_y=spec.error_y,
                alpha=spec.alpha,
                label=spec.label,
                showlegend=show_legend,
                **spec.kwargs
            )
            
            # Store hlines, vlines, and annotations for later processing
            # These will be added as shapes/annotations to the figure
            if hasattr(spec, 'hlines') and spec.hlines:
                if not hasattr(trace, '_hlines'):
                    trace._hlines = []
                trace._hlines = spec.hlines
            
            if hasattr(spec, 'vlines') and spec.vlines:
                if not hasattr(trace, '_vlines'):
                    trace._vlines = []
                trace._vlines = spec.vlines
            
            if hasattr(spec, 'annotations') and spec.annotations:
                if not hasattr(trace, '_annotations'):
                    trace._annotations = []
                trace._annotations = spec.annotations
            
            return trace
        
        elif spec.plot_type == 'histogram':
            return renderers.render_histogram_plotly(
                data=spec.data,
                color=spec.color,
                alpha=spec.alpha,
                bins=spec.kwargs.pop('bins', 30),
                label=spec.label,
                showlegend=show_legend,
                **spec.kwargs
            )
        
        elif spec.plot_type == 'heatmap':
            return renderers.render_heatmap_plotly(
                data=spec.data,
                cmap=spec.kwargs.pop('cmap', None),
                colorscale=spec.kwargs.pop('colorscale', None),
                colorbar_label=spec.colorbar_label,
                **spec.kwargs
            )
        
        elif spec.plot_type == 'bar':
            return renderers.render_bar_plotly(
                data=spec.data,
                x=spec.kwargs.pop('x', None),
                color=spec.color,
                colors=spec.kwargs.pop('colors', None),
                alpha=spec.alpha,
                label=spec.label,
                showlegend=show_legend,
                error_y=spec.kwargs.pop('error_y', None),
                error_x=spec.kwargs.pop('error_x', None),
                **spec.kwargs
            )
        
        elif spec.plot_type == 'violin':
            # Enhanced violin plot with box and points
            # Convert matplotlib-style 'showmeans' to plotly's 'meanline'
            # Make a copy to avoid mutating the original
            plot_kwargs = dict(spec.kwargs)
            showmeans = plot_kwargs.pop('showmeans', None)
            meanline = plot_kwargs.pop('meanline', {})
            
            if showmeans is not None:
                # If showmeans is explicitly set, use it to control meanline
                meanline = {'visible': showmeans}
            elif isinstance(meanline, bool):
                meanline = {'visible': meanline}
            elif not isinstance(meanline, dict):
                meanline = {'visible': True}
            
            return renderers.render_violin_plotly(
                data=spec.data,
                color=spec.color,
                alpha=spec.alpha,
                meanline=meanline,
                showbox=plot_kwargs.pop('showbox', True),
                showpoints=plot_kwargs.pop('showpoints', True),
                label=spec.label,
                showlegend=show_legend,
                **plot_kwargs
            )
        
        elif spec.plot_type == 'box':
            # Map matplotlib 'notch' to plotly 'notched'
            plot_kwargs = dict(spec.kwargs)
            notched = plot_kwargs.pop('notch', plot_kwargs.pop('notched', False))
            
            return renderers.render_box_plotly(
                data=spec.data,
                color=spec.color,
                alpha=spec.alpha,
                label=spec.label,
                showlegend=show_legend,
                notched=notched,
                **plot_kwargs
            )
        
        elif spec.plot_type == 'trajectory':
            # 2D trajectory with time-based coloring
            # Extract x, y from data
            x, y = extract_xy_from_data(spec.data)
            
            # Compute colors if needed
            colors = compute_colors(len(x), color_by=spec.color_by) if spec.color_by else None
            
            # Render using renderer
            trace = render_trajectory_plotly(
                x=x,
                y=y,
                colors=colors,
                cmap=spec.cmap or 'Viridis',
                linewidth=spec.line_width or 1.0,
                alpha=spec.alpha,
                show_points=spec.show_points,
                point_size=spec.marker_size or 10,
                colorbar=spec.colorbar,
                colorbar_label=spec.colorbar_label or 'Time' if spec.color_by else None,
                label=spec.label,
                showlegend=show_legend,
            )
            
            return trace
        
        elif spec.plot_type == 'trajectory3d':
            # 3D trajectory with time-based coloring
            # Extract x, y, z from data
            x, y, z = extract_xyz_from_data(spec.data)
            
            # Compute colors if needed
            colors = compute_colors(len(x), color_by=spec.color_by) if spec.color_by else None
            
            # Render using renderer
            trace = render_trajectory3d_plotly(
                x=x,
                y=y,
                z=z,
                colors=colors,
                cmap=spec.cmap or 'Viridis',
                linewidth=spec.line_width or 2.0,
                alpha=spec.alpha,
                show_points=spec.show_points,
                point_size=spec.marker_size or 10,
                colorbar=spec.colorbar,
                colorbar_label=spec.colorbar_label or 'Time' if spec.color_by else None,
                label=spec.label,
                showlegend=show_legend,
            )
            
            return trace
        
        elif spec.plot_type == 'kde':
            # 2D KDE contour plot
            # Extract x, y from data
            x, y = extract_xy_from_data(spec.data)
            
            # Calculate KDE using utility function
            xi, yi, zi = compute_kde_2d(
                x, y,
                bandwidth=spec.bandwidth,
                grid_size=100,
                expand_fraction=0.1
            )
            
            # Render using renderer
            trace = render_kde_plotly(
                xi=xi,
                yi=yi,
                zi=zi,
                fill=spec.fill,
                n_levels=spec.n_levels,
                cmap=spec.cmap or 'Blues',
                alpha=spec.alpha,
                colorbar=spec.colorbar,
                colorbar_label=spec.colorbar_label,
                label=spec.label,
                showlegend=show_legend,
            )
            
            return trace
        
        elif spec.plot_type == 'grouped_scatter':
            # Grouped scatter - return multiple traces (we'll need to handle this specially)
            if not isinstance(spec.data, dict):
                raise ValueError("grouped_scatter data must be dict mapping group names to (x,y) tuples")
            
            colors = spec.colors or get_default_categorical_colors(len(spec.data))
            
            # Return list of traces (one per group)
            traces = []
            for idx, (name, (x, y)) in enumerate(spec.data.items()):
                color = colors[idx % len(colors)]
                trace = go.Scatter(
                    x=x,
                    y=y,
                    mode='markers',
                    marker=dict(size=spec.marker_size or 20, color=color, opacity=spec.alpha),
                    name=name,
                    showlegend=True,
                )
                traces.append(trace)
                
                # Add convex hull if requested
                if spec.show_hulls and len(x) >= 3:
                    hull_x, hull_y = compute_convex_hull(x, y)
                    if hull_x is not None and hull_y is not None:
                        hull_trace = render_convex_hull_plotly(
                            hull_x=hull_x,
                            hull_y=hull_y,
                            color=color,
                            linewidth=1,
                            alpha=0.3,
                            fill=spec.fill if hasattr(spec, 'fill') else False,
                            fill_alpha=0.1,
                            showlegend=False,
                        )
                        traces.append(hull_trace)
            
            # Return first trace (others will be ignored for now - needs refactoring)
            return traces[0] if traces else None
        
        elif spec.plot_type == 'convex_hull':
            # Just draw convex hull boundary
            x, y = extract_xy_from_data(spec.data)
            
            if len(x) >= 3:
                hull_x, hull_y = compute_convex_hull(x, y)
                if hull_x is not None and hull_y is not None:
                    trace = render_convex_hull_plotly(
                        hull_x=hull_x,
                        hull_y=hull_y,
                        color=spec.color or 'blue',
                        linewidth=spec.line_width or 1,
                        alpha=spec.alpha,
                        fill=spec.fill,
                        fill_alpha=0.2,
                        label=spec.label,
                        showlegend=show_legend,
                    )
                    return trace
                else:
                    print("Warning: Could not compute convex hull")
                    return None
            else:
                return None
        
        elif spec.plot_type == 'boolean_states':
            # Render boolean states as filled regions
            x, states = extract_xy_from_data(spec.data)
            states = states.astype(bool)
            
            traces = renderers.render_boolean_states_plotly(
                x=x,
                states=states,
                true_color=spec.true_color or '#2ca02c',
                false_color=spec.false_color or '#d62728',
                true_label=spec.true_label or 'True',
                false_label=spec.false_label or 'False',
                alpha=spec.alpha,
            )
            # Return first trace (multiple traces need special handling)
            return traces[0] if traces else None
        
        else:
            raise ValueError(f"Unsupported plot type: {spec.plot_type}")


# Convenience functions for common patterns

def plot_comparison_grid(
    data_dict: dict[str, np.ndarray],
    plot_type: PlotType = 'scatter',
    rows: int | None = None,
    cols: int | None = None,
    **kwargs,
) -> Any:
    """
    Create a grid comparing multiple datasets with the same plot type.
    
    Parameters
    ----------
    data_dict : dict
        Dictionary mapping labels to data arrays
    plot_type : PlotType, default='scatter'
        Type of plot for all comparisons ('scatter', 'line', 'histogram', etc.)
    rows : int, optional
        Number of rows (auto-calculated if not specified)
    cols : int, optional
        Number of columns (auto-calculated if not specified)
    **kwargs
        Additional arguments passed to PlotGrid
        
    Returns
    -------
    figure
        The generated figure
        
    Examples
    --------
    >>> data = {
    ...     'Method A': result_a,
    ...     'Method B': result_b,
    ...     'Method C': result_c,
    ... }
    >>> fig = plot_comparison_grid(data, plot_type='histogram', cols=3)
    """
    layout = GridLayoutConfig(rows=rows, cols=cols)
    grid = PlotGrid.from_dict(
        data_dict,
        plot_type=plot_type,
        layout=layout,
        **kwargs
    )
    return grid.plot()


def plot_grouped_comparison(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: str,
    plot_type: PlotType = 'scatter',
    **kwargs,
) -> Any:
    """
    Create overlaid plots grouped by a category.
    
    All groups are plotted in the same subplot with different colors.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with data to plot
    x_col : str
        Column name for x-axis
    y_col : str
        Column name for y-axis
    group_col : str
        Column name for grouping (different colors)
    plot_type : PlotType, default='scatter'
        Type of plot ('scatter', 'line', 'histogram', etc.)
    **kwargs
        Additional arguments passed to PlotGrid
        
    Returns
    -------
    figure
        The generated figure
        
    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'x': [...],
    ...     'y': [...],
    ...     'condition': ['A', 'A', 'B', 'B', ...]
    ... })
    >>> fig = plot_grouped_comparison(df, 'x', 'y', 'condition')
    """
    # Create one plot spec per group, all in position (1, 1)
    groups = data[group_col].unique()
    color_scheme = ColorScheme()
    colors = color_scheme.get_colors(groups)
    
    plot_specs = []
    for group in groups:
        group_data = data[data[group_col] == group]
        if plot_type == 'scatter':
            arr = group_data[[x_col, y_col]].values
        elif plot_type == 'line':
            arr = group_data[y_col].values
        else:
            arr = group_data[y_col].values
        
        spec = PlotSpec(
            data=arr,
            plot_type=plot_type,
            label=str(group),
            color=colors[group],
        )
        plot_specs.append(spec)
    
    # All in single subplot
    layout = GridLayoutConfig(rows=1, cols=1)
    grid = PlotGrid(
        plot_specs=plot_specs,
        layout=layout,
        color_scheme=color_scheme,
        **kwargs
    )
    
    return grid.plot()


# ==============================================================================
# Subplot Utility Functions (Used internally by PlotGrid)
# ==============================================================================

def create_subplot_grid(
    rows: int,
    cols: int,
    config: PlotConfig | None = None,
    subplot_titles: Sequence[str] | None = None,
    shared_xaxes: bool | str = False,
    shared_yaxes: bool | str = False,
    vertical_spacing: float | None = None,
    horizontal_spacing: float | None = None,
    specs: list[list[dict[str, Any]]] | None = None,
    backend: Literal["matplotlib", "plotly"] | None = None,
    projection: str | None = None,
):
    """
    Create a multi-panel subplot grid.
    
    Internal utility function used by PlotGrid for creating subplot layouts.

    Parameters
    ----------
    rows : int
        Number of rows.
    cols : int
        Number of columns.
    config : PlotConfig, optional
        Overall plot configuration.
    subplot_titles : sequence of str, optional
        Titles for each subplot (length should be rows * cols).
    shared_xaxes : bool or str, default=False
        Share x-axes. Can be True, False, 'all', 'rows', or 'columns'.
    shared_yaxes : bool or str, default=False
        Share y-axes. Can be True, False, 'all', 'rows', or 'columns'.
    vertical_spacing : float, optional
        Vertical spacing between subplots (0 to 1).
    horizontal_spacing : float, optional
        Horizontal spacing between subplots (0 to 1).
    specs : list of list of dict, optional
        Specifications for each subplot (plotly only).
        Each dict can contain 'type' (e.g., 'xy', 'scene', etc.).
    backend : {"matplotlib", "plotly"}, optional
        Backend to use.

    Returns
    -------
    matplotlib.figure.Figure and list of Axes, or plotly.graph_objects.Figure
        For matplotlib: tuple of (figure, list of axes).
        For plotly: figure object with subplot structure.
    """
    if config is None:
        config = PlotConfig()

    # Get backend - just use string directly, don't try to instantiate Literal type
    backend_str = get_backend() if backend is None else backend

    if backend_str == 'matplotlib':
        return _create_subplot_grid_matplotlib(
            rows, cols, config, subplot_titles, shared_xaxes, shared_yaxes, plt, projection
        )
    else:
        if not PLOTLY_AVAILABLE:
            raise ValueError("Plotly backend requested but plotly is not installed")
        return _create_subplot_grid_plotly(
            rows, cols, config, subplot_titles, shared_xaxes, shared_yaxes,
            vertical_spacing, horizontal_spacing, specs, go, make_subplots
        )


def _create_subplot_grid_matplotlib(
    rows: int,
    cols: int,
    config: PlotConfig,
    subplot_titles: Sequence[str] | None,
    shared_xaxes: bool | str,
    shared_yaxes: bool | str,
    plt,
    projection: str | None = None,
):
    """Matplotlib implementation of subplot grid."""
    # Convert shared axes parameters
    sharex = "all" if shared_xaxes is True else (shared_xaxes if isinstance(shared_xaxes, str) else False)
    sharey = "all" if shared_yaxes is True else (shared_yaxes if isinstance(shared_yaxes, str) else False)

    # Set up subplot_kw for 3D projection if needed
    subplot_kw = {}
    if projection:
        subplot_kw['projection'] = projection
    
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=config.figsize,
        dpi=config.dpi,
        sharex=sharex if sharex != "rows" and sharex != "columns" else False,
        sharey=sharey if sharey != "rows" and sharey != "columns" else False,
        squeeze=False,
        subplot_kw=subplot_kw,
    )

    # Flatten axes array for easier indexing
    axes_flat = axes.flatten().tolist()

    # Add subplot titles
    if subplot_titles is not None:
        for i, (ax, title) in enumerate(zip(axes_flat, subplot_titles)):
            ax.set_title(title, fontsize=12)

    # Apply overall title - only use suptitle for multiple subplots
    # For single subplot, title will be set on the axis itself
    if config.title and (rows * cols > 1):
        fig.suptitle(config.title, fontsize=14)

    plt.tight_layout()

    return fig, axes_flat


def _create_subplot_grid_plotly(
    rows: int,
    cols: int,
    config: PlotConfig,
    subplot_titles: Sequence[str] | None,
    shared_xaxes: bool | str,
    shared_yaxes: bool | str,
    vertical_spacing: float | None,
    horizontal_spacing: float | None,
    specs: list[list[dict[str, Any]]] | None,
    go,
    make_subplots,
):
    """Plotly implementation of subplot grid."""
    # Convert subplot_titles to list if provided
    titles = list(subplot_titles) if subplot_titles is not None else None

    # Set default spacing if not provided
    if vertical_spacing is None:
        vertical_spacing = 0.1 if rows > 1 else 0.0
    if horizontal_spacing is None:
        horizontal_spacing = 0.1 if cols > 1 else 0.0

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=titles,
        shared_xaxes=shared_xaxes,
        shared_yaxes=shared_yaxes,
        vertical_spacing=vertical_spacing,
        horizontal_spacing=horizontal_spacing,
        specs=specs,
    )

    # Apply overall layout
    width, height = config.figsize
    fig.update_layout(
        title=config.title if config.title else None,
        width=width * config.dpi,
        height=height * config.dpi,
        showlegend=config.legend,
    )

    return fig


def add_trace_to_subplot(fig, trace, row: int, col: int) -> None:
    """
    Add a trace to a specific subplot in a plotly figure.
    
    Internal utility function used by PlotGrid for adding traces to subplots.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        Figure created by create_subplot_grid with plotly backend.
    trace : plotly trace object
        Trace to add (e.g., go.Scatter, go.Scatter3d, go.Bar).
    row : int
        Row position (1-indexed).
    col : int
        Column position (1-indexed).

    Returns
    -------
    plotly.graph_objects.Figure
        Updated figure with trace added.
    """
    if not PLOTLY_AVAILABLE:
        raise ValueError("Plotly is not installed")

    if not isinstance(fig, go.Figure):
        raise TypeError("fig must be a plotly.graph_objects.Figure")

    fig.add_trace(trace, row=row, col=col)
    return fig
