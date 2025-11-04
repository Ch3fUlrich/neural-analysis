"""
Flexible grid configuration system for multi-panel plotting.

This module provides a metadata-driven approach to creating complex
multi-panel plots with minimal code. Instead of manually managing subplot
positions and styling, users provide a structured configuration that
specifies what to plot, where, and how.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Sequence
import numpy as np
import pandas as pd
from pathlib import Path

from .core import PlotConfig


PlotType = Literal['scatter', 'line', 'histogram', 'heatmap', 'scatter3d', 'violin', 'box', 'bar']
BackendType = Literal['matplotlib', 'plotly']

@dataclass
class PlotSpec:
    """
    Specification for a single plot element.
    
    This defines what data to plot and how to style it.
    
    Parameters
    ----------
    data : np.ndarray or pd.DataFrame or dict
        Data to plot. Format depends on plot_type:
        - scatter: (n_samples, 2) or (n_samples, 3) array
        - line: (n_samples,) or (n_samples, n_lines) array
        - histogram/kde: (n_samples,) array
        - heatmap: (n_rows, n_cols) array
        - violin/box: (n_samples,) array or list of arrays
    plot_type : PlotType
        Type of plot: 'scatter', 'line', 'histogram', 'heatmap', 
        'scatter3d', 'violin', 'box', 'bar'
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
    alpha : float, optional
        Transparency (0-1)
    kwargs : dict, optional
        Additional plot-specific arguments
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
    alpha: float = 0.7
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
        from .backend import get_backend, BackendType
        
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
        
        # Create the grid
        backend_type = get_backend() if self.backend is None else BackendType(self.backend)
        
        result = create_subplot_grid(
            rows=rows,
            cols=cols,
            config=self.config,
            subplot_titles=subplot_titles,
            shared_xaxes=self.layout.shared_xaxes,
            shared_yaxes=self.layout.shared_yaxes,
            backend=self.backend,
        )
        
        if backend_type == BackendType.MATPLOTLIB:
            fig, axes = result
            # Flatten axes for easier indexing
            axes_flat = [ax for row in axes for ax in row] if isinstance(axes[0], list) else axes
            
            # Track which labels have been shown per subplot
            legend_tracker = {}
            
            # Plot each group of specs
            for i, spec_group in enumerate(subplot_groups):
                if i >= len(axes_flat):
                    break
                ax = axes_flat[i]
                legend_tracker[i] = set()
                
                for spec in spec_group:
                    self._plot_spec_matplotlib(spec, ax, legend_tracker[i])
            
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
            
            return fig
    
    def _plot_spec_matplotlib(self, spec: PlotSpec, ax, legend_tracker: set):
        """Plot a PlotSpec using matplotlib with renderer functions."""
        import matplotlib.pyplot as plt
        from . import renderers
        
        if ax is None:
            ax = plt.gca()
        
        # Determine if we should show this label
        show_label = spec.label and spec.label not in legend_tracker
        if show_label and spec.label:
            legend_tracker.add(spec.label)
        label_to_use = spec.label if show_label else None
        
        if spec.plot_type == 'scatter':
            renderers.render_scatter_matplotlib(
                ax=ax,
                data=spec.data,
                color=spec.color,
                marker=spec.marker or 'o',
                marker_size=spec.marker_size,
                alpha=spec.alpha,
                label=label_to_use,
                **spec.kwargs
            )
        
        elif spec.plot_type == 'line':
            renderers.render_line_matplotlib(
                ax=ax,
                data=spec.data,
                color=spec.color,
                line_width=spec.line_width or 1.5,
                alpha=spec.alpha,
                label=label_to_use,
                **spec.kwargs
            )
        
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
                alpha=spec.alpha,
                **spec.kwargs
            )
        
        elif spec.plot_type == 'violin':
            # Enhanced violin plot with box and points
            renderers.render_violin_matplotlib(
                ax=ax,
                data=spec.data,
                color=spec.color,
                alpha=spec.alpha,
                showmeans=spec.kwargs.pop('showmeans', True),
                showmedians=spec.kwargs.pop('showmedians', True),
                showbox=spec.kwargs.pop('showbox', True),
                showpoints=spec.kwargs.pop('showpoints', True),
                label=label_to_use,
                **spec.kwargs
            )
        
        elif spec.plot_type == 'box':
            renderers.render_box_matplotlib(
                ax=ax,
                data=spec.data,
                color=spec.color,
                alpha=spec.alpha,
                label=label_to_use,
                **spec.kwargs
            )
        
        if spec.title:
            ax.set_title(spec.title)
        if legend_tracker:  # Only show legend if there are labeled items
            ax.legend()
    
    def _plot_spec_plotly(self, spec: PlotSpec, legend_tracker: set):
        """Plot a PlotSpec using plotly with renderer functions (returns trace)."""
        from . import renderers
        
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ValueError("Plotly backend requested but plotly is not installed")
        
        # Determine if we should show this label in legend
        show_legend = spec.label and spec.label not in legend_tracker
        if show_legend and spec.label:
            legend_tracker.add(spec.label)
        
        if spec.plot_type == 'scatter':
            return renderers.render_scatter_plotly(
                data=spec.data,
                color=spec.color,
                marker=spec.marker or 'circle',
                marker_size=spec.marker_size,
                alpha=spec.alpha,
                label=spec.label,
                showlegend=show_legend,
                **spec.kwargs
            )
        
        elif spec.plot_type == 'scatter3d':
            return renderers.render_scatter3d_plotly(
                data=spec.data,
                color=spec.color,
                marker_size=spec.marker_size,
                alpha=spec.alpha,
                label=spec.label,
                showlegend=show_legend,
                **spec.kwargs
            )
        
        elif spec.plot_type == 'line':
            return renderers.render_line_plotly(
                data=spec.data,
                color=spec.color,
                line_width=spec.line_width,
                alpha=spec.alpha,
                label=spec.label,
                showlegend=show_legend,
                **spec.kwargs
            )
        
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
                **spec.kwargs
            )
        
        elif spec.plot_type == 'bar':
            return renderers.render_bar_plotly(
                data=spec.data,
                x=spec.kwargs.pop('x', None),
                color=spec.color,
                alpha=spec.alpha,
                label=spec.label,
                showlegend=show_legend,
                **spec.kwargs
            )
        
        elif spec.plot_type == 'violin':
            # Enhanced violin plot with box and points
            meanline = spec.kwargs.pop('meanline', {})
            if isinstance(meanline, bool):
                meanline = {'visible': meanline}
            elif not isinstance(meanline, dict):
                meanline = {'visible': True}
            
            return renderers.render_violin_plotly(
                data=spec.data,
                color=spec.color,
                alpha=spec.alpha,
                meanline=meanline,
                showbox=spec.kwargs.pop('showbox', True),
                showpoints=spec.kwargs.pop('showpoints', True),
                label=spec.label,
                showlegend=show_legend,
                **spec.kwargs
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
    import matplotlib.pyplot as plt
    from .backend import BackendType, get_backend
    
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        PLOTLY_AVAILABLE = True
    except ImportError:
        PLOTLY_AVAILABLE = False
    
    if config is None:
        config = PlotConfig()

    backend_type = get_backend() if backend is None else BackendType(backend)

    if backend_type == BackendType.MATPLOTLIB:
        return _create_subplot_grid_matplotlib(
            rows, cols, config, subplot_titles, shared_xaxes, shared_yaxes, plt
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
):
    """Matplotlib implementation of subplot grid."""
    # Convert shared axes parameters
    sharex = "all" if shared_xaxes is True else (shared_xaxes if isinstance(shared_xaxes, str) else False)
    sharey = "all" if shared_yaxes is True else (shared_yaxes if isinstance(shared_yaxes, str) else False)

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=config.figsize,
        dpi=config.dpi,
        sharex=sharex if sharex != "rows" and sharex != "columns" else False,
        sharey=sharey if sharey != "rows" and sharey != "columns" else False,
        squeeze=False,
    )

    # Flatten axes array for easier indexing
    axes_flat = axes.flatten().tolist()

    # Add subplot titles
    if subplot_titles is not None:
        for i, (ax, title) in enumerate(zip(axes_flat, subplot_titles)):
            ax.set_title(title, fontsize=12)

    # Apply overall title
    if config.title:
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


def add_trace_to_subplot(fig, trace, row: int, col: int):
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
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ValueError("Plotly is not installed")

    if not isinstance(fig, go.Figure):
        raise TypeError("fig must be a plotly.graph_objects.Figure")

    fig.add_trace(trace, row=row, col=col)
    return fig
