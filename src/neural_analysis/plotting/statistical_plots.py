"""
Statistical plotting utilities.

This module provides convenience functions for creating common statistical
visualizations using the PlotGrid system. All functions support both
matplotlib and plotly backends.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from .core import PlotConfig
from .grid_config import GridLayoutConfig, PlotGrid, PlotSpec

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "plot_bar",
    "plot_violin",
    "plot_box",
    "plot_grouped_distributions",
    "plot_comparison_distributions",
]


# Default color palette that works with both matplotlib and plotly
DEFAULT_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
]


def plot_bar(
    data: dict[str, np.ndarray] | list[np.ndarray],
    labels: Sequence[str] | None = None,
    colors: Sequence[str] | None = None,
    orientation: Literal['v', 'h'] = 'v',
    config: PlotConfig | None = None,
    backend: Literal['matplotlib', 'plotly'] | None = None,
    **kwargs,
) -> Any:
    """
    Create a bar plot for comparing multiple groups.
    
    Parameters
    ----------
    data : dict or list of arrays
        If dict: keys are labels, values are data arrays.
        If list: provide labels separately.
    labels : sequence of str, optional
        Labels for each bar (required if data is list).
    colors : sequence of str, optional
        Colors for each bar.
    orientation : {'v', 'h'}, default='v'
        Vertical or horizontal bars.
    config : PlotConfig, optional
        Plot configuration.
    backend : {'matplotlib', 'plotly'}, optional
        Plotting backend.
    **kwargs
        Additional arguments passed to PlotSpec.
    
    Returns
    -------
    matplotlib.figure.Figure or plotly.graph_objects.Figure
        The generated figure.
    
    Examples
    --------
    >>> data = {
    ...     'Group A': np.random.randn(100),
    ...     'Group B': np.random.randn(100) + 1,
    ...     'Group C': np.random.randn(100) + 2,
    ... }
    >>> fig = plot_bar(data, backend='plotly')
    """
    # Convert to dict if needed
    if isinstance(data, list):
        if labels is None:
            labels = [f'Group {i+1}' for i in range(len(data))]
        data_dict = dict(zip(labels, data))
    else:
        data_dict = data
        labels = list(data_dict.keys())
    
    # Compute means and stds for bars
    means = np.array([np.mean(data_dict[label]) for label in labels])
    stds = np.array([np.std(data_dict[label], ddof=1) for label in labels])
    x_positions = np.arange(len(labels))
    
    # Set default colors
    if colors is None:
        colors = [DEFAULT_COLORS[i % len(DEFAULT_COLORS)] for i in range(len(labels))]
    
    # Create a single bar spec with all data
    if orientation == 'v':
        spec = PlotSpec(
            data=means,
            plot_type='bar',
            color=None,  # Will use multiple colors
            subplot_position=0,
            kwargs={
                'x': x_positions,
                'error_y': stds if 'error_y' not in kwargs else kwargs.pop('error_y'),
                'colors': colors,  # Pass colors as array
                **kwargs
            }
        )
    else:  # horizontal
        spec = PlotSpec(
            data=means,
            plot_type='bar',
            color=None,
            subplot_position=0,
            kwargs={
                'orientation': 'h',
                'x': means,
                'y': x_positions,
                'error_x': stds if 'error_x' not in kwargs else kwargs.pop('error_x'),
                'colors': colors,
                **kwargs
            }
        )
    
    grid = PlotGrid(
        plot_specs=[spec],
        config=config or PlotConfig(title="Bar Plot Comparison"),
        layout=GridLayoutConfig(rows=1, cols=1),
        backend=backend,
    )
    
    result = grid.plot()
    
    # Update axes labels
    if backend == 'plotly' or (backend is None and hasattr(result, 'update_xaxes')):
        if orientation == 'v':
            result.update_xaxes(tickvals=x_positions, ticktext=labels)
        else:
            result.update_yaxes(tickvals=x_positions, ticktext=labels)
    else:  # matplotlib
        # Check if result is a Figure or Axes
        import matplotlib.pyplot as plt
        if isinstance(result, plt.Figure):
            ax = result.axes[0]
        else:
            ax = result  # It's already an Axes object
        
        if orientation == 'v':
            ax.set_xticks(x_positions)
            ax.set_xticklabels(labels)
        else:
            ax.set_yticks(x_positions)
            ax.set_yticklabels(labels)
    
    return result


def plot_violin(
    data: dict[str, np.ndarray] | list[np.ndarray],
    labels: Sequence[str] | None = None,
    colors: Sequence[str] | None = None,
    showmeans: bool = True,
    showmedians: bool = True,
    config: PlotConfig | None = None,
    backend: Literal['matplotlib', 'plotly'] | None = None,
    **kwargs,
) -> Any:
    """
    Create violin plots for comparing distributions.
    
    Parameters
    ----------
    data : dict or list of arrays
        If dict: keys are labels, values are data arrays.
        If list: provide labels separately.
    labels : sequence of str, optional
        Labels for each violin (required if data is list).
    colors : sequence of str, optional
        Colors for each violin.
    showmeans : bool, default=True
        Show mean lines in violins.
    showmedians : bool, default=True
        Show median lines in violins.
    config : PlotConfig, optional
        Plot configuration.
    backend : {'matplotlib', 'plotly'}, optional
        Plotting backend.
    **kwargs
        Additional arguments passed to PlotSpec.
    
    Returns
    -------
    matplotlib.figure.Figure or plotly.graph_objects.Figure
        The generated figure.
    
    Examples
    --------
    >>> data = {
    ...     'Control': np.random.randn(200),
    ...     'Treatment': np.random.randn(200) + 0.5,
    ... }
    >>> fig = plot_violin(data, backend='plotly')
    """
    # Convert to dict if needed
    if isinstance(data, list):
        if labels is None:
            labels = [f'Group {i+1}' for i in range(len(data))]
        data_dict = dict(zip(labels, data))
    else:
        data_dict = data
        labels = list(data_dict.keys())
    
    # Set default colors
    if colors is None:
        colors = [DEFAULT_COLORS[i % len(DEFAULT_COLORS)] for i in range(len(labels))]
    
    # Create specs
    specs = []
    for label, color in zip(labels, colors):
        # For plotly, map showmeans/showmedians to meanline
        plot_kwargs = dict(kwargs)
        if backend == 'plotly' and showmeans:
            plot_kwargs['meanline'] = {'visible': True}
        # Remove matplotlib-specific params for plotly
        if backend == 'plotly':
            plot_kwargs.pop('showmedians', None)
        
        spec = PlotSpec(
            data=data_dict[label],
            plot_type='violin',
            label=label,
            color=color,
            subplot_position=0,
            kwargs=plot_kwargs
        )
        specs.append(spec)
    
    grid = PlotGrid(
        plot_specs=specs,
        config=config or PlotConfig(title="Violin Plot Comparison"),
        layout=GridLayoutConfig(rows=1, cols=1),
        backend=backend,
    )
    
    return grid.plot()


def plot_box(
    data: dict[str, np.ndarray] | list[np.ndarray],
    labels: Sequence[str] | None = None,
    colors: Sequence[str] | None = None,
    notch: bool = False,
    config: PlotConfig | None = None,
    backend: Literal['matplotlib', 'plotly'] | None = None,
    **kwargs,
) -> Any:
    """
    Create box plots for comparing distributions.
    
    Parameters
    ----------
    data : dict or list of arrays
        If dict: keys are labels, values are data arrays.
        If list: provide labels separately.
    labels : sequence of str, optional
        Labels for each box (required if data is list).
    colors : sequence of str, optional
        Colors for each box.
    notch : bool, default=False
        Show notched boxes (confidence interval around median).
    config : PlotConfig, optional
        Plot configuration.
    backend : {'matplotlib', 'plotly'}, optional
        Plotting backend.
    **kwargs
        Additional arguments passed to PlotSpec.
    
    Returns
    -------
    matplotlib.figure.Figure or plotly.graph_objects.Figure
        The generated figure.
    
    Examples
    --------
    >>> data = [np.random.randn(100), np.random.randn(100) + 1]
    >>> labels = ['Before', 'After']
    >>> fig = plot_box(data, labels=labels, notch=True, backend='plotly')
    """
    # Convert to dict if needed
    if isinstance(data, list):
        if labels is None:
            labels = [f'Group {i+1}' for i in range(len(data))]
        data_dict = dict(zip(labels, data))
    else:
        data_dict = data
        labels = list(data_dict.keys())
    
    # Set default colors
    if colors is None:
        colors = [DEFAULT_COLORS[i % len(DEFAULT_COLORS)] for i in range(len(labels))]
    
    # Create specs
    specs = []
    for label, color in zip(labels, colors):
        spec = PlotSpec(
            data=data_dict[label],
            plot_type='box',
            label=label,
            color=color,
            subplot_position=0,
            kwargs={'notch': notch, **kwargs}
        )
        specs.append(spec)
    
    grid = PlotGrid(
        plot_specs=specs,
        config=config or PlotConfig(title="Box Plot Comparison"),
        layout=GridLayoutConfig(rows=1, cols=1),
        backend=backend,
    )
    
    return grid.plot()


def plot_grouped_distributions(
    data: dict[str, dict[str, np.ndarray]],
    plot_type: Literal['violin', 'box'] = 'violin',
    colors: Sequence[str] | None = None,
    config: PlotConfig | None = None,
    layout: GridLayoutConfig | None = None,
    backend: Literal['matplotlib', 'plotly'] | None = None,
    **kwargs,
) -> Any:
    """
    Create multiple distribution plots grouped by category.
    
    This creates a grid where each subplot shows one group's distributions
    across multiple conditions.
    
    Parameters
    ----------
    data : dict of dict
        Nested dictionary: {group: {condition: data_array}}.
        Each group gets its own subplot.
    plot_type : {'violin', 'box'}, default='violin'
        Type of distribution plot.
    colors : sequence of str, optional
        Colors for conditions (cycles if more conditions than colors).
    config : PlotConfig, optional
        Plot configuration.
    layout : GridLayoutConfig, optional
        Grid layout configuration.
    backend : {'matplotlib', 'plotly'}, optional
        Plotting backend.
    **kwargs
        Additional arguments passed to PlotSpec.
    
    Returns
    -------
    matplotlib.figure.Figure or plotly.graph_objects.Figure
        The generated figure.
    
    Examples
    --------
    >>> data = {
    ...     'Region A': {
    ...         'Baseline': np.random.randn(100),
    ...         'Stimulus': np.random.randn(100) + 1,
    ...     },
    ...     'Region B': {
    ...         'Baseline': np.random.randn(100),
    ...         'Stimulus': np.random.randn(100) + 0.5,
    ...     },
    ... }
    >>> fig = plot_grouped_distributions(data, plot_type='violin', backend='plotly')
    """
    groups = list(data.keys())
    conditions = list(next(iter(data.values())).keys())
    
    # Set default colors
    if colors is None:
        colors = [DEFAULT_COLORS[i % len(DEFAULT_COLORS)] for i in range(len(conditions))]
    
    # Create specs
    specs = []
    for group_idx, group in enumerate(groups):
        for cond_idx, condition in enumerate(conditions):
            spec = PlotSpec(
                data=data[group][condition],
                plot_type=plot_type,
                subplot_position=group_idx,
                title=group,
                label=condition,
                color=colors[cond_idx % len(colors)],
                kwargs=kwargs
            )
            specs.append(spec)
    
    grid = PlotGrid(
        plot_specs=specs,
        config=config or PlotConfig(title=f"Grouped {plot_type.capitalize()} Plots"),
        layout=layout or GridLayoutConfig(),
        backend=backend,
    )
    
    return grid.plot()


def plot_comparison_distributions(
    data: dict[str, np.ndarray],
    plot_type: Literal['violin', 'box', 'histogram'] = 'violin',
    rows: int | None = None,
    cols: int | None = None,
    colors: Sequence[str] | None = None,
    config: PlotConfig | None = None,
    backend: Literal['matplotlib', 'plotly'] | None = None,
    **kwargs,
) -> Any:
    """
    Create separate distribution plots for each group in a grid.
    
    Each group gets its own subplot for side-by-side comparison.
    
    Parameters
    ----------
    data : dict
        Dictionary mapping group names to data arrays.
    plot_type : {'violin', 'box', 'histogram'}, default='violin'
        Type of distribution plot.
    rows : int, optional
        Number of rows in grid (auto-calculated if not specified).
    cols : int, optional
        Number of columns in grid (auto-calculated if not specified).
    colors : sequence of str, optional
        Colors for each group.
    config : PlotConfig, optional
        Plot configuration.
    backend : {'matplotlib', 'plotly'}, optional
        Plotting backend.
    **kwargs
        Additional arguments passed to PlotSpec.
    
    Returns
    -------
    matplotlib.figure.Figure or plotly.graph_objects.Figure
        The generated figure.
    
    Examples
    --------
    >>> data = {
    ...     'Control': np.random.randn(200),
    ...     'Low Dose': np.random.randn(200) + 0.5,
    ...     'High Dose': np.random.randn(200) + 1.0,
    ... }
    >>> fig = plot_comparison_distributions(data, plot_type='violin', backend='plotly')
    """
    groups = list(data.keys())
    
    # Create specs (each group gets its own subplot)
    specs = []
    
    # Check if data values are nested dicts (multiple conditions per group)
    first_value = next(iter(data.values()))
    is_nested = isinstance(first_value, dict)
    
    if is_nested:
        # Nested dict: {group: {condition: data}}
        # Each group gets multiple traces in one subplot
        conditions = list(first_value.keys())
        if colors is None:
            colors = [DEFAULT_COLORS[i % len(DEFAULT_COLORS)] for i in range(len(conditions))]
        
        for group_idx, group in enumerate(groups):
            for cond_idx, condition in enumerate(conditions):
                spec = PlotSpec(
                    data=data[group][condition],
                    plot_type=plot_type,
                    subplot_position=group_idx,
                    title=group,
                    label=condition,
                    color=colors[cond_idx % len(colors)],
                    kwargs=kwargs
                )
                specs.append(spec)
    else:
        # Simple dict: {group: data}
        # Each group gets its own subplot
        if colors is None:
            colors = [DEFAULT_COLORS[i % len(DEFAULT_COLORS)] for i in range(len(groups))]
        
        for idx, (group, color) in enumerate(zip(groups, colors)):
            spec = PlotSpec(
                data=data[group],
                plot_type=plot_type,
                subplot_position=idx,
                title=group,
                label=group,
                color=color,
                kwargs=kwargs
            )
            specs.append(spec)
    
    grid = PlotGrid(
        plot_specs=specs,
        config=config or PlotConfig(title=f"{plot_type.capitalize()} Comparison"),
        layout=GridLayoutConfig(rows=rows, cols=cols),
        backend=backend,
    )
    
    return grid.plot()
