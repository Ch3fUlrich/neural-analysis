"""
One-dimensional plotting functions.

This module provides convenience functions for creating 1D visualizations using
the PlotGrid system. All functions support both matplotlib and plotly backends.
"""

from typing import Literal, Any
import numpy as np
import numpy.typing as npt

from .grid_config import PlotGrid, PlotSpec, GridLayoutConfig
from .core import PlotConfig

__all__ = [
    "plot_line",
    "plot_multiple_lines",
    "plot_boolean_states",
]


def plot_line(
    data: npt.NDArray,
    x: npt.NDArray | None = None,
    config: PlotConfig | None = None,
    std: npt.NDArray | None = None,
    color: str | None = None,
    linewidth: float = 2.0,
    linestyle: str = '-',
    marker: str | None = None,
    markersize: float = 5.0,
    label: str | None = None,
    backend: Literal["matplotlib", "plotly"] | None = None,
) -> Any:
    """
    Plot a 1D line with optional error bands using PlotGrid.
    
    Parameters
    ----------
    data : ndarray
        1D array of y-values to plot.
    x : ndarray, optional
        1D array of x-values. If None, uses range(len(data)).
    config : PlotConfig, optional
        Plot configuration.
    std : ndarray, optional
        Standard deviation values for error band visualization.
    color, linewidth, linestyle, marker, markersize, label :
        Line styling parameters.
    backend : {'matplotlib', 'plotly'}, optional
        Backend to use.
        
    Returns
    -------
    Figure object from the backend.
    """
    # Validate
    data = np.atleast_1d(data)
    if data.ndim != 1:
        raise ValueError(f"Data must be 1D, got shape {data.shape}")
    
    if x is None:
        x = np.arange(len(data))
    else:
        x = np.atleast_1d(x)
        if len(x) != len(data):
            raise ValueError(f"x and data must have same length")
    
    # Prepare 2D data [x, y]
    line_data = np.column_stack([x, data])

    # Create spec with direct parameter assignment
    spec = PlotSpec(
        data=line_data,
        plot_type='line',
        color=color,
        label=label,
        subplot_position=0,
        line_width=linewidth,
        linestyle=linestyle,
        marker=marker,
        marker_size=markersize,
        error_y=std,
        alpha=1.0,
    )
    
    # Create grid
    grid = PlotGrid(
        plot_specs=[spec],
        layout=GridLayoutConfig(rows=1, cols=1),
        backend=backend,
        config=config,
    )
    
    return grid.plot()


def plot_multiple_lines(
    data_dict: dict[str, npt.NDArray],
    x: npt.NDArray | None = None,
    config: PlotConfig | None = None,
    colors: dict[str, str] | list[str] | None = None,
    linewidth: float = 2.0,
    linestyle: str = '-',
    alpha: float = 0.8,
    backend: Literal["matplotlib", "plotly"] | None = None,
) -> Any:
    """
    Plot multiple lines on the same axes using PlotGrid.
    
    Parameters
    ----------
    data_dict : dict
        Dictionary mapping line labels to 1D data arrays.
    x : ndarray, optional
        Shared x-values. If None, uses range.
    colors : dict or list, optional
        Colors for each line.
    Other parameters : Styling options.
        
    Returns
    -------
    Figure object from the backend.
    """
    if not data_dict:
        raise ValueError("data_dict cannot be empty")
    
    first_data = next(iter(data_dict.values()))
    if x is None:
        x = np.arange(len(first_data))
    
    # Prepare colors
    if colors is None:
        default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        color_map = {label: default_colors[i % len(default_colors)]
                     for i, label in enumerate(data_dict.keys())}
    elif isinstance(colors, dict):
        color_map = colors
    else:
        color_map = {label: colors[i % len(colors)]
                     for i, label in enumerate(data_dict.keys())}
    
    # Create specs
    specs = []
    for label, data in data_dict.items():
        data = np.atleast_1d(data)
        line_data = np.column_stack([x, data])
        spec = PlotSpec(
            data=line_data,
            plot_type='line',
            color=color_map.get(label),
            label=label,
            subplot_position=0,
            line_width=linewidth,
            linestyle=linestyle,
            alpha=alpha,
        )
        specs.append(spec)
    
    grid = PlotGrid(
        plot_specs=specs,
        layout=GridLayoutConfig(rows=1, cols=1),
        backend=backend,
        config=config,
    )
    
    return grid.plot()


def plot_boolean_states(
    states: npt.NDArray,
    x: npt.NDArray | None = None,
    config: PlotConfig | None = None,
    true_color: str = '#2ca02c',
    false_color: str = '#d62728',
    true_label: str = 'True',
    false_label: str = 'False',
    alpha: float = 0.3,
    backend: Literal["matplotlib", "plotly"] | None = None,
) -> Any:
    """
    Visualize boolean states over time using PlotGrid.
    
    Parameters
    ----------
    states : ndarray
        1D boolean array.
    x : ndarray, optional
        X-values. If None, uses range.
    config : PlotConfig, optional
        Plot configuration.
    true_color : str
        Color for True regions.
    false_color : str
        Color for False regions.
    true_label : str
        Label for True regions in legend.
    false_label : str
        Label for False regions in legend.
    alpha : float
        Transparency.
    backend : {'matplotlib', 'plotly'}, optional
        Backend to use.
        
    Returns
    -------
    Figure object from the backend.
    """
    states = np.atleast_1d(states).astype(bool)
    
    if x is None:
        x = np.arange(len(states))
    
    # Prepare 2D data [x, states_numeric] for PlotSpec
    states_numeric = states.astype(float)
    line_data = np.column_stack([x, states_numeric])
    
    # Set ylim to (0, 1) for boolean plots if not already set
    if config is None:
        config = PlotConfig(ylim=(0, 1))
    elif config.ylim is None:
        # Create a new config with ylim set
        config = PlotConfig(
            title=config.title,
            xlabel=config.xlabel,
            ylabel=config.ylabel,
            xlim=config.xlim,
            ylim=(0, 1),  # Force ylim for boolean plots
            grid=config.grid,
            figsize=config.figsize,
            dpi=config.dpi,
            show=config.show,
            legend=config.legend,
            cmap=config.cmap,
            alpha=config.alpha,
        )
    
    # Create PlotSpec for boolean states
    spec = PlotSpec(
        data=line_data,
        plot_type='boolean_states',
        subplot_position=0,
        alpha=alpha,
        true_color=true_color,
        false_color=false_color,
        true_label=true_label,
        false_label=false_label,
    )
    
    # Create grid
    grid = PlotGrid(
        plot_specs=[spec],
        layout=GridLayoutConfig(rows=1, cols=1),
        backend=backend,
        config=config,
    )
    
    return grid.plot()
