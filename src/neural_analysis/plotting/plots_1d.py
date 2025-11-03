"""
One-dimensional plotting functions.

This module provides functions for creating 1D visualizations including:
- Line plots with error bands
- Time series plots
- Loss curves
- Boolean state visualization

All functions support both matplotlib and plotly backends.

Example:
    >>> from neural_analysis.plotting import plot_line, PlotConfig
    >>> config = PlotConfig(title="My Data", xlabel="Time", ylabel="Value")
    >>> plot_line(data, config=config, backend='matplotlib')
"""

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from .backend import BackendType, get_backend
from .core import (
    PlotConfig,
    save_plot,
    apply_layout_matplotlib,
    apply_layout_plotly,
    finalize_plot_matplotlib,
    finalize_plot_plotly,
)

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
    ax: plt.Axes | None = None,
) -> plt.Axes | go.Figure:
    """
    Plot a 1D line with optional error bands.
    
    Creates a line plot with optional standard deviation bands. Supports both
    matplotlib for static publication-quality figures and plotly for interactive
    visualizations.
    
    Parameters
    ----------
    data : ndarray
        1D array of y-values to plot.
    x : ndarray, optional
        1D array of x-values. If None, uses range(len(data)).
    config : PlotConfig, optional
        Plot configuration. If None, uses defaults.
    std : ndarray, optional
        Standard deviation values for error band visualization.
    color : str, optional
        Line color. If None, uses default from backend.
    linewidth : float, default=2.0
        Width of the line.
    linestyle : str, default='-'
        Line style ('-', '--', '-.', ':').
    marker : str, optional
        Marker style ('o', 's', '^', etc.). If None, no markers.
    markersize : float, default=5.0
        Size of markers if marker is specified.
    label : str, optional
        Label for legend.
    backend : {'matplotlib', 'plotly'}, optional
        Backend to use. If None, uses current global backend.
    ax : matplotlib Axes, optional
        Existing axes to plot on (matplotlib only). If None, creates new figure.
        
    Returns
    -------
    matplotlib.axes.Axes or plotly.graph_objects.Figure
        The plot object.
        
    Raises
    ------
    ValueError
        If data is not 1D or if plotly backend requested but not installed.
        
    Examples
    --------
    >>> # Simple line plot
    >>> data = np.array([1, 4, 2, 5, 3])
    >>> plot_line(data, config=PlotConfig(title="My Data"))
    
    >>> # Line plot with error bands
    >>> data = np.random.randn(100).cumsum()
    >>> std = np.ones(100) * 0.5
    >>> plot_line(data, std=std, label="Signal")
    
    >>> # Time series with custom x-axis
    >>> time = np.linspace(0, 10, 100)
    >>> signal = np.sin(time)
    >>> config = PlotConfig(xlabel="Time (s)", ylabel="Amplitude")
    >>> plot_line(signal, x=time, config=config)
    
    >>> # Interactive plotly plot
    >>> plot_line(data, backend='plotly', config=PlotConfig(title="Interactive"))
    """
    # Validate inputs
    data = np.atleast_1d(data)
    if data.ndim != 1:
        raise ValueError(f"Data must be 1D, got shape {data.shape}")
    
    # Set default configuration
    if config is None:
        config = PlotConfig()
    
    # Set x values if not provided
    if x is None:
        x = np.arange(len(data))
    else:
        x = np.atleast_1d(x)
        if len(x) != len(data):
            raise ValueError(
                f"x and data must have same length: {len(x)} != {len(data)}"
            )

    # Determine backend
    backend_type = get_backend() if backend is None else BackendType(backend)

    # Create plot based on backend
    if backend_type == BackendType.MATPLOTLIB:
        return _plot_line_matplotlib(
            x, data, config, std, color, linewidth, linestyle,
            marker, markersize, label, ax
        )
    else:  # PLOTLY
        if not PLOTLY_AVAILABLE:
            raise ValueError("Plotly backend requested but plotly is not installed")
        return _plot_line_plotly(
            x, data, config, std, color, linewidth, linestyle,
            marker, markersize, label
        )


def _plot_line_matplotlib(
    x: npt.NDArray,
    data: npt.NDArray,
    config: PlotConfig,
    std: npt.NDArray | None,
    color: str | None,
    linewidth: float,
    linestyle: str,
    marker: str | None,
    markersize: float,
    label: str | None,
    ax: plt.Axes | None,
) -> plt.Axes:
    """Matplotlib implementation of line plot."""
    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
    
    # Plot line
    ax.plot(
        x, data,
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        marker=marker,
        markersize=markersize,
        label=label,
        alpha=config.alpha
    )
    
    # Add error band if std provided
    if std is not None:
        std = np.atleast_1d(std)
        if len(std) != len(data):
            raise ValueError(
                f"std and data must have same length: {len(std)} != {len(data)}"
            )
        ax.fill_between(
            x,
            data - std,
            data + std,
            color=color,
            alpha=config.alpha * 0.3,
            linewidth=0
        )
    
    # Configure axes using common helper
    apply_layout_matplotlib(ax, config)
    if config.legend and label:
        ax.legend()
    
    # Finalize (save and show)
    finalize_plot_matplotlib(config)
    
    return ax


def _plot_line_plotly(
    x: npt.NDArray,
    data: npt.NDArray,
    config: PlotConfig,
    std: npt.NDArray | None,
    color: str | None,
    linewidth: float,
    linestyle: str,
    marker: str | None,
    markersize: float,
    label: str | None,
) -> go.Figure:
    """Plotly implementation of line plot."""
    fig = go.Figure()
    
    # Convert linestyle to plotly format
    dash_map = {'-': 'solid', '--': 'dash', '-.': 'dashdot', ':': 'dot'}
    dash = dash_map.get(linestyle, 'solid')
    
    # Add error band if std provided
    if std is not None:
        std = np.atleast_1d(std)
        # Upper bound
        fig.add_trace(go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([data + std, (data - std)[::-1]]),
            fill='toself',
            fillcolor=color if color else 'rgba(0,100,200,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Add main line
    mode = 'lines' if marker is None else 'lines+markers'
    fig.add_trace(go.Scatter(
        x=x,
        y=data,
        mode=mode,
        name=label,
        line=dict(
            color=color,
            width=linewidth,
            dash=dash
        ),
        marker=dict(
            symbol=marker if marker else 'circle',
            size=markersize
        ) if marker else None,
        opacity=config.alpha
    ))
    
    # Configure layout using common helper
    apply_layout_plotly(fig, config)
    fig.update_layout(
        showlegend=config.legend and label is not None,
        template='plotly_white'
    )
    
    # Finalize (save and show)
    finalize_plot_plotly(fig, config)
    
    return fig


def plot_multiple_lines(
    data_dict: dict[str, npt.NDArray],
    x: npt.NDArray | None = None,
    config: PlotConfig | None = None,
    colors: list[str] | None = None,
    backend: Literal["matplotlib", "plotly"] | None = None,
) -> plt.Axes | go.Figure:
    """
    Plot multiple lines on the same axes.
    
    Parameters
    ----------
    data_dict : dict
        Dictionary mapping labels to 1D data arrays.
    x : ndarray, optional
        Shared x-values for all lines. If None, uses range(len(data)).
    config : PlotConfig, optional
        Plot configuration.
    colors : list of str, optional
        Colors for each line. If None, uses default color cycle.
    backend : {'matplotlib', 'plotly'}, optional
        Backend to use. If None, uses current global backend.
        
    Returns
    -------
    matplotlib.axes.Axes or plotly.graph_objects.Figure
        The plot object.
        
    Examples
    --------
    >>> data = {
    ...     'sine': np.sin(np.linspace(0, 10, 100)),
    ...     'cosine': np.cos(np.linspace(0, 10, 100))
    ... }
    >>> plot_multiple_lines(data, config=PlotConfig(title="Trig Functions"))
    """
    if config is None:
        config = PlotConfig()
    
    # Determine backend
    backend_type = get_backend() if backend is None else BackendType(backend)
    
    if backend_type == BackendType.MATPLOTLIB:
        fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
        
        for i, (label, data) in enumerate(data_dict.items()):
            color = colors[i] if colors and i < len(colors) else None
            plot_line(
                data, x=x, config=PlotConfig(legend=False, show=False),
                color=color, label=label, backend='matplotlib', ax=ax
            )
        
        # Apply config to final plot using common helper
        apply_layout_matplotlib(ax, config)
        if config.legend:
            ax.legend()
        
        finalize_plot_matplotlib(config)
        
        return ax
    
    else:  # PLOTLY
        if not PLOTLY_AVAILABLE:
            raise ValueError("Plotly backend requested but plotly is not installed")
        
        fig = go.Figure()
        
        for i, (label, data) in enumerate(data_dict.items()):
            data = np.atleast_1d(data)
            x_vals = x if x is not None else np.arange(len(data))
            color = colors[i] if colors and i < len(colors) else None
            
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=data,
                mode='lines',
                name=label,
                line=dict(color=color) if color else None
            ))
        
        # Configure layout using common helper
        apply_layout_plotly(fig, config)
        fig.update_layout(
            showlegend=config.legend,
            template='plotly_white'
        )
        
        finalize_plot_plotly(fig, config)
        
        return fig

def plot_boolean_states(
    states: npt.NDArray,
    x: npt.NDArray | None = None,
    config: PlotConfig | None = None,
    true_color: str = 'blue',
    false_color: str = 'white',
    true_label: str = 'True',
    false_label: str = 'False',
    alpha: float = 0.3,
    backend: Literal["matplotlib", "plotly"] | None = None,
) -> plt.Axes | go.Figure:
    """
    Visualize boolean states over time.
    
    Creates a filled area plot showing regions where a boolean condition is
    True vs False. Useful for visualizing behavioral states, movement periods, etc.
    
    Parameters
    ----------
    states : ndarray
        1D boolean array.
    x : ndarray, optional
        X-axis values. If None, uses range(len(states)).
    config : PlotConfig, optional
        Plot configuration.
    true_color : str, default='blue'
        Color for True states.
    false_color : str, default='white'
        Color for False states.
    true_label : str, default='True'
        Label for True states in legend.
    false_label : str, default='False'
        Label for False states in legend.
    alpha : float, default=0.3
        Transparency of filled regions.
    backend : {'matplotlib', 'plotly'}, optional
        Backend to use. If None, uses current global backend.
        
    Returns
    -------
    matplotlib.axes.Axes or plotly.graph_objects.Figure
        The plot object.
        
    Examples
    --------
    >>> # Visualize movement states
    >>> is_moving = np.array([True, True, False, False, True, True, True])
    >>> config = PlotConfig(title="Movement States", ylabel="State")
    >>> plot_boolean_states(is_moving, config=config, 
    ...                     true_label="Moving", false_label="Stationary")
    """
    states = np.atleast_1d(states).astype(bool)
    if x is None:
        x = np.arange(len(states))
    
    if config is None:
        config = PlotConfig()
    
    # Determine backend
    backend_type = get_backend() if backend is None else BackendType(backend)
    
    if backend_type == BackendType.MATPLOTLIB:
        fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
        
        # Fill regions based on state
        ax.fill_between(
            x, 0, 1,
            where=states,
            color=true_color,
            alpha=alpha,
            step='post',
            label=true_label,
            interpolate=False
        )
        ax.fill_between(
            x, 0, 1,
            where=~states,
            color=false_color,
            alpha=alpha,
            step='post',
            label=false_label,
            interpolate=False
        )
        
        # Configure using common helper
        ax.set_ylim(0, 1)
        apply_layout_matplotlib(ax, config)
        if config.legend:
            ax.legend()
        
        finalize_plot_matplotlib(config)
        
        return ax
    
    else:  # PLOTLY
        if not PLOTLY_AVAILABLE:
            raise ValueError("Plotly backend requested but plotly is not installed")
        
        fig = go.Figure()
        
        # Add filled regions
        # Group consecutive True values
        changes = np.diff(np.concatenate([[False], states, [False]]).astype(int))
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]
        
        for start, end in zip(starts, ends):
            # Convert color to RGBA for plotly
            rgb = plt.matplotlib.colors.to_rgb(true_color)
            rgba = f'rgba{tuple(list(rgb) + [alpha])}'
            fig.add_trace(go.Scatter(
                x=[x[start], x[end-1], x[end-1], x[start]],
                y=[0, 0, 1, 1],
                fill='toself',
                fillcolor=rgba,
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Configure layout using common helper
        apply_layout_plotly(fig, config)
        fig.update_layout(
            yaxis=dict(range=[0, 1]),
            template='plotly_white'
        )
        
        finalize_plot_plotly(fig, config)
        
        return fig
