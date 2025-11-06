"""
Plot rendering functions for the PlotGrid system.

This module provides low-level rendering functions for different plot types.
These functions are used internally by PlotGrid and can be reused across
different plot modules.

All functions follow a consistent interface:
- Take data, styling parameters, and backend-specific axes/figure
- Return trace/plot object
- Support legend tracking to avoid duplicates
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    import plotly.graph_objects as go

try:
    import plotly.graph_objects as go  # type: ignore[no-redef]
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


# ==============================================================================
# Data Extraction Helpers
# ==============================================================================

def extract_xy_from_data(data: dict | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract x and y coordinates from various data formats.
    
    Parameters
    ----------
    data : dict or np.ndarray
        Either a dict with 'x' and 'y' keys, or a (n, 2) array
        
    Returns
    -------
    x, y : tuple of np.ndarray
        X and Y coordinate arrays
        
    Raises
    ------
    ValueError
        If data format is not recognized
    """
    if isinstance(data, dict) and 'x' in data and 'y' in data:
        return np.asarray(data['x']), np.asarray(data['y'])
    elif isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[1] == 2:
        return data[:, 0], data[:, 1]
    else:
        raise ValueError("data must be dict with 'x','y' keys or (n,2) array")


def extract_xyz_from_data(data: dict | np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract x, y, and z coordinates from various data formats.
    
    Parameters
    ----------
    data : dict or np.ndarray
        Either a dict with 'x', 'y', 'z' keys, or a (n, 3) array
        
    Returns
    -------
    x, y, z : tuple of np.ndarray
        X, Y, and Z coordinate arrays
        
    Raises
    ------
    ValueError
        If data format is not recognized
    """
    if isinstance(data, dict) and 'x' in data and 'y' in data and 'z' in data:
        return np.asarray(data['x']), np.asarray(data['y']), np.asarray(data['z'])
    elif isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[1] == 3:
        return data[:, 0], data[:, 1], data[:, 2]
    else:
        raise ValueError("data must be dict with 'x','y','z' keys or (n,3) array")


# ==============================================================================
# Export List
# ==============================================================================

__all__ = [
    "extract_xy_from_data",
    "extract_xyz_from_data",
    "render_scatter_matplotlib",
    "render_scatter_plotly",
    "render_scatter3d_plotly",
    "render_line_matplotlib",
    "render_line_plotly",
    "render_histogram_matplotlib",
    "render_histogram_plotly",
    "render_heatmap_matplotlib",
    "render_heatmap_plotly",
    "render_bar_plotly",
    "render_violin_matplotlib",
    "render_violin_plotly",
    "render_box_matplotlib",
    "render_box_plotly",
    "render_trajectory_matplotlib",
    "render_trajectory_plotly",
    "render_trajectory3d_matplotlib",
    "render_trajectory3d_plotly",
    "render_kde_matplotlib",
    "render_kde_plotly",
    "render_convex_hull_matplotlib",
    "render_convex_hull_plotly",
]


# ============================================================================
# Scatter Plots
# ============================================================================

def render_scatter_matplotlib(
    ax,
    data: npt.NDArray,
    color: str | None = None,
    marker: str = 'o',
    marker_size: float | None = None,
    alpha: float = 0.7,
    label: str | None = None,
    **kwargs
):
    """
    Render a 2D scatter plot using matplotlib.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes or mpl_toolkits.mplot3d.axes3d.Axes3D
        Axes to plot on (can be 2D or 3D)
    data : ndarray
        2D array with shape (n_points, 2) or (n_points, 3) containing coordinates
    color : str, optional
        Color for the markers
    marker : str, default='o'
        Marker style ('o', 's', 'D', '^', etc.)
    marker_size : float, optional
        Size of markers (default: 20)
    alpha : float, default=0.7
        Opacity of markers (0-1)
    label : str, optional
        Label for legend
    **kwargs
        Additional keyword arguments passed to ax.scatter()
        
    Returns
    -------
    PathCollection
        The matplotlib scatter plot object
    """
    if data.shape[1] == 2:
        # 2D scatter
        return ax.scatter(
            data[:, 0], data[:, 1],
            c=color, s=marker_size or 20,
            marker=marker,
            alpha=alpha, label=label,
            **kwargs
        )
    elif data.shape[1] == 3:
        # 3D scatter
        return ax.scatter(
            data[:, 0], data[:, 1], data[:, 2],
            c=color, s=marker_size or 20,
            marker=marker,
            alpha=alpha, label=label,
            **kwargs
        )
    else:
        raise ValueError(f"Scatter plot requires 2D or 3D data, got shape {data.shape}")


def render_scatter_plotly(
    data: npt.NDArray,
    color: str | None = None,
    colors: npt.NDArray | None = None,
    cmap: str | None = None,
    marker: str = 'circle',
    marker_size: float | None = None,
    sizes: npt.NDArray | None = None,
    alpha: float = 0.7,
    label: str | None = None,
    showlegend: bool = True,
    colorbar: bool = False,
    colorbar_label: str | None = None,
    **kwargs
) -> go.Scatter:
    """
    Render a 2D scatter plot using plotly.
    
    Parameters
    ----------
    data : ndarray
        2D array with shape (n_points, 2) containing x, y coordinates
    color : str, optional
        Solid color for all markers (if colors is None)
    colors : ndarray, optional
        Array of color values for colormap (overrides color parameter)
    cmap : str, optional
        Colormap name to use with colors array
    marker : str, default='circle'
        Marker symbol ('circle', 'square', 'diamond', etc.)
    marker_size : float, optional
        Fixed size for all markers (default: 8)
    sizes : ndarray, optional
        Array of marker sizes (overrides marker_size)
    alpha : float, default=0.7
        Opacity of markers (0-1)
    label : str, optional
        Label for legend
    showlegend : bool, default=True
        Whether to show this trace in the legend
    colorbar : bool, default=False
        Whether to show colorbar (when colors is provided)
    colorbar_label : str, optional
        Label for colorbar
    **kwargs
        Additional keyword arguments passed to go.Scatter()
        
    Returns
    -------
    plotly.graph_objects.Scatter
        The plotly scatter trace
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is required for this function")
    
    if data.shape[1] != 2:
        raise ValueError("2D scatter requires 2-column data")
    
    # Build marker dict
    marker_dict = {
        'symbol': marker,
        'size': sizes if sizes is not None else (marker_size or 8),
        'opacity': alpha,
    }
    
    # Handle colors: array of values for colormap or single color
    if colors is not None and len(colors) > 0:
        marker_dict['color'] = colors
        if cmap:
            marker_dict['colorscale'] = cmap
        if colorbar:
            marker_dict['colorbar'] = {'title': colorbar_label or ''}
            marker_dict['showscale'] = True
    elif color is not None:
        marker_dict['color'] = color
    
    return go.Scatter(
        x=data[:, 0], y=data[:, 1],
        mode='markers',
        marker=marker_dict,
        name=label or '',
        showlegend=showlegend,
        **kwargs
    )


def render_scatter3d_plotly(
    data: npt.NDArray,
    color: str | None = None,
    colors: npt.NDArray | None = None,
    cmap: str | None = None,
    marker_size: float | None = None,
    sizes: npt.NDArray | None = None,
    alpha: float = 0.7,
    label: str | None = None,
    showlegend: bool = True,
    colorbar: bool = False,
    colorbar_label: str | None = None,
    **kwargs
) -> go.Scatter3d:
    """
    Render a 3D scatter plot using plotly.
    
    Parameters
    ----------
    data : ndarray
        2D array with shape (n_points, 3) containing x, y, z coordinates
    color : str, optional
        Solid color for all markers (if colors is None)
    colors : ndarray, optional
        Array of color values for colormap (overrides color parameter)
    cmap : str, optional
        Colormap name to use with colors array
    marker_size : float, optional
        Fixed size for all markers (default: 4)
    sizes : ndarray, optional
        Array of marker sizes (overrides marker_size)
    alpha : float, default=0.7
        Opacity of markers (0-1)
    label : str, optional
        Label for legend
    showlegend : bool, default=True
        Whether to show this trace in the legend
    colorbar : bool, default=False
        Whether to show colorbar (when colors is provided)
    colorbar_label : str, optional
        Label for colorbar
    **kwargs
        Additional keyword arguments passed to go.Scatter3d()
        
    Returns
    -------
    plotly.graph_objects.Scatter3d
        The plotly 3D scatter trace
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is required for this function")
    
    if data.shape[1] != 3:
        raise ValueError("3D scatter requires 3-column data")
    
    # Build marker dict
    marker_dict = {
        'size': sizes if sizes is not None else (marker_size or 4),
        'opacity': alpha,
    }
    
    # Handle colors: array of values for colormap or single color
    if colors is not None and len(colors) > 0:
        marker_dict['color'] = colors
        if cmap:
            marker_dict['colorscale'] = cmap
        if colorbar:
            marker_dict['colorbar'] = {'title': colorbar_label or ''}
            marker_dict['showscale'] = True
    elif color is not None:
        marker_dict['color'] = color
    
    return go.Scatter3d(
        x=data[:, 0], y=data[:, 1], z=data[:, 2],
        mode='markers',
        marker=marker_dict,
        name=label or '',
        showlegend=showlegend,
        **kwargs
    )


# ============================================================================
# Line Plots
# ============================================================================

def render_line_matplotlib(
    ax,
    data: npt.NDArray,
    color: str | None = None,
    line_width: float = 1.5,
    linestyle: str = '-',
    marker: str | None = None,
    marker_size: float | None = None,
    error_y: npt.NDArray | None = None,
    alpha: float = 1.0,
    label: str | None = None,
    show_values: bool = False,
    value_format: str = '.3f',
    x_labels: list | None = None,
    **kwargs
):
    """
    Render a line plot using matplotlib with optional error bands.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    data : ndarray
        1D array of y-values, or 2D array with [x, y] columns
    color : str, optional
        Line color
    line_width : float, default=1.5
        Width of the line
    linestyle : str, default='-'
        Line style ('-', '--', '-.', ':', etc.')
    error_y : ndarray, optional
        Error bar values for y-axis (creates shaded error band)
    alpha : float, default=1.0
        Opacity of the line (0-1)
    label : str, optional
        Label for legend
    show_values : bool, default=False
        Whether to show value labels on points
    value_format : str, default='.3f'
        Format string for value labels
    x_labels : list, optional
        Custom labels for x-axis ticks
    **kwargs
        Additional keyword arguments passed to ax.plot()
        
    Returns
    -------
    list
        List of Line2D objects
    """
    # Extract custom parameters that matplotlib doesn't support directly
    kwargs.pop('false_color', None)
    kwargs.pop('true_label', None)
    kwargs.pop('false_label', None)
    # Pop axis label parameters (handled by PlotConfig)
    kwargs.pop('x_label', None)
    kwargs.pop('y_label', None)
    
    # Handle 2D data with [x, y] columns
    if data.ndim == 2 and data.shape[1] == 2:
        x = data[:, 0]
        y = data[:, 1]
    elif data.ndim == 1:
        # 1D data - use indices as x
        x = np.arange(len(data))
        y = data
    else:
        # Multiple lines (old behavior for compatibility)
        # Build kwargs for matplotlib
        plot_kwargs = {}
        if marker is not None:
            plot_kwargs['marker'] = marker
            if marker_size is not None:
                plot_kwargs['markersize'] = marker_size
        
        lines = []
        for i in range(data.shape[1]):
            line = ax.plot(
                data[:, i],
                color=color, linewidth=line_width, linestyle=linestyle,
                alpha=alpha, label=label if i == 0 else None,
                **plot_kwargs,
                **kwargs
            )
            lines.extend(line)
        return lines
    
    # Build kwargs for matplotlib
    plot_kwargs = {}
    if marker is not None:
        plot_kwargs['marker'] = marker
        if marker_size is not None:
            plot_kwargs['markersize'] = marker_size
    
    # Plot the main line
    lines = ax.plot(
        x, y,
        color=color, linewidth=line_width, linestyle=linestyle,
        alpha=alpha, label=label,
        **plot_kwargs
    )
    
    # Add value labels if requested
    if show_values:
        y_range = y.max() - y.min() if len(y) > 0 else 1
        offset = y_range * 0.02
        for xi, yi in zip(x, y):
            ax.text(xi, yi + offset, f'{yi:{value_format}}', 
                   ha='center', va='bottom', fontsize=9)
    
    # Set custom x-axis labels if provided
    if x_labels is not None:
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
    
    # Add error band if provided
    if error_y is not None:
        error_y = np.asarray(error_y)
        fill_color = color if color else lines[0].get_color()
        ax.fill_between(
            x,
            y - error_y,
            y + error_y,
            color=fill_color,
            alpha=0.2,
            linewidth=0,
        )
    
    return lines


def render_line_plotly(
    data: npt.NDArray,
    color: str | None = None,
    line_width: float = 2,
    linestyle: str | None = None,
    error_y: npt.NDArray | None = None,
    alpha: float = 1.0,
    label: str | None = None,
    showlegend: bool = True,
    **kwargs
) -> go.Scatter:
    """
    Render a line plot using plotly with optional error bands.
    
    Parameters
    ----------
    data : ndarray
        1D array of y-values, or 2D array with [x, y] columns
    color : str, optional
        Line color
    line_width : float, default=2
        Width of the line
    linestyle : str, optional
        Line style ('solid', 'dash', 'dot', 'dashdot'). 
        Matplotlib styles ('-', '--', '-.', ':') are auto-converted.
    error_y : ndarray, optional
        Error bar values for y-axis
    alpha : float, default=1.0
        Opacity of the line (0-1)
    label : str, optional
        Label for legend
    showlegend : bool, default=True
        Whether to show this trace in the legend
    **kwargs
        Additional keyword arguments passed to go.Scatter()
        
    Returns
    -------
    plotly.graph_objects.Scatter
        The plotly line trace
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is required for this function")
    
    # Convert matplotlib linestyle to plotly dash style
    linestyle_map = {
        '-': 'solid',
        '--': 'dash',
        '-.': 'dashdot',
        ':': 'dot',
    }
    dash_style = linestyle_map.get(linestyle, linestyle) if linestyle else None
    
    line_dict = {
        'color': color,
        'width': line_width,
    }
    if dash_style:
        line_dict['dash'] = dash_style
    
    # Prepare error_y dict if error values provided
    error_y_dict = None
    if error_y is not None:
        error_y = np.asarray(error_y)
        error_y_dict = dict(
            type='data',
            array=error_y,
            visible=True,
            color=color if color else 'rgba(0,0,0,0.3)',
        )
    
    if data.ndim == 1:
        # 1D data: use indices as x
        return go.Scatter(
            y=data,
            mode='lines',
            line=line_dict,
            error_y=error_y_dict,
            opacity=alpha,
            name=label or '',
            showlegend=showlegend,
            **kwargs
        )
    elif data.shape[1] == 2:
        # 2D data: x and y columns
        return go.Scatter(
            x=data[:, 0],
            y=data[:, 1],
            mode='lines',
            line=line_dict,
            error_y=error_y_dict,
            opacity=alpha,
            name=label or '',
            showlegend=showlegend,
            **kwargs
        )
    else:
        # Multiple y values: use first column
        return go.Scatter(
            y=data[:, 0],
            mode='lines',
            line=dict(color=color, width=line_width),
            error_y=error_y_dict,
            opacity=alpha,
            name=label or '',
            showlegend=showlegend,
            **kwargs
        )


# ============================================================================
# Histogram Plots
# ============================================================================

def render_histogram_matplotlib(
    ax,
    data: npt.NDArray,
    color: str | None = None,
    alpha: float = 0.7,
    bins: int = 30,
    label: str | None = None,
    **kwargs
):
    """
    Render a histogram using matplotlib.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    data : ndarray
        1D array of values
    color : str, optional
        Bar color
    alpha : float, default=0.7
        Opacity of bars (0-1)
    bins : int, default=30
        Number of bins
    label : str, optional
        Label for legend
    **kwargs
        Additional keyword arguments passed to ax.hist()
        
    Returns
    -------
    tuple
        (n, bins, patches) from matplotlib hist()
    """
    return ax.hist(
        data,
        color=color, alpha=alpha,
        label=label, bins=bins,
        **kwargs
    )


def render_histogram_plotly(
    data: npt.NDArray,
    color: str | None = None,
    alpha: float = 0.7,
    bins: int = 30,
    label: str | None = None,
    showlegend: bool = True,
    **kwargs
) -> go.Histogram:
    """
    Render a histogram using plotly.
    
    Parameters
    ----------
    data : ndarray
        1D array of values
    color : str, optional
        Bar color
    alpha : float, default=0.7
        Opacity of bars (0-1)
    bins : int, default=30
        Number of bins
    label : str, optional
        Label for legend
    showlegend : bool, default=True
        Whether to show this trace in the legend
    **kwargs
        Additional keyword arguments passed to go.Histogram()
        
    Returns
    -------
    plotly.graph_objects.Histogram
        The plotly histogram trace
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is required for this function")
    
    return go.Histogram(
        x=data,
        marker=dict(color=color),
        opacity=alpha,
        name=label or '',
        showlegend=showlegend,
        nbinsx=bins,
        **kwargs
    )


# ============================================================================
# Heatmap Plots
# ============================================================================

def render_heatmap_matplotlib(
    ax,
    data: npt.NDArray,
    cmap: str = 'viridis',
    alpha: float = 1.0,
    **kwargs
):
    """
    Render a heatmap using matplotlib.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    data : ndarray
        2D array of values
    cmap : str, default='viridis'
        Colormap name
    alpha : float, default=1.0
        Opacity (0-1)
    **kwargs
        Additional keyword arguments passed to ax.imshow()
        
    Returns
    -------
    AxesImage
        The matplotlib image object
    """
    import matplotlib.pyplot as plt
    
    # Extract parameters that matplotlib doesn't support directly
    colorbar_enabled = kwargs.pop('colorbar', True)
    colorbar_label = kwargs.pop('colorbar_label', None)
    x_labels = kwargs.pop('x_labels', None)
    y_labels = kwargs.pop('y_labels', None)
    show_values = kwargs.pop('show_values', False)
    value_format = kwargs.pop('value_format', '.2f')
    
    im = ax.imshow(
        data,
        cmap=cmap,
        alpha=alpha,
        **kwargs
    )
    
    # Add colorbar if requested
    if colorbar_enabled:
        cbar = plt.colorbar(im, ax=ax)
        if colorbar_label:
            cbar.set_label(colorbar_label)
    
    # Add axis labels if provided
    if x_labels is not None:
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels)
    if y_labels is not None:
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels)
    
    # Add value annotations if requested
    if show_values:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, format(data[i, j], value_format),
                             ha="center", va="center", color="black")
    
    return im


def render_heatmap_plotly(
    data: npt.NDArray,
    cmap: str | None = None,
    colorscale: str | None = None,
    **kwargs
) -> go.Heatmap:
    """
    Render a heatmap using plotly.
    
    Parameters
    ----------
    data : ndarray
        2D array of values
    cmap : str, optional
        Colormap name (matplotlib style, converted to plotly)
    colorscale : str, optional
        Plotly colorscale name (takes precedence over cmap)
    **kwargs
        Additional keyword arguments passed to go.Heatmap()
        
    Returns
    -------
    plotly.graph_objects.Heatmap
        The plotly heatmap trace
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is required for this function")
    
    # Extract colorbar boolean and label, remove incompatible parameters
    colorbar_enabled = kwargs.pop('colorbar', True)
    colorbar_label = kwargs.pop('colorbar_label', None)
    kwargs.pop('x_labels', None)  # Remove - not directly supported
    kwargs.pop('y_labels', None)  # Remove - not directly supported
    kwargs.pop('show_values', None)  # Remove - needs implementation
    kwargs.pop('value_format', None)  # Remove - needs implementation
    kwargs.pop('alpha', None)  # Remove - plotly uses opacity
    
    # Use colorscale if provided, otherwise convert cmap
    scale = colorscale or cmap or 'Viridis'
    
    # Build colorbar config
    colorbar_config = dict(title=colorbar_label or "Value") if colorbar_enabled else None
    
    return go.Heatmap(
        z=data,
        colorscale=scale,
        showlegend=False,
        colorbar=colorbar_config,
        **kwargs
    )


# ============================================================================
# Bar Plots
# ============================================================================

def render_bar_matplotlib(
    ax,
    data: npt.NDArray,
    x: npt.NDArray | None = None,
    color: str | None = None,
    colors: list | None = None,
    alpha: float = 0.7,
    label: str | None = None,
    orientation: str = 'v',
    error_y: npt.NDArray | None = None,
    error_x: npt.NDArray | None = None,
    show_values: bool = False,
    value_format: str = '.3f',
    x_labels: list | None = None,
    **kwargs
) -> Any:
    """
    Render a bar plot using matplotlib.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on
    data : ndarray
        1D array of bar heights
    x : ndarray, optional
        X-axis positions
    color : str, optional
        Single bar color (used if colors is None)
    colors : list, optional
        List of colors for each bar
    alpha : float, default=0.7
        Opacity of bars (0-1)
    label : str, optional
        Label for legend
    orientation : {'v', 'h'}, default='v'
        Vertical or horizontal bars
    error_y : ndarray, optional
        Error bar values for vertical bars
    error_x : ndarray, optional
        Error bar values for horizontal bars
    show_values : bool, default=False
        Whether to show value labels on bars
    value_format : str, default='.3f'
        Format string for value labels
    x_labels : list, optional
        Custom labels for x-axis ticks
    **kwargs
        Additional keyword arguments passed to ax.bar()
        
    Returns
    -------
    matplotlib.container.BarContainer
        The bar container
    """
    # Pop custom parameters that matplotlib doesn't support directly
    kwargs.pop('x_label', None)  # Will be handled by PlotConfig
    kwargs.pop('y_label', None)  # Will be handled by PlotConfig
    
    if x is None:
        x = np.arange(len(data))
    
    # Use colors array if provided, otherwise single color
    bar_color = colors if colors is not None else color
    
    if orientation == 'h':
        bars = ax.barh(x, data, color=bar_color, alpha=alpha, label=label, 
                      xerr=error_x, **kwargs)
        
        # Add value labels if requested
        if show_values:
            for i, (pos, val) in enumerate(zip(x, data)):
                offset = max(data) * 0.02 if error_x is None else max(data) * 0.02 + (error_x[i] if hasattr(error_x, '__getitem__') else 0)
                ax.text(val + offset, pos, f'{val:{value_format}}', 
                       va='center', ha='left', fontsize=9)
        
        # Set custom y-axis labels if provided
        if x_labels is not None:
            ax.set_yticks(x)
            ax.set_yticklabels(x_labels)
    else:
        bars = ax.bar(x, data, color=bar_color, alpha=alpha, label=label,
                     yerr=error_y, **kwargs)
        
        # Add value labels if requested
        if show_values:
            for i, (pos, val) in enumerate(zip(x, data)):
                offset = max(data) * 0.02 if error_y is None else max(data) * 0.02 + (error_y[i] if hasattr(error_y, '__getitem__') else 0)
                ax.text(pos, val + offset, f'{val:{value_format}}', 
                       ha='center', va='bottom', fontsize=9)
        
        # Set custom x-axis labels if provided
        if x_labels is not None:
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels)
    
    return bars


def render_bar_plotly(
    data: npt.NDArray,
    x: npt.NDArray | None = None,
    color: str | None = None,
    colors: list | None = None,
    alpha: float = 0.7,
    label: str | None = None,
    showlegend: bool = True,
    error_y: npt.NDArray | None = None,
    error_x: npt.NDArray | None = None,
    **kwargs
) -> go.Bar:
    """
    Render a bar plot using plotly.
    
    Parameters
    ----------
    data : ndarray
        1D array of bar heights
    x : ndarray, optional
        X-axis positions
    color : str, optional
        Single bar color (used if colors is None)
    colors : list, optional
        List of colors for each bar
    alpha : float, default=0.7
        Opacity of bars (0-1)
    label : str, optional
        Label for legend
    showlegend : bool, default=True
        Whether to show this trace in the legend
    error_y : ndarray, optional
        Error bar values for y-axis
    error_x : ndarray, optional
        Error bar values for x-axis
    **kwargs
        Additional keyword arguments passed to go.Bar()
        
    Returns
    -------
    plotly.graph_objects.Bar
        The plotly bar trace
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is required for this function")
    
    # Use colors array if provided, otherwise single color
    bar_color = colors if colors is not None else color
    
    # Build marker dict
    marker_dict = {'color': bar_color, 'opacity': alpha}
    
    # Build error bars
    error_y_dict = None
    error_x_dict = None
    if error_y is not None:
        error_y_dict = dict(type='data', array=error_y, visible=True)
    if error_x is not None:
        error_x_dict = dict(type='data', array=error_x, visible=True)
    
    return go.Bar(
        x=x,
        y=data if data.ndim == 1 else data[:, 0],
        marker=marker_dict,
        name=label or '',
        showlegend=showlegend,
        error_y=error_y_dict,
        error_x=error_x_dict,
        **kwargs
    )


# ============================================================================
# Violin Plots
# ============================================================================

def render_violin_matplotlib(
    ax,
    data: npt.NDArray,
    position: int = 1,
    color: str | None = None,
    alpha: float = 0.7,
    showmeans: bool = True,
    showmedians: bool = True,
    showbox: bool = True,
    showpoints: bool = True,
    label: str | None = None,
    **kwargs
):
    """
    Render a half violin plot (right side) with points on the left using matplotlib.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    data : ndarray
        1D array of values
    position : int, default=1
        X-axis position for this violin
    color : str, optional
        Violin color
    alpha : float, default=0.7
        Opacity (0-1)
    showmeans : bool, default=True
        Show mean line
    showmedians : bool, default=True
        Show median line
    showbox : bool, default=True
        Show box plot (not used in half-violin implementation)
    showpoints : bool, default=True
        Show individual data points on the left side
    label : str, optional
        Label for legend
    **kwargs
        Additional keyword arguments passed to violinplot()
        
    Returns
    -------
    dict
        Dictionary with violin plot components
    """
    import numpy as np
    
    result = {}
    
    # Filter out plotly-specific parameters from kwargs
    violin_kwargs = {k: v for k, v in kwargs.items() if k not in ['meanline']}
    
    # Create violin plot with minimal extras (we'll customize the body)
    parts = ax.violinplot(
        [data],
        positions=[position],
        showmeans=False,  # We'll draw this ourselves if needed
        showmedians=False,  # We'll draw this ourselves if needed
        showextrema=False,  # Don't show whiskers
        widths=0.6,
        **violin_kwargs
    )
    
    # Modify violin to show only right half
    for pc in parts['bodies']:
        # Get the vertices of the violin
        verts = pc.get_paths()[0].vertices.copy()
        
        # Keep only the right half (x >= position)
        # The violin path is symmetric, so we filter and reconstruct
        right_mask = verts[:, 0] >= position
        right_verts = verts[right_mask]
        
        # Sort by y to ensure proper ordering
        sorted_indices = np.argsort(right_verts[:, 1])
        right_verts = right_verts[sorted_indices]
        
        # Get y range
        y_min = right_verts[:, 1].min()
        y_max = right_verts[:, 1].max()
        
        # Create straight left edge along the centerline
        # We build: bottom point -> all right side points (sorted by y) -> top point -> close
        new_verts = np.vstack([
            [[position, y_min]],  # Bottom of centerline
            right_verts,           # Right side curve
            [[position, y_max]],  # Top of centerline
            [[position, y_min]]   # Close path
        ])
        
        # Update the path
        from matplotlib.path import Path
        pc.get_paths()[0] = Path(new_verts)
        
        # Apply styling
        if color:
            pc.set_facecolor(color)
            pc.set_edgecolor('black')
            pc.set_alpha(alpha)
            pc.set_linewidth(1)
    
    # Manually draw statistics if requested (similar to matplotlib example)
    if showbox or showmeans or showmedians:
        quartile1 = np.percentile(data, 25)
        median = np.percentile(data, 50)
        quartile3 = np.percentile(data, 75)
        mean = np.mean(data)
        
        # Draw box (quartile lines) at the centerline
        if showbox:
            # Vertical line from Q1 to Q3
            ax.vlines(position, quartile1, quartile3, color='k', linestyle='-', lw=5, zorder=4)
        
        # Draw median
        if showmedians:
            ax.scatter([position], [median], marker='o', color='white', s=30, zorder=5, edgecolors='black', linewidths=1.5)
        
        # Draw mean
        if showmeans:
            ax.scatter([position], [mean], marker='D', color='red', s=30, zorder=5, edgecolors='darkred', linewidths=1)
    
    result['violin'] = parts
    
    # Add individual points on the left side if requested
    if showpoints:
        # Add jitter to x positions on the LEFT side
        x_jitter = np.random.normal(position - 0.1, 0.05, size=len(data))
        scatter = ax.scatter(
            x_jitter, data, 
            alpha=alpha * 0.6, 
            s=15, 
            color=color or 'black',
            zorder=3
        )
        result['points'] = scatter
    
    # Add legend entry
    if label:
        from matplotlib.patches import Patch
        legend_patch = Patch(facecolor=color or 'C0', alpha=alpha, label=label)
        # Store for later legend creation
        result['legend_handle'] = legend_patch
    
    return result


def render_violin_plotly(
    data: npt.NDArray,
    color: str | None = None,
    alpha: float = 0.7,
    meanline: dict | None = None,
    showbox: bool = True,
    showpoints: bool = True,
    label: str | None = None,
    showlegend: bool = True,
    **kwargs
) -> go.Violin:
    """
    Render a half violin plot (right side) with points on the left using plotly.
    
    Parameters
    ----------
    data : ndarray
        1D array of values
    color : str, optional
        Violin color
    alpha : float, default=0.7
        Opacity (0-1)
    meanline : dict, optional
        Dictionary with meanline configuration (e.g., {'visible': True})
    showbox : bool, default=True
        Show box plot inside violin
    showpoints : bool, default=True
        Show individual data points on the left
    label : str, optional
        Label for legend
    showlegend : bool, default=True
        Whether to show this trace in the legend
    **kwargs
        Additional keyword arguments passed to go.Violin()
        
    Returns
    -------
    plotly.graph_objects.Violin
        The plotly violin trace
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is required for this function")
    
    # Handle meanline configuration - make it more visible
    if meanline is None:
        meanline = {'visible': True, 'color': color or 'black', 'width': 2}
    elif isinstance(meanline, bool):
        meanline = {'visible': meanline, 'color': color or 'black', 'width': 2}
    elif isinstance(meanline, dict):
        # Enhance existing meanline config
        if 'visible' not in meanline:
            meanline['visible'] = True
        if 'width' not in meanline:
            meanline['width'] = 2
        if 'color' not in meanline:
            meanline['color'] = color or 'black'
    
    # Configure points display on the LEFT side
    if showpoints:
        points = 'all'
        pointpos = -0.8  # Position points to the left (negative = left side)
        jitter = 0.3
    else:
        points = False
        pointpos = 0
        jitter = 0
    
    return go.Violin(
        y=data,
        name=label or '',
        marker=dict(color=color),
        opacity=alpha,
        showlegend=showlegend,
        meanline=meanline,
        box_visible=showbox,
        box=dict(
            visible=showbox,
            fillcolor='rgba(255, 255, 255, 0.5)',  # Semi-transparent white so inner lines are visible
            line=dict(color=color or 'black', width=2),  # Thicker outline
            width=0.3  # Increased box width from default (~0.15)
        ) if showbox else None,
        points=points,
        pointpos=pointpos,
        jitter=jitter,
        side='positive',  # Show only RIGHT half of violin
        **kwargs
    )


# ============================================================================
# Box Plots
# ============================================================================

def render_box_matplotlib(
    ax,
    data: npt.NDArray,
    position: int = 1,
    color: str | None = None,
    alpha: float = 0.7,
    label: str | None = None,
    notch: bool = False,
    showpoints: bool = True,
    **kwargs
):
    """
    Render a box plot with sample points using matplotlib.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    data : ndarray
        1D array of values
    position : int, default=1
        X-axis position for this box plot
    color : str, optional
        Box color
    alpha : float, default=0.7
        Opacity (0-1)
    label : str, optional
        Label for legend
    notch : bool, default=False
        Whether to show notches
    showpoints : bool, default=True
        Whether to show individual sample points
    **kwargs
        Additional keyword arguments passed to ax.boxplot()
        
    Returns
    -------
    dict
        Dictionary with box plot components
    """
    import numpy as np
    
    # Create box plot
    bp = ax.boxplot(
        [data],
        positions=[position],
        widths=0.5,
        patch_artist=True,
        notch=notch,
        showfliers=False,  # Don't show outliers as we'll show all points
        **kwargs
    )
    
    # Apply color
    if color:
        for patch in bp['boxes']:
            patch.set_facecolor(color)
            patch.set_alpha(alpha)
        # Also color the whiskers, caps, and medians
        for element in ['whiskers', 'caps', 'medians']:
            for item in bp[element]:
                item.set_color(color)
                item.set_alpha(alpha)
    
    # Add individual sample points if requested
    if showpoints:
        # Add jitter to x positions
        x_jitter = np.random.normal(position, 0.08, size=len(data))
        scatter = ax.scatter(
            x_jitter, data,
            alpha=alpha * 0.4,
            s=15,
            color=color or 'black',
            zorder=3
        )
        bp['points'] = scatter
    
    # Add legend entry
    if label:
        from matplotlib.patches import Patch
        legend_patch = Patch(facecolor=color or 'C0', alpha=alpha, label=label)
        bp['legend_handle'] = legend_patch
    
    return bp


def render_box_plotly(
    data: npt.NDArray,
    color: str | None = None,
    alpha: float = 0.7,
    label: str | None = None,
    showlegend: bool = True,
    notched: bool = False,
    showpoints: bool = True,
    **kwargs
) -> go.Box:
    """
    Render a box plot with sample points using plotly.
    
    Parameters
    ----------
    data : ndarray
        1D array of values
    color : str, optional
        Box color
    alpha : float, default=0.7
        Opacity (0-1)
    label : str, optional
        Label for legend
    showlegend : bool, default=True
        Whether to show this trace in the legend
    notched : bool, default=False
        Whether to show notches
    showpoints : bool, default=True
        Whether to show individual sample points
    **kwargs
        Additional keyword arguments passed to go.Box()
        
    Returns
    -------
    plotly.graph_objects.Box
        The plotly box trace
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is required for this function")
    
    # Configure points display
    if showpoints:
        boxpoints = 'all'  # Show all points
        jitter = 0.3
        pointpos = 0  # Center the points over the box
    else:
        boxpoints = False
        jitter = 0
        pointpos = 0
    
    return go.Box(
        y=data,
        name=label or '',
        marker=dict(color=color),
        opacity=alpha,
        showlegend=showlegend,
        notched=notched,
        boxpoints=boxpoints,
        jitter=jitter,
        pointpos=pointpos,
        **kwargs
    )


# ============================================================================
# Trajectory Plots
# ============================================================================

def render_trajectory_matplotlib(
    ax,
    x: npt.NDArray,
    y: npt.NDArray,
    colors: npt.NDArray | None = None,
    cmap: str = "viridis",
    linewidth: float = 2.0,
    alpha: float = 1.0,
    show_points: bool = False,
    point_color: str = "black",
    point_size: float = 20.0,
    colorbar: bool = True,
    colorbar_label: str | None = None,
    label: str | None = None,
    **kwargs
) -> Any:
    """
    Render a 2D trajectory using matplotlib LineCollection.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on
    x : array-like
        X coordinates
    y : array-like
        Y coordinates
    colors : array-like, optional
        Color values for each segment (length n-1)
    cmap : str
        Colormap name
    linewidth : float
        Line width
    alpha : float
        Line transparency
    show_points : bool
        Whether to show scatter points
    point_color : str
        Color for scatter points
    point_size : float
        Size of scatter points
    colorbar : bool
        Whether to show colorbar
    colorbar_label : str, optional
        Label for the colorbar
    label : str, optional
        Label for legend
    **kwargs
        Additional arguments for LineCollection
        
    Returns
    -------
    LineCollection
        The matplotlib LineCollection object
    """
    from matplotlib import pyplot as plt
    from matplotlib.collections import LineCollection
    
    # Calculate segments internally
    if len(x) != len(y):
        raise ValueError(f"x and y must have same length, got {len(x)} and {len(y)}")
    
    # Handle single point case - just show as scatter
    if len(x) == 1:
        if show_points:
            ax.scatter(x, y, c=point_color, s=point_size, alpha=alpha, label=label)
        return ax
    
    if len(x) < 1:
        raise ValueError("Need at least 1 point for trajectory")
    
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # Create LineCollection
    if colors is not None:
        lc = LineCollection(segments, cmap=cmap, alpha=alpha, linewidths=linewidth, **kwargs)
        lc.set_array(colors)
    else:
        lc = LineCollection(segments, alpha=alpha, linewidths=linewidth, label=label, **kwargs)
    
    ax.add_collection(lc)
    
    # Add colorbar if requested and colors provided
    if colorbar and colors is not None:
        cbar = plt.colorbar(lc, ax=ax)
        if colorbar_label:
            cbar.set_label(colorbar_label)
    
    # Add scatter points if requested
    if show_points:
        ax.scatter(x, y, c=point_color, s=point_size, zorder=5, alpha=alpha)
    
    # Auto-scale axes
    ax.autoscale()
    
    return lc


def render_trajectory_plotly(
    x: npt.NDArray,
    y: npt.NDArray,
    colors: npt.NDArray | None = None,
    cmap: str = "Viridis",
    linewidth: float = 2.0,
    alpha: float = 1.0,
    show_points: bool = False,
    point_size: float = 5.0,
    colorbar: bool = True,
    colorbar_label: str | None = None,
    label: str | None = None,
    showlegend: bool = True,
    **kwargs
) -> Any:
    """
    Render a 2D trajectory using plotly.
    
    Parameters
    ----------
    x : array-like
        X coordinates
    y : array-like
        Y coordinates
    colors : array-like, optional
        Color values for coloring (same length as x, y)
    cmap : str
        Colormap name
    linewidth : float
        Line width
    alpha : float
        Line transparency
    show_points : bool
        Whether to show markers
    point_size : float
        Size of markers
    colorbar : bool
        Whether to show colorbar
    colorbar_label : str, optional
        Label for the colorbar
    label : str, optional
        Trace name for legend
    showlegend : bool
        Whether to show in legend
    **kwargs
        Additional arguments for go.Scatter
        
    Returns
    -------
    go.Scatter
        The plotly scatter trace
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is required for this function")
    
    marker_config = dict(size=point_size) if show_points else None
    mode = "lines+markers" if show_points else "lines"
    
    if colors is not None:
        # Convert numpy array to list for Plotly compatibility
        colors_list = colors.tolist() if hasattr(colors, 'tolist') else colors
        
        return go.Scatter(
            x=x,
            y=y,
            mode=mode,
            name=label or '',
            line=dict(width=linewidth),
            marker=dict(
                size=point_size,
                color=colors_list,  # Use marker color instead of line color
                colorscale=cmap,
                showscale=colorbar,
                colorbar=dict(title=colorbar_label) if colorbar and colorbar_label else None,
            ),
            opacity=alpha,
            showlegend=showlegend,
            **kwargs
        )
    else:
        return go.Scatter(
            x=x,
            y=y,
            mode=mode,
            name=label or '',
            line=dict(width=linewidth),
            marker=marker_config,
            opacity=alpha,
            showlegend=showlegend,
            **kwargs
        )


def render_trajectory3d_plotly(
    x: npt.NDArray,
    y: npt.NDArray,
    z: npt.NDArray,
    colors: npt.NDArray | None = None,
    cmap: str = "Viridis",
    linewidth: float = 2.0,
    alpha: float = 1.0,
    show_points: bool = False,
    point_size: float = 3.0,
    colorbar: bool = True,
    colorbar_label: str | None = None,
    label: str | None = None,
    showlegend: bool = True,
    **kwargs
) -> Any:
    """
    Render a 3D trajectory using plotly.
    
    Parameters
    ----------
    x : array-like
        X coordinates
    y : array-like
        Y coordinates
    z : array-like
        Z coordinates
    colors : array-like, optional
        Color values for coloring (same length as x, y, z)
    cmap : str
        Colormap name
    linewidth : float
        Line width
    alpha : float
        Line transparency
    show_points : bool
        Whether to show markers
    point_size : float
        Size of markers
    colorbar : bool
        Whether to show colorbar
    colorbar_label : str, optional
        Label for the colorbar
    label : str, optional
        Trace name for legend
    showlegend : bool
        Whether to show in legend
    **kwargs
        Additional arguments for go.Scatter3d
        
    Returns
    -------
    go.Scatter3d
        The plotly 3D scatter trace
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is required for this function")
    
    marker_config = dict(size=point_size) if show_points else dict(size=0.1)
    mode = "lines+markers" if show_points else "lines"
    
    if colors is not None:
        marker_config.update({
            'color': colors,
            'colorscale': cmap,
            'showscale': colorbar,
            'colorbar': dict(title=colorbar_label) if colorbar_label else {}
        })
        
    return go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode=mode,
        name=label or '',
        line=dict(width=linewidth),
        marker=marker_config,
        opacity=alpha,
        showlegend=showlegend,
        **kwargs
    )


def render_trajectory3d_matplotlib(
    ax,
    x: npt.NDArray,
    y: npt.NDArray,
    z: npt.NDArray,
    colors: npt.NDArray | None = None,
    cmap: str = "viridis",
    linewidth: float = 2.0,
    alpha: float = 1.0,
    show_points: bool = False,
    point_size: float = 10.0,
    colorbar: bool = True,
    colorbar_label: str | None = None,
    label: str | None = None,
    **kwargs
) -> Any:
    """
    Render a 3D trajectory using matplotlib Line3DCollection.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes3D
        The 3D axes to plot on
    x : array-like
        X coordinates
    y : array-like
        Y coordinates
    z : array-like
        Z coordinates
    colors : array-like, optional
        Color values for each segment (length n-1)
    cmap : str
        Colormap name
    linewidth : float
        Line width
    alpha : float
        Line transparency
    show_points : bool
        Whether to show scatter points
    point_size : float
        Size of scatter points
    colorbar : bool
        Whether to show colorbar
    colorbar_label : str, optional
        Label for the colorbar
    label : str, optional
        Label for legend
    **kwargs
        Additional arguments for Line3DCollection
        
    Returns
    -------
    Line3DCollection
        The matplotlib Line3DCollection object
    """
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    
    # Calculate segments internally
    if not (len(x) == len(y) == len(z)):
        raise ValueError(f"x, y, z must have same length, got {len(x)}, {len(y)}, {len(z)}")
    if len(x) < 2:
        raise ValueError("Need at least 2 points for trajectory")
    
    points = np.array([x, y, z]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # Create Line3DCollection
    if colors is not None:
        lc = Line3DCollection(segments, cmap=cmap, alpha=alpha, linewidths=linewidth, **kwargs)
        lc.set_array(colors)
    else:
        lc = Line3DCollection(segments, alpha=alpha, linewidths=linewidth, **kwargs)
    
    ax.add_collection3d(lc)
    
    # Add colorbar if requested and colors provided
    if colorbar and colors is not None:
        cbar = plt.colorbar(lc, ax=ax, shrink=0.8, pad=0.1)
        if colorbar_label:
            cbar.set_label(colorbar_label)
    
    # Add scatter points if requested
    if show_points:
        if colors is not None:
            ax.scatter(x, y, z, c=colors, s=point_size, cmap=cmap, alpha=alpha)
        else:
            ax.scatter(x, y, z, s=point_size, alpha=alpha)
    
    # Auto-scale axes
    ax.autoscale()
    
    return lc


# ============================================================================
# KDE (Kernel Density Estimation) Plots
# ============================================================================

def render_kde_matplotlib(
    ax,
    xi: npt.NDArray,
    yi: npt.NDArray,
    zi: npt.NDArray,
    fill: bool = True,
    n_levels: int = 10,
    cmap: str = "viridis",
    alpha: float = 0.6,
    colorbar: bool = True,
    colorbar_label: str | None = None,
    label: str | None = None,
    **kwargs
) -> Any:
    """
    Render a 2D KDE plot using matplotlib contour/contourf.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on
    xi : ndarray
        X coordinates of the grid (2D array)
    yi : ndarray
        Y coordinates of the grid (2D array)
    zi : ndarray
        Density values on the grid (2D array)
    fill : bool
        Whether to fill contours (contourf) or just lines (contour)
    n_levels : int
        Number of contour levels
    cmap : str
        Colormap name
    alpha : float
        Transparency
    colorbar : bool
        Whether to show colorbar
    colorbar_label : str, optional
        Label for the colorbar
    label : str, optional
        Label for legend (note: contours don't support labels well)
    **kwargs
        Additional arguments for contour/contourf
        
    Returns
    -------
    ContourSet
        The matplotlib contour object
    """
    from matplotlib import pyplot as plt
    
    if fill:
        contour = ax.contourf(xi, yi, zi, levels=n_levels, cmap=cmap, alpha=alpha, **kwargs)
    else:
        contour = ax.contour(xi, yi, zi, levels=n_levels, cmap=cmap, alpha=alpha, **kwargs)
    
    if colorbar:
        cbar = plt.colorbar(contour, ax=ax)
        if colorbar_label:
            cbar.set_label(colorbar_label)
    
    return contour


def render_kde_plotly(
    xi: npt.NDArray,
    yi: npt.NDArray,
    zi: npt.NDArray,
    fill: bool = True,
    n_levels: int = 10,
    cmap: str = "Viridis",
    alpha: float = 0.6,
    colorbar: bool = True,
    colorbar_label: str | None = None,
    label: str | None = None,
    showlegend: bool = True,
    **kwargs
) -> Any:
    """
    Render a 2D KDE plot using plotly contour.
    
    Parameters
    ----------
    xi : ndarray
        X coordinates of the grid (2D array)
    yi : ndarray
        Y coordinates of the grid (2D array)
    zi : ndarray
        Density values on the grid (2D array)
    fill : bool
        Whether to fill contours
    n_levels : int
        Number of contour levels
    cmap : str
        Colormap name
    alpha : float
        Transparency
    colorbar : bool
        Whether to show colorbar
    colorbar_label : str, optional
        Label for the colorbar
    label : str, optional
        Trace name for legend
    showlegend : bool
        Whether to show in legend
    **kwargs
        Additional arguments for go.Contour
        
    Returns
    -------
    go.Contour
        The plotly contour trace
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is required for this function")
    
    contour_config = dict(
        start=float(np.min(zi)),
        end=float(np.max(zi)),
        size=(float(np.max(zi)) - float(np.min(zi))) / n_levels
    )
    
    return go.Contour(
        x=xi[0, :] if xi.ndim == 2 else xi,
        y=yi[:, 0] if yi.ndim == 2 else yi,
        z=zi,
        name=label or '',
        colorscale=cmap,
        opacity=alpha,
        showlegend=showlegend,
        showscale=colorbar,
        colorbar=dict(title=colorbar_label) if colorbar_label else {},
        contours=contour_config,
        **kwargs
    )


# ============================================================================
# Convex Hull Plots
# ============================================================================

def render_convex_hull_matplotlib(
    ax,
    hull_x: npt.NDArray,
    hull_y: npt.NDArray,
    color: str = "black",
    linewidth: float = 2.0,
    linestyle: str = "-",
    alpha: float = 1.0,
    fill: bool = False,
    fill_alpha: float = 0.2,
    label: str | None = None,
    **kwargs
) -> Any:
    """
    Render a convex hull boundary using matplotlib.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on
    hull_x : array-like
        X coordinates of hull boundary (closed loop)
    hull_y : array-like
        Y coordinates of hull boundary (closed loop)
    color : str
        Line color
    linewidth : float
        Line width
    linestyle : str
        Line style
    alpha : float
        Line transparency
    fill : bool
        Whether to fill the hull
    fill_alpha : float
        Fill transparency
    label : str, optional
        Label for legend
    **kwargs
        Additional arguments for plot/fill
        
    Returns
    -------
    Line2D or Polygon
        The matplotlib artist object
    """
    # Plot boundary
    line = ax.plot(hull_x, hull_y, color=color, linewidth=linewidth, 
                   linestyle=linestyle, alpha=alpha, label=label, **kwargs)[0]
    
    # Fill if requested
    if fill:
        ax.fill(hull_x, hull_y, color=color, alpha=fill_alpha)
    
    return line


def render_convex_hull_plotly(
    hull_x: npt.NDArray,
    hull_y: npt.NDArray,
    color: str = "black",
    linewidth: float = 2.0,
    alpha: float = 1.0,
    fill: bool = False,
    fill_alpha: float = 0.2,
    label: str | None = None,
    showlegend: bool = True,
    **kwargs
) -> Any:
    """
    Render a convex hull boundary using plotly.
    
    Parameters
    ----------
    hull_x : array-like
        X coordinates of hull boundary (closed loop)
    hull_y : array-like
        Y coordinates of hull boundary (closed loop)
    color : str
        Line color
    linewidth : float
        Line width
    alpha : float
        Line transparency
    fill : bool
        Whether to fill the hull
    fill_alpha : float
        Fill transparency (only applies if fill=True)
    label : str, optional
        Trace name for legend
    showlegend : bool
        Whether to show in legend
    **kwargs
        Additional arguments for go.Scatter
        
    Returns
    -------
    go.Scatter
        The plotly scatter trace
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is required for this function")
    
    fill_mode = "toself" if fill else "none"
    fill_color = color if fill else None
    
    return go.Scatter(
        x=hull_x,
        y=hull_y,
        mode="lines",
        name=label or '',
        line=dict(color=color, width=linewidth),
        fill=fill_mode,
        fillcolor=fill_color,
        opacity=alpha if not fill else fill_alpha,
        showlegend=showlegend,
        **kwargs
    )


def render_boolean_states_matplotlib(
    ax,
    x: npt.NDArray,
    states: npt.NDArray,
    true_color: str = '#2ca02c',
    false_color: str = '#d62728',
    true_label: str = 'True',
    false_label: str = 'False',
    alpha: float = 0.3,
    **kwargs
):
    """
    Render boolean states as filled regions using matplotlib.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    x : ndarray
        X-values (time points)
    states : ndarray
        Boolean array indicating states
    true_color : str, default='#2ca02c'
        Color for True regions
    false_color : str, default='#d62728'
        Color for False regions
    true_label : str, default='True'
        Label for True regions in legend
    false_label : str, default='False'
        Label for False regions in legend
    alpha : float, default=0.3
        Transparency for filled regions
    **kwargs
        Additional arguments (unused, for compatibility)
        
    Returns
    -------
    list
        List of matplotlib artist objects (PolyCollection)
    """
    artists = []
    
    # Find transitions to create segments
    state_changes = np.diff(np.concatenate([[False], states, [False]]).astype(int))
    true_starts = np.where(state_changes == 1)[0]
    true_ends = np.where(state_changes == -1)[0]
    
    # Plot true regions
    for i, (start, end) in enumerate(zip(true_starts, true_ends)):
        poly = ax.axvspan(
            x[start], x[end-1], 
            color=true_color, 
            alpha=alpha, 
            label=true_label if i == 0 else ""
        )
        artists.append(poly)
    
    # Find false regions
    false_starts = np.where(state_changes == -1)[0]
    false_ends = np.where(state_changes == 1)[0]
    
    # Handle edge cases for false regions
    if not states[0]:  # Starts with False
        false_starts = np.concatenate([[0], false_starts])
    if not states[-1]:  # Ends with False
        false_ends = np.concatenate([false_ends, [len(states)]])
    
    # Adjust lengths
    min_len = min(len(false_starts), len(false_ends))
    false_starts = false_starts[:min_len]
    false_ends = false_ends[:min_len]
    
    # Plot false regions
    for i, (start, end) in enumerate(zip(false_starts, false_ends)):
        poly = ax.axvspan(
            x[start], x[end-1] if end < len(x) else x[-1], 
            color=false_color, 
            alpha=alpha, 
            label=false_label if i == 0 else ""
        )
        artists.append(poly)
    
    return artists


def render_boolean_states_plotly(
    x: npt.NDArray,
    states: npt.NDArray,
    true_color: str = '#2ca02c',
    false_color: str = '#d62728',
    true_label: str = 'True',
    false_label: str = 'False',
    alpha: float = 0.3,
    **kwargs
):
    """
    Render boolean states as filled regions using plotly.
    
    Parameters
    ----------
    x : ndarray
        X-values (time points)
    states : ndarray
        Boolean array indicating states
    true_color : str, default='#2ca02c'
        Color for True regions
    false_color : str, default='#d62728'
        Color for False regions
    true_label : str, default='True'
        Label for True regions in legend
    false_label : str, default='False'
        Label for False regions in legend
    alpha : float, default=0.3
        Transparency for filled regions
    **kwargs
        Additional arguments (unused, for compatibility)
        
    Returns
    -------
    list
        List of plotly traces
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is required for this function")
    
    traces = []
    
    # Find transitions to create segments
    state_changes = np.diff(np.concatenate([[False], states, [False]]).astype(int))
    true_starts = np.where(state_changes == 1)[0]
    true_ends = np.where(state_changes == -1)[0]
    
    # Plot true regions
    for i, (start, end) in enumerate(zip(true_starts, true_ends)):
        trace = go.Scatter(
            x=[x[start], x[end-1], x[end-1], x[start], x[start]],
            y=[0, 0, 1, 1, 0],
            fill='toself',
            fillcolor=true_color,
            line=dict(width=0),
            opacity=alpha,
            name=true_label if i == 0 else '',
            showlegend=(i == 0),
            **kwargs
        )
        traces.append(trace)
    
    # Find false regions
    false_starts = np.where(state_changes == -1)[0]
    false_ends = np.where(state_changes == 1)[0]
    
    # Handle edge cases for false regions
    if not states[0]:
        false_starts = np.concatenate([[0], false_starts])
    if not states[-1]:
        false_ends = np.concatenate([false_ends, [len(states)]])
    
    # Adjust lengths
    min_len = min(len(false_starts), len(false_ends))
    false_starts = false_starts[:min_len]
    false_ends = false_ends[:min_len]
    
    # Plot false regions
    for i, (start, end) in enumerate(zip(false_starts, false_ends)):
        x_end = x[end-1] if end < len(x) else x[-1]
        trace = go.Scatter(
            x=[x[start], x_end, x_end, x[start], x[start]],
            y=[0, 0, 1, 1, 0],
            fill='toself',
            fillcolor=false_color,
            line=dict(width=0),
            opacity=alpha,
            name=false_label if i == 0 else '',
            showlegend=(i == 0),
            **kwargs
        )
        traces.append(trace)
    
    return traces

