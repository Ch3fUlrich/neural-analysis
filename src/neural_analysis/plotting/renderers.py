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

from typing import Literal, Any, Optional
import numpy as np
import numpy.typing as npt

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None

__all__ = [
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
]


# ============================================================================
# Scatter Plots
# ============================================================================

def render_scatter_matplotlib(
    ax,
    data: npt.NDArray,
    color: Optional[str] = None,
    marker: str = 'o',
    marker_size: Optional[float] = None,
    alpha: float = 0.7,
    label: Optional[str] = None,
    **kwargs
):
    """
    Render a 2D scatter plot using matplotlib.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    data : ndarray
        2D array with shape (n_points, 2) containing x, y coordinates
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
    if data.shape[1] != 2:
        raise ValueError("Scatter plot requires 2D data (n_points, 2)")
    
    return ax.scatter(
        data[:, 0], data[:, 1],
        c=color, s=marker_size or 20,
        marker=marker,
        alpha=alpha, label=label,
        **kwargs
    )


def render_scatter_plotly(
    data: npt.NDArray,
    color: Optional[str] = None,
    marker: str = 'circle',
    marker_size: Optional[float] = None,
    alpha: float = 0.7,
    label: Optional[str] = None,
    showlegend: bool = True,
    **kwargs
) -> 'go.Scatter':
    """
    Render a 2D scatter plot using plotly.
    
    Parameters
    ----------
    data : ndarray
        2D array with shape (n_points, 2) containing x, y coordinates
    color : str, optional
        Color for the markers
    marker : str, default='circle'
        Marker symbol ('circle', 'square', 'diamond', etc.)
    marker_size : float, optional
        Size of markers (default: 8)
    alpha : float, default=0.7
        Opacity of markers (0-1)
    label : str, optional
        Label for legend
    showlegend : bool, default=True
        Whether to show this trace in the legend
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
    
    return go.Scatter(
        x=data[:, 0], y=data[:, 1],
        mode='markers',
        marker=dict(
            color=color,
            size=marker_size or 8,
            symbol=marker,
            opacity=alpha,
        ),
        name=label or '',
        showlegend=showlegend,
        **kwargs
    )


def render_scatter3d_plotly(
    data: npt.NDArray,
    color: Optional[str] = None,
    marker_size: Optional[float] = None,
    alpha: float = 0.7,
    label: Optional[str] = None,
    showlegend: bool = True,
    **kwargs
) -> 'go.Scatter3d':
    """
    Render a 3D scatter plot using plotly.
    
    Parameters
    ----------
    data : ndarray
        2D array with shape (n_points, 3) containing x, y, z coordinates
    color : str, optional
        Color for the markers
    marker_size : float, optional
        Size of markers (default: 4)
    alpha : float, default=0.7
        Opacity of markers (0-1)
    label : str, optional
        Label for legend
    showlegend : bool, default=True
        Whether to show this trace in the legend
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
    
    return go.Scatter3d(
        x=data[:, 0], y=data[:, 1], z=data[:, 2],
        mode='markers',
        marker=dict(
            color=color,
            size=marker_size or 4,
            opacity=alpha,
        ),
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
    color: Optional[str] = None,
    line_width: float = 1.5,
    alpha: float = 1.0,
    label: Optional[str] = None,
    **kwargs
):
    """
    Render a line plot using matplotlib.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    data : ndarray
        1D array of y-values or 2D array with x, y columns
    color : str, optional
        Line color
    line_width : float, default=1.5
        Width of the line
    alpha : float, default=1.0
        Opacity of the line (0-1)
    label : str, optional
        Label for legend
    **kwargs
        Additional keyword arguments passed to ax.plot()
        
    Returns
    -------
    list
        List of Line2D objects
    """
    if data.ndim == 1:
        return ax.plot(
            data,
            color=color, linewidth=line_width,
            alpha=alpha, label=label,
            **kwargs
        )
    else:
        # Multiple lines
        lines = []
        for i in range(data.shape[1]):
            line = ax.plot(
                data[:, i],
                color=color, linewidth=line_width,
                alpha=alpha, label=label if i == 0 else None,
                **kwargs
            )
            lines.extend(line)
        return lines


def render_line_plotly(
    data: npt.NDArray,
    color: Optional[str] = None,
    line_width: float = 2,
    alpha: float = 1.0,
    label: Optional[str] = None,
    showlegend: bool = True,
    **kwargs
) -> 'go.Scatter':
    """
    Render a line plot using plotly.
    
    Parameters
    ----------
    data : ndarray
        1D array of y-values or 2D array with x, y columns
    color : str, optional
        Line color
    line_width : float, default=2
        Width of the line
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
    
    if data.ndim == 1:
        # 1D data: use indices as x
        return go.Scatter(
            y=data,
            mode='lines',
            line=dict(
                color=color,
                width=line_width,
            ),
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
            line=dict(
                color=color,
                width=line_width,
            ),
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
    color: Optional[str] = None,
    alpha: float = 0.7,
    bins: int = 30,
    label: Optional[str] = None,
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
    color: Optional[str] = None,
    alpha: float = 0.7,
    bins: int = 30,
    label: Optional[str] = None,
    showlegend: bool = True,
    **kwargs
) -> 'go.Histogram':
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
    
    im = ax.imshow(
        data,
        cmap=cmap,
        alpha=alpha,
        **kwargs
    )
    plt.colorbar(im, ax=ax)
    return im


def render_heatmap_plotly(
    data: npt.NDArray,
    cmap: Optional[str] = None,
    colorscale: Optional[str] = None,
    **kwargs
) -> 'go.Heatmap':
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
    
    # Use colorscale if provided, otherwise convert cmap
    scale = colorscale or cmap or 'Viridis'
    
    return go.Heatmap(
        z=data,
        colorscale=scale,
        showlegend=False,
        **kwargs
    )


# ============================================================================
# Bar Plots
# ============================================================================

def render_bar_plotly(
    data: npt.NDArray,
    x: Optional[npt.NDArray] = None,
    color: Optional[str] = None,
    alpha: float = 0.7,
    label: Optional[str] = None,
    showlegend: bool = True,
    **kwargs
) -> 'go.Bar':
    """
    Render a bar plot using plotly.
    
    Parameters
    ----------
    data : ndarray
        1D array of bar heights
    x : ndarray, optional
        X-axis positions
    color : str, optional
        Bar color
    alpha : float, default=0.7
        Opacity of bars (0-1)
    label : str, optional
        Label for legend
    showlegend : bool, default=True
        Whether to show this trace in the legend
    **kwargs
        Additional keyword arguments passed to go.Bar()
        
    Returns
    -------
    plotly.graph_objects.Bar
        The plotly bar trace
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is required for this function")
    
    return go.Bar(
        x=x,
        y=data if data.ndim == 1 else data[:, 0],
        marker=dict(color=color),
        opacity=alpha,
        name=label or '',
        showlegend=showlegend,
        **kwargs
    )


# ============================================================================
# Violin Plots
# ============================================================================

def render_violin_matplotlib(
    ax,
    data: npt.NDArray,
    color: Optional[str] = None,
    alpha: float = 0.7,
    showmeans: bool = True,
    showmedians: bool = True,
    showbox: bool = True,
    showpoints: bool = True,
    label: Optional[str] = None,
    **kwargs
):
    """
    Render a violin plot with optional box plot and points using matplotlib.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    data : ndarray
        1D or 2D array of values
    color : str, optional
        Violin color
    alpha : float, default=0.7
        Opacity (0-1)
    showmeans : bool, default=True
        Show mean line
    showmedians : bool, default=True
        Show median line
    showbox : bool, default=True
        Show box plot on the left side
    showpoints : bool, default=True
        Show individual data points
    label : str, optional
        Label for legend
    **kwargs
        Additional keyword arguments passed to violinplot()
        
    Returns
    -------
    dict
        Dictionary with violin plot components
    """
    # Ensure data is in list format for violinplot
    if data.ndim == 1:
        data_list = [data]
        positions = [1]
    else:
        data_list = [data[:, i] for i in range(data.shape[1])]
        positions = list(range(1, data.shape[1] + 1))
    
    result = {}
    
    # Add box plot on the left side if requested
    if showbox:
        # Shift box plot slightly to the left
        box_positions = [p - 0.2 for p in positions]
        bp = ax.boxplot(
            data_list,
            positions=box_positions,
            widths=0.15,
            patch_artist=True,
            showfliers=False,
            **kwargs
        )
        # Apply color to box
        if color:
            for patch in bp['boxes']:
                patch.set_facecolor(color)
                patch.set_alpha(alpha * 0.5)
        result['box'] = bp
    
    # Create violin plot
    parts = ax.violinplot(
        data_list,
        positions=positions,
        showmeans=showmeans,
        showmedians=showmedians,
        **kwargs
    )
    
    # Apply color to violin parts
    if color:
        for pc in parts['bodies']:
            pc.set_facecolor(color)
            pc.set_alpha(alpha)
    
    result['violin'] = parts
    
    # Add individual points if requested
    if showpoints:
        import numpy as np
        for i, (pos, d) in enumerate(zip(positions, data_list)):
            # Add jitter to x positions
            x_jitter = np.random.normal(pos + 0.15, 0.04, size=len(d))
            ax.scatter(x_jitter, d, alpha=alpha * 0.3, s=10, color=color or 'black')
    
    # Add legend entry
    if label:
        ax.plot([], [], color=color, label=label, linewidth=10)
    
    return result


def render_violin_plotly(
    data: npt.NDArray,
    color: Optional[str] = None,
    alpha: float = 0.7,
    meanline: Optional[dict] = None,
    showbox: bool = True,
    showpoints: bool = True,
    label: Optional[str] = None,
    showlegend: bool = True,
    **kwargs
) -> 'go.Violin':
    """
    Render a violin plot with optional box plot and points using plotly.
    
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
        Show individual data points
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
    
    # Handle meanline configuration
    if meanline is None:
        meanline = {'visible': True}
    elif isinstance(meanline, bool):
        meanline = {'visible': meanline}
    
    # Configure points display
    if showpoints:
        points = 'all'
        pointpos = 0.5  # Position points to the right
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
        points=points,
        pointpos=pointpos,
        jitter=jitter,
        **kwargs
    )


# ============================================================================
# Box Plots
# ============================================================================

def render_box_matplotlib(
    ax,
    data: npt.NDArray,
    color: Optional[str] = None,
    alpha: float = 0.7,
    label: Optional[str] = None,
    notch: bool = False,
    **kwargs
):
    """
    Render a box plot using matplotlib.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    data : ndarray
        1D or 2D array of values
    color : str, optional
        Box color
    alpha : float, default=0.7
        Opacity (0-1)
    label : str, optional
        Label for legend
    notch : bool, default=False
        Whether to show notches
    **kwargs
        Additional keyword arguments passed to ax.boxplot()
        
    Returns
    -------
    dict
        Dictionary with box plot components
    """
    # Ensure data is in list format
    if data.ndim == 1:
        data_list = [data]
    else:
        data_list = [data[:, i] for i in range(data.shape[1])]
    
    bp = ax.boxplot(
        data_list,
        labels=[label] if label and data.ndim == 1 else None,
        patch_artist=True,
        notch=notch,
        **kwargs
    )
    
    # Apply color
    if color:
        for patch in bp['boxes']:
            patch.set_facecolor(color)
            patch.set_alpha(alpha)
    
    return bp


def render_box_plotly(
    data: npt.NDArray,
    color: Optional[str] = None,
    alpha: float = 0.7,
    label: Optional[str] = None,
    showlegend: bool = True,
    notched: bool = False,
    **kwargs
) -> 'go.Box':
    """
    Render a box plot using plotly.
    
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
    **kwargs
        Additional keyword arguments passed to go.Box()
        
    Returns
    -------
    plotly.graph_objects.Box
        The plotly box trace
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is required for this function")
    
    return go.Box(
        y=data,
        name=label or '',
        marker=dict(color=color),
        opacity=alpha,
        showlegend=showlegend,
        notched=notched,
        **kwargs
    )
