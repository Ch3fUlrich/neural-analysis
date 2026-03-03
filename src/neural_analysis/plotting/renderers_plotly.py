"""Plotly rendering functions for the PlotGrid system."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import plotly.graph_objects as go
    from numpy.typing import NDArray


try:
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


# ============================================================================
# Scatter Plots
# ============================================================================


def render_scatter_plotly(
    data: NDArray[np.floating[Any]],
    color: str | None = None,
    colors: NDArray[np.floating[Any]] | None = None,
    cmap: str | None = None,
    marker: str = "circle",
    marker_size: float | None = None,
    sizes: NDArray[np.floating[Any]] | None = None,
    alpha: float = 0.7,
    label: str | None = None,
    showlegend: bool = True,
    colorbar: bool = False,
    colorbar_label: str | None = None,
    **kwargs: Any,
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
    marker_dict: dict[str, Any] = {
        "symbol": marker,
        "size": sizes if sizes is not None else (marker_size or 8),
        "opacity": alpha,
    }

    # Handle colors: array of values for colormap or single color
    if colors is not None and len(colors) > 0:
        marker_dict["color"] = colors
        if cmap:
            marker_dict["colorscale"] = cmap
        if colorbar:
            marker_dict["colorbar"] = {"title": colorbar_label or ""}
            marker_dict["showscale"] = True
    elif color is not None:
        marker_dict["color"] = color

    return go.Scatter(
        x=data[:, 0],
        y=data[:, 1],
        mode="markers",
        marker=marker_dict,
        name=label or "",
        showlegend=showlegend,
        **kwargs,
    )


def render_scatter3d_plotly(
    data: NDArray[np.floating[Any]],
    color: str | None = None,
    colors: NDArray[np.floating[Any]] | None = None,
    cmap: str | None = None,
    marker_size: float | None = None,
    sizes: NDArray[np.floating[Any]] | None = None,
    alpha: float = 0.7,
    label: str | None = None,
    showlegend: bool = True,
    colorbar: bool = False,
    colorbar_label: str | None = None,
    **kwargs: Any,
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
    marker_dict: dict[str, Any] = {
        "size": sizes if sizes is not None else (marker_size or 4),
        "opacity": alpha,
    }

    # Handle colors: array of values for colormap or single color
    if colors is not None and len(colors) > 0:
        marker_dict["color"] = colors
        if cmap:
            marker_dict["colorscale"] = cmap
        if colorbar:
            marker_dict["colorbar"] = {"title": colorbar_label or ""}
            marker_dict["showscale"] = True
    elif color is not None:
        marker_dict["color"] = color

    return go.Scatter3d(
        x=data[:, 0],
        y=data[:, 1],
        z=data[:, 2],
        mode="markers",
        marker=marker_dict,
        name=label or "",
        showlegend=showlegend,
        **kwargs,
    )


# ============================================================================
# Line Plots
# ============================================================================


def render_line_plotly(
    data: NDArray[np.floating[Any]],
    color: str | None = None,
    line_width: float = 2,
    linestyle: str | None = None,
    error_y: NDArray[np.floating[Any]] | None = None,
    alpha: float = 1.0,
    label: str | None = None,
    showlegend: bool = True,
    **kwargs: Any,
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
        "-": "solid",
        "--": "dash",
        "-.": "dashdot",
        ":": "dot",
    }
    dash_style = linestyle_map.get(linestyle, linestyle) if linestyle else None

    line_dict = {
        "color": color,
        "width": line_width,
    }
    if dash_style:
        line_dict["dash"] = dash_style

    # Prepare error_y dict if error values provided
    error_y_dict = None
    if error_y is not None:
        error_y = np.asarray(error_y)
        error_y_dict = dict(
            type="data",
            array=error_y,
            visible=True,
            color=color if color else "rgba(0,0,0,0.3)",
        )

    # Handle dictionary data format
    if isinstance(data, dict):
        if "x" in data and "y" in data:
            x = np.asarray(data["x"])
            y = np.asarray(data["y"])
            return go.Scatter(
                x=x,
                y=y,
                mode="lines",
                line=line_dict,
                error_y=error_y_dict,
                opacity=alpha,
                name=label or "",
                showlegend=showlegend,
                **kwargs,
            )
        else:
            raise ValueError("Dictionary data must contain 'x' and 'y' keys")

    if data.ndim == 1:
        # 1D data: use indices as x
        return go.Scatter(
            y=data,
            mode="lines",
            line=line_dict,
            error_y=error_y_dict,
            opacity=alpha,
            name=label or "",
            showlegend=showlegend,
            **kwargs,
        )
    elif data.shape[1] == 2:
        # 2D data: x and y columns
        return go.Scatter(
            x=data[:, 0],
            y=data[:, 1],
            mode="lines",
            line=line_dict,
            error_y=error_y_dict,
            opacity=alpha,
            name=label or "",
            showlegend=showlegend,
            **kwargs,
        )
    else:
        # Multiple y values: use first column
        return go.Scatter(
            y=data[:, 0],
            mode="lines",
            line=dict(color=color, width=line_width),
            error_y=error_y_dict,
            opacity=alpha,
            name=label or "",
            showlegend=showlegend,
            **kwargs,
        )


# ============================================================================
# Histogram Plots
# ============================================================================


def render_histogram_plotly(
    data: NDArray[np.floating[Any]],
    color: str | None = None,
    alpha: float = 0.7,
    bins: int = 30,
    label: str | None = None,
    showlegend: bool = True,
    **kwargs: Any,
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
        name=label or "",
        showlegend=showlegend,
        nbinsx=bins,
        **kwargs,
    )


# ============================================================================
# Heatmap Plots
# ============================================================================


def render_heatmap_plotly(
    data: NDArray[np.floating[Any]],
    cmap: str | None = None,
    colorscale: str | None = None,
    **kwargs: Any,
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
    colorbar_enabled = kwargs.pop("colorbar", True)
    colorbar_label = kwargs.pop("colorbar_label", None)
    kwargs.pop("x_labels", None)  # Remove - not directly supported
    kwargs.pop("y_labels", None)  # Remove - not directly supported
    kwargs.pop("show_values", None)  # Remove - needs implementation
    kwargs.pop("value_format", None)  # Remove - needs implementation
    kwargs.pop("alpha", None)  # Remove - plotly uses opacity

    # Use colorscale if provided, otherwise convert cmap
    scale = colorscale or cmap or "Viridis"

    # Build colorbar config
    colorbar_config = (
        dict(title=colorbar_label or "Value") if colorbar_enabled else None
    )

    return go.Heatmap(
        z=data, colorscale=scale, showlegend=False, colorbar=colorbar_config, **kwargs
    )


# ============================================================================
# Bar Plots
# ============================================================================


def render_bar_plotly(
    data: NDArray[np.floating[Any]],
    x: NDArray[np.floating[Any]] | None = None,
    color: str | None = None,
    colors: list[str] | None = None,
    alpha: float = 0.7,
    label: str | None = None,
    showlegend: bool = True,
    error_y: NDArray[np.floating[Any]] | None = None,
    error_x: NDArray[np.floating[Any]] | None = None,
    **kwargs: Any,
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
    marker_dict = {"color": bar_color, "opacity": alpha}

    # Build error bars
    error_y_dict = None
    error_x_dict = None
    if error_y is not None:
        error_y_dict = dict(type="data", array=error_y, visible=True)
    if error_x is not None:
        error_x_dict = dict(type="data", array=error_x, visible=True)

    return go.Bar(
        x=x,
        y=data if data.ndim == 1 else data[:, 0],
        marker=marker_dict,
        name=label or "",
        showlegend=showlegend,
        error_y=error_y_dict,
        error_x=error_x_dict,
        **kwargs,
    )


# ============================================================================
# Violin Plots
# ============================================================================


def render_violin_plotly(
    data: NDArray[np.floating[Any]],
    color: str | None = None,
    alpha: float = 0.7,
    meanline: dict[str, Any] | None = None,
    showbox: bool = True,
    showpoints: bool = True,
    label: str | None = None,
    showlegend: bool = True,
    **kwargs: Any,
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
        meanline = {"visible": True, "color": color or "black", "width": 2}
    elif isinstance(meanline, bool):
        meanline = {"visible": meanline, "color": color or "black", "width": 2}
    elif isinstance(meanline, dict):
        # Enhance existing meanline config
        if "visible" not in meanline:
            meanline["visible"] = True
        if "width" not in meanline:
            meanline["width"] = 2
        if "color" not in meanline:
            meanline["color"] = color or "black"

    # Configure points display on the LEFT side
    points: str | bool
    if showpoints:
        points = "all"
        pointpos = -0.8  # Position points to the left (negative = left side)
        jitter = 0.3
    else:
        points = False
        pointpos = 0
        jitter = 0

    return go.Violin(
        y=data,
        name=label or "",
        marker=dict(color=color),
        opacity=alpha,
        showlegend=showlegend,
        meanline=meanline,
        box_visible=showbox,
        box=dict(
            visible=showbox,
            fillcolor="rgba(255, 255, 255, 0.5)",  # Semi-transparent white so inner lines are visible
            line=dict(color=color or "black", width=2),  # Thicker outline
            width=0.3,  # Increased box width from default (~0.15)
        )
        if showbox
        else None,
        points=points,
        pointpos=pointpos,
        jitter=jitter,
        side="positive",  # Show only RIGHT half of violin
        **kwargs,
    )


# ============================================================================
# Box Plots
# ============================================================================


def render_box_plotly(
    data: NDArray[np.floating[Any]],
    color: str | None = None,
    alpha: float = 0.7,
    label: str | None = None,
    showlegend: bool = True,
    notched: bool = False,
    showpoints: bool = True,
    **kwargs: Any,
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
    boxpoints: str | bool
    if showpoints:
        boxpoints = "all"  # Show all points
        jitter = 0.3
        pointpos = 0  # Center the points over the box
    else:
        boxpoints = False
        jitter = 0
        pointpos = 0

    return go.Box(
        y=data,
        name=label or "",
        marker=dict(color=color),
        opacity=alpha,
        showlegend=showlegend,
        notched=notched,
        boxpoints=boxpoints,
        jitter=jitter,
        pointpos=pointpos,
        **kwargs,
    )


# ============================================================================
# Trajectory Plots
# ============================================================================


def render_trajectory_plotly(
    x: NDArray[np.floating[Any]],
    y: NDArray[np.floating[Any]],
    colors: NDArray[np.floating[Any]] | None = None,
    cmap: str = "Viridis",
    linewidth: float = 2.0,
    alpha: float = 1.0,
    show_points: bool = False,
    point_size: float = 5.0,
    colorbar: bool = True,
    colorbar_label: str | None = None,
    label: str | None = None,
    showlegend: bool = True,
    **kwargs: Any,
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
        colors_list = colors.tolist() if hasattr(colors, "tolist") else colors

        return go.Scatter(
            x=x,
            y=y,
            mode=mode,
            name=label or "",
            line=dict(width=linewidth),
            marker=dict(
                size=point_size,
                color=colors_list,  # Use marker color instead of line color
                colorscale=cmap,
                showscale=colorbar,
                colorbar=dict(title=colorbar_label)
                if colorbar and colorbar_label
                else None,
            ),
            opacity=alpha,
            showlegend=showlegend,
            **kwargs,
        )
    else:
        return go.Scatter(
            x=x,
            y=y,
            mode=mode,
            name=label or "",
            line=dict(width=linewidth),
            marker=marker_config,
            opacity=alpha,
            showlegend=showlegend,
            **kwargs,
        )


def render_trajectory3d_plotly(
    x: NDArray[np.floating[Any]],
    y: NDArray[np.floating[Any]],
    z: NDArray[np.floating[Any]],
    colors: NDArray[np.floating[Any]] | None = None,
    cmap: str = "Viridis",
    linewidth: float = 2.0,
    alpha: float = 1.0,
    show_points: bool = False,
    point_size: float = 3.0,
    colorbar: bool = True,
    colorbar_label: str | None = None,
    label: str | None = None,
    showlegend: bool = True,
    **kwargs: Any,
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
        colorbar_dict: dict[str, Any] = {}
        if colorbar_label:
            colorbar_dict["title"] = colorbar_label
        marker_config_dict: dict[str, Any] = {
            "color": colors,
            "colorscale": cmap,
            "showscale": colorbar,
            "colorbar": colorbar_dict,
        }
        marker_config.update(marker_config_dict)

    return go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode=mode,
        name=label or "",
        line=dict(width=linewidth),
        marker=marker_config,
        opacity=alpha,
        showlegend=showlegend,
        **kwargs,
    )


# ============================================================================
# KDE (Kernel Density Estimation) Plots
# ============================================================================


def render_kde_plotly(
    xi: NDArray[np.floating[Any]],
    yi: NDArray[np.floating[Any]],
    zi: NDArray[np.floating[Any]],
    fill: bool = True,
    n_levels: int = 10,
    cmap: str = "Viridis",
    alpha: float = 0.6,
    colorbar: bool = True,
    colorbar_label: str | None = None,
    label: str | None = None,
    showlegend: bool = True,
    **kwargs: Any,
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
        size=(float(np.max(zi)) - float(np.min(zi))) / n_levels,
    )

    return go.Contour(
        x=xi[0, :] if xi.ndim == 2 else xi,
        y=yi[:, 0] if yi.ndim == 2 else yi,
        z=zi,
        name=label or "",
        colorscale=cmap,
        opacity=alpha,
        showlegend=showlegend,
        showscale=colorbar,
        colorbar=dict(title=colorbar_label) if colorbar_label else {},
        contours=contour_config,
        **kwargs,
    )


# ============================================================================
# Convex Hull Plots
# ============================================================================


def render_convex_hull_plotly(
    hull_x: NDArray[np.floating[Any]],
    hull_y: NDArray[np.floating[Any]],
    color: str = "black",
    linewidth: float = 2.0,
    alpha: float = 1.0,
    fill: bool = False,
    fill_alpha: float = 0.2,
    label: str | None = None,
    showlegend: bool = True,
    **kwargs: Any,
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
        name=label or "",
        line=dict(color=color, width=linewidth),
        fill=fill_mode,
        fillcolor=fill_color,
        opacity=alpha if not fill else fill_alpha,
        showlegend=showlegend,
        **kwargs,
    )


# ============================================================================
# Boolean States Plots
# ============================================================================


def render_boolean_states_plotly(
    x: NDArray[np.floating[Any]],
    states: NDArray[np.floating[Any]],
    true_color: str = "#2ca02c",
    false_color: str = "#d62728",
    true_label: str = "True",
    false_label: str = "False",
    alpha: float = 0.3,
    **kwargs: Any,
) -> Any:
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
            x=[x[start], x[end - 1], x[end - 1], x[start], x[start]],
            y=[0, 0, 1, 1, 0],
            fill="toself",
            fillcolor=true_color,
            line=dict(width=0),
            opacity=alpha,
            name=true_label if i == 0 else "",
            showlegend=(i == 0),
            **kwargs,
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
        x_end = x[end - 1] if end < len(x) else x[-1]
        trace = go.Scatter(
            x=[x[start], x_end, x_end, x[start], x[start]],
            y=[0, 0, 1, 1, 0],
            fill="toself",
            fillcolor=false_color,
            line=dict(width=0),
            opacity=alpha,
            name=false_label if i == 0 else "",
            showlegend=(i == 0),
            **kwargs,
        )
        traces.append(trace)

    return traces


# ==============================================================================
# Ellipse Rendering
# ==============================================================================


def render_ellipse_plotly(
    centers: NDArray[np.float64],
    widths: NDArray[np.float64],
    heights: NDArray[np.float64],
    angles: NDArray[np.float64] | None = None,
    color: str = "red",
    alpha: float = 0.3,
    name: str | None = None,
    **kwargs: Any,
) -> list[Any]:
    """
    Render ellipses using plotly shapes.

    Parameters
    ----------
    centers : np.ndarray
        Center coordinates, shape (n_ellipses, n_dims)
    widths : np.ndarray
        Width of each ellipse
    heights : np.ndarray
        Height of each ellipse
    angles : np.ndarray, optional
        Rotation angles in degrees
    color : str
        Fill color
    alpha : float
        Transparency
    name : str, optional
        Trace name for legend
    **kwargs
        Additional arguments

    Returns
    -------
    list
        List of shape dictionaries for plotly
    """
    shapes = []
    n_dims = centers.shape[1] if centers.ndim > 1 else 1

    # Convert color to rgba with alpha
    import plotly.colors as pc

    rgba = f"rgba({pc.hex_to_rgb(color)[0]}, {pc.hex_to_rgb(color)[1]}, {pc.hex_to_rgb(color)[2]}, {alpha})"

    if n_dims == 1:
        # 1D: rectangles
        for i in range(len(centers)):
            center_x = centers[i, 0] if centers.ndim > 1 else centers[i]
            width = widths[i]
            height = heights[i] if heights is not None else 1.0

            shape = dict(
                type="rect",
                x0=center_x - width / 2,
                x1=center_x + width / 2,
                y0=-height / 2,
                y1=height / 2,
                fillcolor=rgba,
                line=dict(width=0),
                layer="below",
            )
            shapes.append(shape)

    elif n_dims == 2:
        # 2D: circles (plotly doesn't support rotated ellipses natively)
        # Approximate with path for rotated ellipses
        for i in range(len(centers)):
            center = centers[i]
            width = widths[i]
            height = heights[i]
            angle_deg = angles[i] if angles is not None else 0
            angle_rad = np.radians(angle_deg)

            # Generate ellipse points
            t = np.linspace(0, 2 * np.pi, 50)
            x = (width / 2) * np.cos(t)
            y = (height / 2) * np.sin(t)

            # Rotate
            x_rot = x * np.cos(angle_rad) - y * np.sin(angle_rad) + center[0]
            y_rot = x * np.sin(angle_rad) + y * np.cos(angle_rad) + center[1]

            # Create path
            path = f"M {x_rot[0]},{y_rot[0]} "
            for j in range(1, len(x_rot)):
                path += f"L {x_rot[j]},{y_rot[j]} "
            path += "Z"

            shape = dict(
                type="path",
                path=path,
                fillcolor=rgba,
                line=dict(width=0),
                layer="below",
            )
            shapes.append(shape)

    # Note: 3D ellipsoids in plotly would require mesh3d traces

    return shapes
