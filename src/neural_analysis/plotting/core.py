"""
Core utilities for the plotting package.

This module provides fundamental utilities used across all plotting functions:
- PlotConfig: Configuration dataclass for plot parameters
- Color utilities: Color generation, alpha calculation, RGBA conversion
- Save utilities: Plot saving with various formats
- Helper functions: Axis configuration, normalization, etc.

Example:
    >>> from neural_analysis.plotting.core import PlotConfig, calculate_alpha
    >>> config = PlotConfig(title="My Plot", figsize=(10, 5))
    >>> alpha_values = calculate_alpha([1, 2, 3, 4, 5])
"""

import colorsys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from sklearn.preprocessing import minmax_scale

from .backend import BackendType

__all__ = [
    "PlotConfig",
    "calculate_alpha",
    "generate_similar_colors",
    "create_rgba_labels",
    "save_plot",
    # Cross-backend helpers
    "resolve_colormap",
    "apply_layout_matplotlib",
    "apply_layout_plotly",
    "apply_layout_plotly_3d",
    "get_default_categorical_colors",
    "finalize_plot_matplotlib",
    "finalize_plot_plotly",
]


@dataclass
class PlotConfig:
    """
    Configuration for plot appearance and behavior.

    This dataclass consolidates all plot configuration parameters into a single
    object for easier management and passing between functions.

    Attributes
    ----------
    title : str, optional
        Plot title
    xlabel : str, optional
        X-axis label
    ylabel : str, optional
        Y-axis label
    zlabel : str, optional
        Z-axis label (for 3D plots)
    xlim : tuple of float, optional
        X-axis limits (min, max)
    ylim : tuple of float, optional
        Y-axis limits (min, max)
    zlim : tuple of float, optional
        Z-axis limits (min, max, for 3D plots)
    figsize : tuple of int, optional
        Figure size in inches (width, height)
    dpi : int, default=100
        Dots per inch for figure resolution
    grid : bool, default=False
        Whether to show grid
    legend : bool, default=True
        Whether to show legend
    tight_layout : bool, default=True
        Whether to use tight layout
    save_path : Path or str, optional
        Path to save the figure. If save_dir is provided, this is ignored.
    save_format : str, default='png'
        Format for saving ('png', 'pdf', 'svg', 'jpg', etc.)
    save_dir : Path or str, optional
        Directory where to save the figure. If provided, filename is generated
        from plot_type and additional_save_title.
    plot_type : str, optional
        Type of plot (e.g., 'heatmap', 'scatter', 'line'). Used in generated filename.
    additional_save_title : str, optional
        Additional text to include in generated filename.
    save_html : bool, default=True
        For plotly plots, whether to also save HTML version alongside other formats.
    show : bool, default=True
        Whether to display the plot
    cmap : str, default='viridis'
        Colormap name
    alpha : float, default=0.8
        Default alpha transparency

    Examples
    --------
    >>> config = PlotConfig(
    ...     title="Neural Activity",
    ...     xlabel="Time (s)",
    ...     ylabel="Activity (a.u.)",
    ...     figsize=(12, 6),
    ...     grid=True
    ... )
    >>> # Use config in plotting function
    >>> plot_line(data, config=config)

    >>> # Save with generated filename
    >>> config = PlotConfig(
    ...     save_dir="output/figures",
    ...     plot_type="scatter",
    ...     additional_save_title="neuron_activity",
    ...     save_format="png"
    ... )
    >>> # Will save as: output/figures/scatter_neuron_activity.png
    """

    title: str | None = None
    xlabel: str | None = None
    ylabel: str | None = None
    zlabel: str | None = None
    xlim: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None
    zlim: tuple[float, float] | None = None
    figsize: tuple[int, int] = (10, 6)
    dpi: int = 100
    grid: bool = False
    legend: bool = True
    tight_layout: bool = True
    save_path: str | Path | None = None
    save_format: str = "png"
    save_dir: str | Path | None = None
    plot_type: str | None = None
    additional_save_title: str | None = None
    save_html: bool = True
    show: bool = True
    cmap: str = "viridis"
    alpha: float = 0.8

    def __post_init__(self):
        """Validate and convert parameters after initialization."""
        if self.save_path is not None:
            self.save_path = Path(self.save_path)
        if self.save_dir is not None:
            self.save_dir = Path(self.save_dir)

    def get_save_path(self) -> Path | None:
        """
        Get the full save path for the plot.

        If save_path is set, returns it directly.
        If save_dir is set, generates filename from plot_type and additional_save_title.
        Otherwise returns None.

        Returns
        -------
        Path or None
            Full path where plot should be saved, or None if no save requested.
        """
        if self.save_path is not None:
            return self.save_path

        if self.save_dir is not None:
            # Generate filename
            parts = []
            if self.plot_type:
                parts.append(self.plot_type)
            if self.additional_save_title:
                parts.append(self.additional_save_title)

            if not parts:
                # Default filename if nothing specified
                parts = ["plot"]

            filename = "_".join(parts) + f".{self.save_format}"
            return self.save_dir / filename

        return None


# ---------------------------------------------
# Cross-backend colormap and layout utilities
# ---------------------------------------------

# Common colormap aliases so a single name works on both backends
_PLOTLY_CMAP_ALIASES: dict[str, str] = {
    # Perceptually uniform
    "viridis": "Viridis",
    "plasma": "Plasma",
    "inferno": "Inferno",
    "magma": "Magma",
    "cividis": "Cividis",
    "turbo": "Turbo",
    # Sequential
    "blues": "Blues",
    "greens": "Greens",
    "reds": "Reds",
    "greys": "Greys",
    "purples": "Purples",
    "oranges": "Oranges",
    # Diverging
    "rdbu": "RdBu",
    "rdylbu": "RdYlBu",
    "rdylgn": "RdYlGn",
    "puor": "PuOr",
    "purd": "PuRd",
    "ylorbr": "YlOrBr",
    "ylorrd": "YlOrRd",
    "ylgn": "YlGn",
    "ylgnbu": "YlGnBu",
}


def resolve_colormap(cmap: str | None, backend: BackendType) -> str:
    """Return a backend-appropriate colormap identifier.

    For matplotlib this returns a valid colormap object (or name accepted by
    matplotlib), for Plotly this returns a colorscale name recognized by Plotly.

    If the provided name is unknown, falls back to Viridis.
    """
    name = (cmap or "viridis").lower()
    if backend == BackendType.MATPLOTLIB:
        # Try the name as-is first, then try capitalized version for sequential cmaps
        for attempt_name in [name, name.capitalize()]:
            try:
                # Use modern colormaps API (Matplotlib 3.7+)
                return plt.colormaps[attempt_name]
            except (KeyError, AttributeError):
                continue

        # Fallback: use viridis if colormap not found
        warnings.warn(
            f"Unknown colormap '{cmap}', using 'viridis' instead.", stacklevel=2
        )
        try:
            return plt.colormaps["viridis"]
        except AttributeError:
            # Very old matplotlib versions
            return plt.colormaps.get_cmap("viridis")
    else:
        # Plotly accepts canonical colorscale names (case-sensitive). Normalize
        # common matplotlib names to their Plotly equivalents; if not found, use
        # the provided name as-is.
        return _PLOTLY_CMAP_ALIASES.get(name, name)


def apply_layout_matplotlib(ax, config: PlotConfig) -> None:
    """Apply common layout (title, labels, limits, grid) for matplotlib."""
    if config.title:
        ax.set_title(config.title)
    if config.xlabel:
        ax.set_xlabel(config.xlabel)
    if config.ylabel:
        ax.set_ylabel(config.ylabel)
    if config.zlabel and hasattr(ax, "set_zlabel"):
        ax.set_zlabel(config.zlabel)
    if config.xlim:
        ax.set_xlim(config.xlim)
    if config.ylim:
        ax.set_ylim(config.ylim)
    if config.zlim and hasattr(ax, "set_zlim"):
        ax.set_zlim(config.zlim)
    if config.grid:
        ax.grid(True, alpha=0.3)
    if config.tight_layout:
        plt.tight_layout()


def apply_layout_plotly(fig, config: PlotConfig) -> None:
    """Apply common layout (title, labels, limits, grid) for Plotly."""
    layout_updates: dict[str, Any] = {}
    if config.title:
        layout_updates["title"] = config.title
    if config.xlabel:
        layout_updates["xaxis_title"] = config.xlabel
    if config.ylabel:
        layout_updates["yaxis_title"] = config.ylabel
    xaxis = {}
    yaxis = {}
    if config.xlim:
        xaxis["range"] = list(config.xlim)
    if config.ylim:
        yaxis["range"] = list(config.ylim)
    xaxis["showgrid"] = bool(config.grid)
    yaxis["showgrid"] = bool(config.grid)
    if xaxis:
        layout_updates["xaxis"] = xaxis
    if yaxis:
        layout_updates["yaxis"] = yaxis
    if config.figsize:
        layout_updates["width"] = int(config.figsize[0] * 100)
        layout_updates["height"] = int(config.figsize[1] * 100)
    fig.update_layout(**layout_updates)


def apply_layout_plotly_3d(fig, config: PlotConfig) -> None:
    """Apply common layout for Plotly 3D plots (scene configuration)."""
    layout_updates: dict[str, Any] = {}
    if config.title:
        layout_updates["title"] = config.title

    scene_dict = {}
    if config.xlabel:
        scene_dict["xaxis_title"] = config.xlabel
    if config.ylabel:
        scene_dict["yaxis_title"] = config.ylabel
    if config.zlabel:
        scene_dict["zaxis_title"] = config.zlabel
    if config.xlim:
        scene_dict["xaxis"] = scene_dict.get("xaxis", {})
        scene_dict["xaxis"]["range"] = list(config.xlim)
    if config.ylim:
        scene_dict["yaxis"] = scene_dict.get("yaxis", {})
        scene_dict["yaxis"]["range"] = list(config.ylim)
    if config.zlim:
        scene_dict["zaxis"] = scene_dict.get("zaxis", {})
        scene_dict["zaxis"]["range"] = list(config.zlim)

    if scene_dict:
        layout_updates["scene"] = scene_dict

    if config.figsize:
        layout_updates["width"] = int(config.figsize[0] * 100)
        layout_updates["height"] = int(config.figsize[1] * 100)

    fig.update_layout(**layout_updates)


def get_default_categorical_colors(n: int) -> list[str]:
    """Return a list of n default categorical colors as hex strings.

    Uses matplotlib's 'tab10' palette to ensure consistent colors across
    backends (matplotlib and plotly).
    """
    try:
        palette = plt.colormaps["tab10"]
    except AttributeError:
        palette = plt.get_cmap("tab10")

    colors: list[str] = []
    for i in range(n):
        r, g, b, _ = palette(i % palette.N)
        colors.append(f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}")
    return colors


def finalize_plot_matplotlib(config: PlotConfig) -> None:
    """Handle save and show for matplotlib plots.

    This consolidates the common pattern of saving and showing matplotlib
    plots based on config settings.
    """
    save_path = config.get_save_path()
    if save_path:
        save_plot(save_path, format=config.save_format, dpi=config.dpi)
    if config.show:
        plt.show()


def finalize_plot_plotly(fig, config: PlotConfig) -> None:
    """Handle save and show for plotly plots.

    This consolidates the common pattern of saving and showing plotly
    plots based on config settings. If save_html is True, also saves
    an HTML version alongside the specified format.
    """
    save_path = config.get_save_path()
    if save_path:
        # Save in the requested format
        if config.save_format == "html":
            fig.write_html(str(save_path))
        else:
            # Save in requested format (png, jpg, pdf)
            try:
                fig.write_image(str(save_path), format=config.save_format)
            except ValueError as e:
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(
                    f"Could not save plotly figure as {config.save_format}: {e}. "
                    f"You may need to install kaleido: pip install kaleido"
                )

            # Also save HTML if requested
            if config.save_html:
                html_path = save_path.with_suffix(".html")
                fig.write_html(str(html_path))

    if config.show:
        from IPython.display import HTML, display

        try:
            display(HTML(fig.to_html()))
        except Exception:
            # Fall back to standard show if not in Jupyter
            fig.show()


def calculate_alpha(
    value: int | float | list[int | float] | npt.NDArray[np.floating[Any]],
    min_value: int | float | None = None,
    max_value: int | float | None = None,
    min_alpha: float = 0.3,
    max_alpha: float = 1.0,
) -> float | list[float]:
    """
    Calculate alpha value(s) based on value's position in range.

    Maps input values to alpha (transparency) values linearly based on their
    position in the min-max range. Useful for encoding additional information
    in scatter plots or other visualizations.

    Parameters
    ----------
    value : float, int, list, or array
        The value(s) for which to calculate alpha. Can be scalar or array-like.
    min_value : float or int, optional
        Minimum value of the range. If None and value is array-like, uses min of array.
    max_value : float or int, optional
        Maximum value of the range. If None and value is array-like, uses max of array.
    min_alpha : float, default=0.3
        Minimum alpha value (fully transparent = 0, fully opaque = 1).
    max_alpha : float, default=1.0
        Maximum alpha value.

    Returns
    -------
    float or list of float
        Calculated alpha value(s) constrained between min_alpha and max_alpha.
        Returns float if input was scalar, list if input was array-like.

    Raises
    ------
    ValueError
        If no values provided or if min_alpha > max_alpha.

    Examples
    --------
    >>> # Single value with explicit range
    >>> calculate_alpha(5, min_value=0, max_value=10)
    0.65

    >>> # Array of values with auto range
    >>> calculate_alpha([1, 2, 3, 4, 5])
    [0.3, 0.475, 0.65, 0.825, 1.0]

    >>> # Custom alpha range
    >>> calculate_alpha([1, 5, 10], min_alpha=0.5, max_alpha=0.9)
    [0.5, 0.7, 0.9]
    """
    # Validate alpha range
    if min_alpha > max_alpha:
        raise ValueError(f"min_alpha ({min_alpha}) must be <= max_alpha ({max_alpha})")

    # Convert to array for uniform processing
    is_scalar = isinstance(value, (int, float))
    values = np.atleast_1d(value)

    if len(values) == 0:
        raise ValueError("No values provided for alpha calculation.")

    # Single value with no range specified returns max alpha
    if len(values) == 1 and (min_value is None or max_value is None):
        return max_alpha if is_scalar else [max_alpha]

    # Determine range
    min_val = min_value if min_value is not None else np.min(values)
    max_val = max_value if max_value is not None else np.max(values)

    # Handle edge case where range is zero
    if max_val == min_val:
        result = [min_alpha] * len(values)
        return result[0] if is_scalar else result

    # Calculate alphas
    normalized = (values - min_val) / (max_val - min_val)
    alphas = min_alpha + (max_alpha - min_alpha) * normalized

    # Ensure alpha stays within specified bounds
    alphas = np.clip(alphas, min_alpha, max_alpha)

    # Return single value or list based on input type
    return float(alphas[0]) if is_scalar else alphas.tolist()


def generate_similar_colors(
    base_color: tuple[float, float, float],
    num_colors: int,
    hue_variation: float = 0.05,
    lightness_variation: float = 0.1,
) -> list[tuple[float, float, float]]:
    """
    Generate a list of similar colors based on a base color.

    Creates color variations by adjusting hue and lightness in the HLS color space.
    Useful for creating cohesive color schemes for grouped data.

    Parameters
    ----------
    base_color : tuple of float
        RGB color tuple (r, g, b) with values in [0, 1].
    num_colors : int
        Number of colors to generate.
    hue_variation : float, default=0.05
        Amount of hue variation per color step.
    lightness_variation : float, default=0.1
        Amount of lightness variation per color step.

    Returns
    -------
    list of tuple
        List of RGB color tuples.

    Examples
    --------
    >>> base = (0.2, 0.4, 0.8)  # Blue
    >>> colors = generate_similar_colors(base, 5)
    >>> len(colors)
    5

    >>> # Use in scatter plot
    >>> for i, color in enumerate(colors):
    ...     plt.scatter(x[i], y[i], color=color)
    """
    base_hls = colorsys.rgb_to_hls(*base_color[:3])
    colors = []

    for i in range(num_colors):
        # Small hue variations
        hue = (base_hls[0] + i * hue_variation) % 1.0

        # Slight lightness variations, clamped to [0, 1]
        lightness = np.clip(
            base_hls[1] + (i - num_colors / 2) * lightness_variation, 0, 1
        )

        # Convert back to RGB
        rgb = colorsys.hls_to_rgb(hue, lightness, base_hls[2])
        colors.append(rgb)

    return colors


def create_rgba_labels(
    values: npt.NDArray[np.floating[Any]],
    alpha: float = 0.8,
    cmap: str = "rainbow",
) -> npt.NDArray[np.floating[Any]]:
    """
    Create RGBA labels using a colormap.

    Maps input values to RGBA colors using a specified colormap. Supports both
    1D and 2D input values. For 2D values, uses a custom 2D colormap.

    Parameters
    ----------
    values : ndarray
        The input values to be mapped to colors. Can be 1D or 2D.
        For 1D: shape (n,) - each value mapped to a color.
        For 2D: shape (n, 2) - each pair mapped to a 2D color (red-blue gradient).
    alpha : float, default=0.8
        The alpha level for the colors (0=transparent, 1=opaque).
    cmap : str, default='rainbow'
        Colormap name for 1D values (ignored for 2D).

    Returns
    -------
    ndarray
        Array of RGBA colors with shape (n, 4).

    Raises
    ------
    ValueError
        If values dimension is not 1 or 2.

    Examples
    --------
    >>> # 1D values
    >>> values_1d = np.array([1, 2, 3, 4, 5])
    >>> colors = create_rgba_labels(values_1d)
    >>> colors.shape
    (5, 4)

    >>> # 2D values (e.g., x-y coordinates)
    >>> values_2d = np.array([[0.1, 0.2], [0.5, 0.5], [0.9, 0.8]])
    >>> colors = create_rgba_labels(values_2d)
    >>> colors.shape
    (3, 4)
    """
    values = np.atleast_1d(values)
    # Use sklearn's optimized min-max scaling
    normalized_values = minmax_scale(values, axis=0)

    if values.ndim == 1:
        # Use matplotlib colormap for 1D
        cmap_obj = plt.colormaps.get_cmap(cmap)
        rgba_colors = np.array([cmap_obj(x) for x in normalized_values])
        rgba_colors[:, 3] = alpha  # Set alpha channel

    elif values.ndim == 2 and values.shape[1] == 2:
        # Custom 2D colormap: first dimension -> red, second -> blue
        rgba_colors = np.zeros((len(values), 4))
        rgba_colors[:, 0] = normalized_values[:, 0]  # Red channel
        rgba_colors[:, 1] = 0.5  # Green channel (constant)
        rgba_colors[:, 2] = normalized_values[:, 1]  # Blue channel
        rgba_colors[:, 3] = alpha  # Alpha channel

    else:
        raise ValueError(
            f"Invalid values dimension: {values.shape}. "
            "Expected 1D array or 2D array with shape (n, 2)."
        )

    return rgba_colors


def save_plot(
    save_path: str | Path,
    format: str = "png",
    dpi: int = 300,
    bbox_inches: str = "tight", **kwargs) -> None:
    """
    Save current matplotlib figure to file.

    Parameters
    ----------
    save_path : str or Path
        Path where to save the figure. Parent directories will be created if needed.
    format : str, default='png'
        File format ('png', 'pdf', 'svg', 'jpg', etc.).
    dpi : int, default=300
        Resolution in dots per inch.
    bbox_inches : str, default='tight'
        Bounding box specification. 'tight' removes extra whitespace.
    **kwargs
        Additional arguments passed to plt.savefig().

    Examples
    --------
    >>> plt.plot([1, 2, 3], [1, 4, 9])
    >>> save_plot('output/my_plot.png', dpi=300)

    >>> # Save as PDF
    >>> save_plot('output/my_plot.pdf', format='pdf')
    """
    save_path = Path(save_path)

    # Create directory if it doesn't exist
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Add extension if not present
    if save_path.suffix == "":
        save_path = save_path.with_suffix(f".{format}")

    # Suppress tight_layout warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
        plt.savefig(
            save_path, format=format, dpi=dpi, bbox_inches=bbox_inches, **kwargs)


def make_list_if_not(value: Any | list[Any]) -> list[Any]:
    """
    Convert value to list if it isn't already.

    Parameters
    ----------
    value : any
        Value to convert.

    Returns
    -------
    list
        Input wrapped in list if not already a list.

    Examples
    --------
    >>> make_list_if_not(5)
    [5]
    >>> make_list_if_not([1, 2, 3])
    [1, 2, 3]
    """
    return value if isinstance(value, list) else [value]
