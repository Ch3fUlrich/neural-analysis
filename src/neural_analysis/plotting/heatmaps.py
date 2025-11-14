"""
Heatmap visualization functions.

This module provides convenience functions for creating heatmaps using
the PlotGrid system. Supports both matplotlib and plotly backends.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt

from .grid_config import GridLayoutConfig, PlotGrid, PlotSpec

if TYPE_CHECKING:
    from .core import PlotConfig

__all__ = ["plot_heatmap"]


def plot_heatmap(
    data: npt.NDArray[np.floating[Any]],
    config: PlotConfig | None = None,
    cmap: str = "viridis",
    colorbar: bool = True,
    colorbar_label: str | None = None,
    alpha: float = 1.0,
    vmin: float | None = None,
    vmax: float | None = None,
    x_labels: list[str] | None = None,
    y_labels: list[str] | None = None,
    show_values: bool = False,
    value_format: str = ".2f",
    backend: Literal["matplotlib", "plotly"] | None = None,
) -> Any:
    """
    Create a heatmap visualization using PlotGrid.

    Parameters
    ----------
    data : ndarray
        2D array of values to visualize as heatmap.
    config : PlotConfig, optional
        Plot configuration.
    cmap : str
        Colormap name.
    colorbar : bool
        Whether to show colorbar.
    colorbar_label : str, optional
        Label for the colorbar.
    alpha : float
        Transparency.
    vmin, vmax : float, optional
        Minimum and maximum values for color scaling.
    x_labels : list[str], optional
        Labels for x-axis ticks.
    y_labels : list[str], optional
        Labels for y-axis ticks.
    show_values : bool
        Whether to annotate cells with their values.
    value_format : str
        Format string for value annotations (e.g., ".1f", ".2f").
    backend : {'matplotlib', 'plotly'}, optional
        Backend to use.

    Returns
    -------
    Figure object from the backend.
    """
    # Validate input dimensionality BEFORE conversion
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError(f"Data must be 2D, got shape {data.shape}")

    # Build kwargs (don't include alpha - it's a PlotSpec field)
    heatmap_kwargs = {
        "cmap": cmap,
        "colorbar": colorbar,
    }

    if vmin is not None:
        heatmap_kwargs["vmin"] = vmin
    if vmax is not None:
        heatmap_kwargs["vmax"] = vmax
    if x_labels is not None:
        heatmap_kwargs["x_labels"] = x_labels
    if y_labels is not None:
        heatmap_kwargs["y_labels"] = y_labels
    if show_values:
        heatmap_kwargs["show_values"] = show_values
        heatmap_kwargs["value_format"] = value_format

    # Create spec
    spec = PlotSpec(
        data=data,
        plot_type="heatmap",
        color=None,
        colorbar_label=colorbar_label,
        subplot_position=0,
        alpha=alpha,  # Store alpha in PlotSpec field, not kwargs
        kwargs=heatmap_kwargs,
    )

    # Create grid
    grid = PlotGrid(
        plot_specs=[spec],
        layout=GridLayoutConfig(rows=1, cols=1),
        backend=backend,
        config=config,
    )

    return grid.plot()
