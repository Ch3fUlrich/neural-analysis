"""
Heatmap visualization functions.

This module provides convenience functions for creating heatmaps using
the PlotGrid system. Supports both matplotlib and plotly backends.
"""

from typing import Literal, Any
import numpy as np
import numpy.typing as npt

from .grid_config import PlotGrid, PlotSpec, GridLayoutConfig
from .core import PlotConfig

__all__ = ["plot_heatmap"]


def plot_heatmap(
    data: npt.NDArray,
    config: PlotConfig | None = None,
    cmap: str = 'viridis',
    colorbar: bool = True,
    alpha: float = 1.0,
    vmin: float | None = None,
    vmax: float | None = None,
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
    alpha : float
        Transparency.
    vmin, vmax : float, optional
        Minimum and maximum values for color scaling.
    backend : {'matplotlib', 'plotly'}, optional
        Backend to use.
        
    Returns
    -------
    Figure object from the backend.
    """
    # Validate input
    data = np.atleast_2d(data)
    if data.ndim != 2:
        raise ValueError(f"Data must be 2D, got shape {data.shape}")
    
    # Build kwargs
    heatmap_kwargs = {
        'cmap': cmap,
        'colorbar': colorbar,
        'alpha': alpha,
    }
    
    if vmin is not None:
        heatmap_kwargs['vmin'] = vmin
    if vmax is not None:
        heatmap_kwargs['vmax'] = vmax
    
    # Create spec
    spec = PlotSpec(
        data=data,
        plot_type='heatmap',
        color=None,
        subplot_position=0,
        **heatmap_kwargs
    )
    
    # Create grid
    grid = PlotGrid(
        plot_specs=[spec],
        layout=GridLayoutConfig(rows=1, cols=1),
        backend=backend,
        config=config,
    )
    
    return grid.plot()
