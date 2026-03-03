"""2D visualization functions for synthetic neural data.

Provides 2D-specific plotting: spatial binning, coverage heatmaps,
radial power spectra, example cell heatmaps, and random cell diagnostics.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from neural_analysis.metrics.pairwise_metrics import spatial_autocorrelation
from neural_analysis.plotting.grid_config import PlotSpec
from neural_analysis.plotting.synthetic_plots_1d import _compute_spatial_bins_1d
from neural_analysis.plotting.synthetic_plots_3d import CELL_TYPE_COLORS


def _compute_spatial_bins_2d(
    positions: npt.NDArray[np.float64],
    activity: npt.NDArray[np.float64],
    arena_size: tuple[float, ...],
    n_bins: int = 30,
    cell_idx: int | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], Any]:
    """Compute 2D spatial binning of activity.

    Args:
        positions: Position array, shape (n_samples, 2).
        activity: Activity array, shape (n_samples, n_cells).
        arena_size: Arena size (width, height).
        n_bins: Number of spatial bins per dimension.
        cell_idx: If specified, only bin this cell. Otherwise average all cells.

    Returns:
        x_bins: X bin edges.
        y_bins: Y bin edges.
        firing_map: 2D firing rate map, shape (n_bins, n_bins).
    """
    # Get arena dimensions
    if isinstance(arena_size, tuple):
        x_max, y_max = arena_size
    else:
        x_max = y_max = arena_size

    # Create spatial bins
    x_bins = np.linspace(0, x_max, n_bins + 1)
    y_bins = np.linspace(0, y_max, n_bins + 1)

    # Initialize firing map with 0 for unvisited bins (not NaN)
    firing_map = np.zeros((n_bins, n_bins), dtype=np.float64)

    # Compute average activity per spatial bin
    for i in range(n_bins):
        for j in range(n_bins):
            mask = (
                (positions[:, 0] >= x_bins[i])
                & (positions[:, 0] < x_bins[i + 1])
                & (positions[:, 1] >= y_bins[j])
                & (positions[:, 1] < y_bins[j + 1])
            )
            if mask.sum() > 0:
                if cell_idx is not None:
                    firing_map[j, i] = activity[mask, cell_idx].mean()
                else:
                    firing_map[j, i] = activity[mask, :].mean()

    return x_bins.astype(np.float64), y_bins.astype(np.float64), firing_map


def _compute_radial_power_spectrum(
    power_spectrum_2d: npt.NDArray[np.float64],
    n_bins: int,
) -> npt.NDArray[np.float64]:
    """Compute radial average of 2D power spectrum.

    Args:
        power_spectrum_2d: 2D power spectrum (already shifted)
        n_bins: Number of radial bins

    Returns:
        radial_profile: 1D radial power profile
    """
    center = np.array(power_spectrum_2d.shape) // 2
    y_coords, x_coords = np.ogrid[
        : power_spectrum_2d.shape[0], : power_spectrum_2d.shape[1]
    ]
    r = np.sqrt((x_coords - center[1]) ** 2 + (y_coords - center[0]) ** 2)
    r = r.astype(int)

    # Compute radial profile
    radial_profile = np.zeros(n_bins)
    for i in range(n_bins):
        mask = r == i
        if mask.sum() > 0:
            radial_profile[i] = power_spectrum_2d[mask].mean()

    return radial_profile


def _create_coverage_heatmap(
    activity: npt.NDArray[np.float64],
    metadata: dict[str, Any],
    subplot_position: int,
) -> PlotSpec | None:
    """Create spatial coverage heatmap for place/grid cells (unified function).

    For grid cells: uses 2D autocorrelation to reveal periodic structure.
    For place cells: uses spatial binning of actual trajectory data.
    """
    positions = metadata.get("positions")
    arena_size = metadata.get("arena_size", (1.0, 1.0))
    cell_type = metadata.get("cell_type", "place")

    if positions is None or positions.shape[1] != 2:
        return None

    # Get arena dimensions
    if isinstance(arena_size, tuple):
        x_max, y_max = arena_size
    else:
        x_max = y_max = arena_size

    # Different strategies for grid vs place cells
    n_bins = 60 if cell_type == "grid" else 30

    # Choose colormap based on cell type
    cmap = "hot" if cell_type == "place" else "viridis"

    if cell_type == "grid":
        # For grid cells: compute autocorrelation using
        # spatial_autocorrelation orchestrator. This gives a cleaner
        # hexagonal pattern than a single noisy cell
        autocorr_normalized, lag_axes = spatial_autocorrelation(
            activity=activity,
            positions=positions,
            arena_size=arena_size,
            n_bins=n_bins,
            n_cells_to_average=10,
        )
        x_lags, y_lags = lag_axes

        spec = PlotSpec(
            data=autocorr_normalized,
            plot_type="heatmap",
            subplot_position=subplot_position,
            title="Grid Field Autocorrelation",
            cmap=cmap,
            colorbar=True,
            colorbar_label="Normalized Autocorr",
            kwargs={
                "x_label": "X Lag (m)",
                "y_label": "Y Lag (m)",
                "aspect": "auto",
                "extent": [x_lags[0], x_lags[-1], y_lags[0], y_lags[-1]],
                "origin": "lower",
            },
        )
    else:
        # For place cells: use spatial binning (average across all cells)
        _, _, firing_map = _compute_spatial_bins_2d(
            positions, activity, arena_size, n_bins=n_bins, cell_idx=None
        )

        spec = PlotSpec(
            data=firing_map,
            plot_type="heatmap",
            subplot_position=subplot_position,
            title="Place Field Coverage",
            cmap=cmap,
            colorbar=True,
            colorbar_label="Avg. Firing Rate (Hz)",
            kwargs={
                "x_label": "X Position (m)",
                "y_label": "Y Position (m)",
                "aspect": "auto",
                "extent": [0, x_max, 0, y_max],
                "origin": "lower",
            },
        )

    return spec


def _create_example_cell_heatmaps(
    activity: npt.NDArray[np.float64],
    metadata: dict[str, Any],
    subplot_position: int,
) -> list[PlotSpec]:
    """Create example cell heatmaps for 1D/2D place/grid cells (3 examples).

    Unified function that handles both 1D (line plots) and 2D (heatmaps).
    """
    specs: list[PlotSpec] = []
    positions = metadata.get("positions")
    arena_size = metadata.get("arena_size", 1.0)
    cell_type = metadata.get("cell_type", "place")
    n_dims = metadata.get("n_dims", 2)

    if positions is None:
        return specs

    # Determine dimensionality
    if n_dims == 1:
        # 1D case: line plots
        if positions.ndim != 2 or positions.shape[1] != 1:
            return specs

        # Get arena length
        if isinstance(arena_size, tuple):
            arena_size = arena_size[0]

        # Choose color based on cell type
        color = CELL_TYPE_COLORS.get(cell_type, "#E74C3C")

        # Create 3 example cell plots
        for cell_idx in range(min(3, activity.shape[1])):
            x_centers, firing_rates = _compute_spatial_bins_1d(
                positions, activity, arena_size, n_bins=50, cell_idx=cell_idx
            )

            spec = PlotSpec(
                data={"x": x_centers, "y": firing_rates},
                plot_type="line",
                subplot_position=subplot_position + cell_idx,
                title=f"{cell_type.title()} Cell {cell_idx} Heatmap",
                color=color,
                line_width=2,
                alpha=0.8,
                kwargs={
                    "x_label": "Position (m)",
                    "y_label": "Firing Rate (Hz)",
                },
            )
            specs.append(spec)

    elif n_dims == 2:
        # 2D case: heatmaps
        if positions.shape[1] != 2:
            return specs

        # Choose colormap based on cell type
        cmap = "hot" if cell_type == "place" else "viridis"

        # Get arena dimensions
        if isinstance(arena_size, tuple):
            x_max, y_max = arena_size
        else:
            x_max = y_max = arena_size

        # Create 3 example cell heatmaps
        for cell_idx in range(min(3, activity.shape[1])):
            x_bins, y_bins, rate_map = _compute_spatial_bins_2d(
                positions, activity, arena_size, n_bins=30, cell_idx=cell_idx
            )

            spec = PlotSpec(
                data=rate_map.T,  # Transpose for correct orientation
                plot_type="heatmap",
                subplot_position=subplot_position + cell_idx,
                title=f"{cell_type.title()} Cell {cell_idx} Heatmap",
                cmap=cmap,
                colorbar=True,
                colorbar_label="Rate (Hz)",
                kwargs={
                    "x_label": "X Position (m)",
                    "y_label": "Y Position (m)",
                    "aspect": "auto",
                    "extent": [0, x_max, 0, y_max],
                    "origin": "lower",
                },
            )
            specs.append(spec)

    return specs


def _create_random_diagnostics(
    activity: npt.NDArray[np.float64],
    metadata: dict[str, Any],
    subplot_position: int,
) -> list[PlotSpec]:
    """Create diagnostic plots to verify random cells lack spatial structure.

    Creates three diagnostic tests:
    1. Single cell "spatial map" (should be uniform/noisy)
    2. Population coverage (should be uniform, no hotspots)
    3. Grid field autocorrelation (should show no periodicity, unlike grid cells)

    This validates that random cells truly lack spatial structure like place/grid cells.

    **Code Reuse:** This function uses `spatial_autocorrelation` orchestrator
    from `neural_analysis.metrics.pairwise_metrics`, which automatically handles
    dimensionality. Same algorithm as grid cells, different interpretation (grid
    cells show hexagonal patterns; random cells show no structure).

    Args:
        activity: Neural activity matrix (n_samples, n_cells).
        metadata: Dataset metadata.
        subplot_position: Starting subplot position.

    Returns:
        List of PlotSpec objects for diagnostic plots (3 plots).
    """
    specs: list[PlotSpec] = []

    # Use uniform positions from metadata (should already be uniform random)
    # These positions test whether random cells incorrectly show spatial structure
    positions = metadata.get("positions")
    arena_size = metadata.get("arena_size", (1.0, 1.0))

    if positions is None:
        raise ValueError("Random cell diagnostics require 'positions' in metadata")

    # 1. Single cell spatial map (should show no structure)
    # Pick the cell with highest variance for best visualization
    cell_variances = activity.var(axis=0)
    test_cell = np.argmax(cell_variances)

    x_bins, y_bins, firing_map = _compute_spatial_bins_2d(
        positions, activity, arena_size, n_bins=20, cell_idx=int(test_cell)
    )

    spec_single = PlotSpec(
        data=firing_map,
        plot_type="heatmap",
        subplot_position=subplot_position,
        title=f"Random Cell {test_cell} 'Spatial Map'",
        cmap="hot",
        colorbar=True,
        colorbar_label="Rate (Hz)",
        kwargs={
            "x_label": "Synthetic X Position",
            "y_label": "Synthetic Y Position",
            "aspect": "auto",
            "extent": [0, arena_size[0], 0, arena_size[1]],
            "origin": "lower",
        },
    )
    specs.append(spec_single)

    # 2. Population coverage (average across all cells - should be uniform)
    _, _, coverage_map = _compute_spatial_bins_2d(
        positions, activity, arena_size, n_bins=20, cell_idx=None
    )

    spec_coverage = PlotSpec(
        data=coverage_map,
        plot_type="heatmap",
        subplot_position=subplot_position + 1,
        title="Population 'Coverage' (Should be Uniform)",
        cmap="viridis",
        colorbar=True,
        colorbar_label="Avg. Rate (Hz)",
        kwargs={
            "x_label": "Synthetic X Position",
            "y_label": "Synthetic Y Position",
            "aspect": "auto",
            "extent": [0, arena_size[0], 0, arena_size[1]],
            "origin": "lower",
        },
    )
    specs.append(spec_coverage)

    # 3. Grid field autocorrelation (should show no periodicity)
    # Use spatial_autocorrelation orchestrator (handles dimensionality automatically)
    autocorr_normalized, lag_axes = spatial_autocorrelation(
        activity=activity,
        positions=positions,
        arena_size=arena_size,
        n_bins=40,  # Moderate resolution
        n_cells_to_average=10,
    )
    x_lags, y_lags = lag_axes

    spec_autocorr = PlotSpec(
        data=autocorr_normalized,
        plot_type="heatmap",
        subplot_position=subplot_position + 1,
        title="Autocorrelation (Should Show No Periodicity)",
        cmap="viridis",
        colorbar=True,
        colorbar_label="Normalized Autocorr",
        kwargs={
            "x_label": "X Lag (m)",
            "y_label": "Y Lag (m)",
            "aspect": "auto",
            "extent": [x_lags[0], x_lags[-1], y_lags[0], y_lags[-1]],
            "origin": "lower",
        },
    )
    specs.append(spec_autocorr)

    return specs
