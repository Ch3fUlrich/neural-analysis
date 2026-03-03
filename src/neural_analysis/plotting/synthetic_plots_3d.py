"""3D visualization functions and shared helpers for synthetic neural data.

Provides 3D-specific plotting: spatial binning, coverage heatmap walls.
Shared helpers: cell type colors, cell color mapping, grid layout utilities,
raster plot creation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from neural_analysis.plotting.grid_config import PlotSpec

CELL_TYPE_COLORS = {
    "place": "#E74C3C",
    "grid": "#3498DB",
    "head_direction": "#2ECC71",
    "random": "#95A5A6",
}


def _compute_spatial_bins_3d(
    positions: npt.NDArray[np.float64],
    activity: npt.NDArray[np.float64],
    arena_size: tuple[float, ...],
    n_bins: int = 20,
    cell_idx: int | None = None,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    Any,
]:
    """Compute 3D spatial binning of activity with interpolation for sparse data.

    Args:
        positions: Position array (n_samples, 3)
        activity: Activity array (n_samples, n_cells)
        arena_size: Arena size tuple (x, y, z)
        n_bins: Number of spatial bins per dimension
        cell_idx: If None, average across all cells; otherwise specific cell

    Returns:
        x_bins: X bin edges
        y_bins: Y bin edges
        z_bins: Z bin edges
        firing_volume: 3D firing rate map (n_bins, n_bins, n_bins)
    """
    from scipy.interpolate import griddata

    x_max, y_max, z_max = arena_size

    x_bins = np.linspace(0, x_max, n_bins + 1)
    y_bins = np.linspace(0, y_max, n_bins + 1)
    z_bins = np.linspace(0, z_max, n_bins + 1)

    # Get bin centers for interpolation
    x_centers = (x_bins[:-1] + x_bins[1:]) / 2
    y_centers = (y_bins[:-1] + y_bins[1:]) / 2
    z_centers = (z_bins[:-1] + z_bins[1:]) / 2

    # Create 3D grid for interpolation
    X, Y, Z = np.meshgrid(x_centers, y_centers, z_centers, indexing="ij")
    grid_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    # Get activity values to interpolate
    if cell_idx is not None:
        activity_values = activity[:, cell_idx]
        # Interpolate activity onto regular grid
        # Use linear interpolation for smoother grid patterns
        firing_volume_flat = griddata(
            positions, activity_values, grid_points, method="linear", fill_value=0.0
        )
    else:
        # For coverage across all cells: interpolate each cell separately then average
        # This preserves spatial structure better than averaging first
        n_cells = activity.shape[1]
        firing_volume_flat = np.zeros_like(grid_points[:, 0])

        for cell_i in range(n_cells):
            cell_activity = activity[:, cell_i]
            cell_interp = griddata(
                positions, cell_activity, grid_points, method="nearest", fill_value=0.0
            )
            firing_volume_flat += cell_interp

        # Average across cells
        firing_volume_flat /= n_cells

    # Reshape to 3D
    firing_volume = firing_volume_flat.reshape((n_bins, n_bins, n_bins)).astype(
        np.float64
    )

    return (
        x_bins.astype(np.float64),
        y_bins.astype(np.float64),
        z_bins.astype(np.float64),
        firing_volume,
    )


def _get_cell_colors(
    activity: npt.NDArray[np.float64], metadata: dict[str, Any]
) -> tuple[str, list[str] | None, list[str]]:
    """Determine cell types and colors from metadata.

    Returns:
        Tuple of (cell_type, cell_types, colors)
    """
    cell_type = metadata.get("cell_type", "unknown")
    cell_types = metadata.get("cell_types")

    if cell_types is not None:
        # Mixed population
        colors = [CELL_TYPE_COLORS.get(ct, "#7F8C8D") for ct in cell_types]
    else:
        # Single cell type
        colors = [CELL_TYPE_COLORS.get(cell_type, "#7F8C8D")] * activity.shape[1]

    return cell_type, cell_types, colors


def _count_fixed_plots(
    metadata: dict[str, Any],
    show_raster: bool,
    show_fields: bool,
    show_behavior: bool,
    show_ground_truth: bool,
    show_embeddings: bool,
    embedding_methods: list[str],
) -> int:
    """Count fixed plots (not including example cells)."""
    n_fixed = 0
    n_fixed += 1 if show_raster else 0  # Raster
    # Coverage (for place/grid cells) or diagnostics (for random cells)
    if show_fields:
        cell_type = metadata.get("cell_type", "unknown")
        if cell_type in ("place", "grid"):
            n_fixed += 1  # Coverage
        elif cell_type == "random":
            n_fixed += 4  # 4 diagnostic plots
    # Behavior trajectory
    n_fixed += 1 if show_behavior and "positions" in metadata else 0
    # Ground truth embedding
    n_fixed += 1 if show_ground_truth and "ground_truth_embedding" in metadata else 0
    # Learned embeddings
    n_fixed += len(embedding_methods) if show_embeddings else 0
    return n_fixed


def _calculate_optimal_example_cells(n_fixed_plots: int, cell_type: str) -> int:
    """Calculate optimal number of example cells to minimize empty subplots."""

    def calc_grid(n_total: int) -> tuple[int, int]:
        """Calculate grid dimensions for given number of plots."""
        if n_total <= 2:
            return 1, n_total
        elif n_total <= 4:
            return 2, 2
        elif n_total <= 6:
            return 2, 3
        else:
            ncols = 3
            nrows = (n_total + ncols - 1) // ncols  # Ceiling division
            return nrows, ncols

    # Find optimal number of example cells (2-4) that minimizes empty subplots
    best_n_examples = 3  # Default (prefer showing all three cell types)
    min_empty = float("inf")

    for n_examples in range(2, 5):  # Try 2, 3, 4 example cells
        total = n_fixed_plots + n_examples
        nrows, ncols = calc_grid(total)
        grid_size = nrows * ncols
        n_empty = grid_size - total

        # For random cells, prefer n_examples=3 to show all cell types
        # unless it would create > 3 empty subplots
        if cell_type == "random" and n_examples == 3 and n_empty <= 3:
            best_n_examples = 3
            break
        # Prefer configurations with fewer empty subplots
        elif n_empty < min_empty:
            min_empty = n_empty
            best_n_examples = n_examples

    return best_n_examples


def _create_raster_plot(
    activity: npt.NDArray[np.float64],
    colors: list[str],
    max_cells: int,
    subplot_position: int,
    cell_types: list[str] | None = None,
) -> PlotSpec:
    """Create raster plot specification as heatmap with proper cell ID ticks.

    For single cell types: shows firing rate in Hz with 'hot' colormap.
    For mixed populations: shows boolean on/off activity colored by cell type.
    """
    n_samples, n_cells = activity.shape

    # Subsample cells if too many
    if n_cells > max_cells:
        step = n_cells // max_cells
        cell_indices = np.arange(0, n_cells, step)[:max_cells]
        activity_sub = activity[:, cell_indices]
        colors_sub = [colors[i] for i in cell_indices] if colors else None
        if cell_types is not None:
            cell_types_sub = [cell_types[i] for i in cell_indices]
        else:
            cell_types_sub = None
    else:
        cell_indices = np.arange(n_cells)
        activity_sub = activity
        colors_sub = colors
        cell_types_sub = cell_types

    # Transpose to (n_cells, n_samples) for imshow-style display
    raster_data = activity_sub.T

    # Create integer xticks for time axis
    n_time_ticks = min(6, n_samples // 50)  # Reasonable number of ticks
    if n_time_ticks > 0:
        time_tick_step = n_samples // n_time_ticks
        time_ticks = np.arange(0, n_samples + 1, time_tick_step)
        time_labels = time_ticks.astype(int)
    else:
        time_ticks = None
        time_labels = None

    # Create integer yticks for cell IDs
    n_cell_ticks = min(10, len(cell_indices))  # Max 10 ticks to avoid crowding
    if n_cell_ticks > 0:
        cell_tick_step = max(1, len(cell_indices) // n_cell_ticks)
        cell_ticks = np.arange(0, len(cell_indices) + 1, cell_tick_step)
        cell_labels = cell_indices[::cell_tick_step].astype(int)
        # Pad labels if needed
        while len(cell_labels) < len(cell_ticks):
            cell_labels = np.append(cell_labels, cell_indices[-1])
        cell_labels = cell_labels[: len(cell_ticks)]
    else:
        cell_ticks = None
        cell_labels = None

    # Check if this is a mixed population (multiple cell types)
    is_mixed = cell_types_sub is not None and len(set(cell_types_sub)) > 1

    if is_mixed:
        # For mixed populations: create boolean activity colored by cell type
        # Create RGB image where each cell's activity is colored by its type
        import matplotlib.colors as mcolors

        # Convert hex colors to RGB
        if colors_sub is None:
            colors_sub = []
        rgb_colors = [mcolors.hex2color(c) for c in colors_sub]

        # Create RGB array: (n_cells, n_samples, 3)
        rgb_raster = np.zeros((*raster_data.shape, 3))

        # Threshold activity to boolean (on/off)
        # Use mean + 0.5*std as threshold for "active"
        threshold = raster_data.mean() + 0.5 * raster_data.std()
        active_mask = raster_data > threshold

        # Apply colors where active
        for i in range(len(cell_indices)):
            rgb_raster[i, active_mask[i], :] = rgb_colors[i]

        spec = PlotSpec(
            data=rgb_raster,
            plot_type="heatmap",
            subplot_position=subplot_position,
            title="Neural Activity Raster (Mixed Population)",
            cmap=None,  # No colormap for RGB
            colorbar=False,  # No colorbar for boolean
            kwargs={
                "x_label": "Time (samples)",
                "y_label": "Cell ID",
                "interpolation": "nearest",
                "aspect": "auto",
                "extent": [0, n_samples, 0, len(cell_indices)],
                "origin": "lower",
                "set_xticks": time_ticks,
                "set_xticklabels": time_labels,
                "set_yticks": cell_ticks,
                "set_yticklabels": cell_labels,
            },
        )
    else:
        # For single cell types: show firing rate with hot colormap
        spec = PlotSpec(
            data=raster_data,
            plot_type="heatmap",
            subplot_position=subplot_position,
            title="Neural Activity Raster",
            cmap="hot",
            colorbar=True,
            colorbar_label="Firing Rate (Hz)",
            kwargs={
                "x_label": "Time (samples)",
                "y_label": "Cell ID",
                "interpolation": "nearest",
                "aspect": "auto",
                "extent": [0, n_samples, 0, len(cell_indices)],
                "origin": "lower",
                "set_xticks": time_ticks,
                "set_xticklabels": time_labels,
                "set_yticks": cell_ticks,
                "set_yticklabels": cell_labels,
            },
        )

    return spec


def _create_coverage_heatmap_3d(
    activity: npt.NDArray[np.float64],
    metadata: dict[str, Any],
    subplot_position: int,
) -> list[PlotSpec] | None:
    """Create 3D spatial coverage visualization.

    For place cells: shows spatial coverage heatmap (average firing rate across all cells).
    For grid cells: shows autocorrelation to reveal periodic structure.
    """
    positions = metadata.get("positions")
    arena_size = metadata.get("arena_size", (1.0, 1.0, 1.0))
    cell_type = metadata.get("cell_type", "place")

    if positions is None or positions.shape[1] != 3:
        return None

    # Get arena dimensions
    x_max, y_max, z_max = arena_size

    # Choose colormap based on cell type
    cmap = "hot" if cell_type == "place" else "viridis"

    specs = []

    if cell_type == "grid":
        # For grid cells: compute autocorrelation to reveal periodic structure
        try:
            # Use spatial binning first
            n_bins = 60
            x_centers, y_centers, z_centers, firing_volume = _compute_spatial_bins_3d(
                positions, activity, arena_size, n_bins=n_bins, cell_idx=0
            )

            # Normalize firing volume to have zero mean for autocorrelation
            firing_volume_centered = firing_volume - np.nanmean(firing_volume)
            firing_volume_centered = np.nan_to_num(firing_volume_centered, nan=0.0)

            # Compute 3D autocorrelation using FFT (faster than spatial method)
            # Use power spectrum approach (Wiener-Khinchin theorem):
            # autocorrelation = IFFT(|FFT(signal)|^2)
            fft_3d = np.fft.fftn(firing_volume_centered)
            power_spectrum = np.abs(fft_3d) ** 2
            autocorr = np.fft.ifftn(power_spectrum).real
            autocorr = np.fft.fftshift(autocorr)  # Center the zero-lag peak

            # Normalize autocorrelation by zero-lag value
            center_idx = tuple(s // 2 for s in autocorr.shape)
            if autocorr[center_idx] != 0:
                autocorr_normalized = autocorr / autocorr[center_idx]
            else:
                autocorr_normalized = autocorr

            # Create orthogonal slices through the center (where zero-lag peak is)
            center_x, center_y, center_z = center_idx
            xy_slice = autocorr_normalized[:, :, center_z]  # XY plane through center Z
            xz_slice = autocorr_normalized[:, center_y, :]  # XZ plane through center Y
            yz_slice = autocorr_normalized[center_x, :, :]  # YZ plane through center X

            # Create lag axes (in meters, centered at 0)
            x_lags = np.linspace(-x_max, x_max, autocorr_normalized.shape[0])
            y_lags = np.linspace(-y_max, y_max, autocorr_normalized.shape[1])
            z_lags = np.linspace(-z_max, z_max, autocorr_normalized.shape[2])

            wall_spec = PlotSpec(
                data={
                    "xy": xy_slice,
                    "xz": xz_slice,
                    "yz": yz_slice,
                    "x_centers": x_lags,
                    "y_centers": y_lags,
                    "z_centers": z_lags,
                    # Add wall positions to place at boundaries
                    "xy_position": z_lags[0],  # Place XY plane at minimum Z
                    "xz_position": y_lags[0],  # Place XZ plane at minimum Y
                    "yz_position": x_lags[0],  # Place YZ plane at minimum X
                },
                plot_type="heatmap_walls",
                subplot_position=subplot_position,
                title="Grid Field Autocorrelation (3D)",
                cmap=cmap,
                colorbar=True,
                colorbar_label="Normalized Autocorr",
                kwargs={},
            )
            specs.append(wall_spec)
        except Exception as e:
            # If anything goes wrong creating the autocorrelation, skip it silently
            import warnings

            warnings.warn(f"Failed to create 3D autocorrelation: {e}", stacklevel=2)
    else:
        # For place cells: use spatial binning of all neurons (coverage heatmap)
        n_bins = 30
        x_centers, y_centers, z_centers, firing_volume = _compute_spatial_bins_3d(
            positions, activity, arena_size, n_bins=n_bins, cell_idx=None
        )

        # Create orthogonal slices through the center
        center_x, center_y, center_z = (
            firing_volume.shape[0] // 2,
            firing_volume.shape[1] // 2,
            firing_volume.shape[2] // 2,
        )
        xy_slice = firing_volume[:, :, center_z]  # XY plane through center Z
        xz_slice = firing_volume[:, center_y, :]  # XZ plane through center Y
        yz_slice = firing_volume[center_x, :, :]  # YZ plane through center X

        wall_spec = PlotSpec(
            data={
                "xy": xy_slice,
                "xz": xz_slice,
                "yz": yz_slice,
                "x_centers": x_centers,
                "y_centers": y_centers,
                "z_centers": z_centers,
                # Add wall positions to place at boundaries
                "xy_position": z_centers[0],  # Place XY plane at minimum Z
                "xz_position": y_centers[0],  # Place XZ plane at minimum Y
                "yz_position": x_centers[0],  # Place YZ plane at minimum X
            },
            plot_type="heatmap_walls",
            subplot_position=subplot_position,
            title="Place Field Coverage (3D)",
            cmap=cmap,
            colorbar=True,
            colorbar_label="Avg. Firing Rate (Hz)",
            kwargs={},
        )
        specs.append(wall_spec)

    return specs
