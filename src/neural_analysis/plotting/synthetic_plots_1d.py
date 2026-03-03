"""1D visualization functions for synthetic neural data.

Provides 1D-specific plotting: spatial binning, coverage histograms,
and head direction tuning curves.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from neural_analysis.plotting.grid_config import PlotSpec
from neural_analysis.plotting.synthetic_plots_3d import CELL_TYPE_COLORS


def _compute_spatial_bins_1d(
    positions: npt.NDArray[np.float64],
    activity: npt.NDArray[np.float64],
    arena_size: float | tuple[float, ...],
    n_bins: int = 50,
    cell_idx: int | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute 1D spatial binning of activity.

    Args:
        positions: Position array, shape (n_samples, 1).
        activity: Activity array, shape (n_samples, n_cells).
        arena_size: Arena size (float or tuple).
        n_bins: Number of spatial bins.
        cell_idx: If specified, only bin this cell. Otherwise average all cells.

    Returns:
        bin_centers: Center position of each bin.
        binned_rates: Average firing rate in each bin.
    """
    # Get arena length
    x_max = arena_size[0] if isinstance(arena_size, tuple) else arena_size

    # Create position bins
    x_bins = np.linspace(0, x_max, n_bins + 1)
    binned_rates = np.zeros(n_bins)

    # Compute average activity per position bin
    for i in range(n_bins):
        mask = (positions[:, 0] >= x_bins[i]) & (positions[:, 0] < x_bins[i + 1])
        if mask.sum() > 0:
            if cell_idx is not None:
                binned_rates[i] = activity[mask, cell_idx].mean()
            else:
                binned_rates[i] = activity[mask, :].mean()

    # Get bin centers for plotting
    bin_centers = (x_bins[:-1] + x_bins[1:]) / 2

    return bin_centers, binned_rates


def _create_coverage_histogram_1d(
    activity: npt.NDArray[np.float64],
    metadata: dict[str, Any],
    subplot_position: int,
) -> PlotSpec | None:
    """Create 1D coverage histogram for place/grid cells (unified function)."""
    positions = metadata.get("positions")
    arena_size = metadata.get("arena_size", 1.0)
    cell_type = metadata.get("cell_type", "place")

    if positions is None or positions.ndim != 2 or positions.shape[1] != 1:
        return None

    # Use helper function for spatial binning
    x_centers, coverage = _compute_spatial_bins_1d(
        positions, activity, arena_size, n_bins=50, cell_idx=None
    )

    # Choose color based on cell type
    color = CELL_TYPE_COLORS.get(cell_type, "#E74C3C")
    title = f"{cell_type.title()} Field Coverage (1D)"

    # Create line plot spec
    spec = PlotSpec(
        data={"x": x_centers, "y": coverage},
        plot_type="line",
        subplot_position=subplot_position,
        title=title,
        color=color,
        line_width=2,
        kwargs={
            "x_label": "Position (m)",
            "y_label": "Avg. Firing Rate (Hz)",
        },
    )

    return spec


def _compute_hd_tuning_curve(
    activity: npt.NDArray[np.float64],
    head_directions: npt.NDArray[np.float64],
    cell_idx: int,
    n_bins: int = 72,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute head direction tuning curve for a single cell.

    Shared helper function to avoid code duplication.

    Args:
        activity: Neural activity matrix (n_samples, n_cells).
        head_directions: Head direction angles in radians [0, 2π), shape (n_samples,).
            Angles are normalized to [0, 2π) if needed.
        cell_idx: Index of the cell to compute tuning for.
        n_bins: Number of angular bins.

    Returns:
        angles_deg: Bin centers in degrees [0, 360).
        rates: Mean firing rates per bin.
    """
    # Normalize head directions to [0, 2π) range
    hd_normalized = head_directions.copy()
    hd_normalized = hd_normalized % (2 * np.pi)
    # Handle negative angles
    hd_normalized[hd_normalized < 0] += 2 * np.pi

    # Create bins from 0 to 2π
    angle_bins = np.linspace(0, 2 * np.pi, n_bins + 1)
    bin_centers = (angle_bins[:-1] + angle_bins[1:]) / 2
    rates = np.zeros(n_bins)

    for i in range(n_bins):
        # Handle the last bin to include 2π (wrap-around)
        if i == n_bins - 1:
            mask = (hd_normalized >= angle_bins[i]) & (
                hd_normalized <= angle_bins[i + 1]
            )
        else:
            mask = (hd_normalized >= angle_bins[i]) & (
                hd_normalized < angle_bins[i + 1]
            )
        if mask.sum() > 0:
            rates[i] = activity[mask, cell_idx].mean()

    angles_deg = np.degrees(bin_centers)
    return angles_deg, rates


def _create_hd_example_cells(
    activity: npt.NDArray[np.float64],
    metadata: dict[str, Any],
    colors: list[str],
    subplot_position: int,
    n_examples: int = 3,
) -> list[PlotSpec]:
    """Create example head direction tuning curves.

    Args:
        activity: Neural activity matrix.
        metadata: Dataset metadata.
        colors: Color list for cells.
        subplot_position: Starting subplot position.
        n_examples: Number of example cells to show (2-4).

    Returns:
        List of PlotSpec objects.
    """
    specs: list[PlotSpec] = []
    preferred_directions = metadata.get("preferred_directions")
    head_directions = metadata.get("head_directions")

    if preferred_directions is None or head_directions is None:
        return specs

    # Find n_examples cells with highest peak firing rates for better visualization
    peak_rates = activity.max(axis=0)
    top_cells = np.argsort(peak_rates)[-n_examples:][::-1]  # Top cells, highest first

    for idx, cell_idx in enumerate(top_cells):
        # Use shared helper function
        angles_deg, rates = _compute_hd_tuning_curve(
            activity, head_directions, int(cell_idx), n_bins=72
        )

        # Get the preferred direction for this cell for the title
        pref_dir_deg = np.degrees(preferred_directions[cell_idx])
        # Normalize to [0, 360) for display
        pref_dir_deg = pref_dir_deg % 360

        spec = PlotSpec(
            data={"x": angles_deg, "y": rates},
            plot_type="line",
            subplot_position=subplot_position + idx,
            title=f"HD Cell {cell_idx} (pref: {pref_dir_deg:.0f}°)",
            color=CELL_TYPE_COLORS["head_direction"],
            line_width=2,
            alpha=0.8,
            kwargs={
                "x_label": "Head Direction (°)",
                "y_label": "Firing Rate (Hz)",
                "set_xlim": (0.0, 360.0),  # Explicitly set full 0-360° range
            },
        )
        specs.append(spec)

    return specs


def _create_random_hd_tuning_examples(
    activity: npt.NDArray[np.float64],
    metadata: dict[str, Any],
    colors: list[str],
    subplot_position: int,
    n_examples: int = 3,
) -> list[PlotSpec]:
    """Create example head direction tuning curves for random cells.

    Random cells should show flat tuning curves (no directional preference),
    demonstrating lack of head direction tuning.

    Args:
        activity: Neural activity matrix (n_samples, n_cells).
        metadata: Dataset metadata.
        colors: Color list for cells.
        subplot_position: Starting subplot position.
        n_examples: Number of example cells to show (2-4).

    Returns:
        List of PlotSpec objects for head direction tuning curves.
    """
    specs: list[PlotSpec] = []
    head_directions = metadata.get("head_directions")

    # Generate head directions if not present (backward compatibility)
    if head_directions is None:
        rng = np.random.default_rng(42)
        n_samples = activity.shape[0]
        head_directions = rng.uniform(-np.pi, np.pi, size=n_samples)

    # Select n_examples cells with highest variance for better visualization
    cell_variances = activity.var(axis=0)
    top_cells = np.argsort(cell_variances)[-n_examples:][::-1]

    for idx, cell_idx in enumerate(top_cells):
        # Use shared helper function (fewer bins for random cells)
        angles_deg, rates = _compute_hd_tuning_curve(
            activity, head_directions, int(cell_idx), n_bins=36
        )

        spec = PlotSpec(
            data={"x": angles_deg, "y": rates},
            plot_type="line",
            subplot_position=subplot_position + idx,
            title=f"Random Cell {cell_idx} HD Tuning (Should be Flat)",
            color=CELL_TYPE_COLORS.get("random", "#95A5A6"),
            line_width=2,
            alpha=0.8,
            kwargs={
                "x_label": "Head Direction (°)",
                "y_label": "Firing Rate (Hz)",
                "set_xlim": (0.0, 360.0),  # Explicitly set full 0-360° range
            },
        )
        specs.append(spec)

    return specs


def _create_hd_tuning_plot(
    activity: npt.NDArray[np.float64],
    metadata: dict[str, Any],
    colors: list[str],
    subplot_position: int,
) -> PlotSpec | None:
    """Create head direction tuning curve plot for first cell.

    Legacy function - consider using _create_hd_example_cells instead.
    """
    preferred_angles = metadata.get("preferred_angles")
    head_directions = metadata.get("head_directions")

    if preferred_angles is None or head_directions is None:
        return None

    # Use shared helper function for first cell
    angles_deg, rates = _compute_hd_tuning_curve(
        activity, head_directions, cell_idx=0, n_bins=36
    )

    spec = PlotSpec(
        data={"x": angles_deg, "y": rates},
        plot_type="line",
        subplot_position=subplot_position,
        title="Head Direction Tuning",
        color=colors[0],
        kwargs={
            "x_label": "Head Direction (°)",
            "y_label": "Firing Rate (Hz)",
            "set_xlim": (0.0, 360.0),  # Explicitly set full 0-360° range
        },
    )

    return spec
