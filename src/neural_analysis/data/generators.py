"""Individual cell type generator functions for synthetic neural data.

This module provides functions to generate firing data for specific neural
cell types: place cells, grid cells, head direction cells, random cells,
and mixed populations. Also includes manifold mapping functions and shape
distance validation data generators.
"""

from __future__ import annotations

from typing import Any, Literal, cast

import numpy as np
import numpy.typing as npt

from neural_analysis.data.trajectories_gen import (
    generate_head_direction,
    generate_position_trajectory,
)


def _compute_grid_pattern_with_harmonics(
    positions: npt.NDArray[np.float64],
    phase_offset: npt.NDArray[np.float64],
    grid_spacing: float,
    axes: list[npt.NDArray[np.float64]] | None = None,
    harmonic_weights: tuple[float, ...] = (1.0, 0.4, 0.2),
) -> npt.NDArray[np.float64]:
    """Compute grid cell firing pattern with harmonics.

    Generates periodic firing pattern using sum of cosines with multiple harmonics
    for biological realism (sharper firing fields).

    Args:
        positions: Position array (n_samples, n_dims)
        phase_offset: Phase offset for this cell (n_dims,)
        grid_spacing: Distance between grid peaks in meters
        axes: Optional list of grid axes for 2D/3D hexagonal grids.
            If None, uses axis-aligned pattern for 1D.
            For 2D: 2 axes at 60° apart (hexagonal)
            For 3D: 4 axes forming tetrahedral symmetry (FCC-like)
        harmonic_weights: Weights for fundamental, 2nd, 3rd harmonics.
            Default: (1.0, 0.4, 0.2) for sharp biological fields.

    Returns:
        Firing rates with harmonics applied (n_samples,)
    """
    n_dims = positions.shape[1]

    if axes is not None and n_dims == 2:
        # 2D hexagonal grid with 3 axes at 60° apart
        proj1 = np.dot(positions - phase_offset, axes[0])
        proj2 = np.dot(positions - phase_offset, axes[1])

        # Sum of cosines pattern with harmonics
        rates = np.zeros(len(positions))
        for harmonic_idx, weight in enumerate(harmonic_weights, start=1):
            freq = 2 * np.pi * harmonic_idx / grid_spacing
            rates += weight * (
                np.cos(freq * proj1)
                + np.cos(freq * proj2)
                + np.cos(freq * (proj1 - proj2))
            )
    elif axes is not None and n_dims == 3:
        # 3D hexagonal/tetrahedral grid (FCC-like structure)
        # Project onto 4 axes with tetrahedral symmetry
        rates = np.zeros(len(positions))
        centered_pos = positions - phase_offset

        for harmonic_idx, weight in enumerate(harmonic_weights, start=1):
            freq = 2 * np.pi * harmonic_idx / grid_spacing
            for axis in axes:
                proj = np.dot(centered_pos, axis)
                rates += weight * np.cos(freq * proj)
    else:
        # 1D: axis-aligned grid pattern
        rates = np.zeros(len(positions))
        for dim in range(n_dims):
            for harmonic_idx, weight in enumerate(harmonic_weights, start=1):
                freq = 2 * np.pi * harmonic_idx / grid_spacing
                rates += weight * np.cos(freq * (positions[:, dim] - phase_offset[dim]))

    return rates


def generate_place_cells(
    n_cells: int = 100,
    n_samples: int = 1000,
    positions: npt.NDArray[np.floating[Any]] | None = None,
    arena_size: float | tuple[float, ...] = (1.0, 1.0),
    field_size: float = 0.2,
    peak_rate: float = 10.0,
    noise_level: float = 0.1,
    sampling_rate: float = 20.0,
    seed: int | None = None,
    plot: bool = True,
) -> tuple[npt.NDArray[np.float64], dict[str, Any]]:
    """Generate place cell firing data in 1D, 2D, or 3D.

    Place cells fire when animal is in specific locations (place fields).
    Each cell has a Gaussian tuning curve centered at a random location.

    Args:
        n_cells: Number of place cells.
        n_samples: Number of time points.
        positions: Optional position trajectory, shape (n_samples, n_dims).
            If None, generates random trajectory.
        arena_size: Size of arena. Float for 1D, tuple for 2D/3D.
        field_size: Standard deviation of Gaussian place field in meters.
        peak_rate: Maximum firing rate in Hz.
        noise_level: Amount of Poisson noise (0=none, 1=full Poisson).
        sampling_rate: Sampling rate in Hz (default: 20 Hz for calcium imaging).
            Controls temporal resolution of neural activity.
        seed: Random seed for reproducibility.
        plot: If True, create comprehensive visualization using PlotGrid system.

    Returns:
        activity: Neural activity matrix, shape (n_samples, n_cells).
            Firing rates in Hz.
        metadata: Dictionary with:
            - 'field_centers': Place field centers, shape (n_cells, n_dims)
            - 'positions': Position trajectory used, shape (n_samples, n_dims)
            - 'cell_type': 'place' for all cells
            - 'n_dims': Dimensionality (1, 2, or 3)
            - 'sampling_rate': Sampling rate in Hz

    Examples:
        >>> # 1D place cells with automatic plotting (20 Hz sampling)
        >>> activity, meta = generate_place_cells(50, 1000, arena_size=2.0, plot=True)
        >>> # 2D place cells without plotting (30 Hz sampling)
        >>> activity, meta = generate_place_cells(50, 1000, arena_size=(1.0, 1.0),
        ...                                        sampling_rate=30.0, plot=False)
        >>> # 3D place cells
        >>> activity, meta = generate_place_cells(50, 1000, arena_size=(1.0, 1.0, 0.5))
    """
    rng = np.random.default_rng(seed)

    # Determine dimensionality
    if isinstance(arena_size, (int, float)):
        n_dims = 1
        arena_size = (float(arena_size),)
    else:
        n_dims = len(arena_size)
        arena_size = tuple(arena_size)

    # Generate or use provided positions
    if positions is None:
        positions = generate_position_trajectory(
            n_samples, arena_size=arena_size, seed=seed
        )
    positions = np.asarray(positions, dtype=np.float64)
    positions = cast("npt.NDArray[np.float64]", positions)
    n_dims = positions.shape[1] if positions.ndim > 1 else 1
    if n_dims == 1 and positions.ndim == 1:
        positions = positions.reshape(-1, 1)

    # Random place field centers
    field_centers: npt.NDArray[np.float64]
    field_radii: npt.NDArray[np.float64]
    field_angles: npt.NDArray[np.float64] | None = None
    if n_dims == 1:
        field_centers = rng.uniform(0, arena_size[0], size=(n_cells, 1))
    else:
        field_centers = rng.uniform([0] * n_dims, arena_size, size=(n_cells, n_dims))
    field_centers = np.asarray(field_centers, dtype=np.float64)

    # Random oval-shaped place fields (anisotropic Gaussian)
    # Each cell has random radii and orientation
    if n_dims == 1:
        field_radii = rng.uniform(0.7, 1.3, size=(n_cells, 1)) * field_size
    elif n_dims == 2:
        # Random aspect ratio (elongation)
        field_radii = rng.uniform(0.6, 1.4, size=(n_cells, 2)) * field_size
        # Random orientation for oval fields
        field_angles = rng.uniform(0, np.pi, size=n_cells)
    elif n_dims == 3:
        field_radii = rng.uniform(0.6, 1.4, size=(n_cells, 3)) * field_size
        field_angles = None
    else:  # pragma: no cover - defensive
        raise ValueError(
            f"Unsupported dimensionality. Expected: 1, 2, or 3. Got: {n_dims!r}"
        )
    field_radii = np.asarray(field_radii, dtype=np.float64)

    # Ensure arrays are set and compute firing rates based on distance to field center
    assert field_centers is not None
    activity = np.zeros((n_samples, n_cells))

    # Low baseline firing rate outside place field (realistic for place cells)
    baseline_rate = peak_rate * 0.01  # 1% of peak rate

    for i in range(n_cells):
        if n_dims == 1:
            # 1D: Simple Gaussian with variable width
            distances = np.abs(positions[:, 0] - field_centers[i, 0])
            rates = baseline_rate + peak_rate * np.exp(
                -(distances**2) / (2 * field_radii[i, 0] ** 2)
            )

        elif n_dims == 2:
            # 2D: Rotated oval (anisotropic Gaussian)
            # Translate to field center
            dx = positions[:, 0] - field_centers[i, 0]
            dy = positions[:, 1] - field_centers[i, 1]

            # Rotate to field orientation
            assert field_angles is not None
            angle = float(field_angles[i])
            dx_rot = dx * np.cos(angle) + dy * np.sin(angle)
            dy_rot = -dx * np.sin(angle) + dy * np.cos(angle)

            # Compute anisotropic distance (Mahalanobis-like distance)
            dist_x = (dx_rot / field_radii[i, 0]) ** 2
            dist_y = (dy_rot / field_radii[i, 1]) ** 2
            rates = baseline_rate + peak_rate * np.exp(-(dist_x + dist_y) / 2)

        elif n_dims == 3:
            # 3D: Ellipsoid (axis-aligned for simplicity)
            dx = (positions[:, 0] - field_centers[i, 0]) / field_radii[i, 0]
            dy = (positions[:, 1] - field_centers[i, 1]) / field_radii[i, 1]
            dz = (positions[:, 2] - field_centers[i, 2]) / field_radii[i, 2]
            rates = baseline_rate + peak_rate * np.exp(-(dx**2 + dy**2 + dz**2) / 2)

        # Add Poisson noise: sample from Poisson distribution then add Gaussian noise
        if noise_level > 0:
            # Poisson noise (spike count variability)
            rates = rng.poisson(rates)
            # Small Gaussian noise on top
            rates = rates + rng.normal(0, noise_level * peak_rate, size=rates.shape)
            # Clip to non-negative
            rates = np.maximum(0, rates)

        activity[:, i] = rates

    metadata = {
        "field_centers": field_centers,
        "field_radii": field_radii,
        "field_angles": field_angles if n_dims == 2 else None,
        "positions": positions,
        "cell_type": "place",
        "arena_size": arena_size,
        "field_size": field_size,
        "n_dims": n_dims,
        "sampling_rate": sampling_rate,
    }

    # Create visualization if requested
    if plot:
        from neural_analysis.plotting.synthetic_plots import plot_synthetic_data

        # Force embeddings to always be 2D, even for 3D spatial environments
        # Raster plots are always 2D heatmaps by design
        plot_synthetic_data(
            activity,
            metadata,
            show_raster=True,
            show_fields=True,
            show_behavior=True,
            show_ground_truth=False,
            show_embeddings=True,
            embedding_methods=["pca", "umap"],
            n_embedding_dims=2,  # Always 2D for embeddings, regardless of spatial dims
        )
    return activity, metadata


def generate_grid_cells(
    n_cells: int = 50,
    n_samples: int = 1000,
    positions: npt.NDArray[np.floating[Any]] | None = None,
    arena_size: float | tuple[float, ...] = (2.0, 2.0),
    grid_spacing: float = 0.4,
    grid_orientation: float = 0.0,
    peak_rate: float = 10.0,
    noise_level: float = 0.1,
    sampling_rate: float = 20.0,
    seed: int | None = None,
    plot: bool = True,
) -> tuple[npt.NDArray[np.float64], dict[str, Any]]:
    """Generate grid cell firing data in 1D, 2D, or 3D.

    Grid cells fire at multiple locations arranged in regular grid pattern.

    Args:
        n_cells: Number of grid cells.
        n_samples: Number of time points.
        positions: Optional position trajectory, shape (n_samples, n_dims).
        arena_size: Size of arena. Float for 1D, tuple for 2D/3D.
        grid_spacing: Distance between grid peaks in meters.
        grid_orientation: Grid rotation in degrees (2D/3D only).
        peak_rate: Maximum firing rate in Hz.
        noise_level: Amount of Poisson noise.
        sampling_rate: Sampling rate in Hz (default: 20 Hz for calcium imaging).
            Controls temporal resolution of neural activity.
        seed: Random seed.
        plot: If True, create comprehensive visualization using PlotGrid system.

    Returns:
        activity: Neural activity matrix, shape (n_samples, n_cells).
        metadata: Dictionary with grid cell parameters.

    Examples:
        >>> # 1D grid cells with plotting
        >>> activity, meta = generate_grid_cells(30, 1000, arena_size=2.0, plot=True)
        >>> # 2D grid cells without plotting
        >>> activity, meta = generate_grid_cells(
        ...     30, 1000, arena_size=(2.0, 2.0), plot=False
        ... )
    """
    rng = np.random.default_rng(seed)

    # Determine dimensionality
    if isinstance(arena_size, (int, float)):
        n_dims = 1
        arena_size = (float(arena_size),)
    else:
        n_dims = len(arena_size)
        arena_size = tuple(arena_size)

    # Generate or use provided positions
    if positions is None:
        positions = generate_position_trajectory(
            n_samples, arena_size=arena_size, seed=seed
        )
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim == 1:
        positions = positions.reshape(-1, 1)
    positions = cast("npt.NDArray[np.float64]", positions)
    n_dims = positions.shape[1] if positions.ndim > 1 else 1

    activity = np.zeros((n_samples, n_cells))

    if n_dims == 1:
        # 1D: Periodic firing with harmonics for biological realism
        phase_offsets = rng.uniform(0, grid_spacing, size=n_cells)

        for i in range(n_cells):
            # Use helper function for harmonic pattern
            rates = _compute_grid_pattern_with_harmonics(
                positions,
                np.array([phase_offsets[i]]),
                grid_spacing,
                axes=None,
                harmonic_weights=(0.6, 0.25, 0.15),
            )

            # Normalize to [0, 1] range, then scale
            rates = (rates - rates.min()) / (rates.max() - rates.min() + 1e-10)
            rates = rates * peak_rate

            # Add noise for biological realism (Poisson + Gaussian)
            if noise_level > 0:
                # Sample from Poisson distribution (spike count variability)
                rates = rng.poisson(rates)
                # Add small Gaussian noise on top
                rates = rates + rng.normal(
                    0, noise_level * peak_rate * 0.1, size=rates.shape
                )
                # Clip to non-negative
                rates = np.maximum(0, rates)

            activity[:, i] = rates

        metadata = {
            "phase_offsets": phase_offsets,
            "positions": positions,
            "cell_type": "grid",
            "grid_spacing": grid_spacing,
            "arena_size": arena_size,
            "n_dims": n_dims,
            "sampling_rate": sampling_rate,
        }

    elif n_dims == 2:
        # 2D: Hexagonal grid pattern with harmonics
        phase_offsets = rng.uniform(
            [0, 0], [grid_spacing, grid_spacing], size=(n_cells, 2)
        )

        # Grid axes (60 degrees apart)
        theta = np.radians(grid_orientation)
        axis1 = np.array([np.cos(theta), np.sin(theta)])
        axis2 = np.array([np.cos(theta + np.pi / 3), np.sin(theta + np.pi / 3)])
        axes = [axis1, axis2]

        for i in range(n_cells):
            # Use helper function for harmonic pattern
            rates = _compute_grid_pattern_with_harmonics(
                positions,
                phase_offsets[i],
                grid_spacing,
                axes=axes,
                harmonic_weights=(1.0, 0.4, 0.2),
            )

            # Normalize to [0, 1], then scale
            rates = (rates - rates.min()) / (rates.max() - rates.min() + 1e-10)
            rates = rates * peak_rate

            # Add Poisson noise (proper spike count variability) + small Gaussian noise
            if noise_level > 0:
                rates = rng.poisson(rates) + rng.normal(
                    0, 0.1 * noise_level * peak_rate, size=rates.shape
                )
                rates = np.maximum(0, rates)  # Clip to non-negative

            activity[:, i] = rates

        metadata = {
            "phase_offsets": phase_offsets,
            "positions": positions,
            "cell_type": "grid",
            "grid_spacing": grid_spacing,
            "grid_orientation": grid_orientation,
            "arena_size": arena_size,
            "n_dims": n_dims,
            "sampling_rate": sampling_rate,
        }

    elif n_dims == 3:
        # 3D: Hexagonal/tetrahedral grid pattern (FCC-like) with harmonics
        # Create hierarchy of grid spacings (biological realism:
        # dorsal-ventral gradient)
        # Generate multiple scales from fine to coarse
        grid_spacings = np.zeros(n_cells)
        cells_per_scale = max(1, n_cells // 5)  # Divide into ~5 scale groups

        for scale_idx in range(5):
            start_idx = scale_idx * cells_per_scale
            end_idx = min((scale_idx + 1) * cells_per_scale, n_cells)
            if start_idx >= n_cells:
                break
            # Scale factor: 1.0, 1.4, 2.0, 2.8, 4.0 (roughly sqrt(2) progression)
            scale_factor = 1.4**scale_idx
            grid_spacings[start_idx:end_idx] = grid_spacing * scale_factor

        # Fill any remaining cells
        if end_idx < n_cells:
            grid_spacings[end_idx:] = grid_spacing * (1.4**4)

        phase_offsets = rng.uniform([0, 0, 0], [grid_spacing] * 3, size=(n_cells, 3))

        # Define 4 tetrahedral axes for 3D hexagonal packing (FCC crystal structure)
        # These create face-centered cubic (FCC) symmetry, similar to 3D grid cells
        theta = np.radians(grid_orientation)
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        # Four axes pointing toward vertices of a tetrahedron
        axis1 = np.array([1, 1, 1]) / np.sqrt(3)
        axis2 = np.array([1, -1, -1]) / np.sqrt(3)
        axis3 = np.array([-1, 1, -1]) / np.sqrt(3)
        axis4 = np.array([-1, -1, 1]) / np.sqrt(3)

        # Apply rotation if specified (rotate around z-axis)
        if grid_orientation != 0:
            rotation_z = np.array([[cos_t, -sin_t, 0], [sin_t, cos_t, 0], [0, 0, 1]])
            axis1 = rotation_z @ axis1
            axis2 = rotation_z @ axis2
            axis3 = rotation_z @ axis3
            axis4 = rotation_z @ axis4

        axes = [axis1, axis2, axis3, axis4]

        for i in range(n_cells):
            # Use cell-specific grid spacing for frequency diversity
            cell_grid_spacing = grid_spacings[i]

            # Use helper function for harmonic pattern with tetrahedral axes
            rates = _compute_grid_pattern_with_harmonics(
                positions,
                phase_offsets[i],
                cell_grid_spacing,
                axes=axes,
                harmonic_weights=(1.0, 0.4, 0.2),
            )

            # Normalize to [0, 1], then scale
            rates = (rates - rates.min()) / (rates.max() - rates.min() + 1e-10)
            rates = rates * peak_rate

            # Add Gaussian noise only (Poisson at high rates approaches Gaussian)
            # For 3D, use only Gaussian noise to preserve spatial structure better
            if noise_level > 0:
                # Scale noise by peak rate for biological realism
                rates = rates + rng.normal(0, noise_level * peak_rate, size=rates.shape)
                rates = np.maximum(0, rates)  # Clip to non-negative

            activity[:, i] = rates

        metadata = {
            "phase_offsets": phase_offsets,
            "grid_spacings": grid_spacings,  # Cell-specific spacings
            "grid_spacing": grid_spacing,  # Base spacing
            "positions": positions,
            "cell_type": "grid",
            "grid_orientation": grid_orientation,
            "arena_size": arena_size,
            "n_dims": n_dims,
            "sampling_rate": sampling_rate,
        }

    # Create visualization if requested
    if plot:
        from neural_analysis.plotting.synthetic_plots import plot_synthetic_data

        plot_synthetic_data(
            activity,
            metadata,
            show_raster=True,
            show_fields=True,
            show_behavior=True,
            show_ground_truth=False,
            show_embeddings=True,
            embedding_methods=["pca", "umap"],
            n_embedding_dims=2,
        )

    return activity, metadata


def generate_head_direction_cells(
    n_cells: int = 60,
    n_samples: int = 1000,
    head_direction: npt.NDArray[np.floating[Any]] | None = None,
    tuning_width: float = np.pi / 6,  # 30 degrees
    peak_rate: float = 10.0,
    noise_level: float = 0.1,
    sampling_rate: float = 20.0,
    seed: int | None = None,
    plot: bool = True,
) -> tuple[npt.NDArray[np.float64], dict[str, Any]]:
    """Generate head direction cell firing data.

    Head direction cells fire when animal's head points in specific direction.

    Args:
        n_cells: Number of head direction cells.
        n_samples: Number of time points.
        head_direction: Optional head direction trajectory in radians,
            shape (n_samples,).
        tuning_width: Width of directional tuning curve (radians).
        peak_rate: Maximum firing rate in Hz.
        noise_level: Amount of Poisson noise.
        sampling_rate: Sampling rate in Hz (default: 20 Hz for calcium imaging).
            Controls temporal resolution of neural activity.
        seed: Random seed.
        plot: If True, create comprehensive visualization using PlotGrid system.

    Returns:
        activity: Neural activity matrix, shape (n_samples, n_cells).
        metadata: Dictionary with preferred directions.

    Examples:
        >>> activity, meta = generate_head_direction_cells(
        ...     n_cells=60,
        ...     tuning_width=np.pi/4,
        ...     plot=True
        ... )
    """
    rng = np.random.default_rng(seed)

    # Generate or use provided head direction
    if head_direction is None:
        head_direction = generate_head_direction(n_samples, seed=seed)

    # Random preferred directions in [-π, +π]
    preferred_dirs = rng.uniform(-np.pi, np.pi, size=n_cells)

    activity = np.zeros((n_samples, n_cells))

    for i in range(n_cells):
        # Circular difference (handles wrap-around at ±π)
        angle_diff = np.abs(head_direction - preferred_dirs[i])
        angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)

        # Von Mises tuning curve (circular Gaussian)
        kappa = 1 / (tuning_width**2)  # Concentration parameter
        rates = peak_rate * np.exp(kappa * (np.cos(angle_diff) - 1))

        # Add Poisson noise
        if noise_level > 0:
            rates = rng.poisson(rates * noise_level) / noise_level

        activity[:, i] = rates

    metadata = {
        "preferred_directions": preferred_dirs,
        "head_directions": head_direction,
        "cell_type": "head_direction",
        "tuning_width": tuning_width,
        "sampling_rate": sampling_rate,
    }

    # Create visualization if requested
    if plot:
        from neural_analysis.plotting.synthetic_plots import plot_synthetic_data

        plot_synthetic_data(
            activity,
            metadata,
            show_raster=True,
            show_fields=True,
            show_behavior=False,
            show_ground_truth=False,
            show_embeddings=True,
            embedding_methods=["pca", "umap"],
            n_embedding_dims=2,
        )

    return activity, metadata


def generate_random_cells(
    n_cells: int = 50,
    n_samples: int = 1000,
    baseline_rate: float = 2.0,
    variability: float = 1.0,
    temporal_smoothness: float = 0.1,
    sampling_rate: float = 20.0,
    seed: int | None = None,
    plot: bool = True,
    arena_size: tuple[float, float] = (1.0, 1.0),
) -> tuple[npt.NDArray[np.float64], dict[str, Any]]:
    """Generate random cells with no specific tuning properties.

    These cells have random firing patterns without spatial or directional tuning.
    Useful for testing robustness of decoding/embedding methods to noise.

    Args:
        n_cells: Number of random cells.
        n_samples: Number of time points.
        baseline_rate: Mean firing rate in Hz.
        variability: Standard deviation of firing rate fluctuations.
        temporal_smoothness: Temporal correlation (0=white noise, 1=highly smooth).
            Controls how smoothly firing rates change over time.
        sampling_rate: Sampling rate in Hz (default: 20 Hz for calcium imaging).
            Controls temporal resolution of neural activity.
        seed: Random seed for reproducibility.
        plot: If True, create comprehensive visualization using PlotGrid system.
        arena_size: (width, height) of arena for generating synthetic position labels.
            Default: (1.0, 1.0) meters.

    Returns:
        activity: Neural activity matrix, shape (n_samples, n_cells).
            Random firing patterns with temporal smoothness.
        metadata: Dictionary with cell parameters and synthetic position labels.

    Examples:
        >>> # Generate noisy cells
        >>> activity, meta = generate_random_cells(
        ...     n_cells=30,
        ...     baseline_rate=3.0,
        ...     variability=2.0,
        ...     temporal_smoothness=0.2,
        ...     plot=True
        ... )
    """
    rng = np.random.default_rng(seed)

    activity = np.zeros((n_samples, n_cells))

    for i in range(n_cells):
        # Generate random walk for each cell
        # Start with white noise
        noise = rng.normal(0, variability, size=n_samples)

        # Apply temporal smoothing using exponential moving average
        if temporal_smoothness > 0:
            alpha = temporal_smoothness
            smoothed = np.zeros(n_samples)
            smoothed[0] = noise[0]
            for t in range(1, n_samples):
                smoothed[t] = alpha * smoothed[t - 1] + (1 - alpha) * noise[t]
            noise = smoothed

        # Add baseline and ensure non-negative
        rates = baseline_rate + noise
        rates = np.maximum(rates, 0)

        # Add Poisson noise for realism
        rates = rng.poisson(rates).astype(np.float64, copy=False)

        activity[:, i] = rates

    # Generate uniform random 2D positions for structure index compatibility
    # These positions are synthetic and NOT used for tuning (cells are random)
    # They provide uniform coverage of the 2D space for fair comparison
    positions = rng.uniform(
        low=[0, 0], high=[arena_size[0], arena_size[1]], size=(n_samples, 2)
    )

    # Generate uniform random head directions in [-π, +π]
    # Also synthetic and NOT used for tuning (cells are random)
    head_directions = rng.uniform(-np.pi, np.pi, size=n_samples)

    metadata = {
        "cell_type": "random",
        "baseline_rate": baseline_rate,
        "variability": variability,
        "temporal_smoothness": temporal_smoothness,
        "sampling_rate": sampling_rate,
        "positions": positions,  # Synthetic uniform positions
        "head_directions": head_directions,  # Synthetic uniform head directions
        "arena_size": arena_size,
    }

    # Create visualization if requested
    if plot:
        from neural_analysis.plotting.synthetic_plots import plot_synthetic_data

        plot_synthetic_data(
            activity,
            metadata,
            show_raster=True,
            show_fields=True,  # Enable to show diagnostic plots
            show_behavior=True,  # Show position trajectory in last row
            show_ground_truth=False,
            show_embeddings=True,
            embedding_methods=["pca", "umap"],
            n_embedding_dims=2,
        )

    return activity, metadata


def generate_mixed_neural_population(
    n_place: int = 50,
    n_grid: int = 30,
    n_hd: int = 20,
    n_samples: int = 1000,
    arena_size: tuple[float, float] = (2.0, 2.0),
    seed: int | None = None,
) -> tuple[npt.NDArray[np.float64], dict[str, Any]]:
    """Generate mixed population of place, grid, and head direction cells.

    Useful for testing cell type classification and decoding methods.

    Args:
        n_place: Number of place cells.
        n_grid: Number of grid cells.
        n_hd: Number of head direction cells.
        n_samples: Number of time points.
        arena_size: (width, height) of arena.
        seed: Random seed.

    Returns:
        activity: Combined neural activity, shape (n_samples, n_place + n_grid + n_hd).
        metadata: Dictionary with cell type labels and parameters.

    Examples:
        >>> activity, meta = generate_mixed_neural_population(
        ...     n_place=50,
        ...     n_grid=30,
        ...     n_hd=20
        ... )
        >>> cell_types = meta['cell_types']  # Array of 'place', 'grid', 'hd' labels
    """
    np.random.default_rng(seed)

    # Generate common behavioral variables
    positions = generate_position_trajectory(
        n_samples, arena_size=arena_size, seed=seed
    )
    head_direction = generate_head_direction(n_samples, seed=seed)

    # Generate each cell type
    place_activity, place_meta = generate_place_cells(
        n_place, n_samples, positions=positions, arena_size=arena_size, seed=seed
    )

    grid_activity, grid_meta = generate_grid_cells(
        n_grid, n_samples, positions=positions, arena_size=arena_size, seed=seed
    )

    hd_activity, hd_meta = generate_head_direction_cells(
        n_hd, n_samples, head_direction=head_direction, seed=seed
    )

    # Combine activity
    activity = np.column_stack([place_activity, grid_activity, hd_activity])

    # Create cell type labels
    cell_types = ["place"] * n_place + ["grid"] * n_grid + ["head_direction"] * n_hd

    metadata = {
        "cell_types": np.array(cell_types),
        "positions": positions,
        "head_direction": head_direction,
        "arena_size": arena_size,
        "n_place": n_place,
        "n_grid": n_grid,
        "n_hd": n_hd,
        "place_meta": place_meta,
        "grid_meta": grid_meta,
        "hd_meta": hd_meta,
    }

    return activity, metadata


def add_noise(
    data: npt.NDArray[np.floating[Any]],
    noise_type: Literal["gaussian", "poisson", "uniform"] = "gaussian",
    noise_level: float = 0.1,
    seed: int | None = None,
) -> npt.NDArray[np.floating[Any]]:
    """Add noise to data.

    Args:
        data: Input data array.
        noise_type: Type of noise to add ('gaussian', 'poisson', 'uniform').
        noise_level: Amount of noise to add.
        seed: Random seed.

    Returns:
        noisy_data: Data with noise added.

    Examples:
        >>> clean_data = np.random.randn(100, 10)
        >>> noisy_data = add_noise(clean_data, 'gaussian', noise_level=0.5)
    """
    rng = np.random.default_rng(seed)

    if noise_type == "gaussian":
        noise = rng.normal(0, noise_level, size=data.shape)
        return cast("npt.NDArray[np.floating[Any]]", data + noise)

    elif noise_type == "poisson":
        # For Poisson noise, we need positive values
        # Treat data as rates and sample from Poisson distribution
        data_positive = np.maximum(data, 0)  # Ensure non-negative
        # Scale by noise_level (higher = more noise)
        if noise_level > 0:
            noisy = rng.poisson(data_positive / noise_level) * noise_level
        else:
            noisy = data_positive
        return cast("npt.NDArray[np.floating[Any]]", noisy)

    elif noise_type == "uniform":
        noise = rng.uniform(-noise_level, noise_level, size=data.shape)
        return cast("npt.NDArray[np.floating[Any]]", data + noise)

    else:
        raise ValueError(
            f"Unknown noise type. Expected: 'gaussian', 'poisson', or 'uniform'. "
            f"Got: {noise_type!r}"
        )


def map_to_ring(
    activity: npt.NDArray[np.float64],
    positions: npt.NDArray[np.float64],
    plot: bool = True,
) -> npt.NDArray[np.float64]:
    """Map population activity to ring manifold (1D circular).

    For place cells or head direction cells, the underlying manifold
    should be a ring (circle). This function computes the population
    vector angle for visualization.

    Args:
        activity: Neural activity matrix, shape (n_samples, n_cells).
        positions: Position or angle values, shape (n_samples,) or (n_samples, 1).
        plot: If True, create visualization using PlotGrid system showing:
            - Original 1D trajectory (line plot colored by time)
            - Ring embedding colored by time
            - Ring embedding colored by position

    Returns:
        ring_coords: Coordinates on ring, shape (n_samples, 2).
            Columns are [cos(angle), sin(angle)].

    Examples:
        >>> activity, meta = generate_place_cells(50, 1000, arena_size=2.0)
        >>> ring_coords = map_to_ring(activity, meta['positions'], plot=True)
    """
    # Flatten positions if needed
    positions_flat = positions.ravel() if positions.ndim > 1 else positions

    # Normalize positions to [0, 2π]
    pos_min, pos_max = positions_flat.min(), positions_flat.max()
    angles = 2 * np.pi * (positions_flat - pos_min) / (pos_max - pos_min)

    # Map to ring
    ring_coords = np.column_stack([np.cos(angles), np.sin(angles)])

    # Create visualization if requested
    if plot:
        from neural_analysis.plotting import (
            GridLayoutConfig,
            PlotConfig,
            PlotSpec,
        )
        from neural_analysis.plotting.grid_dispatch import PlotGrid

        plot_specs = []

        # 1. Original 1D trajectory (line plot with time on x-axis)
        time_array = np.arange(len(positions_flat))
        spec1 = PlotSpec(
            data={"x": time_array, "y": positions_flat},
            plot_type="line",
            subplot_position=0,
            title="1D Position Trajectory",
            color="#3498DB",
            line_width=1.5,
            alpha=0.8,
            kwargs={
                "x_label": "Time (samples)",
                "y_label": "Position (m)",
            },
        )
        plot_specs.append(spec1)

        # 2. Ring embedding colored by time
        spec2 = PlotSpec(
            data={"x": ring_coords[:, 0], "y": ring_coords[:, 1]},
            plot_type="scatter",
            subplot_position=1,
            title="Ring Embedding (S¹) - Colored by Time",
            color_by=cast("Any", time_array),
            cmap="viridis",
            marker_size=10,
            alpha=0.7,
            colorbar=True,
            colorbar_label="Time (samples)",
            equal_aspect=True,
            kwargs={
                "x_label": "cos(θ)",
                "y_label": "sin(θ)",
            },
        )
        plot_specs.append(spec2)

        # 3. Ring embedding colored by position
        spec3 = PlotSpec(
            data={"x": ring_coords[:, 0], "y": ring_coords[:, 1]},
            plot_type="scatter",
            subplot_position=2,
            title="Ring Embedding - Colored by Position",
            color_by=cast("Any", positions_flat),
            cmap="plasma",
            marker_size=10,
            alpha=0.7,
            colorbar=True,
            colorbar_label="Position (m)",
            equal_aspect=True,
            kwargs={
                "x_label": "cos(θ)",
                "y_label": "sin(θ)",
            },
        )
        plot_specs.append(spec3)

        # Create grid with uneven widths: first subplot (1D trajectory) is narrower
        # Width ratios: [1, 2, 2] makes first column half width of ring embeddings
        grid = PlotGrid(
            plot_specs=plot_specs,
            config=PlotConfig(figsize=(15, 5)),
            layout=GridLayoutConfig(rows=1, cols=3, width_ratios=[1, 2, 2]),
            backend="matplotlib",
        )

        grid.plot()

    return ring_coords


def map_to_torus(
    activity: npt.NDArray[np.float64],
    positions: npt.NDArray[np.float64],
    major_radius: float = 2.0,
    minor_radius: float = 1.0,
    plot: bool = True,
) -> npt.NDArray[np.float64]:
    """Map population activity to torus manifold (2D periodic).

    For 2D grid cells, the underlying manifold should be a torus.
    This function maps 2D positions to 3D torus coordinates.

    Args:
        activity: Neural activity matrix, shape (n_samples, n_cells).
        positions: 2D positions, shape (n_samples, 2).
        major_radius: Major radius of torus (distance from center to tube center).
        minor_radius: Minor radius of torus (tube radius).
        plot: If True, create visualization using PlotGrid system showing:
            - Original 2D trajectory
            - Torus embedding colored by time
            - Torus embedding colored by X position

    Returns:
        torus_coords: Coordinates on torus, shape (n_samples, 3).
            3D embedding of the 2D periodic space.

    Examples:
        >>> activity, meta = generate_grid_cells(30, 1000, arena_size=(2.0, 2.0))
        >>> torus_coords = map_to_torus(activity, meta['positions'], plot=True)
    """
    if positions.shape[1] != 2:
        raise ValueError("Positions must be 2D for torus mapping")

    # Normalize positions to [0, 2π] for each dimension
    pos_min = positions.min(axis=0)
    pos_max = positions.max(axis=0)
    theta = 2 * np.pi * (positions[:, 0] - pos_min[0]) / (pos_max[0] - pos_min[0])
    phi = 2 * np.pi * (positions[:, 1] - pos_min[1]) / (pos_max[1] - pos_min[1])

    # Map to torus
    x = (major_radius + minor_radius * np.cos(phi)) * np.cos(theta)
    y = (major_radius + minor_radius * np.cos(phi)) * np.sin(theta)
    z = minor_radius * np.sin(phi)

    torus_coords = np.column_stack([x, y, z])

    # Create visualization if requested
    if plot:
        from neural_analysis.plotting import (
            GridLayoutConfig,
            PlotConfig,
            PlotSpec,
        )
        from neural_analysis.plotting.grid_dispatch import PlotGrid

        plot_specs = []
        time_array = np.arange(len(positions))

        # 1. Original 2D trajectory
        spec1 = PlotSpec(
            data={"x": positions[:, 0], "y": positions[:, 1]},
            plot_type="trajectory",
            subplot_position=0,
            title="2D Position Trajectory",
            color_by=cast("Any", time_array),
            cmap="viridis",
            marker_size=5,
            alpha=0.7,
            colorbar=True,
            colorbar_label="Time (samples)",
            equal_aspect=True,
            kwargs={
                "x_label": "X Position (m)",
                "y_label": "Y Position (m)",
            },
        )
        plot_specs.append(spec1)

        # 2. Torus embedding colored by time (3D)
        spec2 = PlotSpec(
            data={
                "x": torus_coords[:, 0],
                "y": torus_coords[:, 1],
                "z": torus_coords[:, 2],
            },
            plot_type="scatter3d",
            subplot_position=1,
            title="Torus Embedding (T²) - Colored by Time",
            color_by=cast("Any", time_array),
            cmap="viridis",
            marker_size=5,
            alpha=0.7,
            colorbar=True,
            colorbar_label="Time (samples)",
            kwargs={
                "x_label": "X",
                "y_label": "Y",
                "z_label": "Z",
            },
        )
        plot_specs.append(spec2)

        # 3. Torus embedding colored by X position (3D)
        spec3 = PlotSpec(
            data={
                "x": torus_coords[:, 0],
                "y": torus_coords[:, 1],
                "z": torus_coords[:, 2],
            },
            plot_type="scatter3d",
            subplot_position=2,
            title="Torus - Colored by X Position",
            color_by=cast("Any", positions[:, 0]),
            cmap="plasma",
            marker_size=5,
            alpha=0.7,
            colorbar=True,
            colorbar_label="X Position (m)",
            kwargs={
                "x_label": "X",
                "y_label": "Y",
                "z_label": "Z",
            },
        )
        plot_specs.append(spec3)

        # Create grid
        grid = PlotGrid(
            plot_specs=plot_specs,
            config=PlotConfig(figsize=(16, 5)),
            layout=GridLayoutConfig(rows=1, cols=3),
            backend="matplotlib",
        )

        grid.plot()

    return torus_coords


def generate_mixed_population_flexible(
    cell_config: dict[str, dict[str, Any]] | None = None,
    n_samples: int = 1000,
    arena_size: float | tuple[float, ...] = (2.0, 2.0),
    seed: int | None = None,
    plot: bool = True,
) -> tuple[npt.NDArray[np.float64], dict[str, Any]]:
    """Generate flexible mixed neural population with custom configuration.

    Args:
        cell_config: Dictionary specifying cell types and parameters. Format:
            {
                'place': {'n_cells': 50, 'field_size': 0.2, 'noise_level': 0.1},
                'grid': {'n_cells': 30, 'grid_spacing': 0.4, 'noise_level': 0.05},
                'head_direction': {
                    'n_cells': 20, 'tuning_width': np.pi/6, 'noise_level': 0.1
                },
                'random': {'n_cells': 15, 'baseline_rate': 2.0, 'variability': 1.0},
            }
            If None, uses default configuration with all cell types.
        n_samples: Number of time points.
        arena_size: Size of arena (float for 1D, tuple for 2D/3D).
        seed: Random seed for reproducibility.
        plot: If True, create comprehensive visualization using PlotGrid system.

    Returns:
        activity: Combined neural activity, shape (n_samples, total_cells).
        metadata: Dictionary with comprehensive metadata including:
            - 'cell_types': Array of cell type labels
            - 'cell_indices': Dict mapping cell type to indices
            - 'positions': Position trajectory
            - 'head_direction': Head direction trajectory (if applicable)
            - Individual metadata for each cell type

    Examples:
        >>> # Use default configuration with automatic plotting
        >>> activity, meta = generate_mixed_population_flexible(
        ...     n_samples=1500, seed=42, plot=True
        ... )

        >>> # Custom configuration without plotting
        >>> config = {
        ...     'place': {'n_cells': 50, 'field_size': 0.3, 'noise_level': 0.1},
        ...     'grid': {'n_cells': 30, 'grid_spacing': 0.5, 'noise_level': 0.05},
        ...     'head_direction': {'n_cells': 20, 'noise_level': 0.1},
        ...     'random': {'n_cells': 15, 'baseline_rate': 3.0},
        ... }
        >>> activity, meta = generate_mixed_population_flexible(
        ...     config, n_samples=2000, plot=False
        ... )
        >>> # Access specific cell types
        >>> place_indices = meta['cell_indices']['place']
        >>> place_activity = activity[:, place_indices]
    """
    np.random.default_rng(seed)

    # Default configuration if none provided
    if cell_config is None:
        cell_config = {
            "place": {"n_cells": 50, "field_size": 0.2, "noise_level": 0.08},
            "grid": {"n_cells": 30, "grid_spacing": 0.4, "noise_level": 0.05},
            "head_direction": {
                "n_cells": 25,
                "tuning_width": np.pi / 6,
                "noise_level": 0.1,
            },
            "random": {"n_cells": 20, "baseline_rate": 2.0, "variability": 1.5},
        }

    # Generate common behavioral variables
    positions = generate_position_trajectory(
        n_samples, arena_size=arena_size, seed=seed
    )

    # Determine if we need head direction
    needs_hd = "head_direction" in cell_config
    head_direction = generate_head_direction(n_samples, seed=seed) if needs_hd else None

    # Generate each cell type
    all_activity = []
    all_cell_types = []
    cell_indices = {}
    cell_metadata = {}
    current_idx = 0

    for cell_type, params in cell_config.items():
        n_cells = params.pop("n_cells", 10)

        if cell_type == "place":
            activity, meta = generate_place_cells(
                n_cells=n_cells,
                n_samples=n_samples,
                positions=positions,
                arena_size=arena_size,
                seed=seed,
                plot=False,  # Disable individual plots for mixed population
                **params,
            )

        elif cell_type == "grid":
            activity, meta = generate_grid_cells(
                n_cells=n_cells,
                n_samples=n_samples,
                positions=positions,
                arena_size=arena_size,
                seed=seed,
                plot=False,  # Disable individual plots for mixed population
                **params,
            )

        elif cell_type in ["head_direction", "hd"]:
            activity, meta = generate_head_direction_cells(
                n_cells=n_cells,
                n_samples=n_samples,
                head_direction=head_direction,
                seed=seed,
                plot=False,  # Disable individual plots for mixed population
                **params,
            )

        elif cell_type == "random":
            activity, meta = generate_random_cells(
                n_cells=n_cells,
                n_samples=n_samples,
                seed=seed,
                plot=False,  # Disable individual plots for mixed population
                **params,
            )

        else:
            raise ValueError(
                f"Unknown cell type. Expected: 'place', 'grid', "
                f"'head_direction', 'hd', or 'random'. Got: {cell_type!r}"
            )

        # Store activity and metadata
        all_activity.append(activity)
        all_cell_types.extend([cell_type] * n_cells)
        cell_indices[cell_type] = list(range(current_idx, current_idx + n_cells))
        cell_metadata[cell_type] = meta
        current_idx += n_cells

        # Restore n_cells to params
        params["n_cells"] = n_cells

    # Combine all activities
    combined_activity = np.column_stack(all_activity)

    # Merge cell-specific metadata arrays into single arrays
    # Initialize with None for all cells
    n_total_cells = combined_activity.shape[1]
    preferred_directions = np.full(n_total_cells, np.nan)

    # Fill in metadata for each cell type
    for cell_type, idxs in cell_indices.items():
        meta = cell_metadata[cell_type]

        # Handle preferred_directions (for head direction cells)
        if "preferred_directions" in meta and meta["preferred_directions"] is not None:
            preferred_directions[idxs] = meta["preferred_directions"]

    # Create comprehensive metadata
    metadata = {
        "cell_types": np.array(all_cell_types),
        "cell_indices": cell_indices,
        "cell_config": cell_config,
        "positions": positions,
        "head_direction": head_direction,
        "head_directions": head_direction,  # Add plural form for compatibility
        "arena_size": arena_size,
        "n_samples": n_samples,
        "n_dims": 2,  # Mixed populations currently only support 2D
        "preferred_directions": preferred_directions,
        "individual_metadata": cell_metadata,
    }

    # Create visualization if requested
    if plot:
        from neural_analysis.plotting.synthetic_plots import plot_synthetic_data

        plot_synthetic_data(
            combined_activity,
            metadata,
            show_raster=True,
            show_fields=True,
            show_behavior=True,
            show_ground_truth=False,
            show_embeddings=True,
            embedding_methods=["pca", "umap"],
            n_embedding_dims=2,
        )

    return combined_activity, metadata


def generate_cluster_templates(
    n_clusters: int,
    n_features: int,
    cluster_separation: float = 15.0,
    seed: int | None = None,
) -> npt.NDArray[np.float64]:
    """Generate cluster templates with fundamentally different structures.

    Creates distinct cluster templates where each cluster has completely
    different patterns (frequency structures, sparsity patterns, block structures)
    that create shape differences surviving rotation/reflection/translation/scale
    normalization. This is essential for validating shape distance metrics.

    Parameters
    ----------
    n_clusters : int
        Number of distinct clusters to generate.
    n_features : int
        Number of features (dimensions) in the feature space.
    cluster_separation : float, default=15.0
        Separation factor controlling how distinct clusters are.
        Higher values create more separated clusters.
    seed : int or None, default=None
        Random seed for reproducibility.

    Returns
    -------
    templates : ndarray of shape (n_clusters, n_features)
        Cluster templates, each row is a distinct template pattern.

    Examples
    --------
    >>> templates = generate_cluster_templates(n_clusters=5, n_features=100)
    >>> # Each template has fundamentally different structure
    >>> # templates.shape = (5, 100)
    """
    templates = []

    for k in range(n_clusters):
        cluster_rng = np.random.default_rng((seed or 0) + k * 1000)

        # Create completely independent patterns (no shared base)
        # Each cluster is fundamentally different
        pattern = np.zeros(n_features)

        # Method 1: Different frequency structures
        # Each cluster uses different frequency combinations
        for freq_mult in range(1, 6):  # Multiple frequencies
            freq = freq_mult * (k + 1)  # Cluster-specific frequencies
            if freq < n_features / 2:
                x = np.linspace(0, 4 * np.pi, n_features)
                phase = cluster_rng.uniform(0, 2 * np.pi)
                amplitude = cluster_separation / freq_mult
                pattern += amplitude * np.sin(freq * x + phase)

        # Method 2: Different sparsity patterns
        # Each cluster has different important features
        n_active = n_features // (2 + k)  # Different sparsity per cluster
        active_indices = cluster_rng.choice(n_features, size=n_active, replace=False)
        active_values = cluster_rng.normal(0, cluster_separation, size=n_active)
        pattern[active_indices] += active_values

        # Method 3: Different block structures
        # Each cluster has different feature groupings
        n_blocks = 3 + k
        block_size = n_features // n_blocks
        for b in range(n_blocks):
            start = b * block_size
            end = min((b + 1) * block_size, n_features)
            # Different block values per cluster
            block_val = cluster_rng.normal(0, cluster_separation * 0.7)
            pattern[start:end] += block_val

        templates.append(pattern)

    return np.stack(templates, axis=0).astype(np.float64)


def generate_dataset_from_cluster_template(
    template: npt.NDArray[np.float64],
    n_neurons: int,
    noise_scale: float = 0.1,
    cluster_id: int = 0,
    seed: int | None = None,
) -> npt.NDArray[np.float64]:
    """Generate a neural dataset from a cluster template.

    Creates datasets with cluster-specific point arrangements that create
    shape differences surviving rotation/reflection/translation/scale normalization.
    Different clusters have different point cloud geometries (linear, curved, spread).

    Parameters
    ----------
    template : ndarray of shape (n_features,)
        Cluster template pattern to use as base.
    n_neurons : int
        Number of neurons (rows) in the generated dataset.
    noise_scale : float, default=0.1
        Standard deviation of noise to add.
    cluster_id : int, default=0
        Cluster identifier determining the geometric structure type.
        Different cluster_ids create different intrinsic dimensionalities and
        point arrangements.
    seed : int or None, default=None
        Random seed for reproducibility.

    Returns
    -------
    dataset : ndarray of shape (n_neurons, n_features)
        Generated neural dataset with cluster-specific geometry.

    Notes
    -----
    The function creates different geometric structures per cluster:
    - cluster_id % 3 == 0: Points along a curve (1D manifold)
    - cluster_id % 3 == 1: Points in a plane (2D manifold)
    - cluster_id % 3 == 2: Spread distribution (higher intrinsic dim)

    Examples
    --------
    >>> template = generate_cluster_templates(n_clusters=1, n_features=100)[0]
    >>> dataset = generate_dataset_from_cluster_template(
    ...     template, n_neurons=50, cluster_id=0
    ... )
    >>> # dataset.shape = (50, 100)
    """
    rng = np.random.default_rng(seed)
    n_features = template.shape[0]
    cluster_rng = np.random.default_rng((seed or 0) + cluster_id * 1000)

    X = np.zeros((n_neurons, n_features))

    # Create different point arrangements per cluster
    # This creates geometric shape differences

    # Different intrinsic dimensionality per cluster
    n_intrinsic = max(3, min(12, n_features // (2 + cluster_id)))

    # Generate points with cluster-specific geometry
    # Initialize intrinsic with correct shape and dtype
    intrinsic: npt.NDArray[np.float64] = np.zeros(
        (n_neurons, n_intrinsic), dtype=np.float64
    )

    if cluster_id % 3 == 0:
        # Type 0: Points along a curve (1D manifold embedded in high-D)
        t = np.linspace(0, 2 * np.pi, n_neurons)
        for dim in range(n_intrinsic):
            intrinsic[:, dim] = np.sin((dim + 1) * t).astype(np.float64)
    elif cluster_id % 3 == 1:
        # Type 1: Points in a plane (2D manifold)
        t1 = np.linspace(0, 2 * np.pi, int(np.sqrt(n_neurons)))
        t2 = np.linspace(0, 2 * np.pi, int(np.sqrt(n_neurons)))
        T1, T2 = np.meshgrid(t1, t2)
        if n_intrinsic >= 2:
            intrinsic[: len(T1.ravel()), 0] = T1.ravel()[:n_neurons].astype(np.float64)
            intrinsic[: len(T2.ravel()), 1] = T2.ravel()[:n_neurons].astype(np.float64)
        # Fill remaining with small values
        if n_intrinsic > 2:
            intrinsic[:, 2:] = cluster_rng.normal(
                0, 0.1, size=(n_neurons, n_intrinsic - 2)
            ).astype(np.float64)
    else:
        # Type 2: More spread distribution (higher intrinsic dim)
        intrinsic = cluster_rng.normal(0, 1, size=(n_neurons, n_intrinsic)).astype(
            np.float64
        )

    # Create cluster-specific embedding
    from scipy.linalg import svd

    embedding = cluster_rng.normal(0, 1, size=(n_intrinsic, n_features))
    U, s, Vt = svd(embedding, full_matrices=False)
    embedding_ortho = U @ Vt

    # Embed to feature space
    X_embedded = intrinsic @ embedding_ortho

    # Add template with appropriate scaling
    template_norm = np.linalg.norm(template)
    embedded_norm = (
        np.linalg.norm(X_embedded) / np.sqrt(n_neurons) if n_neurons > 0 else 1
    )
    if template_norm > 0 and embedded_norm > 0:
        scale = embedded_norm / template_norm
        X = X_embedded + template[None, :] * scale
    else:
        X = X_embedded + template[None, :]

    # Add small noise
    noise = rng.normal(0, noise_scale, size=(n_neurons, n_features))
    X = X + noise

    # Ensure float64 dtype explicitly
    result: npt.NDArray[np.float64] = X.astype(np.float64)
    return result


def generate_shape_distance_datasets(
    n_datasets: int = 100,
    n_clusters: int = 5,
    min_neurons: int = 50,
    max_neurons: int = 200,
    n_features: int = 300,
    cluster_separation: float = 15.0,
    noise_scale: float = 0.1,
    seed: int | None = None,
) -> tuple[list[npt.NDArray[np.float64]], npt.NDArray[np.int_]]:
    """Generate multiple neural datasets with distinct cluster structure.

    Generates K neural datasets, each with different numbers of neurons but
    belonging to one of n_clusters. Each cluster has fundamentally different
    structure that survives rotation/reflection/translation/scale normalization,
    making it suitable for validating shape distance metrics.

    Parameters
    ----------
    n_datasets : int, default=100
        Total number of datasets to generate.
    n_clusters : int, default=5
        Number of distinct clusters (each dataset belongs to one cluster).
    min_neurons : int, default=50
        Minimum number of neurons per dataset.
    max_neurons : int, default=200
        Maximum number of neurons per dataset.
    n_features : int, default=300
        Number of features (dimensions) per dataset.
    cluster_separation : float, default=15.0
        Separation factor for cluster templates (higher = more distinct).
    noise_scale : float, default=0.1
        Noise level added to each dataset.
    seed : int or None, default=None
        Random seed for reproducibility.

    Returns
    -------
    datasets : list of ndarray
        List of K datasets, each of shape (n_neurons_i, n_features).
        Each dataset has a different number of neurons.
    labels : ndarray of shape (n_datasets,)
        Cluster labels for each dataset (0 to n_clusters-1).

    Examples
    --------
    >>> datasets, labels = generate_shape_distance_datasets(
    ...     n_datasets=50,
    ...     n_clusters=5,
    ...     min_neurons=30,
    ...     max_neurons=80,
    ...     n_features=100
    ... )
    >>> # datasets[0].shape might be (45, 100)
    >>> # datasets[1].shape might be (67, 100)
    >>> # labels indicates which cluster each dataset belongs to
    """
    rng = np.random.default_rng(seed if seed is not None else 1)
    templates = generate_cluster_templates(
        n_clusters, n_features, cluster_separation=cluster_separation, seed=seed
    )
    datasets: list[npt.NDArray[np.float64]] = []
    labels = np.zeros(n_datasets, dtype=int)

    for i in range(n_datasets):
        cluster_id = i % n_clusters
        labels[i] = cluster_id
        template = templates[cluster_id]
        n_neurons = int(rng.integers(min_neurons, max_neurons + 1))
        X = generate_dataset_from_cluster_template(
            template,
            n_neurons,
            noise_scale=noise_scale,
            cluster_id=cluster_id,
            seed=seed,
        )
        datasets.append(X)

    return datasets, labels
