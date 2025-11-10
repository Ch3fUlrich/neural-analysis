"""Synthetic dataset generators for testing neural analysis methods.

This module provides functions to generate various types of synthetic data:
- Manifold data (swiss roll, s-curve, circles, moons, blobs, etc.)
- Neural data (place cells, grid cells, head direction cells)
- Behavioral data (position, velocity, head direction)
- Noisy versions of all data types

The main entry point is `generate_data()` which provides a unified interface
for all dataset types with consistent parameters and outputs.

These datasets are useful for:
- Testing dimensionality reduction algorithms
- Validating decoding methods
- Benchmarking neural analysis pipelines
- Creating reproducible examples and tutorials

Examples:
    Generate Swiss roll manifold:
        >>> data, labels = generate_data('swiss_roll', n_samples=1000, noise=0.1)

    Generate place cell data with behavioral labels:
        >>> data, labels = generate_data(
        ...     'place_cells',
        ...     n_samples=1000,
        ...     n_features=50,
        ...     noise=0.1
        ... )
        >>> # labels contains position trajectory

    Generate grid cells with position labels:
        >>> data, labels = generate_data('grid_cells', n_samples=1000, n_features=30)

    Generate classification dataset:
        >>> data, labels = generate_data('blobs', n_samples=500, n_features=10, n_classes=3)
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from sklearn.datasets import (
    make_blobs,
    make_circles,
    make_classification,
    make_moons,
    make_regression,
    make_s_curve,
    make_swiss_roll,
)

# Type aliases
DatasetType = Literal[
    'swiss_roll', 's_curve', 'blobs', 'moons', 'circles', 
    'classification', 'regression',
    'place_cells', 'grid_cells', 'head_direction_cells', 'mixed_cells',
    'position_trajectory', 'head_direction'
]


def generate_data(
    dataset_type: DatasetType,
    n_samples: int = 1000,
    n_features: int | None = None,
    n_classes: int | None = None,
    noise: float = 0.0,
    seed: int | None = None,
    **kwargs: Any,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64] | dict[str, Any]]:
    """Generate synthetic datasets with unified interface.

    Main orchestrator function that provides access to all dataset types
    (sklearn manifolds, classification data, and custom neural data) through
    a single standardized interface.

    Args:
        dataset_type: Type of dataset to generate. Options:
            Manifolds: 'swiss_roll', 's_curve'
            Classification: 'blobs', 'moons', 'circles', 'classification'
            Regression: 'regression'
            Neural: 'place_cells', 'grid_cells', 'head_direction_cells', 'mixed_cells'
            Behavioral: 'position_trajectory', 'head_direction'
        n_samples: Number of data points to generate.
        n_features: Number of features/dimensions. For neural data, this is
            the number of cells. For manifolds, this parameter may be ignored
            (they have fixed dimensions).
        n_classes: Number of classes/clusters (for classification datasets).
        noise: Noise level (interpretation depends on dataset type).
        seed: Random seed for reproducibility.
        **kwargs: Additional dataset-specific parameters.

    Returns:
        data: Generated data array, shape (n_samples, n_features).
        labels: Labels/ground truth for the data. For:
            - Manifolds: continuous values for coloring (1D array)
            - Classification: class labels (1D array)
            - Neural data: behavioral variables (dict with positions, angles, etc.)

    Examples:
        >>> # Manifold for dimensionality reduction
        >>> data, labels = generate_data('swiss_roll', n_samples=1000, noise=0.1)
        >>> # data.shape = (1000, 3), labels.shape = (1000,)

        >>> # Classification dataset
        >>> data, labels = generate_data('blobs', n_samples=500, n_classes=3)

        >>> # Place cells with position labels
        >>> activity, meta = generate_data('place_cells', n_samples=1000, n_features=50)
        >>> positions = meta['positions']  # (1000, 2) trajectory
        >>> # Use positions for coloring/analysis

        >>> # Mixed neural population
        >>> activity, meta = generate_data(
        ...     'mixed_cells',
        ...     n_samples=2000,
        ...     n_place=50,
        ...     n_grid=30,
        ...     n_hd=20
        ... )
        >>> cell_types = meta['cell_types']  # ['place', 'grid', 'head_direction']
        >>> positions = meta['positions']
    """
    # Normalize dataset type
    dataset_type = dataset_type.lower()

    # Route to appropriate generator
    if dataset_type == 'swiss_roll':
        return _generate_swiss_roll(n_samples, noise, seed)
    
    elif dataset_type == 's_curve':
        return _generate_s_curve(n_samples, noise, seed)
    
    elif dataset_type == 'blobs':
        n_features = n_features or 2
        n_classes = n_classes or 3
        return _generate_blobs(n_samples, n_features, n_classes, noise, seed, **kwargs)
    
    elif dataset_type == 'moons':
        return _generate_moons(n_samples, noise, seed)
    
    elif dataset_type == 'circles':
        return _generate_circles(n_samples, noise, seed, **kwargs)
    
    elif dataset_type == 'classification':
        n_features = n_features or 20
        n_classes = n_classes or 2
        return _generate_classification(n_samples, n_features, n_classes, noise, seed, **kwargs)
    
    elif dataset_type == 'regression':
        n_features = n_features or 10
        return _generate_regression(n_samples, n_features, noise, seed, **kwargs)
    
    elif dataset_type == 'place_cells':
        n_features = n_features or 100  # Number of cells
        return _generate_place_cells(n_samples, n_features, noise, seed, **kwargs)
    
    elif dataset_type == 'grid_cells':
        n_features = n_features or 50
        return _generate_grid_cells(n_samples, n_features, noise, seed, **kwargs)
    
    elif dataset_type == 'head_direction_cells':
        n_features = n_features or 60
        return _generate_head_direction_cells(n_samples, n_features, noise, seed, **kwargs)
    
    elif dataset_type == 'mixed_cells':
        return _generate_mixed_cells(n_samples, seed, **kwargs)
    
    elif dataset_type == 'position_trajectory':
        return _generate_position_trajectory(n_samples, seed, **kwargs)
    
    elif dataset_type == 'head_direction':
        return _generate_head_direction(n_samples, seed, **kwargs)
    
    else:
        raise ValueError(
            f"Unknown dataset type: {dataset_type}. "
            f"Available types: swiss_roll, s_curve, blobs, moons, circles, "
            f"classification, regression, place_cells, grid_cells, "
            f"head_direction_cells, mixed_cells, position_trajectory, head_direction"
        )


# ============================================================================
# Manifold Datasets (sklearn wrappers)
# ============================================================================

def _generate_swiss_roll(
    n_samples: int,
    noise: float,
    seed: int | None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Generate Swiss roll manifold using scikit-learn."""
    x, t = make_swiss_roll(n_samples=n_samples, noise=noise, random_state=seed)
    return x, t


def _generate_s_curve(
    n_samples: int,
    noise: float,
    seed: int | None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Generate S-curve manifold using scikit-learn."""
    X, t = make_s_curve(n_samples=n_samples, noise=noise, random_state=seed)
    return X, t


def _generate_blobs(
    n_samples: int,
    n_features: int,
    n_centers: int,
    noise: float,
    seed: int | None,
    **kwargs: Any,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """Generate isotropic Gaussian blobs using scikit-learn."""
    cluster_std = kwargs.get('cluster_std', 1.0 + noise)
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_centers,
        cluster_std=cluster_std,
        random_state=seed,
    )
    return X, y


def _generate_moons(
    n_samples: int,
    noise: float,
    seed: int | None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """Generate two interleaving half circles using scikit-learn."""
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    return X, y


def _generate_circles(
    n_samples: int,
    noise: float,
    seed: int | None,
    **kwargs: Any,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """Generate large circle containing smaller circle using scikit-learn."""
    factor = kwargs.get('factor', 0.5)
    X, y = make_circles(
        n_samples=n_samples,
        noise=noise,
        factor=factor,
        random_state=seed,
    )
    return X, y


def _generate_classification(
    n_samples: int,
    n_features: int,
    n_classes: int,
    noise: float,
    seed: int | None,
    **kwargs: Any,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """Generate random classification problem using scikit-learn."""
    n_informative = kwargs.get('n_informative', min(n_features, 2 * n_classes))
    n_redundant = kwargs.get('n_redundant', 0)
    
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_classes=n_classes,
        flip_y=noise,
        random_state=seed,
    )
    return X, y


def _generate_regression(
    n_samples: int,
    n_features: int,
    noise: float,
    seed: int | None,
    **kwargs: Any,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Generate random regression problem using scikit-learn."""
    n_informative = kwargs.get('n_informative', min(n_features, 10))
    
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        noise=noise,
        random_state=seed,
    )
    return X, y


# ============================================================================
# Behavioral Data Generator Wrappers
# ============================================================================

def _generate_position_trajectory(
    n_samples: int,
    seed: int | None,
    **kwargs: Any,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Generate position trajectory - returns (positions, positions) for label compatibility."""
    arena_size = kwargs.get('arena_size', (1.0, 1.0))
    speed = kwargs.get('speed', 0.1)
    turning_rate = kwargs.get('turning_rate', 0.3)
    
    positions = generate_position_trajectory(
        n_samples=n_samples,
        arena_size=arena_size,
        speed=speed,
        turning_rate=turning_rate,
        seed=seed,
    )
    return positions, positions  # Return positions as both data and labels


def _generate_head_direction(
    n_samples: int,
    seed: int | None,
    **kwargs: Any,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Generate head direction - returns (angles, angles) for label compatibility."""
    turning_rate = kwargs.get('turning_rate', 0.1)
    
    angles = generate_head_direction(
        n_samples=n_samples,
        turning_rate=turning_rate,
        seed=seed,
    )
    # Return as column vector for data, 1D for labels
    return angles.reshape(-1, 1), angles


# ============================================================================
# Neural Data Generator Wrappers
# ============================================================================

def _generate_place_cells(
    n_samples: int,
    n_features: int,
    noise: float,
    seed: int | None,
    **kwargs: Any,
) -> tuple[npt.NDArray[np.float64], dict[str, Any]]:
    """Generate place cells - returns (activity, metadata with positions)."""
    arena_size = kwargs.get('arena_size', (1.0, 1.0))
    field_size = kwargs.get('field_size', 0.2)
    peak_rate = kwargs.get('peak_rate', 10.0)
    sampling_rate = kwargs.get('sampling_rate', 20.0)
    positions = kwargs.get('positions')
    
    activity, metadata = generate_place_cells(
        n_cells=n_features,
        n_samples=n_samples,
        positions=positions,
        arena_size=arena_size,
        field_size=field_size,
        peak_rate=peak_rate,
        noise_level=noise,
        sampling_rate=sampling_rate,
        seed=seed,
    )
    return activity, metadata


def _generate_grid_cells(
    n_samples: int,
    n_features: int,
    noise: float,
    seed: int | None,
    **kwargs: Any,
) -> tuple[npt.NDArray[np.float64], dict[str, Any]]:
    """Generate grid cells - returns (activity, metadata with positions)."""
    arena_size = kwargs.get('arena_size', (2.0, 2.0))
    grid_spacing = kwargs.get('grid_spacing', 0.4)
    grid_orientation = kwargs.get('grid_orientation', 0.0)
    peak_rate = kwargs.get('peak_rate', 10.0)
    sampling_rate = kwargs.get('sampling_rate', 20.0)
    positions = kwargs.get('positions')
    
    activity, metadata = generate_grid_cells(
        n_cells=n_features,
        n_samples=n_samples,
        positions=positions,
        arena_size=arena_size,
        grid_spacing=grid_spacing,
        grid_orientation=grid_orientation,
        peak_rate=peak_rate,
        noise_level=noise,
        sampling_rate=sampling_rate,
        seed=seed,
    )
    return activity, metadata


def _generate_head_direction_cells(
    n_samples: int,
    n_features: int,
    noise: float,
    seed: int | None,
    **kwargs: Any,
) -> tuple[npt.NDArray[np.float64], dict[str, Any]]:
    """Generate head direction cells - returns (activity, metadata with angles)."""
    tuning_width = kwargs.get('tuning_width', np.pi / 6)
    peak_rate = kwargs.get('peak_rate', 10.0)
    sampling_rate = kwargs.get('sampling_rate', 20.0)
    head_direction = kwargs.get('head_direction')
    
    activity, metadata = generate_head_direction_cells(
        n_cells=n_features,
        n_samples=n_samples,
        head_direction=head_direction,
        tuning_width=tuning_width,
        peak_rate=peak_rate,
        noise_level=noise,
        sampling_rate=sampling_rate,
        seed=seed,
    )
    return activity, metadata


def _generate_mixed_cells(
    n_samples: int,
    seed: int | None,
    **kwargs: Any,
) -> tuple[npt.NDArray[np.float64], dict[str, Any]]:
    """Generate mixed population - returns (activity, metadata)."""
    n_place = kwargs.get('n_place', 50)
    n_grid = kwargs.get('n_grid', 30)
    n_hd = kwargs.get('n_hd', 20)
    arena_size = kwargs.get('arena_size', (2.0, 2.0))
    
    activity, metadata = generate_mixed_neural_population(
        n_place=n_place,
        n_grid=n_grid,
        n_hd=n_hd,
        n_samples=n_samples,
        arena_size=arena_size,
        seed=seed,
    )
    return activity, metadata


# ============================================================================
# Core Behavioral Data Generators (Public API)
# ============================================================================

def _generate_smooth_speeds(
    n_samples: int,
    speed_range: tuple[float, float],
    rng: np.random.Generator,
) -> npt.NDArray[np.float64]:
    """Generate smooth speed profile using Ornstein-Uhlenbeck process.
    
    Creates smooth, continuous speed variations within specified range.
    
    Args:
        n_samples: Number of time points.
        speed_range: Tuple of (min_speed, max_speed).
        rng: Random number generator.
        
    Returns:
        speeds: Array of speeds, shape (n_samples,).
    """
    min_speed, max_speed = speed_range
    
    # Initialize with random speed
    current_speed = rng.uniform(min_speed, max_speed)
    speeds = np.zeros(n_samples)
    speeds[0] = current_speed
    
    # Parameters for Ornstein-Uhlenbeck process
    speed_mean = (min_speed + max_speed) / 2
    speed_tau = 0.98  # Temporal correlation (high = smooth changes)
    speed_sigma = (max_speed - min_speed) / 6  # Noise level
    
    for i in range(1, n_samples):
        # Ornstein-Uhlenbeck process for smooth speed changes
        current_speed = (
            speed_tau * current_speed + 
            (1 - speed_tau) * speed_mean + 
            speed_sigma * rng.normal()
        )
        # Clip to valid range
        current_speed = np.clip(current_speed, min_speed, max_speed)
        speeds[i] = current_speed
    
    return speeds


def _trajectory_1d(
    n_samples: int,
    arena_size: tuple[float, ...],
    speeds: npt.NDArray[np.float64],
    turning_rate: float,
    rng: np.random.Generator,
) -> npt.NDArray[np.float64]:
    """Generate 1D trajectory with bouncing off walls.
    
    Args:
        n_samples: Number of time points.
        arena_size: (length,) of arena.
        speeds: Speed at each time point.
        turning_rate: How quickly direction changes.
        rng: Random number generator.
        
    Returns:
        positions: Position trajectory, shape (n_samples, 1).
    """
    positions = np.zeros((n_samples, 1))
    positions[0, 0] = arena_size[0] / 2  # Start in center
    
    # Initial velocity direction
    velocity = rng.choice([-1, 1]) * speeds[0]
    
    for i in range(1, n_samples):
        # Update velocity direction with smooth turning
        velocity_change = rng.normal(0, turning_rate * speeds[i])
        velocity = velocity + velocity_change
        
        # Apply current speed magnitude
        if abs(velocity) > 0:
            velocity = (velocity / abs(velocity)) * speeds[i]
        else:
            velocity = rng.choice([-1, 1]) * speeds[i]
        
        new_pos = positions[i - 1, 0] + velocity
        
        # Bounce off walls
        if new_pos < 0:
            velocity = abs(velocity)
            new_pos = 0
        elif new_pos > arena_size[0]:
            velocity = -abs(velocity)
            new_pos = arena_size[0]
        
        positions[i, 0] = new_pos
    
    return positions


def _trajectory_2d(
    n_samples: int,
    arena_size: tuple[float, ...],
    speeds: npt.NDArray[np.float64],
    turning_rate: float,
    rng: np.random.Generator,
) -> npt.NDArray[np.float64]:
    """Generate 2D trajectory with bouncing off walls.
    
    Args:
        n_samples: Number of time points.
        arena_size: (width, height) of arena.
        speeds: Speed at each time point.
        turning_rate: How quickly direction changes.
        rng: Random number generator.
        
    Returns:
        positions: Position trajectory, shape (n_samples, 2).
    """
    positions = np.zeros((n_samples, 2))
    positions[0] = [arena_size[0] / 2, arena_size[1] / 2]  # Start in center
    
    # Initial direction
    direction = rng.uniform(0, 2 * np.pi)
    
    for i in range(1, n_samples):
        direction += rng.normal(0, turning_rate)
        
        dx = speeds[i] * np.cos(direction)
        dy = speeds[i] * np.sin(direction)
        new_pos = positions[i - 1] + [dx, dy]
        
        # Bounce off walls
        if new_pos[0] < 0 or new_pos[0] > arena_size[0]:
            direction = np.pi - direction
            new_pos[0] = np.clip(new_pos[0], 0, arena_size[0])
        
        if new_pos[1] < 0 or new_pos[1] > arena_size[1]:
            direction = -direction
            new_pos[1] = np.clip(new_pos[1], 0, arena_size[1])
        
        positions[i] = new_pos
    
    return positions


def _trajectory_3d(
    n_samples: int,
    arena_size: tuple[float, ...],
    speeds: npt.NDArray[np.float64],
    turning_rate: float,
    rng: np.random.Generator,
) -> npt.NDArray[np.float64]:
    """Generate 3D trajectory with bouncing off walls.
    
    Args:
        n_samples: Number of time points.
        arena_size: (width, height, depth) of arena.
        speeds: Speed at each time point.
        turning_rate: How quickly direction changes.
        rng: Random number generator.
        
    Returns:
        positions: Position trajectory, shape (n_samples, 3).
    """
    positions = np.zeros((n_samples, 3))
    positions[0] = [s / 2 for s in arena_size]  # Start in center
    
    # Initial spherical direction
    theta = rng.uniform(0, 2 * np.pi)  # azimuth
    phi = rng.uniform(0, np.pi)  # elevation
    
    for i in range(1, n_samples):
        theta += rng.normal(0, turning_rate)
        phi += rng.normal(0, turning_rate / 2)
        phi = np.clip(phi, 0, np.pi)  # Keep elevation valid
        
        dx = speeds[i] * np.sin(phi) * np.cos(theta)
        dy = speeds[i] * np.sin(phi) * np.sin(theta)
        dz = speeds[i] * np.cos(phi)
        new_pos = positions[i - 1] + [dx, dy, dz]
        
        # Bounce off walls
        for dim in range(3):
            if new_pos[dim] < 0 or new_pos[dim] > arena_size[dim]:
                new_pos[dim] = np.clip(new_pos[dim], 0, arena_size[dim])
                # Reverse relevant direction component
                if dim == 0:
                    theta = np.pi - theta
                elif dim == 1:
                    theta = -theta
                else:
                    phi = np.pi - phi
        
        positions[i] = new_pos
    
    return positions


def generate_position_trajectory(
    n_samples: int = 1000,
    arena_size: float | tuple[float, ...] = (1.0, 1.0),
    speed: float | None = None,
    speed_range: tuple[float, float] = (0.02, 0.2),
    turning_rate: float = 0.3,
    seed: int | None = None,
) -> npt.NDArray[np.float64]:
    """Generate realistic position trajectory for a freely moving animal.

    Simulates random walk with momentum in 1D, 2D, or 3D space with
    realistic speed variations.

    Args:
        n_samples: Number of time points.
        arena_size: Size of the arena in meters. Can be:
            - float: 1D linear track of length arena_size
            - Tuple[float, float]: 2D arena (width, height)
            - Tuple[float, float, float]: 3D arena (width, height, depth)
        speed: DEPRECATED. Average movement speed in meters per timestep.
            If provided, overrides speed_range with fixed speed.
        speed_range: Tuple of (min_speed, max_speed) in meters per timestep.
            Speed smoothly varies within this range. Default: (0.02, 0.2) m/s.
        turning_rate: How quickly direction changes (0=straight, 1=random).
        seed: Random seed for reproducibility.

    Returns:
        positions: Position trajectory, shape (n_samples, n_dims).
            Each row is coordinates in meters.

    Examples:
        >>> # 1D trajectory
        >>> pos_1d = generate_position_trajectory(1000, arena_size=2.0)
        >>> # 2D trajectory
        >>> pos_2d = generate_position_trajectory(1000, arena_size=(2.0, 2.0))
        >>> # 3D trajectory
        >>> pos_3d = generate_position_trajectory(1000, arena_size=(2.0, 2.0, 1.5))
    """
    rng = np.random.default_rng(seed)
    
    # Handle deprecated speed parameter
    if speed is not None:
        speed_range = (speed, speed)
    
    # Determine dimensionality
    if isinstance(arena_size, (int, float)):
        n_dims = 1
        arena_size = (float(arena_size),)
    else:
        n_dims = len(arena_size)
        arena_size = tuple(arena_size)
    
    # Generate smooth speed profile
    speeds = _generate_smooth_speeds(n_samples, speed_range, rng)
    
    # Generate trajectory based on dimensionality
    if n_dims == 1:
        positions = _trajectory_1d(n_samples, arena_size, speeds, turning_rate, rng)
    elif n_dims == 2:
        positions = _trajectory_2d(n_samples, arena_size, speeds, turning_rate, rng)
    elif n_dims == 3:
        positions = _trajectory_3d(n_samples, arena_size, speeds, turning_rate, rng)
    else:
        raise ValueError(f"Unsupported number of dimensions: {n_dims}")
    
    return positions


def generate_head_direction(
    n_samples: int = 1000,
    turning_rate: float = 0.1,
    seed: int | None = None,
) -> npt.NDArray[np.float64]:
    """Generate head direction trajectory.

    Simulates angular position of animal's head over time.

    Args:
        n_samples: Number of time points.
        turning_rate: Standard deviation of angular velocity (radians per timestep).
        seed: Random seed for reproducibility.

    Returns:
        angles: Head direction angles in radians, shape (n_samples,).
            Values are in range [0, 2π].

    Examples:
        >>> hd = generate_head_direction(1000, turning_rate=0.2)
        >>> # Convert to degrees for plotting
        >>> hd_deg = np.degrees(hd)
    """
    rng = np.random.default_rng(seed)

    angles = np.zeros(n_samples)
    angles[0] = rng.uniform(0, 2 * np.pi)

    for i in range(1, n_samples):
        angles[i] = angles[i - 1] + rng.normal(0, turning_rate)

    # Wrap to [0, 2π]
    angles = np.mod(angles, 2 * np.pi)

    return angles


def generate_place_cells(
    n_cells: int = 100,
    n_samples: int = 1000,
    positions: npt.NDArray | None = None,
    arena_size: float | tuple[float, ...] = (1.0, 1.0),
    field_size: float = 0.2,
    peak_rate: float = 10.0,
    noise_level: float = 0.1,
    sampling_rate: float = 20.0,
    seed: int | None = None,
    plot: bool = True,
) -> tuple[npt.NDArray[np.float64], dict]:
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
    else:
        n_dims = positions.shape[1] if positions.ndim > 1 else 1
        if n_dims == 1 and positions.ndim == 1:
            positions = positions.reshape(-1, 1)

    # Random place field centers
    if n_dims == 1:
        field_centers = rng.uniform(0, arena_size[0], size=(n_cells, 1))
    else:
        field_centers = rng.uniform(
            [0] * n_dims, arena_size, size=(n_cells, n_dims)
        )

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
        # For 3D, we use ellipsoid without rotation (simplified)
        field_angles = None
    
    # Compute firing rates based on distance to field center
    activity = np.zeros((n_samples, n_cells))
    
    # Low baseline firing rate outside place field (realistic for place cells)
    baseline_rate = peak_rate * 0.01  # 1% of peak rate

    for i in range(n_cells):
        if n_dims == 1:
            # 1D: Simple Gaussian with variable width
            distances = np.abs(positions[:, 0] - field_centers[i, 0])
            rates = baseline_rate + peak_rate * np.exp(-(distances ** 2) / (2 * field_radii[i, 0] ** 2))
        
        elif n_dims == 2:
            # 2D: Rotated oval (anisotropic Gaussian)
            # Translate to field center
            dx = positions[:, 0] - field_centers[i, 0]
            dy = positions[:, 1] - field_centers[i, 1]
            
            # Rotate to field orientation
            angle = field_angles[i]
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
        'field_centers': field_centers,
        'field_radii': field_radii,
        'field_angles': field_angles if n_dims == 2 else None,
        'positions': positions,
        'cell_type': 'place',
        'arena_size': arena_size,
        'field_size': field_size,
        'n_dims': n_dims,
        'sampling_rate': sampling_rate,
    }

    # Create visualization if requested
    if plot:
        from neural_analysis.plotting.synthetic_plots import plot_synthetic_data
        # Force embeddings to always be 2D, even for 3D spatial environments
        # Raster plots are always 2D heatmaps by design
        plot_synthetic_data(
            activity, metadata,
            show_raster=True,
            show_fields=True,
            show_behavior=True,
            show_ground_truth=False,
            show_embeddings=True,
            embedding_methods=['pca', 'umap'],
            n_embedding_dims=2,  # Always 2D for embeddings, regardless of spatial dims
        )

    return activity, metadata


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
                np.cos(freq * proj1) +
                np.cos(freq * proj2) +
                np.cos(freq * (proj1 - proj2))
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


def generate_grid_cells(
    n_cells: int = 50,
    n_samples: int = 1000,
    positions: npt.NDArray | None = None,
    arena_size: float | tuple[float, ...] = (2.0, 2.0),
    grid_spacing: float = 0.4,
    grid_orientation: float = 0.0,
    peak_rate: float = 10.0,
    noise_level: float = 0.1,
    sampling_rate: float = 20.0,
    seed: int | None = None,
    plot: bool = True,
) -> tuple[npt.NDArray[np.float64], dict]:
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
        >>> activity, meta = generate_grid_cells(30, 1000, arena_size=(2.0, 2.0), plot=False)
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
    else:
        n_dims = positions.shape[1] if positions.ndim > 1 else 1
        if n_dims == 1 and positions.ndim == 1:
            positions = positions.reshape(-1, 1)

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
                harmonic_weights=(0.6, 0.25, 0.15)
            )
            
            # Normalize to [0, 1] range, then scale
            rates = (rates - rates.min()) / (rates.max() - rates.min() + 1e-10)
            rates = rates * peak_rate
            
            # Add noise for biological realism (Poisson + Gaussian)
            if noise_level > 0:
                # Sample from Poisson distribution (spike count variability)
                rates = rng.poisson(rates)
                # Add small Gaussian noise on top
                rates = rates + rng.normal(0, noise_level * peak_rate * 0.1, size=rates.shape)
                # Clip to non-negative
                rates = np.maximum(0, rates)
            
            activity[:, i] = rates
        
        metadata = {
            'phase_offsets': phase_offsets,
            'positions': positions,
            'cell_type': 'grid',
            'grid_spacing': grid_spacing,
            'arena_size': arena_size,
            'n_dims': n_dims,
            'sampling_rate': sampling_rate,
        }
    
    elif n_dims == 2:
        # 2D: Hexagonal grid pattern with harmonics
        phase_offsets = rng.uniform([0, 0], [grid_spacing, grid_spacing], size=(n_cells, 2))
        
        # Grid axes (60 degrees apart)
        theta = np.radians(grid_orientation)
        axis1 = np.array([np.cos(theta), np.sin(theta)])
        axis2 = np.array([np.cos(theta + np.pi/3), np.sin(theta + np.pi/3)])
        axes = [axis1, axis2]
        
        for i in range(n_cells):
            # Use helper function for harmonic pattern
            rates = _compute_grid_pattern_with_harmonics(
                positions, 
                phase_offsets[i], 
                grid_spacing,
                axes=axes,
                harmonic_weights=(1.0, 0.4, 0.2)
            )
            
            # Normalize to [0, 1], then scale
            rates = (rates - rates.min()) / (rates.max() - rates.min() + 1e-10)
            rates = rates * peak_rate
            
            # Add Poisson noise (proper spike count variability) + small Gaussian noise
            if noise_level > 0:
                rates = rng.poisson(rates) + rng.normal(0, 0.1 * noise_level * peak_rate, size=rates.shape)
                rates = np.maximum(0, rates)  # Clip to non-negative
            
            activity[:, i] = rates
        
        metadata = {
            'phase_offsets': phase_offsets,
            'positions': positions,
            'cell_type': 'grid',
            'grid_spacing': grid_spacing,
            'grid_orientation': grid_orientation,
            'arena_size': arena_size,
            'n_dims': n_dims,
            'sampling_rate': sampling_rate,
        }
    
    elif n_dims == 3:
        # 3D: Hexagonal/tetrahedral grid pattern (FCC-like) with harmonics
        # Create hierarchy of grid spacings (biological realism: dorsal-ventral gradient)
        # Generate multiple scales from fine to coarse
        grid_spacings = np.zeros(n_cells)
        cells_per_scale = max(1, n_cells // 5)  # Divide into ~5 scale groups
        
        for scale_idx in range(5):
            start_idx = scale_idx * cells_per_scale
            end_idx = min((scale_idx + 1) * cells_per_scale, n_cells)
            if start_idx >= n_cells:
                break
            # Scale factor: 1.0, 1.4, 2.0, 2.8, 4.0 (roughly sqrt(2) progression)
            scale_factor = 1.4 ** scale_idx
            grid_spacings[start_idx:end_idx] = grid_spacing * scale_factor
        
        # Fill any remaining cells
        if end_idx < n_cells:
            grid_spacings[end_idx:] = grid_spacing * (1.4 ** 4)
        
        phase_offsets = rng.uniform(
            [0, 0, 0], 
            [grid_spacing] * 3, 
            size=(n_cells, 3)
        )
        
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
            rotation_z = np.array([
                [cos_t, -sin_t, 0],
                [sin_t, cos_t, 0],
                [0, 0, 1]
            ])
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
                harmonic_weights=(1.0, 0.4, 0.2)
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
            'phase_offsets': phase_offsets,
            'grid_spacings': grid_spacings,  # Cell-specific spacings
            'grid_spacing': grid_spacing,  # Base spacing
            'positions': positions,
            'cell_type': 'grid',
            'grid_orientation': grid_orientation,
            'arena_size': arena_size,
            'n_dims': n_dims,
            'sampling_rate': sampling_rate,
        }

    # Create visualization if requested
    if plot:
        from neural_analysis.plotting.synthetic_plots import plot_synthetic_data
        plot_synthetic_data(
            activity, metadata,
            show_raster=True,
            show_fields=True,
            show_behavior=True,
            show_ground_truth=False,
            show_embeddings=True,
            embedding_methods=['pca', 'umap'],
            n_embedding_dims=2,
        )

    return activity, metadata


def generate_head_direction_cells(
    n_cells: int = 60,
    n_samples: int = 1000,
    head_direction: npt.NDArray | None = None,
    tuning_width: float = np.pi / 6,  # 30 degrees
    peak_rate: float = 10.0,
    noise_level: float = 0.1,
    sampling_rate: float = 20.0,
    seed: int | None = None,
    plot: bool = True,
) -> tuple[npt.NDArray[np.float64], dict]:
    """Generate head direction cell firing data.

    Head direction cells fire when animal's head points in specific direction.

    Args:
        n_cells: Number of head direction cells.
        n_samples: Number of time points.
        head_direction: Optional head direction trajectory in radians, shape (n_samples,).
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

    # Random preferred directions
    preferred_dirs = rng.uniform(0, 2 * np.pi, size=n_cells)

    activity = np.zeros((n_samples, n_cells))

    for i in range(n_cells):
        # Circular difference
        angle_diff = np.abs(head_direction - preferred_dirs[i])
        angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)

        # Von Mises tuning curve (circular Gaussian)
        kappa = 1 / (tuning_width ** 2)  # Concentration parameter
        rates = peak_rate * np.exp(kappa * (np.cos(angle_diff) - 1))

        # Add Poisson noise
        if noise_level > 0:
            rates = rng.poisson(rates * noise_level) / noise_level

        activity[:, i] = rates

    metadata = {
        'preferred_directions': preferred_dirs,
        'head_directions': head_direction,
        'cell_type': 'head_direction',
        'tuning_width': tuning_width,
        'sampling_rate': sampling_rate,
    }

    # Create visualization if requested
    if plot:
        from neural_analysis.plotting.synthetic_plots import plot_synthetic_data
        plot_synthetic_data(
            activity, metadata,
            show_raster=True,
            show_fields=True,
            show_behavior=False,
            show_ground_truth=False,
            show_embeddings=True,
            embedding_methods=['pca', 'umap'],
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
) -> tuple[npt.NDArray[np.float64], dict]:
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
    
    Returns:
        activity: Neural activity matrix, shape (n_samples, n_cells).
            Random firing patterns with temporal smoothness.
        metadata: Dictionary with cell parameters.
    
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
                smoothed[t] = alpha * smoothed[t-1] + (1 - alpha) * noise[t]
            noise = smoothed
        
        # Add baseline and ensure non-negative
        rates = baseline_rate + noise
        rates = np.maximum(rates, 0)
        
        # Add Poisson noise for realism
        rates = rng.poisson(rates)
        
        activity[:, i] = rates
    
    metadata = {
        'cell_type': 'random',
        'baseline_rate': baseline_rate,
        'variability': variability,
        'temporal_smoothness': temporal_smoothness,
        'sampling_rate': sampling_rate,
    }
    
    # Create visualization if requested
    if plot:
        from neural_analysis.plotting.synthetic_plots import plot_synthetic_data
        plot_synthetic_data(
            activity, metadata,
            show_raster=True,
            show_fields=False,
            show_behavior=False,
            show_ground_truth=False,
            show_embeddings=True,
            embedding_methods=['pca'],
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
) -> tuple[npt.NDArray[np.float64], dict]:
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
    cell_types = (['place'] * n_place +
                  ['grid'] * n_grid +
                  ['head_direction'] * n_hd)

    metadata = {
        'cell_types': np.array(cell_types),
        'positions': positions,
        'head_direction': head_direction,
        'arena_size': arena_size,
        'n_place': n_place,
        'n_grid': n_grid,
        'n_hd': n_hd,
        'place_meta': place_meta,
        'grid_meta': grid_meta,
        'hd_meta': hd_meta,
    }

    return activity, metadata


def add_noise(
    data: npt.NDArray,
    noise_type: Literal['gaussian', 'poisson', 'uniform'] = 'gaussian',
    noise_level: float = 0.1,
    seed: int | None = None,
) -> npt.NDArray:
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

    if noise_type == 'gaussian':
        noise = rng.normal(0, noise_level, size=data.shape)
        return data + noise

    elif noise_type == 'poisson':
        # For Poisson noise, we need positive values
        # Treat data as rates and sample from Poisson distribution
        data_positive = np.maximum(data, 0)  # Ensure non-negative
        # Scale by noise_level (higher = more noise)
        if noise_level > 0:
            noisy = rng.poisson(data_positive / noise_level) * noise_level
        else:
            noisy = data_positive
        return noisy

    elif noise_type == 'uniform':
        noise = rng.uniform(-noise_level, noise_level, size=data.shape)
        return data + noise

    else:
        raise ValueError(f"Unknown noise type: {noise_type}")


# ============================================================================
# Backward-Compatible Convenience Functions
# ============================================================================

def generate_swiss_roll(
    n_samples: int = 1000,
    noise: float = 0.0,
    seed: int | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Generate Swiss roll manifold dataset using scikit-learn.

    Classic 3D manifold that lies on a 2D surface. Useful for testing
    dimensionality reduction algorithms like Isomap, LLE, and UMAP.

    Args:
        n_samples: Number of points to generate.
        noise: Standard deviation of Gaussian noise added to the data.
        seed: Random seed for reproducibility.

    Returns:
        points: Swiss roll coordinates, shape (n_samples, 3).
        colors: Color values along the roll, shape (n_samples,).
            Useful for verifying that manifold structure is preserved.

    Examples:
        >>> points, colors = generate_swiss_roll(1000, noise=0.1)
        >>> # Visualize with color coding
        >>> from neural_analysis.plotting import plot_scatter_3d
        >>> fig = plot_scatter_3d(
        ...     points[:, 0], points[:, 1], points[:, 2],
        ...     colors=colors
        ... )
    """
    return generate_data('swiss_roll', n_samples=n_samples, noise=noise, seed=seed)


def generate_s_curve(
    n_samples: int = 1000,
    noise: float = 0.0,
    seed: int | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Generate S-curve manifold dataset using scikit-learn.

    Another classic 3D manifold on a 2D surface, shaped like an 'S'.

    Args:
        n_samples: Number of points to generate.
        noise: Standard deviation of Gaussian noise.
        seed: Random seed for reproducibility.

    Returns:
        points: S-curve coordinates, shape (n_samples, 3).
        colors: Color values along the curve, shape (n_samples,).

    Examples:
        >>> points, colors = generate_s_curve(1000, noise=0.05)
    """
    return generate_data('s_curve', n_samples=n_samples, noise=noise, seed=seed)


# ============================================================================
# Manifold Mapping Functions
# ============================================================================

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
    if positions.ndim > 1:
        positions_flat = positions.ravel()
    else:
        positions_flat = positions
    
    # Normalize positions to [0, 2π]
    pos_min, pos_max = positions_flat.min(), positions_flat.max()
    angles = 2 * np.pi * (positions_flat - pos_min) / (pos_max - pos_min)
    
    # Map to ring
    ring_coords = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Create visualization if requested
    if plot:
        from neural_analysis.plotting.grid_config import PlotSpec, PlotGrid
        from neural_analysis.plotting.grid_config import PlotConfig, GridLayoutConfig
        
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
            color_by=time_array,
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
            color_by=positions_flat,
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
        
        # Create grid
        grid = PlotGrid(
            plot_specs=plot_specs,
            config=PlotConfig(figsize=(15, 5)),
            layout=GridLayoutConfig(rows=1, cols=3),
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
    theta = (
        2 * np.pi * (positions[:, 0] - pos_min[0]) / (pos_max[0] - pos_min[0])
    )
    phi = (
        2 * np.pi * (positions[:, 1] - pos_min[1]) / (pos_max[1] - pos_min[1])
    )
    
    # Map to torus
    x = (major_radius + minor_radius * np.cos(phi)) * np.cos(theta)
    y = (major_radius + minor_radius * np.cos(phi)) * np.sin(theta)
    z = minor_radius * np.sin(phi)
    
    torus_coords = np.column_stack([x, y, z])
    
    # Create visualization if requested
    if plot:
        from neural_analysis.plotting.grid_config import (
            GridLayoutConfig,
            PlotConfig,
            PlotGrid,
            PlotSpec,
        )
        
        plot_specs = []
        time_array = np.arange(len(positions))
        
        # 1. Original 2D trajectory
        spec1 = PlotSpec(
            data={"x": positions[:, 0], "y": positions[:, 1]},
            plot_type="trajectory",
            subplot_position=0,
            title="2D Position Trajectory",
            color_by=time_array,
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
            data={"x": torus_coords[:, 0], "y": torus_coords[:, 1], "z": torus_coords[:, 2]},
            plot_type="scatter3d",
            subplot_position=1,
            title="Torus Embedding (T²) - Colored by Time",
            color_by=time_array,
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
            data={"x": torus_coords[:, 0], "y": torus_coords[:, 1], "z": torus_coords[:, 2]},
            plot_type="scatter3d",
            subplot_position=2,
            title="Torus - Colored by X Position",
            color_by=positions[:, 0],
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
                'head_direction': {'n_cells': 20, 'tuning_width': np.pi/6, 'noise_level': 0.1},
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
        >>> activity, meta = generate_mixed_population_flexible(n_samples=1500, seed=42, plot=True)
        
        >>> # Custom configuration without plotting
        >>> config = {
        ...     'place': {'n_cells': 50, 'field_size': 0.3, 'noise_level': 0.1},
        ...     'grid': {'n_cells': 30, 'grid_spacing': 0.5, 'noise_level': 0.05},
        ...     'head_direction': {'n_cells': 20, 'noise_level': 0.1},
        ...     'random': {'n_cells': 15, 'baseline_rate': 3.0},
        ... }
        >>> activity, meta = generate_mixed_population_flexible(config, n_samples=2000, plot=False)
        >>> # Access specific cell types
        >>> place_indices = meta['cell_indices']['place']
        >>> place_activity = activity[:, place_indices]
    """
    np.random.default_rng(seed)
    
    # Default configuration if none provided
    if cell_config is None:
        cell_config = {
            'place': {'n_cells': 50, 'field_size': 0.2, 'noise_level': 0.08},
            'grid': {'n_cells': 30, 'grid_spacing': 0.4, 'noise_level': 0.05},
            'head_direction': {'n_cells': 25, 'tuning_width': np.pi/6, 'noise_level': 0.1},
            'random': {'n_cells': 20, 'baseline_rate': 2.0, 'variability': 1.5},
        }
    
    # Generate common behavioral variables
    positions = generate_position_trajectory(
        n_samples, arena_size=arena_size, seed=seed
    )
    
    # Determine if we need head direction
    needs_hd = 'head_direction' in cell_config
    head_direction = generate_head_direction(n_samples, seed=seed) if needs_hd else None
    
    # Generate each cell type
    all_activity = []
    all_cell_types = []
    cell_indices = {}
    cell_metadata = {}
    current_idx = 0
    
    for cell_type, params in cell_config.items():
        n_cells = params.pop('n_cells', 10)
        
        if cell_type == 'place':
            activity, meta = generate_place_cells(
                n_cells=n_cells,
                n_samples=n_samples,
                positions=positions,
                arena_size=arena_size,
                seed=seed,
                plot=False,  # Disable individual plots for mixed population
                **params
            )
        
        elif cell_type == 'grid':
            activity, meta = generate_grid_cells(
                n_cells=n_cells,
                n_samples=n_samples,
                positions=positions,
                arena_size=arena_size,
                seed=seed,
                plot=False,  # Disable individual plots for mixed population
                **params
            )
        
        elif cell_type in ['head_direction', 'hd']:
            activity, meta = generate_head_direction_cells(
                n_cells=n_cells,
                n_samples=n_samples,
                head_direction=head_direction,
                seed=seed,
                plot=False,  # Disable individual plots for mixed population
                **params
            )
        
        elif cell_type == 'random':
            activity, meta = generate_random_cells(
                n_cells=n_cells,
                n_samples=n_samples,
                seed=seed,
                plot=False,  # Disable individual plots for mixed population
                **params
            )
        
        else:
            raise ValueError(f"Unknown cell type: {cell_type}")
        
        # Store activity and metadata
        all_activity.append(activity)
        all_cell_types.extend([cell_type] * n_cells)
        cell_indices[cell_type] = list(range(current_idx, current_idx + n_cells))
        cell_metadata[cell_type] = meta
        current_idx += n_cells
        
        # Restore n_cells to params
        params['n_cells'] = n_cells
    
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
        if 'preferred_directions' in meta and meta['preferred_directions'] is not None:
            preferred_directions[idxs] = meta['preferred_directions']
    
    # Create comprehensive metadata
    metadata = {
        'cell_types': np.array(all_cell_types),
        'cell_indices': cell_indices,
        'cell_config': cell_config,
        'positions': positions,
        'head_direction': head_direction,
        'head_directions': head_direction,  # Add plural form for compatibility
        'arena_size': arena_size,
        'n_samples': n_samples,
        'n_dims': 2,  # Mixed populations currently only support 2D
        'preferred_directions': preferred_directions,
        'individual_metadata': cell_metadata,
    }
    
    # Create visualization if requested
    if plot:
        from neural_analysis.plotting.synthetic_plots import plot_synthetic_data
        plot_synthetic_data(
            combined_activity, metadata,
            show_raster=True,
            show_fields=True,
            show_behavior=True,
            show_ground_truth=False,
            show_embeddings=True,
            embedding_methods=['pca', 'umap'],
            n_embedding_dims=2,
        )
    
    return combined_activity, metadata
