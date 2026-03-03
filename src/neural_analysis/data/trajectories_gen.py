"""Trajectory generation functions for synthetic neural data.

This module provides functions to generate realistic animal trajectories
in 1D, 2D, and 3D environments, including position trajectories and
head direction signals.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import numpy.typing as npt


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
            speed_tau * current_speed
            + (1 - speed_tau) * speed_mean
            + speed_sigma * rng.normal()
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
            - tuple[float, float]: 2D arena (width, height)
            - tuple[float, float, float]: 3D arena (width, height, depth)
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
        raise ValueError(
            f"Unsupported number of dimensions. Expected: 1, 2, or 3. Got: {n_dims!r}"
        )

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
            Values are in range [0, 2π).

    Examples:
        >>> hd = generate_head_direction(1000, turning_rate=0.2)
        >>> # Convert to degrees for plotting
        >>> hd_deg = np.degrees(hd)
    """
    rng = np.random.default_rng(seed)

    angles = np.zeros(n_samples)
    angles[0] = rng.uniform(-np.pi, np.pi)

    for i in range(1, n_samples):
        angles[i] = angles[i - 1] + rng.normal(0, turning_rate)

    # Wrap to [0, 2π)
    angles = np.angle(np.exp(1j * angles))
    angles = (angles + 2 * np.pi) % (2 * np.pi)

    return cast("npt.NDArray[np.float64]", angles)
