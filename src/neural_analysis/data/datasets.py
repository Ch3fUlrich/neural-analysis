"""Dataset assembly, orchestration, and preset dataset functions.

This module provides the main ``generate_data()`` orchestrator function that
offers a unified interface for all dataset types, along with backward-compatible
convenience functions and sklearn dataset wrappers.
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

from neural_analysis.data.generators import (
    generate_grid_cells,
    generate_head_direction_cells,
    generate_mixed_neural_population,
    generate_place_cells,
    generate_random_cells,
    generate_shape_distance_datasets,
)
from neural_analysis.data.trajectories_gen import (
    generate_head_direction,
    generate_position_trajectory,
)

# Type aliases
DatasetType = Literal[
    "swiss_roll",
    "s_curve",
    "blobs",
    "moons",
    "circles",
    "classification",
    "regression",
    "place_cells",
    "grid_cells",
    "head_direction_cells",
    "mixed_cells",
    "position_trajectory",
    "head_direction",
    "shape_distance_clusters",  # For shape distance validation
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
    dataset_type = dataset_type.lower()  # type: ignore[assignment]

    # Route to appropriate generator
    match dataset_type:
        case "swiss_roll":
            return _generate_swiss_roll(n_samples, noise, seed)

        case "s_curve":
            return _generate_s_curve(n_samples, noise, seed)

        case "blobs":
            n_features = n_features or 2
            n_classes = n_classes or 3
            data, labels = _generate_blobs(
                n_samples, n_features, n_classes, noise, seed, **kwargs
            )
            return data, labels.astype(np.float64)

        case "moons":
            data, labels = _generate_moons(n_samples, noise, seed)
            return data, labels.astype(np.float64)

        case "circles":
            data, labels = _generate_circles(n_samples, noise, seed, **kwargs)
            return data, labels.astype(np.float64)

        case "classification":
            n_features = n_features or 20
            n_classes = n_classes or 2
            data, labels = _generate_classification(
                n_samples, n_features, n_classes, noise, seed, **kwargs
            )
            return data, labels.astype(np.float64)

        case "regression":
            n_features = n_features or 10
            return _generate_regression(n_samples, n_features, noise, seed, **kwargs)

        case "place_cells":
            n_features = n_features or 100  # Number of cells
            return _generate_place_cells(n_samples, n_features, noise, seed, **kwargs)

        case "grid_cells":
            n_features = n_features or 50
            return _generate_grid_cells(n_samples, n_features, noise, seed, **kwargs)

        case "random_cells":
            n_features = n_features or 50
            return _generate_random_cells(n_samples, n_features, noise, seed, **kwargs)

        case "head_direction_cells":
            n_features = n_features or 60
            return _generate_head_direction_cells(
                n_samples, n_features, noise, seed, **kwargs
            )

        case "mixed_cells":
            return _generate_mixed_cells(n_samples, seed, **kwargs)

        case "position_trajectory":
            return _generate_position_trajectory(n_samples, seed, **kwargs)

        case "head_direction":
            return _generate_head_direction(n_samples, seed, **kwargs)

        case "shape_distance_clusters":
            data, labels = _generate_shape_distance_clusters(
                n_samples, n_features, seed, **kwargs
            )
            # Convert labels to float64 array to match return type
            # (shape_distance_clusters returns int labels, but generate_data expects float64 or dict)
            return data, labels.astype(np.float64)

        case _:
            raise ValueError(
                f"Unknown dataset type: {dataset_type}. "
                f"Available types: swiss_roll, s_curve, blobs, moons, circles, "
                f"classification, regression, place_cells, grid_cells, random_cells, "
                f"head_direction_cells, mixed_cells, position_trajectory, "
                f"head_direction, shape_distance_clusters"
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
    x, t = make_s_curve(n_samples=n_samples, noise=noise, random_state=seed)
    return x, t


def _generate_blobs(
    n_samples: int,
    n_features: int,
    n_centers: int,
    noise: float,
    seed: int | None,
    **kwargs: Any,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """Generate isotropic Gaussian blobs using scikit-learn."""
    cluster_std = kwargs.get("cluster_std", 1.0 + noise)
    x, y = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_centers,
        cluster_std=cluster_std,
        random_state=seed,
    )
    return x, y


def _generate_moons(
    n_samples: int,
    noise: float,
    seed: int | None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """Generate two interleaving half circles using scikit-learn."""
    x, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    return x, y


def _generate_circles(
    n_samples: int,
    noise: float,
    seed: int | None,
    **kwargs: Any,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """Generate large circle containing smaller circle using scikit-learn."""
    factor = kwargs.get("factor", 0.5)
    x, y = make_circles(
        n_samples=n_samples,
        noise=noise,
        factor=factor,
        random_state=seed,
    )
    return x, y


def _generate_classification(
    n_samples: int,
    n_features: int,
    n_classes: int,
    noise: float,
    seed: int | None,
    **kwargs: Any,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """Generate random classification problem using scikit-learn."""
    n_informative = kwargs.get("n_informative", min(n_features, 2 * n_classes))
    n_redundant = kwargs.get("n_redundant", 0)

    x, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_classes=n_classes,
        flip_y=noise,
        random_state=seed,
    )
    return x, y


def _generate_regression(
    n_samples: int,
    n_features: int,
    noise: float,
    seed: int | None,
    **kwargs: Any,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Generate random regression problem using scikit-learn."""
    n_informative = kwargs.get("n_informative", min(n_features, 10))

    x, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        noise=noise,
        random_state=seed,
    )
    return x, y


# ============================================================================
# Behavioral Data Generator Wrappers
# ============================================================================


def _generate_position_trajectory(
    n_samples: int,
    seed: int | None,
    **kwargs: Any,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Generate position trajectory - returns (positions, positions)
    for label compatibility.
    """
    arena_size = kwargs.get("arena_size", (1.0, 1.0))
    speed = kwargs.get("speed", 0.1)
    turning_rate = kwargs.get("turning_rate", 0.3)

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
    turning_rate = kwargs.get("turning_rate", 0.1)

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
    arena_size = kwargs.get("arena_size", (1.0, 1.0))
    field_size = kwargs.get("field_size", 0.2)
    peak_rate = kwargs.get("peak_rate", 10.0)
    sampling_rate = kwargs.get("sampling_rate", 20.0)
    positions = kwargs.get("positions")
    plot = kwargs.get("plot", True)  # Default to True for visualization

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
        plot=plot,  # Pass plot parameter through
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
    arena_size = kwargs.get("arena_size", (2.0, 2.0))
    grid_spacing = kwargs.get("grid_spacing", 0.4)
    grid_orientation = kwargs.get("grid_orientation", 0.0)
    peak_rate = kwargs.get("peak_rate", 10.0)
    sampling_rate = kwargs.get("sampling_rate", 20.0)
    positions = kwargs.get("positions")
    plot = kwargs.get("plot", True)  # Default to True for visualization

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
        plot=plot,  # Pass plot parameter through
    )
    return activity, metadata


def _generate_random_cells(
    n_samples: int,
    n_features: int,
    noise: float,
    seed: int | None,
    **kwargs: Any,
) -> tuple[npt.NDArray[np.float64], dict[str, Any]]:
    """Generate random cells - returns (activity, metadata with positions)."""
    baseline_rate = kwargs.get("baseline_rate", 2.0)
    variability = kwargs.get("variability", noise * 10.0)  # Scale noise to variability
    temporal_smoothness = kwargs.get("temporal_smoothness", 0.1)
    sampling_rate = kwargs.get("sampling_rate", 20.0)
    arena_size = kwargs.get("arena_size", (1.0, 1.0))
    plot = kwargs.get("plot", True)  # Default to True for visualization

    activity, metadata = generate_random_cells(
        n_cells=n_features,
        n_samples=n_samples,
        baseline_rate=baseline_rate,
        variability=variability,
        temporal_smoothness=temporal_smoothness,
        sampling_rate=sampling_rate,
        arena_size=arena_size,
        seed=seed,
        plot=plot,  # Pass plot parameter through
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
    tuning_width = kwargs.get("tuning_width", np.pi / 6)
    peak_rate = kwargs.get("peak_rate", 10.0)
    sampling_rate = kwargs.get("sampling_rate", 20.0)
    head_direction = kwargs.get("head_direction")

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
    n_place = kwargs.get("n_place", 50)
    n_grid = kwargs.get("n_grid", 30)
    n_hd = kwargs.get("n_hd", 20)
    arena_size = kwargs.get("arena_size", (2.0, 2.0))

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
# Shape Distance Validation Data Wrappers
# ============================================================================


def _generate_shape_distance_clusters(
    n_samples: int,
    n_features: int | None,
    seed: int | None,
    **kwargs: Any,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int_]]:
    """Internal function for generate_data() to create shape distance clusters."""
    n_features = n_features or 100
    n_datasets = n_samples
    n_clusters = kwargs.get("n_clusters", 5)
    min_neurons = kwargs.get("min_neurons", 50)
    max_neurons = kwargs.get("max_neurons", 200)
    cluster_separation = kwargs.get("cluster_separation", 15.0)
    noise_scale = kwargs.get("noise", 0.1)

    datasets, labels = generate_shape_distance_datasets(
        n_datasets=n_datasets,
        n_clusters=n_clusters,
        min_neurons=min_neurons,
        max_neurons=max_neurons,
        n_features=n_features,
        cluster_separation=cluster_separation,
        noise_scale=noise_scale,
        seed=seed,
    )

    # For compatibility with generate_data interface, we need to return
    # a single array. However, datasets have different sizes, so we can't
    # stack them. Instead, return the first dataset and labels.
    # Note: This is a limitation - users should call generate_shape_distance_datasets
    # directly for proper usage.
    if len(datasets) > 0:
        return datasets[0], labels
    else:
        return np.zeros((0, n_features), dtype=np.float64), np.zeros(0, dtype=int)


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
    return generate_data("swiss_roll", n_samples=n_samples, noise=noise, seed=seed)  # type: ignore[return-value]


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
    data, labels = generate_data("s_curve", n_samples=n_samples, noise=noise, seed=seed)
    if not isinstance(labels, np.ndarray):
        labels = np.asarray(labels, dtype=np.float64)
    return np.asarray(data, dtype=np.float64), np.asarray(labels, dtype=np.float64)
