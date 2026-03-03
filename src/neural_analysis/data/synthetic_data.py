"""Synthetic dataset generators for testing neural analysis methods.

This module is a facade that re-exports all public functions from the
generators, trajectories_gen, and datasets submodules.

The main entry point is ``generate_data()`` which provides a unified interface
for all dataset types with consistent parameters and outputs.
"""

from neural_analysis.data.datasets import (
    DatasetType,
    generate_data,
    generate_s_curve,
    generate_swiss_roll,
)
from neural_analysis.data.generators import (
    add_noise,
    generate_cluster_templates,
    generate_dataset_from_cluster_template,
    generate_grid_cells,
    generate_head_direction_cells,
    generate_mixed_neural_population,
    generate_mixed_population_flexible,
    generate_place_cells,
    generate_random_cells,
    generate_shape_distance_datasets,
    map_to_ring,
    map_to_torus,
)
from neural_analysis.data.trajectories_gen import (
    generate_head_direction,
    generate_position_trajectory,
)

__all__ = [
    "DatasetType",
    "generate_data",
    "generate_position_trajectory",
    "generate_head_direction",
    "generate_place_cells",
    "generate_grid_cells",
    "generate_head_direction_cells",
    "generate_random_cells",
    "generate_mixed_neural_population",
    "generate_mixed_population_flexible",
    "map_to_ring",
    "map_to_torus",
    "generate_cluster_templates",
    "generate_dataset_from_cluster_template",
    "generate_shape_distance_datasets",
    "add_noise",
    "generate_swiss_roll",
    "generate_s_curve",
]
