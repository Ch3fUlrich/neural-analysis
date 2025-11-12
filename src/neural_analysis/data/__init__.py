"""Data generation and synthetic dataset utilities for neural analysis.

This module provides functions for generating synthetic neural datasets including
place cells, grid cells, head direction cells, and mixed populations.
"""

from neural_analysis.data.synthetic_data import (
    generate_data,
    generate_grid_cells,
    generate_head_direction_cells,
    generate_mixed_population_flexible,
    generate_place_cells,
    generate_random_cells,
)

__all__ = [
    "generate_data",
    "generate_grid_cells",
    "generate_head_direction_cells",
    "generate_mixed_population_flexible",
    "generate_place_cells",
    "generate_random_cells",
]
