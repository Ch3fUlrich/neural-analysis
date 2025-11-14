"""neural_analysis - Tools for neural data analysis and dimensionality reduction.

This package provides:
- Synthetic neural data generation (place cells, grid cells, head direction cells)
- Manifold mapping functions (ring, torus)
- Decoding methods (k-NN, population vector)
- Visualization tools (coming soon)
"""

__version__ = "0.0.0"

# Import key functions for convenient access
from neural_analysis.data.synthetic_data import (
    generate_grid_cells,
    generate_head_direction_cells,
    generate_mixed_population_flexible,
    generate_place_cells,
    generate_random_cells,
    map_to_ring,
    map_to_torus,
)
from neural_analysis.learning.decoding import (
    compare_highd_lowd_decoding,
    cross_validated_knn_decoder,
    evaluate_decoder,
    knn_decoder,
    population_vector_decoder,
)

__all__ = [
    # Synthetic data generation
    "generate_place_cells",
    "generate_grid_cells",
    "generate_head_direction_cells",
    "generate_random_cells",
    "generate_mixed_population_flexible",
    # Manifold mapping
    "map_to_ring",
    "map_to_torus",
    # Decoding
    "knn_decoder",
    "population_vector_decoder",
    "cross_validated_knn_decoder",
    "compare_highd_lowd_decoding",
    "evaluate_decoder",
]


def hello() -> str:
    """Simple placeholder function used by tests.

    Returns:
        A greeting string.
    """
    return "hello from neural_analysis"
