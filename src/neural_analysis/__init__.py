"""neural_analysis - Tools for neural data analysis and dimensionality reduction.

This package provides:
- Synthetic neural data generation (place cells, grid cells, head direction cells)
- Metrics (pairwise distances, shape distances, outliers)
- Embeddings (PCA, UMAP, t-SNE, MDS, Isomap, LLE, Spectral)
- Learning (decoding, classification, clustering)
- Topology (structure index)
- Visualization (PlotGrid system with matplotlib/plotly)
"""

__version__ = "0.1.0"

# Data generation
from neural_analysis.data.datasets import generate_data
from neural_analysis.data.synthetic_data import (
    generate_grid_cells,
    generate_head_direction_cells,
    generate_mixed_population_flexible,
    generate_place_cells,
    generate_random_cells,
    generate_shape_distance_datasets,
    map_to_ring,
    map_to_torus,
)

# Embeddings
from neural_analysis.embeddings.dimensionality_reduction import (
    compute_embedding,
    # compute_multiple_embeddings
)

# Learning
from neural_analysis.learning.classification import (
    classify_cells,
    cluster_cells,
    compare_classifiers,
    compare_clusterers,
    evaluate_classifier,
    evaluate_clustering,
    extract_cell_features,
    fit_clusterer,
    train_classifier,
)
from neural_analysis.learning.decoding import (
    compare_highd_lowd_decoding,
    cross_validated_knn_decoder,
    evaluate_decoder,
    knn_decoder,
    population_vector_decoder,
)

# Metrics
from neural_analysis.metrics.distributions import shape_distance
from neural_analysis.metrics.outliers import filter_outlier
from neural_analysis.metrics.pairwise_metrics import compute_pairwise_matrix

# Pipeline
from neural_analysis.pipeline import PipelineConfig, PipelineResult, run_analysis

# Plotting
from neural_analysis.plotting.grid_config import (
    GridLayoutConfig,
    PlotGrid,
    PlotSpec,
)

# Topology
from neural_analysis.topology.structure_index import compute_structure_index

__all__ = [
    # Data generation
    "generate_data",
    "generate_place_cells",
    "generate_grid_cells",
    "generate_head_direction_cells",
    "generate_random_cells",
    "generate_mixed_population_flexible",
    "generate_shape_distance_datasets",
    "map_to_ring",
    "map_to_torus",
    # Embeddings
    "compute_embedding",
    "compute_multiple_embeddings",
    # Learning
    "classify_cells",
    "cluster_cells",
    "compare_classifiers",
    "compare_clusterers",
    "evaluate_classifier",
    "evaluate_clustering",
    "extract_cell_features",
    "fit_clusterer",
    "train_classifier",
    "knn_decoder",
    "population_vector_decoder",
    "cross_validated_knn_decoder",
    "compare_highd_lowd_decoding",
    "evaluate_decoder",
    # Metrics
    "compute_pairwise_matrix",
    "shape_distance",
    "filter_outlier",
    # Plotting
    "PlotGrid",
    "PlotSpec",
    "GridLayoutConfig",
    # Topology
    "compute_structure_index",
    # Pipeline
    "run_analysis",
    "PipelineConfig",
    "PipelineResult",
]


def hello() -> str:
    """Simple placeholder function used by tests.

    Returns:
        A greeting string.
    """
    return "hello from neural_analysis"
