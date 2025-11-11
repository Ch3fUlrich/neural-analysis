"""Topology and structure analysis for neural data.

This module provides functions for quantifying topological structure in neural
representations, including the structure index metric which measures how well
behavioral structure is preserved in neural activity patterns.

Key functions:
    - compute_structure_index: Compute structure index metric
    - compute_structure_index_sweep: Parameter sweep with automatic caching
    - load_structure_index_results: Load cached results from HDF5
    - draw_overlap_graph: Visualize bin-group relationships

The structure index quantifies preservation of behavioral structure in neural
space by measuring overlaps between neural representations of different
behavioral states.

Example:
    >>> from neural_analysis.topology import compute_structure_index
    >>> # Generate sample data
    >>> neural_data = np.random.rand(1000, 10)
    >>> position = np.random.rand(1000, 2)
    >>> # Compute structure index
    >>> SI, bin_info, overlap_mat, shuffled_SI = compute_structure_index(
    ...     data=neural_data,
    ...     label=position,
    ...     n_bins=10,
    ...     n_neighbors=15,
    ...     num_shuffles=100
    ... )
    >>> print(f"Structure Index: {SI:.3f}")
"""

from __future__ import annotations

from .plotting import (
    plot_structure_index,
    plot_structure_index_comparison,
)
from .structure_index import (
    compute_structure_index,
    compute_structure_index_sweep,
    draw_overlap_graph,
    load_structure_index_results,
)

__all__ = [
    "compute_structure_index",
    "compute_structure_index_sweep",
    "draw_overlap_graph",
    "load_structure_index_results",
    "plot_structure_index",
    "plot_structure_index_comparison",
]
