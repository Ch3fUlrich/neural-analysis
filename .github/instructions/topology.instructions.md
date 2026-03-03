---
name: 'Topology'
description: 'Structure Index computation and topology analysis'
applyTo: 'src/neural_analysis/topology/**'
---

# Topology Instructions

## Structure Index (SI)

The Structure Index quantifies how well behavioral structure (e.g., spatial position) is preserved in neural population activity space.

### Algorithm

1. **Bin** neural data by behavioral label → N-dimensional grid
2. **Compute** pairwise overlap between bins via k-NN (or radius) neighborhood queries
3. **Build** weighted directed graph of overlap
4. **Calculate:** SI = 2 × (1 − d̄ / (n_bins − 1)) − 1, clamped ≥ 0
5. **Shuffle** labels `num_shuffles` times for null distribution

### Key Functions

- `compute_structure_index(data, label, n_bins, n_neighbors)` — Core computation. Accepts `StructureIndexConfig` or kwargs.
- `compute_structure_index_sweep(data, labels, n_neighbors_list, n_bins_list)` — Parameter sweep with HDF5 caching per combination
- `draw_overlap_graph(overlap_mat)` — NetworkX directed graph visualization
- `load_structure_index_results(save_path, dataset_name)` — Load cached results

### Performance

- FAISS-accelerated k-NN when available (guarded import).
- Redis caching via StorageManager.
- Progress bars for long computations.
- Incremental HDF5 persistence per parameter combination.

### Visualization (`topology/plotting.py`)

- 3-panel view: scatter + overlap heatmap + directed graph
- Parameter sweep line plots
- Multi-dataset comparison plots

All plots go through PlotGrid. Use `PlotSpec` objects and existing rendering infrastructure.
