# Structure Index Documentation

## Overview

The **Structure Index (SI)** is a dimensionless metric that quantifies how well neural population activity organizes according to external behavioral or stimulus variables. It captures the degree to which neural representations form a structured manifold aligned with task-relevant features.

**Key Concept:** If neural activity truly represents behavioral states (e.g., position, velocity, direction), then similar behavioral states should correspond to similar neural patterns, forming a coherent manifold structure.

---

## Mathematical Background

### Core Idea

The Structure Index quantifies manifold coherence by:

1. **Binning** neural data according to behavioral labels
2. **Computing overlap** between neural activity patterns in neighboring bins
3. **Building a graph** where nodes are bins and edges represent neural overlap
4. **Measuring coherence** via graph connectivity metrics
5. **Comparing to chance** using shuffled null distributions

### Algorithm Steps

```
1. Bin behavioral space (e.g., position grid)
2. For each bin pair (i, j):
   - Collect neural states in each bin
   - Compute k-NN overlap between neural clouds
   - Weight by behavioral distance
3. Construct weighted directed overlap graph
4. Compute structure index from graph properties
5. Generate null distribution via label shuffling
```

### Structure Index Formula

The SI is computed from the overlap matrix **O**, where O[i,j] represents the fraction of k-nearest neighbors from bin i that come from bin j:

```
SI = f(O, distance_matrix)
```

The exact formula depends on the implementation, but generally captures how well neural overlaps respect behavioral proximity.

---

## Key Parameters

### `n_bins` - Binning Resolution

**Description:** Number of bins per behavioral dimension

**Impact:**
- **Low (5-10):** Coarse binning, robust to noise, may miss fine structure
- **Medium (10-20):** Good balance for most datasets
- **High (20+):** Fine-grained, sensitive to small-scale structure, needs more data

**Guidelines:**
- Start with 10 bins for 1D, 10-15 for 2D
- Ensure adequate samples per bin (>20 recommended)
- Higher dimensions need fewer bins to avoid sparsity

### `n_neighbors` - Neighborhood Size

**Description:** Number of nearest neighbors for overlap computation

**Impact:**
- **Low (5-10):** Local structure, sensitive to noise
- **Medium (15-30):** Captures local manifold structure
- **High (50+):** Global structure, may smooth over details

**Guidelines:**
- Start with 15-20 neighbors
- Should be << samples per bin
- Scale with dataset size

### `distance_metric` - Behavioral Distance

**Description:** Metric for measuring behavioral similarity

**Options:**
- **`"euclidean"`**: Straight-line distance (default)
- **`"geodesic"`**: Manifold distance via Isomap

**Guidelines:**
- Use Euclidean for simple behavioral spaces
- Use geodesic for nonlinear manifolds
- Geodesic is slower but captures curved structure

### `num_shuffles` - Statistical Testing

**Description:** Number of label permutations for null distribution

**Default:** 100

**Guidelines:**
- 100 shuffles for quick testing
- 1000+ for publication-quality p-values
- Trade-off: computation time vs statistical precision

---

## Usage Examples

### Basic Structure Index

```python
import numpy as np
from neural_analysis.topology import compute_structure_index

# Neural activity: (n_samples, n_neurons)
data = np.random.randn(1000, 50)

# Behavioral labels: (n_samples, n_dims)
labels = np.random.randn(1000, 2)  # 2D position

# Compute structure index
SI, bin_info, overlap_mat, shuffled_SI = compute_structure_index(
    data=data,
    label=labels,
    n_bins=10,
    n_neighbors=15,
    distance_metric="euclidean",
    num_shuffles=100
)

print(f"Structure Index: {SI:.3f}")
print(f"Chance level (mean): {np.mean(shuffled_SI):.3f}")
print(f"Chance level (std): {np.std(shuffled_SI):.3f}")

# Statistical significance
p_value = np.mean(shuffled_SI >= SI)
print(f"P-value: {p_value:.4f}")
```

### Parameter Sweep with HDF5 Saving

```python
from neural_analysis.topology import compute_structure_index_sweep

# Define parameter grid
n_neighbors_list = [10, 15, 20, 30]
n_bins_list = [5, 10, 15, 20]

# Run sweep with automatic saving
results = compute_structure_index_sweep(
    data=data,
    labels=labels,
    dataset_name="session_001",
    save_path="output/structure_indices.h5",  # Default if None
    n_neighbors_list=n_neighbors_list,
    n_bins_list=n_bins_list,
    distance_metric="euclidean",
    num_shuffles=100,
    verbose=True
)

# Results is dict: {(n_bins, n_neighbors): {'structure_index': SI, ...}}
for params, result in results.items():
    n_bins, n_neighbors = params
    SI = result['structure_index']
    print(f"n_bins={n_bins}, n_neighbors={n_neighbors}: SI={SI:.3f}")
```

**Default Save Path:** `./output/structure_indices.h5`

---

## Interpreting Results

### Structure Index Value

| SI Range | Interpretation |
|----------|----------------|
| **SI < 0.5** | Weak or no structure; neural activity doesn't track behavioral variables |
| **0.5 ≤ SI < 0.7** | Moderate structure; some alignment between neural and behavioral spaces |
| **0.7 ≤ SI < 0.85** | Strong structure; clear manifold organization |
| **SI ≥ 0.85** | Very strong structure; tight coupling between neural and behavioral states |

**Note:** Interpretation depends on dataset characteristics and parameters.

### Statistical Significance

```python
# Compute z-score
z_score = (SI - np.mean(shuffled_SI)) / np.std(shuffled_SI)
print(f"Z-score: {z_score:.2f}")

# Empirical p-value
p_value = np.mean(shuffled_SI >= SI)
print(f"P-value: {p_value:.4f}")

# Significant if p < 0.05 or z > 2
is_significant = (p_value < 0.05) or (z_score > 2)
```

### Overlap Matrix

The overlap matrix **O** shows how neural activity in one behavioral bin overlaps with other bins:

```python
import matplotlib.pyplot as plt

# Visualize overlap matrix
plt.figure(figsize=(8, 6))
plt.imshow(overlap_mat, cmap='viridis', aspect='auto')
plt.colorbar(label='Overlap Fraction')
plt.xlabel('Target Bin')
plt.ylabel('Source Bin')
plt.title('Neural Overlap Matrix')
plt.show()
```

**Interpretation:**
- **Diagonal dominance:** Strong structure (bins overlap with themselves)
- **Off-diagonal spread:** Weak structure (bins overlap broadly)
- **Block structure:** Clusters in behavioral space

---

## Visualization

### Overlap Graph

```python
from neural_analysis.topology import draw_overlap_graph
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 10))
draw_overlap_graph(
    overlap_mat,
    ax=ax,
    node_cmap=plt.cm.tab10,
    edge_cmap=plt.cm.Greys,
    scale_edges=5
)
plt.title('Structure Index Overlap Graph')
plt.show()
```

**Elements:**
- **Nodes:** Behavioral bins
- **Edges:** Neural overlap (directed, weighted)
- **Edge thickness:** Overlap strength
- **Node colors:** Bin identity

### Null Distribution

```python
# Compare observed SI to null distribution
plt.figure(figsize=(8, 5))
plt.hist(shuffled_SI, bins=30, alpha=0.7, label='Null (shuffled)')
plt.axvline(SI, color='red', linestyle='--', linewidth=2, label=f'Observed (SI={SI:.3f})')
plt.xlabel('Structure Index')
plt.ylabel('Frequency')
plt.legend()
plt.title('Structure Index vs Null Distribution')
plt.show()
```

---

## HDF5 Storage

### File Structure

Results are stored hierarchically:

```
output/structure_indices.h5
├── session_001/
│   ├── nbins10_nneigh15/
│   │   ├── @structure_index: 0.847
│   │   ├── @n_bins: 10
│   │   ├── @n_neighbors: 15
│   │   ├── overlap_matrix: [10 × 10]
│   │   └── shuffled_si: [100]
│   └── nbins20_nneigh30/
│       ├── @structure_index: 0.792
│       └── ...
└── session_002/
    └── nbins10_nneigh15/
        └── ...
```

See [HDF5 Structure Documentation](hdf5_structure_README.md) for details.

### Loading Results

```python
from neural_analysis.utils.io import (
    load_results_from_hdf5_dataset,
    get_hdf5_result_summary
)

# Load all results for a dataset
results = load_results_from_hdf5_dataset(
    "output/structure_indices.h5",
    dataset_name="session_001"
)

# Load specific parameter combination
results = load_results_from_hdf5_dataset(
    "output/structure_indices.h5",
    dataset_name="session_001",
    result_key="nbins10_nneigh15"
)

# Access data
si_value = results["session_001"]["nbins10_nneigh15"]["attributes"]["structure_index"]
overlap = results["session_001"]["nbins10_nneigh15"]["arrays"]["overlap_matrix"]
shuffled = results["session_001"]["nbins10_nneigh15"]["arrays"]["shuffled_si"]
```

### Summary DataFrame

```python
# Get summary of all results
df = get_hdf5_result_summary("output/structure_indices.h5")

print(df[['dataset_name', 'n_bins', 'n_neighbors', 'structure_index']])

# Analyze parameter effects
import seaborn as sns
sns.scatterplot(data=df, x='n_neighbors', y='structure_index', hue='n_bins')
plt.show()
```

---

## Batch Processing Workflow

### 1. Generate Synthetic Data

```python
from neural_analysis.data.synthetic_data import generate_data

# Place cells
place_data, place_labels = generate_data(
    'place_cells',
    n_samples=1000,
    n_features=50,
    noise=0.1
)

# Grid cells
grid_data, grid_labels = generate_data(
    'grid_cells',
    n_samples=1000,
    n_features=50
)

# Random (control)
random_data, random_labels = generate_data(
    'classification',
    n_samples=1000,
    n_features=50
)
```

### 2. Compute Structure Index for Each

```python
datasets = {
    "place_cells": (place_data, place_labels['position']),
    "grid_cells": (grid_data, grid_labels['position']),
    "random": (random_data, random_labels),
}

for name, (data, labels) in datasets.items():
    results = compute_structure_index_sweep(
        data=data,
        labels=labels,
        dataset_name=name,
        save_path="output/structure_indices.h5",
        n_neighbors_list=[10, 15, 20],
        n_bins_list=[10, 15, 20],
        verbose=False
    )
    print(f"{name}: Computed {len(results)} parameter combinations")
```

### 3. Load and Compare

```python
# Load all results
df = get_hdf5_result_summary("output/structure_indices.h5")

# Compare datasets
print(df.groupby('dataset_name')['structure_index'].describe())

# Best parameters for each dataset
best_idx = df.groupby('dataset_name')['structure_index'].idxmax()
best_params = df.loc[best_idx]
print(best_params[['dataset_name', 'n_bins', 'n_neighbors', 'structure_index']])
```

---

## Parameter Selection Tips

### Starting Points

**For most datasets:**
- `n_bins = 10` (1D), `10-15` (2D)
- `n_neighbors = 15-20`
- `distance_metric = "euclidean"`
- `num_shuffles = 100` (testing), `1000` (final)

### Tuning Guidelines

#### Sample Size

| n_samples | n_bins | n_neighbors |
|-----------|--------|-------------|
| < 500 | 5-8 | 10-15 |
| 500-2000 | 10-15 | 15-25 |
| > 2000 | 15-20 | 20-40 |

#### Dimensionality

| Label Dims | n_bins (per dim) | Total Bins |
|------------|------------------|------------|
| 1D | 10-20 | 10-20 |
| 2D | 10-15 | 100-225 |
| 3D | 5-10 | 125-1000 |

**Rule:** Total bins should be < n_samples / 20

#### Signal Strength

**Strong signal (e.g., place cells):**
- Can use more bins
- Smaller neighborhoods OK
- Euclidean distance sufficient

**Weak signal (e.g., mixed populations):**
- Fewer bins (robust to noise)
- Larger neighborhoods
- Consider geodesic distance

### Parameter Sweep Strategy

```python
# 1. Quick scan
n_bins_coarse = [5, 10, 15, 20]
n_neighbors_coarse = [10, 20, 30]

# 2. Refine around best
best_bins = 10  # from coarse scan
best_neighbors = 20

n_bins_fine = [8, 10, 12, 14]
n_neighbors_fine = [15, 20, 25, 30]

# 3. Final parameters
final_bins = 10
final_neighbors = 20
final_shuffles = 1000
```

---

## Common Issues and Solutions

### Issue 1: Low Structure Index

**Possible causes:**
- Neural activity doesn't track behavioral variables
- Too few samples per bin
- Inappropriate parameters
- Noisy labels

**Solutions:**
- Check data quality and alignment
- Use fewer bins or more data
- Try parameter sweep
- Filter outliers

### Issue 2: High but Non-significant SI

**Possible causes:**
- Small null distribution variance
- Too few shuffles
- Overfitting to noise

**Solutions:**
- Increase `num_shuffles` to 1000+
- Check if SI >> mean(shuffled_SI)
- Compute z-score

### Issue 3: Empty Bins

**Error:** `ValueError: Empty bin detected`

**Solutions:**
- Reduce `n_bins`
- Remove outliers
- Check label distribution
- Use `discrete_label=True` if appropriate

### Issue 4: Memory Issues

**Large datasets cause OOM errors**

**Solutions:**
- Subsample data
- Use `data_indices` parameter
- Reduce `n_bins` and `n_neighbors`
- Process in batches

---

## Complete Example

```python
import numpy as np
import matplotlib.pyplot as plt
from neural_analysis.topology import (
    compute_structure_index,
    compute_structure_index_sweep,
    draw_overlap_graph
)
from neural_analysis.utils.io import get_hdf5_result_summary

# Generate synthetic place cell data
from neural_analysis.data.synthetic_data import generate_data
data, labels = generate_data(
    'place_cells',
    n_samples=1000,
    n_features=50,
    noise=0.1,
    seed=42
)
position = labels['position']  # Extract position labels

# 1. Single computation
SI, bin_info, overlap_mat, shuffled_SI = compute_structure_index(
    data=data,
    label=position,
    n_bins=10,
    n_neighbors=15,
    num_shuffles=100
)

print(f"Structure Index: {SI:.3f}")
print(f"Z-score: {(SI - np.mean(shuffled_SI)) / np.std(shuffled_SI):.2f}")

# 2. Visualize overlap graph
fig, ax = plt.subplots(figsize=(10, 10))
draw_overlap_graph(overlap_mat, ax=ax)
plt.title(f'Structure Index = {SI:.3f}')
plt.savefig('overlap_graph.png', dpi=150, bbox_inches='tight')

# 3. Parameter sweep
results = compute_structure_index_sweep(
    data=data,
    labels=position,
    dataset_name="place_cells_example",
    save_path="output/si_results.h5",
    n_neighbors_list=[10, 15, 20, 25],
    n_bins_list=[8, 10, 12, 15],
    num_shuffles=100
)

# 4. Analyze results
df = get_hdf5_result_summary("output/si_results.h5")

# Find best parameters
best_idx = df['structure_index'].idxmax()
best_result = df.loc[best_idx]
print(f"\nBest parameters:")
print(f"  n_bins = {best_result['n_bins']}")
print(f"  n_neighbors = {best_result['n_neighbors']}")
print(f"  SI = {best_result['structure_index']:.3f}")

# 5. Visualize parameter effects
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Effect of n_bins
df.groupby('n_bins')['structure_index'].mean().plot(ax=ax1, marker='o')
ax1.set_xlabel('Number of Bins')
ax1.set_ylabel('Mean Structure Index')
ax1.set_title('Effect of Binning Resolution')

# Effect of n_neighbors
df.groupby('n_neighbors')['structure_index'].mean().plot(ax=ax2, marker='o')
ax2.set_xlabel('Number of Neighbors')
ax2.set_ylabel('Mean Structure Index')
ax2.set_title('Effect of Neighborhood Size')

plt.tight_layout()
plt.savefig('parameter_effects.png', dpi=150, bbox_inches='tight')
```

---

## References

1. **Bernardi et al. (2020).** "The Geometry of Abstraction in the Hippocampus and Prefrontal Cortex." *Cell*, 183(4), 954-967.
   - Original Structure Index paper
   - Applications to neural manifolds

2. **Chung & Abbott (1993).** "A Compact Firing Rate Model of the CA3 Region."
   - k-NN overlap methods
   - Neural manifold theory

---

## See Also

- [Distribution Comparisons Documentation](distributions_README.md)
- [HDF5 File Structure](hdf5_structure_README.md)
- [Structure Index Examples Notebook](../examples/structure_index_examples.ipynb)
