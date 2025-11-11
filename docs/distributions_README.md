# Distribution Comparison Module

## Overview

The `neural_analysis.metrics.distributions` module provides comprehensive tools for comparing probability distributions and neural population representations. It supports multiple distance metrics for both distribution-level and shape-level comparisons, with built-in HDF5 I/O for efficient batch processing.

**Key Features:**
- 📊 Multiple distribution metrics (Wasserstein, Kolmogorov-Smirnov, Jensen-Shannon, etc.)
- 🔷 Shape distance methods (Procrustes, one-to-one matching, soft optimal transport)
- 💾 Automatic HDF5 saving/loading with hierarchical organization
- 🔍 Filtering and querying capabilities
- 🚀 Batch processing with caching
- 📈 Summary statistics and DataFrame outputs

---

## Distribution Metrics

### Available Metrics

| Metric | Description | Use Case | Output |
|--------|-------------|----------|--------|
| `wasserstein` | Earth Mover's Distance | General distribution comparison | float |
| `ks` (Kolmogorov-Smirnov) | Maximum CDF difference | Distribution equality testing | float |
| `js` (Jensen-Shannon) | Information divergence | Symmetric probability comparison | float |
| `euclidean` | L2 distance between means | Simple centroid comparison | float |
| `cosine` | Cosine similarity | Directional similarity | float |
| `mahalanobis` | Covariance-weighted distance | Accounting for correlations | float |

### Distribution Comparison

```python
from neural_analysis.metrics.distributions import compare_distributions

# 1D distributions (e.g., firing rates)
dist1 = np.random.randn(1000)  # Neural activity population 1
dist2 = np.random.randn(1000) + 0.5  # Population 2 (shifted)

# Compute Wasserstein distance
distance = compare_distributions(dist1, dist2, metric="wasserstein")
print(f"Wasserstein distance: {distance:.3f}")

# Kolmogorov-Smirnov test
ks_stat = compare_distributions(dist1, dist2, metric="ks")
print(f"KS statistic: {ks_stat:.3f}")

# Jensen-Shannon divergence
js_div = compare_distributions(dist1, dist2, metric="js")
print(f"JS divergence: {js_div:.3f}")
```

### Multi-dimensional Distributions

```python
# Neural population activity (samples × neurons)
pop1 = np.random.randn(500, 50)  # 500 timepoints, 50 neurons
pop2 = np.random.randn(500, 50)

# Multidimensional Wasserstein distance
distance = compare_distributions(pop1, pop2, metric="wasserstein")
```

---

## Shape Distance Methods

Shape distances compare the geometric structure of neural population representations, accounting for point-to-point correspondences.

### Available Methods

| Method | Description | Best For | Returns Pairs |
|--------|-------------|----------|---------------|
| `procrustes` | Orthogonal Procrustes alignment | Rotational invariance | ✅ |
| `one-to-one` | Optimal bipartite matching | Permutation invariance | ✅ |
| `soft-matching` | Entropic optimal transport | Partial matches, regularization | ✅ |

### Shape Distance Comparison

```python
from neural_analysis.metrics.distributions import compare_distributions

# Neural representations (neurons × features)
rep1 = np.random.randn(50, 10)  # 50 neurons, 10-dim embedding
rep2 = np.random.randn(50, 10)

# Procrustes distance with point correspondences
distance, pairs = compare_distributions(
    rep1, rep2, 
    metric="procrustes"
)
print(f"Procrustes distance: {distance:.3f}")
print(f"Number of matched pairs: {len(pairs)}")

# One-to-one matching with custom metric
distance, pairs = compare_distributions(
    rep1, rep2,
    metric="one-to-one",
    matching_metric="euclidean"
)

# Soft matching with regularization
distance, pairs = compare_distributions(
    rep1, rep2,
    metric="soft-matching",
    approx=True,
    reg=0.1
)
```

---

## Saving Results to HDF5

### Single Comparison

```python
# Save individual comparison
distance = compare_distributions(
    pop1, pop2,
    metric="wasserstein",
    dataset_i="condition_A",
    dataset_j="condition_B",
    comparison_name="experiment_001",
    save_path="output/comparisons.h5"
)

# Shape comparison with pairs saved
distance, pairs = compare_distributions(
    rep1, rep2,
    metric="procrustes",
    dataset_i="session1",
    dataset_j="session2",
    comparison_name="shape_analysis",
    save_path="output/comparisons.h5"
)
```

**Default Path:** `./output/distribution_comparisons.h5` (when `save_path` not specified)

### Batch Comparisons

```python
from neural_analysis.metrics.distributions import (
    pairwise_distribution_comparison_batch
)

# Multiple datasets
data = {
    "condition_A": np.random.randn(1000, 50),
    "condition_B": np.random.randn(1000, 50),
    "condition_C": np.random.randn(1000, 50),
}

# Compare all pairs with multiple metrics
metrics = ["wasserstein", "euclidean", "cosine"]
df = pairwise_distribution_comparison_batch(
    data,
    metrics=metrics,
    comparison_name="session_001",
    save_path="output/comparisons.h5"
)

print(df.head())
```

**Output DataFrame:**

| dataset_i | dataset_j | metric | value | n_samples_i | n_samples_j | ... |
|-----------|-----------|--------|-------|-------------|-------------|-----|
| condition_A | condition_B | wasserstein | 0.523 | 1000 | 1000 | ... |
| condition_A | condition_C | wasserstein | 0.812 | 1000 | 1000 | ... |
| condition_B | condition_C | wasserstein | 0.691 | 1000 | 1000 | ... |

### Batch with Shape Metrics

```python
# Shape comparison of neural representations
data = {
    "session_1": np.random.randn(50, 10),
    "session_2": np.random.randn(50, 10),
    "session_3": np.random.randn(50, 10),
}

# Multiple shape metrics with custom parameters
metrics = {
    'procrustes': {},
    'one-to-one': {'metric': 'euclidean'},
    'soft-matching': {'reg': 0.05, 'approx': True}
}

df = pairwise_distribution_comparison_batch(
    data,
    metrics=metrics,
    comparison_name="shape_analysis",
    save_path="output/shape_comparisons.h5"
)
```

---

## Loading and Filtering Results

### Load All Comparisons

```python
from neural_analysis.metrics.distributions import (
    load_distribution_comparisons,
    get_comparison_summary
)

# Load entire comparison group
results = load_distribution_comparisons(
    "output/comparisons.h5",
    comparison_name="session_001"
)

# Access specific result
result_key = "conditionA_vs_conditionB_wasserstein"
if result_key in results["session_001"]:
    result_data = results["session_001"][result_key]
    value = result_data["attributes"]["value"]
    metric = result_data["attributes"]["metric"]
```

### Filter by Attributes

```python
# Filter by first dataset
results = load_distribution_comparisons(
    "output/comparisons.h5",
    comparison_name="session_001",
    dataset_i="condition_A"
)

# Filter by metric
results = load_distribution_comparisons(
    "output/comparisons.h5",
    dataset_i="condition_A",
    dataset_j="condition_B",
    metric="wasserstein"
)

# Multiple filters
results = load_distribution_comparisons(
    "output/comparisons.h5",
    comparison_name="experiment_001",
    metric="procrustes"
)
```

### Summary DataFrames

```python
# Get summary of all comparisons in file
df = get_comparison_summary("output/comparisons.h5")
print(df.columns)
# ['comparison_name', 'result_key', 'dataset_i', 'dataset_j', 
#  'metric', 'value', 'n_samples_i', 'n_samples_j', ...]

# Filter specific comparison group
df_session = get_comparison_summary(
    "output/comparisons.h5",
    comparison_name="session_001"
)

# Analyze results
print(df_session.groupby('metric')['value'].describe())
```

---

## Caching and Regeneration

### Automatic Caching

```python
# First call: computes and saves
df = pairwise_distribution_comparison_batch(
    data,
    metrics=["wasserstein", "euclidean"],
    comparison_name="session_001",
    save_path="output/comparisons.h5"
)

# Second call: loads cached results (much faster!)
df = pairwise_distribution_comparison_batch(
    data,
    metrics=["wasserstein", "euclidean"],
    comparison_name="session_001",
    save_path="output/comparisons.h5"
)
```

### Force Regeneration

```python
# Recompute all comparisons
df = pairwise_distribution_comparison_batch(
    data,
    metrics=["wasserstein"],
    comparison_name="session_001",
    save_path="output/comparisons.h5",
    regenerate=True  # Force recomputation
)
```

### Incremental Updates

```python
# Add new metric to existing comparisons
df = pairwise_distribution_comparison_batch(
    data,
    metrics=["cosine"],  # New metric
    comparison_name="session_001",
    save_path="output/comparisons.h5"
)
# Only computes missing comparisons, keeps existing ones
```

---

## Advanced Usage

### Group Comparisons

```python
from neural_analysis.metrics.distributions import (
    compare_distribution_groups
)

# Compare within groups (e.g., trial variability)
groups = {
    "trial_1": np.random.randn(1000, 50),
    "trial_2": np.random.randn(1000, 50),
    "trial_3": np.random.randn(1000, 50),
}

# Within-group comparisons
within_dists = compare_distribution_groups(
    groups,
    compare_type="inside",
    metric="wasserstein"
)

# Between-group comparisons
between_dists = compare_distribution_groups(
    groups,
    compare_type="between",
    metric="euclidean"
)
```

### Custom Metric Parameters

```python
# Mahalanobis with custom covariance
distance = compare_distributions(
    pop1, pop2,
    metric="mahalanobis",
    VI=custom_covariance_inv
)

# Soft matching with high regularization
distance, pairs = compare_distributions(
    rep1, rep2,
    metric="soft-matching",
    reg=0.5,
    approx=False  # Exact solver
)
```

---

## HDF5 File Structure

Comparisons are stored in a hierarchical structure:

```
output/distribution_comparisons.h5
├── session_001/
│   ├── conditionA_vs_conditionB_wasserstein/
│   │   ├── @dataset_i: "conditionA"
│   │   ├── @dataset_j: "conditionB"
│   │   ├── @metric: "wasserstein"
│   │   ├── @value: 0.523
│   │   ├── @n_samples_i: 1000
│   │   └── @n_samples_j: 1000
│   └── conditionA_vs_conditionB_procrustes/
│       ├── @value: 0.156
│       ├── pair_indices: [50 × 2] int64
│       └── pair_values: [50] float64
└── experiment_002/
    └── trialA_vs_trialB_euclidean/
        └── @value: 12.45
```

See [HDF5 Structure Documentation](hdf5_structure_README.md) for details.

---

## Complete Example

```python
import numpy as np
from neural_analysis.metrics.distributions import (
    compare_distributions,
    pairwise_distribution_comparison_batch,
    load_distribution_comparisons,
    get_comparison_summary
)

# Generate synthetic neural data
np.random.seed(42)
data = {
    "baseline": np.random.randn(500, 50),
    "stimulus": np.random.randn(500, 50) + 0.5,
    "recovery": np.random.randn(500, 50) + 0.2,
}

# 1. Single comparison
dist = compare_distributions(
    data["baseline"],
    data["stimulus"],
    metric="wasserstein",
    dataset_i="baseline",
    dataset_j="stimulus",
    comparison_name="experiment_001",
    save_path="results/comparisons.h5"
)
print(f"Baseline vs Stimulus: {dist:.3f}")

# 2. Batch comparison with multiple metrics
metrics = ["wasserstein", "euclidean", "cosine"]
df = pairwise_distribution_comparison_batch(
    data,
    metrics=metrics,
    comparison_name="experiment_001",
    save_path="results/comparisons.h5"
)
print("\nAll pairwise comparisons:")
print(df[['dataset_i', 'dataset_j', 'metric', 'value']])

# 3. Load and filter
results = load_distribution_comparisons(
    "results/comparisons.h5",
    comparison_name="experiment_001",
    dataset_i="baseline"
)
print(f"\nLoaded {len(results['experiment_001'])} comparisons")

# 4. Generate summary
summary = get_comparison_summary("results/comparisons.h5")
print("\nSummary statistics:")
print(summary.groupby('metric')['value'].agg(['mean', 'std', 'min', 'max']))
```

---

## Performance Tips

### 1. Use Batch Processing
```python
# ❌ Slow: Individual comparisons
for i, j in pairs:
    compare_distributions(data[i], data[j], ...)

# ✅ Fast: Batch with caching
pairwise_distribution_comparison_batch(data, metrics, ...)
```

### 2. Filter Early
```python
# ❌ Load all, then filter
results = load_distribution_comparisons("file.h5")
filtered = [r for r in results if r['metric'] == 'wasserstein']

# ✅ Filter during load
results = load_distribution_comparisons("file.h5", metric="wasserstein")
```

### 3. Leverage Caching
```python
# First run: slow
df = pairwise_distribution_comparison_batch(data, metrics, save_path="cache.h5")

# Subsequent runs: instant (loads from HDF5)
df = pairwise_distribution_comparison_batch(data, metrics, save_path="cache.h5")
```

### 4. Use Approximate Methods
```python
# Exact: slow for large datasets
distance, pairs = compare_distributions(
    large_rep1, large_rep2,
    metric="soft-matching",
    approx=False
)

# Approximate: much faster
distance, pairs = compare_distributions(
    large_rep1, large_rep2,
    metric="soft-matching",
    approx=True  # Use Sinkhorn iterations
)
```

---

## API Reference

### Core Functions

- **`compare_distributions()`**: Single comparison with optional saving
- **`pairwise_distribution_comparison_batch()`**: All-pairs batch comparison
- **`compare_distribution_groups()`**: Within/between group comparisons

### I/O Functions

- **`save_distribution_comparison()`**: Save single result to HDF5
- **`load_distribution_comparisons()`**: Load with filtering
- **`get_comparison_summary()`**: Generate summary DataFrame

### Metrics

**Distribution:** `wasserstein`, `ks`, `js`, `euclidean`, `cosine`, `mahalanobis`  
**Shape:** `procrustes`, `one-to-one`, `soft-matching`

---

## See Also

- [HDF5 File Structure](hdf5_structure_README.md)
- [Structure Index Documentation](structure_index_README.md)
- [Examples Notebook](../examples/structure_index_examples.ipynb)
