# HDF5 File Structure Documentation

## Overview

The `neural_analysis` package uses a hierarchical HDF5 file structure to efficiently store and retrieve analysis results. This document explains the organization of HDF5 files for both **distribution comparisons** and **structure index** results.

## Hierarchical Structure

### General Pattern

```
file.h5
├── dataset_name_1/                    # Top-level group (comparison or analysis name)
│   ├── result_key_1/                  # Result identifier
│   │   ├── @attribute_1               # Scalar metadata (HDF5 attributes)
│   │   ├── @attribute_2
│   │   ├── array_1                    # Array dataset
│   │   └── array_2                    # Array dataset
│   └── result_key_2/
│       ├── @attribute_1
│       └── array_1
└── dataset_name_2/
    └── result_key_3/
        └── @attribute_1
```

**Key Components:**
- **Top-level groups**: Named by dataset, session, or comparison name
- **Result keys**: Unique identifiers for each analysis result
- **Attributes** (prefixed with `@`): Scalar values stored as HDF5 attributes
- **Datasets**: Array data stored as HDF5 datasets

---

## Distribution Comparisons

### File Structure

Default path: `./output/distribution_comparisons.h5`

```
distribution_comparisons.h5
├── session_001/                                    # comparison_name
│   ├── conditionA_vs_conditionB_wasserstein/      # result_key
│   │   ├── @dataset_i: "conditionA"               # First dataset name
│   │   ├── @dataset_j: "conditionB"               # Second dataset name
│   │   ├── @metric: "wasserstein"                 # Distance metric used
│   │   ├── @value: 0.523                          # Computed distance
│   │   ├── @n_samples_i: 100                      # Sample count (dataset i)
│   │   ├── @n_samples_j: 120                      # Sample count (dataset j)
│   │   ├── @n_features_i: 50                      # Feature count (dataset i)
│   │   └── @n_features_j: 50                      # Feature count (dataset j)
│   ├── conditionA_vs_conditionC_wasserstein/
│   │   ├── @dataset_i: "conditionA"
│   │   ├── @dataset_j: "conditionC"
│   │   ├── @metric: "wasserstein"
│   │   └── @value: 0.812
│   └── conditionA_vs_conditionB_procrustes/       # Shape metric with pairs
│       ├── @dataset_i: "conditionA"
│       ├── @dataset_j: "conditionB"
│       ├── @metric: "procrustes"
│       ├── @value: 0.156
│       ├── pair_indices: [50 × 2] int64           # Point-to-point indices
│       └── pair_values: [50] float64              # Correspondence values
└── experiment_002/                                 # Different comparison group
    └── trialA_vs_trialB_euclidean/
        ├── @dataset_i: "trialA"
        ├── @dataset_j: "trialB"
        ├── @metric: "euclidean"
        └── @value: 12.45
```

### Result Key Format

Format: `{dataset_i}_vs_{dataset_j}_{metric}`

**Examples:**
- `conditionA_vs_conditionB_wasserstein`
- `session1_vs_session2_ks`
- `neuron_pop1_vs_neuron_pop2_procrustes`

### Attributes vs Datasets

**Attributes** (scalars):
- `dataset_i`, `dataset_j`: Dataset names
- `metric`: Distance/similarity metric
- `value`: Computed comparison result
- `n_samples_i`, `n_samples_j`: Sample counts
- `n_features_i`, `n_features_j`: Feature dimensions

**Datasets** (arrays):
- `pair_indices`: Point-to-point correspondences (shape metrics only)
- `pair_values`: Correspondence values (shape metrics only)

---

## Structure Index Results

### File Structure

Default path: `./output/structure_indices.h5`

```
structure_indices.h5
├── session_001/                                    # dataset_name
│   ├── nbins10_nneigh15/                          # result_key (parameter combo)
│   │   ├── @structure_index: 0.847                # Main SI value
│   │   ├── @n_bins: 10                            # Binning parameter
│   │   ├── @n_neighbors: 15                       # k-NN parameter
│   │   ├── @distance_metric: "euclidean"          # Distance used
│   │   ├── @num_shuffles: 100                     # Shuffle iterations
│   │   ├── @n_samples: 1000                       # Data points
│   │   ├── @n_features: 50                        # Neural dimensions
│   │   ├── @label_dims: 2                         # Behavioral dimensions
│   │   ├── overlap_matrix: [10 × 10] float64     # Bin-to-bin overlap
│   │   └── shuffled_si: [100] float64            # Null distribution
│   ├── nbins10_nneigh20/                          # Different parameters
│   │   ├── @structure_index: 0.821
│   │   ├── @n_bins: 10
│   │   ├── @n_neighbors: 20
│   │   ├── overlap_matrix: [10 × 10] float64
│   │   └── shuffled_si: [100] float64
│   └── nbins15_nneigh15/
│       ├── @structure_index: 0.792
│       ├── @n_bins: 15
│       ├── @n_neighbors: 15
│       ├── overlap_matrix: [15 × 15] float64
│       └── shuffled_si: [100] float64
└── session_002/
    └── nbins10_nneigh15/
        ├── @structure_index: 0.623
        ├── overlap_matrix: [10 × 10] float64
        └── shuffled_si: [100] float64
```

### Result Key Format

Format: `nbins{n_bins}_nneigh{n_neighbors}`

**Examples:**
- `nbins10_nneigh15`
- `nbins20_nneigh30`
- `nbins5_nneigh10`

### Attributes vs Datasets

**Attributes** (scalars):
- `structure_index`: Main SI value
- `n_bins`: Number of bins per dimension
- `n_neighbors`: k-NN parameter
- `distance_metric`: Distance metric used
- `num_shuffles`: Number of shuffle iterations
- `n_samples`: Number of data points
- `n_features`: Neural dimensionality
- `label_dims`: Behavioral dimensionality

**Datasets** (arrays):
- `overlap_matrix`: Bin-to-bin overlap matrix [n_bins × n_bins]
- `shuffled_si`: Null distribution from shuffling [num_shuffles]

---

## Access Examples

### Python API

#### Distribution Comparisons

```python
from neural_analysis.metrics.distributions import (
    load_distribution_comparisons,
    get_comparison_summary
)

# Load all comparisons from a session
results = load_distribution_comparisons(
    "output/comparisons.h5",
    comparison_name="session_001"
)

# Filter by specific datasets and metric
results = load_distribution_comparisons(
    "output/comparisons.h5",
    comparison_name="session_001",
    dataset_i="conditionA",
    metric="wasserstein"
)

# Get summary DataFrame
df = get_comparison_summary("output/comparisons.h5")
print(df[['dataset_i', 'dataset_j', 'metric', 'value']])
```

#### Structure Index

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

# Get summary of all structure index results
df = get_hdf5_result_summary("output/structure_indices.h5")
print(df[['dataset_name', 'n_bins', 'n_neighbors', 'structure_index']])
```

### Direct HDF5 Access

```python
import h5py

# Open file
with h5py.File("output/comparisons.h5", "r") as f:
    # Navigate hierarchy
    session = f["session_001"]
    result = session["conditionA_vs_conditionB_wasserstein"]
    
    # Read attributes
    value = result.attrs["value"]
    metric = result.attrs["metric"]
    
    # Read datasets (if present)
    if "pair_indices" in result:
        pairs = result["pair_indices"][:]
```

---

## Design Principles

### 1. Hierarchical Organization
- Groups organize related results
- Result keys uniquely identify analyses
- Easy navigation and filtering

### 2. Scalars as Attributes
- Metadata stored as HDF5 attributes
- Fast access without loading large arrays
- Queryable for filtering

### 3. Arrays as Datasets
- Large numerical data stored efficiently
- Compressed by default (gzip)
- Lazy loading possible

### 4. Consistent Naming
- Predictable result key formats
- Standard attribute names
- Clear semantic meaning

### 5. Incremental Updates
- Add results without rewriting file
- Multiple comparison groups coexist
- Caching-friendly for batch processing

---

## Benefits

✅ **Efficient Storage**: Compressed arrays, minimal overhead  
✅ **Fast Filtering**: Query by attributes without loading data  
✅ **Scalability**: Handles large-scale batch analyses  
✅ **Flexibility**: Multiple comparison groups in one file  
✅ **Reproducibility**: Complete metadata with results  
✅ **Interoperability**: Standard HDF5 format, language-agnostic

---

## See Also

- [Distribution Comparisons Documentation](distributions_README.md)
- [Structure Index Documentation](structure_index_README.md)
- [HDF5 Examples Notebook](../examples/io_h5io_examples.ipynb)
