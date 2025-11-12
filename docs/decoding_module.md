# Decoding Module Documentation

## Overview

The `neural_analysis.learning.decoding` module provides functions for decoding behavioral variables (position, head direction, etc.) from neural activity or low-dimensional embeddings.

**Key Features:**
- Population vector decoder (weighted average, peak methods)
- k-Nearest Neighbors (k-NN) decoder
- Cross-validation support
- High-D vs Low-D comparison tools
- Works on both raw activity and embeddings

## Installation Requirements

```bash
pip install numpy scikit-learn
```

## Quick Start

```python
from neural_analysis.data.synthetic_data import generate_place_cells
from neural_analysis.learning.decoding import knn_decoder, population_vector_decoder

# Generate synthetic place cells
activity, meta = generate_place_cells(n_cells=50, n_samples=1000, n_dims=2)

# Split into train/test
train_act, test_act = activity[:700], activity[700:]
train_pos, test_pos = meta['positions'][:700], meta['positions'][700:]

# Decode with k-NN
decoded_knn = knn_decoder(train_act, train_pos, test_act, k=5)

# Decode with population vector
decoded_pv = population_vector_decoder(
    test_act, meta['field_centers'], method='weighted_average'
)

# Compute decoding error
import numpy as np
error_knn = np.linalg.norm(decoded_knn - test_pos, axis=1).mean()
error_pv = np.linalg.norm(decoded_pv - test_pos, axis=1).mean()

print(f"k-NN error: {error_knn:.4f}")
print(f"Population vector error: {error_pv:.4f}")
```

## Functions

### `population_vector_decoder`

Classic neuroscience method that uses known tuning properties (field centers) to decode.

**Parameters:**
- `activity`: Neural activity, shape `(n_samples, n_cells)`
- `field_centers`: Preferred locations, shape `(n_cells, n_dims)`
- `method`: `'weighted_average'` or `'peak'`

**Returns:**
- `decoded_positions`: Decoded positions, shape `(n_samples, n_dims)`

**Example:**
```python
from neural_analysis.learning.decoding import population_vector_decoder

decoded = population_vector_decoder(
    activity, meta['field_centers'], method='weighted_average'
)
```

**Methods:**
- **Weighted average**: Weights field centers by activity (more robust)
- **Peak**: Uses position of most active cell (simpler, less accurate)

---

### `knn_decoder`

k-Nearest Neighbors decoder that works on both high-D activity and low-D embeddings.

**Parameters:**
- `train_activity`: Training data, shape `(n_train, n_features)`
- `train_labels`: Training labels, shape `(n_train, n_dims)`
- `test_activity`: Test data, shape `(n_test, n_features)`
- `k`: Number of neighbors (default: 5)
- `weights`: `'uniform'` or `'distance'` (default: `'distance'`)
- `metric`: Distance metric (default: `'euclidean'`)

**Returns:**
- `decoded_labels`: Decoded labels, shape `(n_test, n_dims)`

**Example:**
```python
from neural_analysis.learning.decoding import knn_decoder

# Decode from high-D activity
decoded_highd = knn_decoder(train_act, train_pos, test_act, k=5)

# Decode from low-D embedding
import umap
embedding = umap.UMAP(n_components=3).fit_transform(activity)
train_emb, test_emb = embedding[:700], embedding[700:]
decoded_lowd = knn_decoder(train_emb, train_pos, test_emb, k=10)
```

**Weight Options:**
- **uniform**: All neighbors weighted equally
- **distance**: Weight by inverse distance (closer neighbors have more influence)

---

### `cross_validated_knn_decoder`

k-NN decoder with k-fold cross-validation for robust performance estimation.

**Parameters:**
- `activity`: Neural activity or embedding, shape `(n_samples, n_features)`
- `labels`: True labels, shape `(n_samples, n_dims)`
- `k`: Number of neighbors (default: 5)
- `n_folds`: Number of CV folds (default: 5)
- `weights`: `'uniform'` or `'distance'`
- `metric`: Distance metric
- `return_predictions`: Return predictions for each fold (default: False)

**Returns:**
- Dictionary with:
  - `mean_r2`, `std_r2`: R² scores
  - `mean_mse`, `std_mse`: Mean squared errors
  - `mean_error`, `std_error`: Euclidean errors
  - `r2_scores`, `mse_scores`, `euclidean_errors`: Per-fold scores
  - `predictions`: (optional) Predictions for each fold

**Example:**
```python
from neural_analysis.learning.decoding import cross_validated_knn_decoder

metrics = cross_validated_knn_decoder(
    activity, meta['positions'], k=5, n_folds=5
)

print(f"R²: {metrics['mean_r2']:.3f} ± {metrics['std_r2']:.3f}")
print(f"Error: {metrics['mean_error']:.3f} ± {metrics['std_error']:.3f}")
```

---

### `compare_highd_lowd_decoding`

Compare decoding performance on high-D activity vs low-D embedding.

**Key Use Case:** Evaluate if dimensionality reduction preserves decodable information.

**Parameters:**
- `activity`: High-D neural activity, shape `(n_samples, n_cells)`
- `embedding`: Low-D embedding, shape `(n_samples, n_components)`
- `labels`: True behavioral labels, shape `(n_samples, n_dims)`
- `k`: Number of neighbors (default: 5)
- `n_folds`: Number of CV folds (default: 5)

**Returns:**
- Dictionary with:
  - `high_d`: Metrics from high-D decoding
  - `low_d`: Metrics from low-D decoding
  - `performance_ratio`: low-D R² / high-D R²
  - `error_increase`: Relative error increase
  - `dimensionality_reduction`: String like "80 → 10"
  - `information_preserved`: Boolean (ratio > 0.8)

**Example:**
```python
from neural_analysis.learning.decoding import compare_highd_lowd_decoding
from sklearn.decomposition import PCA

activity, meta = generate_place_cells(100, 1500, n_dims=2)

# Create PCA embedding
pca = PCA(n_components=10)
embedding = pca.fit_transform(activity)

# Compare decoding
comparison = compare_highd_lowd_decoding(
    activity, embedding, meta['positions'], k=5
)

print(f"High-D R²: {comparison['high_d']['mean_r2']:.3f}")
print(f"Low-D R²: {comparison['low_d']['mean_r2']:.3f}")
print(f"Performance ratio: {comparison['performance_ratio']:.2%}")
print(f"Dimensionality: {comparison['dimensionality_reduction']}")
```

---

### `evaluate_decoder`

Unified interface for evaluating different decoder types on train/test split.

**Parameters:**
- `train_activity`: Training data, shape `(n_train, n_features)`
- `train_labels`: Training labels, shape `(n_train, n_dims)`
- `test_activity`: Test data, shape `(n_test, n_features)`
- `test_labels`: Test labels, shape `(n_test, n_dims)`
- `decoder`: `'knn'` or `'population_vector'`
- `**decoder_params`: Decoder-specific parameters

**Returns:**
- Dictionary with `r2_score`, `mse`, `mean_error`, `decoder`

**Example:**
```python
from neural_analysis.learning.decoding import evaluate_decoder

# Evaluate k-NN
metrics_knn = evaluate_decoder(
    train_act, train_pos, test_act, test_pos,
    decoder='knn', k=5
)

# Evaluate population vector
metrics_pv = evaluate_decoder(
    train_act, train_pos, test_act, test_pos,
    decoder='population_vector',
    field_centers=meta['field_centers'],
    method='weighted_average'
)

print(f"k-NN R²: {metrics_knn['r2_score']:.3f}")
print(f"Population vector R²: {metrics_pv['r2_score']:.3f}")
```

## Complete Examples

### Example 1: Basic Decoding Comparison

```python
import numpy as np
from neural_analysis.data.synthetic_data import generate_place_cells
from neural_analysis.learning.decoding import (
    knn_decoder,
    population_vector_decoder,
)

# Generate data
activity, meta = generate_place_cells(
    n_cells=50, n_samples=1000, n_dims=2,
    field_size=0.2, seed=42
)

# Split train/test
split = 700
train_act, test_act = activity[:split], activity[split:]
train_pos, test_pos = meta['positions'][:split], meta['positions'][split:]

# Decode with k-NN
decoded_knn = knn_decoder(train_act, train_pos, test_act, k=5)
error_knn = np.linalg.norm(decoded_knn - test_pos, axis=1).mean()

# Decode with population vector
decoded_pv = population_vector_decoder(
    test_act, meta['field_centers'], method='weighted_average'
)
error_pv = np.linalg.norm(decoded_pv - test_pos, axis=1).mean()

print(f"k-NN error: {error_knn:.4f} m")
print(f"Population vector error: {error_pv:.4f} m")
```

### Example 2: Cross-Validation

```python
from neural_analysis.data.synthetic_data import generate_place_cells
from neural_analysis.learning.decoding import cross_validated_knn_decoder

# Generate data
activity, meta = generate_place_cells(
    n_cells=80, n_samples=1500, n_dims=2, seed=123
)

# Cross-validate with different k values
for k in [3, 5, 10, 20]:
    metrics = cross_validated_knn_decoder(
        activity, meta['positions'], k=k, n_folds=5
    )
    print(f"k={k:2d}: R²={metrics['mean_r2']:.3f} ± {metrics['std_r2']:.3f}, "
          f"Error={metrics['mean_error']:.4f} ± {metrics['std_error']:.4f}")
```

### Example 3: High-D vs Low-D Comparison

```python
import numpy as np
from sklearn.decomposition import PCA
import umap

from neural_analysis.data.synthetic_data import generate_place_cells
from neural_analysis.learning.decoding import compare_highd_lowd_decoding

# Generate data
activity, meta = generate_place_cells(
    n_cells=100, n_samples=1500, n_dims=2,
    field_size=0.2, noise_level=0.08, seed=42
)

# Compare different embeddings
embeddings = {
    'PCA-10': PCA(n_components=10).fit_transform(activity),
    'PCA-5': PCA(n_components=5).fit_transform(activity),
    'UMAP-3': umap.UMAP(n_components=3).fit_transform(activity),
    'UMAP-2': umap.UMAP(n_components=2).fit_transform(activity),
}

print(f"High-D: {activity.shape[1]} dimensions")
print("-" * 60)

for name, embedding in embeddings.items():
    comparison = compare_highd_lowd_decoding(
        activity, embedding, meta['positions'], k=5, n_folds=5
    )
    
    print(f"{name} ({embedding.shape[1]} dims):")
    print(f"  R² ratio: {comparison['performance_ratio']:.2%}")
    print(f"  Error increase: {comparison['error_increase']:.2%}")
    print(f"  Information preserved: {comparison['information_preserved']}")
```

### Example 4: Mixed Cell Populations

```python
from neural_analysis.data.synthetic_data import generate_mixed_population_flexible
from neural_analysis.learning.decoding import cross_validated_knn_decoder

# Generate mixed population (place + grid + HD + random cells)
activity, meta = generate_mixed_population_flexible(
    n_samples=1200, seed=42
)

print(f"Total cells: {activity.shape[1]}")
print(f"Cell types: {meta['cell_types']}")

# Evaluate decoding
metrics = cross_validated_knn_decoder(
    activity, meta['positions'], k=5, n_folds=5
)

print(f"\nDecoding performance:")
print(f"R²: {metrics['mean_r2']:.3f} ± {metrics['std_r2']:.3f}")
print(f"Mean error: {metrics['mean_error']:.4f} ± {metrics['std_error']:.4f} m")
```

## Performance Tips

### Choosing k

- **Small k (3-5)**: More sensitive to local structure, higher variance
- **Large k (10-20)**: Smoother predictions, more robust to noise
- **Rule of thumb**: k ≈ sqrt(n_cells) for high-D, k ≈ 2-3× n_components for low-D

### Distance Weights

- **Distance weighting**: Usually better for spatial decoding (closer neighbors matter more)
- **Uniform weighting**: Can be more robust with very noisy data

### Metrics

- **Euclidean distance**: Default, works well for spatial variables
- **Manhattan distance**: Can be more robust to outliers
- **Cosine distance**: Good for directional variables (head direction)

### Cross-Validation

- Use **5-fold CV** for datasets with 500-2000 samples
- Use **10-fold CV** for larger datasets (>2000 samples)
- Use **leave-one-out CV** only for very small datasets (<100 samples)

## Expected Performance

### High-Dimensional Activity (Raw Neural Data)

| Cell Type | Typical R² | Typical Error |
|-----------|-----------|---------------|
| Place cells (50+) | 0.75-0.85 | 0.15-0.25 m |
| Grid cells (30+) | 0.70-0.80 | 0.20-0.30 m |
| Mixed population | 0.65-0.80 | 0.20-0.35 m |

### Low-Dimensional Embeddings

**Performance depends on:**
1. Embedding method (UMAP > Isomap > PCA for nonlinear manifolds)
2. Number of components (more = better, but diminishing returns)
3. Original data quality (noise level, cell count)

**Typical performance ratios (low-D R² / high-D R²):**
- PCA (10 components): 0.80-0.90
- UMAP (3 components): 0.85-0.95
- Isomap (5 components): 0.80-0.90

## Troubleshooting

### Low R² scores (<0.5)

- **Check data quality**: Ensure cells are tuned to the variable
- **Increase k**: Try larger k values
- **Use distance weighting**: Switch from `weights='uniform'` to `weights='distance'`
- **Check for outliers**: Remove or handle outliers in activity/labels

### High decoding error

- **Increase cell count**: More cells = better decoding
- **Reduce noise**: Lower `noise_level` in data generation
- **Check embedding**: Compare high-D vs low-D performance

### Performance ratio < 0.7

- **Increase embedding dimensions**: Use more components
- **Try different embedding**: UMAP often better than PCA for neural data
- **Check manifold structure**: Ensure embedding captures true structure

## References

1. **Population Vector Decoding**: Georgopoulos et al. (1986). "Neuronal population coding of movement direction." *Science*.
2. **k-NN for Neural Decoding**: Zhang et al. (1998). "Interpreting neuronal population activity by reconstruction." *Journal of Neurophysiology*.
3. **Embedding Quality**: Mimaroglu & Aksoy (2021). "DIPPER: A spatiotemporal data imputation framework." *Data Mining and Knowledge Discovery*.

## See Also

- `neural_analysis.data.synthetic_data`: Generate synthetic neural data
- Jupyter notebook: `notebooks/synthetic_datasets_demo.ipynb`
- Examples: `scripts/test_decoding_simple.py`
