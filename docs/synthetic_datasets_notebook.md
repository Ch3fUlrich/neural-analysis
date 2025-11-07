# Synthetic Datasets Demonstration Notebook

## Location
`notebooks/synthetic_datasets_demo.ipynb`

## Overview
Comprehensive Jupyter notebook demonstrating all synthetic dataset generation capabilities including neural data (place cells, grid cells, head direction cells) in 1D/2D/3D, manifold mappings, and sklearn benchmark datasets.

## Contents

### Part 1: Neural Data - Place Cells (1D/2D/3D)
- **1D Place Cells**: Linear track with localized firing fields
- **2D Place Cells**: Open field with Gaussian fields tiling the space
- **3D Place Cells**: Volumetric environment with 3D place fields
- Visualizations: Raster plots + behavioral trajectories using GridPlot layout

### Part 2: Grid Cells (1D/2D/3D)
- **1D Grid Cells**: Periodic firing along linear track
- **2D Grid Cells**: Hexagonal grid pattern (biologically accurate)
- **3D Grid Cells**: Cubic grid pattern in volumetric space
- Visualizations: Raster plots + trajectories + firing rate maps

### Part 3: Head Direction Cells
- Von Mises tuning curves for angular selectivity
- Polar plots showing tuning curves
- Preferred direction distributions
- Raster plots with head direction trajectories

### Part 4: Manifold Mappings
- **Ring (S¹) for Place Cells**: Map 1D positions to 2D circle
- **Ring (S¹) for HD Cells**: Map angles to circular manifold
- **Torus (T²) for Grid Cells**: Map 2D periodic space to 3D torus
- Visualizations showing perfect manifold structure

### Part 5: Mixed Neural Populations
- `generate_mixed_population_flexible()` with dictionary configuration
- Example configuration:
  ```python
  cell_config = {
      'place': {'n_cells': 60, 'field_size': 0.18, 'noise_level': 0.08},
      'grid': {'n_cells': 40, 'grid_spacing': 0.35, 'noise_level': 0.05},
      'head_direction': {'n_cells': 30, 'tuning_width': np.pi/5, 'noise_level': 0.1}
  }
  ```
- Color-coded raster plots showing different cell types
- Separate visualizations for each cell type

### Part 6: Embedding Quality Analysis
- **Perfect Place Cells**: Noise-free → clean embeddings
- **Noisy Place Cells**: Varying noise levels (0%, 10%, 30%, 50%)
  - Shows graceful degradation of embedding quality
- **Mixed Populations**: Heterogeneous cell types
  - Compares pure place cells vs. place-only from mixed vs. full mixed population
  - Demonstrates increased complexity in mixed embeddings

### Part 7: sklearn Manifold Datasets
- **Swiss Roll**: Classic 3D manifold on 2D surface
- **S-Curve**: S-shaped 3D manifold
- **Blobs**: Isotropic Gaussian clusters
- **Moons**: Two interleaving half circles
- **Circles**: Concentric circles
- **Dimensionality Reduction Comparison**:
  - PCA (linear)
  - Isomap (geodesic distances)
  - UMAP (topological structure)
  - Shows which methods successfully "unroll" manifolds

## Key Features

### Multi-Dimensional Support ✅
All neural data generators support:
- 1D environments (linear tracks): `arena_size=2.0`
- 2D environments (open fields): `arena_size=(1.0, 1.0)`
- 3D environments (volumetric): `arena_size=(1.0, 1.0, 0.5)`

### Manifold Theory Integration ✅
- Ring (S¹): Circular manifold for 1D periodic spaces (place cells, head direction)
- Torus (T²): 2D periodic manifold for grid cells
- Population vector decoding for trajectory tracking

### Flexible Configuration ✅
Dictionary-based configuration for mixed populations:
- Specify cell types, numbers, and parameters
- Custom noise levels per cell type
- Realistic test scenarios for embedding methods

### Visualization Best Practices ✅
- GridPlot layouts for organized multi-panel figures
- Raster plots with behavioral trajectories
- Time-colored trajectories
- Position-colored embeddings
- 3D plots for volumetric data and manifolds

## Key Functions Demonstrated

### Neural Data Generation
- `generate_place_cells()` - Gaussian place fields in 1D/2D/3D
- `generate_grid_cells()` - Periodic grids in 1D/2D/3D
- `generate_head_direction_cells()` - Angular tuning
- `generate_mixed_population_flexible()` - Dictionary-configured mixtures

### Manifold Mapping
- `map_to_ring()` - 1D → 2D circle (S¹)
- `map_to_torus()` - 2D → 3D torus (T²)
- `population_vector_decoder()` - Activity → decoded position

### sklearn Datasets
- `generate_swiss_roll()` - 3D Swiss roll manifold
- `generate_s_curve()` - 3D S-curve manifold
- `generate_data()` - Unified interface for all types

## Usage Examples

### Generate 2D Place Cells
```python
activity, meta = generate_place_cells(
    n_cells=50,
    n_samples=1000,
    arena_size=(1.0, 1.0),
    field_size=0.15,
    noise_level=0.05,
    seed=42
)
```

### Generate Mixed Population
```python
config = {
    'place': {'n_cells': 50, 'noise_level': 0.1},
    'grid': {'n_cells': 30, 'grid_spacing': 0.4, 'noise_level': 0.05},
    'head_direction': {'n_cells': 20, 'noise_level': 0.1}
}
activity, meta = generate_mixed_population_flexible(
    cell_config=config,
    n_samples=1500,
    arena_size=(2.0, 2.0),
    seed=42
)
```

### Map to Manifolds
```python
# Ring mapping for place cells
ring_coords = map_to_ring(activity, meta['positions'])

# Torus mapping for grid cells
torus_coords = map_to_torus(activity, meta['positions'], R=2.0, r=1.0)
```

## Scientific Applications

### Testing Dimensionality Reduction
- Perfect datasets: Verify methods recover true structure
- Noisy datasets: Test robustness to noise
- Mixed datasets: Evaluate performance on heterogeneous data

### Decoding Validation
- Population vector decoding with known ground truth
- Compare weighted average vs. peak methods
- Trajectory reconstruction on manifolds

### Cell Type Classification
- Train classifiers on mixed populations
- Test with known cell type labels
- Evaluate on realistic noise levels

### Manifold Learning
- Ring embeddings for place/HD cells (S¹)
- Torus embeddings for grid cells (T²)
- Compare learned vs. true manifolds

## Visualization Guidelines

### GridPlot Layout
```python
fig = plt.figure(figsize=(16, 8))
gs = GridSpec(rows, cols, height_ratios=[...], hspace=0.3, wspace=0.3)
```

### Raster Plot with Behavior
- Top panel: Raster plot (time × cells)
- Bottom panel(s): Behavioral variables (position, angle)
- Color-code cell types in mixed populations

### 3D Visualizations
- Use `projection='3d'` for trajectories and manifolds
- Color by time or position
- Adjust viewing angle for clarity

## Reproducibility
All datasets are reproducible with random seeds:
```python
SEED = 42
activity, meta = generate_place_cells(..., seed=SEED)
```

## Dependencies
- numpy
- matplotlib
- sklearn (for datasets and dimensionality reduction)
- umap-learn (for UMAP)
- neural_analysis.synthetic_data (custom module)

## Future Extensions
- [ ] Add more cell types (border cells, speed cells)
- [ ] Implement more manifold types (sphere, Klein bottle)
- [ ] Add time-varying dynamics (remapping, drift)
- [ ] Include spike train generation (Poisson processes)
- [ ] Add more embedding methods (t-SNE, PaCMAP)

## Notes
- All neural data includes comprehensive metadata dictionaries
- Manifold mappings preserve topological structure
- Mixed populations allow realistic testing scenarios
- Embeddings show progressive degradation: perfect → noisy → mixed
