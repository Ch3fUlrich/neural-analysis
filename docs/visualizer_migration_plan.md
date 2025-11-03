# Visualizer.py Migration Plan

## Overview
The `Visualizer.py` file (~7685 lines) contains a large collection of visualization functions with significant code duplication and mixed use of matplotlib and plotly. This document outlines the migration strategy to create a modular, maintainable, and well-tested visualization library.

## Current Issues Identified
1. **Code Duplication**: Many similar plotting functions with slight variations
2. **Mixed Backends**: Both matplotlib and plotly used without clear separation
3. **Inconsistent API**: Some methods are instance methods, others are static, some standalone functions
4. **Missing Type Hints**: Many functions lack proper type annotations
5. **Incomplete Documentation**: Missing or inadequate docstrings and examples
6. **Poor Modularity**: Single monolithic file with 99+ functions
7. **Unclear Dependencies**: Uses `from Helper import *` making dependencies unclear

## Proposed Module Structure

```
src/neural_analysis/visualization/
├── __init__.py                 # Main API exports
├── backend.py                  # Backend selection (matplotlib/plotly)
├── core.py                     # Core utilities, colors, axis config
├── plots_1d.py                 # 1D plotting functions
├── plots_2d.py                 # 2D plotting functions
├── plots_3d.py                 # 3D plotting functions
├── embeddings.py               # Embedding visualizations (2D/3D)
├── heatmaps.py                 # Heatmap visualizations
├── statistical.py              # Histograms, KDE, distributions
├── neural.py                   # Neural-specific plots (rasters, traces)
└── animations.py               # Animation functions
```

## Function Categories and Migration Plan

### 1. Core Utilities (core.py)
**Functions to migrate:**
- `calculate_alpha()` - Alpha value calculation based on value ranges
- `generate_similar_colors()` - Color generation utilities
- `create_rgba_labels()` - RGBA label creation
- `create_RGBA_colors_from_2d()` - 2D to RGBA conversion
- `auto_fontsize()` - Automatic font size calculation
- `_get_cmap()` - Colormap retrieval
- `_get_base_color()` - Base color extraction
- `_get_alpha_colors()` - Alpha color generation
- `save_plot()` - Plot saving with format options
- `create_save_path()` - Path creation for saving

**Refactoring needed:**
- Remove duplicates in color generation
- Create consistent API for color utilities
- Add proper type hints for all functions
- Add comprehensive docstrings with examples

### 2. Backend Selection (backend.py)
**New module to create:**
- `BackendEnum` - Enum for matplotlib/plotly selection
- `get_backend()` - Get current backend
- `set_backend()` - Set backend preference
- `BackendAdapter` - Abstract base class for backend-specific implementations
- `MatplotlibAdapter` - Matplotlib-specific plotting
- `PlotlyAdapter` - Plotly-specific plotting

**Design pattern:**
Use Strategy pattern to allow runtime backend selection while maintaining consistent API.

### 3. Plot Configuration (core.py)
**Functions to consolidate:**
- `default_plot_attributes()` - Default plot parameters
- `define_plot_parameter()` - Parameter definition
- `default_plot_start()` - Plot initialization
- `default_plot_ending()` - Plot finalization
- `plot_ending()` - Alternative ending function

**Refactoring strategy:**
- Create a `PlotConfig` dataclass for plot configuration
- Consolidate start/end functions into context manager
- Example: `with plot_context(config) as ctx: ...`

### 4. 1D Plots (plots_1d.py)
**Functions to migrate:**
- `data_plot_1D()` - 1D data plotting with multiple series
- `plot_line()` - Line plot with std bands
- `plot_losses()` - Multiple losses comparison

**Key improvements:**
- Add backend parameter to all functions
- Consolidate line plotting logic
- Support both continuous and categorical x-axes
- Add markers, error bands, annotations

### 5. 2D Plots (plots_2d.py)
**Functions to migrate:**
- `data_plot_2D()` - 2D scatter/trajectory plots
- `plot_2d_group_scatter()` - Grouped scatter plots
- `plot_2d_kde_dict_of_dicts()` - KDE plots
- `plot_2d_line_dict_of_dicts()` - Line plots with grouping
- `plot_2d_colormap()` - 2D colormap visualization

**Refactoring needed:**
- Remove duplicate scatter plot implementations
- Unify color mapping logic
- Support both matplotlib and plotly
- Add convex hull option (using `add_hull`)

### 6. 3D Plots (plots_3d.py)
**Functions to migrate:**
- `plot_3D_group_scatter()` - 3D scatter plots with grouping
- `add_hull()` - Convex hull for 2D/3D (move to embeddings.py)

**Key features:**
- Interactive 3D with plotly
- Static 3D with matplotlib
- Hull visualization
- Proper axis configuration

### 7. Embedding Visualizations (embeddings.py)
**Functions to migrate:**
- `plot_embedding()` - Main embedding function (2D/3D router)
- `plot_embedding_2d()` - 2D embeddings
- `plot_embedding_3d()` - 3D embeddings
- `plot_multiple_embeddings()` - Multiple embedding comparison
- `plot_simple_embedd()` - Simplified embedding plot
- `add_1d_colormap_legend()` - 1D colormap legend
- `add_2d_colormap_legend()` - 2D colormap legend
- `add_hull()` - Convex hull visualization

**Major improvements:**
- Separate dimension reduction (PCA/MDS) into preprocessing
- Clean up legend creation
- Support multiple labeling schemes
- Add interactive mode with plotly

### 8. Heatmaps (heatmaps.py)
**Functions to migrate:**
- `plot_heatmap()` - Main heatmap function
- `plot_cell_activites_heatmap()` - Cell activity heatmaps
- `plot_heatmap_dict_of_dicts()` - Nested dictionary heatmaps
- `plot_mean_std_heatmap()` - Mean/std heatmaps
- `plot_batch_heatmap()` - Batch heatmap plotting

**Consolidation:**
- Create single `plot_heatmap()` function with options
- Support different data structures (arrays, dicts, DataFrames)
- Add annotation options
- Support both backends

### 9. Statistical Plots (statistical.py)
**Functions to migrate:**
- `plot_histogram()` - Histogram plotting
- `create_histogram()` - Histogram creation
- `histogam_subplot()` - Histogram subplot (fix typo)
- `plot_kde()` - KDE plotting
- `kdeplot_from_dict()` - KDE from dictionary
- `plot_1d_kde_dict_of_dicts()` - 1D KDE grouped
- `density()` - Density plots
- `violin_plot()` - Violin plots
- `linebar_df_group_plot()` - Line and bar combinations

**Improvements:**
- Unify histogram and KDE interfaces
- Add statistical test annotations
- Support multiple distributions
- Add box plots and violin plots

### 10. Neural-Specific Plots (neural.py)
**Functions to migrate:**
- `plot_neural_activity_raster()` - Raster plots
- `plot_traces_shifted()` - Shifted trace plots
- `traces_subplot()` - Trace subplots
- `plot_single_cell_activity()` - Single cell visualization
- `plot_multi_task_cell_activity_pos_by_time()` - Multi-task cell activity
- `plot_all_cells_modular()` - Modular cell plotting

**Key features:**
- Spike raster plots with proper binning
- Calcium trace visualization
- Multi-neuron comparison
- Time-aligned activity

### 11. Decoding & Analysis Plots (statistical.py)
**Functions to migrate:**
- `plot_decoding_score()` - Decoding performance
- `plot_decoding_statistics()` - Decoding statistics
- `plot_decoding_statistics_line()` - Line-based decoding stats
- `plot_discrete_decoding_statistics_bar()` - Bar plots for discrete
- `plot_continuous_decoding_statistics_bar()` - Bar plots for continuous
- `plot_consistency_scores()` - Consistency scoring
- `plot_zscore()` - Z-score visualization
- `plot_si_rates()` - Spatial information rates

**Consolidation:**
- Create unified decoding visualization API
- Support multiple metrics
- Add statistical significance indicators

### 12. Specialized Plots
**Functions to migrate:**
- `barplot_from_dict()` - Bar plots from dictionaries
- `barplot_from_dict_of_dicts()` - Nested bar plots
- `lineplot_from_dict_of_dicts()` - Nested line plots
- `plot_structure_index()` - Structure index visualization
- `plot_corr_hist_heat_salience()` - Combined correlation plots
- `plot_group_distr_similarities()` - Distribution similarity plots
- `pca_component_variance_plot()` - PCA variance explained

### 13. Animations (animations.py)
**Functions to migrate:**
- `animate_2D_positions()` - 2D position animation

**Improvements:**
- Add 3D animation support
- Support export to video formats
- Interactive playback controls with plotly

## Helper Functions to Identify/Replace
Since the file uses `from Helper import *`, we need to identify which helper functions are used:
- `make_list_ifnot()` - List conversion utility
- `force_equal_dimensions()` - Dimension matching
- `range_to_times_xlables_xpos()` - Time label generation
- `values_to_groups()` - Value grouping
- `normalize_01()` - Normalization to [0,1]
- `is_rgba()` - RGBA format check
- `global_logger` - Logging utility
- `do_critical()` - Critical error handling
- `pca_numba()` / `mds_numba()` - Dimensionality reduction

**Action:** Create explicit imports or implement these in core.py if they're simple utilities.

## Implementation Strategy

### Phase 1: Core Infrastructure (Week 1)
1. Create module structure
2. Implement `backend.py` with adapter pattern
3. Implement `core.py` with color utilities, PlotConfig, and common helpers
4. Write tests for core functionality

### Phase 2: Basic Plots (Week 1-2)
1. Implement `plots_1d.py` with comprehensive tests
2. Implement `plots_2d.py` with comprehensive tests
3. Implement `plots_3d.py` with comprehensive tests
4. Test backend switching

### Phase 3: Specialized Visualizations (Week 2-3)
1. Implement `embeddings.py` with tests
2. Implement `heatmaps.py` with tests
3. Implement `statistical.py` with tests
4. Implement `neural.py` with tests

### Phase 4: Advanced Features (Week 3-4)
1. Implement `animations.py`
2. Add interactive plotly features
3. Performance optimization
4. Comprehensive integration tests

### Phase 5: Documentation & Examples (Week 4)
1. Write comprehensive docstrings for all functions
2. Create usage examples notebook
3. Create migration guide for users
4. Write API reference documentation

## Testing Strategy

### Unit Tests
- Test each function with various input types
- Test edge cases (empty data, single point, etc.)
- Test type validation
- Test backend switching

### Integration Tests
- Test complex plotting workflows
- Test multiple plots in subplots
- Test saving/loading functionality
- Test both backends produce similar results

### Comparison Tests
- Compare new implementation with original where possible
- Visual regression testing for critical plots
- Performance benchmarks

## API Design Principles

1. **Consistent Interface**: All plot functions follow similar patterns
2. **Backend Agnostic**: User can switch between matplotlib/plotly easily
3. **Sensible Defaults**: Works out-of-box with minimal parameters
4. **Flexible**: Advanced users can customize everything
5. **Type Safe**: Full type hints for IDE support
6. **Well Documented**: Comprehensive docstrings with examples
7. **Testable**: Pure functions where possible, injectable dependencies

## Example API Usage

```python
from neural_analysis.visualization import (
    set_backend, 
    plot_1d, 
    plot_embedding_2d,
    PlotConfig
)

# Set backend preference
set_backend('matplotlib')  # or 'plotly'

# Simple 1D plot
plot_1d(data, backend='matplotlib', title="My Plot")

# Configured plot
config = PlotConfig(
    title="Neural Activity",
    xlabel="Time (s)",
    ylabel="Activity",
    figsize=(10, 5)
)
plot_1d(data, config=config)

# Embedding visualization
plot_embedding_2d(
    embedding=coords,
    labels=labels,
    show_hulls=True,
    backend='plotly',  # Interactive!
    save_path='embedding.html'
)
```

## Migration Checklist

- [ ] Set up module structure
- [ ] Implement backend adapter system
- [ ] Migrate core utilities
- [ ] Migrate 1D plotting functions
- [ ] Migrate 2D plotting functions
- [ ] Migrate 3D plotting functions
- [ ] Migrate embedding visualizations
- [ ] Migrate heatmaps
- [ ] Migrate statistical plots
- [ ] Migrate neural-specific plots
- [ ] Migrate animations
- [ ] Write unit tests for all modules
- [ ] Write integration tests
- [ ] Create usage examples
- [ ] Write documentation
- [ ] Performance optimization
- [ ] Final code review

## Notes on Specific Refactorings

### Color System
Currently has multiple color generation methods that overlap:
- `generate_similar_colors()` - HSL-based variation
- `create_rgba_labels()` - Colormap-based
- `_get_base_color()` - Base color extraction
- `_get_alpha_colors()` - Alpha variation

**Proposed**: Single `ColorManager` class with methods for each use case.

### Plot Configuration
Multiple overlapping configuration functions:
- `default_plot_attributes()`
- `define_plot_parameter()`
- `default_plot_start()`
- `default_plot_ending()`

**Proposed**: Single `PlotConfig` dataclass with context manager support.

### Duplicate Implementations
Many functions like `plot_2d_*` have similar implementations. Consolidate into single function with parameters:
- `plot_scatter_2d(data, groupby=None, kde=False, ...)`

## Dependencies to Add
- `matplotlib` >= 3.5
- `plotly` >= 5.0
- `numpy` >= 1.20
- `scipy` >= 1.7
- `pandas` >= 1.3
- `scikit-learn` >= 1.0 (for PCA/MDS if not in Helper)
- `seaborn` >= 0.11 (optional, for enhanced styling)

## Success Criteria
1. All functions from original file migrated
2. 90%+ test coverage
3. Both matplotlib and plotly backends work
4. API documentation complete
5. Performance equal or better than original
6. No code duplication
7. Type hints on all public functions
8. Examples for all major functions
