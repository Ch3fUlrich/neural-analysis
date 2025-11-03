# Plotting Module Migration Progress

## Overview
This document tracks the migration of the legacy `Visualizer.py` class (7,685 lines) into a modern, modular plotting system.

## Migration Goals
1. ✅ Reduce code duplication across 1D, 2D, and 3D plotting functions
2. ✅ Create consistent cross-backend (matplotlib/plotly) support
3. ✅ Improve modularity and testability
4. ✅ Modernize matplotlib API usage
5. ✅ Fix all pytest warnings
6. 🔄 Migrate all functions from legacy Visualizer.py
7. 🔄 Add comprehensive tests and examples

## Completed Work

### 1. Core Infrastructure (`plots/core.py`)
- ✅ Created cross-backend helper functions:
  - `resolve_colormap()` - Modern colormap resolution (plt.colormaps[])
  - `apply_layout_matplotlib()` / `apply_layout_plotly()` - Consistent layout application
  - `finalize_plot_matplotlib()` / `finalize_plot_plotly()` - Save/show/return logic
  - `get_default_categorical_colors()` - Modern categorical color palette
- ✅ Fixed all matplotlib deprecation warnings
- ✅ Added proper warning stacklevels for better debugging

### 2. 1D Plotting Module (`plots/plots_1d.py`)
- ✅ Migrated functions:
  - `plot_line()` - Single or multiple line plots with optional std/confidence bands
  - `plot_multiple_lines()` - Multiple lines with automatic coloring
  - `plot_boolean_states()` - Binary state visualization over time
- ✅ All functions use cross-backend helpers
- ✅ 28 passing tests

### 3. 2D Plotting Module (`plots/plots_2d.py`)
- ✅ Migrated functions:
  - `plot_scatter()` - 2D scatter plots with color mapping
  - `plot_trajectory()` - 2D paths with optional time-based coloring
  - `plot_grouped_scatter()` - Multiple groups with optional convex hulls
  - `plot_kde()` - 2D kernel density estimation
- ✅ Fixed "No data for colormapping" warnings (conditional cmap passing)
- ✅ 38 passing tests

### 4. 3D Plotting Module (`plots/plots_3d.py`) **NEW!**
- ✅ Migrated functions:
  - `plot_scatter_3d()` - 3D scatter plots with color mapping
  - `plot_trajectory_3d()` - 3D paths with Line3DCollection for time coloring
- ✅ Both matplotlib (Axes3D) and plotly (Scatter3d) backends
- ✅ 10 passing tests

## Code Quality Improvements

### Lines of Code Reduced
- **~150+ lines** of duplicated code eliminated through helper functions
- Cross-backend logic now centralized in `core.py`

### Test Coverage
- **Total Tests**: 81 (all passing)
  - 1D plots: 28 tests
  - 2D plots: 38 tests
  - 3D plots: 10 tests
  - Other: 5 tests
- **Test Execution Time**: ~6 seconds (clean, no warnings)

### API Modernization
- ❌ Old: `cm.get_cmap('viridis')` (deprecated)
- ✅ New: `plt.colormaps['viridis']` or `plt.colormaps.get_cmap('viridis')`
- ✅ Automatic colormap name capitalization (e.g., "blues" → "Blues")
- ✅ Conditional cmap parameter passing (only when colors is an array)

## Pending Migration from Visualizer.py

### High Priority (Foundation for other features)
- 🔄 Embedding visualizations:
  - `plot_embedding()` - Generic embedding wrapper
  - `plot_embedding_2d()` - 2D embeddings (UMAP, t-SNE, PCA)
  - `plot_embedding_3d()` - 3D embeddings with convex hulls
  - `plot_simple_embedd()` - Simplified embedding plots

### Medium Priority (Specialized plots)
- 🔄 Heatmaps:
  - `plot_batch_heatmap()` - Multiple heatmaps in grid
  - `plot_2d_colormap()` - 2D colormapped visualizations
- 🔄 KDE plots:
  - `plot_kde()` - General KDE visualization
  - `plot_1d_kde_dict_of_dicts()` - Hierarchical 1D KDE
  - `plot_2d_kde_dict_of_dicts()` - Hierarchical 2D KDE
- 🔄 Neural-specific plots:
  - Neural activity rasters
  - Spike timing plots
  - Neural decoding statistics

### Lower Priority (Simple or specialized)
- 🔄 `plot_line()` - May overlap with existing plot_line
- 🔄 `plot_all_cells_modular()` - Modular cell visualization

## File Structure

```
src/neural_analysis/plotting/
├── __init__.py              # Public API exports
├── core.py                  # Backend-agnostic utilities (362 lines)
├── plots_1d.py             # Line plots (264 lines)
├── plots_2d.py             # 2D scatter/trajectory/KDE (504 lines)
└── plots_3d.py             # 3D scatter/trajectory (453 lines) ⭐ NEW!

tests/
├── test_plots_1d.py        # 28 tests
├── test_plots_2d.py        # 38 tests
└── test_plots_3d.py        # 10 tests ⭐ NEW!

todo/
└── Visualizer.py           # Legacy code (7,685 lines, being migrated)
```

## Next Steps

### Immediate Actions
1. ✅ Create `test_plots_3d.py` with comprehensive tests
2. 📝 Create `examples/plots_3d_examples.ipynb` demonstrating 3D capabilities
3. 🔄 Begin embedding module migration (`plots_embeddings.py`)
4. 🔄 Mark or remove migrated 3D functions from `Visualizer.py`

### Workflow for Continued Migration
1. **Identify** function(s) from Visualizer.py to migrate
2. **Create** new module or add to existing module
3. **Implement** using modern API and cross-backend helpers
4. **Test** with comprehensive pytest suite
5. **Document** with examples/notebook
6. **Mark** original function as migrated in Visualizer.py
7. **Update** this progress document

## Success Metrics

### Before Migration
- ❌ 7,685 lines in single monolithic file
- ❌ Heavy code duplication across plot types
- ❌ Inconsistent backend support
- ❌ Deprecated matplotlib API usage
- ❌ Pytest warnings on every run
- ❌ Limited test coverage

### After Migration (Current State)
- ✅ **~1,583 lines** across 4 modular files
- ✅ Minimal code duplication (shared helpers)
- ✅ Consistent matplotlib + plotly backends
- ✅ Modern matplotlib API throughout
- ✅ **Zero pytest warnings**
- ✅ **81 tests** with 100% pass rate
- ✅ Clear separation of concerns
- ✅ Foundation for 3D visualization features

### Target State
- 🎯 Complete migration of all useful Visualizer functions
- 🎯 90%+ test coverage
- 🎯 Comprehensive example notebooks
- 🎯 Deprecation or removal of Visualizer.py
- 🎯 Full documentation of new plotting API

---

**Last Updated**: 2024 (after plots_3d.py creation)
**Status**: 🟢 On track - Core infrastructure complete, 3D foundation established
