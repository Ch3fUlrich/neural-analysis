# Neural Analysis Project - Consolidated TODO & Migration Plan

**Last Updated**: November 2025
**Current Test Status**: 181/181 passing (100% coverage) ✅
**Repository**: neural-analysis (migration branch)

**Total Legacy Code to Migrate**: ~16,134 lines across 6 files in `/todo` folder

---

## 📊 RECENT ACCOMPLISHMENTS

### December 2024 - Code Reorganization ✅
1. **Shape Distance Functions Migrated**:
   - ✅ Moved all shape.py functions to distributions.py
   - ✅ Created modular functions: `shape_distance_procrustes()`, `shape_distance_one_to_one()`, `shape_distance_soft_matching()`
   - ✅ Created general batch framework: `pairwise_distribution_comparison_batch()`
   - ✅ Updated __init__.py imports
   - ✅ Deleted shape.py
   - ✅ Updated test_new_modules.ipynb

**Rationale**: Shape distances ARE distribution comparisons (comparing neural population activity distributions in feature space). Consolidation improves code organization and enables unified batch processing framework.

### November 2024 - Test Suite & Documentation ✅
1. ✅ PlotGrid architecture verification (all 15 functions consistent)
2. ✅ README.md documentation (features, usage examples, testing)
3. ✅ Duplicate test file removed (181 unique tests, down from 196)
4. ✅ Type annotation fixes (renderers.py plotly types)
5. ✅ All tests passing (100% coverage maintained)
6. ✅ 100% Test Coverage achieved (181 unique tests)
7. ✅ PlotGrid Architecture verified and documented

---

## 🚀 HIGH PRIORITY TASKS

### 1. TODO Folder Migration - CRITICAL 🔴

**Total Legacy Code**: ~16,134 lines requiring evaluation and migration

| File | Lines | Status | Priority | Action |
|------|-------|--------|----------|--------|
| `Manimeasure.py` | 1,882 | ❌ Not Migrated | 🔴 CRITICAL | Core analysis pipeline (embeddings, decoding, similarity) |
| `Visualizer.py` | 7,586 | 🟡 Partial | 🟡 MEDIUM | Legacy plotting (many already migrated to PlotGrid) |
| `Helper.py` | 4,731 | ❌ Not Migrated | 🟢 LOW-MEDIUM | Utility functions (needs detailed review) |
| `structure_index.py` | 864 | ❌ Not Migrated | 🟡 MEDIUM | Structure index topology analysis |
| `yaml_creator.py` | 686 | ❌ Not Migrated | ⚪ LOW | YAML metadata creation (pipeline-specific) |
| `restructure.py` | 391 | ❌ Not Migrated | ⚪ LOW | Data folder restructuring (pipeline-specific) |

---

### 1.1 Manimeasure.py - TOP PRIORITY 🔴

**File**: `todo/Manimeasure.py` (1,882 lines)
#### A. Embedding Methods (Lines 47-320) - CRITICAL

**Functions to Migrate**:
- `create_multiple_embeddings()` - Multi-panel embedding comparison plots
- `_add_embedding_subplot()` - Helper for subplot creation  
- `simple_embedd()` - Core embedding wrapper (TSNE, UMAP, PCA, MDS, Isomap, LLE, PTU, Spectral)

**Target Location**:
```
src/neural_analysis/embeddings/
├── __init__.py
├── dimensionality_reduction.py  # Core embedding functions
└── visualization.py              # Embedding plotting with PlotGrid
```

**Implementation Plan**:
1. Create `compute_embedding()` function with unified API
2. Implement `plot_multiple_embeddings()` using PlotGrid system
3. Support methods: PCA, UMAP, t-SNE, MDS, Isomap, LLE, Spectral
4. Add comprehensive testing with synthetic data
5. Create example notebook: `examples/embeddings_demo.ipynb`

**Dependencies**:
- scikit-learn (PCA, MDS, Isomap, LLE, SpectralEmbedding)
- umap-learn (UMAP) - ✅ Available with Python 3.12
- Optional: parallel_transport_unfolding (PTU)

**Estimated Effort**: 8-12 hours

---

#### B. Decoding Analysis (Lines 322-664) - CRITICAL

**Functions to Migrate**:
- `decoding_analysis()` - Main decoder training and evaluation
- `get_train_test_data()` - Data splitting for cross-validation
- `compute_decoding_accuracy()` - Model performance metrics

**Target Location**:
```
src/neural_analysis/decoding/
├── __init__.py
├── decoders.py          # Decoder classes and training
├── evaluation.py        # Performance metrics
└── cross_validation.py  # CV splitting strategies
```

**Implementation Plan**:
1. Create decoder wrapper classes for common sklearn models
2. Implement cross-validation framework
3. Add performance metrics (accuracy, confusion matrix, decoding curves)
4. Support multiple decoder types (SVM, LogisticRegression, RandomForest, etc.)
5. Add visualization functions using PlotGrid
6. Create example notebook: `examples/decoding_demo.ipynb`

**Dependencies**:
- scikit-learn (models, metrics, CV)
- scipy (statistics)

**Estimated Effort**: 10-15 hours

---

#### C. Similarity Analysis (Lines 666-1,882) - HIGH

**Functions to Migrate**:
- Population similarity metrics
- Representational similarity analysis (RSA)
- Cross-session comparisons

**Note**: Some similarity functions may already exist in `src/neural_analysis/metrics/similarity.py`. Check for overlaps before migration.

**Target Location**: `src/neural_analysis/metrics/similarity.py` (extend existing module)

**Estimated Effort**: 6-8 hours

---

### 1.2 Visualizer.py - Evaluate and Complete 🟡

**File**: `todo/Visualizer.py` (7,586 lines)
**Status**: 🟡 PARTIALLY MIGRATED

**Action Items**:
1. **Audit Existing Functions**: Identify which plotting functions have been migrated to PlotGrid
2. **Identify Unique Functions**: Find plotting capabilities NOT yet in new system
3. **Migration Decision Matrix**:
   - ✅ Already migrated → Delete from Visualizer.py
   - 🔄 Unique & useful → Migrate to PlotGrid system
   - ❌ Obsolete/redundant → Document and deprecate

**Key Areas to Check**:
- Tuning curves (likely migrated)
- Raster plots (likely migrated)
- Population heatmaps (likely migrated)
- Custom statistical plots (check coverage)
- 3D visualizations (check coverage)

**Estimated Effort**: 4-6 hours audit + variable migration time

---

### 1.3 Helper.py - Review and Organize 🟢

**File**: `todo/Helper.py` (4,731 lines)
**Status**: ❌ NOT MIGRATED

**Action Items**:
1. **Categorize Functions**:
   - Data preprocessing → `src/neural_analysis/utils/preprocessing.py`
   - I/O operations → `src/neural_analysis/utils/io.py`
   - Validation → `src/neural_analysis/utils/validation.py`
   - Mathematical utilities → `src/neural_analysis/utils/math_utils.py`
   - Redundant (duplicates numpy/scipy) → Document and deprecate

2. **Migration Strategy**:
   - Create inventory of all functions
   - Check for numpy/scipy equivalents
   - Migrate unique, useful functions
   - Add type hints and proper documentation
   - Write unit tests for each migrated function

**Estimated Effort**: 8-12 hours

---

### 1.4 Structure Index (structure_index.py) - Topology Module 🟡

**File**: `todo/structure_index.py` (864 lines)
**Status**: ❌ NOT MIGRATED

**Note**: Some structure index functionality may already exist in `src/neural_analysis/topology/`.

**Action Items**:
1. Compare with existing `topology/` module
2. Migrate unique functionality
3. Create example notebook
4. Add comprehensive tests

**Estimated Effort**: 4-6 hours

---

### 1.5 Pipeline Utilities - LOW PRIORITY ⚪

**Files**: `yaml_creator.py` (686 lines), `restructure.py` (391 lines)
**Status**: ❌ NOT MIGRATED
**Priority**: ⚪ LOW - Pipeline-specific utilities

**Purpose**:
- `yaml_creator.py`: Creates YAML metadata files from Excel spreadsheets for SERBRA data pipeline
- `restructure.py`: Reorganizes data directories to match SERBRA pipeline structure

**Assessment**:
- Very specific to SERBRA data loading/organization pipeline
- Not general-purpose analysis code
- One-time or occasional use

**Options**:
1. **Move to scripts/data_preprocessing/** if actively used
2. **Keep in todo/** as pipeline utility (import as-needed)
3. **Document and deprecate** if no longer actively used

**Recommended Action**: Move to `scripts/data_preprocessing/` if actively used, otherwise document and leave in todo/

**Estimated Effort**: 1-2 hours each (if moved)

---

### 1.6 TODO Folder Notebooks

**Notebooks to Review**:
- `Make_yaml_steffen.ipynb` - YAML creation workflow
- `Restructure_make_yaml_nathlie_openfield.ipynb` - Data restructuring workflow

**Action**:
- Review for useful patterns
- Convert to example notebook if generally useful
- Otherwise, keep as reference in `todo/Notebooks/`

**Estimated Effort**: 1 hour review each

---

## 2. MIGRATION ROADMAP & TIMELINE

### 2.1 Phase 1: Critical Analysis Functions (Priority 🔴)

**Goal**: Migrate core analysis capabilities
**Estimated Time**: 3-4 weeks (part-time) or 1-1.5 weeks (full-time)

| Task | Subtasks | Effort | Dependencies |
|------|----------|--------|--------------|
| **Embeddings Module** | Compute embedding, multi-embedding plots, PCA variance | 8-12 hrs | umap-learn (Python 3.12+) |
| **Decoding Module** | k-NN decoder, metrics, visualization | 6-10 hrs | scikit-learn |
| **Feature Similarity** | Population similarity, visualization | 4-6 hrs | scipy, sklearn |
| **Structure Index** | Migrate structure_index.py, wrapper, viz | 8-10 hrs | networkx, sklearn |
| **Testing** | Unit tests for all above | 10-15 hrs | pytest |
| **Documentation** | Example notebooks for all above | 8-12 hrs | jupyter |
| **Total** | - | **50-73 hrs** | - |

**Deliverables**:
- `src/neural_analysis/embeddings/` (with tests & examples)
- `src/neural_analysis/decoding/` (with tests & examples)
- `src/neural_analysis/topology/` (enhanced with structure index)
- Extended `src/neural_analysis/metrics/similarity.py`
- 4-5 new example notebooks

---

### 2.2 Phase 2: Visualizer.py Remaining Functions (Priority 🟡)

**Goal**: Complete visualization system migration
**Estimated Time**: 1-2 weeks (part-time) or 3-5 days (full-time)

| Task | Subtasks | Effort | Dependencies |
|------|----------|--------|--------------|
| **Batch Heatmaps** | plot_heatmap_grid implementation | 3-4 hrs | PlotGrid |
| **Utilities** | Extract useful utilities to core.py | 3-5 hrs | - |
| **Colormap Functions** | Integrate to core.py | 2-4 hrs | - |
| **Specialized Plots** | Evaluate & migrate if needed | 2-4 hrs | - |
| **Testing** | Unit tests for above | 3-5 hrs | pytest |
| **Documentation** | Update existing examples | 2-3 hrs | jupyter |
| **Total** | - | **15-25 hrs** | - |

**Key Functions to Evaluate**:
- `plot_simple_embedd()` - Move to embeddings/visualization.py
- `pca_component_variance_plot()` - Move to embeddings/visualization.py
- `plot_batch_heatmap()` - Create `plot_heatmap_grid()` in heatmaps.py
- `plot_all_cells_modular()` - Cell-wise activity heatmaps
- Animation functions - LOW PRIORITY (defer)

**Deliverables**:
- Enhanced `src/neural_analysis/plotting/heatmaps.py`
- Enhanced `src/neural_analysis/plotting/core.py`
- Updated example notebooks

---

### 2.3 Phase 3: Helper.py Review & Migration (Priority 🟢)

**Goal**: Migrate useful utilities, remove redundant code
**Estimated Time**: 1-2 weeks (part-time) or 3-5 days (full-time)

| Task | Subtasks | Effort | Dependencies |
|------|----------|--------|--------------|
| **Catalog** | List all functions in Helper.py | 1-2 hrs | - |
| **Cross-Reference** | Compare with existing utils/ and stdlib | 2-3 hrs | - |
| **Migrate** | Move unique utilities to utils/ | 5-10 hrs | Variable |
| **Testing** | Unit tests for migrated functions | 3-6 hrs | pytest |
| **Total** | - | **11-21 hrs** | - |

**Strategy**:
1. Generate complete function list from Helper.py
2. Cross-reference with:
   - `src/neural_analysis/utils/` modules
   - numpy/scipy/sklearn standard functions
3. Categorize:
   - **Keep & Migrate**: Unique, useful utilities → appropriate utils/ modules
   - **Document as Available**: Already in numpy/scipy/sklearn
   - **Obsolete**: No longer needed → document and deprecate

**Deliverables**:
- Enhanced `src/neural_analysis/utils/` modules
- List of deprecated functions (documented)
- Tests for all migrated utilities

---

### 2.4 Phase 4: Pipeline Utilities & Cleanup (Priority ⚪)

**Goal**: Organize pipeline-specific code, archive completed work
**Estimated Time**: 2-3 days (part-time) or 1 day (full-time)

| Task | Subtasks | Effort | Dependencies |
|------|----------|--------|--------------|
| **yaml_creator.py** | Move to scripts/ or document | 1-2 hrs | - |
| **restructure.py** | Move to scripts/ or document | 1-2 hrs | - |
| **Notebooks** | Review & archive/convert | 2 hrs | - |
| **Documentation** | Archive completed docs | 1 hr | - |
| **Cleanup** | Remove/deprecate todo/ folder | 1-2 hrs | All above complete |
| **Total** | - | **6-9 hrs** | - |

**Deliverables**:
- `scripts/data_preprocessing/` (if pipeline utils kept)
- `docs/archive/` with historical documentation
- Clean, empty (or removed) `todo/` folder

---

### 2.5 Total Migration Effort Summary

| Phase | Priority | Effort (hours) | Weeks (part-time) | Days (full-time) |
|-------|----------|----------------|-------------------|------------------|
| Phase 1: Critical Analysis | 🔴 | 50-73 | 3-4 | 7-10 |
| Phase 2: Visualization | 🟡 | 15-25 | 1-2 | 2-3 |
| Phase 3: Helper.py | 🟢 | 11-21 | 1-2 | 2-3 |
| Phase 4: Cleanup | ⚪ | 6-9 | <1 | 1 |
| **Total** | - | **82-128 hrs** | **5-9 weeks** | **12-17 days** |

**Assumptions**:
- Part-time: 15-20 hours/week
- Full-time: 6-8 hours/day
- Estimates include implementation, testing, documentation

---

## 3. CODE QUALITY & MAINTENANCE

### 2.1 Documentation ✅ COMPLETE

**Completed**:
- ✅ README.md with comprehensive usage examples
- ✅ Architecture documentation (PlotGrid system)
- ✅ Test status documentation
- ✅ Example notebooks verified

**Remaining**:
- 📝 API reference documentation (Sphinx)
- 📝 Migration guide from Visualizer.py to PlotGrid
- 📝 Comprehensive plotting cookbook

**Estimated Effort**: 6-8 hours

---

### 2.2 Example Notebooks - Needs Refresh

**Status**: Code verified correct, outputs stale

**Action Items**:
1. Re-execute all notebooks with current code:
   - `embeddings_demo.ipynb` (⚠️ needs creation after migration)
   - `io_h5io_examples.ipynb`
   - `logging_examples.ipynb`
   - `metrics_examples.ipynb`
   - `neural_analysis_demo.ipynb`
   - `plots_1d_examples.ipynb`
   - `plots_2d_examples.ipynb`
   - `plots_3d_examples.ipynb`
   - `plotting_grid_showcase.ipynb`
   - `statistical_plots_examples.ipynb`
   - `synthetic_datasets_example.ipynb`

2. Update outputs and save
3. Verify all plots render correctly in both backends

**Estimated Effort**: 2-3 hours

---

### 2.3 Test Notebook Updates ✅ COMPLETE

**Completed December 2024**:
- ✅ Updated `test_new_modules.ipynb` for shape distance migration
- ✅ Updated imports to use distributions.py instead of shape.py
- ✅ Updated test functions for new API
- ✅ Added tests for general batch comparison framework

---

## 4. TECHNICAL DEBT

### 4.1 Type Annotations - Mostly Complete ✅

**Status**: Main codebase has type hints

**Remaining Work**:
- Add type hints to any migrated functions from todo/
- Ensure 100% coverage with mypy
- Add py.typed marker for library usage

**Estimated Effort**: 2-3 hours

---

### 4.2 Plotly Forward References - RESOLVED ✅

**Status**: ✅ COMPLETE (fixed in renderers.py)

---

### 4.3 Python 3.14 Compatibility - BLOCKED 🚫

**Issue**: Numba not yet compatible with Python 3.14
**Status**: BLOCKED - waiting on upstream
**Reference**: `todo/numba_python314.md`
**Action**: Monitor numba releases

---

## 5. RISK ASSESSMENT & MITIGATION

### 5.1 Technical Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| **Python version incompatibility** | High (blocks embeddings) | Low | ✅ Resolved - Using Python 3.12 with full UMAP support |
| **Breaking API changes** | Medium | Low | Maintain backward compatibility layer |
| **Missing dependencies** | Medium | Low | Add optional dependencies, graceful degradation |
| **Performance regression** | Medium | Medium | Profile before/after, optimize if needed |
| **Test coverage gaps** | High | Medium | Require >90% coverage for all new code |

### 5.2 Process Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| **Scope creep** | High | Medium | Stick to documented plan, defer enhancements |
| **Incomplete migration** | High | Low | Follow checklist, mark progress in this doc |
| **Lost functionality** | High | Low | Comprehensive testing, example notebooks |
| **Timeline delays** | Medium | Medium | Focus on critical features first |

---

## 6. TESTING STRATEGY

### 6.1 Testing Requirements

All migrated code must include:

1. **Unit Tests** (pytest)
   - Test each function independently
   - Test edge cases and error conditions
   - Target: >90% code coverage

2. **Integration Tests**
   - Test pipelines (e.g., embedding → decoding)
   - Test with real-world data patterns
   - Test backend compatibility (matplotlib/plotly)

3. **Example Notebooks**
   - Demonstrate typical usage
   - Show advanced features
   - Executable and reproducible

### 6.2 Test Data

Use existing test fixtures from `tests/`:
- Synthetic datasets (`test_synthetic_data.py`)
- Small real-world examples
- Edge cases (from existing tests)

### 6.3 Continuous Integration

Ensure all tests pass before merging:

```bash
pytest tests/ -v --cov=src/neural_analysis --cov-report=html
```

Target: Maintain **100% test passage** (currently 181/181 tests passing)

---

## 7. DOCUMENTATION STRATEGY

### 7.1 Code Documentation

All migrated functions must have:

1. **Type Hints**
   ```python
   def compute_embedding(
       data: npt.NDArray[np.floating],
       method: Literal["pca", "umap", "tsne"] = "pca",
       n_components: int = 2,
   ) -> npt.NDArray[np.floating]:
   ```

2. **Docstrings** (NumPy style)
   ```python
   """Compute dimensionality reduction embedding.

   Parameters
   ----------
   data : ndarray of shape (n_samples, n_features)
       Input data matrix.
   method : {'pca', 'umap', 'tsne'}, default='pca'
       Embedding method to use.
   n_components : int, default=2
       Number of dimensions in output.

   Returns
   -------
   embedding : ndarray of shape (n_samples, n_components)
       Reduced dimensionality representation.
   """
   ```

3. **Inline Comments**
   - Explain complex algorithms
   - Document non-obvious decisions
   - Note edge cases

### 7.2 Example Notebooks

Create comprehensive example notebooks for each new module:

| Module | Notebook | Status |
|--------|----------|--------|
| embeddings | `embeddings_demo.ipynb` | 📋 Planned (Phase 1) |
| decoding | `decoding_examples.ipynb` | 📋 Planned (Phase 1) |
| similarity | `similarity_examples.ipynb` | 📋 Planned (Phase 1) |
| topology | `structure_index_examples.ipynb` | 📋 Planned (Phase 1) |

**Notebook Structure**:
1. Import and setup
2. Generate/load example data
3. Basic usage examples
4. Advanced usage examples
5. Visualization examples
6. Parameter tuning guidance

### 7.3 API Documentation

Update Sphinx documentation:
- Add new modules to `docs/api/index.rst`
- Run `make html` to generate docs
- Verify examples render correctly

---

## 8. BACKWARD COMPATIBILITY

### 8.1 Deprecation Strategy

For functions being replaced (not just moved):

1. **Keep old function** with deprecation warning:
   ```python
   import warnings
   
   def old_function(*args, **kwargs):
       warnings.warn(
           "old_function is deprecated, use new_function instead",
           DeprecationWarning,
           stacklevel=2
       )
       return new_function(*args, **kwargs)
   ```

2. **Document in changelog**:
   - What's deprecated
   - What to use instead
   - When it will be removed (e.g., version X.Y.Z)

3. **Provide migration guide** in docs

### 8.2 Import Compatibility

Maintain import paths where possible:

```python
# Old import (still works)
from neural_analysis.plotting import plot_scatter_3d

# New import (preferred)
from neural_analysis.plotting.plots_3d import plot_scatter_3d
```

---

## 9. FUTURE ENHANCEMENTS

### 9.1 Performance Optimization

**Potential Improvements**:
- Profile batch comparison functions
- Consider parallel processing for large datasets
- Optimize HDF5 I/O patterns
- Add caching for expensive computations

**Priority**: LOW (only if performance issues identified)

---

### 9.2 Additional Metrics

**Possible Additions**:
- More distribution distance metrics
- Additional shape comparison methods
- Temporal analysis metrics
- Information-theoretic measures

**Priority**: MEDIUM (as needed by research projects)

---

### 9.3 Interactive Visualization

**Ideas**:
- Dash/Streamlit dashboard for data exploration
- Interactive plotly widgets in notebooks
- Real-time plotting for streaming data

**Priority**: LOW (research nice-to-have)

---

## 10. DEPRECATED/ARCHIVED

### Files Removed ✅
- ✅ `src/neural_analysis/metrics/shape.py` (migrated to distributions.py)
- ✅ `tests/test_plotting_new.py.duplicate_backup` (removed duplicate)

### Documentation Archived
- ✅ `plotting_integration_summary.md` (migration complete)
- ✅ `synthetic_data_enhancements_completed.md` (enhancements complete)

---

## 11. MIGRATION WORKFLOW TEMPLATE

For each function/module to migrate:

1. **Assess Necessity**
   - Is it used? Check grep/references
   - Does numpy/scipy/sklearn provide this?
   - Does it duplicate existing code?

2. **Plan Location**
   - Which module does it belong in?
   - Does it fit existing API patterns?

3. **Refactor Code**
   - Add type hints
   - Add comprehensive docstring (NumPy style)
   - Improve error handling
   - Follow project style guide

4. **Test Implementation**
   - Write unit tests (target >90% coverage)
   - Test edge cases
   - Test with synthetic and real data

5. **Document**
   - Add to example notebook
   - Update API documentation
   - Add usage examples to docstring

6. **Verify**
   - All tests pass
   - Linting clean
   - Type checking passes
   - Example notebook runs successfully

7. **Mark Complete**
   - Update this TODO
   - Remove from legacy file
   - Delete legacy file when empty

---

## 12. QUICK REFERENCE

### Current Module Structure
```
src/neural_analysis/
├── decoding.py              # Placeholder - needs migration
├── embeddings/              # ⚠️ NOT YET CREATED - HIGH PRIORITY
├── example.py               # Example/template code
├── metrics/
│   ├── distance.py         # Distance metrics
│   ├── distributions.py    # Distribution comparison + shape distances ✅
│   ├── outliers.py         # Outlier detection
│   └── similarity.py       # Similarity metrics
├── plotting/
│   ├── core/               # PlotGrid system ✅
│   ├── plots_1d.py         # 1D plots ✅
│   ├── plots_2d.py         # 2D plots ✅
│   ├── plots_3d.py         # 3D plots ✅
│   ├── plots_heatmaps.py   # Heatmaps ✅
│   ├── plots_statistical.py # Statistical plots ✅
│   └── synthetic_plots.py  # Synthetic data viz ✅
├── synthetic_data.py       # Data generation ✅
├── topology/               # Structure index and topology ✅
└── utils/
    ├── io/                 # I/O utilities ✅
    ├── preprocessing.py    # Data preprocessing ✅
    └── validation.py       # Input validation ✅
```

### Test Status
- **Total Tests**: 181 unique tests
- **Passing**: 181/181 (100%) ✅
- **Coverage**: 100% of migrated code
- **Test Organization**: Organized by module in tests/

### Commands
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/neural_analysis --cov-report=html

# Run specific module tests
pytest tests/test_distributions.py -v

# Run linting
make lint

# Run type checking
mypy src/
```

---

## 13. NOTES & DECISIONS

### 13.1 Architecture Decisions
1. **Shape distances in distributions.py**: Shape comparison treats neural populations as distributions in feature space, making it conceptually a distribution comparison method.

2. **General batch framework**: `pairwise_distribution_comparison_batch()` provides unified interface for any pairwise metric computation with HDF5 caching.

3. **PlotGrid system**: All plotting uses PlotGrid for consistency, testability, and backend agnosticism.

### 13.2 Migration Philosophy

- **Preserve functionality** without blindly copying code
- **Improve architecture** with modern Python practices
- **Add comprehensive testing** for reliability
- **Create examples** for usability

### 13.3 Success Metrics

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Tests passing | 181/181 (100%) | 181/181 (100%) | ✅ Maintained |
| Code coverage | 100% (migrated code) | >90% (all code) | 🔄 In Progress |
| Legacy code remaining | ~16,134 lines | 0 lines | ❌ To Do |
| Example notebooks | 11 | 15-16 | 🔄 4-5 more needed |
| API documentation | Partial | Complete | 🔄 In Progress |

---

**End of TODO**
