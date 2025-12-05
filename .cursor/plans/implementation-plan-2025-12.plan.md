# Implementation Plan - Neural Analysis Migration

**Created**: December 2025
**Status**: Planning Phase
**Goal**: Complete migration of legacy code from `/todo` folder to modernized codebase

---

## Executive Summary

This plan outlines the systematic migration of ~16,134 lines of legacy code from the `/todo` folder into the modernized `neural-analysis` codebase. The migration follows a phased approach prioritizing critical analysis functions, then visualization, utilities, and finally cleanup.

**Current State**:

- ✅ Core infrastructure complete (PlotGrid, metrics, plotting)
- ✅ 181/181 tests passing (100% coverage)
- ❌ ~16,134 lines of legacy code in `/todo` folder
- ❌ Missing: Embeddings visualization, Decoding framework, Graph analysis
- ❌ Code quality issues: mypy/ruff errors in plotting module
- ❌ Bug fixes needed: shape distance metrics, phase3_api_demo.ipynb issues

**Target State**:

- All critical analysis functions migrated and tested
- Complete documentation and example notebooks
- Zero legacy code in `/todo` folder
- All mypy/ruff errors fixed
- All known bugs resolved
- Full feature parity with legacy system

---

## Phase 0: Code Quality & Critical Bug Fixes (Priority 🔴)

### 0.1 Fix Type Checking and Linting for Plotting Module

**Status**: ✅ COMPLETE - HIGH PRIORITY

**Current State**:

- ✅ All mypy and ruff errors fixed in `neural_analysis.plotting`
- ✅ All plotting modules pass type checking
- ✅ Zero errors remaining

**Tasks**:

1. ✅ Remove plotting module exclusions from `pyproject.toml` (lines 164-174)
2. ✅ Run mypy on plotting modules to identify all type errors (~200 errors found)
3. ✅ Fix all ruff linting errors (all fixed)
4. ✅ Fix type annotations systematically (all ~200 mypy errors fixed)

- `plotting/renderers.py` (~70 errors)
- `plotting/grid_config.py` (~60 errors)
- `plotting/core.py` (~15 errors)
- `plotting/statistical_plots.py` (~10 errors)
- `plotting/plots_1d.py`, `plots_2d.py`, `plots_3d.py` (~20 errors each)
- `plotting/synthetic_plots.py` (~15 errors)
- `plotting/embeddings.py` (~10 errors)

5. ⏳ Verify all tests still pass after fixes
6. ⏳ Ensure no regressions in functionality

**Dependencies**: None
**Estimated Effort**: 8-12 hours (revised - more errors than initially estimated)
**Deliverables**: All plotting modules pass mypy and ruff checks

---

### 0.2 Fix Shape Distance Metric Bugs

**Status**: ❌ Not Started - HIGH PRIORITY

**Current State**:

- Shape distance property violation: soft-matching ≤ one-to-one ≤ procrustes not always satisfied
- Need to verify metric computation correctness after changing default to sqeuclidean

**Tasks**:

1. **Fix one-to-one matching comparison** (`phase3_api_demo.ipynb`)

- Ensure shape distance property: soft-matching ≤ one-to-one ≤ procrustes
- Test all three shape metrics with same datasets
- Verify ordering: procrustes_distance >= one-to-one_distance >= soft-matching_distance

2. **Verify shape_distance_soft_matching() metric computation** (`distributions.py:1410-1465`)

- After changing default metric to sqeuclidean for `shape_distance_one_to_one()`
- Check if square root is needed at end of `shape_distance_soft_matching()`
- Verify mathematical correctness of metric computation
- Ensure consistency with `shape_distance_one_to_one()` (lines 1025-1367)

3. **Fix shape distance violations in phase3_api_demo.ipynb**

- Compare all 3 shape metrics with each other (cell 15+)
- Find and fix computation logic errors
- Ensure results comparing same dataset shapes follow ordering rule
- Rerun cells to verify fixes

**Dependencies**: None
**Estimated Effort**: 3-5 hours
**Deliverables**: All shape distance metrics compute correctly and satisfy ordering property

---

### 0.3 Fix phase3_api_demo.ipynb Issues

**Status**: ❌ Not Started - HIGH PRIORITY

**Current State**:

- Multiple issues identified in `examples/phase3_api_demo.ipynb`
- Need comprehensive fixes and verification

**Tasks**:

1. **Fix compute_all_pairs() inf/nan issue** (cell 19)

- `compute_all_pairs(sessions, metric='wasserstein')` should never return inf
- Check wasserstein distance computation
- Ensure all distances are finite and non-nan
- Add validation to prevent inf/nan results

2. **Investigate Kolmogorov-Smirnov distance** (cell 20)

- Check why KS distance is 1 when comparing condition A vs B
- Verify if result should be different
- Fix computation if incorrect

3. **Enhance performance comparison visualization** (cell 22)

- Provide all possible within distance metrics in one subplot with `parallel=True`
- Create second subplot with `parallel=False`
- Verify that parallel and non-parallel solutions produce same results
- If results differ, find and fix computation error
- Rerun cell to test

4. **Fix table of contents** (cell 2)

- Fix broken links in table of contents
- Ensure all navigation links work correctly

5. **Convert all plots to GridPlot logic** (starting from cell 15)

- Replace any non-GridPlot plotting code
- Ensure all visualizations use PlotGrid system
- Maintain visual consistency

**Dependencies**: None
**Estimated Effort**: 4-6 hours
**Deliverables**: All cells in phase3_api_demo.ipynb work correctly, all plots use GridPlot

---

### 0.4 Refactor Auto-Save/Load Logic

**Status**: ❌ Not Started - HIGH PRIORITY

**Current State**:

- Auto-save/load logic in `pairwise_metrics.py:1312-1358` is specific to pairwise metrics
- Need general, reusable function for other future methods

**Tasks**:

1. **Design general auto-save/load function**

- Analyze current implementation in `pairwise_metrics.py:1312-1358`
- Design strongly general interface
- Plan proper abstraction for reuse

2. **Implement in appropriate location**

- Move to `utils/io.py` or `utils/comparison_store.py`
- Create general function that can be used by any computation method
- Maintain backward compatibility with existing pairwise_metrics usage

3. **Refactor pairwise_metrics.py to use new function**

- Replace existing auto-save/load logic
- Use new general function
- Verify functionality unchanged

4. **Add tests and documentation**

- Unit tests for general auto-save/load function
- Documentation with usage examples
- Update function registry

**Dependencies**: None
**Estimated Effort**: 3-4 hours
**Deliverables**: General auto-save/load function in utils, pairwise_metrics.py refactored

---

### 0.5 Fix Head Direction Plotting

**Status**: ❌ Not Started

**Current State**:

- `synthetic_data.py:779-815` generates head direction values between 0 to 2π
- `synthetic_plots.py:1605-1661` plotting function doesn't show full range properly

**Tasks**:

1. Update head direction plotting function in `synthetic_plots.py:1605-1661`
2. Ensure plots show full range 0-2π for head direction example cells
3. Fix circular axis handling if needed
4. Verify plots display correctly

**Dependencies**: None
**Estimated Effort**: 1-2 hours
**Deliverables**: Head direction plots show full 0-2π range correctly

---

### 0.6 Verify Jensen-Shannon Divergence Binning

**Status**: ❌ Not Started

**Current State**:

- Adaptive binning implemented in `jensen_shannon_divergence()` to handle memory issues
- Need to verify method doesn't destroy calculation accuracy

**Tasks**:

1. Review binning method in `distributions.py` (see `PHASE4_PROGRESS.md:149-188`)
2. Verify adaptive binning formula: `bins_adaptive = max(3, bins^(3/D))`
3. Test that results are not significantly affected by bin reduction
4. Document trade-offs between memory and accuracy
5. Add validation tests if needed

**Dependencies**: None
**Estimated Effort**: 1-2 hours
**Deliverables**: Verified binning method maintains calculation accuracy

---

## Phase 1: Critical Analysis Functions (Priority 🔴)

### 1.1 Embeddings Module - Complete Migration

**Status**: Partial - Core `compute_embedding()` exists, but visualization and multi-embedding comparison missing

**Current State**:

- ✅ `src/neural_analysis/embeddings/dimensionality_reduction.py` - Core embedding computation
- ✅ `src/neural_analysis/embeddings/visualization.py` - Exists but needs enhancement
- ❌ Multi-embedding comparison plots missing
- ❌ Integration with legacy `create_multiple_embeddings()` from `Manimeasure.py`

**Tasks**:

1. **Audit existing embeddings module**

- Review `dimensionality_reduction.py` for completeness
- Check `visualization.py` for existing functionality
- Compare with `Manimeasure.py` functions (lines 47-320)

2. **Migrate `create_multiple_embeddings()` from Manimeasure.py**

- Convert to PlotGrid-based system
- Support all embedding methods (PCA, UMAP, t-SNE, MDS, Isomap, LLE, Spectral, PTU)
- Add comprehensive tests

3. **Migrate `simple_embedd()` visualization**

- Integrate with existing `compute_embedding()`
- Add PlotGrid visualization wrapper
- Support multiple plot types (center, center_std, samples, flow, annotate_dots)

4. **Enhance PCA variance analysis**

- Add power-law fitting for eigenvalue spectra (from TODO section 1.8.B)
- Create visualization for explained variance with power-law fit
- Add α (slope) as dimensionality descriptor

5. **Create example notebook**

- `examples/embeddings_demo.ipynb`
- Show all embedding methods
- Compare methods side-by-side
- Demonstrate power-law analysis

**Dependencies**: scikit-learn, umap-learn, plotly
**Estimated Effort**: 8-12 hours
**Deliverables**: Enhanced embeddings module, tests, example notebook

---

### 1.2 Decoding Module - Complete Framework

**Status**: Partial - Basic `knn_decoder` exists in `learning/decoding.py`, but full framework missing

**Current State**:

- ✅ `src/neural_analysis/learning/decoding.py` - Basic k-NN decoder exists
- ❌ Missing: Full decoding analysis framework from `Manimeasure.py` (lines 322-664)
- ❌ Missing: Cross-validation framework
- ❌ Missing: Multiple decoder types (SVM, LogisticRegression, RandomForest)
- ❌ Missing: Comprehensive evaluation metrics

**Tasks**:

1. **Audit existing decoding module**

- Review `learning/decoding.py` for existing functionality
- Compare with `Manimeasure.py` functions

2. **Create decoder wrapper classes**

- Base decoder class with unified interface
- Implement wrappers for: SVM, LogisticRegression, RandomForest, k-NN
- Support both regression and classification

3. **Implement cross-validation framework**

- Migrate `get_train_test_data()` from Manimeasure.py
- Support KFold, StratifiedKFold, TimeSeriesSplit
- Add train/test splitting utilities

4. **Add comprehensive evaluation**

- Migrate `compute_decoding_accuracy()` from Manimeasure.py
- Add metrics: accuracy, confusion matrix, ROC curves, decoding curves
- Create visualization functions using PlotGrid

5. **Create main decoding analysis function**

- Migrate `decoding_analysis()` from Manimeasure.py
- Unified interface for decoder training and evaluation
- Support multiple decoder types in single call

6. **Create example notebook**

- `examples/decoding_demo.ipynb`
- Show all decoder types
- Demonstrate cross-validation
- Visualize decoding performance

**Dependencies**: scikit-learn, scipy
**Estimated Effort**: 10-15 hours
**Deliverables**: Complete decoding module, tests, example notebook

---

### 1.3 Similarity Analysis - Extend Existing Module

**Status**: Partial - Basic similarity metrics exist, but population similarity and RSA missing

**Current State**:

- ✅ `src/neural_analysis/metrics/similarity.py` - Basic similarity metrics exist
- ❌ Missing: Population similarity metrics from `Manimeasure.py` (lines 666-1,882)
- ❌ Missing: Representational Similarity Analysis (RSA)
- ❌ Missing: Cross-session comparisons

**Tasks**:

1. **Audit existing similarity module**

- Review `metrics/similarity.py` for existing functionality
- Identify gaps vs. Manimeasure.py

2. **Add population similarity metrics**

- Population-level similarity calculations
- Cross-population comparisons
- Temporal similarity analysis

3. **Implement Representational Similarity Analysis (RSA)**

- RSA computation functions
- RSA visualization
- Integration with existing similarity metrics

4. **Add cross-session comparison tools**

- Session-to-session similarity
- Stability metrics
- Visualization for cross-session analysis

5. **Create example notebook**

- `examples/similarity_analysis_demo.ipynb`
- Show population similarity
- Demonstrate RSA
- Cross-session comparisons

**Dependencies**: scipy, scikit-learn
**Estimated Effort**: 6-8 hours
**Deliverables**: Extended similarity module, tests, example notebook

---

### 1.4 Structure Index - Topology Module Enhancement

**Status**: Unknown - Need to compare `todo/structure_index.py` with existing `topology/` module

**Current State**:

- ✅ `src/neural_analysis/topology/` - Exists but needs audit
- ❌ `todo/structure_index.py` (864 lines) - Not migrated

**Tasks**:

1. **Compare existing topology module with structure_index.py**

- Identify overlapping functionality
- Find unique features in structure_index.py

2. **Migrate unique functionality**

- Integrate missing structure index methods
- Ensure compatibility with existing topology module

3. **Create unified structure index interface**

- Single entry point for structure index computation
- Support multiple topology analysis methods

4. **Add visualization**

- Structure index plots using PlotGrid
- Topology visualization

5. **Create example notebook**

- `examples/structure_index_demo.ipynb`
- Show structure index computation
- Visualize topology analysis

**Dependencies**: networkx, scikit-learn
**Estimated Effort**: 4-6 hours
**Deliverables**: Enhanced topology module, tests, example notebook

---

### 1.5 Shape Similarity Visualization

**Status**: ❌ Not Started - HIGH PRIORITY

**Current State**:

- Legacy code exists in `Manimeasure.py:1355-1421`
- Need modern, general, modular implementation using PlotGrid

**Tasks**:

1. **Design visualization system**

- Analyze legacy code in `Manimeasure.py:1355-1421`
- Design general, modular plotting function for shape similarity
- Plan to use GridPlot logic from `grid_config.py`
- Ensure all rendering happens in `renderers.py`

2. **Update function registry**

- Run `scripts/generate_function_registry.py`
- Check available functions using registry
- Follow instructions in `.github/instructions/claude.instructions.md`

3. **Implement visualization function**

- Use example datasets to compare shapes
- Create visualizations with arrows showing grouping and behavior in space
- Use MDS to embed distance matrix into lower dimensions
- Compare MDS to 20 dimensions + PCA to 2D
- Use embedding methods from `dimensionality_reduction.py`
- Integrate with GridPlot system

4. **Create example notebook**

- Test visualization with various datasets
- Show grouping and proper behavior in space
- Demonstrate MDS and PCA embedding comparisons
- Verify arrows and visual elements work correctly

**Dependencies**: scikit-learn, plotly
**Estimated Effort**: 6-8 hours
**Deliverables**: General shape similarity visualization function, example notebook

---

### 1.6 Function Registry Update & Notebook Fixes

**Status**: ❌ Not Started

**Current State**:

- Function registry may be outdated
- Notebooks have issues that need fixing

**Tasks**:

1. **Update function registry**

- Run `scripts/generate_function_registry.py`
- Check `.github/instructions/claude.instructions.md` for proper procedure
- Verify registry is up to date

2. **Fix synthetic_datasets_example.ipynb**

- Run all cells starting from beginning
- Fix any issues encountered
- Ensure all outputs are correct
- Update outputs if needed

3. **Fix io_h5io_examples.ipynb**

- Run all cells
- Fix any issues encountered
- Ensure all I/O operations work correctly
- Verify HDF5 read/write operations

**Dependencies**: None
**Estimated Effort**: 2-4 hours
**Deliverables**: Updated function registry, fixed notebooks

---

### 1.7 Synthetic Datasets & Classification Enhancement

**Status**: ❌ Not Started

**Current State**:

- Synthetic data generation exists in `synthetic_data.py`
- Need to enhance and create comprehensive analysis notebook
- Need cell type classifier

**Tasks**:

1. **Create classification.py module**

- Implement cell type classifier
- Support classification on mixed populations
- Integrate with existing synthetic data generators

2. **Enhance synthetic datasets usage**

- Use datasets to test neural analysis methods
- Benchmark dimensionality reduction algorithms
- Validate decoding approaches with known ground truth
- Test cell type classification on mixed populations
- Study how noise affects embedding quality

3. **Create comprehensive analysis notebook**

- `examples/neural_analysis_example.ipynb`
- Use `synthetic_data.py` for creating datasets
- Use `structure_index.py` for topology analysis
- Use shape similarity from `distributions.py`
- Include all analysis types:
 - Dimensionality reduction benchmarking
 - Decoding validation
 - Cell type classification
 - Noise impact analysis
 - Embedding quality assessment

**Available Functions** (from summary):

- Place cells (1D, 2D, 3D)
- Grid cells (1D, 2D, 3D)
- Head direction cells
- Mixed populations
- Manifold mappings (Ring S¹, Torus T²)
- Population vector decoding
- sklearn datasets (Swiss roll, S-curve, blobs, moons, circles)

**Dependencies**: scikit-learn, numpy, scipy
**Estimated Effort**: 8-12 hours
**Deliverables**: `classification.py` module, comprehensive `neural_analysis_example.ipynb`

---

## Phase 2: Visualization Completion (Priority 🟡)

### 2.1 Visualizer.py Audit and Migration

**Status**: 🟡 PARTIALLY MIGRATED - Many functions already in PlotGrid, but some unique functions remain

**Current State**:

- ✅ PlotGrid system complete
- ✅ Most common plots migrated
- ❌ `todo/Visualizer.py` (7,586 lines) - Needs comprehensive audit

**Tasks**:

1. **Comprehensive function audit**

- List all functions in Visualizer.py
- Check against existing PlotGrid functions
- Identify unique/valuable functions

2. **Migration decision matrix**

- ✅ Already migrated → Document and remove from Visualizer.py
- 🔄 Unique & useful → Migrate to PlotGrid system
- ❌ Obsolete/redundant → Document and deprecate

3. **Migrate unique functions**

- Batch heatmap functions → `plotting/plots_heatmaps.py`
- Specialized statistical plots → `plotting/plots_statistical.py`
- Embedding-specific plots → `embeddings/visualization.py`
- Animation functions (if needed) → Defer or create new module

4. **Extract utilities**

- Colormap functions → `plotting/core.py`
- Helper utilities → `plotting/core.py`

5. **Update documentation**

- Migration guide from Visualizer.py to PlotGrid
- Function mapping table

**Dependencies**: PlotGrid system
**Estimated Effort**: 4-6 hours audit + variable migration time
**Deliverables**: Cleaned Visualizer.py, enhanced plotting modules, migration guide

---

## Phase 3: Utilities Migration (Priority 🟢)

### 3.1 Helper.py Review and Migration

**Status**: ❌ NOT MIGRATED - 4,731 lines of utility functions

**Current State**:

- ✅ `src/neural_analysis/utils/` - Exists with some utilities
- ❌ `todo/Helper.py` (4,731 lines) - Needs comprehensive review

**Tasks**:

1. **Generate function inventory**

- List all functions in Helper.py
- Categorize by purpose (preprocessing, I/O, validation, math, etc.)

2. **Cross-reference with existing code**

- Check against `src/neural_analysis/utils/` modules
- Check against numpy/scipy/sklearn standard functions
- Identify duplicates

3. **Categorize functions**

- **Keep & Migrate**: Unique, useful utilities
- **Document as Available**: Already in numpy/scipy/sklearn
- **Obsolete**: No longer needed

4. **Migrate unique functions**

- Data preprocessing → `utils/preprocessing.py`
- I/O operations → `utils/io/`
- Validation → `utils/validation.py`
- Mathematical utilities → `utils/math_utils.py` (create if needed)

5. **Add tests and documentation**

- Unit tests for all migrated functions
- Type hints and docstrings
- Usage examples

**Dependencies**: None
**Estimated Effort**: 8-12 hours
**Deliverables**: Enhanced utils modules, deprecated functions list, tests

---

## Phase 4: New Research Features (Priority 🟡)

### 4.1 Grid Cell Analysis

**Status**: ❌ Not Started

**Tasks**:

1. Implement torus mapping algorithm
2. Create trajectory visualization with angle-based coloring
3. Add spatial continuity analysis
4. Integrate with PlotGrid
5. Create example notebook

**Dependencies**: numpy, scipy, matplotlib/plotly
**Estimated Effort**: 6-8 hours

---

### 4.2 High Dimensional Data Analysis

**Status**: ❌ Not Started

**Tasks**:

1. Implement tangeling calculation function
2. Create power-law fitting for eigenvalue spectra
3. Enhance PCA plots with power-law analysis
4. Add visualization functions
5. Create example notebook

**Dependencies**: numpy, scipy, scikit-learn
**Estimated Effort**: 8-12 hours

---

### 4.3 Graph Analysis

**Status**: ❌ Not Started - HIGH PRIORITY

**Tasks**:

1. Implement comprehensive graph property extraction
2. Create graph creation pipeline with multiple metrics
3. Add STTC (spike time tiling coefficient) calculation
4. Implement community detection algorithms
5. Implement heat equation analysis for module detection
6. Create graph visualization functions
7. Integrate with PlotGrid
8. Create example notebook

**Dependencies**: networkx, scipy, scikit-learn, igraph (optional)
**Estimated Effort**: 20-30 hours

---

### 4.4 High Dimensionality Analysis Based on Extracted Properties

**Status**: ❌ Not Started

**Tasks**:

1. Design property aggregation framework
2. Implement property merging for neurons, datasets, animals
3. Create similarity analysis on integrated representations
4. Add clustering methods
5. Implement trajectory analysis for learning experiments
6. Add visualization functions
7. Create example notebook

**Dependencies**: numpy, scipy, scikit-learn, all other analysis modules
**Estimated Effort**: 12-18 hours

---

## Phase 5: Storage System Integration (Priority 🔴)

### 5.1 Multi-layer Storage System

**Status**: In progress

**Tasks**:

1. Finalize storage helper stack (`src/neural_analysis/utils/storage/`)
2. Ensure high-volume workflows leverage unified manager
3. Provide reproducible setup via Docker/Compose
4. Document usage in `examples/storage_demo.ipynb`
5. Backfill regression tests
6. Finish documentation updates
7. Harden cache invalidation + SQL query helpers
8. Run end-to-end validation

**Dependencies**: Redis, DuckDB, HDF5
**Estimated Effort**: Variable (ongoing)

---

## Phase 6: Cleanup and Documentation (Priority ⚪)

### 6.1 Pipeline Utilities

**Tasks**:

1. Review `yaml_creator.py` and `restructure.py`
2. Move to `scripts/data_preprocessing/` if actively used
3. Document if pipeline-specific

**Estimated Effort**: 1-2 hours each

---

### 6.2 Documentation Completion

**Tasks**:

1. API reference documentation (Sphinx)
2. Migration guide from Visualizer.py to PlotGrid
3. Comprehensive plotting cookbook
4. Refresh example notebooks (re-execute and update outputs)

**Estimated Effort**: 6-8 hours + 2-3 hours for notebooks

---

### 6.3 Final Cleanup

**Tasks**:

1. Remove/deprecate empty todo/ folder
2. Archive completed documentation
3. Final test suite run
4. Update README with final status

**Estimated Effort**: 1-2 hours

---

## Implementation Strategy

### Workflow for Each Migration Task

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

- Update TODO.md
- Remove from legacy file
- Delete legacy file when empty

---

## Risk Mitigation

### Technical Risks

- **Breaking changes**: Maintain backward compatibility where possible
- **Performance regression**: Profile before/after, optimize if needed
- **Missing dependencies**: Add optional dependencies, graceful degradation
- **Test coverage gaps**: Require >90% coverage for all new code

### Process Risks

- **Scope creep**: Stick to documented plan, defer enhancements
- **Incomplete migration**: Follow checklist, mark progress in TODO.md
- **Lost functionality**: Comprehensive testing, example notebooks
- **Timeline delays**: Focus on critical features first

---

## Success Metrics

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Tests passing | 181/181 (100%) | 181+ (100%) | ✅ Maintained |
| Code coverage | 100% (migrated) | >90% (all code) | 🔄 In Progress |
| Legacy code remaining | ~16,134 lines | 0 lines | ❌ To Do |
| Example notebooks | 11 | 15-16 | 🔄 4-5 more needed |
| API documentation | Partial | Complete | 🔄 In Progress |
| Mypy errors | Excluded | 0 errors | 🔄 In Progress (~200 remaining) |
| Ruff errors | 11 errors | 0 errors | ✅ Fixed |

---

## Timeline Estimate

| Phase | Priority | Effort (hours) | Weeks (part-time) | Days (full-time) |
|-------|----------|----------------|-------------------|------------------|
| Phase 0: Code Quality & Bug Fixes | 🔴 | 16-25 | 1-2 | 2-3 |
| Phase 1: Critical Analysis | 🔴 | 28-41 | 2-3 | 4-6 |
| Phase 2: Visualization | 🟡 | 15-25 | 1-2 | 2-3 |
| Phase 3: Helper.py | 🟢 | 11-21 | 1-2 | 2-3 |
| Phase 4: New Features | 🟡 | 46-68 | 3-4 | 6-9 |
| Phase 5: Storage | 🔴 | Variable | Ongoing | Ongoing |
| Phase 6: Cleanup | ⚪ | 10-15 | <1 | 1-2 |
| **Total** | - | **126-210 hrs** | **8-14 weeks** | **17-33 days** |

**Assumptions**:

- Part-time: 15-20 hours/week
- Full-time: 6-8 hours/day
- Estimates include implementation, testing, documentation

---

**End of Implementation Plan**