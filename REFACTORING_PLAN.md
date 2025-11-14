# Refactoring Plan: Consolidate similarity.py into distance.py

**STATUS**: ✅ COMPLETED (November 13, 2025)

## Analysis Summary

### Current Structure:
1. **distance.py** (~400 lines):
   - Point-to-point distances: euclidean, manhattan, mahalanobis, cosine
   - Pairwise distance computation with numba acceleration
   - Plugin system for metric dispatch

2. **similarity.py** (~1150 lines):
   - Correlation functions: Pearson, Spearman, Kendall
   - Cosine/angular similarity matrices
   - Spatial autocorrelation (1D, 2D, 3D)
   - Orchestration functions: correlation(), spatial_autocorrelation()

3. **distributions.py** (~1850 lines):
   - Uses distance.py for pairwise_distance
   - Distribution comparisons (Wasserstein, KS, JS)
   - Shape distance methods
   - Group comparison functions

### Code Duplication Identified:

1. **Pairwise Computation Pattern** (similarity.py and distance.py):
   - Both have numba-accelerated pairwise loops
   - Similar error handling for empty/mismatched arrays
   - Duplicate input validation

2. **Cosine Similarity** (appears in both modules):
   - distance.py: cosine_similarity() function
   - similarity.py: cosine_similarity_matrix() function
   - Can be unified with single implementation

3. **Parallel Dispatch Logic**:
   - Both modules check NUMBA_AVAILABLE and dispatch
   - Similar pattern: if parallel and NUMBA_AVAILABLE: ...

## Refactoring Strategy

### Phase 1: Module Renaming
- Rename `distance.py` → `pairwise_metrics.py`
  - Better describes unified functionality
  - Encompasses: distances, similarities, correlations, autocorrelations

### Phase 2: Code Consolidation
Migrate from similarity.py to pairwise_metrics.py:

**Section A: Correlation Functions** (keep together)
- correlation() orchestrator
- correlation_matrix()
- _correlation_matrix_parallel()
- _spearman_numba(), _kendall_numba(), _rank_data_numba()

**Section B: Similarity Matrices** (unify with existing cosine)
- Merge cosine_similarity() and cosine_similarity_matrix()
- Keep angular_similarity_matrix()
- Keep similarity_matrix() orchestrator
- _cosine_similarity_matrix_parallel()
- _angular_similarity_matrix_parallel()

**Section C: Spatial Autocorrelation** (new section)
- spatial_autocorrelation() orchestrator
- _compute_1d_autocorrelation()
- _compute_2d_autocorrelation()
- _compute_3d_autocorrelation()

**Section D: Refactored Helpers** (extract common patterns)
- _validate_pairwise_inputs()
- _dispatch_parallel()
- _pairwise_loop_template()

### Phase 3: Update Dependencies

**Files to update:**
1. `src/neural_analysis/metrics/__init__.py`
   - Update lazy imports
   - Export from pairwise_metrics instead of similarity/distance

2. `src/neural_analysis/plotting/synthetic_plots.py`
   - Change: `from neural_analysis.metrics.similarity import spatial_autocorrelation`
   - To: `from neural_analysis.metrics.pairwise_metrics import spatial_autocorrelation`

3. `src/neural_analysis/metrics/distributions.py`
   - Change: `from .distance import pairwise_distance`
   - To: `from .pairwise_metrics import pairwise_distance`
   - Ensure distribution_distance works with new metric names

4. Test files:
   - Update all test imports
   - Verify test coverage maintained

### Phase 4: Delete Old Module
- Remove `src/neural_analysis/metrics/similarity.py`

## New Module Structure: pairwise_metrics.py

```
# =============================================================================
# Module: pairwise_metrics.py
# =============================================================================

# Imports and setup (numba, logging)

__all__ = [
    # Point-to-point distances
    "euclidean_distance",
    "manhattan_distance", 
    "mahalanobis_distance",
    # Pairwise computations
    "pairwise_distance",
    # Correlation
    "correlation",
    "correlation_matrix",
    # Similarity
    "cosine_similarity",
    "angular_similarity",
    "similarity_matrix",
    # Spatial autocorrelation
    "spatial_autocorrelation",
]

# =============================================================================
# Section 1: Point-to-Point Distance Metrics
# =============================================================================
# (Existing from distance.py)

# =============================================================================
# Section 2: Correlation Functions
# =============================================================================
# (From similarity.py)

# =============================================================================
# Section 3: Similarity Matrices
# =============================================================================
# (Merged from both modules)

# =============================================================================
# Section 4: Spatial Autocorrelation
# =============================================================================
# (From similarity.py)

# =============================================================================
# Section 5: Pairwise Computation System
# =============================================================================
# (Unified from both modules)

# =============================================================================
# Section 6: Parallel Computation Helpers
# =============================================================================
# (Unified numba implementations)
```

## Risk Mitigation

1. **Backward Compatibility**: Update __init__.py to re-export with old names
2. **Test Coverage**: Run full test suite after each phase
3. **Incremental Migration**: Commit after each successful phase
4. **Deprecation Warnings**: Could add warnings for old imports (optional)

## Execution Order

1. ✅ Create this plan document
2. ✅ Create new pairwise_metrics.py with merged content
3. ✅ Update imports in synthetic_plots.py
4. ✅ Update imports in distributions.py
5. ✅ Update __init__.py
6. ✅ Update test files
7. ✅ Run tests (65 tests passing)
8. ✅ Run linting
9. ✅ Delete similarity.py and old distance.py
10. ✅ Final verification

## Completion Summary (November 13, 2025)

### Phase 1: Module Consolidation ✅

**What was accomplished:**
- Consolidated `distance.py` (496 lines) and `similarity.py` (1150 lines) into unified `pairwise_metrics.py` (803 lines)
- Eliminated ~200 lines of duplicate code (pairwise computation patterns, cosine similarity)
- Updated all imports across codebase:
  - `src/neural_analysis/metrics/__init__.py` (lazy imports)
  - `src/neural_analysis/plotting/synthetic_plots.py` (spatial_autocorrelation)
  - `src/neural_analysis/metrics/distributions.py` (pairwise_distance)
  - `examples/neural_analysis_demo.ipynb` (notebook imports)
- All 65 distance/similarity/distribution tests passing (100%)
- Function registry updated (14 functions in pairwise_metrics)
- Old modules permanently deleted

### Phase 2: Unified Pairwise Computation System ✅

**Problem identified:**
After Phase 1, duplication still existed between `pairwise_metrics.py` and `distributions.py`:
- Both had pairwise iteration logic (lines 265-336 in pairwise_metrics, 1323-1554 in distributions)
- Manual metric dispatch in both modules
- Duplicate input validation and result handling

**Solution implemented:**
1. **Created unified pairwise system** in `pairwise_metrics.py`:
   - `_validate_pairwise_inputs()` - Input validation helper
   - `compute_pairwise_matrix()` - Unified interface for all pairwise metrics
   - Supports point-to-point (euclidean, manhattan, cosine, mahalanobis)
   - Supports distribution-level (wasserstein, kolmogorov-smirnov, jensen-shannon)
   - Supports shape metrics (procrustes, one-to-one, soft-matching)

2. **Refactored distributions.py** to use unified system:
   - `compare_distributions()` now calls `compute_pairwise_matrix()`
   - `pairwise_distribution_comparison_batch()` now calls `compute_pairwise_matrix()`
   - Eliminated ~230 lines of duplicate pairwise logic
   - Single source of responsibility for pairwise computations

3. **Enhanced `pairwise_distance()`**:
   - Now uses `_validate_pairwise_inputs()` helper
   - Improved documentation
   - More modular and maintainable

**Test results:**
- ✅ All 77 metrics tests passing (100%)
- ✅ No regressions in functionality
- ✅ Function registry updated (15 functions in pairwise_metrics)

**Benefits realized:**
- ✅ Single source of truth for all pairwise metrics
- ✅ Better code organization (logical grouping)
- ✅ Easier maintenance (one module vs. two)
- ✅ Clearer API (unified interface)
- ✅ Reduced code duplication (~430 lines eliminated total)
- ✅ Single source of responsibility for pairwise logic
- ✅ More robust and modular architecture

## Benefits

- **Eliminates ~200 lines of duplicate code**
- **Single source of truth for metrics**
- **Better organization**: all pairwise operations in one place
- **Easier to maintain**: one module vs three
- **Clearer API**: unified orchestration functions
