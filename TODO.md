# Detailed Task List

> Every task below maps to a section in [plan.md](plan.md). Tasks are ordered within each phase by dependency.

## Phase 0 — Foundation

- [x] **0.1.1** Read current `.github/workflows/ci.yml` and identify commented-out triggers
- [x] **0.1.2** Uncomment `push` and `pull_request` triggers for `main` and `migration` branches
- [x] **0.1.3** Verify CI runs on a test push to a feature branch
- [x] **0.2.1** Run `uv run mypy src tests` and catalog all existing type errors
- [x] **0.2.2** Fix mypy errors in `utils/` (lowest dependency layer first)
- [x] **0.2.3** Fix mypy errors in `data/`
- [x] **0.2.4** Fix mypy errors in `metrics/`
- [x] **0.2.5** Fix mypy errors in `embeddings/` and `learning/`
- [x] **0.2.6** Fix mypy errors in `topology/`
- [x] **0.2.7** Fix mypy errors in `plotting/`
- [x] **0.2.8** Fix mypy errors in `tests/`
- [x] **0.2.9** Run `uv run ruff format --check .` and fix all format violations
- [x] **0.2.10** Remove `continue-on-error: true` from mypy and ruff format CI steps
- [x] **0.2.11** Confirm CI is green with blocking gates
- [x] **0.3.1** Change version from `0.0.0` to `0.1.0` in `pyproject.toml`
- [x] **0.3.2** Create `.github/workflows/release.yml` for tag-triggered PyPI publishing
- [x] **0.3.3** Test the release workflow with a dry-run (no actual publish)
- [x] **0.4.1** Rewrite `Dockerfile` to use `COPY --from=ghcr.io/astral-sh/uv:latest` and `uv sync`
- [x] **0.4.2** Remove all `pip install` commands from Dockerfile
- [x] **0.4.3** Update `docker-compose.yml` if needed for UV compatibility
- [x] **0.4.4** Test `docker-compose up` and verify the environment works end-to-end

## Phase 1 — Code Quality

- [x] **1.1.1** Split `renderers.py` → `renderers_matplotlib.py` + `renderers_plotly.py`
- [x] **1.1.2** Create `renderers.py` facade that re-exports all functions from both new files
- [x] **1.1.3** Run tests to confirm zero regressions after renderer split
- [x] **1.1.4** Split `grid_config.py` → `grid_config.py` (core dataclasses) + `grid_dispatch.py` (dispatch + PlotGrid logic)
- [x] **1.1.5** Run tests to confirm zero regressions after grid_config split
- [x] **1.1.6** Split `synthetic_data.py` → `generators.py` + `trajectories_gen.py` + `datasets.py`
- [x] **1.1.7** Create `synthetic_data.py` facade that re-exports from new files
- [x] **1.1.8** Run tests to confirm zero regressions after synthetic_data split
- [x] **1.1.9** Split `synthetic_plots.py` → `synthetic_plots_1d.py` + `synthetic_plots_2d.py` + `synthetic_plots_3d.py`
- [x] **1.1.10** Run tests to confirm zero regressions after synthetic_plots split
- [x] **1.1.11** Split `pairwise_metrics.py` → `pairwise_core.py` + `pairwise_numba.py`
- [x] **1.1.12** Create `pairwise_metrics.py` facade that re-exports from both new files
- [x] **1.1.13** Run tests to confirm zero regressions after pairwise_metrics split
- [x] **1.2.1** Define `MATPLOTLIB_RENDERERS: dict[str, Callable]` mapping all 16 plot types
- [x] **1.2.2** Define `PLOTLY_RENDERERS: dict[str, Callable]` mapping all 16 plot types
- [x] **1.2.3** Replace `_plot_spec_matplotlib` if/elif chain with registry lookup
- [x] **1.2.4** Replace `_plot_spec_plotly` if/elif chain with registry lookup
- [x] **1.2.5** Add descriptive `ValueError` when plot type is not found in registry
- [x] **1.2.6** Run renderer tests to confirm all 16 types still work on both backends
- [x] **1.3.1** Delete `src/neural_analysis/utils/preprocessing.py`
- [x] **1.3.2** Remove `preprocessing` from `utils/__init__.py` exports
- [x] **1.3.3** Search for and update any remaining imports of `preprocessing`
- [x] **1.4.1** Create `src/neural_analysis/core/__init__.py` and `core/results.py`
- [x] **1.4.2** Define `AnalysisResult`, `MetricResult`, `EmbeddingResult`, `DecodingResult` dataclasses
- [x] **1.4.3** Add result dataclass returns to `compute_embedding()` alongside existing tuple return
- [x] **1.4.4** Add result dataclass returns to `compute_pairwise_matrix()` alongside existing return
- [x] **1.4.5** Add result dataclass returns to decoding functions alongside existing return
- [x] **1.4.6** Add deprecation warnings on old tuple returns
- [x] **1.4.7** Write tests for all new result dataclasses

## Phase 2 — Usability

- [x] **2.1.1** Identify the top-15 most-used public functions across examples and notebooks
- [x] **2.1.2** Add convenience imports to `src/neural_analysis/__init__.py` for those functions
- [x] **2.1.3** Define `__all__` in the root `__init__.py`
- [x] **2.1.4** Verify no circular import issues with lazy imports where needed
- [x] **2.1.5** Update example notebooks to use short imports
- [x] **2.2.1** Create `src/neural_analysis/pipeline.py`
- [x] **2.2.2** Implement `PipelineConfig` and `PipelineResult` dataclasses
- [x] **2.2.3** Implement `run_analysis()` function with generate → embed → decode → SI steps
- [x] **2.2.4** Write tests for the pipeline with default config
- [x] **2.2.5** Write tests for the pipeline with custom config overrides
- [x] **2.2.6** Add a pipeline example to `examples/`
- [x] **2.3.1** Audit all `match/case` default branches for error message quality
- [x] **2.3.2** Audit all `raise ValueError` calls across `metrics/`
- [x] **2.3.3** Audit all `raise ValueError` calls across `embeddings/` and `learning/`
- [x] **2.3.4** Audit all `raise ValueError` calls across `topology/`
- [x] **2.3.5** Ensure every error follows pattern: "what happened" + "what expected" + "what received"
- [x] **2.3.6** Add `LogFileReference` paths to all user-facing errors (see 2.6)
- [x] **2.4.1** Create `StructureIndexConfig` dataclass
- [x] **2.4.2** Refactor `compute_structure_index` to accept config + kwargs
- [x] **2.4.3** Create `EmbeddingConfig` dataclass
- [x] **2.4.4** Refactor `compute_embedding` to accept config + kwargs
- [x] **2.4.5** Create `MetricConfig` dataclass for `compute_pairwise_matrix`
- [x] **2.4.6** Write tests for config-object API alongside kwargs API
- [x] **2.5.1** Create `src/neural_analysis/utils/progress.py` with `get_progress_bar()`
- [x] **2.5.2** Replace all ad-hoc `tqdm` calls in `topology/` with `get_progress_bar()`
- [x] **2.5.3** Replace all ad-hoc `tqdm` calls in `metrics/` with `get_progress_bar()`
- [x] **2.5.4** Add `NEURAL_ANALYSIS_QUIET` env var support to suppress progress bars in CI
- [x] **2.6.1** Implement `LogConfig` dataclass with multi-file settings
- [x] **2.6.2** Implement `LogFileReference` class with `error_log()`, `debug_log()`, `session_dir()` methods
- [x] **2.6.3** Implement `get_log_dir()` function
- [x] **2.6.4** Implement `_make_session_id()` with UTC timestamp format
- [x] **2.6.5** Rewrite `configure_logging()` to create session directory with 4 log files (all, info, warnings, errors)
- [x] **2.6.6** Add `RotatingFileHandler` with configurable `max_bytes` and `backup_count`
- [x] **2.6.7** Add `logging.captureWarnings(True)` integration to route `warnings.warn()` to warning log
- [x] **2.6.8** Implement `_PrintCapture` class for optional `print()` redirection
- [x] **2.6.9** Ensure backward compatibility: existing `get_logger()`, `log_section()`, `log_kv()`, `@log_calls()` work unchanged
- [x] **2.6.10** Add `file_path` fallback mode for legacy single-file usage
- [x] **2.6.11** Update `do_critical()` in `validation.py` to include `LogFileReference` paths
- [x] **2.6.12** Audit all `raise` statements in `metrics/` and add `LogFileReference.debug_log()` to messages
- [x] **2.6.13** Audit all `raise` statements in `topology/` and add log references
- [x] **2.6.14** Audit all `raise` statements in `embeddings/` and `learning/` and add log references
- [x] **2.6.15** Audit all `raise` statements in `plotting/` and add log references
- [x] **2.6.16** Add `exc_info=True` to all `logger.error()` calls that precede a re-raise
- [x] **2.6.17** Search codebase for any remaining `print()` calls in `src/` and convert to `logger.info()`
- [x] **2.6.18** Write tests for multi-file log creation (verify files exist and contain expected levels)
- [x] **2.6.19** Write tests for `LogFileReference` returning correct paths
- [x] **2.6.20** Write tests for warning capture routing
- [x] **2.6.21** Write tests for print capture (opt-in mode)
- [x] **2.6.22** Write tests for log rotation (file size threshold)
- [x] **2.6.23** Write tests for backward compatibility (old `configure_logging(file_path=...)` still works)
- [x] **2.6.24** Update `docs/logging.md` with new multi-file system documentation
- [x] **2.6.25** Update `.github/instructions/python.instructions.md` logging guidance
- [x] **2.6.26** Add `logs/` to `.gitignore`

## Phase 3 — Reproducibility

- [x] **3.1.1** Create `src/neural_analysis/utils/reproducibility.py`
- [x] **3.1.2** Implement `reproducible()` context manager with numpy/random state save/restore
- [x] **3.1.3** Add `rng` parameter to `generate_data()` and all synthetic generators
- [x] **3.1.4** Add `random_state` parameter to `compute_embedding()` for all 7 methods
- [x] **3.1.5** Write tests verifying identical results with same seed
- [x] **3.1.6** Update example notebooks to use `reproducible()` context
- [x] **3.2.1** Verify `uv.lock` is committed and up-to-date
- [x] **3.2.2** Add `uv lock --check` step to CI if not already present
- [x] **3.2.3** Document lock file workflow in `CONTRIBUTING.md`
- [x] **3.3.1** Create `src/neural_analysis/utils/provenance.py`
- [x] **3.3.2** Implement `get_provenance()` returning library version, Python version, platform, timestamps, dep versions
- [x] **3.3.3** Inject provenance as `_provenance_*` attributes in `save_result_to_hdf5_dataset()`
- [x] **3.3.4** Inject provenance in `save_comparison()` calls
- [x] **3.3.5** Write tests verifying provenance attributes exist in saved HDF5 files
- [x] **3.4.1** Add a CI step to syntax-check all `examples/*_marimo_nb.py` files
- [x] **3.4.2** Start with `continue-on-error: true` (non-blocking)
- [x] **3.4.3** Fix any notebooks that fail the syntax check
- [x] **3.4.4** Once stable, remove `continue-on-error` to make notebook validation blocking

## Phase 4 — Testing

- [x] **4.1.1** Inventory all test files by source module (list which `test_*` files map to which `src/` module)
- [x] **4.1.2** Identify duplicate/overlapping test cases across `_additional`, `_comprehensive`, `_final`, `_more` files
- [x] **4.1.3** Merge `test_plotting*.py` files into a single `test_plotting.py` using test classes
- [x] **4.1.4** Merge `test_metrics*.py` files into a single `test_metrics.py` using test classes
- [x] **4.1.5** Merge `test_storage*.py` files into a single `test_storage.py` using test classes
- [x] **4.1.6** Merge other fragmented test files following the same pattern
- [x] **4.1.7** Verify test count and coverage are unchanged after consolidation
- [x] **4.1.8** Delete empty or fully-merged test files
- [x] **4.2.1** Identify embedding tests that can be parametrized across 7 methods
- [x] **4.2.2** Refactor with `@pytest.mark.parametrize("method", [...])` — 7 embedding methods (incl. UMAP with skipif)
- [x] **4.2.3** Identify plotting tests that can be parametrized across backends × plot types
- [x] **4.2.4** Refactor with `@pytest.mark.parametrize("backend", ["matplotlib", "plotly"])` — 6 backend values
- [x] **4.2.5** Identify metric tests that can be parametrized across metric types
- [x] **4.2.6** Refactor with `@pytest.mark.parametrize("metric", [...])` — similarity (5) and correlation (3) methods
- [x] **4.2.7** Identify classifier/clusterer tests that can be parametrized — 9 classifiers, 5 clusterers, 4 k-values
- [x] **4.2.8** Verify parametrized tests produce the same pass/fail results as the originals — 1615 passed, 22 skipped
- [x] **4.3.1** Create session-scoped fixtures in `tests/conftest.py` for common datasets
- [x] **4.3.2** Add `place_cells_2d` fixture (50 cells, 2000 timesteps)
- [x] **4.3.3** Add `random_activity` fixture (1000×50 matrix, seeded)
- [x] **4.3.4** Add `small_matrix` function-scoped fixture for quick unit tests
- [x] **4.3.5** Refactor tests to use shared fixtures instead of inline data generation
- [x] **4.3.6** Verify test performance improves (less redundant data generation)
- [x] **4.4.1** Run `uv run pytest --cov` and record current coverage percentage
- [x] **4.4.2** Set `--cov-fail-under=70` in `pyproject.toml` (or current value, whichever is lower)
- [x] **4.4.3** Verify CI fails when coverage drops below threshold
- [x] **4.4.4** Increase threshold by 5% increments as new tests are added

## Phase 5 — Documentation

- [x] **5.1.1** Add `--check` flag support to `scripts/generate_function_registry.py`
- [x] **5.1.2** Add a CI step that runs `generate_function_registry.py --check`
- [x] **5.1.3** Add `sphinx.ext.autodoc` and `sphinx.ext.napoleon` to `docs/conf.py`
- [x] **5.1.4** Add `autodoc_default_options` configuration
- [x] **5.1.5** Generate API reference RST stubs for each module
- [x] **5.1.6** Verify `uv run make html` in `docs/` builds without warnings
- [x] **5.2.1** Create `CHANGELOG.md` with the [Keep a Changelog](https://keepachangelog.com/) format
- [x] **5.2.2** Backfill `[0.1.0]` entry with current library features
- [x] **5.2.3** Add `[Unreleased]` section for ongoing work
- [x] **5.2.4** Document the changelog update workflow in `CONTRIBUTING.md`
- [x] **5.3.1** Create `.github/workflows/release.yml` with tag-triggered publishing
- [x] **5.3.2** Configure trusted publishing (OIDC) on PyPI for the repository
- [x] **5.3.3** Test with a `v0.1.0-rc1` pre-release tag to TestPyPI
- [x] **5.3.4** Document the release process (version bump → changelog → commit → tag → push)

## Phase 6 — Performance

- [x] **6.1.1** Create `scripts/benchmark.py` with timing harness
- [x] **6.1.2** Add benchmarks for pairwise euclidean distance (100×5000)
- [x] **6.1.3** Add benchmarks for Procrustes shape distance (100×5000)
- [x] **6.1.4** Add benchmarks for PCA embedding (100×5000 → 3D)
- [x] **6.1.5** Add benchmarks for structure index computation
- [x] **6.1.6** Create `benchmarks/` directory and save baseline results
- [x] **6.1.7** Document how to run and compare benchmarks in `CONTRIBUTING.md`
- [x] **6.2.1** Add `ttl` parameter to `StorageManager.save_data()` (already existed as `cache_ttl`)
- [x] **6.2.2** Set suggested TTL values: 300s (fast), 3600s (moderate), 86400s (expensive)
- [x] **6.2.3** Update structure index sweep to use 86400s TTL
- [x] **6.2.4** Update batch comparison functions to use appropriate TTL values
- [x] **6.2.5** Write tests for TTL override behavior

## Cross-Cutting Tasks

- [x] **X.1** Update `.gitignore` to include `logs/`, `benchmarks/latest.json`
- [x] **X.2** Update `docs/function_registry.md` after all API changes
- [x] **X.3** Update all 10 `.github/instructions/*.instructions.md` files to reflect new patterns
- [x] **X.4** Run final full CI: `uv run ruff check src tests --fix && uv run ruff format . && uv run mypy src tests && uv run pytest -v -n auto --cov`
- [x] **X.5** Update `TODO.md` with completed items and any new follow-ups discovered
- [x] **X.6** Update `CHANGELOG.md` with all changes made during this refactoring

**Total: ~150 individual tasks across 7 phases + cross-cutting.**
**Completed: All 150 tasks.**
