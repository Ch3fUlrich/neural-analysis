# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Migrated legacy `Manimeasure.py` embedding functions into `src/neural_analysis/embeddings/`
- Migrated legacy `Manimeasure.py` decoding functions into `src/neural_analysis/decoding/`
- Added `examples/decoding_demo_marimo_nb.py` for demonstrating decoding usage

### Fixed
- Fixed bug in decoding logic causing indexing out-of-bounds due to continuous label dimensional casting
- Resolved mock issues in `test_storage_redis.py` tests
- Fixed numerous linting issues across Marimo notebooks to pass `make check`

### Added
- `run_analysis()` pipeline for end-to-end generate → embed → decode → SI workflow
- `PipelineConfig` and `PipelineResult` dataclasses
- `StructureIndexConfig` dataclass for structured configuration
- `reproducible()` context manager for deterministic seed management
- `get_provenance()` for environment metadata capture
- `get_progress_bar()` standardized tqdm wrapper with NEURAL_ANALYSIS_QUIET support
- Enhanced multi-file logging system with session directories
- `LogFileReference` class for error message log path references
- `AnalysisResult`, `MetricResult`, `EmbeddingResult`, `DecodingResult` result dataclasses
- Renderer registry pattern replacing if/elif dispatch chains
- Convenience imports in top-level `neural_analysis` package
- Shared test fixtures in `tests/conftest.py`
- Coverage threshold (70%) enforced via `fail_under`
- `--check` flag for `generate_function_registry.py`
- Notebook validation step in CI
- CHANGELOG.md

### Changed
- CI triggers enabled for push/PR (previously manual-only)
- Quality gates (mypy, ruff format, lockfile) now blocking in CI
- Improved error messages across 10 modules following "what happened + expected + got" pattern
- Consolidated 68 test files down to 44 by merging `_additional`/`_comprehensive`/`_final`/`_more` suffixed files
- Split large source files: renderers, grid_config, synthetic_data, synthetic_plots, pairwise_metrics
- Removed dead code (`preprocessing.py`)
- Dockerfile rewritten to use UV instead of pip

### Fixed
- All mypy strict-mode errors resolved across src/ and tests/
- All ruff lint and format violations resolved

## [0.1.0] - 2025-01-01

### Added
- Modular architecture: data, metrics, embeddings, learning, topology, plotting, utils
- Synthetic data generators: place cells, grid cells, head direction cells, random cells, mixed populations
- 7 embedding methods: PCA, t-SNE, UMAP, MDS, Isomap, LLE, Spectral
- 9 supervised classifiers and 7 unsupervised clusterers
- Pairwise distance/similarity metrics with Numba acceleration
- Distribution comparison: Jensen-Shannon, Wasserstein, KL divergence, shape distances
- Structure Index computation with parameter sweeps
- PlotGrid visualization system (16 plot types, dual matplotlib/plotly backend)
- Three-layer storage stack (HDF5 + DuckDB + Redis)
- Structured logging with decorators and context managers
- 17 Marimo example notebooks
- Docker and docker-compose support
- GitHub Actions CI pipeline
