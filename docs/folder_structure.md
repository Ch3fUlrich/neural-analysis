# Repository Structure

## Current Structure

```
neural-analysis/
│
├── .github/                        # GitHub configuration
│   └── workflows/
│       └── ci.yml                  # GitHub Actions CI/CD pipeline
│
├── docs/                           # Comprehensive documentation
│   ├── DOCUMENTATION_CLEANUP_SUMMARY.md  # Documentation consolidation record
│   ├── folder_structure.md         # This file - Repository structure
│   ├── project_goal.md             # Project mission and goals
│   ├── testing_and_ci.md           # Testing, linting, CI/CD guide (consolidated)
│   ├── setup_script_usage.md       # Setup script comprehensive guide
│   ├── logging.md                  # Logging configuration guide
│   │
│   ├── plotgrid.md                 # Comprehensive PlotGrid system guide (consolidated)
│   ├── plotting_architecture.md    # Plotting module architecture and design patterns
│   ├── historical_migrations.md    # Historical migration reference (consolidated)
│   │
│   ├── decoding_module.md          # Decoding functions documentation
│   ├── distributions.md            # Distribution comparison module
│   ├── hdf5_structure.md          # HDF5 file structure reference
│   ├── structure_index.md          # Structure index documentation
│   ├── synthetic_datasets_notebook.md  # Synthetic datasets guide
│   ├── function_registry.md        # Function registry documentation (auto-generated)
│   │
│   ├── _build/                     # Sphinx build output
│   ├── api/                        # API documentation
│   ├── conf.py                     # Sphinx configuration
│   ├── index.rst                   # Sphinx index
│   ├── contributing.rst            # Contributing guide (RST)
│   ├── examples.rst                # Examples documentation
│   ├── installation.rst            # Installation guide
│   ├── quickstart.rst              # Quick start guide
│   ├── requirements.txt            # Docs dependencies
│   └── Makefile                    # Documentation build automation
│
├── examples/                       # Jupyter notebooks for demos and tutorials
│   ├── embeddings_demo.ipynb       # Embeddings and dimensionality reduction
│   ├── io_h5io_examples.ipynb      # HDF5 I/O operations
│   ├── logging_examples.ipynb      # Logging system usage
│   ├── metrics_examples.ipynb      # Metrics and distance calculations
│   ├── neural_analysis_demo.ipynb  # General neural analysis demo
│   ├── plots_1d_examples.ipynb     # 1D plotting examples
│   ├── plots_2d_examples.ipynb     # 2D plotting examples
│   ├── plots_3d_examples.ipynb     # 3D plotting examples
│   ├── plotting_grid_showcase.ipynb # PlotGrid system showcase
│   ├── statistical_plots_examples.ipynb # Statistical plotting
│   ├── structure_index_examples.ipynb # Structure indexing
│   ├── synthetic_datasets_example.ipynb # Synthetic data generation
│   ├── random_cells_diagnostics_example.py # Random cells diagnostics
│   └── output/                     # Notebook outputs
│
├── scripts/                        # Automation and utility scripts
│   ├── setup_env.sh                # Environment bootstrap script
│   ├── run_ci_locally.sh           # Local CI runner with act
│   ├── generate_function_registry.py # Function registry generator
│   └── execute_notebooks.py        # Notebook execution automation
│
├── src/                            # Main package source code
│   └── neural_analysis/
│       ├── __init__.py             # Package initialization
│       │
│       ├── data/                   # Data generation and management
│       │   ├── __init__.py
│       │   └── synthetic_data.py   # Synthetic dataset generation
│       │
│       ├── decoding.py             # Neural decoding algorithms (LEGACY - use learning/)
│       │
│       ├── embeddings/             # Dimensionality reduction and embeddings
│       │   ├── __init__.py
│       │   ├── dimensionality_reduction.py  # PCA, UMAP, t-SNE, etc.
│       │   └── visualization.py    # Embedding visualization helpers
│       │
│       ├── learning/               # Machine learning and decoding
│       │   ├── __init__.py
│       │   └── decoding.py         # Neural decoding models
│       │
│       ├── metrics/                # Quantitative analysis metrics
│       │   ├── __init__.py
│       │   ├── distance.py         # Distance metrics (Euclidean, Mahalanobis, etc.)
│       │   ├── distributions.py    # Distribution comparison (KS, Anderson-Darling)
│       │   ├── outliers.py         # Outlier detection methods
│       │   └── similarity.py       # Similarity measures
│       │
│       ├── plotting/               # Modular plotting system (backend-agnostic)
│       │   ├── __init__.py
│       │   ├── backend.py          # Backend selection (matplotlib/plotly)
│       │   ├── core.py             # Core plotting utilities
│       │   ├── grid_config.py      # PlotGrid system (metadata-driven layouts)
│       │   ├── renderers.py        # Low-level rendering primitives
│       │   ├── plots_1d.py         # 1D plots (line, bar, histogram, etc.)
│       │   ├── plots_2d.py         # 2D plots (scatter, density, contour, etc.)
│       │   ├── plots_3d.py         # 3D plots (surface, scatter, trajectory, etc.)
│       │   ├── heatmaps.py         # Heatmap visualizations
│       │   ├── statistical_plots.py # Statistical plots (violin, box, swarm, etc.)
│       │   ├── synthetic_plots.py  # Synthetic data visualization
│       │   └── embeddings.py       # Embedding-specific plots
│       │
│       ├── topology/               # Topological analysis
│       │   ├── __init__.py
│       │   ├── structure_index.py  # Structure index calculations
│       │   └── plotting.py         # Topology visualization
│       │
│       └── utils/                  # General utility functions
│           ├── __init__.py
│           ├── geometry.py         # Geometric calculations
│           ├── io.py               # File I/O operations (HDF5, etc.)
│           ├── logging.py          # Logging configuration and utilities
│           ├── preprocessing.py    # Signal processing and preprocessing
│           ├── trajectories.py     # Trajectory analysis utilities
│           └── validation.py       # Input validation and type checking
│
├── tests/                          # Unit and integration tests (mirrors src/)
│   ├── test_decoding.py            # Decoding tests
│   ├── test_embeddings.py          # Embeddings tests
│   ├── test_metrics_distance.py    # Distance metrics tests
│   ├── test_metrics_distributions.py # Distribution tests
│   ├── test_metrics_outliers.py    # Outlier detection tests
│   ├── test_similarity.py          # Similarity measures tests
│   ├── test_plots_1d.py            # 1D plotting tests
│   ├── test_plots_2d.py            # 2D plotting tests
│   ├── test_plots_3d.py            # 3D plotting tests
│   ├── test_plots_heatmaps_subplots.py # Heatmap tests
│   ├── test_structure_index.py     # Structure index tests
│   ├── test_synthetic_data.py      # Synthetic data tests
│   ├── test_utils_io.py            # I/O utilities tests
│   ├── test_utils_io_h5io.py       # HDF5 I/O tests
│   ├── test_utils_preprocessing.py # Preprocessing tests
│   ├── test_utils_validation.py    # Validation tests
│   └── test_placeholder.py         # Placeholder test
│
├── todo/                           # Work in progress and legacy code
│   ├── Helper.py                   # Legacy helper utilities
│   ├── Manimeasure.py              # Legacy measurement utilities
│   ├── Visualizer.py               # LEGACY - Migrated to modular plotting/
│   ├── restructure.py              # Restructuring utilities
│   ├── structure_index.py          # Legacy structure index
│   ├── yaml_creator.py             # YAML configuration creator
│   └── Notebooks/                  # Work-in-progress notebooks
│
├── .gitignore                      # Git ignore patterns
├── .pre-commit-config.yaml         # Pre-commit hooks (ruff, mypy)
├── .python-version                 # Python version (3.14)
├── .editorconfig                   # Editor configuration
├── CONTRIBUTING.md                 # Contribution guidelines
├── LICENSE                         # MIT License
├── Makefile                        # Build automation
├── PYTHON_312_MIGRATION.md         # Python 3.12 migration notes
├── README.md                       # Project overview and quick start
├── TODO.md                         # Project todo list
├── pyproject.toml                  # Project metadata, dependencies, tool configs
└── uv.lock                         # Locked dependency versions (DO NOT EDIT)

```

## Module Organization Philosophy

The repository follows a **modular, layered architecture** designed for maintainability, testability, and reusability:

### Core Principles

1. **Separation of Concerns**: Each module has a clear, focused purpose
2. **DRY (Don't Repeat Yourself)**: Check `docs/function_registry.md` before writing new code
3. **Backend-Agnostic**: Plotting system works with both matplotlib and plotly
4. **Type-Safe**: Extensive type hints and mypy validation
5. **Tested**: High test coverage with pytest (204/205 tests passing)

### Module Hierarchy

```
utils/          → Core utilities (I/O, logging, validation, preprocessing)
    ↓
data/           → Data generation (synthetic datasets)
    ↓
metrics/        → Quantitative analysis (distances, distributions, outliers)
embeddings/     → Dimensionality reduction (PCA, UMAP, t-SNE)
topology/       → Topological analysis (structure index)
learning/       → Machine learning (decoding models)
    ↓
plotting/       → Visualization (PlotGrid, 1D/2D/3D, statistical)
```

### Key Design Decisions

**Plotting System**:
- **Layer 1**: `renderers.py` - Low-level primitives (scatter, line, bar, etc.)
- **Layer 2**: `grid_config.py` - PlotGrid system (metadata-driven multi-panel layouts)
- **Layer 3**: `plots_1d.py`, `plots_2d.py`, `plots_3d.py`, `statistical_plots.py` - High-level plotting functions
- **Backend**: `backend.py` - Automatic matplotlib ↔ plotly switching

**Legacy Code**:
- `decoding.py` in root → Moving to `learning/decoding.py`
- `Visualizer.py` in `todo/` → Fully migrated to modular `plotting/` system
- See `docs/historical_migrations.md` for migration history

**Documentation**:
- **Consolidated**: PlotGrid (5→1), Migrations (3→1), Testing/CI (3→1)
- **Module-Specific**: Each major module has dedicated docs
- See `docs/DOCUMENTATION_CLEANUP_SUMMARY.md` for cleanup details

## 🚨 Critical Workflow Rules

### NEVER Push Without CI Passing

**Before every push to GitHub:**
```bash
# MANDATORY - Run local CI
./scripts/run_ci_locally.sh

# If act/Docker not available, run checks manually:
uv run -- ruff check src tests
uv run -- mypy src tests
uv run -- pytest -v
```

**Why?**
- Catches issues before they reach GitHub
- Saves CI minutes
- Maintains code quality standards
- Prevents broken builds on main branch

### Branch Protection

- ✅ Main branch is protected
- ✅ Direct pushes to main are blocked
- ✅ All changes must go through pull requests
- ✅ CI must pass before merging
- ✅ Use feature branches: `feat/`, `fix/`, `chore/`

### Workflow Steps

1. **Create feature branch**
   ```bash
   git checkout -b feat/your-feature
   ```

2. **Make changes and test locally**
   ```bash
   # Make code changes
   # Add tests
   # Run local CI
   ./scripts/run_ci_locally.sh
   ```

3. **Commit changes**
   ```bash
   git add .
   git commit -m "feat: description"
   ```

4. **Push to feature branch**
   ```bash
   git push origin feat/your-feature
   ```

5. **Create pull request**
   - GitHub Actions will run automatically
   - All checks must pass
   - Review and merge when approved

## Quick Navigation Guide

### For New Users

**Getting Started**:
1. Read `README.md` - Project overview and quick start
2. Read `docs/installation.rst` - Installation instructions
3. Read `docs/quickstart.rst` - Quick start guide
4. Explore `examples/` - Jupyter notebooks with interactive demos

**Learning the System**:
- **Plotting**: `docs/plotgrid.md` - Comprehensive PlotGrid guide with 7 examples
- **Logging**: `docs/logging.md` - Logging system usage
- **Testing**: `docs/testing_and_ci.md` - How to run tests and CI
- **Function Registry**: `docs/function_registry.md` - Available functions

### For Contributors

**Development Setup**:
1. Run `scripts/setup_env.sh` - Bootstrap environment
2. Read `CONTRIBUTING.md` - Contribution guidelines
3. Read `docs/testing_and_ci.md` - Testing and CI workflow
4. Check `docs/function_registry.md` - Avoid duplicating existing code

**Architecture Documentation**:
- `docs/folder_structure.md` (this file) - Repository structure
- `docs/plotting_architecture.md` - Plotting system design patterns
- `docs/historical_migrations.md` - Past refactorings and lessons learned

**Before Adding Code**:
1. **Check registry**: `python scripts/generate_function_registry.py` (updates registry)
2. **Search for similar functions**: Check `docs/function_registry.md`
3. **Reuse existing code**: DRY principle
4. **Add tests**: Mirror `src/` structure in `tests/`
5. **Update docs**: Add docstrings and update relevant docs

### For Specific Tasks

| Task | Where to Look |
|------|---------------|
| Create plots | `src/neural_analysis/plotting/` + `docs/plotgrid.md` |
| Generate synthetic data | `src/neural_analysis/data/synthetic_data.py` + `examples/synthetic_datasets_example.ipynb` |
| Calculate distances | `src/neural_analysis/metrics/distance.py` + `examples/metrics_examples.ipynb` |
| Compare distributions | `src/neural_analysis/metrics/distributions.py` + `docs/distributions.md` |
| Dimensionality reduction | `src/neural_analysis/embeddings/` + `examples/embeddings_demo.ipynb` |
| Neural decoding | `src/neural_analysis/learning/decoding.py` + `docs/decoding_module.md` |
| Structure index | `src/neural_analysis/topology/structure_index.py` + `docs/structure_index.md` |
| Read/write HDF5 | `src/neural_analysis/utils/io.py` + `docs/hdf5_structure.md` |
| Configure logging | `src/neural_analysis/utils/logging.py` + `docs/logging.md` |
| Preprocess signals | `src/neural_analysis/utils/preprocessing.py` |

### Documentation Map

| Category | Files | Purpose |
|----------|-------|---------|
| **Getting Started** | README.md, docs/quickstart.rst, docs/installation.rst | Quick start and setup |
| **Plotting** | docs/plotgrid.md, docs/plotting_architecture.md | Comprehensive plotting guide |
| **Testing/CI** | docs/testing_and_ci.md | Testing, linting, CI/CD |
| **Module Docs** | docs/decoding_module.md, docs/distributions.md, docs/structure_index.md | Module-specific documentation |
| **Data Formats** | docs/hdf5_structure.md, docs/synthetic_datasets_notebook.md | Data structure reference |
| **Development** | CONTRIBUTING.md, docs/function_registry.md, docs/setup_script_usage.md | Development workflow |
| **History** | docs/historical_migrations.md, docs/DOCUMENTATION_CLEANUP_SUMMARY.md | Project history and refactorings |
| **Utilities** | docs/logging.md | Logging and utilities |

## Related Documentation

- **Project Goals**: See `docs/project_goal.md` for detailed project vision and objectives
- **Contributing**: See `CONTRIBUTING.md` for contribution guidelines and best practices
- **Function Registry**: See `docs/function_registry.md` for complete function catalog
- **Setup Guide**: See `docs/setup_script_usage.md` for environment setup details
- **Migration History**: See `docs/historical_migrations.md` for past refactorings

---

**Last Updated**: January 2025 (Documentation Cleanup)  
**Status**: ✅ Comprehensive documentation with 14 focused files (down from 30+)
