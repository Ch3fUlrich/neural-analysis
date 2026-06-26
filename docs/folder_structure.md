# Repository Structure

## Current Structure

```
neural-analysis/

 .github/
    instructions/               # Agent/IDE coding instructions (11 files)
    workflows/
        ci.yml                  # CI pipeline (ruff, mypy, pytest, notebooks)
        release.yml             # Tag-triggered PyPI publishing

 docs/                           # Documentation
    folder_structure.md         # This file
    function_registry.md        # Auto-generated function registry
    plotgrid.md                 # PlotGrid system guide
    storage.md                  # Storage stack (HDF5, DuckDB, Redis)
    hdf5_structure.md           # HDF5 file structure reference
    logging.md                  # Logging configuration guide
    testing_and_ci.md           # Testing, CI/CD guide
    distributions.md            # Distribution comparison module
    structure_index.md          # Structure index documentation
    synthetic_datasets_notebook.md  # Synthetic datasets guide
    marimo_guide.md             # Marimo notebook usage guide
    setup_script_usage.md       # Setup script guide
    conf.py                     # Sphinx configuration
    index.rst                   # Sphinx index
    api/                        # API reference stubs
    _build/                     # Sphinx build output
    legacy/                     # Historical/archived docs
        PHASE4_PROGRESS.md
        historical_migrations.md
        plotting_architecture.md
        jupyter_to_marimo_conversion.md
        long_plotgrid.md
        storage_best_practices.md
        storage_stack.md

 examples/                       # Marimo notebooks (interactive demos)
    *_marimo_nb.py              # 16 marimo notebooks
    __marimo__/                 # Marimo HTML exports and session data
    output/                     # Notebook output files (HDF5)

 legacy/                         # Archived notebooks and scripts
    *.ipynb                     # 16 legacy Jupyter notebooks
    *.py                        # Legacy scripts

 scripts/
    benchmark.py                # Performance benchmarking suite
    generate_function_registry.py  # Function registry generator (--check flag)
    run_ci_locally.sh           # Local CI via act + Docker
    setup_env.sh                # Environment bootstrap
    convert_jupyter_to_marimo.py   # Jupyter  Marimo converter
    execute_notebooks.py        # Notebook execution automation

 benchmarks/                     # Benchmark result files

 src/neural_analysis/
    __init__.py                 # Convenience imports for top-level API
    pipeline.py                 # run_analysis() end-to-end pipeline
   
    core/
       __init__.py
       results.py              # AnalysisResult, MetricResult, EmbeddingResult, DecodingResult
   
    data/
       __init__.py
       synthetic_data.py       # Facade re-exporting from generators/trajectories/datasets
       generators.py           # Cell-type generators (place, grid, HD, random)
       trajectories_gen.py     # Trajectory generation (1D/2D/3D)
       datasets.py             # generate_data() dispatcher + dataset builders
   
    embeddings/
       __init__.py
       dimensionality_reduction.py  # compute_embedding()  7 methods
       visualization.py        # Embedding visualization helpers
   
    learning/
       __init__.py
       classification.py       # 9 supervised + 7 unsupervised classifiers
       cross_validation.py     # k-fold cross-validation
       decoders.py             # Core decoding algorithms
       decoding.py             # Neural decoders (PV, k-NN, cross-validated)
       evaluation.py           # Decoder evaluation metrics
   
    metrics/
       __init__.py
       pairwise_metrics.py     # Facade re-exporting from core + numba
       pairwise_core.py        # Core pairwise dispatcher + similarity matrices
       pairwise_numba.py       # Numba-accelerated parallel implementations
       distributions.py        # Shape distances, distribution metrics, batch processing
       outliers.py             # 5 outlier detection methods
   
    plotting/
       __init__.py
       core.py                 # PlotConfig, BackendType, backend utilities
       grid_config.py          # PlotSpec, PlotGrid, GridLayoutConfig dataclasses
       grid_dispatch.py        # Renderer registries + PlotGrid dispatch logic
       renderers.py            # Facade re-exporting from both renderer files
       renderers_matplotlib.py # Matplotlib render functions (16 plot types)
       renderers_plotly.py     # Plotly render functions (16 plot types)
       plots_1d.py             # 1D plots (line, bar, histogram, boolean)
       plots_2d.py             # 2D plots (scatter, trajectory, KDE, contour)
       plots_3d.py             # 3D plots (scatter, surface, trajectory)
       heatmaps.py             # Heatmap visualizations
       statistical_plots.py    # Violin, box, grouped distributions
       synthetic_plots.py      # Facade re-exporting from _1d/_2d/_3d
       synthetic_plots_1d.py   # 1D synthetic data visualizations
       synthetic_plots_2d.py   # 2D synthetic data visualizations
       synthetic_plots_3d.py   # 3D synthetic data visualizations
       shape_distance.py       # MDS/shape distance visualizations
       embeddings.py           # Embedding scatter plots
   
    topology/
       __init__.py
       structure_index.py      # compute_structure_index, parameter sweeps
       plotting.py             # SI visualization (scatter, heatmap, graph)
   
    utils/
        __init__.py
        io.py                   # File I/O (HDF5 helpers, h5io)
        logging.py              # Logging (LogConfig, LogFileReference, multi-file)
        validation.py           # Input validation, do_critical()
        progress.py             # get_progress_bar() (replaces ad-hoc tqdm)
        reproducibility.py      # reproducible() context manager, seed utilities
        provenance.py           # get_provenance() for versioning metadata
        comparison_store.py     # HDF5-backed comparison caching
        subsampling.py          # Subsampling utilities for large datasets
        geometry.py             # Geometric calculations
        trajectories.py         # Trajectory analysis utilities
        storage/                # Three-layer storage stack
            __init__.py
            config.py           # StorageConfig
            manager.py          # StorageManager orchestrator
            hdf5_backend.py     # HDF5 backend
            sql_backend.py      # DuckDB backend
            redis_backend.py    # Redis backend

 tests/                          # 44 test files, ~1600 tests
    conftest.py                 # Shared fixtures (place_cells_2d, random_activity, etc.)
    test_*.py                   # One primary file per source module
    ...

 CHANGELOG.md                    # Release notes (Keep a Changelog format)
 CONTRIBUTING.md                 # Contribution guidelines
 TODO.md                         # Task tracking
 pyproject.toml                  # Project metadata, dependencies, tool configs
 uv.lock                         # Locked dependencies (DO NOT EDIT)
 Dockerfile                      # Docker image (UV-based)
 docker-compose.yml              # Docker Compose (app + redis)
 Makefile                        # Build automation
 README.md                       # Project overview
```

## Module Hierarchy

Lower layers never import from upper layers:

```
utils  data  metrics  embeddings  learning  topology  plotting
                                                      
                                               core/results.py
                                               pipeline.py
```

## Key Design Patterns

- **Facade modules**: `synthetic_data.py`, `pairwise_metrics.py`, `renderers.py`, `synthetic_plots.py` re-export from split implementation files for backward compatibility.
- **Renderer registry**: `MATPLOTLIB_RENDERERS` and `PLOTLY_RENDERERS` dicts in `grid_dispatch.py` replace if/elif dispatch chains.
- **Config dataclasses**: `PipelineConfig`, `MetricConfig`, `EmbeddingConfig`, `StructureIndexConfig` group parameters.
- **Result dataclasses**: `AnalysisResult`, `MetricResult`, `EmbeddingResult`, `DecodingResult` provide structured returns.
- **Convenience imports**: Top-level `neural_analysis` package exports ~30 commonly used functions.
- **Reproducibility**: `rng` parameter convention, `reproducible(seed)` context manager.
- **Provenance**: `get_provenance()` attaches version/platform metadata to saved results.
- **Progress**: `get_progress_bar()` replaces ad-hoc `tqdm` calls.

## Quick Navigation

| Task | Where to Look |
|------|---------------|
| Generate data | `from neural_analysis import generate_data` |
| Compute distances | `from neural_analysis import compute_pairwise_matrix` |
| Shape distances | `from neural_analysis import shape_distance` |
| Embeddings | `from neural_analysis import compute_embedding` |
| Decoding | `from neural_analysis import knn_decoder, population_vector_decoder` |
| Classification | `from neural_analysis import run_classifier` |
| Structure index | `from neural_analysis import compute_structure_index` |
| Full pipeline | `from neural_analysis import run_analysis, PipelineConfig` |
| Plotting | `from neural_analysis.plotting import PlotGrid, PlotSpec, plot_bar, plot_violin` |
| Storage | `from neural_analysis.utils.storage import StorageManager` |
| Logging | `from neural_analysis.utils import configure_logging, get_logger` |

---

**Last Updated**: June 2025
