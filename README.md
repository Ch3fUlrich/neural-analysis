# neural-analysis

**An automated pipeline for building and testing neural analysis methods**, following golden programming rules for reproducibility and maintainability.

This repository uses Astral `uv` for reproducible Python environments, enforces quality gates through CI, and provides an automated development workflow.

## Features

### Unified Plotting System
- **Multi-backend support**: matplotlib and plotly with unified API
- **Plot types**: 1D (line, histogram, boolean states), 2D (scatter, trajectory, KDE, grouped scatter), 3D (scatter, trajectory), statistical (violin, box, bar), and heatmaps
- **Grid layouts**: Flexible subplot grids with customizable layouts
- **Rich configuration**: Colors, markers, labels, error bars, and styling options
- **100% test coverage**: 181/181 tests passing ✅

### Key Capabilities
- Error bar support for line plots
- **Reference lines**: Add horizontal/vertical lines with annotations to any plot
- Color gradients for trajectories
- Grouped scatter plots with convex hulls
- Interactive plotly visualizations
- Heatmaps with custom labels and value annotations
- Boolean state visualization with customizable regions

### Multi-Layer Storage & Benchmarks
- **HDF5** remains the ground-truth store for all analyses.
- **DuckDB** mirrors HDF5 metadata for instant SQL queries (`datasets`, `comparisons`, `chunks` tables).
- **Redis** caches hot comparison/structure-index rows with configurable TTL.
- **StorageManager** automatically cascades Redis ➜ DuckDB ➜ HDF5 with graceful degradation.
- `examples/storage_demo.ipynb` benchmarks write speed, reload speed, and disk usage for:
  - Pandas + HDF5
  - HDF5 + DuckDB metadata
  - HDF5 + Redis cache
  - Full stack (Redis + DuckDB + HDF5)

See [`docs/storage_stack.md`](docs/storage_stack.md) for architecture diagrams, configuration flags, and notebook walkthroughs.

## Quick Start

1. Install uv (see https://docs.astral.sh/uv/)

2. From project root:

```bash
# create a lockfile and install dependencies (creates `.venv`)
uv lock
uv sync

# run format/lint/tests via uv-run
uv run ruff
uv run mypy
uv run pytest -q
```


## Architecture & Capabilities

- **[Architecture Recommendations](docs/architecture.md)**: Detailed breakdown of the repository structure and guidelines for codebase modularity.
- **[Analysis Capabilities](docs/analysis_capabilities.md)**: Comprehensive guide on synthetic data generation, pairwise metrics, embeddings, decoding, topological analysis, and visualization dataflow within this repository.



## Usage

For complete usage examples including advanced plotting and configurations, see the **[Usage Examples Guide](docs/usage_examples.md)**.



## Development

For comprehensive instructions on setting up the environment, using the Makefile, Docker testing, and CI, please refer to the **[Development Guide](docs/development_guide.md)**.


## Notes

### Requirements

- Python >= 3.10 (recommended: 3.12)
- Git
- Dev tools (pytest, ruff, mypy, pre-commit, etc.) are in the `dev` optional dependency group
- CI enforces linting, type-checking, and testing on all PRs
- See `CONTRIBUTING.md` for detailed development guidelines

## Testing

**Status**: 100% test coverage (181/181 tests passing) ✅

The project has comprehensive test coverage across all plotting functionality:
- All plot types (1D, 2D, 3D, statistical, heatmaps)
- Both matplotlib and plotly backends
- Edge cases (single-point trajectories, invalid inputs)
- API parameters and configuration options
- Error handling and validation

Run tests:
```bash
make test          # Run all tests
make test-cov      # Run with coverage report
make test-fast     # Run in parallel (faster)
```

## Example Notebooks with Outputs

This repository includes example marimo notebooks demonstrating various neural analysis methods. To view notebooks with executed outputs:

**Location**: All exported HTML notebooks with outputs are saved in `examples/__marimo__/`

**Available notebooks**:
- `examples/__marimo__/metrics_examples_marimo_nb.html` - Metrics and distance calculations
- `examples/__marimo__/structure_index_examples_marimo_nb.html` - Structure index analysis
- `examples/__marimo__/neural_analysis_example_marimo_nb.html` - Comprehensive neural analysis
- `examples/__marimo__/synthetic_datasets_example_marimo_nb.html` - Synthetic data generation
- And more...

**To view**: Simply open any `.html` file in `examples/__marimo__/` in your web browser. These HTML files contain fully executed notebooks with all outputs, plots, and visualizations.

**To regenerate**: Export a marimo notebook to HTML with outputs:
```bash
uv run marimo export html examples/notebook.py -o examples/__marimo__/notebook.html
```

The `__marimo__/` directory is automatically created by marimo when exporting notebooks. This directory also contains session data and snapshots for interactive editing.
