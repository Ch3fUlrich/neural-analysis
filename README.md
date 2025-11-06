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

## Usage

### Basic Plotting Examples

```python
import numpy as np
from neural_analysis.plotting import plot_line, plot_scatter_2d, plot_heatmap

# Line plot with error bars
x = np.linspace(0, 10, 100)
y = np.sin(x)
error_y = 0.1 * np.ones_like(y)
plot_line(x, y, error_y=error_y, backend="matplotlib")

# 2D scatter plot
x = np.random.randn(100)
y = np.random.randn(100)
plot_scatter_2d(x, y, backend="plotly")

# Heatmap with labels
data = np.random.rand(5, 5)
plot_heatmap(
    data,
    x_labels=["A", "B", "C", "D", "E"],
    y_labels=["1", "2", "3", "4", "5"],
    show_values=True,
    colorbar=True,
    backend="matplotlib"
)
```

### Backend Selection

Choose between matplotlib and plotly backends:

```python
# Static matplotlib plots (publication-ready)
plot_scatter_2d(x, y, backend="matplotlib")

# Interactive plotly plots (exploration)
plot_scatter_2d(x, y, backend="plotly")
```

### Advanced Features

```python
# Trajectory with color gradient
from neural_analysis.plotting import plot_trajectory_2d

trajectory = np.random.randn(100, 2)
plot_trajectory_2d(trajectory, color_by="time", backend="plotly")

# Grouped scatter with convex hulls
from neural_analysis.plotting import plot_grouped_scatter_2d

points = np.random.randn(100, 2)
labels = np.random.choice(["A", "B", "C"], 100)
plot_grouped_scatter_2d(points, labels, show_hulls=True, backend="matplotlib")

# Reference lines and annotations (PlotGrid)
from neural_analysis.plotting import PlotGrid, PlotSpec, PlotConfig

# Create line plot with threshold lines and annotations
x = np.linspace(0, 10, 100)
y = np.exp(-0.5 * x) + np.random.normal(0, 0.1, 100)

spec = PlotSpec(
    data={'x': x, 'y': y},
    plot_type='line',
    color='steelblue',
    line_width=2,
    # Horizontal threshold line
    hlines=[{'y': 0.5, 'color': 'red', 'linestyle': '--', 'label': 'Threshold'}],
    # Vertical marker line
    vlines=[{'x': 5.0, 'color': 'orange', 'linestyle': ':', 'label': 'Key point'}],
    # Text annotation with arrow
    annotations=[{
        'text': 'Important event',
        'xy': (5.0, 0.5),
        'xytext': (6.0, 0.8),
        'arrowprops': {'color': 'darkred'}
    }]
)

grid = PlotGrid(plot_specs=[spec], config=PlotConfig())
grid.plot()  # Works with both matplotlib and plotly backends!
```

## Development with Makefile

The project includes a **Makefile** with common development tasks for improved productivity:

```bash
make help          # Show all available commands
make install       # Install project dependencies
make install-dev   # Install with dev dependencies
make test          # Run tests
make test-cov      # Run tests with coverage report
make test-fast     # Run tests in parallel
make lint          # Check code with ruff
make lint-fix      # Auto-fix linting issues
make format        # Format code with ruff
make format-check  # Check if code is formatted
make type-check    # Run mypy type checking
make check         # Run all checks (lint, format, type, test)
make ci            # Run local CI with act
make clean         # Remove build artifacts and caches
make lock          # Update dependency lockfile
make sync          # Sync environment with lockfile
make update        # Update all dependencies
```

**Most useful commands:**
- `make check` - Run all quality checks before committing
- `make format` - Auto-format your code
- `make test-cov` - Run tests and see coverage report
- `make ci` - Test with local CI before pushing

## Notes

### Requirements

- Python >= 3.10 (recommended: 3.12)
- Git
- Dev tools (pytest, ruff, mypy, pre-commit, etc.) are in the `dev` optional dependency group
- CI enforces linting, type-checking, and testing on all PRs
- See `CONTRIBUTING.md` for detailed development guidelines

## Automatic Setup Script

This repository includes an **interactive setup script** to bootstrap your development environment on Ubuntu/WSL.

### Interactive Mode (Recommended)

```bash
# Simply run the script - it will guide you through the setup
./scripts/setup_env.sh
```

The script will ask you to choose:
- UV package manager installation method (installer/pipx/pip)
- Whether to install full development environment (Python packages, act, Docker)
- Whether to run validation checks after setup

### Non-Interactive Mode (For Automation)

Set environment variables to control the installation:

```bash
# Full development environment (Python packages + act + Docker)
INSTALL_DEV=1 ./scripts/setup_env.sh

# Minimal installation (no dev tools)
INSTALL_DEV=0 ./scripts/setup_env.sh

# Full dev setup with validation
INSTALL_DEV=1 RUN_TESTS=1 ./scripts/setup_env.sh
```

See [docs/setup_script_usage.md](docs/setup_script_usage.md) for complete documentation.

## Local CI Testing (Optional but Recommended)

Test your changes locally before pushing using `act` to run GitHub Actions workflows:

```bash
# Using Makefile (recommended)
make ci

# Or run the local CI script (installs Docker and act if needed)
./scripts/run_ci_locally.sh

# Or use act directly
act -W .github/workflows/ci.yml
```

This runs the exact same checks as GitHub Actions, helping you catch issues early.

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

## Setup Details

The script will attempt to install system build dependencies (via apt), install `uv` (multiple methods supported), create or refresh `uv.lock`, and then sync the project environment.

When you choose to install the development environment (or pass `INSTALL_DEV=1`), the script will install:
- Python development packages from the `dev` optional-dependencies group in `pyproject.toml` (pytest, mypy, ruff, pre-commit, etc.)
- **act** - tool for running GitHub Actions CI locally
- **Docker** - required by act (prompts for installation if not present)
- Pre-commit hooks (if inside a git repository)

If you skip the dev environment (`INSTALL_DEV=0`), only core dependencies will be installed.
