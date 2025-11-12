# Testing, Linting, and CI/CD Guide

This document provides comprehensive guidance on testing, linting, type checking, and CI/CD for the neural-analysis package.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Testing](#testing)
3. [Linting and Formatting](#linting-and-formatting)
4. [Type Checking](#type-checking)
5. [Local CI with act](#local-ci-with-act)
6. [GitHub Actions CI/CD](#github-actions-cicd)
7. [Current CI Status](#current-ci-status)
8. [Development Workflow](#development-workflow)
9. [Troubleshooting](#troubleshooting)

---

## Quick Start

### 🚨 Critical Rule: CI Must Pass Before Pushing

**NEVER push code without passing local CI checks first.**

```bash
# ALWAYS run this before pushing
./scripts/run_ci_locally.sh

# Or run individual checks
uv run ruff check src tests --fix
uv run mypy src tests
uv run pytest -v
```

### Installation

```bash
# Using setup script (recommended)
INSTALL_DEV=1 ./scripts/setup_env.sh

# Or manual installation
uv sync --locked --all-extras
```

---

## Testing

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run tests in parallel (faster)
uv run pytest -n auto

# Run specific test file
uv run pytest tests/test_plots_1d.py

# Run specific test function
uv run pytest tests/test_plots_1d.py::test_plot_line

# Run tests matching pattern
uv run pytest -k "test_scatter"
```

### Test Coverage

```bash
# Run with coverage report
uv run pytest --cov

# Generate HTML coverage report
uv run pytest --cov --cov-report=html

# View HTML report
open htmlcov/index.html  # macOS/Linux
start htmlcov/index.html  # Windows
```

### Test Organization

```
tests/
├── test_decoding.py              # Decoding module tests
├── test_embeddings.py            # Embeddings tests
├── test_metrics_distance.py      # Distance metrics
├── test_metrics_distributions.py # Distribution metrics
├── test_metrics_outliers.py      # Outlier detection
├── test_plots_1d.py             # 1D plotting tests
├── test_plots_2d.py             # 2D plotting tests
├── test_plots_3d.py             # 3D plotting tests
├── test_plots_heatmaps_subplots.py  # Heatmaps/subplots
├── test_similarity.py           # Similarity metrics
├── test_synthetic_data.py       # Synthetic data generation
├── test_utils_io.py             # IO utilities
├── test_utils_io_h5io.py        # HDF5 IO
├── test_utils_preprocessing.py  # Preprocessing
└── test_utils_validation.py     # Validation utilities
```

### Test Statistics (Current)

- **Total Tests**: 204
- **Pass Rate**: 100% (all passing)
- **Coverage**: 53.09%
- **Execution Time**: ~15 seconds
- **Warnings**: 0

### Writing Tests

Follow pytest conventions:

```python
import numpy as np
import pytest
from neural_analysis.plotting import plot_line

def test_plot_line_basic():
    """Test basic line plotting."""
    data = np.sin(np.linspace(0, 2*np.pi, 100))
    fig = plot_line(data=data, backend='matplotlib')
    assert fig is not None

def test_plot_line_with_config():
    """Test line plotting with configuration."""
    data = np.sin(np.linspace(0, 2*np.pi, 100))
    config = PlotConfig(title='Sin Wave', xlabel='Time', ylabel='Value')
    fig = plot_line(data=data, config=config, backend='matplotlib')
    assert fig is not None

@pytest.mark.parametrize("backend", ['matplotlib', 'plotly'])
def test_plot_line_backends(backend):
    """Test line plotting with both backends."""
    data = np.sin(np.linspace(0, 2*np.pi, 100))
    fig = plot_line(data=data, backend=backend)
    assert fig is not None
```

---

## Linting and Formatting

### Ruff

Ruff is an ultra-fast Python linter and formatter that replaces flake8, black, isort, and more.

```bash
# Check code
uv run ruff check src tests

# Auto-fix issues
uv run ruff check src tests --fix

# Format code
uv run ruff format src tests

# Check specific file
uv run ruff check src/neural_analysis/plotting/grid_config.py
```

### Configuration

Ruff is configured in `pyproject.toml`:

```toml
[tool.ruff]
line-length = 88
target-version = "py314"

[tool.ruff.lint]
select = [
    "E",    # pycodestyle errors
    "W",    # pycodestyle warnings
    "F",    # pyflakes
    "I",    # isort
    "N",    # pep8-naming
    "UP",   # pyupgrade
    "B",    # flake8-bugbear
    "C4",   # flake8-comprehensions
]
ignore = [
    "E501",  # line too long (handled by formatter)
]

[tool.ruff.lint.per-file-ignores]
"__init__.py" = ["F401"]  # Allow unused imports in __init__.py
```

### Common Ruff Fixes

```bash
# Remove unused imports
uv run ruff check --select F401 --fix

# Fix import ordering
uv run ruff check --select I --fix

# Remove trailing whitespace
uv run ruff check --select W291,W293 --fix
```

---

## Type Checking

### Mypy

Mypy enforces and validates type hints.

```bash
# Check types
uv run mypy src tests

# Check specific file
uv run mypy src/neural_analysis/plotting/grid_config.py

# Verbose output
uv run mypy src tests --verbose

# Show error codes
uv run mypy src tests --show-error-codes
```

### Configuration

Mypy is configured in `pyproject.toml`:

```toml
[tool.mypy]
python_version = "3.14"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = false  # Set to true for strict mode
```

### Type Hint Examples

```python
from __future__ import annotations
from typing import Literal, Sequence
import numpy as np
import pandas as pd

def plot_scatter(
    data: np.ndarray | pd.DataFrame,
    backend: Literal['matplotlib', 'plotly'] = 'matplotlib',
    colors: Sequence[str] | None = None,
    marker_size: float = 10.0,
    alpha: float = 0.7,
) -> Figure:
    """
    Plot scatter plot.
    
    Args:
        data: Data array (n, 2) or DataFrame with x, y columns
        backend: Backend to use for plotting
        colors: Optional color sequence
        marker_size: Size of markers
        alpha: Transparency (0-1)
        
    Returns:
        Figure object (matplotlib or plotly)
    """
    ...
```

### Known Type Issues

Current mypy status: **686 errors across 31 files**
- **Impact**: Non-blocking, purely for type safety
- **Priority**: Low - can be addressed incrementally
- **Main sources**: Missing return type annotations in tests, some parameter type issues

---

## Local CI with act

### Why Run CI Locally?

- **Faster feedback**: Catch issues before pushing to GitHub
- **Save CI minutes**: Reduce GitHub Actions usage
- **Offline testing**: Test workflows without internet access
- **Debugging**: Easier to debug workflow issues

### Installation

```bash
# On Ubuntu/WSL
sudo apt update
sudo apt install docker.io -y
sudo systemctl start docker
sudo systemctl enable docker

# Install act
curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash

# Verify installation
docker --version
act --version
```

### Usage

```bash
# Using the provided script (recommended)
./scripts/run_ci_locally.sh

# Using act directly
act -W .github/workflows/ci.yml

# Verbose output
act -W .github/workflows/ci.yml -v

# Dry run
act -n

# Run specific job
act -j test
```

### Common Options

```bash
# Pull latest Docker images
act --pull

# Run without pulling images
act --pull=false

# Use specific platform
act --platform ubuntu-latest=catthehacker/ubuntu:act-latest

# Pass secrets
act --secret-file .secrets

# Pass environment variables
act --env VAR=value
```

### Troubleshooting act

**Docker Permission Denied:**
```bash
sudo usermod -aG docker $USER
newgrp docker
```

**act Not Found:**
```bash
# Ensure /usr/local/bin is in PATH
echo $PATH

# Or install to ~/.local/bin
curl https://raw.githubusercontent.com/nektos/act/master/install.sh | bash -s -- -b ~/.local/bin
```

**Workflow Fails Locally but Passes on GitHub:**
- Check Docker image compatibility
- Ensure all secrets/environment variables are set
- Some GitHub-specific features may not work identically locally

---

## GitHub Actions CI/CD

### Workflow Configuration

Located at `.github/workflows/ci.yml`:

```yaml
name: CI
on:
  push:
    branches: [ main, master ]
  pull_request:
    branches: [ main, master ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.12'  # Note: Using 3.12 for full compatibility
      
      - name: Install system dependencies
        run: sudo apt-get update && sudo apt-get install -y cmake
      
      - name: Install uv
        run: |
          curl -LsSf https://astral.sh/uv/install.sh | sh
          echo "$HOME/.local/bin" >> $GITHUB_PATH
      
      - name: Sync environment
        run: uv sync --locked --all-extras
      
      - name: Run ruff (lint)
        run: uv run ruff check src tests
      
      - name: Run mypy (type check)
        run: uv run mypy src tests || true  # Allow warnings
      
      - name: Run tests
        run: uv run pytest -v --cov
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          token: ${{ secrets.CODECOV_TOKEN }}
```

### CI Checks

The CI pipeline runs:
1. **Ruff**: Linting and formatting checks
2. **Mypy**: Type checking (warnings allowed)
3. **Pytest**: All tests with coverage
4. **Codecov**: Coverage upload (optional)

### Branch Protection

Main branch is protected with:
- ✅ Require passing CI checks before merge
- ✅ Require pull request reviews
- ✅ No direct pushes to main
- ✅ Up-to-date branches required

---

## Current CI Status

### ✅ Status: PASSING

Last Updated: 2025-01

#### Test Results
- **Tests Passed**: 204/205
- **Tests Skipped**: 1 (UMAP - optional dependency)
- **Coverage**: 53.09%
- **Python Version**: 3.12 (for full compatibility)
- **Execution Time**: ~15 seconds
- **Warnings**: 0

#### Type Checking
- **Status**: 686 mypy errors (non-blocking)
- **Impact**: Does not block CI, purely for type safety
- **Priority**: Low - addressing incrementally
- **Files Affected**: Mainly `plotting/` and `tests/`

#### Dependencies Status

All dependencies working on Python 3.12:

| Package | Version | Status | Notes |
|---------|---------|--------|-------|
| numpy | 2.2.1 | ✅ Working | |
| scipy | 1.14.1 | ✅ Working | |
| scikit-learn | 1.7.2 | ✅ Working | |
| matplotlib | 3.10.0 | ✅ Working | |
| plotly | 5.24.1 | ✅ Working | |
| pot | 0.9.6 | ✅ Working | Builds from source with cmake |
| numba | 0.62.1 | ✅ Working | Pre-built wheels available |
| umap-learn | 0.5.9 | ✅ Working | Pre-built wheels available |

### Known Issues

1. **numba/umap-learn**: Optional dependencies, may not be available on all platforms
2. **Type annotations**: 686 mypy errors (non-blocking, can be addressed incrementally)
3. **Coverage**: At 53%, goal is 90%+

---

## Development Workflow

### Core Tools

| Purpose | Tool | Version |
|---------|------|---------|
| Package Manager | uv | Latest |
| Testing | pytest + pytest-xdist | Latest |
| Coverage | pytest-cov | Latest |
| Type Checking | mypy | Latest |
| Linting | ruff | Latest |
| Local CI | act | Latest |
| CI/CD | GitHub Actions | N/A |

### Daily Workflow

1. **Write code** with type hints and docstrings
   ```python
   def process_data(values: list[float]) -> float:
       """Calculate mean of values.
       
       Args:
           values: List of numeric values
           
       Returns:
           Arithmetic mean
       """
       return sum(values) / len(values)
   ```

2. **Run quick checks** (optional, for fast feedback)
   ```bash
   uv run ruff check src tests --fix
   uv run mypy src tests
   uv run pytest -v
   ```

3. **Run local CI** (REQUIRED before pushing)
   ```bash
   ./scripts/run_ci_locally.sh
   ```

4. **Commit and push** (only after local CI passes)
   ```bash
   git add .
   git commit -m "feat: add new feature"
   git push origin feature-branch
   ```

5. **GitHub Actions runs** automatically
   - All checks must pass before merging
   - Branch protection enforces this

### Pre-Push Workflow

```bash
# Step 1: Lint and fix
uv run ruff check src tests --fix
uv run ruff format src tests

# Step 2: Type check
uv run mypy src tests

# Step 3: Test
uv run pytest -v -n auto --cov

# Step 4: Local CI (runs all above + more)
./scripts/run_ci_locally.sh

# Step 5: Push (only after all pass)
git push origin your-branch
```

### Fast Iteration

```bash
# Watch mode for tests
uv run ptw  # Requires pytest-watch

# Run tests on specific file
uv run pytest tests/test_plots_1d.py -v

# Run tests matching pattern
uv run pytest -k "scatter"

# Run with coverage for specific file
uv run pytest tests/test_plots_1d.py --cov=src/neural_analysis/plotting/plots_1d.py
```

---

## Troubleshooting

### Tests fail locally but pass in CI

```bash
# Ensure locked dependencies
uv sync --locked --all-extras

# Check Python version
python --version  # Should be 3.12+

# Clear cache
uv cache clean
uv sync --locked --all-extras
```

### Ruff or mypy not found

```bash
# Install dev dependencies
uv sync --locked --all-extras

# Verify installation
uv run ruff --version
uv run mypy --version
```

### Local CI fails to run

```bash
# Check Docker
docker ps

# Check act
act --version

# Run with verbose logging
./scripts/run_ci_locally.sh -v

# Check Docker permissions
sudo usermod -aG docker $USER
newgrp docker
```

### Import errors in tests

```bash
# Install in editable mode
uv pip install -e .

# Or sync with all extras
uv sync --locked --all-extras
```

### Coverage too low

```bash
# Generate HTML report to see what's missing
uv run pytest --cov --cov-report=html

# View report
open htmlcov/index.html

# Add tests for uncovered code
# See test_*.py files for examples
```

---

## Best Practices

### Testing

1. **Write tests for all new features**
2. **Test both backends** (matplotlib and plotly)
3. **Use parametrize** for testing multiple cases
4. **Test edge cases** (empty data, single point, NaN values)
5. **Keep tests fast** (use small datasets)

### Type Hints

1. **Add type hints to all functions**
2. **Use `from __future__ import annotations`** for forward references
3. **Use Literal types** for string enums
4. **Use `Sequence` instead of `list`** for read-only parameters
5. **Use `| None` instead of `Optional`** (modern syntax)

### Code Style

1. **Run `ruff check --fix`** before committing
2. **Keep lines under 88 characters** (ruff enforces this)
3. **Use docstrings** for all public functions (Google style)
4. **Import order**: stdlib, third-party, local
5. **Avoid wildcard imports** (`from module import *`)

### CI/CD

1. **Always run local CI before pushing**
2. **Never force push to main**
3. **Never bypass checks with `--no-verify`**
4. **Keep CI fast** (<5 minutes ideal)
5. **Monitor CI status** and fix failures immediately

---

## Summary

### Quick Commands

```bash
# Install
uv sync --locked --all-extras

# Lint
uv run ruff check src tests --fix

# Type check
uv run mypy src tests

# Test
uv run pytest -v

# Coverage
uv run pytest --cov

# Local CI
./scripts/run_ci_locally.sh
```

### Enforcement Rules

1. ✅ **Local CI must pass** before pushing
2. ✅ **GitHub Actions must pass** before merging
3. ✅ **Branch protection** prevents direct pushes to main
4. ✅ **All PRs require passing CI** checks
5. ✅ **No bypassing checks** with force push or `--no-verify`

### Success Metrics

- ✅ 204/205 tests passing (99.5%)
- ✅ 53% coverage (target: 90%)
- ✅ 0 pytest warnings
- ✅ All dependencies working
- ✅ Fast CI execution (~15s locally, ~3min on GitHub)

---

**The testing and CI system ensures continuous code quality without manual intervention. Always run local CI before pushing!**
