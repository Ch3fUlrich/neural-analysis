# Modern Python Testing & Linting Setup for neural-analysis

This guide describes the **fully automated, production-grade testing system** for this project. It ensures **early error detection**, **consistent style**, and **automatic verification** locally and in CI/CD.

## 🚨 Critical Rule: CI Must Pass Before Pushing

**NEVER push code to GitHub without running and passing local CI checks first.**

```bash
# ALWAYS run this before pushing
./scripts/run_ci_locally.sh

# Or run individual checks
uv run -- ruff check src tests
uv run -- mypy src tests  
uv run -- pytest
```

**Why?** This saves CI minutes, catches issues early, and maintains code quality standards.

## Core Tools in This Project

| Purpose | Tool | Version | Description |
|----------|------|---------|--------------|
| Package Manager | `uv` | Latest | Fast, reliable Python package management |
| Testing | `pytest` + `pytest-xdist` | Latest | Fast, parallelized testing framework |
| Coverage | `pytest-cov` + `coverage` | Latest | Measures test coverage |
| Type Checking | `mypy` | Latest | Enforces and validates type hints |
| Linting & Formatting | `ruff` | Latest | Ultra-fast linter + formatter (replaces flake8, black, isort) |
| Docstring Validation | `pydocstyle` | Latest | Ensures docstring presence and format |
| Pre-Commit Hooks | `pre-commit` | Latest | Runs checks before commits (local only) |
| Local CI Testing | `act` | Latest | Runs GitHub Actions locally with Docker |
| Continuous Integration | GitHub Actions | N/A | Runs all checks automatically on every push or PR |

## Installation (This Project)

**Using the provided setup script (recommended):**
```bash
# Install everything including dev tools
INSTALL_DEV=1 ./scripts/setup_env.sh

# Or interactive mode
./scripts/setup_env.sh  # Answer 'yes' for dev packages
```

**Manual installation:**
```bash
# Install uv if not present
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies (includes all dev tools)
uv sync --locked --all-extras

# Install pre-commit hooks (optional, for local use)
uv run -- pre-commit install
```

## Configuration Files (This Project)

### pyproject.toml
Our project configuration (see `pyproject.toml` in root):
```toml
[project]
name = "neural-analysis"
requires-python = ">=3.14"
dependencies = []

[project.optional-dependencies]
dev = [
    "pytest",
    "pytest-xdist",
    "pytest-cov",
    "coverage",
    "mypy",
    "ruff",
    "pydocstyle",
    "pre-commit",
    "flake8",
    "build",
]

[tool.ruff]
line-length = 88
target-version = "py314"
```

### .pre-commit-config.yaml
Pre-commit hooks for local development (see `.pre-commit-config.yaml` in root):
```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml

  - repo: https://github.com/pre-commit/mirrors-flake8
    rev: 6.1.0
    hooks:
      - id: flake8
```

### .github/workflows/ci.yml
GitHub Actions CI workflow (see `.github/workflows/ci.yml` in root):
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
          python-version: '3.14'
      - name: Install uv
        run: |
          curl -sSfL https://install.astral.sh | sh
          echo "$HOME/.local/bin" >> $GITHUB_PATH
      - name: Sync environment
        run: uv sync --locked --all-extras
      - name: Run ruff (lint)
        run: uv run -- ruff check .
      - name: Run mypy (type check)
        run: uv run -- mypy src tests
      - name: Run tests
        run: uv run -- pytest -q
```

## Recommended Workflow (This Project)

### 🚨 MANDATORY: Before Every Push

```bash
# Step 1: Run local CI (this runs ALL checks)
./scripts/run_ci_locally.sh

# Step 2: If local CI passes, push
git push origin your-branch
```

### Daily Development Workflow

1. **Write code** with full type hints and docstrings
   ```python
   def process_data(values: list[float]) -> float:
       """Calculate mean of values.
       
       Args:
           values: List of numeric values
           
       Returns:
           Arithmetic mean of input values
       """
       return sum(values) / len(values)
   ```

2. **Run quick checks locally** (optional, for fast feedback)
   ```bash
   # Quick linting
   uv run -- ruff check src tests
   
   # Type checking
   uv run -- mypy src tests
   
   # Run tests
   uv run -- pytest -v
   ```

3. **Run local CI before committing** (REQUIRED)
   ```bash
   ./scripts/run_ci_locally.sh
   ```
   This runs the exact same checks as GitHub Actions.

4. **Commit** (pre-commit hooks run automatically if installed)
   ```bash
   git add .
   git commit -m "feat: add new feature"
   ```

5. **Push** (only after local CI passes)
   ```bash
   git push origin your-branch
   ```

6. **GitHub Actions runs** automatically and verifies your changes
   - All checks must pass before merging to main
   - Branch protection enforces this requirement

### Fast Iteration Workflow

For quick feedback during development:
```bash
# Watch mode with pytest (install pytest-watch)
uv run -- ptw

# Quick ruff check on save (use editor integration)
# VS Code: Install "Ruff" extension
# Vim/Neovim: Use ALE or coc-ruff

# Type check on save (use editor integration)
# VS Code: Install "Pylance" 
```

## Benefits

- ✅ **Detects errors early** — before pushing to GitHub
- ✅ **Fast** — parallel testing and linting with uv
- ✅ **Consistent quality** — enforced style, types, and docs
- ✅ **Fully automated** — from local dev to CI/CD
- ✅ **Reproducible** — locked dependencies via uv.lock
- ✅ **Local CI testing** — catch issues before using GitHub Actions minutes
- ✅ **Enterprise-grade** — mirrors setups used by Google, Meta, Microsoft

## Summary

| Feature | Tool | Command |
|----------|------|---------|
| Package Management | `uv` | `uv sync --locked --all-extras` |
| Unit + Integration Tests | `pytest` | `uv run -- pytest` |
| Parallel Execution | `pytest-xdist` | `uv run -- pytest -n auto` |
| Test Coverage | `pytest-cov` | `uv run -- pytest --cov` |
| Type Checking | `mypy` | `uv run -- mypy src tests` |
| Linting + Formatting | `ruff` | `uv run -- ruff check .` |
| Docstring Rules | `pydocstyle` | Included in ruff |
| Auto-check on Commit | `pre-commit` | `uv run -- pre-commit run --all-files` |
| Local CI Testing | `act` | `./scripts/run_ci_locally.sh` |
| CI/CD Integration | GitHub Actions | Automatic on push/PR |

## Enforcement Rules

1. **Local CI must pass** before pushing
2. **GitHub Actions must pass** before merging to main
3. **Branch protection** prevents direct pushes to main
4. **All PRs require passing CI** checks
5. **No bypassing checks** with `--no-verify` or force push

## Troubleshooting

### Tests fail locally but pass in CI
```bash
# Ensure you're using the locked dependencies
uv sync --locked --all-extras

# Check Python version matches CI
python --version  # Should be 3.14+
```

### Ruff or mypy not found
```bash
# Install dev dependencies
uv sync --locked --all-extras

# Verify installation
uv run -- ruff --version
uv run -- mypy --version
```

### Local CI fails to run
```bash
# Check Docker is running
docker ps

# Check act is installed
act --version

# See detailed logs
./scripts/run_ci_locally.sh -v
```

### Pre-commit hooks not working
```bash
# Reinstall hooks
uv run -- pre-commit install

# Run manually to debug
uv run -- pre-commit run --all-files
```

Result: A fast, modern, fully automated testing system that continuously ensures correctness, readability, and maintainability without manual testing. **Always run local CI before pushing!**

