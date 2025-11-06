# Contributing to neural-analysis

Thank you for your interest in contributing! This document provides guidelines for setting up your development environment and contributing to the project.

## Project Goals

This repository provides an **automated pipeline for building and testing neural analysis methods**, following golden programming rules:

- **Reproducible environments** using Astral `uv` and a committed `uv.lock`
- **Automated quality gates** enforced in CI (linting, type-checking, testing)
- **Pre-commit hooks** to catch issues before they reach the repository
- **Clear documentation** for contributors and users

## Development Setup

### Prerequisites

- Python 3.10 or newer (recommended: 3.12)
- `uv` package manager (installation instructions below)
- Git

### Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd neural-analysis
   ```

2. **Run the setup script** (Ubuntu/WSL)
   ```bash
   chmod +x ./scripts/setup_env.sh
   ./scripts/setup_env.sh
   ```
   
   Answer `yes` when prompted to install development packages. The script will:
   - Install system build dependencies
   - Install or update `uv`
   - Generate/refresh `uv.lock`
   - Create a virtual environment (`.venv`)
   - Install dev dependencies (ruff, mypy, pytest, pre-commit, etc.)
   - Install pre-commit hooks

3. **Manual setup** (alternative)
   ```bash
   # Install uv (if not already installed)
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Install dependencies
   uv sync --locked --all-extras
   
   # Install pre-commit hooks
   uv run -- pre-commit install
   ```

## Development Workflow

### Running CI Locally (Recommended Before Pushing)

Before pushing changes, you can run the full CI pipeline locally using `act` to catch issues early:

```bash
# First time setup (install Docker and act)
# On Ubuntu/WSL:
sudo apt update && sudo apt install docker.io -y
sudo systemctl start docker
sudo systemctl enable docker

# Add your user to docker group (optional, to avoid sudo)
sudo usermod -aG docker $USER
newgrp docker

# Install act
curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash

# Run the local CI script
chmod +x ./scripts/run_ci_locally.sh
./scripts/run_ci_locally.sh
```

This will:
- Run the exact same checks as GitHub Actions
- Validate your changes before pushing
- Save CI minutes and catch issues faster

**Manual act usage:**
```bash
# Run specific workflow
act -W .github/workflows/ci.yml

# Run with verbose output
act -W .github/workflows/ci.yml -v

# Run specific job
act -j test
```

### Running Tests

```bash
# Run all tests
uv run -- pytest

# Run tests with verbose output
uv run -- pytest -v

# Run tests in parallel (faster)
uv run -- pytest -n auto

# Run with coverage
uv run -- pytest --cov=neural_analysis --cov-report=html
```

### Linting and Type Checking

```bash
# Run ruff linter
uv run -- ruff check src tests

# Auto-fix linting issues
uv run -- ruff check src tests --fix

# Run type checker
uv run -- mypy src tests

# Run all pre-commit hooks manually
uv run -- pre-commit run --all-files
```

### Pre-Commit Hooks

Pre-commit hooks run automatically before each commit. They will:
- Check and format code with `ruff`
- Run type checks with `mypy`
- Validate YAML files
- Fix trailing whitespace and end-of-file issues

If hooks fail, fix the reported issues and try committing again.

To skip hooks temporarily (not recommended):
```bash
git commit --no-verify
```

## Code Standards

### Style Guidelines

- **PEP 8**: Follow Python's style guide (enforced by `ruff`)
- **Type hints**: Use type annotations for all function signatures
- **Docstrings**: Document all public modules, classes, and functions
- **Line length**: Maximum 100 characters (configured in `pyproject.toml`)

### Testing Guidelines

- Write tests for all new features and bug fixes
- Aim for high test coverage (target: >80%)
- Use descriptive test names: `test_<what>_<condition>_<expected>`
- Place tests in `tests/` directory, mirroring `src/` structure

### Example Test

```python
from neural_analysis.example import mean

def test_mean_simple_list():
    """Test mean calculation with a simple list of numbers."""
    assert mean([1.0, 2.0, 3.0]) == 2.0

def test_mean_empty_raises():
    """Test that mean raises ValueError for empty sequences."""
    import pytest
    with pytest.raises(ValueError):
        mean([])
```

## Making a Pull Request

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write code following the style guidelines
   - Add or update tests
   - Update documentation if needed

3. **Run quality checks locally**
   ```bash
   uv run -- ruff check src tests
   uv run -- mypy src tests
   uv run -- pytest
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```
   
   Use conventional commit prefixes:
   - `feat:` for new features
   - `fix:` for bug fixes
   - `docs:` for documentation changes
   - `test:` for test additions/changes
   - `refactor:` for code refactoring
   - `chore:` for maintenance tasks

5. **Push and create a pull request**
   ```bash
   git push -u origin feature/your-feature-name
   ```
   
   Then open a PR on GitHub. The CI will automatically run all checks.

## Continuous Integration

All pull requests must pass CI checks:

- ✅ `uv.lock` is up-to-date
- ✅ Ruff linting passes
- ✅ Mypy type checking passes
- ✅ All tests pass
- ✅ Pre-commit hooks pass

CI runs on every push and pull request. Check the Actions tab for detailed results.

## Project Structure

```
neural-analysis/
├── src/neural_analysis/    # Main package source code
├── tests/                   # Unit and integration tests
├── docs/                    # Documentation
├── scripts/                 # Development and setup scripts
├── .github/workflows/       # CI/CD configuration
├── pyproject.toml          # Project metadata and dependencies
├── uv.lock                 # Locked dependency versions
└── README.md               # Project overview
```

## Getting Help

- Check the `docs/` directory for detailed documentation
- Review existing tests in `tests/` for examples
- Open an issue for bugs or feature requests
- Ask questions in pull request discussions

## License

By contributing to this project, you agree that your contributions will be licensed under the project's license (see `LICENSE` file).
