# Development Guide

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

## Multi-Service Docker Environment

The repository includes a Compose stack that bundles the Python app, Redis, and PostgreSQL (placeholder for future metadata use). This is the fastest way to get Redis-backed integration tests running without local installs.

```bash
./scripts/setup_docker.sh  # builds images and launches redis/db/app
./scripts/dev_docker.sh    # open a shell inside the Python container
```

Inside the container the default environment variables already point at the Compose services:

```
REDIS_HOST=redis
REDIS_PORT=6379
SQL_DB_PATH=/app/metadata.duckdb
```

Run benchmarks or tests exactly as you would on the host (for example `uv run pytest`, `uv run mypy`, or opening `examples/storage_demo.ipynb`).

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

## Setup Details

The script will attempt to install system build dependencies (via apt), install `uv` (multiple methods supported), create or refresh `uv.lock`, and then sync the project environment.

When you choose to install the development environment (or pass `INSTALL_DEV=1`), the script will install:
- Python development packages from the `dev` optional-dependencies group in `pyproject.toml` (pytest, mypy, ruff, pre-commit, etc.)
- **act** - tool for running GitHub Actions CI locally
- **Docker** - required by act (prompts for installation if not present)
- Pre-commit hooks (if inside a git repository)

If you skip the dev environment (`INSTALL_DEV=0`), only core dependencies will be installed.
