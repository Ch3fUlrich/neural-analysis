# neural-analysis

**An automated pipeline for building and testing neural analysis methods**, following golden programming rules for reproducibility and maintainability.

This repository uses Astral `uv` for reproducible Python environments, enforces quality gates through CI, and provides an automated development workflow.

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

## Notes

- Python >= 3.14 required
- Dev tools (pytest, ruff, mypy, pre-commit, etc.) are in the `dev` optional dependency group
- CI enforces linting, type-checking, and testing on all PRs
- See `CONTRIBUTING.md` for detailed development guidelines

## Automatic Setup Script

This repository includes a convenience script to bootstrap a development environment on Ubuntu/WSL.

```bash
# from the project root
chmod +x ./scripts/setup_env.sh
# Interactive: you'll be asked whether to install development packages (dev group)
./scripts/setup_env.sh

# Non-interactive examples:
# Install uv via pipx, install dev packages and run tests
UV_INSTALL_METHOD=pipx INSTALL_DEV=1 RUN_TESTS=1 ./scripts/setup_env.sh

# Use pip installer, skip dev packages
UV_INSTALL_METHOD=pip INSTALL_DEV=0 ./scripts/setup_env.sh

# Force regeneration of uv.lock
UV_FORCE_LOCK=1 ./scripts/setup_env.sh
```

One-liner: Run `./scripts/setup_env.sh` (answer yes to install dev extras) to bootstrap the uv-managed environment and install pre-commit hooks in a normal git clone.

The script will attempt to install system build dependencies (via apt), install `uv` (multiple methods supported), create or refresh `uv.lock`, and then sync the project environment.

The script asks whether to install the development dependency group (the `dev` optional-dependencies in `pyproject.toml`). If you choose to install dev packages (or pass `INSTALL_DEV=1`), the script will install the dev group and then install pre-commit hooks. If you skip dev packages, the script will not install pre-commit or other dev tools.
