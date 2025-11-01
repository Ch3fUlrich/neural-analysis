# Project goal

This repository's main goal is to provide an automated pipeline for building and testing neural analysis methods. The project is organized so you can run reproducible tooling (lint, type-check, tests, and packaging) with Astral `uv` and a committed `uv.lock`. Continuous integration enforces quality gates and the repository follows golden programming rules for maintainability and reproducibility.

Key points:
- Use `uv` and a committed `uv.lock` to create reproducible environments.
- Provide a clear `scripts/setup_env.sh` to bootstrap development on Ubuntu/WSL.
- Enforce ruff, mypy, pytest and pre-commit checks in CI.
- Keep the documentation in `/docs` up to date with workflow and developer instructions.
