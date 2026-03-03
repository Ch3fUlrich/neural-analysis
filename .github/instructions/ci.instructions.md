---
name: 'CI/CD and Infrastructure'
description: 'Rules for CI pipelines, Docker, scripts, and build configuration'
applyTo: '.github/workflows/**,scripts/**,Makefile,Dockerfile,docker-compose.yml,pyproject.toml'
---

# CI/CD and Infrastructure Instructions

## GitHub Actions (`ci.yml`)

Two jobs:
1. **`test`** — Ubuntu-latest, Python 3.12: lock check → ruff check → ruff format → mypy → pytest (with coverage, `fail_under=70`) → Codecov upload → notebook validation
2. **`docs`** — Sphinx HTML build, artifact upload (7-day retention)

Triggers: push to `main`/`migration`, PRs to `main`, manual dispatch.

All quality gates are **blocking** (no `continue-on-error`).

## Release Workflow (`release.yml`)

Tag-triggered (`v*`) PyPI publishing via trusted OIDC. Builds with Hatchling, publishes to PyPI.

## Local CI

```bash
./scripts/run_ci_locally.sh           # Run full CI via act + Docker
make check                             # Quick: lint + format + types + tests
make ci                                # Full: equivalent to GitHub Actions
```

## Docker

- `Dockerfile`: python:3.12-slim base, includes redis-server, build-essential, git.
- `docker-compose.yml`: Two services — `app` (dev env) + `redis` (redis:7-alpine).
- Dockerfile uses UV for dependency installation (`COPY --from=ghcr.io/astral-sh/uv:latest`).

## Build System

- Hatchling backend (`pyproject.toml`).
- Version: `0.1.0` (semver, documented in `CHANGELOG.md`).
- Optional dependency groups: `viz` (matplotlib, plotly, marimo), `storage` (redis, duckdb), `dev` (ruff, mypy, pytest, etc.).

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/run_ci_locally.sh` | Full CI via act + Docker |
| `scripts/setup_env.sh` | Environment setup (flags: `INSTALL_DEV=1`, `RUN_LOCAL_CI=1`) |
| `scripts/convert_jupyter_to_marimo.py` | Migrate Jupyter → Marimo notebooks |
| `scripts/generate_function_registry.py` | Auto-generate function registry docs (`--check` flag for CI validation) |
| `scripts/benchmark.py` | Performance benchmarking suite (pairwise, shape, PCA, SI) |

## Makefile Targets

14 targets covering: `install`, `dev`, `lint`, `format`, `typecheck`, `test`, `coverage`, `check`, `docs`, `clean`, `ci`, `docker-build`, `docker-up`, `docker-down`.
