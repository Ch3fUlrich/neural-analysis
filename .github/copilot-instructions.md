# Neural Analysis — Repository Instructions

> Python library for analyzing neural population data: synthetic generation, metrics, embeddings, decoding, topology, and visualization.

## Tech Stack

- **Python** ≥3.10 <3.14, managed exclusively with **UV** (Astral)
- **Build:** Hatchling (PEP 517)
- **Core:** NumPy, SciPy, Pandas, scikit-learn
- **Optional:** Numba (JIT), UMAP, POT (optimal transport), FAISS (k-NN), Redis, DuckDB
- **Plotting:** Matplotlib + Plotly via the PlotGrid system
- **Storage:** HDF5 (required) + DuckDB (optional index) + Redis (optional cache)
- **Quality:** ruff (lint+format), mypy (strict), pytest + pytest-xdist + pytest-cov

## Commands

Always use UV. Never invoke `python`, `pip`, or `pytest` directly.

```bash
# Environment
uv sync                                 # Install all deps
uv sync --locked --all-extras           # Locked install with optional deps
uv add PACKAGE                          # Add dependency
uv add --dev PACKAGE                    # Add dev dependency

# Quality (run in order)
uv run ruff check src tests --fix       # Lint
uv run ruff format .                    # Format
uv run mypy src tests                   # Type check
uv run pytest -v -n auto --cov         # Test with coverage

# Full CI
./scripts/run_ci_locally.sh             # GitHub Actions locally via act+Docker
```

## Architecture

```
src/neural_analysis/
├── data/          # Synthetic neural data generators (place, grid, HD, random cells)
├── metrics/       # Pairwise distances, distributions, shape metrics, outliers
├── embeddings/    # Dimensionality reduction (7 methods) + visualization
├── learning/      # Decoding and classification (9 supervised, 7 unsupervised)
├── topology/      # Structure Index computation + visualization
├── plotting/      # PlotGrid system (16 plot types, dual matplotlib/plotly backend)
└── utils/         # I/O, logging, validation, comparison store, geometry
    └── storage/   # StorageManager orchestrator: HDF5 → DuckDB → Redis
```

**Dependency flow (lower never imports upper):**
`utils → data → metrics → embeddings → learning → topology → plotting`

## Key Systems

- **PlotGrid** — All visualization goes through `PlotSpec` → `PlotGrid` → renderer. Never use raw matplotlib/plotly.
- **StorageManager** — Context-managed (`with StorageManager() as sm:`). Never handcraft Redis keys.
- **Logging** — Use `get_logger(__name__)`, `log_kv`, `log_section`, `@log_calls`. Never use `print()`.
- **Patterns** — `match/case` dispatch, `(data, metadata)` return tuples, factory constructors, incremental resume.

## Workflow

1. **Deconstruct** — Clarify intent, inputs, outputs, constraints.
2. **Diagnose** — Check existing code/tests/docs for prior art. Consult `docs/folder_structure.md`.
3. **Develop** — Small, type-hinted units with Google-style docstrings and `@log_calls`.
4. **Verify** — Run `ruff check`, `mypy`, `pytest`, then `./scripts/run_ci_locally.sh`.

When adding/modifying functions: update `docs/function_registry.md`, add/extend tests, update `TODO.md`.

## Git

- Never push to `main`. Use feature branches: `feat/...`, `fix/...`, `chore/...`
- Conventional commits: `feat:`, `fix:`, `docs:`, `test:`, `ci:`, `refactor:`

## Boundaries

**Always:** Use UV, PlotGrid, StorageManager, structured logging. Run lint+types+tests before merging.
**Ask first:** Major refactors, schema changes, new dependencies, CI/CD changes, or after 3 failed attempts.
**Never:** Direct `python`/`pip`, `--no-verify`, push to `main`, bypass StorageManager/PlotGrid/logging, commit secrets.
