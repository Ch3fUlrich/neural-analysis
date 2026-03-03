---
name: 'Testing'
description: 'Test conventions, patterns, and quality gates'
applyTo: 'tests/**'
---

# Testing Instructions

## Running Tests

```bash
uv run pytest -v -n auto --cov         # Full suite with coverage
uv run pytest tests/test_specific.py   # Single file
uv run pytest -k "test_name"           # By name pattern
```

Never use `python -m pytest` or `pytest` directly.

## Naming Convention

```
test_<what>_<condition>_<expected>
```

Example: `test_shape_distance_empty_input_raises_error`

## Patterns

- Use existing test files as templates for new tests.
- Prefer extending existing tests over duplicating test files with `_additional` / `_more` suffixes.
- Test both matplotlib and plotly backends for plotting code.
- Test edge cases: empty inputs, single-element arrays, invalid parameters, missing optional deps.
- Use `pytest.raises` for expected exceptions; use `pytest.approx` for floating-point comparisons.
- Fixtures should be scoped appropriately: `session` for expensive data, `function` for mutable state.

## Shared Fixtures (`tests/conftest.py`)

- `place_cells_2d` (session) — 50 cells, 2000 timesteps; returns (activity, metadata).
- `random_activity` (session) — 1000×50 seeded random matrix.
- `random_labels` (session) — 1000×2 seeded random labels.
- `small_matrix` (function) — 10×5 matrix for quick unit tests.

Use these instead of duplicating data generation in each test file.

## Coverage

- Source: `src/`
- Branch coverage enabled.
- **Threshold: `fail_under = 70`** — tests fail if coverage drops below 70%.
- Reports: terminal (show_missing) + HTML + XML.
- Excludes: `__init__.py`, `pragma: no cover`, `TYPE_CHECKING`, `__repr__`, `abstractmethod`.

## Quality Gates (run in order)

```bash
uv run ruff check src tests --fix    # Lint (15 rule sets)
uv run ruff format .                  # Format
uv run mypy src tests                 # Type check (strict)
uv run pytest -v -n auto --cov       # Tests + coverage
```

All four must pass before committing. For full CI: `./scripts/run_ci_locally.sh`.

## Test Organization

Tests live in `tests/` with one primary file per source module (44 consolidated files). Integration tests covering cross-module workflows should be clearly named (e.g., `test_integration_*`). Avoid creating many small files for the same module — consolidate into the existing file. Current baseline: 1593 tests passing, 22 skipped.
