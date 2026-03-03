# Testing and CI Overview

This document shows how to run tests and checks locally and how they relate to CI. Use these commands before opening or updating a PR.

---

## 1. Testing & Checks Matrix

| Tool    | Purpose                 | Command (local)                               | When to run                  |
|---------|-------------------------|-----------------------------------------------|------------------------------|
| `pytest`| Tests & coverage        | `uv run pytest -v -n auto --cov`              | Before every push            |
| `ruff`  | Linting & formatting    | `uv run ruff check src tests --fix`           | Before running tests         |
| `mypy`  | Static type checking    | `uv run mypy src tests`                       | Before running tests         |
| CI script| Full local CI pipeline | `./scripts/run_ci_locally.sh`                 | Before opening/merging a PR  |

If in doubt: run **ruff  mypy  pytest  CI script** in that order.

---

## 2. Standard Workflows

### 2.1 Run everything locally

```bash
uv run ruff check src tests --fix
uv run mypy src tests
uv run pytest -v -n auto --cov
./scripts/run_ci_locally.sh
```

Use this sequence before pushing or updating a PR.

---

### 2.2 Minimal quick check

For small changes:

```bash
uv run ruff check src tests
uv run pytest -v
```

Only skip `mypy` / full CI script when changes are obviously isolated and low-risk.

---

## 3. CI Pipeline

CI config lives in `.github/workflows/ci.yml`. On push and PR:

| Step | Command | Notes |
|------|---------|-------|
| 1. Lint | `uv run ruff check src tests scripts --output-format=github` | |
| 2. Format | `uv run ruff format --check src tests scripts` | |
| 3. Types | `uv run mypy src/neural_analysis tests scripts --ignore-missing-imports` | |
| 4. Tests | `uv run pytest tests/ --cov --cov-report=xml -v --maxfail=5` | Coverage threshold: **70%** (`fail_under = 70`) |
| 5. Notebooks | Import-check all `examples/*_marimo_nb.py` files | `continue-on-error: true` |
| 6. Coverage | Upload to Codecov | |

A separate `release.yml` workflow publishes to PyPI on tag pushes.

### 3.1 Running CI via act (optional)

```bash
act -W .github/workflows/ci.yml -v
```

Requires Docker and `act` installed.

---

## 4. Coverage Threshold

Coverage is enforced at **70%** minimum via `pyproject.toml`:

```toml
[tool.coverage.report]
fail_under = 70
```

If a PR drops coverage below this threshold, CI will fail. When adding new code, add corresponding tests to maintain coverage.

---

## 5. Test Structure

### 5.1 File Organization

Tests live in `tests/` with **44 test files** and **~1600 tests**:

- One primary test file per source module (e.g., `test_validation.py` for `utils/validation.py`)
- `conftest.py` provides shared fixtures used across multiple test files

### 5.2 Shared Fixtures (`conftest.py`)

Common fixtures avoid duplicate setup:

```python
@pytest.fixture
def place_cells_2d():
    """10 place cells in a 2D environment."""
    ...

@pytest.fixture
def random_activity():
    """100-neuron random activity matrix."""
    ...

@pytest.fixture
def rng():
    """Reproducible RNG for tests."""
    return np.random.default_rng(42)
```

Use these fixtures in tests instead of creating ad-hoc data:

```python
def test_embedding(place_cells_2d, rng):
    data, _ = place_cells_2d
    result = compute_embedding(data, method="pca", rng=rng)
    assert result.shape[1] == 2
```

### 5.3 Test Conventions

- Use `uv run pytest`  never call `pytest` or `python -m pytest` directly.
- Prefer extending existing test files over adding new ones.
- Use the `rng` fixture or `np.random.default_rng(seed)` for reproducibility.
- Mark slow tests with `@pytest.mark.slow` if they take >5 seconds.
- Skip tests with missing optional dependencies: `@pytest.mark.skipif(not HAS_NUMBA, ...)`.

---

## 6. Triage Checklist for CI Failures

When CI fails:

1. Open the CI logs and identify which job failed (lint, type, tests).
2. Reproduce locally with the corresponding command(s).
3. Ensure your local environment matches CI (`uv sync`, extras).
4. Fix the failing code or tests; do not disable tests.
5. Re-run local checks.
6. Push changes; confirm CI is now green.

If tests pass locally but fail in CI:

- Re-run `uv sync` to ensure locked dependencies match CI.
- Look for platform-specific assumptions (paths, temp dirs, etc.).
