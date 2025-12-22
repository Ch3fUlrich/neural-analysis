# `docs/testing_and_ci.md` (restructured)

```markdown
# Testing and CI Overview

This document shows how to run tests and checks locally and how they relate to CI. Use these commands before opening or updating a PR.

---

## 1. Testing & Checks Matrix

| Tool    | Purpose                 | Command (local)                               | When to run                  |
|---------|-------------------------|-----------------------------------------------|------------------------------|
| `pytest`| Tests & coverage        | `uv run pytest -v -n auto --cov`              | Before every push            |
| `ruff`  | Linting & formatting    | `uv run ruff check src tests --fix`           | Before running tests         |
| `mypy`  | Static type checking    | `uv run mypy src tests`                       | Before running tests         |
| CI script| Full local CI pipeline| `./scripts/run_ci_locally.sh`                 | Before opening/merging a PR  |

If in doubt: run **ruff → mypy → pytest → CI script** in that order.

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

Only skip `mypy` / full CI script when changes are obviously isolated and low‑risk.

---

## 3. CI Pipeline

- CI config lives in `.github/workflows/ci.yml`.
- On push and PR, CI runs:
  - Ruff (lint)
  - Mypy (types)
  - Pytest (tests & coverage)
- A PR should be merged only when all checks are green.

### 3.1 Running CI via act (optional)

```bash
act -W .github/workflows/ci.yml -v
```

Use this when you want to simulate GitHub Actions locally (requires Docker and `act` installed).

---

## 4. Triage Checklist for CI Failures

When CI fails:

1. Open the CI logs and identify which job failed (lint, type, tests).
2. Reproduce locally with the corresponding command(s).
3. Ensure your local environment matches CI (Python version, `uv sync`, extras).
4. Fix the failing code or tests; do not disable tests.
5. Re‑run local checks.
6. Push changes; confirm CI is now green.

If tests pass locally but fail in CI:

- Re‑run `uv sync` to ensure locked dependencies match CI.
- Look for platform‑specific assumptions (paths, temp dirs, etc.).

---

## 5. Conventions

- Use `uv run` for all Python commands.
- Aim for high coverage by updating existing tests where possible.
- Prefer small, focused tests that clearly document behaviour.
```