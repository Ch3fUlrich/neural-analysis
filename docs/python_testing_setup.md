# Modern Python Testing & Linting Setup

This guide describes a **fully automated, production-grade testing system** for Python projects. It ensures **early error detection**, **consistent style**, and **automatic verification** locally and in CI/CD.

## Core Tools

| Purpose | Tool | Description |
|----------|------|--------------|
| Testing | `pytest` + `pytest-xdist` | Fast, parallelized testing framework |
| Coverage | `coverage.py` | Measures test coverage |
| Type Checking | `mypy` | Enforces and validates type hints |
| Linting & Formatting | `ruff` | Ultra-fast linter + formatter (replaces flake8, black, isort) |
| Docstring Validation | `pydocstyle` + `ruff` | Ensures docstring presence and format |
| Pre-Commit Hooks | `pre-commit` | Runs linting and type checks before commits |
| Environment Automation | `tox` / `nox` | Runs tests and checks in isolated environments |
| Continuous Integration | GitHub Actions | Runs all checks automatically on every push or PR |

## Local Installation

```bash
pip install pytest pytest-xdist coverage mypy ruff pydocstyle pre-commit tox
pre-commit install
```

## Configuration Files

### .pre-commit-config.yaml
```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.0
    hooks:
      - id: ruff
      - id: ruff-format
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.11.2
    hooks:
      - id: mypy
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: check-docstring-first
      - id: check-yaml
      - id: end-of-file-fixer
      - id: trailing-whitespace
```

### pyproject.toml
```toml
[tool.pytest.ini_options]
addopts = "-ra -q --cov --cov-report=term-missing"
testpaths = ["tests"]

[tool.ruff]
line-length = 88
target-version = "py311"
select = ["E", "F", "I", "D"]
ignore = ["D203", "D213"]

[tool.mypy]
python_version = "3.11"
ignore_missing_imports = true
disallow_untyped_defs = true
warn_unused_ignores = true
warn_return_any = true
strict = true
```

### tox.ini
```ini
[tox]
envlist = py39, py310, py311, lint, type

[testenv]
deps = pytest pytest-xdist coverage
commands = pytest -n auto --cov

[testenv:lint]
deps = ruff
commands = ruff check .

[testenv:type]
deps = mypy
commands = mypy .
```

### .github/workflows/test.yml
```yaml
name: CI
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install pytest pytest-xdist coverage mypy ruff
      - run: ruff check .
      - run: mypy .
      - run: pytest -n auto --cov
```

## Recommended Workflow

1. Write code with full type hints and docstrings.
2. Run locally:
```bash
pytest -n auto --cov
mypy .
ruff check .
```
3. Commit → `pre-commit` auto-fixes and validates.
4. Push → GitHub Actions reruns tests, linting, and type checks in clean environments.

## Benefits

- Detects errors early — before running or deploying code
- Fast — parallel testing and linting
- Consistent quality — enforced style, types, and docs
- Fully automated — from local dev to CI/CD
- Enterprise-grade — mirrors setups used by Google, Meta, Microsoft

## Summary

| Feature | Tool |
|----------|------|
| Unit + Integration Tests | `pytest` |
| Parallel Execution | `pytest-xdist` |
| Type Checking | `mypy` |
| Linting + Formatting | `ruff` |
| Docstring Rules | `pydocstyle` |
| Auto-check on Commit | `pre-commit` |
| Multi-env Testing | `tox` |
| CI/CD Integration | GitHub Actions |
| Coverage | `coverage.py` |

Result: A fast, modern, fully automated testing system that continuously ensures correctness, readability, and maintainability without manual testing.

