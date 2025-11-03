# Instructions for Claude (AI Assistant) Working on neural-analysis

This document provides critical instructions for Claude Sonnet (or any AI assistant) when making changes to this repository.

## 🚨 CRITICAL RULES - NEVER VIOLATE

### Rule 1: Always Run Local CI Before Pushing
**NEVER push code without running and passing local CI checks first.**

```bash
# MANDATORY before every push
./scripts/run_ci_locally.sh
```

If you cannot run `act` (Docker/local CI), then run all checks manually:
```bash
uv run -- ruff check src tests
uv run -- mypy src tests
uv run -- pytest -v
```

**Why?** This ensures code quality, saves GitHub Actions minutes, and maintains repository standards.

### Rule 2: Never Push Directly to Main
- Main branch is protected
- All changes must go through pull requests
- CI must pass before merging
- Use feature branches: `feat/description`, `fix/description`, `chore/description`

### Rule 3: Follow Project Structure
```
neural-analysis/
├── src/neural_analysis/     # Main package code
├── tests/                    # All tests mirror src/ structure
├── docs/                     # Documentation
├── scripts/                  # Automation scripts
├── .github/workflows/        # CI/CD pipelines
├── pyproject.toml           # Project config and dependencies
└── uv.lock                  # Locked dependency versions
```

### Rule 4: Use uv for All Package Management
```bash
# NEVER use pip install
# ALWAYS use uv

# Add dependency
uv add package-name

# Add dev dependency  
uv add --dev package-name

# Install dependencies
uv sync --locked --all-extras

# Run commands
uv run -- command
```

## Project Overview

### Purpose
Automated pipeline for building and testing neural analysis methods with:
- Reproducible environments (uv + uv.lock)
- Automated quality gates (ruff, mypy, pytest)
- Local CI testing (act)
- GitHub Actions CI/CD

### Tech Stack
- **Python**: 3.14+
- **Package Manager**: uv (Astral)
- **Testing**: pytest + pytest-xdist + pytest-cov
- **Type Checking**: mypy
- **Linting**: ruff
- **Pre-commit**: Local hooks (optional)
- **CI**: GitHub Actions + act (local)

### Key Files
- `pyproject.toml` - Project metadata, dependencies, tool configuration
- `uv.lock` - Locked dependency versions (DO NOT manually edit)
- `.github/workflows/ci.yml` - CI pipeline configuration
- `scripts/setup_env.sh` - Environment bootstrap script
- `scripts/run_ci_locally.sh` - Local CI runner

## Development Workflow for Claude

### When Making Changes

1. **Understand the request** - Read user requirements carefully
2. **Check existing code** - Use grep/search tools to understand context
3. **Make changes** - Edit files following project conventions
4. **Run quality checks** - ALWAYS run before pushing:
   ```bash
   ./scripts/run_ci_locally.sh
   ```
5. **Commit changes** - Use conventional commit messages
6. **Push to feature branch** - Never push directly to main
7. **Create/update PR** - Ensure CI passes before requesting merge

### Conventional Commit Messages

```bash
# Feature
git commit -m "feat: add new neural analysis method"

# Bug fix
git commit -m "fix: correct calculation in normalize function"

# Documentation
git commit -m "docs: update README with usage examples"

# Tests
git commit -m "test: add tests for edge cases in mean function"

# Refactor
git commit -m "refactor: simplify data preprocessing pipeline"

# Chore (maintenance)
git commit -m "chore: update dependencies in pyproject.toml"

# CI/CD changes
git commit -m "ci: update GitHub Actions workflow"
```

### Code Style Guidelines

#### Python Code
```python
"""Module docstring explaining purpose."""
from __future__ import annotations

from typing import Sequence


def function_name(param: type) -> return_type:
    """Function docstring.
    
    Args:
        param: Description of parameter
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When validation fails
    """
    # Implementation
    pass


class ClassName:
    """Class docstring."""
    
    def __init__(self, value: int) -> None:
        """Initialize the class."""
        self.value = value
```

#### Type Hints
- ✅ Always use type hints for function parameters and returns
- ✅ Use modern type syntax: `list[str]` not `List[str]`
- ✅ Use `from __future__ import annotations` for forward references
- ✅ Use `Sequence`, `Mapping` for generic containers when appropriate

#### Docstrings
- ✅ Use Google-style docstrings
- ✅ Document all public functions, classes, and modules
- ✅ Include Args, Returns, Raises sections as needed
- ✅ Keep docstrings concise but informative

#### Testing
- ✅ Write tests for all new functions
- ✅ Test normal cases, edge cases, and error cases
- ✅ Use descriptive test names: `test_<what>_<condition>_<expected>`
- ✅ Place tests in `tests/` mirroring `src/` structure

### Commands Reference

#### Environment Setup
```bash
# Initial setup
./scripts/setup_env.sh

# With options
INSTALL_DEV=1 RUN_LOCAL_CI=1 ./scripts/setup_env.sh
```

#### Running Checks
```bash
# Local CI (MANDATORY before push)
./scripts/run_ci_locally.sh

# Individual checks
uv run -- ruff check src tests
uv run -- ruff check src tests --fix  # Auto-fix
uv run -- mypy src tests
uv run -- pytest
uv run -- pytest -v  # Verbose
uv run -- pytest -n auto  # Parallel
uv run -- pytest --cov  # With coverage
```

#### Package Management
```bash
# Sync dependencies
uv sync --locked --all-extras

# Add dependency
uv add package-name

# Add dev dependency
uv add --dev package-name

# Update lock file
uv lock

# Remove package
uv remove package-name
```

#### Git Operations
```bash
# Create feature branch
git checkout -b feat/your-feature

# Stage changes
git add .

# Commit with message
git commit -m "feat: description"

# Push to feature branch
git push origin feat/your-feature

# Create PR (if gh CLI available)
gh pr create --title "feat: description" --body "Details" --base main
```

## Common Tasks for Claude

### Adding a New Feature
1. Create feature branch: `git checkout -b feat/feature-name`
2. Implement the feature in `src/neural_analysis/`
3. Add tests in `tests/`
4. Update documentation if needed
5. Run local CI: `./scripts/run_ci_locally.sh`
6. Commit: `git commit -m "feat: add feature"`
7. Push: `git push origin feat/feature-name`
8. Create PR

### Fixing a Bug
1. Create fix branch: `git checkout -b fix/bug-description`
2. Fix the bug in source code
3. Add test case that would catch the bug
4. Run local CI: `./scripts/run_ci_locally.sh`
5. Commit: `git commit -m "fix: resolve bug"`
6. Push and create PR

### Adding Dependencies
1. Add to pyproject.toml: `uv add package-name`
2. Or add to dev dependencies: `uv add --dev package-name`
3. Verify lock file updated: check `uv.lock`
4. Test that it works: `uv sync --locked --all-extras`
5. Commit: `git commit -m "chore: add package-name dependency"`

### Updating Documentation
1. Edit files in `docs/` or `README.md` or `CONTRIBUTING.md`
2. Ensure markdown is properly formatted
3. Run local CI to catch any issues: `./scripts/run_ci_locally.sh`
4. Commit: `git commit -m "docs: update documentation"`

## CI/CD Pipeline

### GitHub Actions Workflow
Location: `.github/workflows/ci.yml`

Steps executed on every push/PR:
1. Checkout code
2. Set up Python 3.14
3. Install uv
4. Verify uv.lock is up-to-date
5. Sync dependencies (locked)
6. Run ruff linting
7. Run mypy type checking
8. Run pytest tests

All steps must pass before PR can be merged.

### Local CI with act
```bash
# Run full CI locally
./scripts/run_ci_locally.sh

# Or use act directly
act -W .github/workflows/ci.yml

# With verbose output
act -W .github/workflows/ci.yml -v
```

**First run requires Docker images (~1-2 GB download)**

## Troubleshooting for Claude

### CI Fails with "uv: command not found"
```bash
# Reinstall uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

### Tests Pass Locally but Fail in CI
```bash
# Use locked dependencies
uv sync --locked --all-extras

# Check Python version
python --version  # Should be 3.14+
```

### Cannot Push to Main
- This is correct behavior! Main is protected.
- Create a feature branch instead
- Push to feature branch and create PR

### Local CI Script Fails
```bash
# Check Docker is running
docker ps

# Install Docker if missing
sudo apt install docker.io -y
sudo systemctl start docker

# Install act if missing
curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash
```

## Documentation Map

| File | Purpose |
|------|---------|
| `README.md` | Project overview and quick start |
| `CONTRIBUTING.md` | Contribution guidelines |
| `docs/python_testing_setup.md` | Detailed testing documentation |
| `docs/local_ci_testing.md` | Local CI with act guide |
| `docs/folder_structure.md` | Repository structure |
| `docs/project_goal.md` | Project mission and goals |
| `docs/claude.md` | This file - Instructions for AI assistants |

## Quality Standards

### All Code Must Have
- ✅ Type hints on all function signatures
- ✅ Docstrings on all public functions/classes/modules
- ✅ Unit tests with >80% coverage
- ✅ Pass ruff linting
- ✅ Pass mypy type checking
- ✅ Pass pytest tests

### All PRs Must Have
- ✅ Descriptive title and description
- ✅ All CI checks passing (green checkmarks)
- ✅ No merge conflicts with main
- ✅ Conventional commit messages

### Never
- ❌ Push directly to main
- ❌ Skip CI checks
- ❌ Use `pip install` (use `uv` instead)
- ❌ Commit without running local CI
- ❌ Use `--no-verify` to skip pre-commit hooks
- ❌ Force push to shared branches
- ❌ Manually edit `uv.lock`

## Success Checklist for Claude

Before saying "Done" to the user, verify:

- [ ] All changes committed with descriptive messages
- [ ] Local CI passed (`./scripts/run_ci_locally.sh`)
- [ ] Changes pushed to feature branch (not main)
- [ ] PR created (if requested)
- [ ] No errors in terminal output
- [ ] Documentation updated if needed
- [ ] Tests added for new features
- [ ] Code follows style guidelines

## Getting Help

If you (Claude) encounter issues:
1. Check error messages carefully
2. Review relevant documentation in `docs/`
3. Check if similar issues exist in git history
4. Consult `CONTRIBUTING.md` for workflow guidance
5. Ask the user for clarification if needed

## Summary

**Most Important Rules:**
1. 🚨 **ALWAYS run `./scripts/run_ci_locally.sh` before pushing**
2. 🚨 **NEVER push directly to main** - use feature branches
3. 🚨 **ALWAYS use `uv`** for package management, not pip
4. 🚨 **ALWAYS write tests** for new code
5. 🚨 **ALWAYS use type hints** and docstrings

Follow these rules and the project will maintain high quality standards! 🚀
