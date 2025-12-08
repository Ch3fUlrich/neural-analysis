---
applyTo: '**'
---
This document provides essential instructions for Claude Sonnet (or any AI assistant) when contributing to this repository. Follow these strictly to maintain code quality, consistency, and reproducibility.
## 🚨 Critical Rules - Never Violate
### Rule -1: **Never Run Python Directly - Always Use UV Package Manager**
**Never execute Python commands directly. Always use `uv` to manage the environment and run Python.**
**🚫 Forbidden:**
```bash
python script.py
python -m pytest
python -c "import something"
pip install package
```
**✅ Required:**
```bash
uv run python script.py
uv run pytest
uv run python -c "import something"
uv add package
uv sync
```
**For tests:**
```bash
uv run pytest tests/
uv run pytest tests/test_specific.py -v
```
**For scripts:**
```bash
uv run python scripts/my_script.py
```
**Why UV?**
- Ensures reproducible environments via lockfile.
- Handles virtual environments automatically.
- Faster and more consistent than pip/venv.
- Prevents system Python conflicts.
**Environment checks:**
```bash
uv run python --version
uv pip list
```
### Rule 0: **Always Use PlotGrid for Plotting - Never Use Raw Matplotlib/Pyplot**
**All plotting must use the PlotGrid system from `grid_config.py`.**
**🚫 Forbidden:**
```python
import matplotlib.pyplot as plt
plt.scatter(x, y)
plt.figure()
plt.show()
import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter(...))
```
**✅ Required:**
```python
from neural_analysis.plotting import PlotGrid, PlotSpec, GridLayoutConfig
specs = [
    PlotSpec(data=data1, plot_type='scatter', label='A', color='blue'),
    PlotSpec(data=data2, plot_type='scatter', label='B', color='red'),
]
grid = PlotGrid(plot_specs=specs, layout=GridLayoutConfig(rows=1, cols=2))
fig = grid.plot()
```
**Helpers (use PlotGrid internally):**
```python
from neural_analysis.plotting import plot_bar, plot_violin, plot_line
fig = plot_bar(data={'A': arr1, 'B': arr2})
```
**Why PlotGrid?**
- Metadata-driven and backend-agnostic (matplotlib ↔ plotly).
- Consistent styling, multi-panel layouts, legend deduplication.
- Type-safe with Literal hints.
**Exception:** Use matplotlib/pyplot only for post-processing PlotGrid outputs (e.g., annotations).
### Rule 1: Always Run Local CI Before Pushing
**Never push without passing local CI.**
```bash
./scripts/run_ci_locally.sh
```
**Manual alternative (if act fails):**
```bash
uv run ruff check src tests
uv run mypy src tests
uv run pytest -v
```
**Why?** Ensures quality, saves CI resources.
### Rule 2: Never Push Directly to Main
- Use feature branches: `feat/description`, `fix/description`, `chore/description`.
- All changes via pull requests (PRs).
- CI must pass before merging.
### Rule 3: Follow Project Structure
**CRITICAL**: The repository has a comprehensive modular structure. **Always consult `docs/folder_structure.md` for the complete, up-to-date repository organization.**
**Quick Overview**:
```
neural-analysis/
├── src/neural_analysis/     # Main package code
│   ├── data/                # Synthetic data generation
│   ├── embeddings/          # Dimensionality reduction (PCA, UMAP, t-SNE)
│   ├── learning/            # ML models and neural decoding
│   ├── metrics/             # Distance, distributions, outliers, similarity
│   ├── plotting/            # PlotGrid system (12 files, backend-agnostic)
│   ├── topology/            # Structure index and topological analysis
│   └── utils/               # I/O, logging, preprocessing, validation
│
├── tests/                   # Unit tests (mirrors src/ structure)
├── examples/                # Jupyter notebooks (12+ demos)
├── docs/                    # Documentation (14 consolidated files)
├── scripts/                 # Automation (setup, CI, function registry)
├── .github/workflows/       # CI/CD pipeline
├── pyproject.toml           # Config and dependencies
└── uv.lock                  # Locked versions (DO NOT EDIT)
```
**Module Hierarchy** (dependencies flow downward):
```
utils/ → data/ → metrics/embeddings/topology/learning/ → plotting/
```
**Key Points**:
- **Plotting System**: 3 layers (renderers.py → grid_config.py → plots_1d/2d/3d.py)
- **Backend-Agnostic**: Matplotlib ↔ Plotly automatic switching
- **Legacy Code**: `decoding.py` (root) → use `learning/decoding.py`
- **Full Details**: See `docs/folder_structure.md` (343 lines with navigation guide)
**Documentation Structure**:
- `docs/plotgrid.md` - Comprehensive PlotGrid guide (consolidated from 5 files)
- `docs/testing_and_ci.md` - Testing/CI guide (consolidated from 3 files)
- `docs/historical_migrations.md` - Migration history (consolidated from 3 files)
- `docs/folder_structure.md` - **Repository structure with navigation guide**
- See `docs/DOCUMENTATION_CLEANUP_SUMMARY.md` for documentation overview
### Rule 4: Use UV for All Package Management
**Never use pip.**
```bash
uv add package-name            # Add dependency
uv add --dev package-name      # Add dev dependency
uv sync --locked --all-extras  # Install
uv run command                 # Run in env
uv lock                        # Update lockfile
uv remove package-name         # Remove
```
### Rule 5: Maximize Code Reuse - Check Registry First
**Before writing code, check `docs/function_registry.md`.**
```bash
python3 scripts/generate_function_registry.py  # Update registry (generates markdown only)
```
**Reuse Guidelines:**
- Search registry for similar functions.
- Use renderers.py for low-level plotting.
- Refactor duplicates into shared functions.
- Layers: renderers.py (primitives) → grid_config.py (PlotGrid) → statistical_plots.py (high-level).
**Why?** Reduces duplication (DRY), improves maintainability.
### Rule 6: Apply 4-D Methodology for Optimal Input Understanding
**Before responding or implementing, use the 4-D Methodology to deeply analyze user inputs for coding, math, analysis, physics, and computation problems, ensuring accurate, logical, and verifiable solutions:**
1. **Deconstruct:** Extract core intent, key entities (e.g., variables, constraints), and map provided vs. missing information.
2. **Diagnose:** Identify ambiguities, assess specificity/complexity, and check for gaps in clarity.
3. **Develop:** Select techniques by type (e.g., chain-of-thought for complex; modular code for coding; numerical methods for physics); assign specialist roles; enhance with structured reasoning.
4. **Deliver:** Provide clear, executable outputs (e.g., code, equations) with verification steps (e.g., tests, proofs).
**Why?** Ensures thorough comprehension, mathematical/computational correctness, and efficient problem-solving across domains.
## Project Overview
**Purpose:** Automated, reproducible neural analysis pipeline with quality gates.
**Tech Stack:**
- Python 3.14+
- UV (package manager)
- pytest (testing), mypy (types), ruff (linting)
- GitHub Actions + act (CI)
**Key Files:**
- `pyproject.toml`: Metadata, deps, tools.
- `uv.lock`: Locked deps.
- `.github/workflows/ci.yml`: CI config.
- `scripts/setup_env.sh`: Bootstrap.
- `scripts/run_ci_locally.sh`: Local CI.
## Development Workflow
1. Understand request.
2. Check existing code/registry.
3. Implement changes.
4. Run local CI.
5. Commit with conventional messages.
6. Push to feature branch.
7. Create/update PR.
**Conventional Commits:**
- `feat: ...` (new feature)
- `fix: ...` (bug fix)
- `docs: ...` (documentation)
- `test: ...` (tests)
- `refactor: ...` (refactor)
- `chore: ...` (maintenance)
- `ci: ...` (CI changes)
**Git Operations:**
```bash
git checkout -b feat/your-feature
git add .
git commit -m "feat: description"
git push origin feat/your-feature
gh pr create --title "feat: description" --body "Details" --base main  # If gh CLI available
```
## Logging
Avoid `print()` in library code. Use logger:
```python
from neural_analysis.utils import configure_logging, get_logger, log_kv, log_section
configure_logging(level="INFO")  # Optional: file_path="logs/run.log"
log = get_logger(__name__)
log.info("Message")
log_kv("metrics", {"acc": 0.93})
log_section("Phase")
```
All functions should have useful logging output tracked via logging. Use decorator `@log_calls(level=logging.DEBUG)` or similar depending on the type of function.
See `docs/logging.md` for details.
## Code Style Guidelines
**Python Code:**
```python
"""Module docstring."""
from __future__ import annotations
from typing import Sequence
def function_name(param: type) -> return_type:
    """Google-style docstring.
    Args:
        param: Description.
    Returns:
        Description.
    Raises:
        ValueError: If invalid.
    """
    pass
```
- Use type hints (e.g., `list[str]`, `Sequence`).
- Google-style docstrings for public elements.
- Tests: Descriptive names, high overall coverage; update existing functions, not just add new ones.
- Ensure code is mathematically correct, well structured, usable, readable and fulfills all best practices; ruff and mypy compatible without errors.
## Commands Reference
**Setup:**
```bash
./scripts/setup_env.sh  # Optional: INSTALL_DEV=1 RUN_LOCAL_CI=1
```
**Checks:**
```bash
uv run ruff check src tests --fix
uv run mypy src tests
uv run pytest -v -n auto --cov
```
Run ruff, mypy, and pytest before local CI; fix errors first before CI, commit, and push.
**Environment:**
See Rule -1 and Rule 4.
Do not run Python code in the terminal but rather create a Jupyter notebook for running tests. Run tests in parallel if possible to make the pytests run faster.
Always use MCP servers when possible (e.g., ruff, mypy, pytest from VSCode plugins; gh for commit/push). After push, check GitHub workflow success; fix issues if failed.
## Common Tasks
**Add Feature:**
- Branch: `feat/name`
- Implement + tests + docs.
- Local CI.
- Commit, push, PR.
**Fix Bug:**
- Branch: `fix/name`
- Fix + test.
- Local CI.
- Commit, push, PR.
**Add Dependency:**
- `uv add package-name`
- Sync, test.
- Commit: `chore: add dependency`
**Update Docs:**
- Edit docs/README.
- Local CI.
- Commit: `docs: update`
Always add/update todos in todo.md properly; update when tasks added/completed.
Always update function_registry if new function created or strongly modified.
Ensure all files updated when function moved/renamed.
Do not recreate files that are already present; search for the file name in the repository first.
## CI/CD Pipeline
**Workflow:** `.github/workflows/ci.yml` - Lint, type check, test on push/PR.
**Local:**
```bash
./scripts/run_ci_locally.sh
act -W .github/workflows/ci.yml -v
```
## Troubleshooting
**UV Missing:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```
**Tests Pass Locally, Fail CI:**
- Sync locked deps.
- Check Python 3.14+.
**Cannot Push Main:** Use feature branch.
**Local CI Fails:**
- Ensure Docker running.
- Install act if needed.
**Known Issues:**
- NumPy <2.3 pinned for compatibility.
- Coverage issues: Prioritize test passing.
- Notebook imports: Restart kernel or clear sys.modules after changes.
## Documentation Map
**📁 Core Documentation** (14 consolidated files):
| Category | Files | Purpose |
|----------|-------|---------|
| **Structure** | `docs/folder_structure.md` | **Complete repository structure (343 lines with navigation)** |
| **Getting Started** | `README.md`, `CONTRIBUTING.md` | Overview, quick start, contribution guide |
| **Plotting** | `docs/plotgrid.md` | Comprehensive PlotGrid guide (consolidated from 5 files) |
| **Testing/CI** | `docs/testing_and_ci.md` | Testing, linting, CI/CD (consolidated from 3 files) |
| **Migrations** | `docs/historical_migrations.md` | Migration history (consolidated from 3 files) |
| **Module Docs** | `docs/decoding_module.md`, `docs/distributions.md`, `docs/structure_index.md` | Module-specific documentation |
| **Data Formats** | `docs/hdf5_structure.md`, `docs/synthetic_datasets_notebook.md` | Data structure reference |
| **Development** | `docs/function_registry.md`, `docs/setup_script_usage.md`, `docs/logging.md` | Development workflow |
| **Architecture** | `docs/plotting_architecture.md` | Plotting system design patterns |
**🎯 Quick Links**:
- **Repository Structure**: `docs/folder_structure.md` - **Always check this first for file locations**
- **Function Registry**: `docs/function_registry.md` - Check before writing new code (DRY)
- **PlotGrid Guide**: `docs/plotgrid.md` - All plotting documentation in one place
- **Testing Guide**: `docs/testing_and_ci.md` - How to run tests and CI
- **Cleanup Summary**: `docs/DOCUMENTATION_CLEANUP_SUMMARY.md` - Documentation consolidation record
## Quality Standards
**Code:**
- Type hints, docstrings, high overall pytest coverage; update existing functions.
- Pass ruff, mypy, pytest.
**PRs:**
- Descriptive title/body.
- Passing CI.
- No conflicts.
- Conventional commits.
**Never:**
- Push to main.
- Skip CI.
- Use pip.
- Edit uv.lock manually.
- Force push shared branches.
- Run Python in terminal (use Jupyter notebooks for tests).
- Recreate existing files (search repository first).
## Success Checklist
Before "Done":
- [ ] Changes committed descriptively.
- [ ] Local CI passed.
- [ ] Pushed to feature branch.
- [ ] PR created.
- [ ] No errors.
- [ ] Docs/tests updated.
- [ ] Style followed.
## Getting Help
- Check errors/docs/git history.
- Review CONTRIBUTING.md.
- Ask user for clarification.
## Standard Procedures for Module Development
**Checklist:**
1. **Implement:** Type hints, docstrings, examples, backends.
2. **Lint/Type:** Ruff --fix, mypy --strict.
3. **Test:** Pytest -v, then with --cov (high coverage goal); run in parallel.
4. **Notebook:** examples/[module]_examples.ipynb with demos.
5. **pyproject.toml:** Update deps if needed, sync, install -e.
**Quick Commands:**
```bash
uv run ruff check [file] --fix
uv run ruff format [file]
uv run pytest [test_file] -v
uv sync && uv pip install -e ".[dev,viz]"
```