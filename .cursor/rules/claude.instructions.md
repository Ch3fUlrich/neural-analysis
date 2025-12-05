applyTo: '**'  
---  
# Development Guidelines for AI Assistants

This document defines how Claude Sonnet (or any AI assistant) must behave in this repository. Treat items in **Critical Rules** as non‑negotiable.

## Philosophy

### Core beliefs
- **Incremental progress over big bangs**: Make small changes that compile and pass tests.
- **Learning from existing code**: Study existing modules, tests, and docs before implementing.
- **Pragmatic over dogmatic**: Prefer boring, robust solutions that fit the current architecture.
- **Clear intent over clever code**: If a change needs a long explanation to be readable, simplify it.

### Simplicity & design
- **Single responsibility** per function/class; avoid “god functions” and over‑general abstractions.
- **Avoid premature abstractions**; refactor only when duplication or complexity demands it.
- **No clever tricks**: Favour explicit code over magic; make control flow obvious.

### Technical standards
- **Composition over inheritance**: Inject dependencies instead of subclassing where possible.
- **Interfaces over singletons**: Design for testing and flexibility; avoid global state.
- **Explicit over implicit**: Keep data flow and dependencies visible and traceable.
- **Test‑driven when possible**: Let tests guide behaviour; never disable tests to “get CI green”.

### Error handling
- **Fail fast with context**: Raise clear, descriptive errors early rather than hiding problems.
- **Handle errors at the right level**: Do not swallow exceptions in low‑level helpers; either propagate with context or convert to domain‑specific errors.
- **Never silently swallow exceptions**: Logging without surfacing an error is not acceptable for unexpected failures.

## MCP and tool‑use behaviour

- Prefer **MCP‑backed tools** (ruff, mypy, pytest integrations, `gh` for git/PRs) whenever available.
- Use **Context7‑style tools** (or equivalent) to validate library documentation when making design decisions.
- Use **searxng** if primary web search/fetch fails, and escalate to Tavily only if searxng is insufficient.

## Critical Rules

1. **Never run Python directly; always use UV**  
   - Forbidden: `python …`, `python -m pytest`, `pip install …`.  
   - Required: `uv run python …`, `uv run pytest …`, `uv add …`, `uv sync`.  
   - Environment checks: `uv run python --version`, `uv pip list`.  

2. **Always use PlotGrid for plotting**  
   - Forbidden in new code: direct `matplotlib.pyplot` or `plotly` usage, except minimal post‑processing of PlotGrid results.  
   - Required: Plot through `neural_analysis.plotting` (`PlotGrid`, `PlotSpec`, `GridLayoutConfig`, or helpers like `plot_bar`, `plot_violin`, `plot_line`).  

3. **Always run local CI before pushing**  
   - Primary command: `./scripts/run_ci_locally.sh`.  
   - Fallback (if `act` or the script fails):  
     - `uv run ruff check src tests`  
     - `uv run mypy src tests`  
     - `uv run pytest -v`  

4. **Never push directly to `main`**  
   - Use feature branches (`feat/…`, `fix/…`, `chore/…`), and merge only via PR after passing CI.  

5. **Use UV for all package management**  
   - Dependency changes: `uv add`, `uv add --dev`, `uv remove`, `uv lock`, `uv sync --locked --all-extras`.  
   - Never edit `uv.lock` by hand; never use `pip` directly.  

6. **Respect storage system defaults**  
   - Use `with StorageManager() as sm:` (or `sm.close()`) for all storage access to release Redis sockets and DuckDB connections promptly.  
   - Follow the HDF5 → DuckDB → Redis priority set by `StorageManager` and `StorageConfig`.  
   - Never handcraft Redis keys; use the storage or I/O utilities that respect `StorageConfig.cache_namespace` (default `neural_analysis`).  
   - Persist pandas/NumPy results via `save_result_to_hdf5_dataset` / `load_results_from_hdf5_dataset` helpers so attributes, DuckDB indices, and cache invalidation stay correct.  

7. **Maximise code reuse; do not re‑invent systems**  
   - Before writing a new function, run `python3 scripts/generate_function_registry.py` and inspect `docs/function_registry.md` for similar behaviour.  
   - Reuse the PlotGrid stack (`renderers.py → grid_config.py → plots_1d/2d/3d.py`) for plotting.  
   - Reuse logging and storage utilities instead of introducing new logging/storage patterns.  

8. **Planning and failure discipline**  
   - Commit working code incrementally; never commit code that does not compile or has failing tests.
   - Update plan documentation and TODOs as you go; do not let them drift.
   - After **three failed attempts** at an approach (failing tests, stuck design, or repeated CI failures), stop, reassess, and either simplify or ask for clarification.

9. **Never bypass quality gates**  
   - Never use `--no-verify` to bypass commit hooks.
   - Never disable tests instead of fixing them.

## Project integration

- **Repository structure**: Always start with `docs/folder_structure.md` for up‑to‑date module layout, legacy locations, and navigation tips.  
- **Module hierarchy**: Respect the documented dependency flow (`utils → data → metrics/embeddings/topology/learning → plotting`) and do not introduce upward dependencies.  
- **Documentation map**: Use the consolidated docs (`docs/plotgrid.md`, `docs/testing_and_ci.md`, `docs/hdf5_structure.md`, `docs/logging.md`, etc.) as the source of truth instead of duplicating explanations here.  

### Function and file lifecycle

- When you add or significantly modify a function:  
  - Update `docs/function_registry.md`.  
  - Ensure existing tests are updated or extended rather than just adding new ones.  
  - Ensure all references are updated if functions are moved or renamed.  
- When you add tasks or discover follow‑ups:  
  - Add or update entries in `todo.md` and mark them completed when resolved.  

## Execution and tooling

### UV usage

- All Python execution (scripts, tests, ad‑hoc commands) goes through `uv run`.  
- All dependency management goes through UV; sync environments with `uv sync` before running CI or tests.  

### Logging

- Library code must not use `print()` for runtime information.  
- Use `configure_logging`, `get_logger`, `log_kv`, and `log_section` utilities, as well as decorators like `@log_calls`, to provide structured, searchable logs.  

### CI / CD and Git

- Local: `./scripts/run_ci_locally.sh` (or the ruff / mypy / pytest trio) before every push.  
- Remote: Use GitHub Actions as the final arbiter of CI; investigate and fix any failing workflow before merging.  
- Git workflow:  
  - Branch: `feat/...`, `fix/...`, `chore/...`  
  - Conventional commits (`feat: …`, `fix: …`, `docs: …`, `test: …`, `ci: …`)  
  - Use `gh` CLI for PR creation when available.  

## Development workflow

1. **Deconstruct**: Clarify the request; extract the core intent, inputs, outputs, constraints, and missing information.  
2. **Diagnose**: Identify ambiguity and complexity; cross‑check existing code, tests, and docs for similar patterns.  
3. **Develop**:  
   - Choose techniques appropriate to the task (e.g., modular code, numerical methods, PlotGrid recipes).  
   - Write small, type‑hinted units with Google‑style docstrings, logging, and tests.  
4. **Deliver and verify**:  
   - Run ruff, mypy, pytest, then the local CI script.  
   - Ensure examples/notebooks are updated when new features are added.  
   - Commit, push to a feature branch, open/update PR, and confirm CI is green.  

## Command reference (summary)

- Setup: `./scripts/setup_env.sh` (supports flags like `INSTALL_DEV=1`, `RUN_LOCAL_CI=1`).  
- Lint: `uv run ruff check src tests --fix`.  
- Types: `uv run mypy src tests`.  
- Tests: `uv run pytest -v -n auto --cov`.  
- Local CI: `./scripts/run_ci_locally.sh`.  
- CI via act: `act -W .github/workflows/ci.yml -v`.  