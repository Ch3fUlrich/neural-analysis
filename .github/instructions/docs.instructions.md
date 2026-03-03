---
name: 'Documentation'
description: 'Conventions for docs, examples, and function registry updates'
applyTo: 'docs/**,examples/**,*.md'
---

# Documentation Instructions

## Key Documentation Files

| File | Purpose |
|------|---------|
| `docs/folder_structure.md` | Module layout and legacy locations — consult before assuming organization |
| `docs/function_registry.md` | Registry of all public functions — update when adding/modifying functions |
| `docs/plotgrid.md` | PlotGrid system usage and architecture |
| `docs/storage.md` | Storage stack documentation |
| `docs/hdf5_structure.md` | HDF5 schema and hierarchy |
| `docs/logging.md` | Logging system and utilities |
| `docs/testing_and_ci.md` | Test and CI conventions |
| `TODO.md` | Task tracking — update when adding tasks or completing work |
| `CHANGELOG.md` | Release notes in Keep a Changelog format — update for user-facing changes |

## When to Update Docs

- **Adding a function:** Update `docs/function_registry.md` with name, module, signature, and description.
- **Changing an API:** Update docstrings, function registry, and any affected example notebooks.
- **Adding a module:** Update `docs/folder_structure.md` with the new module's purpose and location.
- **Adding a feature:** Add/update entries in `TODO.md` and `CHANGELOG.md`.
- **Checking registry freshness:** Run `uv run python scripts/generate_function_registry.py --check`.

## Example Notebooks

- All examples use **Marimo** (not Jupyter) and live in `examples/`.
- Legacy Jupyter notebooks are in `legacy/` — do not add new ones there.
- Notebook filenames end with `_marimo_nb.py`.
- When an API changes, verify affected example notebooks still run correctly.

## Writing Style

- Use concise, factual language. Avoid filler.
- Code examples should be copy-pasteable and use UV execution.
- Document the "why" alongside the "what" for architecture decisions.
