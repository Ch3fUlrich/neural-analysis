---
name: 'Python Standards'
description: 'Core Python coding conventions for the neural-analysis library'
applyTo: '**/*.py'
---

# Python Standards

## Style

- Type hints on all function signatures (parameters and return types).
- Google-style docstrings on all public functions and classes.
- No `print()` in library code — use `get_logger(__name__)` and structured logging (`log_kv`, `log_section`, `@log_calls`).
- Prefer `match/case` dispatch over long `if/elif` chains for method/metric selection.
- For large dispatch tables (e.g., renderers), use registry dicts (`REGISTRY: dict[str, Callable]`) instead of `match/case`.
- Return `(data, metadata)` tuples from generators and analysis functions to preserve provenance.
- Use result dataclasses (`AnalysisResult`, `MetricResult`, `EmbeddingResult`, `DecodingResult` from `core/results.py`) for structured returns.
- Use config dataclasses (`PipelineConfig`, `MetricConfig`, `EmbeddingConfig`, `StructureIndexConfig`) to group related parameters.

## Error Handling

- Fail fast with clear, contextual errors using `do_critical(exc, message)` for fatal issues.
- Propagate errors with added context rather than silently swallowing them.
- Use domain-specific exceptions; never catch broad `Exception` without re-raising.

## Imports and Dependencies

- Use lazy imports (`importlib.import_module` or in-function imports) to avoid circular dependencies.
- Optional dependencies (numba, umap, POT, faiss, redis, duckdb) must be guarded with `try/except ImportError` and provide pure-Python fallbacks.
- Respect the dependency flow: `utils → data → metrics → embeddings → learning → topology → plotting`. Lower layers never import from upper layers.

## Code Organization

- Favor composition over inheritance; avoid global state and singletons.
- Keep functions focused — if a function exceeds ~150 lines, consider decomposition.
- Use factory patterns with `match/case` for creating classifiers, clusterers, and metrics.
- Unified API entry points (e.g., `generate_data`, `compute_pairwise_matrix`, `shape_distance`, `compute_embedding`) dispatch to specific implementations.
- Use `get_progress_bar()` from `utils/progress.py` instead of ad-hoc `tqdm` calls. Respects `NEURAL_ANALYSIS_QUIET` env var.
- Use `reproducible(seed)` context manager from `utils/reproducibility.py` for deterministic execution.
- Use `get_provenance()` from `utils/provenance.py` to embed version/platform info in saved results.
- Accept an `rng` parameter (int or numpy Generator) for seed management in stochastic functions.
