---
name: 'Metrics'
description: 'Conventions for distance, distribution, shape, and outlier metrics'
applyTo: 'src/neural_analysis/metrics/**'
---

# Metrics Instructions

## Module Layout

- `pairwise_metrics.py` — Facade re-exporting from `pairwise_core.py` + `pairwise_numba.py`
- `pairwise_core.py` — Core pairwise metric dispatcher, spatial autocorrelation, similarity matrices
- `pairwise_numba.py` — Numba-accelerated parallel implementations
- `distributions.py` — Shape distances (Procrustes, one-to-one, soft matching), distribution metrics (Wasserstein, KS, JSD)
- `outliers.py` — Five outlier detection methods via `match/case`

## API Conventions

- **`compute_pairwise_matrix(x, y, metric, parallel)`** — Unified entry point. Routes to the correct metric category automatically.
- **`compare_datasets(data, data2, mode, metric)`** — High-level orchestration: handles `within`/`between`/`all-pairs` with HDF5 caching.
- **`shape_distance(mtx1, mtx2, method, metric)`** — Main shape distance API. Uses automatic subsampling for large datasets.
- **`filter_outlier(data, method)`** — Unified outlier detection dispatcher.

## Adding a New Metric

1. Add the metric name to the appropriate typed constant set (`POINT_TO_POINT_METRICS`, `DISTRIBUTION_METRICS`, `SHAPE_METRICS`, or `SCALAR_METRICS`).
2. Implement the computation function in `pairwise_core.py`, matching existing signatures.
3. Add a `case "metric_name":` branch in the dispatcher or register it in the dispatch table.
4. Include Numba-accelerated parallel implementation in `pairwise_numba.py` where applicable (guarded by `try/except ImportError`).
5. Add tests covering normal inputs, edge cases, and comparison against a reference implementation.
6. Update `docs/function_registry.md`.

`MetricConfig` dataclass can be used to pass grouped metric parameters.

## Computation Patterns

- Use `run_with_subsampling(func, arrays, subsamples, repeats)` for expensive pairwise computations.
- Use `pairwise_distribution_comparison_batch(data, metrics)` for batch all-pairs with HDF5 persistence and incremental resume.
- Cache results via comparison_store: `try_load_cached_comparison` → compute → `save_comparison_result`.
- Matrix preprocessing: `modify_matrix(mtx, whiten, normalize, scale_variance)` before shape distances.
