---
name: 'Storage System'
description: 'Rules for the three-layer storage stack: HDF5, DuckDB, Redis'
applyTo: 'src/neural_analysis/utils/storage/**,src/neural_analysis/utils/io.py,src/neural_analysis/utils/comparison_store.py'
---

# Storage Instructions

## Three-Layer Stack

```
┌─────────────────────────────────────┐
│         StorageManager              │  ← Unified API
│  save_data / load_data / query_data │
├──────────┬──────────┬───────────────┤
│  Redis   │  DuckDB  │    HDF5       │
│  (cache) │  (index) │  (persistent) │
│ Optional │ Optional │   Required    │
└──────────┴──────────┴───────────────┘
```

**Cascade:** Redis (fastest) → DuckDB (SQL queries) → HDF5 (authoritative).
**Graceful degradation:** Missing Redis or DuckDB never crashes; each layer checks `is_available()`.

## Rules

- Always use `with StorageManager() as sm:` to release resources.
- Respect the priority: HDF5 → DuckDB → Redis as configured in `StorageConfig`.
- Never handcraft Redis keys. Keys follow: `cache:{namespace}:{key}`.
- Persist pandas/NumPy with `save_result_to_hdf5_dataset` / `load_results_from_hdf5_dataset`.
- Use `save_comparison` / `load_comparison` / `try_load_cached_comparison` from comparison_store for pairwise results.
- Use `get_missing_comparisons` for incremental resume on long computations.

## HDF5 Schema

```
file.h5
├── {dataset_name}/
│   ├── {result_key}/
│   │   ├── [attributes]: scalar metadata
│   │   └── [datasets]: array data
```

Two concrete schemas exist:
- **Distribution comparisons:** `{di}_vs_{dj}_{metric}` in `distribution_comparisons.h5`
- **Structure index:** `nbins{n}_nneigh{k}` in `structure_indices.h5`

Compression: gzip level 6, shuffle filter, chunk threshold 10,000.

## DuckDB Metadata

Three tables: `datasets`, `comparisons`, `chunks`. Uses `INSERT OR REPLACE` for idempotency. JSON metadata column for flexible attributes.

## Redis Cache

- Pickle protocol (highest) for serialization.
- Size guard: rejects items exceeding `cache_max_size_mb` (default 100 MB).
- TTL default: 3600s. Auth via `redis_password`.

## Configuration

All settings via `NEURAL_ANALYSIS_*` environment variables:
`USE_REDIS`, `REDIS_HOST`, `REDIS_PORT`, `USE_SQL`, `SQL_PATH`, `CACHE_TTL`, `CACHE_NAMESPACE`.
