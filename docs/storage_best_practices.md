# Storage Best Practices

This repository follows the recommendations from the Redis community, in particular the guidance captured in [Redis Best Practices – Expert Tips for High Performance](https://www.dragonflydb.io/guides/redis-best-practices). The goal is to keep caching predictable, portable, and safe across notebooks, scripts, and services.

## Applied Redis Recommendations

| Principle | Implementation |
|-----------|----------------|
| Namespaced keys | `StorageConfig.cache_namespace` prefixes every Redis key so multiple experiments can share one server without collisions. |
| Deterministic TTLs | The existing `cache_ttl` value is honored everywhere, and all notebook benchmarks reset artifacts between runs to avoid stale data. |
| Secure/authenticated clients | `StorageConfig` already exposes host, port, DB, and password parameters; the new namespace field completes the minimal best‑practice set recommended in the article. |
| Resource cleanup | `RedisCache.close()` is called by `StorageManager.close()` (and via the context manager), ensuring that sockets are returned to the OS whenever a script/notebook finishes. |

## Layered Storage Defaults

1. **HDF5** stays the authoritative, lossless store for arrays, scalars, and metadata.
2. **DuckDB** indexes metadata for ad‑hoc SQL queries. The database file (`.duckdb`) persists on disk, so metadata is preserved across kernel restarts. Connections are "short‑lived" in the sense that they're opened per operation and closed via `SQLMetadata.close()`, but the indexed data remains on disk—no reload needed after restart.
3. **Redis** caches hot rows under a namespace for fast reloads. Since Redis runs as a separate service, cached data persists across kernel restarts (until TTL expires or the Redis server is restarted).
4. **StorageManager** orchestrates all three layers and can now be used as a context manager:

```python
from neural_analysis.utils.storage import StorageManager

with StorageManager() as sm:
    sm.save_data(...)
    rows = sm.query_data({"metric": "wasserstein"})
```

### Persistence Across Kernel Restarts

**What persists:**
- **HDF5 files**: All data remains on disk (source of truth)
- **DuckDB metadata**: The `.duckdb` file contains all indexed metadata. On kernel restart, you simply open a new connection—no re-indexing required.
- **Redis cache**: Hot rows remain cached (if Redis server is still running) until TTL expires

**What doesn't persist:**
- In-memory Python objects (DataFrames, arrays) are lost on kernel restart
- Active DuckDB/Redis connections are closed (but data remains)

**On kernel restart workflow:**
1. HDF5 files are already on disk (no action needed)
2. DuckDB metadata file exists—just query it (no re-indexing)
3. Redis cache may still have hot rows—first load checks cache, then falls back to HDF5 if miss
4. New connections are opened automatically when you use `StorageManager` or call `save_result_to_hdf5_dataset()`

Restarting a Python or Jupyter kernel frees every in‑process reference automatically. When long‑running notebooks are interrupted, calling `sm.close()` (or leaving the `with` block) closes DuckDB connections and Redis sockets, so no stray memory is held in the OS.

## Benchmarking & Recommended Profile

`examples/storage_demo.ipynb` now runs three rounds per storage profile (HDF5 only, HDF5+SQL, HDF5+Redis, and the full stack). The notebook aggregates average/standard deviation, selects the fastest loader, and reconfigures the global `StorageConfig` so subsequent cells (and any imported modules) run against that recommended profile automatically.

The helper `run_benchmark()` deletes old artifacts before each pass, reads summary statistics, and explicitly calls `gc.collect()` so every run starts from a clean state.

## Adding New Pandas Payloads

To store a new analysis result:

1. Prepare scalar metadata describing the payload (`dataset_i`, `dataset_j`, `metric`, etc.).
2. Package any NumPy arrays in a `dict[str, np.ndarray]`.
3. Call `save_result_to_hdf5_dataset()` (from `io.py`) with the HDF5 path, dataset name, result key, and the two dictionaries.

```python
from neural_analysis.utils.io import save_result_to_hdf5_dataset

save_result_to_hdf5_dataset(
    save_path="output/custom_results.h5",
    dataset_name="my_session",
    result_key="bins16_neighbors20",
    scalar_data={"metric": "wasserstein", "dataset_i": "a", "dataset_j": "b"},
    array_data={"values": values_array},
)
```

`io.py` owns every HDF5 read/write helper in the codebase. It normalizes attributes, handles DataFrame round‑trips, and now resolves a single `StorageManager` instance through `_resolve_storage_manager()`. That keeps the notebook, CLI scripts, and tests in sync and guarantees Redis/DuckDB indexing is always available when enabled.

When defining a new result format (for example, different DataFrame columns), you only need to:

1. Decide on the scalar attributes that should appear in DuckDB.
2. Save any NumPy arrays or DataFrame payloads through the existing helpers.
3. Reuse the benchmarking notebook (or import `run_benchmark`) to validate the new workload across the four storage profiles.

Because key naming, TTLs, and cache namespaces are now centralized in `StorageConfig`, additional result types automatically inherit the same best practices without any extra plumbing.

## Automatic Orchestration

The storage orchestration happens **automatically** and is **transparent to users**. When you call `save_result_to_hdf5_dataset()`:

1. **HDF5 write** happens first (always, as the source of truth)
2. **DuckDB indexing** happens automatically if `use_sql_index=True` (default)
3. **Redis caching** happens automatically if `use_cache=True` (default) and Redis is available

No additional user code is required. The function detects available backends and uses them gracefully:

```python
# This single call handles all three layers automatically
save_result_to_hdf5_dataset(
    save_path="results.h5",
    dataset_name="animal1/task1/velocity_data",
    result_key="analysis_v1",
    scalar_data={"metric": "wasserstein", "n_trials": 100},
    array_data={"values": velocity_array}
)
# ✅ Saved to HDF5
# ✅ Indexed in DuckDB (if enabled)
# ✅ Cached in Redis (if enabled and available)
```

On subsequent loads, the system automatically:
1. Checks Redis cache first (fastest)
2. Falls back to DuckDB metadata query (if cache miss)
3. Falls back to direct HDF5 read (if metadata unavailable)

See `examples/storage_demo.ipynb` for benchmarks showing the speedup from cached loads.

## Why Orchestration Matters: HDF5-Only vs. Multi-Layer

**HDF5-only is fastest for simple, single-user workflows** where:
- You always load entire datasets
- No need for ad-hoc queries across many files
- Minimal repeated access patterns

**However, for complex, multi-modality workflows** (e.g., `animal1/task1/neural_photon_data`, `animal1/task1/neural_probe_data`, `animal1/task1/position_data`, `animal1/task1/velocity_data`, `animal1/task2/velocity_data`, `animal2/task1/velocity_data`, etc.), orchestration provides critical benefits:

### 1. **Metadata Indexing (DuckDB)**
Without DuckDB, finding "all velocity_data comparisons for animal1" requires:
- Opening every HDF5 file
- Scanning all groups
- Filtering in Python

With DuckDB, you run a single SQL query:
```python
storage_manager.query_data({"dataset_i": "animal1", "modality": "velocity_data"})
```

### 2. **Selective Loading**
HDF5 supports chunking and partial reads, but you need to know exactly where data lives. DuckDB metadata tells you the `file_path` and `group_path` before you touch HDF5, enabling efficient slice loading.

### 3. **Cache Speedup**
Redis caches frequently accessed rows. In interactive notebooks, you might:
- Load `animal1/task1/velocity_data` → cache hit (instant)
- Load `animal1/task1/position_data` → cache miss, load from HDF5, cache for next time
- Reload `animal1/task1/velocity_data` → cache hit (instant)

Without Redis, every reload reads from disk.

### 4. **Concurrency Safety**
HDF5 file locking can cause contention when multiple workers write simultaneously. The orchestrated system:
- Writes metadata to DuckDB (fast, concurrent-safe)
- Caches in Redis (fast, concurrent-safe)
- Uses HDF5 as the durable store (single-writer per file, but metadata/cache reduce contention)

### 5. **Operational Best Practices**
Following Redis best practices (namespaced keys, TTLs, connection cleanup) ensures:
- Multiple experiments can share one Redis server
- Cache doesn't grow unbounded
- Resources are released properly

**Bottom line:** For simple workflows, HDF5-only is fine. For complex, multi-modality, slice-heavy workflows, the orchestrated stack provides query speed, cache hits, and operational safety that HDF5 alone cannot match.




