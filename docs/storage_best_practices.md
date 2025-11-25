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
2. **DuckDB** indexes metadata for ad‑hoc SQL queries. Connections are now short‑lived and released through `SQLMetadata.close()`.
3. **Redis** caches hot rows under a namespace for fast reloads.
4. **StorageManager** orchestrates all three layers and can now be used as a context manager:

```python
from neural_analysis.utils.storage import StorageManager

with StorageManager() as sm:
    sm.save_data(...)
    rows = sm.query_data({"metric": "wasserstein"})
```

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




