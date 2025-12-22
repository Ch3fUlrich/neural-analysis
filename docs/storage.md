# `docs/storage.md` (new, merged)

# Storage System Overview

The storage stack uses three coordinated layers:

- **HDF5** – authoritative, lossless store for arrays, scalars, and metadata.
- **DuckDB** – disk‑backed metadata index for fast, SQL‑like queries.
- **Redis** – in‑memory cache for hot rows, namespaced by experiment.
- **StorageManager** – orchestrator that hides layer details behind a single API.

All helpers and defaults are designed so notebooks, scripts, and services share the same behaviour without extra wiring.

## 1. HDF5 Layout

### 1.1 General pattern

```yaml
file.h5
├── dataset_name/                 # Top-level group (session / dataset / comparison)
│   ├── result_key/               # Result identifier (parameters / pair)
│   │   ├── @attribute_1          # Scalar metadata (HDF5 attributes)
│   │   ├── @attribute_2
│   │   ├── array_1               # Array dataset
│   │   └── array_2               # Array dataset
└── ...
```

Key concepts:

- **Dataset name** – top‑level group (session, dataset, comparison group).
- **Result key** – unique identifier per analysis result.
- **Attributes** – scalar metadata stored as HDF5 attributes.
- **Datasets** – large numerical arrays stored as HDF5 datasets.

---

### 1.2 Distribution comparisons

**Default path**: `./output/distribution_comparisons.h5`

**Result key format**:  
`{dataset_i}_vs_{dataset_j}_{metric}`  
Examples: `conditionA_vs_conditionB_wasserstein`, `session1_vs_session2_ks`.

**Attributes (scalars)**

| Name           | Meaning                             |
|----------------|-------------------------------------|
| `dataset_i`    | Name of first dataset               |
| `dataset_j`    | Name of second dataset              |
| `metric`       | Distance / similarity metric        |
| `value`        | Computed metric value               |
| `n_samples_i`  | Number of samples (dataset i)       |
| `n_samples_j`  | Number of samples (dataset j)       |
| `n_features_i` | Feature count (dataset i)           |
| `n_features_j` | Feature count (dataset j)           |

**Datasets (arrays; optional, metric‑dependent)**

| Name           | Shape          | Meaning                          |
|----------------|----------------|----------------------------------|
| `pair_indices` | `[n_pairs, 2]` | Point‑to‑point correspondence    |
| `pair_values`  | `[n_pairs]`    | Correspondence scores / weights  |

---

### 1.3 Structure index results

**Default path**: `./output/structure_indices.h5`

**Result key format**:  
`nbins{n_bins}_nneigh{n_neighbors}`  
Examples: `nbins10_nneigh15`, `nbins20_nneigh30`.

**Attributes (scalars)**

| Name             | Meaning                          |
|------------------|----------------------------------|
| `structure_index`| Main SI value                    |
| `n_bins`         | Number of bins per dimension     |
| `n_neighbors`    | k‑NN parameter                   |
| `distance_metric`| Distance metric used             |
| `num_shuffles`   | Number of shuffles               |
| `n_samples`      | Number of data points            |
| `n_features`     | Neural dimensionality            |
| `label_dims`     | Behavioural dimensionality       |

**Datasets (arrays)**

| Name             | Shape           | Meaning                          |
|------------------|-----------------|----------------------------------|
| `overlap_matrix` | `[n_bins, n_bins]` | Bin‑to‑bin overlap matrix    |
| `shuffled_si`    | `[num_shuffles]`   | Null distribution from shuffles |

---

## 2. Core Helpers and APIs

### 2.1 HDF5 helpers

All HDF5 read/write operations should go through the I/O helpers in `neural_analysis.utils.io`.

**Saving**

```python
from neural_analysis.utils.io import save_result_to_hdf5_dataset

save_result_to_hdf5_dataset(
    save_path="output/custom_results.h5",
    dataset_name="session_001",
    result_key="bins16_neighbors20",
    scalar_data={"metric": "wasserstein", "dataset_i": "a", "dataset_j": "b"},
    array_data={"values": values_array},
)
```

This single call:

1. Writes to HDF5 (source of truth).
2. Updates DuckDB metadata index (if enabled).
3. Updates Redis cache (if enabled and available).

**Loading**

```python
from neural_analysis.utils.io import load_results_from_hdf5_dataset

results = load_results_from_hdf5_dataset(
    "output/structure_indices.h5",
    dataset_name="session_001",
    result_key="nbins10_nneigh15",
)
```

Use `get_hdf5_result_summary` and similar helpers when you need a pandas summary rather than raw arrays.

---

### 2.2 StorageManager

`StorageManager` is the main entry point for orchestrated storage.

```python
from neural_analysis.utils.storage import StorageManager

with StorageManager() as sm:
    sm.save_data(...)
    rows = sm.query_data({"metric": "wasserstein"})
```

Responsibilities:

- Expose a single API for saving/loading/querying results.
- Manage HDF5, DuckDB, and Redis backends.
- Ensure connections and sockets are cleaned up (`close()` or context manager).

Always prefer `with StorageManager() as sm:` over manually constructing DuckDB or Redis clients.

---

## 3. DuckDB Metadata Index

DuckDB tracks scalar metadata across results to make querying fast and persistent.

### 3.1 What is indexed

Typical fields:

- Dataset identifiers and modalities.
- Metric names and parameter combinations.
- Paths into HDF5 (file path, group path, result key).

### 3.2 Usage patterns

- Use `StorageManager.query_data(...)` or summary helpers to locate relevant results quickly.
- After kernel restart, the `.duckdb` file is still on disk; open a new connection and query immediately—no re‑indexing required.

Example pattern:

```
rows = sm.query_data({"dataset_i": "animal1", "modality": "velocity_data"})
```

---

## 4. Redis Cache Best Practices

Redis is used as a namespaced cache for frequently accessed rows.

### 4.1 Principles

| Principle           | Implementation detail                                       |
|---------------------|-------------------------------------------------------------|
| Namespaced keys     | `StorageConfig.cache_namespace` prefixes every key          |
| Predictable TTLs    | A single `cache_ttl` is honoured consistently               |
| Authenticated client| Host, port, DB, password configurable via `StorageConfig`   |
| Resource cleanup    | `StorageManager.close()` closes Redis connections           |

### 4.2 Behaviour

- Loads check Redis first, then DuckDB, then HDF5.
- Cached rows survive kernel restarts as long as Redis runs and TTL has not expired.
- Key naming and TTLs are centralised in configuration; do not build custom keys.

---

## 5. Persistence Across Kernel Restarts

### 5.1 What persists

- **HDF5 files** – complete data, on disk.  
- **DuckDB metadata** – `.duckdb` file on disk.  
- **Redis cache** – hot rows, as long as the Redis server and TTL allow.

### 5.2 What does not persist

- In‑memory Python objects (arrays, DataFrames).
- Active DuckDB and Redis connections.

### 5.3 Recommended restart workflow

1. Re‑create your `StorageManager` (or call helpers that resolve it).
2. Query DuckDB to discover results.
3. Load from Redis/HDF5 as needed; no manual re‑indexing.

---

## 6. HDF5‑Only vs Orchestrated Stack

**HDF5‑only is adequate when:**

- Single user, simple workflows.
- Always loading full datasets from a small number of files.
- Minimal repeated access or cross‑file querying.

**The full stack is recommended when:**

- Many datasets, sessions, and modalities.
- Frequent cross‑file or cross‑session queries.
- Heavy interactive use (e.g. notebooks) where caching saves time.
- Multiple workers / processes reading the same results.

Orchestration gives:

- Fast metadata queries (DuckDB).
- Cache hits for repeated loads (Redis).
- Controlled, safe use of HDF5 as the durable store.

---

## 7. Design Principles Summary

- **Hierarchical organisation** – groups for datasets, keys for results.  
- **Scalars as attributes** – filter and query without loading arrays.  
- **Arrays as datasets** – compressed, chunked, lazily loadable.  
- **Consistent naming** – predictable result_key and attribute names.  
- **Incremental updates** – append results; avoid rewriting whole files.

For implementation details and examples, see `examples/storage_demo.ipynb`.