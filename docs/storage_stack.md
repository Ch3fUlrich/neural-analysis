# Multi-Layer Storage Stack

This document explains how neural-analysis persists large experiment outputs using a hybrid stack:

1. **HDF5** — canonical, lossless storage of arrays and scalar metadata.
2. **DuckDB** — fast, file-based SQL index that mirrors HDF5 group/attribute metadata for instant queries.
3. **Redis** — optional in-memory cache for hot results and repeated comparison batches.
4. **StorageManager** — orchestrates every read/write, cascading through Redis → DuckDB → HDF5 with graceful fallbacks.

The system guarantees deterministic writes even when Redis or DuckDB are offline. Each layer is optional but enabled by default whenever the dependency is installed.

---

## Architecture Overview

```
client code (metrics, io, notebooks)
        │
        ▼
StorageManager (config-driven)
        │─────────────┬─────────────┐
        ▼             ▼             ▼
 Redis cache    DuckDB metadata    HDF5 file
(hot rows)      (index/search)     (ground truth)
```

### StorageManager responsibilities

- Normalizes `StorageConfig` (environment variables or manual instantiation).
- Saves scalar metadata as HDF5 attributes, arrays as datasets, and mirrors both into DuckDB tables (`datasets`, `comparisons`, `chunks`).
- Provides cache helpers (`cache_get`, `cache_set`, `invalidate_cache`) for higher-level modules and notebooks.
- Ensures every public API can operate with *only* HDF5 present so CI/air-gapped environments stay functional.

---

## Configuration

The `StorageConfig` class reads sensible defaults but every field can be overridden:

| Environment variable | Purpose | Default |
|----------------------|---------|---------|
| `NEURAL_ANALYSIS_USE_REDIS` | Enable Redis caching | `true` (if `redis` installed) |
| `NEURAL_ANALYSIS_REDIS_HOST`, `PORT`, `DB`, `PASSWORD` | Redis connection | `localhost`, `6379`, `0`, empty |
| `NEURAL_ANALYSIS_USE_SQL` | Enable DuckDB metadata | `true` |
| `NEURAL_ANALYSIS_SQL_PATH` | DuckDB file path | `.neural_analysis_meta.db` |
| `NEURAL_ANALYSIS_CACHE_TTL` | Redis TTL in seconds | `3600` |
| `NEURAL_ANALYSIS_CACHE_MAX_SIZE_MB` | Max serialized payload | `100` |

Programmatic override example:

```python
from neural_analysis.utils.storage.config import StorageConfig, set_config

set_config(
    StorageConfig(
        use_redis=True,
        redis_host="localhost",
        redis_port=6379,
        use_sql=True,
        sql_path="output/storage_benchmarks/full_stack.duckdb",
        cache_ttl=1800,
    )
)
```

---

## Running the Storage Benchmark

The notebook `examples/storage_demo.ipynb` produces repeatable metrics comparing:

1. Pandas + direct HDF5
2. HDF5 + DuckDB metadata
3. HDF5 + Redis cache
4. Full stack (Redis + DuckDB + HDF5)

Steps:

1. Install dependencies (`uv sync --all-extras` so `redis` and `duckdb` are available).
2. Ensure Redis is running locally (or via Docker, see below).
3. Launch Jupyter: `uv run jupyter lab`.
4. Open `examples/storage_demo.ipynb` and execute all cells.

The notebook:

- Reconfigures `StorageConfig` before each run.
- Writes pairwise comparison batches to separate artifacts (e.g., `output/storage_benchmarks/full_stack.h5` and `.duckdb`).
- Measures write time, reload time, file size, and row counts.
- Computes relative speedups vs. the pure HDF5 baseline.

Use the resulting DataFrame to capture the performance guarantees requested in the storage demo requirements.

---

## Dockerized Reference Stack

The repo ships with a development-friendly Compose file:

```
docker-compose.yml
├─ app      (Python + uv environment)
├─ redis    (alpine image)
└─ db       (PostgreSQL placeholder for future metadata)
```

Helper scripts:

```bash
./scripts/setup_docker.sh  # builds the image and starts redis/db/app
./scripts/dev_docker.sh    # opens a shell inside the app container
```

Within the container:

```bash
uv sync --all-extras
uv run pytest
uv run mypy
```

The container is configured with `REDIS_HOST=redis`, `REDIS_PORT=6379`, and `SQL_DB_PATH=/app/metadata.duckdb`, so the default `StorageConfig` immediately points at the services launched by Compose. This ensures integration benchmarks or Redis-backed tests never silently fall back to “HDF5-only” mode unless an operator disables them explicitly.

---

## Updating Documentation & Results

- Record benchmark tables or plots inside `examples/storage_demo.ipynb` so reviewers can verify speed/space savings.
- If you introduce a new storage backend (e.g., S3, parquet), expand this document with another layer and note which APIs were adapted.
- For reproducibility, commit **both** the DuckDB metadata file path and the HDF5 artifact path used in benchmarks (or describe how to regenerate them) rather than the binary data itself.

This document should be referenced from the main README and any onboarding docs so new contributors understand how Redis/DuckDB/HDF5 cooperate.




