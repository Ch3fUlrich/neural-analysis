<!-- 79900d25-d9aa-43fb-8044-0460db2ba40d 2cf1afde-a995-4ffc-98b2-2a8d9b31d7a1 -->
# Database Integration System for Fast I/O

## Overview

Create a multi-layer storage architecture combining Redis (in-memory cache), SQL database (metadata indexing), and HDF5 (persistent storage) to dramatically improve read/write performance for large neural datasets. The system must gracefully fallback to HDF5-only mode when Redis or SQL are unavailable.

## Architecture

### Layer 1: Redis Cache (Fastest - Optional)

- **Purpose**: In-memory caching for hot/intermediate data
- **TTL**: Configurable expiration (default: 1 hour)
- **Serialization**: Pickle or msgpack for numpy/pandas objects
- **Fallback**: Skip if Redis unavailable, continue to SQL/HDF5

### Layer 2: SQL Metadata Index (Fast - Optional)  

- **Purpose**: Fast queries for dataset locations, metadata filtering
- **Database**: DuckDB (embedded, no server required)
- **Tables**: datasets, comparisons, chunks, experiments
- **Fallback**: Skip if SQL unavailable, query HDF5 directly

### Layer 3: HDF5 Storage (Persistent - Required)

- **Purpose**: Compressed, hierarchical data storage
- **Current**: Already implemented in `io.py`
- **Always Available**: Core storage layer, no fallback needed

## Implementation Plan

### Phase 1: Core Infrastructure (Priority: CRITICAL)

#### 1.1 Create Storage Backend Abstraction

**File**: `src/neural_analysis/utils/storage/__init__.py`

- Create abstract base classes for storage backends
- Define unified interface: `save()`, `load()`, `query()`, `exists()`
- Support for: Redis, SQL, HDF5

#### 1.2 Redis Cache Module

**File**: `src/neural_analysis/utils/storage/redis_cache.py`

- Wrapper for Redis operations with graceful degradation
- Functions:
- `get_cached(key: str) -> Any | None`
- `set_cached(key: str, value: Any, ttl: int = 3600) -> bool`
- `invalidate_cache(pattern: str) -> int`
- `is_available() -> bool`
- Key naming: `cache:{dataset_name}:{result_key}:{slice_info}`
- Serialization: pickle for numpy/pandas (with size limits)

#### 1.3 SQL Metadata Module  

**File**: `src/neural_analysis/utils/storage/sql_metadata.py`

- DuckDB-based metadata indexing
- Tables:
- `datasets`: id, file_path, group_path, created_at, metadata_json
- `comparisons`: id, dataset_i, dataset_j, metric, mode, file_path, group_path, value_type
- `chunks`: dataset_id, chunk_key, file_path, offset, size, compression
- Functions:
- `index_dataset(file_path, group_path, metadata) -> str`
- `query_datasets(filters: dict) -> pd.DataFrame`
- `query_comparisons(filters: dict) -> pd.DataFrame`
- `is_available() -> bool`

#### 1.4 Unified Storage Manager

**File**: `src/neural_analysis/utils/storage/manager.py`

- Orchestrates Redis -> SQL -> HDF5 access pattern
- Functions:
- `save_data(key, data, metadata, use_cache=True) -> None`
- `load_data(key, use_cache=True) -> Any`
- `query_data(filters, use_sql=True) -> pd.DataFrame`
- Automatic fallback logic:

1. Try Redis cache (if enabled and available)
2. Try SQL metadata query (if enabled and available)  
3. Fallback to HDF5 direct access (always works)

### Phase 2: Update Existing I/O Functions

#### 2.1 Enhance `io.py` Functions

**File**: `src/neural_analysis/utils/io.py`

- Update `save_hdf5()`: Add optional cache/SQL indexing
- Update `load_hdf5()`: Check cache first, then SQL, then HDF5
- Update `save_result_to_hdf5_dataset()`: Index in SQL, cache in Redis
- Update `load_results_from_hdf5_dataset()`: Query SQL for fast filtering
- Maintain backward compatibility: All functions work without Redis/SQL

#### 2.2 Enhance `comparison_store.py`

**File**: `src/neural_analysis/utils/comparison_store.py`

- Update `save_comparison()`: Use storage manager
- Update `load_comparison()`: Use storage manager with cache
- Update `query_comparisons()`: Use SQL for fast queries
- Add cache invalidation on overwrite

### Phase 3: Configuration & Dependencies

#### 3.1 Add Optional Dependencies

**File**: `pyproject.toml`

- Add to `[project.optional-dependencies]`:
- `storage = ["redis>=5.0", "duckdb>=0.10"]`
- Keep Redis and DuckDB as optional (not required)

#### 3.2 Configuration System

**File**: `src/neural_analysis/utils/storage/config.py`

- Environment variables or config file:
- `NEURAL_ANALYSIS_USE_REDIS`: Enable Redis (default: auto-detect)
- `NEURAL_ANALYSIS_USE_SQL`: Enable SQL (default: auto-detect)
- `NEURAL_ANALYSIS_REDIS_HOST`: Redis host (default: localhost)
- `NEURAL_ANALYSIS_REDIS_PORT`: Redis port (default: 6379)
- `NEURAL_ANALYSIS_SQL_PATH`: SQL database path (default: `.neural_analysis_meta.db`)
- `NEURAL_ANALYSIS_CACHE_TTL`: Cache TTL in seconds (default: 3600)

### Phase 4: Integration Points

#### 4.1 Update Batch Comparison Functions

**Files**: `src/neural_analysis/metrics/distributions.py`, `src/neural_analysis/metrics/pairwise_metrics.py`

- Update `pairwise_distribution_comparison_batch()` to use storage manager
- Cache intermediate results in Redis
- Index results in SQL for fast querying

#### 4.2 Update Structure Index Functions

**File**: `src/neural_analysis/topology/structure_index.py`

- Update `compute_structure_index_sweep()` to use storage manager
- Cache parameter sweeps in Redis
- Index results in SQL

### Phase 5: Testing & Documentation

#### 5.1 Comprehensive Tests

**File**: `tests/test_storage_*.py`

- Test Redis cache (with and without Redis available)
- Test SQL metadata (with and without DuckDB available)
- Test fallback to HDF5-only mode
- Test cache invalidation
- Test concurrent access
- Integration tests for full storage manager

#### 5.2 Example Notebook

**File**: `examples/storage_demo.ipynb`

- Demonstrate Redis caching
- Demonstrate SQL metadata queries
- Show performance comparisons
- Show fallback behavior

#### 5.3 Documentation

- Update `README.md` with storage architecture
- Document configuration options
- Migration guide from HDF5-only to multi-layer

## Key Design Principles

1. **Graceful Degradation**: Always fallback to HDF5 if Redis/SQL unavailable
2. **Backward Compatibility**: All existing code works without changes
3. **Optional Dependencies**: Redis and DuckDB are optional, not required
4. **Modular Design**: Each storage layer is independent and testable
5. **Performance**: Cache hot data, index metadata, compress cold data
6. **Type Safety**: Full type hints, mypy compliance

## Files to Create

- `src/neural_analysis/utils/storage/__init__.py`
- `src/neural_analysis/utils/storage/redis_cache.py`
- `src/neural_analysis/utils/storage/sql_metadata.py`
- `src/neural_analysis/utils/storage/manager.py`
- `src/neural_analysis/utils/storage/config.py`
- `tests/test_storage_redis.py`
- `tests/test_storage_sql.py`
- `tests/test_storage_manager.py`
- `examples/storage_demo.ipynb`
- `Dockerfile` (development container)
- `docker-compose.yml` (multi-container setup)
- `scripts/setup_docker.sh` (setup script)
- `scripts/dev_docker.sh` (development helper script)
- `.dockerignore` (exclude unnecessary files)
- `docs/docker_setup.md` (Docker setup guide)

## Files to Modify

- `pyproject.toml` (add optional dependencies)
- `src/neural_analysis/utils/io.py` (integrate storage manager)
- `src/neural_analysis/utils/comparison_store.py` (use storage manager)
- `src/neural_analysis/metrics/distributions.py` (use storage manager)
- `src/neural_analysis/metrics/pairwise_metrics.py` (use storage manager)
- `src/neural_analysis/topology/structure_index.py` (use storage manager)
- `TODO.md` (add this task)
- `todo/todo_integrate_databases.md` (improve with codebase-specific details)

## Estimated Effort

- Phase 1 (Core Infrastructure): 15-20 hours
- Phase 2 (Update I/O Functions): 8-12 hours
- Phase 3 (Configuration): 3-5 hours
- Phase 4 (Integration): 6-10 hours
- Phase 5 (Testing & Docs): 10-15 hours
- **Total**: 42-62 hours (1-2 weeks full-time, 3-4 weeks part-time)

### To-dos

- [ ] Create storage backend abstraction module with abstract base classes and unified interface
- [ ] Implement Redis cache module with graceful degradation and pickle serialization
- [ ] Implement SQL metadata module using DuckDB with tables for datasets, comparisons, chunks
- [ ] Create unified storage manager that orchestrates Redis -> SQL -> HDF5 access with fallback
- [ ] Add optional dependencies (redis, duckdb) to pyproject.toml
- [ ] Create configuration system for storage settings (env vars, config file)
- [ ] Update io.py functions (save_hdf5, load_hdf5, save_result_to_hdf5_dataset) to use storage manager
- [ ] Update comparison_store.py functions to use storage manager with caching
- [ ] Update batch comparison functions in distributions.py and pairwise_metrics.py
- [ ] Update structure_index.py functions to use storage manager
- [ ] Write comprehensive tests for Redis cache (with/without Redis available)
- [ ] Write comprehensive tests for SQL metadata (with/without DuckDB available)
- [ ] Write integration tests for storage manager and fallback behavior
- [ ] Create example notebook demonstrating storage system usage and performance
- [ ] Update TODO.md with database integration task and reference todo_integrate_databases.md
- [ ] Improve todo_integrate_databases.md with codebase-specific implementation details