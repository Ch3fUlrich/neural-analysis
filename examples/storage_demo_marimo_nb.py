import marimo

__generated_with = "0.18.3"

app = marimo.App(width="full")


@app.cell(hide_code=True)
def __():
    import marimo as mo

    return mo


@app.cell
def _(mo):  # noqa: N803
    mo.md(r"""
    # Storage Demo

    This notebook illustrates the new multi-layer storage workflow that combines Redis caching, DuckDB metadata indexing, and persistent HDF5 storage. We benchmark different configurations (plain Pandas + HDF5, +SQL indexing, +Redis caching, and the full stack) to measure write speed, reload speed, and on-disk footprint.
    """)
    return


@app.cell
def _():  # noqa: N803
    import gc
    from pathlib import Path
    from time import perf_counter

    import numpy as np
    import pandas as pd

    try:
        from neural_analysis.metrics.distributions import (
            pairwise_distribution_comparison_batch,
        )
    except ImportError:
        # Fallback for environments where the notebook is executed against an older install
        # (e.g., stale site-packages or a different checkout on Windows). We import the
        # module and fetch the attribute dynamically if it exists.
        from neural_analysis.metrics import distributions as _distributions

        if not hasattr(_distributions, "pairwise_distribution_comparison_batch"):
            raise

        pairwise_distribution_comparison_batch = (
            _distributions.pairwise_distribution_comparison_batch
        )

    from neural_analysis.utils.io import get_hdf5_result_summary
    from neural_analysis.utils.storage.config import StorageConfig, set_config

    rng = np.random.default_rng(42)
    datasets = {
        "condition_a": rng.normal(size=(1280, 800)),
        "condition_b": rng.normal(loc=0.75, size=(1280, 800)),
        "condition_c": rng.normal(loc=-0.75, size=(1280, 800)),
        "condition_d": rng.normal(loc=0.5, size=(1280, 800)),
        # "condition_e": rng.normal(loc=-0.5, size=(1280, 800)),
        # "condition_f": rng.normal(loc=0.25, size=(1280, 800)),
        # "condition_g": rng.normal(loc=-0.25, size=(1280, 800)),
        # "condition_h": rng.normal(loc=0.75, size=(1280, 800)),
        # "condition_i": rng.normal(loc=-0.75, size=(1280, 800)),
        # "condition_j": rng.normal(loc=0.5, size=(1280, 800)),
        # "condition_k": rng.normal(loc=-0.5, size=(1280, 800)),
    }

    output_dir = Path("output/storage_benchmarks")
    output_dir.mkdir(parents=True, exist_ok=True)
    return (
        StorageConfig,
        datasets,
        gc,
        get_hdf5_result_summary,
        output_dir,
        pairwise_distribution_comparison_batch,
        pd,
        perf_counter,
        set_config,
    )


@app.cell
def _(
    StorageConfig,
    datasets,
    gc,
    get_hdf5_result_summary,
    output_dir,
    pairwise_distribution_comparison_batch,
    perf_counter,
    set_config,
):
    def run_benchmark(
        label: str, use_cache: bool, use_sql_index: bool
    ) -> dict[str, float | str]:
        """Run write/load benchmark for a given storage configuration."""
        comparison_name = f"storage_demo_{label}"
        save_path = output_dir / f"{label}.h5"
        meta_path = output_dir / f"{label}.duckdb"

        if save_path.exists():
            save_path.unlink()
        if meta_path.exists():
            meta_path.unlink()

        storage_cfg = StorageConfig(
            use_redis=use_cache,
            use_sql=use_sql_index,
            redis_host="localhost",
            redis_port=6379,
            sql_path=meta_path,
        )
        set_config(storage_cfg)

        kwargs = dict(
            data=datasets,
            metrics={"wasserstein": {}, "procrustes": {}},
            comparison_name=comparison_name,
            save_path=save_path,
            progress=False,
            use_cache=use_cache,
            use_sql_index=use_sql_index,
        )

        write_start = perf_counter()
        pairwise_distribution_comparison_batch(**kwargs, regenerate=True)
        write_seconds = perf_counter() - write_start

        load_start = perf_counter()
        pairwise_distribution_comparison_batch(**kwargs, regenerate=False)
        load_seconds = perf_counter() - load_start

        file_mb = save_path.stat().st_size / (1024 * 1024)
        summary = get_hdf5_result_summary(save_path)
        rows = len(summary)
        del summary
        gc.collect()

        return {
            "label": label,
            "use_cache": use_cache,
            "use_sql_index": use_sql_index,
            "write_seconds": write_seconds,
            "load_seconds": load_seconds,
            "file_mb": file_mb,
            "rows": rows,
            "artifact_path": str(save_path),
        }

    bench_configs = [
        ("hdf5_only", False, False, "Pandas + HDF5"),
        ("hdf5_sql", False, True, "HDF5 + DuckDB metadata"),
        ("hdf5_redis", True, False, "HDF5 + Redis cache"),
        ("full_stack", True, True, "Redis + SQL + HDF5"),
    ]
    return bench_configs, run_benchmark


@app.cell
def _(StorageConfig, bench_configs, output_dir, pd, run_benchmark, set_config):  # noqa: N803
    BENCHMARK_RUNS = 3
    records = []
    for label, use_cache, use_sql, description in bench_configs:
        for run_idx in range(1, BENCHMARK_RUNS + 1):
            result = run_benchmark(label, use_cache, use_sql)
            result["description"] = description
            result["run"] = run_idx
            records.append(result)
    benchmark_df = pd.DataFrame(records)
    agg_df = (
        benchmark_df.groupby(
            ["label", "description", "use_cache", "use_sql_index"], as_index=False
        )
        .agg(
            write_seconds_mean=("write_seconds", "mean"),
            write_seconds_std=("write_seconds", "std"),
            load_seconds_mean=("load_seconds", "mean"),
            load_seconds_std=("load_seconds", "std"),
            file_mb_mean=("file_mb", "mean"),
            rows_mean=("rows", "mean"),
        )
        .sort_values("load_seconds_mean")
    )
    baseline_write = agg_df.loc[
        agg_df["label"] == "hdf5_only", "write_seconds_mean"
    ].iloc[0]
    baseline_load = agg_df.loc[
        agg_df["label"] == "hdf5_only", "load_seconds_mean"
    ].iloc[0]
    agg_df["write_speedup_vs_hdf5"] = baseline_write / agg_df["write_seconds_mean"]
    agg_df["load_speedup_vs_hdf5"] = baseline_load / agg_df["load_seconds_mean"]
    recommended_row = agg_df.iloc[0]
    recommended_label = recommended_row["label"]
    _recommended_use_cache = bool(recommended_row["use_cache"])
    recommended_use_sql = bool(recommended_row["use_sql_index"])
    recommended_cfg = StorageConfig(
        use_redis=_recommended_use_cache,
        use_sql=recommended_use_sql,
        redis_host="localhost",
        redis_port=6379,
        sql_path=output_dir / f"{recommended_label}.duckdb",
    )
    set_config(recommended_cfg)
    run_benchmark(recommended_label, _recommended_use_cache, recommended_use_sql)
    best_path = output_dir / f"{recommended_label}.h5"
    # Materialize the recommended artifact so downstream cells can inspect it
    agg_df
    return (best_path,)


@app.cell
def _(mo):  # noqa: N803
    mo.md(r"""
    ### Recommended configuration

    The table above is averaged across three independent runs per storage profile. The winning profile is stored in `recommended_row` and its artifacts were regenerated automatically. You can now reuse this profile globally:

    ```python
    recommended_cfg
    ```
    """)
    return


@app.cell
def _(best_path, get_hdf5_result_summary):  # noqa: N803
    best_summary = get_hdf5_result_summary(best_path)
    best_summary[["dataset_i", "dataset_j", "metric", "value"]].head()
    return


@app.cell
def _(mo):  # noqa: N803
    mo.md(r"""
    ## Automatic Orchestration & Cache Speedup

    The storage system automatically orchestrates HDF5, DuckDB, and Redis without requiring any additional user code. The following example demonstrates the cache speedup on repeated loads:
    """)
    return


@app.cell
def _(best_path, gc, get_hdf5_result_summary, perf_counter):  # noqa: N803
    _recommended_use_cache = True
    if _recommended_use_cache:
        from neural_analysis.utils.storage.manager import StorageManager

        with StorageManager() as sm:
            print("First load (cache miss - reads from HDF5, caches result):")
            first_start = perf_counter()
            first_summary = get_hdf5_result_summary(best_path)
            first_time = perf_counter() - first_start
            print(f"  Time: {first_time:.4f} seconds")
            print(f"  Rows loaded: {len(first_summary)}")
            cache_key = f"demo_summary:{best_path}"
            sm.cache_set(cache_key, first_summary, ttl=300)
            print("  ✅ Cached summary data")
            del first_summary
            gc.collect()
            print("\nSecond load (cache hit - reads from Redis):")
            second_start = perf_counter()
            cached_summary = sm.cache_get(cache_key)
            second_time = perf_counter() - second_start
            print(f"  Time: {second_time:.4f} seconds")
            print(
                f"  Rows loaded: {(len(cached_summary) if cached_summary is not None else 0)}"
            )
            if cached_summary is not None and second_time > 0:
                speedup = first_time / second_time
                print(f"\n🚀 Cache speedup: {speedup:.2f}x faster")
                print(f"   ({first_time * 1000:.2f}ms → {second_time * 1000:.2f}ms)")
            else:
                print("\n⚠️  Cache miss or load too fast to measure")
            print("\n📝 Note: In practice, pairwise_distribution_comparison_batch()")
            print("   automatically caches results when use_cache=True.")
            print(
                "   The cache persists across kernel restarts (if Redis server is running)."
            )
    else:
        print("⚠️  Redis caching is disabled in recommended config.")
        print("   Enable use_redis=True to see cache speedup benefits.")
    return


@app.cell
def _(mo):  # noqa: N803
    mo.md(r"""
    ### Persistence Across Kernel Restarts

    **Important:** The orchestration is fully automatic and transparent:

    1. **On first save:** Data is written to HDF5, indexed in DuckDB (if enabled), and cached in Redis (if enabled)
    2. **On subsequent loads:** System checks Redis cache first, then DuckDB metadata, then HDF5
    3. **After kernel restart:**
       - HDF5 files persist (source of truth)
       - DuckDB metadata file persists (`.duckdb` file on disk) - **no re-indexing needed**
       - Redis cache persists (if Redis server is still running) - **hot rows remain cached**

    You don't need to reload data into DuckDB after a kernel restart. The `.duckdb` file already contains all indexed metadata—just query it. Redis cache also persists across restarts (until TTL expires), so frequently accessed data loads instantly.
    """)
    return


@app.cell
def _(best_path, get_hdf5_result_summary):  # noqa: N803
    summary = get_hdf5_result_summary(best_path)
    summary["value"].describe()
    return
