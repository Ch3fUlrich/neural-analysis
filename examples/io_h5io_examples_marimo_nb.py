import marimo

__generated_with = "0.18.3"

app = marimo.App(width="full")


@app.cell(hide_code=True)
def __():
    import marimo as mo

    return mo


@app.cell
def _(mo):
    mo.md(r"""
    # HDF5 I/O with h5io

    This notebook demonstrates how to save and load arrays and DataFrames using `neural_analysis.utils.h5io` and how to filter pairs when loading via `load_hdf5`.
    """)
    return


@app.cell
def _():
    # Imports and helpers
    import tempfile
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from neural_analysis.utils import h5io
    from neural_analysis.utils.io import load_hdf5

    tmpdir = tempfile.TemporaryDirectory()
    base = Path(tmpdir.name)
    print("Using temp dir:", base)
    return base, h5io, load_hdf5, np, pd, tmpdir


@app.cell
def _(base, h5io, np):
    # Example 1: Array roundtrip with labels and attrs
    path = base / "array_demo.h5"
    data = np.random.randn(100, 10).astype(np.float32)
    labels = np.array([f"sample_{i}" for i in range(data.shape[0])])
    _attrs = {
        "description": "random normal features",
        "version": 1,
        "metadata": {"source": "synthetic", "dims": list(data.shape)},
    }
    h5io(path, task="save", data=data, labels=labels, attrs=_attrs)
    print("Saved to", path)
    loaded_data, loaded_labels = h5io(path, task="load")
    assert isinstance(loaded_data, np.ndarray)
    np.testing.assert_allclose(loaded_data, data)
    # Save
    assert list(loaded_labels) == list(labels)
    # Load
    # Validate roundtrip
    print("Array roundtrip OK:", loaded_data.shape)
    return


@app.cell
def _(base, h5io, np, pd):
    # Example 2: DataFrame roundtrip
    path_df = base / "df_demo.h5"
    _df = pd.DataFrame(
        {
            "neuron_id": [f"n{i}" for i in range(5)],
            "firing_rate": np.random.rand(5),
            "condition": ["A", "B", "A", "B", "A"],
        }
    )
    labels_df = ["trial_1", "trial_2", "trial_3", "trial_4", "trial_5"]
    h5io(path_df, task="save", data=_df, labels=labels_df)
    loaded_df, loaded_labels_df = h5io(path_df, task="load")
    assert isinstance(loaded_df, pd.DataFrame)
    pd.testing.assert_frame_equal(
        loaded_df.reset_index(drop=True), _df.reset_index(drop=True)
    )
    assert list(loaded_labels_df) == labels_df
    # Validate
    print("DataFrame roundtrip OK:", loaded_df.shape)
    return


@app.cell
def _(base, display, h5io, load_hdf5, pd):
    # Example 3: Filtering pairs on load (via load_hdf5)
    path_pairs = base / "pairs_demo.h5"
    pairs_df = pd.DataFrame(
        {
            "item_i": ["A", "A", "B", "C"],
            "item_j": ["B", "C", "C", "D"],
            "score": [0.1, 0.8, 0.5, 0.9],
        }
    )
    h5io(path_pairs, task="save", data=pairs_df, labels=None)
    wanted = [("A", "C"), ("B", "C")]
    (loaded_filtered, _), _attrs = load_hdf5(
        path_pairs, filter_pairs=wanted, return_attrs=True
    )
    print("Filtered rows:")
    display(loaded_filtered)
    assert set(zip(loaded_filtered["item_i"], loaded_filtered["item_j"])) == set(wanted)
    # Validate only desired pairs
    print("Filter pairs OK")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Advanced HDF5 Operations

    The following examples demonstrate the hierarchical HDF5 structure and advanced functions for saving/loading result datasets with mixed scalar and array data.
    """)
    return


@app.cell
def _(base, np):
    # Example 4: save_result_to_hdf5_dataset with mixed data types
    from neural_analysis.utils.io import (
        get_hdf5_dataset_names,
        get_hdf5_result_summary,
        load_results_from_hdf5_dataset,
        save_result_to_hdf5_dataset,
    )

    hdf5_path = base / "hierarchical_results.h5"
    for session in ["session_001", "session_002"]:
        for condition_pair in [
            ("condA", "condB"),
            ("condA", "condC"),
            ("condB", "condC"),
        ]:
            _result_key = f"{condition_pair[0]}_vs_{condition_pair[1]}_wasserstein"
            value = np.random.rand() * 100
            n_pairs = 50
            # Create a hierarchical HDF5 file with multiple result datasets
            pair_indices = [(i, j) for i in range(10) for j in range(10) if i < j][
                :n_pairs
            ]
            pair_values = np.random.rand(n_pairs)
            # Save multiple results under different comparison groups
            _pairs_indices = np.array(pair_indices, dtype=np.int32)
            pairs_vals = np.array(pair_values, dtype=np.float64)
            save_result_to_hdf5_dataset(
                save_path=hdf5_path,
                dataset_name=session,
                result_key=_result_key,
                scalar_data={
                    "dataset_i": condition_pair[0],
                    "dataset_j": condition_pair[1],
                    "metric": "wasserstein",
                    "value": float(value),
                    "n_samples_i": 100,
                    "n_samples_j": 100,
                    "timestamp": "2024-01-15",
                },
                array_data={
                    "pairs_indices": _pairs_indices,
                    "pairs_values": pairs_vals,
                },
            )  # Generate sample results
    print(f"✓ Saved hierarchical results to {hdf5_path.name}")
    print(
        "  Structure: dataset_name / result_key / {scalar_data, array_data}"
    )  # Scalar metric  # Generate pairwise comparison data (could be large arrays)  # Create structured array that HDF5 can handle  # Store as separate arrays: indices and values  # Save with scalar attributes and array datasets
    return (
        get_hdf5_dataset_names,
        get_hdf5_result_summary,
        hdf5_path,
        load_results_from_hdf5_dataset,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ### Viewing Dataset Names

    Use `get_hdf5_dataset_names()` to list all comparison groups and result keys in the HDF5 file.
    """)
    return


@app.cell
def _(get_hdf5_dataset_names, hdf5_path):
    # List all datasets in the hierarchical HDF5 file
    dataset_names = get_hdf5_dataset_names(hdf5_path)

    print("Dataset names in HDF5 file:")
    for name in dataset_names:
        print(f"  {name}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Loading with Filtering

    Use `load_results_from_hdf5_dataset()` to load results with optional filtering by comparison name, dataset names, or metric.
    """)
    return


@app.cell
def _(hdf5_path, load_results_from_hdf5_dataset):
    # Example 6a: Load all results from session_001
    results_session1 = load_results_from_hdf5_dataset(
        save_path=hdf5_path, dataset_name="session_001"
    )

    print(f"Loaded {len(results_session1['session_001'])} results from session_001")
    print("\nSample result keys:")
    for key in list(results_session1["session_001"].keys())[:3]:
        print(f"  {key}")
    return (results_session1,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Summary DataFrame

    Use `get_hdf5_result_summary()` to generate a pandas DataFrame with all results and their metadata for easy analysis.
    """)
    return


@app.cell
def _(get_hdf5_result_summary, hdf5_path):
    # Example 7: Generate summary DataFrame
    summary_df = get_hdf5_result_summary(
        save_path=hdf5_path, dataset_name="session_001"
    )

    print("Summary DataFrame:")
    print(
        summary_df[
            ["dataset_i", "dataset_j", "metric", "value", "n_samples_i", "n_samples_j"]
        ].head(10)
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Accessing Array Data

    Results can contain both scalar attributes and large array datasets (like pairwise comparison matrices).
    """)
    return


@app.cell
def _(results_session1):
    # Example 6b: Access specific result details
    _result_key = list(results_session1["session_001"].keys())[0]
    result = results_session1["session_001"][_result_key]
    print(f"Result: {_result_key}")
    print(f"  Attributes: {result['attributes']}")
    print(f"  Arrays available: {list(result['arrays'].keys())}")
    _pairs_indices = result["arrays"]["pairs_indices"]
    pairs_values = result["arrays"]["pairs_values"]
    # Access the pairs arrays (now split into indices and values)
    print(f"  Pairs shape: {_pairs_indices.shape}, values: {pairs_values.shape}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Phase 4B: Auto-Save/Load with compare_datasets()

    The `compare_datasets()` function now supports automatic result caching via HDF5:
    - **save_path**: Path to HDF5 file for caching results
    - **regenerate**: Force recomputation even if cached result exists
    - **dataset_names**: Required for mode="between" to identify datasets in cache

    Benefits:
    - **Instant loading**: Skip expensive computations for repeated analyses
    - **Reproducibility**: Cached results with metadata
    - **Easy comparison**: Test different metrics without recomputing
    """)
    return


@app.cell
def _(base, np):
    # Import metrics functions
    import time

    from neural_analysis.metrics.pairwise_metrics import compare_datasets

    np.random.seed(42)
    control_data = np.random.randn(100, 10)
    # Generate test datasets
    treatment_data = np.random.randn(100, 10) + 0.5
    cache_file = base / "comparison_cache.h5"
    print("=" * 60)  # Shifted distribution
    print("AUTO-SAVE/LOAD DEMONSTRATION")
    # Path for caching
    print("=" * 60)
    return cache_file, compare_datasets, control_data, time, treatment_data


@app.cell
def _(cache_file, compare_datasets, control_data, time, treatment_data):
    # First call: Compute and save
    print("\n1. First call: Computing and saving...")
    _start = time.time()
    result1 = compare_datasets(
        control_data,
        treatment_data,
        mode="between",
        metric="wasserstein",
        save_path=cache_file,
        dataset_names=("control", "treatment"),
    )
    elapsed1 = time.time() - _start
    result1_value = result1["value"] if isinstance(result1, dict) else result1
    print(f"   Result: {result1_value:.6f}")
    print(f"   Time: {elapsed1:.4f}s")
    # Result is a dict with 'value' key for between mode
    print(f"   Cache file created: {cache_file.exists()}")
    return elapsed1, result1_value


@app.cell
def _(
    cache_file,
    compare_datasets,
    control_data,
    elapsed1,
    result1_value,
    time,
    treatment_data,
):
    # Second call: Load from cache (instant!)
    print("\n2. Second call: Loading from cache...")
    _start = time.time()
    result2 = compare_datasets(
        control_data,
        treatment_data,
        mode="between",
        metric="wasserstein",
        save_path=cache_file,
        dataset_names=("control", "treatment"),
        regenerate=False,
    )
    elapsed2 = time.time() - _start  # These datasets are ignored - loads from cache
    result2_value = result2 if isinstance(result2, (int, float)) else result2["value"]
    print(f"   Result: {result2_value:.6f}")
    print(f"   Time: {elapsed2:.4f}s")
    print(f"   Speedup: {elapsed1 / elapsed2:.1f}x faster!")
    # When loading from cache, result is a float directly
    print(
        f"   Results match: {result2_value == result1_value}"
    )  # Default: use cached result
    return (result2_value,)


@app.cell
def _(
    cache_file,
    compare_datasets,
    control_data,
    result1_value,
    time,
    treatment_data,
):
    # Force regeneration with modified data
    print("\n3. Force regeneration (regenerate=True)...")
    treatment_modified = treatment_data + 0.5  # Further shift
    _start = time.time()
    result3 = compare_datasets(
        control_data,
        treatment_modified,
        mode="between",
        metric="wasserstein",
        save_path=cache_file,
        dataset_names=("control", "treatment"),
        regenerate=True,
    )
    elapsed3 = time.time() - _start
    result3_value = result3["value"] if isinstance(result3, dict) else result3
    print(f"   Result: {result3_value:.6f}")  # Different data
    print(f"   Time: {elapsed3:.4f}s")
    # Result is a dict with 'value' key for between mode
    print(
        f"   Changed: {abs(result3_value - result1_value) > 0.001}"
    )  # Force recomputation
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### All-Pairs Mode with Caching

    All-pairs mode also supports caching for efficiency when comparing multiple datasets.
    """)
    return


@app.cell
def _(base, compare_datasets, np, time):
    # Create multiple datasets
    datasets = {
        "control": np.random.randn(50, 8),
        "treatment_A": np.random.randn(50, 8) + 0.3,
        "treatment_B": np.random.randn(50, 8) + 0.7,
        "treatment_C": np.random.randn(50, 8) + 1.0,
    }
    cache_all_pairs = base / "all_pairs_cache.h5"
    print("\n" + "=" * 60)
    print("ALL-PAIRS CACHING")
    print("=" * 60)
    print("\n1. Computing all-pairs (6 comparisons)...")
    _start = time.time()
    all_pairs_results = compare_datasets(
        datasets,
        mode="all-pairs",
        metric="wasserstein",
        save_path=cache_all_pairs,
        show_progress=False,
    )
    elapsed_compute = time.time() - _start
    print(f"   Time: {elapsed_compute:.4f}s")
    print(f"   Comparisons: {sum(len(v) for v in all_pairs_results.values())} pairs")
    print("\n   Sample results:")
    for i, (key_i, inner) in enumerate(all_pairs_results.items()):
        # First call: Compute all pairs
        if i < 2:
            for key_j, dist in inner.items():
                # Display sample results
                print(f"      {key_i} → {key_j}: {dist:.4f}")  # Show first 2 datasets
    return all_pairs_results, cache_all_pairs, datasets, elapsed_compute


@app.cell
def _(
    all_pairs_results,
    cache_all_pairs,
    compare_datasets,
    datasets,
    elapsed_compute,
    time,
):
    # Second call: Load from cache
    print("\n2. Loading from cache...")
    _start = time.time()
    loaded_results = compare_datasets(
        datasets,
        mode="all-pairs",
        metric="wasserstein",
        save_path=cache_all_pairs,
        regenerate=False,
    )
    elapsed_load = time.time() - _start
    print(f"   Time: {elapsed_load:.4f}s")
    print(f"   Speedup: {elapsed_compute / elapsed_load:.1f}x faster!")
    matches = all(
        
            loaded_results[k1][k2] == all_pairs_results[k1][k2]
            for k1 in all_pairs_results
            for k2 in all_pairs_results[k1]
        
    )
    # Verify results match
    print(f"   Results match: {matches}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Inspection with comparison_store

    Use the comparison_store API to inspect cached comparisons:
    """)
    return


@app.cell
def _(cache_file, result2_value):
    # Query cached comparisons
    from neural_analysis.utils.comparison_store import (
        load_comparison,
        query_comparisons,
    )

    print("\n" + "=" * 60)
    print("INSPECTING CACHED COMPARISONS")
    print("=" * 60)
    _df = query_comparisons(cache_file)
    print("\nCached comparisons (between-mode):")
    # Query all comparisons in the between-mode cache
    print(_df[["metric", "mode", "dataset_i", "dataset_j"]])
    direct_load = load_comparison(
        cache_file, metric="wasserstein", dataset_i="control", dataset_j="treatment"
    )
    print(f"\nDirect load result: {direct_load:.6f}")
    # Load specific comparison directly
    print(f"Matches cached: {direct_load == result2_value}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Best Practices

    **When to use auto-save:**
    - ✅ Expensive computations (high-D data, many samples)
    - ✅ Repeated analyses with same data
    - ✅ Batch processing pipelines
    - ✅ Exploratory data analysis workflows

    **When regenerate=True:**
    - ✅ Data has been updated
    - ✅ Metric parameters changed
    - ✅ Force cache refresh
    - ✅ Debugging/validation

    **Caching tips:**
    - Use descriptive dataset_names for easy identification
    - Organize cache files by experiment/session
    - Query comparisons to avoid redundant computation
    - Clean up old cache files periodically
    """)
    return


@app.cell
def _(tmpdir):
    # Cleanup
    tmpdir.cleanup()
    print("✓ Cleaned up temporary directory")
    return
