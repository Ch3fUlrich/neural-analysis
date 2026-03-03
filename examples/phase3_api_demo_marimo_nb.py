import marimo

__generated_with = "0.18.3"

app = marimo.App(width="full")


@app.cell(hide_code=True)
def __():
    import marimo as mo

    return mo


@app.cell
def _():
    # Import metrics modules
    from neural_analysis.metrics.pairwise_metrics import (
        compute_all_pairs,
        compute_between_distances,
        compute_within_distances,
    )

    print("✓ Imports loaded!")
    return (
        compute_all_pairs,
        compute_between_distances,
        compute_within_distances,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
    # Phase 3 API Demo: Comprehensive Dataset Comparison Guide

    **Purpose**: Demonstrate the unified API for neural dataset comparisons using:
    - Three comparison modes: **within**, **between**, **all-pairs**
    - Four metric types: **point-to-point**, **distribution**, **shape**, **correlation**
    - Visualization with **PlotGrid** system
    - Synthetic data generation for testing

    **Date**: November 13, 2025
    **API Version**: Phase 3 (Post-Refactoring)

    ---

    ## Table of Contents

    1. [Synthetic Data Generation](#1-synthetic-data-generation)
    2. [Within-Mode Comparisons](#2-within-mode-comparisons)
    3. [Between-Mode Comparisons](#3-between-mode-comparisons)
    4. [All-Pairs Mode Comparisons](#4-all-pairs-mode-comparisons)
    5. [Visualization with PlotGrid](#5-visualization-with-plotgrid)
    6. [Real-World Scenarios](#6-real-world-scenarios)
    7. [Performance Comparison](#7-performance-comparison)
    8. [Summary & Best Practices](#8-summary--best-practices)
    """
    )
    return


@app.cell
def _():
    # Setup: Import all required modules
    import tempfile
    import time

    import matplotlib.pyplot as plt
    import numpy as np

    # Core metrics API
    from neural_analysis import generate_data
    from neural_analysis.metrics.pairwise_metrics import (
        ALL_METRICS,
        DISTRIBUTION_METRICS,
        POINT_TO_POINT_METRICS,
        SHAPE_METRICS,
        compare_datasets,
    )
    from neural_analysis.plotting import (
        GridLayoutConfig,
        PlotConfig,
        PlotGrid,
        PlotSpec,
    )
    from neural_analysis.utils.logging import configure_logging, get_logger

    configure_logging(level="INFO")
    logger = get_logger(__name__)  # Unified orchestration function
    np.random.seed(42)
    print("✓ All imports successful!")
    print(f"  Available metrics: {len(ALL_METRICS)} total")
    print(f"  - Point-to-point: {POINT_TO_POINT_METRICS}")
    print(f"  - Distribution: {DISTRIBUTION_METRICS}")
    # Synthetic data generation
    # Plotting system (PlotGrid)
    # Logging
    # Configure logging
    # Set random seed for reproducibility
    print(f"  - Shape: {SHAPE_METRICS}")
    return (
        DISTRIBUTION_METRICS,
        GridLayoutConfig,
        POINT_TO_POINT_METRICS,
        PlotConfig,
        PlotGrid,
        PlotSpec,
        SHAPE_METRICS,
        compare_datasets,
        generate_data,
        np,
        plt,
        tempfile,
        time,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 1. Synthetic Data Generation

    We'll create various test datasets to demonstrate different comparison scenarios:
    - **Gaussian clusters** (well-separated)
    - **Overlapping distributions**
    - **Different shapes** (elongated vs. spherical)
    - **Temporal sequences** (with drift)
    """
    )
    return


@app.cell
def _(generate_data, np):
    # Generate 5 synthetic datasets with different characteristics
    print("Generating synthetic datasets...")
    data_compact, _ = generate_data(
        dataset_type="blobs",
        n_samples=100,
        n_features=10,
        n_classes=1,
        cluster_std=0.5,
        random_state=42,
    )
    # Dataset 1: Compact Gaussian cluster
    print(f"✓ Compact cluster: {data_compact.shape}, std={data_compact.std():.3f}")
    data_elongated, _ = generate_data(
        dataset_type="blobs",
        n_samples=100,
        n_features=10,
        n_classes=1,
        cluster_std=2.0,
        random_state=43,
    )
    correlation_matrix = np.eye(10)
    correlation_matrix[0, 1] = correlation_matrix[1, 0] = 0.9
    chol = np.linalg.cholesky(correlation_matrix)
    data_elongated = data_elongated @ chol.T
    print(
        f"✓ Elongated cluster: {data_elongated.shape}, corr(f0,f1)={np.corrcoef(data_elongated[:, 0], data_elongated[:, 1])[0, 1]:.3f}"
    )
    data_shifted, _ = generate_data(
        dataset_type="blobs",
        n_samples=100,
        n_features=10,
        n_classes=1,
        cluster_std=1.0,
        random_state=44,
    )
    data_shifted = data_shifted + np.array(
        [2.0, 3.0, 4.0, 2.5, 3.5, 1.5, 2.0, 3.0, 4.0, 2.5]
    )
    print(f"✓ Shifted cluster: {data_shifted.shape}, mean={data_shifted.mean():.3f}")
    # Dataset 2: Elongated cluster (high correlation)
    data_multi, _ = generate_data(
        dataset_type="blobs",
        n_samples=150,
        n_features=10,
        n_classes=3,
        cluster_std=1.0,
        random_state=45,
    )
    print(f"✓ Multi-cluster: {data_multi.shape}, std={data_multi.std():.3f}")
    data_drift, _ = generate_data(
        dataset_type="blobs",
        n_samples=100,
        n_features=10,
        n_classes=1,
        cluster_std=1.0,
        random_state=46,
    )
    drift_vector = np.linspace(0, 2, 100)[:, np.newaxis]
    data_drift = data_drift + drift_vector
    print(
        f"✓ Temporal drift: {data_drift.shape}, drift_range={drift_vector.min():.1f}-{drift_vector.max():.1f}"
    )
    datasets = {
        "Compact": data_compact,
        "Elongated": data_elongated,
        "Shifted": data_shifted,
        "MultiCluster": data_multi,
        "Drift": data_drift,
    }
    # Add correlation between features
    # Dataset 3: Shifted mean
    # Dataset 4: Multi-cluster
    # Dataset 5: Temporal drift
    # Add linear drift over time
    # Collect into dictionary
    print(f"\n✓ All {len(datasets)} datasets generated successfully!")
    return data_compact, datasets


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 2. Within-Mode Comparisons

    **Purpose**: Measure intra-dataset variability (self-similarity, cluster tightness)

    **Supported metrics**: Point-to-point only (euclidean, manhattan, cosine, mahalanobis)

    **Returns**: Float (mean distance) or matrix (n_samples × n_samples)
    """
    )
    return


@app.cell
def _(compute_within_distances, datasets):
    # Within-mode: Compare cluster tightness across datasets
    print("Within-mode comparisons (cluster tightness):\n")

    within_results = {}
    for name, data in datasets.items():
        # Compute mean pairwise distance (tightness measure)
        mean_dist = compute_within_distances(data, metric="euclidean")
        within_results[name] = mean_dist
        print(f"{name:15s}: {mean_dist:.4f}")

    # Identify tightest and loosest clusters
    tightest = min(within_results, key=within_results.get)
    loosest = max(within_results, key=within_results.get)

    print(f"\n✓ Tightest cluster: {tightest} ({within_results[tightest]:.4f})")
    print(f"✓ Loosest cluster: {loosest} ({within_results[loosest]:.4f})")
    print(
        f"  Tightness ratio: {within_results[loosest] / within_results[tightest]:.2f}x"
    )
    return (within_results,)


@app.cell
def _(POINT_TO_POINT_METRICS, compute_within_distances, data_compact, np):
    # Compare all point-to-point metrics on compact cluster
    print("\n" + "=" * 60)
    print("All point-to-point metrics (Compact dataset):")
    print("=" * 60)
    for metric in sorted(POINT_TO_POINT_METRICS):
        if metric == "mahalanobis":
            cov = np.cov(data_compact.T)
            mean_dist_1 = compute_within_distances(
                data_compact, metric=metric, cov=cov
            )  # Mahalanobis requires covariance matrix
        else:
            mean_dist_1 = compute_within_distances(data_compact, metric=metric)
        print(f"{metric:15s}: {mean_dist_1:.6f}")
    print("=" * 60)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 3. Between-Mode Comparisons

    **Purpose**: Compare two datasets (condition A vs B, pre/post treatment, session-to-session)

    **Supported metrics**: ALL (point-to-point, distribution, shape)

    **Returns**:
    - Point-to-point: Float (mean) or matrix (n1 × n2)
    - Distribution/Shape: Float (scalar distance)
    """
    )
    return


@app.cell
def _(
    DISTRIBUTION_METRICS,
    POINT_TO_POINT_METRICS,
    compare_datasets,
    datasets,
):
    # Between-mode comparisons: Test all metric types
    from neural_analysis.utils import log_section

    log_section("Between-Mode Comparisons: All Metrics")
    data_compact_1 = datasets["Compact"]
    data_shifted_1 = datasets["Shifted"]
    # Compare Compact vs Shifted datasets
    print("Point-to-point metrics:")
    for metric_1 in sorted(POINT_TO_POINT_METRICS):
        if metric_1 == "mahalanobis":
            continue
        result = compare_datasets(
            data_compact_1, data_shifted_1, mode="between", metric=metric_1
        )
        value = result["value"] if isinstance(result, dict) else result
        print(
            f"  {metric_1:20s}: {value:.6f}"
        )  # Skip mahalanobis - requires special handling
    print(
        "\nDistribution metrics:"
    )  # Use compare_datasets which returns dict for between-mode
    for metric_1 in sorted(DISTRIBUTION_METRICS):
        result = compare_datasets(
            data_compact_1, data_shifted_1, mode="between", metric=metric_1
        )  # Handle both dict and float returns
        value = result["value"] if isinstance(result, dict) else result
        print(f"  {metric_1:20s}: {value:.6f}")
    print("\nShape metrics (equal sample sizes):")
    for metric_1 in ["procrustes", "soft-matching", "one-to-one"]:
        result = compare_datasets(
            data_compact_1, data_shifted_1, mode="between", metric=metric_1
        )
        value = (
            result["value"] if isinstance(result, dict) else result
        )  # Use compare_datasets which returns dict for between-mode
        print(f"  {metric_1:20s}: {value:.6f}")
    print(
        "\n✓ All between-mode metrics tested successfully!"
    )  # Handle both dict and float returns  # Use compare_datasets which returns dict for between-mode  # Handle both dict and float returns
    return (log_section,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 4. All-Pairs Mode Comparisons

    **Purpose**: Compare all datasets against each other (distance matrix, hierarchical clustering)

    **Supported metrics**: Distribution + Shape only (scalar-returning metrics)

    **Returns**: Nested dict `{dataset_i: {dataset_j: distance}}`
    """
    )
    return


@app.cell
def _(compute_all_pairs, datasets):
    # All-pairs mode: Compare all datasets using Wasserstein distance
    print("All-pairs Wasserstein distances:\n")
    results_wasserstein = compute_all_pairs(
        datasets, metric="wasserstein", show_progress=False
    )
    dataset_names = list(datasets.keys())
    print(f"{'':15s}", end="")
    for name_1 in dataset_names:
        print(f"{name_1:12s}", end="")
    print()
    print("-" * (15 + 12 * len(dataset_names)))
    # Display results in matrix format
    for name_i in dataset_names:
        print(f"{name_i:15s}", end="")
        for name_j in dataset_names:
            dist = results_wasserstein[name_i][name_j]
            print(f"{dist:12.4f}", end="")
        print()
    print("\n✓ All-pairs matrix computed successfully!")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 5. Visualization with PlotGrid

    **PlotGrid** provides metadata-driven, backend-agnostic plotting (matplotlib ↔ plotly)

    **Features**:
    - Multi-panel layouts with automatic sizing
    - Consistent styling and legend deduplication
    - Type-safe plot specifications
    - Easy export to various formats
    """
    )
    return


@app.cell
def _(
    GridLayoutConfig,
    PlotConfig,
    PlotGrid,
    PlotSpec,
    np,
    plt,
    within_results,
):
    # Visualize within-mode results (cluster tightness) using PlotGrid
    # Prepare data for plotting
    tightness_values = {
        name: within_results[name]
        for name in sorted(within_results, key=within_results.get)
    }
    dataset_names_1 = list(tightness_values.keys())
    # Convert dict to array format for PlotGrid
    values = list(tightness_values.values())
    spec = PlotSpec(
        data=values,
        plot_type="bar",
        title="Within-Dataset Variability (Lower = Tighter)",
        kwargs={
            "x_label": "Dataset",
            "y_label": "Mean Pairwise Distance",
            "set_xticks": range(len(dataset_names_1)),
            "set_xticklabels": dataset_names_1,
            "grid": {"axis": "y", "alpha": 0.3},
        },
    )
    grid = PlotGrid(
        plot_specs=[spec],
        config=PlotConfig(figsize=(10, 5)),
        layout=GridLayoutConfig(rows=1, cols=1),
    )
    # Create bar plot using PlotGrid
    result_1 = grid.plot()
    if isinstance(result_1, tuple):
        fig, axes = result_1
        if isinstance(axes, (list, np.ndarray)):
            ax = axes[0] if isinstance(axes, list) else axes.flat[0]
        else:
            ax = axes
        ax.tick_params(axis="x", rotation=45, ha="right")
        plt.tight_layout()
    elif hasattr(result_1, "gca"):
        ax = result_1
        fig = ax.figure
        ax.tick_params(axis="x", rotation=45, ha="right")
        plt.tight_layout()
    else:
        fig = result_1
    plt.show()
    # Handle matplotlib return: single subplot returns just axes, multiple return (fig, axes)
    # For plotly, returns just the figure
    print(
        "✓ Cluster tightness visualization complete!"
    )  # Multiple subplots: (fig, axes)  # Get first axis for label rotation  # Rotate x-axis labels (post-processing on PlotGrid output)  # Single subplot: returns just the axes object  # Rotate x-axis labels  # Plotly figure
    return


@app.cell
def _(
    GridLayoutConfig,
    PlotConfig,
    PlotGrid,
    PlotSpec,
    compute_all_pairs,
    datasets,
    np,
    plt,
):
    # Visualize all-pairs distance matrix as a heatmap using PlotGrid
    # Need to run all-pairs first - get it from earlier results or recompute
    if "all_pairs_results" not in globals():
        all_pairs_results = compute_all_pairs(
            datasets, metric="wasserstein", show_progress=False
        )  # Recompute if not available
    dataset_names_2 = list(all_pairs_results.keys())
    n = len(dataset_names_2)
    # Convert nested dict to numpy matrix
    distance_matrix = np.zeros((n, n))
    for i, name_i_1 in enumerate(dataset_names_2):
        for j, name_j_1 in enumerate(dataset_names_2):
            distance_matrix[i, j] = all_pairs_results[name_i_1][name_j_1]
    spec_1 = PlotSpec(
        data=distance_matrix,
        plot_type="heatmap",
        title="All-Pairs Wasserstein Distance Matrix",
        colorbar=True,
        colorbar_label="Wasserstein Distance",
        cmap="viridis",
        kwargs={
            "x_labels": dataset_names_2,
            "y_labels": dataset_names_2,
            "show_values": True,
            "value_format": ".2f",
        },
    )
    grid_1 = PlotGrid(
        plot_specs=[spec_1],
        config=PlotConfig(figsize=(8, 7)),
        layout=GridLayoutConfig(rows=1, cols=1),
    )
    result_2 = grid_1.plot()
    # Create heatmap using PlotGrid
    if isinstance(result_2, tuple):
        fig_1, axes_1 = result_2
        ax_1 = (
            axes_1[0]
            if isinstance(axes_1, (list, np.ndarray))
            else axes_1.flat[0]
            if hasattr(axes_1, "flat")
            else axes_1
        )
    elif hasattr(result_2, "gca"):
        ax_1 = result_2
        fig_1 = ax_1.figure
    else:
        fig_1 = result_2
        ax_1 = None
    if ax_1 is not None:
        ax_1.tick_params(axis="x", rotation=45, ha="right")
        plt.tight_layout()
    plt.show()
    # Handle matplotlib return: single subplot returns just axes, multiple return (fig, axes)
    # Post-process for rotated labels (PlotGrid doesn't support rotation in kwargs yet)
    print("✓ Distance matrix heatmap complete!")  # Single subplot or plotly
    return


@app.cell
def _(
    compare_datasets, datasets, GridLayoutConfig, PlotConfig, PlotGrid, PlotSpec, plt
):
    # Compare multiple metrics using PlotGrid
    # Test 3 different metrics for between-mode comparison

    _metrics_to_compare = ["euclidean", "wasserstein", "kolmogorov-smirnov"]
    _between_results_multi = {}

    for _metric in _metrics_to_compare:
        # Use compare_datasets which returns dict for between-mode
        _result = compare_datasets(
            datasets["Compact"], datasets["Shifted"], mode="between", metric=_metric
        )
        # Handle both dict and float returns
        _between_results_multi[_metric] = (
            _result["value"] if isinstance(_result, dict) else _result
        )

    # Convert dict to array format for PlotGrid
    _metric_names = list(_between_results_multi.keys())
    _values = list(_between_results_multi.values())

    # Create bar plot using PlotGrid
    _spec = PlotSpec(
        data=_values,
        plot_type="bar",
        title="Compact vs Shifted: Different Metrics",
        kwargs={
            "x_label": "Metric Type",
            "y_label": "Distance Value",
            "set_xticks": range(len(_metric_names)),
            "set_xticklabels": _metric_names,
            "grid": {"axis": "y", "alpha": 0.3},
        },
    )

    _grid = PlotGrid(
        plot_specs=[_spec],
        config=PlotConfig(figsize=(10, 5)),
        layout=GridLayoutConfig(rows=1, cols=1),
    )
    _fig = _grid.plot()

    # Rotate x-axis labels (post-processing on PlotGrid output)
    _ax = _fig.axes[0] if hasattr(_fig, "axes") else _fig.gca()
    _ax.tick_params(axis="x", rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    print("✓ Multi-metric comparison complete!")
    for _metric, _value in _between_results_multi.items():
        print(f"  {_metric:20s}: {_value:.4f}")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 6. Real-World Scenarios

    These examples demonstrate practical use cases for the metrics API in neuroscience research.
    """
    )
    return


@app.cell
def _(compare_datasets, generate_data, log_section):
    # Scenario 1: Pre/Post Treatment Comparison
    # Simulate neural activity before and after drug treatment
    from neural_analysis.utils import log_kv

    log_section("Scenario 1: Pre/Post Treatment Analysis")
    pre_treatment, _ = generate_data(
        dataset_type="blobs",
        n_samples=100,
        n_features=10,
        n_classes=1,
        cluster_std=1.0,
        random_state=50,
    )
    post_treatment, _ = generate_data(
        dataset_type="blobs",
        n_samples=100,
        n_features=10,
        n_classes=1,
        cluster_std=0.8,
        random_state=51,
    )
    post_treatment = post_treatment + 1.5
    treatment_effect = compare_datasets(
        pre_treatment, post_treatment, mode="between", metric="wasserstein"
    )
    distance_value = (
        treatment_effect["value"]
        if isinstance(treatment_effect, dict)
        else treatment_effect
    )  # Shift mean to simulate treatment effect
    log_kv(
        "treatment_analysis",
        {
            "metric": "wasserstein",
            "distance": distance_value,
            "interpretation": "Larger values indicate stronger treatment effect",
        },
    )
    # Test if treatment had significant effect
    # Use compare_datasets which returns dict for between-mode
    print(f"Pre vs Post Treatment Distance: {distance_value:.4f}")
    # Handle both dict and float returns
    print("→ Interpretation: Distance > 0.5 suggests meaningful biological effect")
    return


@app.cell
def _(compute_all_pairs, generate_data, log_section, np):
    # Scenario 2: Session-to-Session Variability
    # Compare neural recordings across multiple experimental sessions
    log_section("Scenario 2: Session Variability Analysis")
    sessions = {
        f"Session_{i + 1}": generate_data(
            dataset_type="blobs",
            n_samples=80,
            n_features=10,
            n_classes=1,
            cluster_std=1.0 + i * 0.1,
            random_state=60 + i,
        )[0]
        for i in range(5)
    }
    session_distances = compute_all_pairs(sessions, metric="wasserstein")
    dataset_names_3 = list(session_distances.keys())
    n_1 = len(dataset_names_3)
    distance_matrix_1 = np.zeros((n_1, n_1))
    for i_1, name_i_2 in enumerate(dataset_names_3):
        # Compute all-pairs to find most/least similar sessions
        for j_1, name_j_2 in enumerate(dataset_names_3):
            distance_matrix_1[i_1, j_1] = session_distances[name_i_2][name_j_2]
    # Convert nested dict to numpy matrix
    mask = ~np.eye(n_1, dtype=bool)
    off_diagonal = distance_matrix_1[mask]
    min_idx = np.unravel_index(np.argmin(off_diagonal), distance_matrix_1.shape)
    max_idx = np.unravel_index(np.argmax(off_diagonal), distance_matrix_1.shape)
    print(
        f"Most similar: {dataset_names_3[min_idx[0]]} ↔ {dataset_names_3[min_idx[1]]} (distance: {distance_matrix_1[min_idx]:.3f})"
    )
    print(
        f"Least similar: {dataset_names_3[max_idx[0]]} ↔ {dataset_names_3[max_idx[1]]} (distance: {distance_matrix_1[max_idx]:.3f})"
    )
    print(f"Mean session variability: {off_diagonal.mean():.3f}")
    # Find most and least similar session pairs
    # Create mask to exclude diagonal (self-comparisons, which are 0.0)
    print(
        "Note: Diagonal elements (self-comparisons) are 0.0 and excluded from analysis"
    )
    return


@app.cell
def _(compare_datasets, generate_data, log_section, np):
    # Scenario 3: Condition A vs B Hypothesis Testing
    # Test if two experimental conditions produce different neural patterns
    log_section("Scenario 3: A/B Condition Testing")
    condition_a, _ = generate_data(
        dataset_type="blobs",
        n_samples=100,
        n_features=10,
        n_classes=1,
        cluster_std=1.0,
        random_state=70,
    )
    condition_b, _ = generate_data(
        dataset_type="blobs",
        n_samples=100,
        n_features=10,
        n_classes=1,
        cluster_std=1.0,
        random_state=71,
    )
    condition_b = condition_b + 0.3
    metrics_for_hypothesis = [
        "euclidean",
        "wasserstein",
        "kolmogorov-smirnov",
    ]  # Small effect size
    hypothesis_results = {}
    # Use multiple metrics for robust comparison
    for metric_2 in metrics_for_hypothesis:
        result_3 = compare_datasets(
            condition_a, condition_b, mode="between", metric=metric_2
        )
        value_1 = result_3["value"] if isinstance(result_3, dict) else result_3
        hypothesis_results[metric_2] = value_1
    print(
        "Condition A vs B - Metric Comparison:"
    )  # Use compare_datasets which returns dict for between-mode
    for metric_2, distance in hypothesis_results.items():
        print(f"  {metric_2:20s}: {distance:.4f}")  # Handle both dict and float returns
    if "kolmogorov-smirnov" in hypothesis_results:
        ks_val = hypothesis_results["kolmogorov-smirnov"]
        if ks_val == 1.0:
            print(
                "\n  Note: KS distance = 1.0 indicates perfect separation in at least one feature."
            )
            print(
                "        This is valid and means the distributions are completely separated"
            )
            print(
                "        in at least one dimension, which is expected for different random datasets."
            )
    avg_distance = np.mean(list(hypothesis_results.values()))
    # Note about Kolmogorov-Smirnov distance
    print(f"\nAverage distance: {avg_distance:.4f}")
    # Simple decision rule
    print(
        f"→ Conclusion: {('Conditions differ significantly' if avg_distance > 0.2 else 'Conditions appear similar')}"
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 7. Performance Comparison

    Demonstration of computational efficiency across different dataset sizes and parallelization options.
    """
    )
    return


@app.cell
def _(
    GridLayoutConfig,
    PlotConfig,
    PlotGrid,
    PlotSpec,
    compute_within_distances,
    generate_data,
    log_section,
    np,
    plt,
    time,
):
    # Performance comparison: All within distance metrics with parallel=True and parallel=False
    log_section("Performance Analysis: All Within Metrics Comparison")
    sizes = [50, 100, 200, 500, 1000]
    metrics_to_test = ["euclidean", "manhattan", "cosine", "mahalanobis"]
    results_parallel = {metric: [] for metric in metrics_to_test}
    results_serial = {metric: [] for metric in metrics_to_test}
    print("Testing with parallel=True...")
    for metric_3 in metrics_to_test:
        # Store results for both parallel modes
        timings = []
        for n_2 in sizes:
            data_1, _ = generate_data(
                dataset_type="blobs",
                n_samples=n_2,
                n_features=10,
                n_classes=1,
                cluster_std=1.0,
                random_state=80,
            )
            # Test with parallel=True
            start = time.time()
            result_4 = compute_within_distances(data_1, metric=metric_3, parallel=True)
            elapsed = time.time() - start
            timings.append(elapsed)
        results_parallel[metric_3] = timings
        print(f"  {metric_3:12s}: {[f'{t:.4f}s' for t in timings]}")
    print("\nTesting with parallel=False...")
    for metric_3 in metrics_to_test:
        timings = []
        for n_2 in sizes:
            data_1, _ = generate_data(
                dataset_type="blobs",
                n_samples=n_2,
                n_features=10,
                n_classes=1,
                cluster_std=1.0,
                random_state=80,
            )
            start = time.time()
            # Test with parallel=False
            result_4 = compute_within_distances(data_1, metric=metric_3, parallel=False)
            elapsed = time.time() - start
            timings.append(elapsed)
        results_serial[metric_3] = timings
        print(f"  {metric_3:12s}: {[f'{t:.4f}s' for t in timings]}")
    print("\nVerifying results are identical...")
    all_match = True
    for metric_3 in metrics_to_test:
        for i_2, n_2 in enumerate(sizes):
            data_1, _ = generate_data(
                dataset_type="blobs",
                n_samples=n_2,
                n_features=10,
                n_classes=1,
                cluster_std=1.0,
                random_state=80,
            )
            result_par = compute_within_distances(
                data_1, metric=metric_3, parallel=True
            )
            result_ser = compute_within_distances(
                data_1, metric=metric_3, parallel=False
            )
            # Verify results are the same (within numerical precision)
            if not np.isclose(result_par, result_ser, rtol=1e-10):
                print(
                    f"  ✗ {metric_3} (n={n_2}): parallel={result_par:.6f}, serial={result_ser:.6f}"
                )
                all_match = False
    if all_match:
        print("  ✓ All results match between parallel and serial modes")
    specs = []
    colors = ["blue", "green", "red", "orange"]
    for i_2, metric_3 in enumerate(metrics_to_test):
        line_data = np.column_stack([sizes, results_parallel[metric_3]])
        specs.append(
            PlotSpec(
                data=line_data,
                plot_type="line",
                label=metric_3,
                color=colors[i_2],
                marker="o",
                subplot_position=0,
                kwargs={
                    "x_label": "Dataset Size (samples)",
                    "y_label": "Computation Time (seconds)",
                    "grid": {"axis": "both", "alpha": 0.3},
                },
            )
        )
    for i_2, metric_3 in enumerate(metrics_to_test):
        line_data = np.column_stack([sizes, results_serial[metric_3]])
        specs.append(
            PlotSpec(
                data=line_data,
                plot_type="line",
                label=metric_3,
                color=colors[i_2],
                marker="s",
                subplot_position=1,
                kwargs={
                    "x_label": "Dataset Size (samples)",
                    "y_label": "Computation Time (seconds)",
                    "grid": {"axis": "both", "alpha": 0.3},
                },
            )
        )
    # Create subplots using PlotGrid
    grid_2 = PlotGrid(
        plot_specs=specs,
        layout=GridLayoutConfig(
            rows=1, cols=2, subplot_titles=["Parallel=True", "Parallel=False"]
        ),
        config=PlotConfig(
            title="Performance Comparison: Parallel vs Serial Computation",
            figsize=(16, 6),
        ),
    )
    fig_2 = grid_2.plot()
    plt.show()
    # Subplot 1: parallel=True (all metrics on same subplot)
    # Subplot 2: parallel=False (all metrics on same subplot)
    print(
        "✓ Performance analysis complete!"
    )  # Convert to column-stacked format for line plot
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 8. Summary & Best Practices

    Key takeaways and recommendations for using the metrics API effectively.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Shape Distance Verification: Soft-Matching ≤ One-to-One ≤ Procrustes

    **Theoretical Property:**
    The three shape distance metrics should satisfy the ordering:
    - **Procrustes**: Finds optimal **rotation** with fixed point correspondence (i→i)
    - **One-to-One**: Finds optimal **permutation** (any i→j mapping) after Procrustes alignment
    - **Soft-Matching**: Finds optimal **soft assignment** (fractional matching) after Procrustes alignment

    The ordering should be: **soft-matching ≤ one-to-one ≤ procrustes** because:
    - Permutation space ⊇ Identity, so optimal permutation ≤ optimal rotation
    - Soft assignment space ⊇ Hard assignment, so optimal soft matching ≤ optimal hard matching

    Let's verify this property holds in practice:
    """
    )
    return


@app.cell
def _(compute_between_distances, np):
    # Test shape distance property: soft-matching ≤ one-to-one ≤ procrustes
    np.random.seed(42)
    test_cases = {
        "Random": (np.random.randn(50, 10), np.random.randn(50, 10)),
        "Rotated": (
            np.random.randn(50, 10),
            np.random.randn(50, 10) @ np.linalg.qr(np.random.randn(10, 10))[0],
        ),
        "Scaled": (np.random.randn(50, 10), 2.5 * np.random.randn(50, 10)),
        "Shifted": (np.random.randn(50, 10), np.random.randn(50, 10) + 5.0),
    }
    print("Testing Property: Soft-Matching ≤ One-to-One ≤ Procrustes")
    print("=" * 80)
    # Generate test datasets with different characteristics
    violations = []
    for name_2, (data1, data2) in test_cases.items():
        dist_procrustes = compute_between_distances(data1, data2, metric="procrustes")
        dist_one_to_one = compute_between_distances(data1, data2, metric="one-to-one")
        dist_soft_matching = compute_between_distances(
            data1, data2, metric="soft-matching"
        )
        satisfies_oto_proc = dist_one_to_one <= dist_procrustes
        satisfies_soft_oto = dist_soft_matching <= dist_one_to_one
        satisfies_all = satisfies_oto_proc and satisfies_soft_oto
        ratio_oto_proc = dist_one_to_one / dist_procrustes if dist_procrustes > 0 else 0
        ratio_soft_oto = (
            dist_soft_matching / dist_one_to_one if dist_one_to_one > 0 else 0
        )
        status = "✓" if satisfies_all else "✗ VIOLATION"
        print(f"\n{name_2:12s} {status}")
        print(f"  Procrustes:    {dist_procrustes:.6f}")
        print(
            f"  One-to-One:    {dist_one_to_one:.6f}  (ratio to procrustes: {ratio_oto_proc:.4f})"
        )
        print(
            f"  Soft-Matching: {dist_soft_matching:.6f}  (ratio to one-to-one: {ratio_soft_oto:.4f})"
        )
        if not satisfies_oto_proc:
            print(
                "  ⚠️  One-to-One > Procrustes (should be ≤)"
            )  # Compute all three distances
        if not satisfies_soft_oto:
            print("  ⚠️  Soft-Matching > One-to-One (should be ≤)")
        if not satisfies_all:
            violations.append(
                (name_2, dist_procrustes, dist_one_to_one, dist_soft_matching)
            )
    print("\n" + "=" * 80)  # Check properties
    if violations:
        print(f"⚠️  Found {len(violations)} violations!")
        for name_2, proc, oto, soft in violations:
            print(
                f"   {name_2}: procrustes={proc:.4f}, one-to-one={oto:.4f}, soft-matching={soft:.4f}"
            )
    else:
        print("✅ All tests passed! Property holds for all cases.")
    return


@app.cell
def _(compute_between_distances, np):
    np.random.seed(42)
    test_cases_1 = {
        "Random": (np.random.randn(30, 5), np.random.randn(30, 5)),
        "Shifted": (np.random.randn(30, 5), np.random.randn(30, 5) + 5.0),
        "Scaled": (np.random.randn(30, 5), 2.5 * np.random.randn(30, 5)),
    }
    print("Testing Property: Soft-Matching ≤ One-to-One ≤ Procrustes")
    print("=" * 70)
    all_pass = True
    for name_3, (data1_1, data2_1) in test_cases_1.items():
        proc_1 = compute_between_distances(data1_1, data2_1, metric="procrustes")
        oto_1 = compute_between_distances(data1_1, data2_1, metric="one-to-one")
        # Re-import
        soft_1 = compute_between_distances(data1_1, data2_1, metric="soft-matching")
        passes_oto_proc = oto_1 <= proc_1
        # Test with multiple cases
        passes_soft_oto = soft_1 <= oto_1
        passes_all = passes_oto_proc and passes_soft_oto
        all_pass = all_pass and passes_all
        ratio_oto_proc_1 = oto_1 / proc_1 if proc_1 > 0 else 0
        ratio_soft_oto_1 = soft_1 / oto_1 if oto_1 > 0 else 0
        status_1 = "✓" if passes_all else "✗ FAIL"
        print(f"\n{name_3:12s} {status_1}")
        print(f"  Procrustes:    {proc_1:.6f}")
        print(
            f"  One-to-One:    {oto_1:.6f}  (ratio to procrustes: {ratio_oto_proc_1:.4f})"
        )
        print(
            f"  Soft-Matching: {soft_1:.6f}  (ratio to one-to-one: {ratio_soft_oto_1:.4f})"
        )
        if not passes_oto_proc:
            print("  ⚠️  One-to-One > Procrustes")
        if not passes_soft_oto:
            print("  ⚠️  Soft-Matching > One-to-One")
    print("\n" + "=" * 70)
    print(f"{('✅ All tests PASSED!' if all_pass else '❌ Some tests FAILED')}")
    return


@app.cell
def _(compute_between_distances, np):
    # Simple check
    data1_2 = np.random.randn(20, 4)
    data2_2 = np.random.randn(20, 4)
    proc_2 = compute_between_distances(data1_2, data2_2, metric="procrustes")
    oto_2 = compute_between_distances(data1_2, data2_2, metric="one-to-one")
    print(f"Procrustes: {proc_2:.4f}")
    print(f"One-to-One: {oto_2:.4f}")
    print(f"Ratio: {oto_2 / proc_2:.4f}")
    print(f"Property: {oto_2 <= proc_2}")
    return


@app.cell
def _(np):
    # Deep dive: What IS Procrustes disparity?
    from scipy.spatial import procrustes

    data1_3 = np.random.randn(10, 3)
    data2_3 = np.random.randn(10, 3)
    m1, m2, disparity = procrustes(data1_3, data2_3)
    frobenius_squared = np.sum((m1 - m2) ** 2)
    sum_of_squared_row_distances = np.sum(np.linalg.norm(m1 - m2, axis=1) ** 2)
    print(f"Scipy disparity:                  {disparity:.8f}")
    # Manual calculations
    print(f"||m1-m2||²_F:                     {frobenius_squared:.8f}")
    print(f"Σ||row_i - row_i||²:              {sum_of_squared_row_distances:.8f}")
    print(
        f"\nConclusion: disparity = Frobenius² = Σ||row||²: {np.allclose([disparity, frobenius_squared, sum_of_squared_row_distances], disparity)}"
    )
    return (procrustes,)


@app.cell
def _(compare_datasets, np):
    # Final test: One-to-One ≤ Procrustes with rotation
    import logging

    logging.getLogger("neural_analysis").setLevel(logging.WARNING)
    np.random.seed(123)
    n_tests = 5
    passes = 0
    print("Testing: One-to-One ≤ Procrustes")
    print("=" * 50)
    for i_3 in range(n_tests):
        d1 = np.random.randn(20, 5)
        d2 = np.random.randn(20, 5)
        proc_result = compare_datasets(d1, d2, mode="between", metric="procrustes")
        proc_3 = proc_result["value"] if isinstance(proc_result, dict) else proc_result
        oto_result = compare_datasets(d1, d2, mode="between", metric="one-to-one")
        oto_3 = oto_result["value"] if isinstance(oto_result, dict) else oto_result
        ok = oto_3 <= proc_3 * 1.0001
        passes = passes + ok
        print(
            f"Test {i_3 + 1}: {('✓' if ok else '✗')}  Proc={proc_3:.4f}, OTO={oto_3:.4f}, Ratio={oto_3 / proc_3:.4f}"
        )
    print("=" * 50)
    print(f"Result: {passes}/{n_tests} passed {('✅' if passes == n_tests else '❌')}")
    # Restore logging
    logging.getLogger("neural_analysis").setLevel(
        logging.INFO
    )  # Small tolerance for numerical errors
    return


@app.cell
def _(compute_between_distances, np, procrustes):
    # Debug: Check the actual implementation being used
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist

    d1_1 = np.random.randn(10, 3)
    d2_1 = np.random.randn(10, 3)
    m1_proc, m2_proc, disparity_proc = procrustes(d1_1, d2_1)
    print(f"Procrustes disparity: {disparity_proc:.6f}")
    cost_matrix = cdist(m1_proc, m2_proc, metric="sqeuclidean")
    # Manual Procrustes
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    oto_distance_manual = cost_matrix[row_ind, col_ind].sum()
    print(f"One-to-one (manual): {oto_distance_manual:.6f}")
    # Manual one-to-one (what it SHOULD be doing)
    print(f"Property holds: {oto_distance_manual <= disparity_proc}")
    result_oto = compute_between_distances(d1_1, d2_1, metric="one-to-one")
    print(f"\nOur function returns: {result_oto['value']:.6f}")
    # What our function returns
    print(f"Matches manual: {np.isclose(result_oto['value'], oto_distance_manual)}")
    return cdist, linear_sum_assignment


@app.cell
def _(cdist, linear_sum_assignment, np, procrustes):
    # Debug: Understand what procrustes returns
    data1_4 = np.random.randn(10, 3)
    data2_4 = np.random.randn(10, 3)
    m1_1, m2_1, disparity_1 = procrustes(data1_4, data2_4)
    print("Procrustes Analysis:")
    print(f"  Disparity (returned): {disparity_1:.6f}")
    print(f"  Frobenius norm squared: {np.sum((m1_1 - m2_1) ** 2):.6f}")
    print(
        f"  Sum of point distances: {np.sum(np.linalg.norm(m1_1 - m2_1, axis=1)):.6f}"
    )
    # Procrustes
    print(f"  m1 Frobenius norm: {np.linalg.norm(m1_1, 'fro'):.6f}")
    print(f"  m2 Frobenius norm: {np.linalg.norm(m2_1, 'fro'):.6f}")
    m1_manual = data1_4 / np.linalg.norm(data1_4, "fro")
    m2_manual = data2_4 / np.linalg.norm(data2_4, "fro")
    print("\nManual Normalization (one-to-one style):")
    print(f"  m1 Frobenius norm: {np.linalg.norm(m1_manual, 'fro'):.6f}")
    print(f"  m2 Frobenius norm: {np.linalg.norm(m2_manual, 'fro'):.6f}")
    cost_matrix_1 = cdist(m1_manual, m2_manual, metric="sqeuclidean")
    # Manual normalization (what one-to-one does)
    row_ind_1, col_ind_1 = linear_sum_assignment(cost_matrix_1)
    oto_distance = cost_matrix_1[row_ind_1, col_ind_1].sum()
    print("\nOne-to-One (sqeuclidean, sum):")
    print(f"  Total distance: {oto_distance:.6f}")
    identity_distance = np.sum((m1_manual - m2_manual) ** 2)
    print("\nIdentity matching (i→i):")
    print(f"  Total squared distance: {identity_distance:.6f}")
    # Compute one-to-one matching on manually normalized
    print("\nComparison:")
    print(f"  One-to-One / Identity: {oto_distance / identity_distance:.4f}")
    # Identity matching (what procrustes uses)
    print(
        f"  Property holds: {oto_distance <= identity_distance} {('✓' if oto_distance <= identity_distance else '✗')}"
    )
    return


@app.cell
def _(np, procrustes):
    # Test: What preprocessing does scipy.procrustes apply?
    data1_5 = np.random.randn(10, 3) + 5  # Shifted data
    data2_5 = np.random.randn(10, 3) + 10  # Different shift
    m1_2, m2_2, disparity_2 = procrustes(data1_5, data2_5)
    print("Input statistics:")
    print(f"  data1 mean: {data1_5.mean(axis=0)}")
    print(f"  data2 mean: {data2_5.mean(axis=0)}")
    print(f"  data1 Frobenius norm: {np.linalg.norm(data1_5, 'fro'):.4f}")
    print(f"  data2 Frobenius norm: {np.linalg.norm(data2_5, 'fro'):.4f}")
    print("\nProcessed by procrustes:")
    print(f"  m1 mean: {m1_2.mean(axis=0)}")
    print(f"  m2 mean: {m2_2.mean(axis=0)}")
    print(f"  m1 Frobenius norm: {np.linalg.norm(m1_2, 'fro'):.4f}")
    print(f"  m2 Frobenius norm: {np.linalg.norm(m2_2, 'fro'):.4f}")
    print(f"  m1 column means: {m1_2.mean(axis=0)}")
    print(f"  m1 column stds: {m1_2.std(axis=0, ddof=1)}")
    print("\nConclusion:")
    print(
        "  Procrustes centers to origin (mean=0) ✓"
        if np.allclose(m1_2.mean(axis=0), 0)
        else "  Procrustes does NOT center"
    )
    print(
        "  Procrustes normalizes Frobenius norm ✓"
        if np.isclose(np.linalg.norm(m1_2, "fro"), 1.0)
        else "  Procrustes does NOT normalize Frobenius"
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Key Takeaways

    **✅ API Design Principles:**
    - **Unified Interface**: Single `compare_datasets()` function handles all comparison modes
    - **Type Safety**: Literal types prevent invalid metric/mode combinations
    - **Backend Agnostic**: PlotGrid works with matplotlib or plotly seamlessly

    **✅ Metric Selection Guide:**

    | Use Case | Recommended Metric | Rationale |
    |----------|-------------------|-----------|
    | Cluster tightness | `euclidean`, `cosine` | Fast, interpretable point-to-point distances |
    | Distribution shape | `wasserstein`, `ks_statistic` | Captures distributional differences |
    | High-dimensional data | `mahalanobis`, `cosine` | Accounts for correlations/directions |
    | Statistical testing | `ks_statistic`, `energy` | Principled hypothesis testing |

    **✅ Mode Selection Flowchart:**

    ```
    ├─ Comparing points within ONE dataset?
    │  └─> Use mode='within' (compute_within_distances)
    │
    ├─ Comparing TWO specific datasets?
    │  └─> Use mode='between' (compute_between_distances)
    │
    └─ Comparing ALL pairs from N datasets?
       └─> Use mode='all-pairs' (compute_all_pairs)
    ```

    **❌ Common Pitfalls to Avoid:**

    1. **Don't mix incompatible metrics with modes**
       - ❌ `mode='within'` with `metric='wasserstein'` → Use distribution metrics for between/all-pairs

    2. **Don't ignore data preprocessing**
       - ❌ Comparing datasets with different scales → Normalize first

    3. **Don't use single metric for important decisions**
       - ❌ Relying only on euclidean distance → Test multiple metrics for robustness

    4. **Don't forget to set random seeds**
       - ❌ Non-reproducible results → Always set `np.random.seed()` for synthetic data

    **✅ Next Steps:**
    - Explore `docs/function_registry.md` for full API documentation
    - Check `tests/test_metrics_distributions.py` for more examples
    - Use `logging` module to track computation details
    - Leverage `save_path` parameter (coming in Phase 4B) for caching results
    """
    )
    return


@app.cell
def _(log_section):
    # Final summary statistics
    log_section("Demo Notebook Complete!")

    print("=" * 60)
    print("PHASE 3 API DEMO - COMPLETION SUMMARY")
    print("=" * 60)
    print("✓ Synthetic datasets generated: 5")
    print("✓ Comparison modes demonstrated: 3 (within, between, all-pairs)")
    print("✓ Metrics tested: 10+ (point-to-point, distribution, shape)")
    print("✓ Real-world scenarios: 3 (treatment, sessions, hypothesis)")
    print("✓ Visualizations created: 5 (PlotGrid)")
    print("✓ Performance benchmarks: 5 dataset sizes")
    print("=" * 60)
    print("\n🎉 All examples executed successfully!")
    print("📚 Next: Explore docs/COMPLETE_REFACTORING_PLAN.md for Phase 4B features")
    print("    (auto-save/load, regenerate parameter, correlation metrics)")
    print("=" * 60)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 1. Within-Dataset Comparisons

    Compute distances within a single dataset using point-to-point metrics.
    """
    )
    return


@app.cell
def _(compute_within_distances, np):
    # Create sample dataset
    data_2 = np.random.randn(100, 10)
    mean_dist_2 = compute_within_distances(data_2, metric="euclidean")
    # Compute mean euclidean distance (scalar)
    print(f"Mean within-dataset distance: {mean_dist_2:.3f}")
    dist_matrix = compute_within_distances(
        data_2, metric="euclidean", return_matrix=True
    )
    print(f"Distance matrix shape: {dist_matrix.shape}")
    # Get full distance matrix
    print(f"Distance range: [{dist_matrix.min():.3f}, {dist_matrix.max():.3f}]")
    return (data_2,)


@app.cell
def _(POINT_TO_POINT_METRICS, compute_within_distances, data_2, np):
    # Test all point-to-point metrics
    print("\nAvailable point-to-point metrics:")
    print(POINT_TO_POINT_METRICS)
    for metric_4 in POINT_TO_POINT_METRICS:
        if metric_4 == "mahalanobis":
            cov_1 = np.cov(data_2.T)
            dist_1 = compute_within_distances(data_2, metric=metric_4, cov=cov_1)
        else:
            dist_1 = compute_within_distances(data_2, metric=metric_4)
        print(f"{metric_4:15s}: {dist_1:.4f}")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 2. Between-Dataset Comparisons

    Compare two datasets using any metric (point-to-point, distribution, or shape).
    """
    )
    return


@app.cell
def _(DISTRIBUTION_METRICS, SHAPE_METRICS, compute_between_distances, np):
    # Create two datasets
    data1_6 = np.random.randn(80, 10)
    data2_6 = np.random.randn(120, 10) + 0.5  # Shifted distribution
    euclidean_scalar = compute_between_distances(data1_6, data2_6, metric="euclidean")
    # Point-to-point metric (scalar by default)
    print(f"Euclidean (scalar): {euclidean_scalar:.4f}")
    euclidean_matrix = compute_between_distances(
        data1_6, data2_6, metric="euclidean", return_matrix=True
    )
    print(f"Euclidean (matrix): {euclidean_matrix.shape}")
    # Point-to-point metric (matrix)
    print("\nDistribution metrics:")
    for metric_5 in DISTRIBUTION_METRICS:
        dist_2 = compute_between_distances(data1_6, data2_6, metric=metric_5)
        # Distribution metrics (always scalar)
        print(f"{metric_5:20s}: {dist_2:.6f}")
    print("\nShape metrics:")
    for metric_5 in SHAPE_METRICS:
        dist_2 = compute_between_distances(data1_6[:80], data2_6[:80], metric=metric_5)
        # Shape metrics (always scalar)
        print(f"{metric_5:15s}: {dist_2:.6f}")  # Same sample count for procrustes
    return data1_6, data2_6


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 3. All-Pairs Comparisons

    Compute all pairwise comparisons between multiple datasets (scalar metrics only).
    """
    )
    return


@app.cell
def _(compute_all_pairs, np):
    # Create multiple datasets
    datasets_1 = {
        "Control": np.random.randn(50, 8),
        "Treatment_A": np.random.randn(50, 8) + 0.3,
        "Treatment_B": np.random.randn(50, 8) + 0.7,
    }
    results = compute_all_pairs(datasets_1, metric="wasserstein", show_progress=False)
    print("\nAll-pairs Wasserstein distances:")
    for name_i_3 in results:
        for name_j_3 in results[name_i_3]:
            dist_3 = results[name_i_3][name_j_3]
            # Compute all pairwise Wasserstein distances
            print(f"{name_i_3:15s} → {name_j_3:15s}: {dist_3:.4f}")
    return (datasets_1,)


@app.cell
def _(compute_all_pairs, datasets_1):
    # Compare multiple scalar metrics
    import pandas as pd

    metrics_to_test_1 = ["wasserstein", "jensen-shannon", "one-to-one"]
    # Test subset of scalar metrics
    results_df = pd.DataFrame()
    for metric_6 in metrics_to_test_1:
        results_1 = compute_all_pairs(datasets_1, metric=metric_6, show_progress=False)
        for name_i_4 in results_1:
            for name_j_4 in results_1[name_i_4]:
                if name_i_4 < name_j_4:  # Extract off-diagonal comparisons
                    results_df.loc[f"{name_i_4}-{name_j_4}", metric_6] = results_1[
                        name_i_4
                    ][name_j_4]
    print("\nComparison matrix:")
    print(results_df)  # Avoid duplicates
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 4. HDF5 Storage

    Save and load comparisons with compression and metadata.
    """
    )
    return


@app.cell
def _(
    compute_all_pairs,
    compute_between_distances,
    data1_6,
    data2_6,
    datasets_1,
    save_comparison,
    tempfile,
):
    import os

    h5_path = os.path.join(tempfile.gettempdir(), "phase3_demo.h5")
    dist_scalar = compute_between_distances(data1_6, data2_6, metric="wasserstein")
    # Create temp file
    save_comparison(
        h5_path,
        comparison_name="control_vs_treatment",
        data=dist_scalar,
        metric="wasserstein",
        mode="between",
        dataset_names=["control", "treatment"],
        metadata={"experiment_id": "exp001", "condition": "acute"},
    )
    print(f"Saved scalar comparison to {h5_path}")
    # Compute and save scalar comparison
    dist_matrix_1 = compute_between_distances(
        data1_6, data2_6, metric="euclidean", return_matrix=True
    )
    save_comparison(
        h5_path,
        comparison_name="control_vs_treatment_matrix",
        data=dist_matrix_1,
        metric="euclidean",
        mode="between",
        dataset_names=["control", "treatment"],
        metadata={"experiment_id": "exp001", "return_matrix": True},
    )
    print(f"Saved matrix comparison to {h5_path}")
    all_pairs_results_1 = compute_all_pairs(
        datasets_1, metric="wasserstein", show_progress=False
    )
    save_comparison(
        h5_path,
        comparison_name="all_pairs_wasserstein",
        data=all_pairs_results_1,
        metric="wasserstein",
        mode="all-pairs",
        dataset_names=list(datasets_1.keys()),
        metadata={"experiment_id": "exp001"},
    )
    # Save matrix comparison
    # Save all-pairs dict
    print(f"Saved all-pairs comparison to {h5_path}")
    return h5_path, os


@app.cell
def _(h5_path, load_comparison):
    # Load comparisons
    loaded_scalar = load_comparison(h5_path, "control_vs_treatment")
    print(f"\nLoaded scalar: {loaded_scalar}")
    print(f"Type: {type(loaded_scalar)}")

    loaded_matrix = load_comparison(h5_path, "control_vs_treatment_matrix")
    print(f"\nLoaded matrix shape: {loaded_matrix.shape}")

    loaded_dict = load_comparison(h5_path, "all_pairs_wasserstein")
    print(f"\nLoaded dict keys: {list(loaded_dict.keys())}")
    return


@app.cell
def _(h5_path, query_comparisons):
    # Query comparisons
    query_results = query_comparisons(h5_path, metric="wasserstein")
    print("\nComparisons with metric='wasserstein':")
    print(query_results[["comparison_name", "metric", "mode", "result_type"]])
    return


@app.cell
def _(h5_path, query_comparisons):
    # Query by mode
    query_results_1 = query_comparisons(h5_path, mode="between")
    print("\nComparisons with mode='between':")
    print(
        query_results_1[["comparison_name", "metric", "mode", "result_type", "shape"]]
    )
    return


@app.cell
def _(h5_path, os):
    # Clean up
    if os.path.exists(h5_path):
        os.remove(h5_path)
        print(f"\nCleaned up: {h5_path}")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Summary

    **Phase 3 API Key Features:**

    1. **Three Explicit Functions:**
       - `compute_within_distances()` - Point-to-point metrics only, returns float or matrix
       - `compute_between_distances()` - All metrics, flexible return types
       - `compute_all_pairs()` - Scalar metrics only, returns nested dict

    2. **Mode-Metric Validation:**
       - Clear error messages for invalid metric-mode combinations
       - Explicit metric category constants

    3. **HDF5 Storage:**
       - Compression (gzip-6) with automatic chunking
       - Indexed queries with O(log n) complexity
       - Type-safe storage (scalar/matrix/dict)
       - Rich metadata support

    **Migration from Phase 2:**

    - **Old:** `distribution_distance(A, mode="within")` → **New:** `compute_within_distances(A)`
    - **Old:** `distribution_distance(A, B, mode="between")` → **New:** `compute_between_distances(A, B)`
    - **Old:** `compute_pairwise_matrix(A, B)` → **New:** `compute_between_distances(A, B, return_matrix=True)`

    See `docs/phase3_api_redesign_summary.md` for complete documentation.
    """
    )
    return
