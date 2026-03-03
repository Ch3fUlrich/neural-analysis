import marimo

__generated_with = "0.18.3"

app = marimo.App(width="full")


@app.cell(hide_code=True)
def __():
    import marimo as mo

    return mo


@app.cell
def _(mo):
    mo.md(
        r"""
    # New PlotGrid System - Quick Examples

    The neural_analysis package now includes a flexible, metadata-driven plotting system that makes creating complex multi-panel visualizations easy. Key features:

    ## Features
    - **DataFrame-driven**: Specify plots using structured data
    - **Multiple traces per subplot**: Use `subplot_position` to overlay multiple data series
    - **Automatic layout**: Grid size auto-calculated from number of plots
    - **Color schemes**: Automatic color assignment by group
    - **Single API**: Works for scatter, line, histogram, heatmap, 3D plots
    - **Both backends**: Matplotlib and Plotly supported
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # Metrics Package Examples

    This notebook demonstrates the neural_analysis metrics package with:
    - Distance metrics (Euclidean, Mahalanobis, Cosine)
    - Distribution comparison (6 statistical methods)
    - Outlier detection (5 different algorithms)
    - Visual comparisons showing strengths and weaknesses
    - Ground truth validation
    """
    )
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go

    from neural_analysis.metrics import (
        compare_datasets,  # Replaces compare_distributions and compare_distribution_groups
        cosine_similarity,
        euclidean_distance,
        filter_outlier,
        mahalanobis_distance,  # Use new unified function
    )
    from neural_analysis.plotting import (
        GridLayoutConfig,  # Grid layout configuration
        PlotConfig,
        PlotGrid,  # New flexible grid system
        PlotSpec,
        add_trace_to_subplot,
        plot_comparison_grid,
        plot_heatmap,
    )

    np.random.seed(42)
    print("Imports successful!")
    return (
        GridLayoutConfig,
        PlotConfig,
        PlotGrid,
        PlotSpec,
        add_trace_to_subplot,
        compare_datasets,
        cosine_similarity,
        euclidean_distance,
        filter_outlier,
        go,
        mahalanobis_distance,
        np,
        pd,
        plot_comparison_grid,
        plot_heatmap,
        sys,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 1. Distance Metrics Comparison

    ### 1.1 Euclidean vs Mahalanobis Distance

    Euclidean distance treats all dimensions equally, while Mahalanobis accounts for correlations.
    """
    )
    return


@app.cell
def _(euclidean_distance, mahalanobis_distance, np):
    # Create two distributions: one spherical, one elongated
    n_points = 200

    # Spherical distribution (uncorrelated)
    spherical = np.random.randn(n_points, 2)

    # Elongated distribution (correlated)
    cov_elongated = np.array([[1.0, 0.9], [0.9, 1.0]])
    elongated = np.random.multivariate_normal([0, 0], cov_elongated, n_points)

    # Test points at same Euclidean distance but different Mahalanobis distances
    test_point_aligned = np.array([1.5, 1.5])  # Along correlation
    test_point_perpendicular = np.array([1.5, -1.5])  # Perpendicular to correlation

    # Compute distances
    mean_elongated = np.mean(elongated, axis=0)
    cov_elongated_emp = np.cov(elongated.T)

    euc_aligned = euclidean_distance(test_point_aligned, mean_elongated)
    euc_perp = euclidean_distance(test_point_perpendicular, mean_elongated)
    maha_aligned = mahalanobis_distance(
        test_point_aligned, mean_elongated, cov_elongated_emp
    )
    maha_perp = mahalanobis_distance(
        test_point_perpendicular, mean_elongated, cov_elongated_emp
    )

    print("Test Point Aligned [1.5, 1.5]:")
    print(f"  Euclidean: {euc_aligned:.3f}")
    print(f"  Mahalanobis: {maha_aligned:.3f}")
    print("\nTest Point Perpendicular [1.5, -1.5]:")
    print(f"  Euclidean: {euc_perp:.3f}")
    print(f"  Mahalanobis: {maha_perp:.3f}")
    print("\nKey Insight:")
    print(f"  Euclidean distances are equal: {abs(euc_aligned - euc_perp) < 0.01}")
    print(
        f"  Mahalanobis correctly identifies perpendicular as farther: {maha_perp > maha_aligned}"
    )
    return elongated, spherical, test_point_aligned, test_point_perpendicular


@app.cell
def _(
    PlotConfig,
    PlotGrid,
    PlotSpec,
    add_trace_to_subplot,
    elongated,
    go,
    spherical,
    test_point_aligned,
    test_point_perpendicular,
):
    _plot_specs = [
        PlotSpec(
            data=spherical,
            plot_type="scatter",
            title="Spherical Distribution",
            label="Spherical",
            color="blue",
            alpha=0.5,
            marker_size=4,
        ),
        PlotSpec(
            data=elongated,
            plot_type="scatter",
            title="Elongated Distribution (Correlated)",
            label="Elongated",
            color="green",
            alpha=0.5,
            marker_size=4,
        ),
    ]
    _grid = PlotGrid(
        plot_specs=_plot_specs,
        config=PlotConfig(title="Distance Metric Comparison", figsize=(10, 5)),
        backend="plotly",
    )
    _fig = _grid.plot()
    for col in [1, 2]:
        add_trace_to_subplot(
            _fig,
            go.Scatter(
                x=[test_point_aligned[0]],
                y=[test_point_aligned[1]],
                mode="markers",
                marker=dict(size=12, color="red", symbol="x"),
                name="Aligned",
                showlegend=col == 2,
            ),
            row=1,
            col=col,
        )
        add_trace_to_subplot(
            _fig,
            go.Scatter(
                x=[test_point_perpendicular[0]],
                y=[test_point_perpendicular[1]],
                mode="markers",
                marker=dict(size=12, color="orange", symbol="x"),
                name="Perpendicular",
                showlegend=col == 2,
            ),
            row=1,
            col=col,
        )
    _fig.update_xaxes(title_text="X", range=[-4, 4])
    _fig.update_yaxes(title_text="Y", range=[-4, 4])
    _fig.show()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### 1.2 Cosine Similarity for Direction

    Cosine similarity measures directional similarity, invariant to magnitude.
    """
    )
    return


@app.cell
def _(cosine_similarity, euclidean_distance, np):
    # Create vectors with different magnitudes but similar directions
    v1 = np.array([1, 2, 3])
    v2 = np.array([2, 4, 6])  # Same direction, 2x magnitude
    v3 = np.array([1, 2, -3])  # Different direction
    v4 = np.array([-1, -2, -3])  # Opposite direction

    print("Cosine Similarity Examples:")
    print(
        f"v1 vs v2 (scaled version): {cosine_similarity(v1, v2):.3f} → 1.0 (identical direction)"
    )
    print(f"v1 vs v3 (partial match): {cosine_similarity(v1, v3):.3f}")
    print(
        f"v1 vs v4 (opposite): {cosine_similarity(v1, v4):.3f} → -1.0 (opposite direction)"
    )

    print("\nEuclidean distances (NOT scale-invariant):")
    print(f"v1 vs v2: {euclidean_distance(v1, v2):.3f}")
    print(f"v1 vs v3: {euclidean_distance(v1, v3):.3f}")
    print(f"v1 vs v4: {euclidean_distance(v1, v4):.3f}")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 2. Distribution Comparison Methods

    Compare 6 statistical methods on synthetic distributions with known ground truth.
    """
    )
    return


@app.cell
def _(compare_datasets, np, pd):
    # Create test distributions with controlled differences
    n_samples = 300
    scenarios = {
        "Identical": (
            np.random.randn(n_samples, 3),
            np.random.randn(n_samples, 3),
            "Same distribution",
        ),
        "Small Shift": (
            np.random.randn(n_samples, 3),
            np.random.randn(n_samples, 3) + 0.5,
            "Slight location difference",
        ),
        "Large Shift": (
            np.random.randn(n_samples, 3),
            np.random.randn(n_samples, 3) + 2.0,
            "Major location difference",
        ),
        "Different Shape": (
            np.random.randn(n_samples, 3),
            np.random.randn(n_samples, 3) * 2.0,
            "Same center, different spread",
        ),
    }
    # Ground truth: 4 scenarios with known similarity
    metrics = [
        "wasserstein",
        "kolmogorov-smirnov",
        "jensen-shannon",
        "euclidean",
        "mahalanobis",
        "cosine",
    ]
    results = {}
    for _metric in metrics:
        results[_metric] = {}
        for _scenario_name, (_p1, _p2, _desc) in scenarios.items():
            _result = compare_datasets(_p1, _p2, mode="between", metric=_metric)
            _dist = _result["value"] if isinstance(_result, dict) else _result
            results[_metric][_scenario_name] = _dist
    df_results = pd.DataFrame(results).T
    print("\nDistribution Comparison Results:")
    print("=" * 80)
    print(df_results.round(3))
    print("\nInterpretation:")
    print("- Lower = more similar (except cosine: higher = more similar)")
    print("- Identical should be ~0 for distance metrics, ~1 for cosine")
    # Test all metrics
    # Display as DataFrame
    print("- Large Shift should show biggest difference")
    return metrics, scenarios


@app.cell
def _(GridLayoutConfig, PlotConfig, PlotGrid, PlotSpec, scenarios):
    # Now we can use PlotGrid with multiple traces per subplot!
    # Create plot specifications where multiple specs share the same subplot_position
    _plot_specs = []
    for _i, (_scenario_name, (_p1, _p2, _desc)) in enumerate(scenarios.items()):
        _plot_specs.append(
            PlotSpec(
                data=_p1[:, :2],
                plot_type="scatter",
                subplot_position=_i,
                title=_scenario_name,
                label="Dist 1",
                color="blue",
                alpha=0.4,
                marker_size=3,
            )
        )
        _plot_specs.append(
            PlotSpec(
                data=_p2[:, :2],
                plot_type="scatter",
                subplot_position=_i,
                label="Dist 2",
                color="red",
                alpha=0.4,
                marker_size=3,
            )
        )  # First distribution (blue)
    _grid = PlotGrid(
        plot_specs=_plot_specs,
        config=PlotConfig(
            title="Ground Truth Scenarios for Distribution Comparison", figsize=(10, 8)
        ),
        layout=GridLayoutConfig(rows=2, cols=2),
        backend="plotly",
    )
    _fig = _grid.plot()
    _fig.update_xaxes(title_text="Dimension 1")  # First 2 dimensions
    _fig.update_yaxes(title_text="Dimension 2")
    _fig.show()  # Group by scenario
    # Create grid - it will automatically arrange 4 subplots (one per unique subplot_position)
    print(
        "\n✓ Using PlotGrid with subplot_position for multiple traces per subplot!"
    )  # Second distribution (red)  # Same subplot as above
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### 2.1 Method Sensitivity Analysis

    Test how each metric responds to increasing distribution shift.
    """
    )
    return


@app.cell
def _(
    GridLayoutConfig,
    PlotConfig,
    PlotGrid,
    PlotSpec,
    compare_datasets,
    metrics,
    np,
):
    # Vary shift magnitude
    shifts = np.linspace(0, 3, 15)
    p1_base = np.random.randn(200, 3)
    sensitivity_results = {_metric: [] for _metric in metrics}
    for shift in shifts:
        p2_shifted = np.random.randn(200, 3) + shift
        for _metric in metrics:
            _result = compare_datasets(
                p1_base, p2_shifted, mode="between", metric=_metric
            )
            _dist = _result["value"] if isinstance(_result, dict) else _result
            sensitivity_results[_metric].append(_dist)
    _plot_specs = []
    colors = ["blue", "red", "green", "orange", "purple", "brown"]
    for _i, _metric in enumerate(metrics):
        # Use PlotGrid to create multi-line plot (all metrics in one subplot)
        line_data = np.column_stack([shifts, sensitivity_results[_metric]])
        _plot_specs.append(
            PlotSpec(
                data=line_data,
                plot_type="line",
                subplot_position=0,
                label=_metric,
                color=colors[_i % len(colors)],
                line_width=2,
                alpha=0.8,
            )
        )
    _grid = PlotGrid(
        plot_specs=_plot_specs,
        config=PlotConfig(
            title="Metric Sensitivity to Distribution Shift", figsize=(8, 5)
        ),
        layout=GridLayoutConfig(rows=1, cols=1),
        backend="plotly",
    )
    _fig = _grid.plot()
    _fig.update_xaxes(
        title_text="Shift Magnitude (std devs)"
    )  # Create x-y pairs for line plot
    _fig.update_yaxes(title_text="Distance/Similarity")
    _fig.show()
    print("\n✓ Using PlotGrid for multi-line plot with all metrics overlaid!")
    print("\nKey Observations:")
    print("- Wasserstein: Linear response to shift (good for quantifying displacement)")
    print("- K-S: Saturates quickly (good for detecting any difference)")
    print(
        "- Euclidean: Simple linear response (centroid distance)"
    )  # All in same subplot
    print("- Mahalanobis: Accounts for variance (more stable)")
    # Create single subplot with all metrics
    print("- Cosine: Relatively stable (direction-based)")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 2.5 Shape Distance Validation

    Validate that shape distance methods (soft-matching, one-to-one, procrustes) maintain theoretical ordering and are on the same scale.

    **Theoretical ordering**: soft-matching ≤ one-to-one ≤ procrustes

    This ordering holds because:
    - **Soft-matching**: Allows fractional assignment (most flexible)
    - **One-to-one**: Enforces hard assignment (less flexible)
    - **Procrustes**: Preserves point correspondence (least flexible)

    All methods normalize matrices to unit Frobenius norm and use squared distances for comparability.
    """
    )
    return


@app.cell
def _(sys):
    # Import shape distance function
    from pathlib import Path

    # Run validation using the test script
    examples_dir = Path.cwd() / "examples"
    if str(examples_dir) not in sys.path:
        sys.path.insert(0, str(examples_dir))
    # Add examples directory to path for imports
    from test_shape_distance_validation import (
        print_validation_summary,
        validate_shape_distance_ordering,
    )

    results_1 = validate_shape_distance_ordering(
        n_pairs=10, n_samples=100, neuron_range=(15, 50), repeats=5, seed=42
    )
    # Run validation with smaller parameters for notebook demo
    print_validation_summary(
        results_1
    )  # Reduced for faster execution  # Smaller range for faster execution  # Fewer repeats for faster execution
    return (results_1,)


@app.cell
def _(GridLayoutConfig, PlotConfig, PlotGrid, PlotSpec, np, results_1):
    if "summary" in results_1:
        summary = results_1["summary"]
        methods_to_plot = ["soft-matching-exact", "one-to-one", "procrustes"]
        distances = [summary[m]["mean"] for m in methods_to_plot if m in summary]
        method_labels = [
            m.replace("-", " ").title() for m in methods_to_plot if m in summary
        ]
        if distances:
            _plot_specs = [
                PlotSpec(
                    data=np.array(distances),
                    plot_type="bar",
                    title="Mean Shape Distances Across Methods",
                    color="steelblue",
                    alpha=0.7,
                    kwargs={
                        "x": list(range(len(method_labels))),
                        "show_values": True,
                        "value_format": ".4f",
                        "x_label": "Method",
                        "y_label": "Mean Distance",
                        "set_xticks": list(range(len(method_labels))),
                        "set_xticklabels": method_labels,
                    },
                )
            ]
            _grid = PlotGrid(
                plot_specs=_plot_specs,
                config=PlotConfig(figsize=(8, 5)),
                layout=GridLayoutConfig(rows=1, cols=1),
                backend="plotly",
            )
            _fig = _grid.plot()
            _fig.update_xaxes(title_text="Method")
            _fig.update_yaxes(title_text="Mean Distance")
            _fig.show()
            print(
                "\n✓ Methods should maintain ordering: soft-matching ≤ one-to-one ≤ procrustes"
            )
    return


@app.cell
def _(np):
    # Generate datasets with distinct clusters and visualize using MDS
    from neural_analysis import generate_shape_distance_datasets
    from neural_analysis.plotting.shape_distance import plot_shape_distance_mds

    print("Generating datasets with distinct clusters for MDS visualization...")
    # Generate datasets with distinct cluster structure
    datasets, labels = generate_shape_distance_datasets(
        n_datasets=30,
        n_clusters=5,
        min_neurons=20,
        max_neurons=50,
        n_features=50,
        seed=42,
    )
    print(f"Generated {len(datasets)} datasets with {len(np.unique(labels))} clusters")
    print(
        f"Cluster distribution: {dict(zip(*np.unique(labels, return_counts=True)))}"
    )  # Reduced for faster execution
    print("\nComputing distance matrices and creating MDS visualizations...")
    _fig = plot_shape_distance_mds(
        datasets=datasets,
        methods=["procrustes", "one-to-one", "soft-matching"],
        labels=labels,
        backend="matplotlib",
        figsize=(12, 12),
        max_neurons=30,
        show_progress=True,
    )
    print("\n✓ MDS plots show how well each method separates clusters")
    # Visualize using MDS (distances computed automatically using compare_datasets)
    print(
        "  Good clustering = points of same color cluster together"
    )  # Reduced for faster execution  # Limit for speed
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 3. Outlier Detection Comparison

    Test 5 methods on synthetic data with known outliers.
    """
    )
    return


@app.cell
def _(filter_outlier, np, pd):
    # Create ground truth data
    n_inliers = 200
    n_outliers = 20
    n_dims = 3
    inliers = np.random.randn(n_inliers, n_dims) * 0.5
    # Inliers: tight cluster
    outliers = np.random.randn(n_outliers, n_dims) * 3 + np.array([5, 5, 5])
    data_with_outliers = np.vstack([inliers, outliers])
    # Outliers: far from cluster
    ground_truth = np.concatenate([np.ones(n_inliers), np.zeros(n_outliers)]).astype(
        bool
    )
    methods = ["iqr", "zscore", "isolation", "lof", "elliptic"]
    # Combine
    contamination = n_outliers / (n_inliers + n_outliers)
    outlier_results = {}
    for _method in methods:
        # Test all methods
        filtered, _mask = filter_outlier(
            data_with_outliers,
            method=_method,
            contamination=contamination,
            threshold=3.0,
            return_mask=True,
        )
        true_positives = np.sum(_mask & ground_truth)
        false_positives = np.sum(_mask & ~ground_truth)
        true_negatives = np.sum(~_mask & ~ground_truth)
        false_negatives = np.sum(~_mask & ground_truth)
        precision = (
            true_positives / (true_positives + false_positives)
            if true_positives + false_positives > 0
            else 0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if true_positives + false_negatives > 0
            else 0
        )
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if precision + recall > 0
            else 0
        )
        outlier_results[_method] = {
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "Detected Outliers": np.sum(~_mask),
            "Kept Inliers": np.sum(_mask),
        }
    df_outlier = pd.DataFrame(outlier_results).T
    print("\nOutlier Detection Performance:")
    print("=" * 80)
    print(df_outlier.round(3))
    print(
        f"\nGround Truth: {n_inliers} inliers, {n_outliers} outliers"
    )  # Compute metrics
    print("\nBest Method: Highest F1-Score balances precision and recall")
    return contamination, data_with_outliers, inliers, methods, outliers


@app.cell
def _(
    GridLayoutConfig,
    PlotConfig,
    PlotGrid,
    PlotSpec,
    contamination,
    data_with_outliers,
    filter_outlier,
    inliers,
    methods,
    outliers,
):
    # Visualize outlier detection results using PlotGrid
    _plot_specs = []
    _plot_specs.append(
        PlotSpec(
            data=inliers,
            plot_type="scatter3d",
            subplot_position=0,
            label="True Inliers",
            color="blue",
            marker_size=3,
            title="Original Data",
        )
    )
    # Subplot 0: Original data with ground truth
    _plot_specs.append(
        PlotSpec(
            data=outliers,
            plot_type="scatter3d",
            subplot_position=0,
            label="True Outliers",
            color="red",
            marker_size=5,
        )
    )
    for idx, _method in enumerate(methods, start=1):
        _, _mask = filter_outlier(
            data_with_outliers,
            method=_method,
            contamination=contamination,
            return_mask=True,
        )
        kept = data_with_outliers[_mask]
        removed = data_with_outliers[~_mask]
        _plot_specs.append(
            PlotSpec(
                data=kept,
                plot_type="scatter3d",
                subplot_position=idx,
                label="Kept",
                color="green",
                marker_size=3,
                alpha=0.6,
                title=_method.upper(),
            )
        )
        _plot_specs.append(
            PlotSpec(
                data=removed,
                plot_type="scatter3d",
                subplot_position=idx,
                label="Removed",
                color="orange",
                marker_size=3,
            )
        )
    _grid = PlotGrid(
        plot_specs=_plot_specs,
        layout=GridLayoutConfig(rows=2, cols=3),
        config=PlotConfig(
            title="Outlier Detection Method Comparison (3D)", figsize=(12, 10)
        ),
        backend="plotly",
    )
    _fig = _grid.plot()
    _fig.show()
    # Each outlier detection method gets its own subplot (positions 1-5)
    # Create the grid with PlotGrid
    print(
        "\n✓ Using PlotGrid with subplot_position for 3D outlier detection visualization!"
    )  # Add kept points (inliers)  # Add removed points (outliers)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 4. Group Distribution Analysis

    Compare multiple groups (e.g., neural activity across trials/conditions).
    """
    )
    return


@app.cell
def _(compare_datasets, np, pd):
    # Simulate neural data: 3 conditions with different activity patterns
    n_trials = 40
    _n_neurons = 50
    conditions = {
        "Baseline": np.random.randn(n_trials, _n_neurons) * 0.5,
        "Stimulus_A": np.random.randn(n_trials, _n_neurons) * 0.5
        + np.array([1.0] * _n_neurons),
        "Stimulus_B": np.random.randn(n_trials, _n_neurons) * 0.8
        + np.array([0.5] * _n_neurons),
    }
    between_results = compare_datasets(
        conditions, mode="all-pairs", metric="wasserstein"
    )
    condition_names = list(conditions.keys())
    dist_matrix = np.zeros((len(condition_names), len(condition_names)))
    for _i, name_i in enumerate(condition_names):
        for _j, name_j in enumerate(condition_names):
            dist_matrix[_i, _j] = between_results[name_i][name_j]
    print("\nBetween-Condition Distance Matrix (Wasserstein):")
    print(
        pd.DataFrame(dist_matrix, index=condition_names, columns=condition_names).round(
            3
        )
    )
    # Compute within-group variability for each condition
    inside_results = {}
    for name in condition_names:
        _within_dist = compare_datasets(
            conditions[name], mode="within", metric="euclidean"
        )
        inside_results[name] = _within_dist
    # Between-group comparison
    print("\nWithin-Condition Variability:")
    for _i, name in enumerate(condition_names):
        # Convert to distance matrix
        # Within-group variability
        _mean = (
            inside_results[name]
            if isinstance(inside_results[name], (int, float))
            else float(inside_results[name])
        )
        print(f"{name}: mean={_mean:.3f}")
    return condition_names, dist_matrix


@app.cell
def _(PlotConfig, condition_names, dist_matrix, plot_heatmap):
    _fig = plot_heatmap(
        dist_matrix,
        config=PlotConfig(
            title="Condition Similarity Matrix (Lower = More Similar)",
            xlabel="Condition",
            ylabel="Condition",
            figsize=(7, 6),
        ),
        x_labels=condition_names,
        y_labels=condition_names,
        cmap="Viridis",
        show_values=True,
        value_format=".2f",
        colorbar_label="Euclidean Distance",
        backend="plotly",
    )
    _fig.show()
    print("\nInterpretation:")
    print("- Baseline vs Stimulus_A: Largest difference (highest activity change)")
    print("- Stimulus_A vs Stimulus_B: Moderate difference)")
    print("- Diagonal: Zero (self-comparison)")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 5. Real-World Example: Neural Manifold Analysis

    Simulate neural population responses and analyze with metrics.
    """
    )
    return


@app.cell
def _(compare_datasets, np, pd):
    # Simulate neural population with 2D preferred direction manifold
    _n_neurons = 100
    n_timepoints = 50
    n_trials_per_angle = 20
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    neural_data = {}
    for angle in angles:
        preferred_pattern = np.array([np.cos(angle), np.sin(angle)])
        trials = []
        for _ in range(n_trials_per_angle):  # Each angle has a preferred neural pattern
            manifold_activity = preferred_pattern + np.random.randn(2) * 0.2
            projection_matrix = np.random.randn(_n_neurons, 2)
            neural_activity = (
                projection_matrix @ manifold_activity
                + np.random.randn(_n_neurons) * 0.1
            )  # Generate trials with noise
            trials.append(neural_activity)
        neural_data[f"angle_{int(np.degrees(angle))}"] = np.array(trials)
    print(
        "\nNeural Manifold Analysis:"
    )  # High-dimensional neural activity projects onto 2D manifold
    print("Comparing activity patterns across movement directions\n")
    _result_adj = compare_datasets(
        neural_data["angle_0"],
        neural_data["angle_45"],
        mode="between",
        metric="mahalanobis",
    )  # Random projection to high-dim space
    _result_opp = compare_datasets(
        neural_data["angle_0"],
        neural_data["angle_180"],
        mode="between",
        metric="mahalanobis",
    )
    dist_adjacent = (
        _result_adj["value"] if isinstance(_result_adj, dict) else _result_adj
    )
    dist_opposite = (
        _result_opp["value"] if isinstance(_result_opp, dict) else _result_opp
    )
    print(f"Distance between adjacent angles (0° vs 45°): {dist_adjacent:.3f}")
    print(f"Distance between opposite angles (0° vs 180°): {dist_opposite:.3f}")
    print(
        f"\nValidation: Opposite directions should be farther: {dist_opposite > dist_adjacent}"
    )
    angle_names = list(neural_data.keys())
    angle_matrix = compare_datasets(neural_data, mode="all-pairs", metric="euclidean")
    matrix_vals = np.array(
        [
            [angle_matrix[name_i][name_j] for name_j in angle_names]
            for name_i in angle_names
        ]
    )
    print("\nFull Angular Distance Matrix:")
    # Compare adjacent angles (should be similar) vs opposite angles (should differ)
    # Adjacent angles
    # Opposite angles
    # Full comparison matrix
    print(pd.DataFrame(matrix_vals, index=angle_names, columns=angle_names).round(2))
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 6. Summary: Method Selection Guide

    ### Distance Metrics

    | Metric | Best For | Pros | Cons |
    |--------|----------|------|------|
    | **Euclidean** | Spherical clusters, equal feature importance | Simple, interpretable, fast | Ignores correlations, scale-sensitive |
    | **Mahalanobis** | Elongated/correlated clusters | Accounts for covariance structure | Requires sufficient samples, computationally expensive |
    | **Cosine** | High-dimensional sparse data, direction matters | Scale-invariant, good for angles | Ignores magnitude differences |

    **Decision Tree:**
    - **Data is correlated/elongated?** → Use Mahalanobis
    - **Direction matters more than magnitude?** → Use Cosine
    - **Simple spherical clusters?** → Use Euclidean

    ### Distribution Comparison Metrics

    | Metric | Best For | Pros | Cons |
    |--------|----------|------|------|
    | **Wasserstein** | Quantifying displacement, optimal transport | Interpretable distance, geometric meaning | Computationally expensive for large samples |
    | **Kolmogorov-Smirnov** | Fast hypothesis testing, 1D data | Fast, distribution-free, exact p-values | 1D only, sensitive to location not shape |
    | **Jensen-Shannon** | Symmetric divergence, probability comparison | Bounded [0,1], symmetric, smooth | Histogram-based, bin size matters |
    | **Euclidean (means)** | Quick centroid comparison | Very fast, simple | Ignores distribution shape/variance |
    | **Cosine (means)** | Directional similarity of distributions | Scale-invariant | Ignores magnitude and variance |

    **Decision Tree:**
    - **Need geometric interpretation?** → Use Wasserstein
    - **One-dimensional data with hypothesis testing?** → Use K-S
    - **Comparing probability distributions symmetrically?** → Use Jensen-Shannon
    - **Quick centroid check for correlated data?** → Use Mahalanobis (means)
    - **Just checking if distributions differ?** → Use K-S or Wasserstein

    ### Outlier Detection Methods

    | Method | Best For | Pros | Cons |
    |--------|----------|------|------|
    | **IQR** | Univariate, Gaussian-like data | Fast, simple, interpretable | Feature-by-feature only, assumes normality |
    | **Z-Score (MAD)** | Univariate, robust to some outliers | Robust to outliers in computation | Still feature-by-feature, needs sufficient data |
    | **Isolation Forest** | High-dimensional, complex structures | Fast, works in high-dim, no assumptions | Black-box, hyperparameter tuning needed |
    | **Local Outlier Factor** | Varying density clusters | Detects local density deviations | Slow on large data, sensitive to k parameter |
    | **Elliptic Envelope** | Gaussian distributions, correlated features | Accounts for covariance | Assumes Gaussian, needs sufficient samples |

    **Decision Tree:**
    - **Univariate data?** → Use IQR or Z-Score (MAD)
    - **High-dimensional (>10D)?** → Use Isolation Forest
    - **Varying cluster densities?** → Use LOF
    - **Correlated Gaussian data?** → Use Elliptic Envelope
    - **Need interpretable threshold?** → Use IQR or Z-Score

    ### Practical Tips

    1. **Start Simple**: Try Euclidean distance, K-S test, or IQR first for baseline
    2. **Visualize**: Plot your data to understand structure before choosing metrics
    3. **Validate**: Use ground truth scenarios (like in this notebook) to test metric behavior
    4. **Computational Cost**: Consider sample size - Wasserstein/Mahalanobis are expensive
    5. **Combine Methods**: Use multiple metrics to get complementary views
    6. **Scale Your Data**: Standardize features unless using scale-invariant metrics (cosine)

    ### Neural-Specific Considerations

    - **Population activity**: Use Euclidean or Mahalanobis for state space distances
    - **Tuning curves**: Cosine similarity works well for directional selectivity
    - **Trial-to-trial variability**: Wasserstein captures geometric displacement
    - **Multi-neuron correlations**: Mahalanobis accounts for covariance structure
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 7. PlotGrid System - Complete Usage Guide

    This notebook demonstrates the new flexible plotting system. Here's a comprehensive guide:
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
    pd,
    plot_comparison_grid,
):
    # Complete PlotGrid Usage Examples

    # Example 1: Simple comparison grid from dictionary
    print("=== Example 1: Comparison Grid ===")
    test_data = {
        "Dataset A": np.random.randn(100, 2),
        "Dataset B": np.random.randn(100, 2) + 1,
        "Dataset C": np.random.randn(100, 2) - 1,
    }

    fig1 = plot_comparison_grid(
        test_data,
        plot_type="scatter",
        cols=3,
        config=PlotConfig(title="Comparison of Three Datasets", figsize=(12, 4)),
        backend="plotly",
    )
    fig1.show()

    # Example 2: Multi-trace subplot (multiple datasets in same subplot)
    print("\n=== Example 2: Multi-Trace Subplot ===")
    specs_multi = [
        PlotSpec(
            data=test_data["Dataset A"],
            plot_type="scatter",
            subplot_position=0,
            label="A",
            color="blue",
            alpha=0.5,
            marker_size=5,
        ),
        PlotSpec(
            data=test_data["Dataset B"],
            plot_type="scatter",
            subplot_position=0,
            label="B",
            color="red",
            alpha=0.5,
            marker_size=5,
        ),
        PlotSpec(
            data=test_data["Dataset C"],
            plot_type="scatter",
            subplot_position=0,
            label="C",
            color="green",
            alpha=0.5,
            marker_size=5,
            title="All Datasets Overlaid",
        ),
    ]

    grid_multi = PlotGrid(
        plot_specs=specs_multi,
        config=PlotConfig(figsize=(7, 5)),
        layout=GridLayoutConfig(rows=1, cols=1),
        backend="plotly",
    )
    fig2 = grid_multi.plot()
    fig2.show()

    # Example 3: DataFrame-based configuration with grouping
    print("\n=== Example 3: DataFrame Configuration ===")
    plot_df = pd.DataFrame(
        {
            "data": [
                np.random.randn(100, 2),
                np.random.randn(100, 2) + 1,
                np.random.randn(100, 2),
                np.random.randn(100, 2) + 1,
            ],
            "plot_type": ["scatter", "scatter", "scatter", "scatter"],
            "title": ["Control 1", "Treatment 1", "Control 2", "Treatment 2"],
            "group": ["control", "treatment", "control", "treatment"],
        }
    )

    grid_df = PlotGrid.from_dataframe(
        plot_df,
        group_by="group",  # Auto-assigns colors by group
        config=PlotConfig(title="Grouped by Condition", figsize=(10, 8)),
        layout=GridLayoutConfig(rows=2, cols=2),
        backend="plotly",
    )
    fig3 = grid_df.plot()
    fig3.show()

    # Example 4: Mixed plot types in one grid
    print("\n=== Example 4: Mixed Plot Types ===")
    time_series = np.random.randn(100)
    histogram_data = np.random.randn(500)
    scatter_data = np.random.randn(50, 2)

    specs_mixed = [
        PlotSpec(
            data=np.column_stack([np.arange(100), time_series]),
            plot_type="line",
            title="Time Series",
            color="blue",
            line_width=2,
        ),
        PlotSpec(
            data=histogram_data,
            plot_type="histogram",
            title="Distribution",
            color="green",
            kwargs={"bins": 30},
        ),
        PlotSpec(
            data=scatter_data,
            plot_type="scatter",
            title="Scatter Plot",
            color="red",
            marker_size=8,
        ),
    ]

    grid_mixed = PlotGrid(
        plot_specs=specs_mixed,
        config=PlotConfig(title="Mixed Plot Types", figsize=(12, 4)),
        layout=GridLayoutConfig(rows=1, cols=3),
        backend="plotly",
    )
    fig4 = grid_mixed.plot()
    fig4.show()

    print("\n✓ All PlotGrid examples completed!")
    print("\nKey Takeaways:")
    print("1. Use plot_comparison_grid() for quick multi-panel comparisons")
    print("2. Use subplot_position to overlay multiple traces in same subplot")
    print("3. Use DataFrame.from_dataframe() for structured, metadata-driven plots")
    print("4. Mix different plot types (scatter, line, histogram, heatmap) in one grid")
    print("5. Automatic color schemes and grid sizing")
    print("6. Works with both matplotlib and plotly backends")
    return
