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
    # Shape Similarity Analysis - Complete Examples

    This notebook demonstrates shape distance metrics for comparing neural population activity matrices.
    Shape similarity measures how similar the "code" or representation is between two populations.

    ## Methods Demonstrated

    1. **Procrustes**: Optimal orthogonal alignment (rotation/reflection). Best for aligned data with fixed neuron identity.
    2. **One-to-one**: Optimal hard assignment (Hungarian algorithm). Best for shuffled or unknown neuron identities.
    3. **Soft-matching**: Optimal transport with soft assignment. Best for different-sized populations.

    ## Theoretical Ordering

    The methods follow a theoretical ordering: **soft-matching ≤ one-to-one ≤ procrustes**
    (more flexibility → lower distance → better matching)
    """
    )
    return


@app.cell
def _():
    # Import required libraries
    from typing import Any

    import numpy as np
    import numpy.typing as npt

    from neural_analysis.metrics.distributions import shape_distance
    from neural_analysis.metrics.pairwise_metrics import compare_datasets
    from neural_analysis.plotting import (
        GridLayoutConfig,
        PlotConfig,
        PlotGrid,
        PlotSpec,
    )
    from neural_analysis.plotting.shape_distance import (
        embed_mds,
        embed_mds_pca,
        plot_shape_distance_mds,
    )

    # Set random seed for reproducibility
    np.random.seed(42)

    print("✓ Imports successful")
    return (
        Any,
        GridLayoutConfig,
        PlotConfig,
        PlotGrid,
        PlotSpec,
        compare_datasets,
        embed_mds,
        embed_mds_pca,
        np,
        npt,
        plot_shape_distance_mds,
        shape_distance,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 1. Generate Diverse Test Datasets

    We'll create datasets with varying neuron counts and features to demonstrate
    how different methods handle different scenarios.
    """
    )
    return


@app.cell
def _(np):
    # Generate diverse test datasets
    # For all-pairs comparison, we need equal-sized datasets for procrustes
    # But we'll show different-sized examples in later sections

    n_datasets = 15  # Reduced for faster computation
    n_features = 50  # Number of features/conditions
    n_neurons = 40  # Fixed neuron count for all-pairs comparison

    datasets = []
    dataset_names = []

    for i in range(n_datasets):
        # Create datasets with some structure (not completely random)
        base_pattern = np.random.randn(n_neurons, n_features)
        # Add some cluster structure
        if i < 5:
            # Cluster 0: similar patterns
            base_pattern += np.random.randn(1, n_features) * 0.3
        elif i < 10:
            # Cluster 1: different patterns
            base_pattern += np.random.randn(1, n_features) * 0.3 + 1.0
        else:
            # Cluster 2: intermediate patterns
            base_pattern += np.random.randn(1, n_features) * 0.3 + 0.5

        datasets.append(base_pattern.astype(np.float64))
        dataset_names.append(f"Dataset_{i}")

    # Create cluster labels for visualization
    labels = np.array([0] * 5 + [1] * 5 + [2] * 5)

    print(f"✓ Generated {len(datasets)} datasets")
    print(f"  Neuron count: {n_neurons} (same for all, required for procrustes)")
    print(f"  Features: {datasets[0].shape[1]}")
    print(
        f"  Clusters: {len(np.unique(labels))} (sizes: {[np.sum(labels == _i) for _i in np.unique(labels)]})"
    )

    return dataset_names, datasets, labels, n_datasets, n_features, n_neurons


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 2. Compute Shape Distance Matrices

    Compute pairwise distance matrices for all three methods using `compare_datasets`.
    """
    )
    return


@app.cell
def _(compare_datasets, datasets, dataset_names, np):
    # Convert datasets to dict format for compare_datasets
    datasets_dict = {name: data for name, data in zip(dataset_names, datasets)}

    # Methods to compute
    methods = ["procrustes", "one-to-one", "soft-matching"]
    distance_matrices = {}

    print("Computing distance matrices...")
    for method in methods:
        print(f"  Computing {method} distances...")
        result = compare_datasets(
            datasets_dict,
            mode="all-pairs",
            metric=method,
            show_progress=False,
        )

        # Convert dict result to symmetric matrix
        n = len(datasets)
        _D = np.zeros((n, n), dtype=np.float64)
        for _i, name_i in enumerate(dataset_names):
            for _j, name_j in enumerate(dataset_names):
                _D[_i, _j] = result[name_i][name_j]

        distance_matrices[method] = _D
        print(
            f"    ✓ {method}: min={_D.min():.4f}, max={_D.max():.4f}, mean={_D.mean():.4f}"
        )

    print("\n✓ All distance matrices computed")
    return distance_matrices, methods


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 3. MDS Embeddings Visualization

    Visualize the distance matrices using Multidimensional Scaling (MDS) embeddings.
    This shows how datasets cluster in the shape similarity space.
    """
    )
    return


@app.cell
def _(labels, plot_shape_distance_mds, distance_matrices):
    # Plot MDS embeddings for all methods
    fig = plot_shape_distance_mds(
        distance_matrices=distance_matrices,
        labels=labels,
        backend="matplotlib",
        figsize=(14, 10),
    )
    return fig


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 4. Distance Matrix Heatmaps

    Visualize the raw distance matrices as heatmaps to see pairwise similarities.
    """
    )
    return


@app.cell
def _(GridLayoutConfig, PlotConfig, PlotGrid, PlotSpec, distance_matrices, methods):
    # Create heatmap visualizations for each distance matrix
    _plot_specs_heatmap = []

    for _idx_heatmap, _method_heatmap in enumerate(methods):
        _D_heatmap = distance_matrices[_method_heatmap]
        _spec_heatmap = PlotSpec(
            data=_D_heatmap,
            plot_type="heatmap",
            subplot_position=_idx_heatmap,
            title=f"{_method_heatmap.capitalize()} Distance Matrix",
            colorbar=True,
            colorbar_label="Distance",
            kwargs={
                "cmap": "viridis_r",  # Reversed: lower distances = brighter
            },
        )
        _plot_specs_heatmap.append(_spec_heatmap)

    _grid_heatmap = PlotGrid(
        plot_specs=_plot_specs_heatmap,
        config=PlotConfig(figsize=(14, 4)),
        layout=GridLayoutConfig(rows=1, cols=3),
        backend="matplotlib",
    )

    _fig_heatmap = _grid_heatmap.plot()
    return _fig_heatmap, _grid_heatmap, _plot_specs_heatmap


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 5. Method Comparison: Theoretical Ordering

    Compare distances from different methods for the same pairs to verify
    the theoretical ordering: soft-matching ≤ one-to-one ≤ procrustes
    """
    )
    return


@app.cell
def _(distance_matrices, methods, np):
    # Extract distances for a few example pairs
    n_pairs_to_show = 10
    pair_indices = [(i, j) for i in range(5) for j in range(i + 1, 5)][:n_pairs_to_show]

    # Collect distances for each pair
    comparison_data = {
        "pair": [],
        "procrustes": [],
        "one-to-one": [],
        "soft-matching": [],
    }

    for _i_pair, _j_pair in pair_indices:
        pair_name = f"{_i_pair}-{_j_pair}"
        comparison_data["pair"].append(pair_name)
        comparison_data["procrustes"].append(
            distance_matrices["procrustes"][_i_pair, _j_pair]
        )
        comparison_data["one-to-one"].append(
            distance_matrices["one-to-one"][_i_pair, _j_pair]
        )
        comparison_data["soft-matching"].append(
            distance_matrices["soft-matching"][_i_pair, _j_pair]
        )

    print("Distance comparison for sample pairs:")
    print(
        f"{'Pair':<10} {'Procrustes':<12} {'One-to-one':<12} {'Soft-matching':<12} {'Ordering':<15}"
    )
    print("-" * 70)

    ordering_violations = 0
    for _i_comp in range(len(comparison_data["pair"])):
        proc = comparison_data["procrustes"][_i_comp]
        oto = comparison_data["one-to-one"][_i_comp]
        soft = comparison_data["soft-matching"][_i_comp]

        # Check ordering
        if proc > oto:
            ordering = "⚠️ VIOLATION"
            ordering_violations += 1
        elif oto > soft:
            ordering = "ℹ️ Expected"
        else:
            ordering = "✅ Correct"

        print(
            f"{comparison_data['pair'][_i_comp]:<10} "
            f"{proc:<12.6f} {oto:<12.6f} {soft:<12.6f} {ordering:<15}"
        )

    print(
        f"\nOrdering violations: {ordering_violations}/{len(comparison_data['pair'])}"
    )

    return comparison_data, n_pairs_to_show, ordering_violations, pair_indices


@app.cell
def _(GridLayoutConfig, PlotConfig, PlotGrid, PlotSpec, comparison_data, methods, np):
    # Visualize the comparison as bar plots

    # Create grouped bar plot
    _x_pos_bar = np.arange(len(comparison_data["pair"]))
    _width_bar = 0.25

    _plot_specs_bar = []
    _colors_bar = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # Blue, orange, green

    for _idx_bar, _method_bar in enumerate(methods):
        _values_bar = comparison_data[_method_bar]
        _spec_bar = PlotSpec(
            data=_values_bar,
            plot_type="bar",
            subplot_position=0,
            label=_method_bar.capitalize(),
            color=_colors_bar[_idx_bar],
            alpha=0.7,
            kwargs={
                "x": _x_pos_bar + _idx_bar * _width_bar,
                "width": _width_bar,
            },
        )
        _plot_specs_bar.append(_spec_bar)

    _grid_bar = PlotGrid(
        plot_specs=_plot_specs_bar,
        config=PlotConfig(
            figsize=(14, 6),
            xlabel="Pair",
            ylabel="Distance",
        ),
        layout=GridLayoutConfig(rows=1, cols=1),
        backend="matplotlib",
    )

    _result_bar = _grid_bar.plot()
    if isinstance(_result_bar, tuple):
        _fig_bar, _axes_bar = _result_bar
        _ax_bar = _axes_bar[0] if isinstance(_axes_bar, list) else _axes_bar
    else:
        _ax_bar = _result_bar
        _fig_bar = _ax_bar.get_figure() if hasattr(_ax_bar, "get_figure") else None

    # Set x-axis labels
    _ax_bar.set_xticks(_x_pos_bar + _width_bar)
    _ax_bar.set_xticklabels(comparison_data["pair"], rotation=45, ha="right")
    _ax_bar.set_title("Distance Comparison Across Methods")
    _ax_bar.legend()
    _ax_bar.grid(True, alpha=0.3)

    return (
        _ax_bar,
        _colors_bar,
        _fig_bar,
        _grid_bar,
        _plot_specs_bar,
        _width_bar,
        _x_pos_bar,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 6. Individual Pair Analysis

    Detailed analysis of a specific pair showing how each method computes the distance.
    """
    )
    return


@app.cell
def _(datasets, np, shape_distance):
    # Select a specific pair for detailed analysis
    pair_idx = (0, 1)
    mtx1 = datasets[pair_idx[0]]
    mtx2 = datasets[pair_idx[1]]

    print(f"Analyzing pair: Dataset {pair_idx[0]} vs Dataset {pair_idx[1]}")
    print(f"  mtx1.shape: {mtx1.shape}")
    print(f"  mtx2.shape: {mtx2.shape}")
    print()

    # Compute distances with all methods
    pair_results = {}
    for _method_pair in ["procrustes", "one-to-one", "soft-matching"]:
        if _method_pair == "soft-matching":
            dist, pairs, meta = shape_distance(
                mtx1, mtx2, method=_method_pair, metric="sqeuclidean", approx=False
            )
        else:
            dist, pairs, meta = shape_distance(
                mtx1, mtx2, method=_method_pair, metric="sqeuclidean"
            )

        pair_results[_method_pair] = {
            "distance": float(dist),
            "pairs": pairs,
            "meta": meta,
        }

        print(f"{_method_pair.capitalize()}:")
        if isinstance(dist, np.ndarray):
            dist_val = float(np.mean(dist))
            print(f"  Distance: {dist_val:.6f} (mean over {len(dist)} runs)")
        else:
            dist_val = float(dist)
            print(f"  Distance: {dist_val:.6f}")
        print(f"  Pairs: {len(pairs) if pairs else 0} correspondences")
        print()

    # Check ordering
    proc_dist = pair_results["procrustes"]["distance"]
    oto_dist = pair_results["one-to-one"]["distance"]
    soft_dist = pair_results["soft-matching"]["distance"]

    print("Ordering check:")
    print(f"  Procrustes: {proc_dist:.6f}")
    print(f"  One-to-one: {oto_dist:.6f}")
    print(f"  Soft-matching: {soft_dist:.6f}")

    if proc_dist > oto_dist:
        print("  ⚠️  VIOLATION: procrustes > one-to-one")
    elif oto_dist > soft_dist:
        print(
            "  ℹ️  Expected: one-to-one > soft-matching (soft handles different sizes)"
        )
    else:
        print("  ✅ Correct ordering: procrustes ≤ one-to-one ≤ soft-matching")

    return mtx1, mtx2, pair_idx, pair_results


@app.cell
def _(GridLayoutConfig, PlotConfig, PlotGrid, PlotSpec, mtx1, mtx2, np):
    # Visualize the two matrices as heatmaps
    _plot_specs_pair = [
        PlotSpec(
            data=mtx1,
            plot_type="heatmap",
            subplot_position=0,
            title=f"Matrix 1 (shape: {mtx1.shape})",
            colorbar=True,
            colorbar_label="Activity",
            kwargs={"cmap": "RdBu_r"},
        ),
        PlotSpec(
            data=mtx2,
            plot_type="heatmap",
            subplot_position=1,
            title=f"Matrix 2 (shape: {mtx2.shape})",
            colorbar=True,
            colorbar_label="Activity",
            kwargs={"cmap": "RdBu_r"},
        ),
    ]

    _grid_pair = PlotGrid(
        plot_specs=_plot_specs_pair,
        config=PlotConfig(figsize=(12, 5)),
        layout=GridLayoutConfig(rows=1, cols=2),
        backend="matplotlib",
    )

    _fig_pair = _grid_pair.plot()
    return _fig_pair, _grid_pair, _plot_specs_pair


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 7. Different-Sized Populations

    Demonstrate how soft-matching handles different-sized populations natively,
    while procrustes and one-to-one require subsampling.
    """
    )
    return


@app.cell
def _(np, shape_distance):
    # Create matrices with very different sizes
    mtx_small = np.random.randn(30, 50).astype(np.float64)
    mtx_large = np.random.randn(80, 50).astype(np.float64)

    print(f"Small matrix: {mtx_small.shape}")
    print(f"Large matrix: {mtx_large.shape}")
    print()

    # Test each method
    results = {}

    # Procrustes - requires subsampling
    print("Procrustes (requires equal sizes, will auto-subsample):")
    dist_proc, pairs_proc, meta_proc = shape_distance(
        mtx_small, mtx_large, method="procrustes", metric="sqeuclidean"
    )
    results["procrustes"] = {"dist": dist_proc, "meta": meta_proc}
    dist_proc_val = (
        float(np.mean(dist_proc))
        if isinstance(dist_proc, np.ndarray)
        else float(dist_proc)
    )
    print(f"  Distance: {dist_proc_val:.6f}")
    print(f"  Auto-subsampling: {meta_proc.get('auto_subsampling', False)}")
    print()

    # One-to-one - requires subsampling
    print("One-to-one (requires equal sizes, will auto-subsample):")
    dist_oto, pairs_oto, meta_oto = shape_distance(
        mtx_small, mtx_large, method="one-to-one", metric="sqeuclidean"
    )
    results["one-to-one"] = {"dist": dist_oto, "meta": meta_oto}
    dist_oto_val = (
        float(np.mean(dist_oto))
        if isinstance(dist_oto, np.ndarray)
        else float(dist_oto)
    )
    print(f"  Distance: {dist_oto_val:.6f}")
    print(f"  Auto-subsampling: {meta_oto.get('auto_subsampling', False)}")
    print()

    # Soft-matching - handles different sizes natively
    print("Soft-matching (handles different sizes natively, no subsampling):")
    dist_soft, pairs_soft, meta_soft = shape_distance(
        mtx_small, mtx_large, method="soft-matching", metric="sqeuclidean", approx=False
    )
    results["soft-matching"] = {"dist": dist_soft, "meta": meta_soft}
    dist_soft_val = (
        float(np.mean(dist_soft))
        if isinstance(dist_soft, np.ndarray)
        else float(dist_soft)
    )
    print(f"  Distance: {dist_soft_val:.6f}")
    print(f"  Auto-subsampling: {meta_soft.get('auto_subsampling', False)}")
    print()

    print("Summary:")
    print(f"  Procrustes: {dist_proc_val:.6f} (subsampled)")
    print(f"  One-to-one: {dist_oto_val:.6f} (subsampled)")
    print(f"  Soft-matching: {dist_soft_val:.6f} (full matrices)")

    return mtx_large, mtx_small, results


@app.cell
def _(GridLayoutConfig, PlotConfig, PlotGrid, PlotSpec, mtx_large, mtx_small):
    # Visualize the different-sized matrices
    _plot_specs_size = [
        PlotSpec(
            data=mtx_small,
            plot_type="heatmap",
            subplot_position=0,
            title=f"Small Matrix ({mtx_small.shape[0]} neurons)",
            colorbar=True,
            colorbar_label="Activity",
            kwargs={"cmap": "viridis"},
        ),
        PlotSpec(
            data=mtx_large,
            plot_type="heatmap",
            subplot_position=1,
            title=f"Large Matrix ({mtx_large.shape[0]} neurons)",
            colorbar=True,
            colorbar_label="Activity",
            kwargs={"cmap": "viridis"},
        ),
    ]

    _grid_size = PlotGrid(
        plot_specs=_plot_specs_size,
        config=PlotConfig(figsize=(12, 5)),
        layout=GridLayoutConfig(rows=1, cols=2),
        backend="matplotlib",
    )

    _fig_size = _grid_size.plot()
    return _fig_size, _grid_size, _plot_specs_size


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 8. Summary Statistics

    Compute summary statistics across all pairs to understand the distribution
    of distances for each method.
    """
    )
    return


@app.cell
def _(distance_matrices, methods, np):
    # Compute summary statistics for each method
    print("Summary Statistics Across All Pairs:")
    print("=" * 60)

    for _method_stats in methods:
        _D_stats = distance_matrices[_method_stats]
        # Extract upper triangle (excluding diagonal)
        _mask_stats = np.triu(np.ones_like(_D_stats, dtype=bool), k=1)
        _distances_stats = _D_stats[_mask_stats]

        print(f"\n{_method_stats.capitalize()}:")
        print(f"  Mean: {np.mean(_distances_stats):.6f}")
        print(f"  Std:  {np.std(_distances_stats):.6f}")
        print(f"  Min:  {np.min(_distances_stats):.6f}")
        print(f"  Max:  {np.max(_distances_stats):.6f}")
        print(f"  Median: {np.median(_distances_stats):.6f}")

    return


@app.cell
def _(GridLayoutConfig, PlotConfig, PlotGrid, PlotSpec, distance_matrices, methods, np):
    # Create histogram comparison
    _plot_specs_hist = []

    for _idx_hist, _method_hist in enumerate(methods):
        _D_hist = distance_matrices[_method_hist]
        # Extract upper triangle (excluding diagonal)
        _mask_hist = np.triu(np.ones_like(_D_hist, dtype=bool), k=1)
        _distances_hist = _D_hist[_mask_hist]

        _spec_hist = PlotSpec(
            data=_distances_hist,
            plot_type="histogram",
            subplot_position=_idx_hist,
            title=f"{_method_hist.capitalize()} Distance Distribution",
            label=_method_hist,
            alpha=0.7,
            kwargs={"bins": 30},
        )
        _plot_specs_hist.append(_spec_hist)

    _grid_hist = PlotGrid(
        plot_specs=_plot_specs_hist,
        config=PlotConfig(figsize=(14, 4), xlabel="Distance", ylabel="Frequency"),
        layout=GridLayoutConfig(rows=1, cols=3),
        backend="matplotlib",
    )

    _fig_hist = _grid_hist.plot()
    return _fig_hist, _grid_hist, _plot_specs_hist


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Conclusion

    This notebook demonstrated:

    1. **Shape distance computation** using three methods: Procrustes, one-to-one, and soft-matching
    2. **MDS visualization** showing how datasets cluster in shape similarity space
    3. **Theoretical ordering** verification: soft-matching ≤ one-to-one ≤ procrustes
    4. **Different-sized populations** handling: soft-matching works natively, others require subsampling
    5. **Distance matrix analysis** through heatmaps and histograms

    ### When to Use Each Method:

    - **Procrustes**: Best when neuron identities are known and aligned (same recording session, same order)
    - **One-to-one**: Best when neurons are shuffled or identities unknown (cross-subject, shuffled data)
    - **Soft-matching**: Best for different-sized populations or when you need a smooth, differentiable metric

    All methods are normalized to be on comparable scales, making them suitable for direct comparison.
    """
    )
    return
