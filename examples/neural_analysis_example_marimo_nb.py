import marimo

__generated_with = "0.18.3"
app = marimo.App(width="full", auto_download=["html"])


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Comprehensive Neural Analysis Example

    This notebook demonstrates comprehensive neural analysis methods using synthetic datasets. It covers:

    1. **Test Neural Analysis Methods** - Apply structure_index and shape_distance to various datasets
       - Shape distance validation (soft-matching, one-to-one, procrustes ordering)
    2. **Benchmark Dimensionality Reduction** - Compare PCA, UMAP, t-SNE, MDS, Isomap, LLE, Spectral
    3. **Validate Decoding Approaches** - Test population vector and k-NN decoders with known ground truth
    4. **Test Cell Type Classification** - Apply supervised and unsupervised classifiers to mixed populations
    5. **Study Noise Effects** - Analyze how noise affects embedding quality and analysis methods

    All visualizations use the **PlotGrid system** for consistency.
    """)
    return


@app.cell
def _():
    # Imports
    import time
    import warnings
    from pathlib import Path

    import numpy as np
    import pandas as pd

    warnings.filterwarnings("ignore")

    # Neural analysis imports
    from neural_analysis import (
        compare_classifiers,
        compare_clusterers,
        compute_embedding,
        compute_structure_index,
        evaluate_decoder,
        extract_cell_features,
        generate_grid_cells,
        generate_head_direction_cells,
        generate_mixed_population_flexible,
        generate_place_cells,
        knn_decoder,
        population_vector_decoder,
        shape_distance,
    )
    from neural_analysis.plotting import (
        GridLayoutConfig,
        PlotConfig,
        PlotGrid,
        PlotSpec,
    )

    # Set random seed
    print("✓ Imports successful")
    return (
        GridLayoutConfig,
        Path,
        PlotConfig,
        PlotGrid,
        PlotSpec,
        compare_classifiers,
        compare_clusterers,
        compute_embedding,
        compute_structure_index,
        evaluate_decoder,
        extract_cell_features,
        generate_grid_cells,
        generate_head_direction_cells,
        generate_mixed_population_flexible,
        generate_place_cells,
        knn_decoder,
        np,
        pd,
        population_vector_decoder,
        shape_distance,
        time,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## Section 1: Test Neural Analysis Methods with Datasets

    Generate multiple synthetic datasets and apply structure_index and shape_distance to quantify manifold organization.
    """)
    return


@app.cell
def _(
    generate_grid_cells,
    generate_head_direction_cells,
    generate_mixed_population_flexible,
    generate_place_cells,
):
    # Generate different cell types
    print("Generating synthetic datasets...")

    # Place cells
    place_activity, place_meta = generate_place_cells(
        n_cells=50, n_samples=1000, arena_size=(2.0, 2.0), seed=42, plot=False
    )

    # Grid cells
    grid_activity, grid_meta = generate_grid_cells(
        n_cells=50, n_samples=1000, arena_size=(2.0, 2.0), seed=43, plot=False
    )

    # Head direction cells
    hd_activity, hd_meta = generate_head_direction_cells(
        n_cells=50, n_samples=1000, seed=44, plot=False
    )

    # Mixed population
    mixed_activity, mixed_meta = generate_mixed_population_flexible(
        n_samples=1000, seed=45, plot=False
    )

    print(f"Place cells: {place_activity.shape}")
    print(f"Grid cells: {grid_activity.shape}")
    print(f"Head direction cells: {hd_activity.shape}")
    print(f"Mixed population: {mixed_activity.shape}")
    return (
        grid_activity,
        grid_meta,
        hd_activity,
        hd_meta,
        mixed_activity,
        mixed_meta,
        place_activity,
        place_meta,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ### 1.1 Shape Distance Validation

    Validate that shape distance methods maintain theoretical ordering: soft-matching ≤ one-to-one ≤ procrustes.
    """)
    return


@app.cell
def _(Path):
    # Run shape distance validation
    import sys

    examples_dir = Path.cwd() / "examples"
    if str(examples_dir) not in sys.path:
        # Add examples directory to path for imports
        sys.path.insert(0, str(examples_dir))
    from test_shape_distance_validation import (
        print_validation_summary,
        validate_shape_distance_ordering,
    )

    print("Running shape distance validation...")
    print("=" * 70)
    validation_results = validate_shape_distance_ordering(
        n_pairs=5, n_samples=100, neuron_range=(20, 40), repeats=3, seed=42
    )
    # Run validation with smaller parameters for notebook demo
    print_validation_summary(validation_results)  # Small number for quick demo
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 1.2 Structure Index Analysis

    Compute structure index for each dataset to quantify manifold organization.
    """)
    return


@app.cell
def _(
    compute_structure_index,
    grid_activity,
    grid_meta,
    hd_activity,
    hd_meta,
    mixed_activity,
    mixed_meta,
    np,
    place_activity,
    place_meta,
):
    # Compute structure index for each dataset
    print("\nComputing Structure Index...")
    datasets = {
        "place": (place_activity, place_meta["positions"]),
        "grid": (grid_activity, grid_meta["positions"]),
        "head_direction": (hd_activity, hd_meta["head_directions"].reshape(-1, 1)),
        "mixed": (mixed_activity, mixed_meta["positions"]),
    }
    si_results = {}
    for name, (_activity, labels) in datasets.items():
        try:
            _si, _, _, _ = compute_structure_index(
                _activity,
                labels,
                n_bins=10,
                n_neighbors=15,
                num_shuffles=3,
            )
            si_results[name] = _si
            print(f"  {name}: SI = {_si:.3f}")
        except Exception as e:
            print(f"  {name}: Error - {e}")
            si_results[name] = np.nan
    return


@app.cell
def _(grid_activity, mixed_activity, place_activity, shape_distance):
    # Compute shape similarity between datasets
    print("\nComputing shape similarity...")
    place_sample = place_activity[:500]
    # Compare place vs grid
    grid_sample = grid_activity[:500]
    _shape_dist, _, _ = shape_distance(
        place_sample.T, place_sample.T, method="procrustes"
    )
    print(f"Place vs Place (Procrustes): {_shape_dist:.3f}")
    _shape_dist2, _, _ = shape_distance(
        place_sample.T, grid_sample.T, method="procrustes"
    )
    print(f"Place vs Grid (Procrustes): {_shape_dist2:.3f}")
    mixed_sample = mixed_activity[:500]
    shape_dist3, _, _ = shape_distance(
        place_sample.T, mixed_sample.T, method="procrustes"
    )
    # Compare place vs mixed
    print(f"Place vs Mixed (Procrustes): {shape_dist3.mean():.3f}")
    mixed_sample = mixed_activity[:500]
    shape_dist3, _, _ = shape_distance(
        grid_sample.T, mixed_sample.T, method="procrustes"
    )
    # Compare place vs mixed
    print(f"Grid vs Mixed (Procrustes): {shape_dist3.mean():.3f}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Section 2: Benchmark Dimensionality Reduction Algorithms

    Compare multiple DR methods on synthetic datasets with known structure.
    """)
    return


@app.cell
def _(compute_embedding, place_activity, place_meta, time):
    data = place_activity
    labels_1 = place_meta["positions"]
    print("Computing embeddings...")
    methods = ["pca", "umap", "tsne", "mds", "isomap", "lle", "spectral"]
    embeddings = {}
    times = {}
    for _method in methods:
        try:
            start = time.time()
            _emb = compute_embedding(
                data, method=_method, n_components=2, random_state=42
            )
            elapsed = time.time() - start
            embeddings[_method] = _emb
            times[_method] = elapsed
            print(f"  {_method}: {elapsed:.3f}s")
        except Exception as e:
            print(f"  {_method}: Error - {e}")
            embeddings[_method] = None
    return embeddings, labels_1


@app.cell
def _(compute_structure_index, embeddings, labels_1, np):
    print("\nEvaluating embeddings with structure index...")
    embedding_si = {}
    for _method, _emb in embeddings.items():
        if _emb is not None:
            try:
                _si, _, _, _ = compute_structure_index(
                    _emb, labels_1, n_bins=10, n_neighbors=15, num_shuffles=1
                )
                embedding_si[_method] = _si
                print(f"  {_method}: SI = {_si:.3f}")
            except Exception as e:
                print(f"  {_method}: SI computation failed - {e}")
                embedding_si[_method] = np.nan
    return


@app.cell
def _(GridLayoutConfig, PlotConfig, PlotGrid, PlotSpec, embeddings, labels_1):
    import matplotlib.pyplot as _plt

    _plot_specs = []
    methods_to_plot = ["pca", "umap", "tsne", "isomap"]
    for idx, _method in enumerate(methods_to_plot):
        if embeddings.get(_method) is not None:
            _emb = embeddings[_method]
            spec = PlotSpec(
                data={"x": _emb[:, 0], "y": _emb[:, 1]},
                plot_type="scatter",
                subplot_position=idx,
                title=f"{_method.upper()} Embedding",
                color_by=labels_1[:, 0],
                cmap="viridis",
                marker_size=10,
                alpha=0.6,
                kwargs={"x_label": "Dim 1", "y_label": "Dim 2"},
            )
            _plot_specs.append(spec)
    if _plot_specs:
        _grid = PlotGrid(
            plot_specs=_plot_specs,
            config=PlotConfig(figsize=(16, 4)),
            layout=GridLayoutConfig(rows=1, cols=len(_plot_specs)),
            backend="matplotlib",
        )
        _fig = _grid.plot()
        # Display the plot
        # matplotlib backend: returns (fig, axes) tuple for multiple subplots, or just axes for single subplot
        # plotly backend: returns figure object
        if isinstance(_fig, tuple):
            # Multiple subplots (matplotlib)
            _fig_obj, _axes = _fig
            _plt.tight_layout()
            _plt.show()
        else:
            # Single subplot (matplotlib) or plotly backend
            _plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Section 3: Validate Decoding Approaches with Known Ground Truth

    Test decoding methods on place cells, grid cells, and head direction cells with known positions/angles.
    """)
    return


@app.cell
def _(knn_decoder, np, place_activity, place_meta, population_vector_decoder):
    # Test population vector decoder on place cells
    print("Testing Population Vector Decoder...")
    decoded_pos = population_vector_decoder(
        place_activity, place_meta["field_centers"], method="weighted_average"
    )

    # Compute decoding error
    errors = np.linalg.norm(decoded_pos - place_meta["positions"], axis=1)
    print(f"  Mean error: {errors.mean():.3f} m")
    print(f"  Median error: {np.median(errors):.3f} m")

    # Test k-NN decoder
    print("\nTesting k-NN Decoder...")
    n_train = 700
    train_act = place_activity[:n_train]
    train_pos = place_meta["positions"][:n_train]
    test_act = place_activity[n_train:]
    test_pos = place_meta["positions"][n_train:]

    decoded_knn = knn_decoder(train_act, train_pos, test_act, k=5)
    errors_knn = np.linalg.norm(decoded_knn - test_pos, axis=1)
    print(f"  Mean error: {errors_knn.mean():.3f} m")
    print(f"  Median error: {np.median(errors_knn):.3f} m")
    return n_train, test_act, test_pos, train_act, train_pos


@app.cell
def _(
    compute_embedding,
    evaluate_decoder,
    n_train,
    place_activity,
    test_act,
    test_pos,
    train_act,
    train_pos,
):
    # Compare high-D vs low-D decoding
    print("\nComparing High-D vs Low-D Decoding...")

    # High-D (raw activity)
    metrics_highd = evaluate_decoder(
        train_act, train_pos, test_act, test_pos, decoder="knn", k=5
    )

    # Low-D (PCA embedding)
    pca_emb = compute_embedding(
        place_activity, method="pca", n_components=10, random_state=42
    )
    train_emb = pca_emb[:n_train]
    test_emb = pca_emb[n_train:]

    metrics_lowd = evaluate_decoder(
        train_emb, train_pos, test_emb, test_pos, decoder="knn", k=5
    )

    print(
        f"High-D R²: {metrics_highd['r2_score']:.3f}, Error: {metrics_highd['mean_error']:.3f}"
    )
    print(
        f"Low-D R²: {metrics_lowd['r2_score']:.3f}, Error: {metrics_lowd['mean_error']:.3f}"
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Section 4: Test Cell Type Classification on Mixed Populations

    Apply supervised and unsupervised classifiers to identify cell types in mixed populations.
    """)
    return


@app.cell
def _(
    compare_classifiers,
    extract_cell_features,
    mixed_activity,
    mixed_meta,
    np,
):
    print("Extracting features from mixed population...")
    mixed_features = extract_cell_features(mixed_activity, mixed_meta)
    mixed_cell_types = mixed_meta["cell_types"]
    print(f"Features shape: {mixed_features.shape}")
    print(f"Cell types: {np.unique(mixed_cell_types, return_counts=True)}")
    n_train_1 = len(mixed_features) // 2
    train_feat = mixed_features[:n_train_1]
    train_labels = mixed_cell_types[:n_train_1]
    test_feat = mixed_features[n_train_1:]
    test_labels = mixed_cell_types[n_train_1:]
    print("\nComparing supervised classifiers...")
    classifier_results = compare_classifiers(
        train_feat,
        train_labels,
        test_feat,
        test_labels,
        methods=["random_forest", "svc", "knn", "logistic_regression"],
        random_state=42,
    )
    for _method, _metrics in classifier_results.items():
        if "error" not in _metrics:
            print(
                f"  {_method}: Accuracy = {_metrics['accuracy']:.3f}, F1 = {_metrics['f1']:.3f}"
            )
    return mixed_cell_types, mixed_features


@app.cell
def _(compare_clusterers, mixed_cell_types, mixed_features, np):
    # Test unsupervised clustering
    print("\nComparing unsupervised clusterers...")
    n_clusters = len(np.unique(mixed_cell_types))
    clustering_results = compare_clusterers(
        mixed_features,
        n_clusters=n_clusters,
        true_labels=mixed_cell_types,
        methods=["kmeans", "gaussian_mixture", "agglomerative"],
        random_state=42,
    )
    for _method, _metrics in clustering_results.items():
        if "error" not in _metrics:
            # Display results
            print(
                f"  {_method}: ARI = {_metrics.get('adjusted_rand_score', np.nan):.3f}, Silhouette = {_metrics['silhouette_score']:.3f}"
            )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Section 5: Study How Noise Affects Embedding Quality

    Generate datasets with varying noise levels and analyze how noise affects embedding quality, structure index, and decoding performance.
    """)
    return


@app.cell
def _(
    compute_embedding,
    compute_structure_index,
    evaluate_decoder,
    generate_place_cells,
    np,
    pd,
    shape_distance,
):
    print("Generating datasets with varying noise...")
    noise_levels = np.linspace(0.0, 1.0, 6)
    noise_results = []
    for noise in noise_levels:
        print(f"\nNoise level: {noise:.2f}")
        _activity, meta = generate_place_cells(
            n_cells=25,
            n_samples=1000,
            arena_size=(2.0, 2.0),
            noise_level=noise,
            seed=42,
            plot=False,
        )
        positions = meta["positions"]
        try:
            _si, _, _, _ = compute_structure_index(
                _activity, positions, n_bins=10, n_neighbors=15, num_shuffles=5
            )
        except:
            _si = np.nan
        embeddings_noise = {}
        for _method in ["pca", "umap"]:
            try:
                _emb = compute_embedding(
                    _activity, method=_method, n_components=2, random_state=42
                )
                embeddings_noise[_method] = _emb
            except:
                embeddings_noise[_method] = None
        if noise > 0 and embeddings_noise.get("pca") is not None:
            clean_activity, _ = generate_place_cells(
                n_cells=25,
                n_samples=1000,
                arena_size=(2.0, 2.0),
                noise_level=0.0,
                seed=42,
                plot=False,
            )
            clean_emb = compute_embedding(
                clean_activity, method="pca", n_components=2, random_state=42
            )
            _shape_dist, _, _ = shape_distance(
                clean_emb[:500], embeddings_noise["pca"][:500], method="procrustes"
            )
        else:
            _shape_dist = 0.0
        n_train_2 = 700
        train_act_1 = _activity[:n_train_2]
        train_pos_1 = positions[:n_train_2]
        test_act_1 = _activity[n_train_2:]
        test_pos_1 = positions[n_train_2:]
        _metrics = evaluate_decoder(
            train_act_1, train_pos_1, test_act_1, test_pos_1, decoder="knn", k=5
        )
        noise_results.append(
            {
                "noise": noise,
                "si": _si,
                "shape_dist": _shape_dist,
                "decoding_error": _metrics["mean_error"],
                "decoding_r2": _metrics["r2_score"],
            }
        )
        print(
            f"  SI: {_si:.3f}, Shape dist: {_shape_dist:.3f}, Decoding error: {_metrics['mean_error']:.3f}"
        )
    noise_df = pd.DataFrame(noise_results)
    print("\nNoise Analysis Results:")
    print(noise_df.round(3))
    return (noise_df,)


@app.cell
def _(GridLayoutConfig, PlotConfig, PlotGrid, PlotSpec, noise_df):
    # Visualize noise effects using PlotGrid
    import matplotlib.pyplot as _plt

    _plot_specs = []
    spec1 = PlotSpec(
        data={"x": noise_df["noise"], "y": noise_df["si"]},
        plot_type="line",
        subplot_position=0,
        title="Structure Index vs Noise",
        color="steelblue",
        marker="o",
        line_width=2,
        kwargs={"x_label": "Noise Level", "y_label": "Structure Index"},
    )
    # Structure Index vs Noise
    _plot_specs.append(spec1)
    spec2 = PlotSpec(
        data={"x": noise_df["noise"], "y": noise_df["decoding_error"]},
        plot_type="line",
        subplot_position=1,
        title="Decoding Error vs Noise",
        color="coral",
        marker="o",
        line_width=2,
        kwargs={"x_label": "Noise Level", "y_label": "Mean Error (m)"},
    )
    _plot_specs.append(spec2)
    spec3 = PlotSpec(
        data={"x": noise_df["noise"], "y": noise_df["shape_dist"]},
        plot_type="line",
        subplot_position=2,
        title="Shape Distance vs Noise",
        color="green",
        marker="o",
        line_width=2,
        kwargs={"x_label": "Noise Level", "y_label": "Shape Distance"},
    )
    _plot_specs.append(spec3)
    _grid = PlotGrid(
        plot_specs=_plot_specs,
        config=PlotConfig(figsize=(15, 5)),
        layout=GridLayoutConfig(rows=1, cols=3),
        backend="matplotlib",
    )
    # Decoding Error vs Noise
    # Shape Distance vs Noise
    _fig = _grid.plot()
    # Display the plot - matplotlib returns (fig, axes) tuple
    if isinstance(_fig, tuple):
        _fig_obj, _axes = _fig
        _plt.tight_layout()
        _plt.show()
    else:
        _plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary

    This comprehensive example demonstrated:

    1. **Structure Index Analysis** - Quantified manifold organization across different cell types
    2. **Dimensionality Reduction Benchmarking** - Compared 7 DR methods on synthetic data
    3. **Decoding Validation** - Tested population vector and k-NN decoders with ground truth
    4. **Cell Type Classification** - Applied supervised and unsupervised methods to mixed populations
    5. **Noise Impact Study** - Analyzed how noise affects embedding quality and decoding performance

    All methods are integrated and ready for use with your neural data!
    """)
    return


if __name__ == "__main__":
    app.run()
