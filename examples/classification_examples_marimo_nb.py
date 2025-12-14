import marimo


__generated_with = "0.18.3"

app = marimo.App(width="full")


@app.cell(hide_code=True)
def __():

    import marimo as mo

    return mo


@app.cell()
def _(mo):
    mo.md(
        r"""
    # Classification and Clustering Examples

    This notebook demonstrates the comprehensive classification and clustering methods available in `neural_analysis.learning.classification`.

    ## Features Demonstrated

    1. **Feature Extraction** - Extract activity-based features from neural data
    2. **Supervised Classification** - 9 different classifiers (Random Forest, SVM, Logistic Regression, etc.)
    3. **Unsupervised Clustering** - 7 different clusterers (KMeans, DBSCAN, Gaussian Mixture, etc.)
    4. **Performance Comparison** - Benchmark all methods on the same data
    5. **Integration** - Use with structure_index and shape_distance for validation

    All visualizations use the **PlotGrid system** for consistency.
    """
    )
    return


@app.cell
def _():
    # Imports
    import numpy as np
    import pandas as pd
    from pathlib import Path
    import warnings

    warnings.filterwarnings("ignore")

    # Neural analysis imports
    from neural_analysis.data.synthetic_data import generate_mixed_population_flexible
    from neural_analysis.learning.classification import (
        classify_cells,
        cluster_cells,
        compare_classifiers,
        compare_clusterers,
        evaluate_classifier,
        evaluate_clustering,
        extract_cell_features,
        fit_clusterer,
        train_classifier,
    )
    from neural_analysis.plotting import (
        PlotGrid,
        PlotSpec,
        GridLayoutConfig,
        PlotConfig,
    )

    # Set random seed
    np.random.seed(42)

    print("✓ Imports successful")
    return (
        GridLayoutConfig,
        PlotConfig,
        PlotGrid,
        PlotSpec,
        classify_cells,
        cluster_cells,
        compare_classifiers,
        compare_clusterers,
        evaluate_classifier,
        evaluate_clustering,
        extract_cell_features,
        generate_mixed_population_flexible,
        np,
        pd,
        train_classifier,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Section 1: Generate Synthetic Data and Extract Features

    We'll generate a mixed population of neural cells (place, grid, head_direction, random) and extract features from their activity patterns.
    """
    )
    return


@app.cell
def _(generate_mixed_population_flexible, np):
    # Generate mixed population
    activity, meta = generate_mixed_population_flexible(
        n_samples=1000,
        seed=42,
        plot=False,  # Disable automatic plotting, we'll use PlotGrid
    )

    print(f"Activity shape: {activity.shape}")
    print(f"Cell types: {np.unique(meta['cell_types'], return_counts=True)}")
    print(f"Total cells: {len(meta['cell_types'])}")
    return activity, meta


@app.cell
def _(activity, extract_cell_features, meta):
    # Extract features from cell activity
    features = extract_cell_features(activity, meta)

    print(f"Features shape: {features.shape}")
    print(f"Number of features per cell: {features.shape[1]}")

    # Display feature names (approximate)
    feature_names = [
        "mean_rate",
        "std_rate",
        "cv",
        "peak_rate",
        "sparsity",
        "autocorr",
        "spatial_info",
        "periodicity",
        "directional_tuning",
    ]
    print(f"\nFeature types: {feature_names[:features.shape[1]]}")
    return (features,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Section 2: Supervised Classification

    We'll train and test multiple supervised classifiers to identify cell types from their activity features.
    """
    )
    return


@app.cell
def _(features, meta, np):
    # Split into train/test
    cell_types = meta["cell_types"]
    n_train = len(features) // 2

    train_features = features[:n_train]
    train_labels = cell_types[:n_train]
    test_features = features[n_train:]
    test_labels = cell_types[n_train:]

    print(f"Training samples: {len(train_features)}")
    print(f"Test samples: {len(test_features)}")
    print(f"Training labels: {np.unique(train_labels, return_counts=True)}")
    return cell_types, test_features, test_labels, train_features, train_labels


@app.cell
def _(
    classify_cells,
    evaluate_classifier,
    test_features,
    test_labels,
    train_features,
    train_labels,
):
    # Test individual classifier
    predictions = classify_cells(
        train_features,
        train_labels,
        test_features,
        method="random_forest",
        random_state=42,
    )
    _metrics = evaluate_classifier(test_labels, predictions)
    print("Random Forest Results:")
    print(f"  Accuracy: {_metrics['accuracy']:.3f}")
    print(f"  Precision: {_metrics['precision']:.3f}")
    # Evaluate
    print(f"  Recall: {_metrics['recall']:.3f}")
    print(f"  F1: {_metrics['f1']:.3f}")
    return


@app.cell
def _(
    compare_classifiers,
    np,
    pd,
    test_features,
    test_labels,
    train_features,
    train_labels,
):
    # Compare all supervised classifiers
    print("Comparing all supervised classifiers...")
    results = compare_classifiers(
        train_features,
        train_labels,
        test_features,
        test_labels,
        methods=[
            "random_forest",
            "svc",
            "svc_rbf",
            "logistic_regression",
            "knn",
            "naive_bayes",
        ],
        random_state=42,
    )
    results_df = pd.DataFrame(
        {
            method: {
                "accuracy": _metrics.get("accuracy", np.nan),
                "precision": _metrics.get("precision", np.nan),
                "recall": _metrics.get("recall", np.nan),
                "f1": _metrics.get("f1", np.nan),
                "time": _metrics.get("time", np.nan),
            }
            for method, _metrics in results.items()
            if "error" not in _metrics
        }
    ).T
    print("\nClassifier Comparison:")
    # Create results DataFrame
    print(results_df.round(3))
    return (results_df,)


@app.cell
def _(GridLayoutConfig, PlotConfig, PlotGrid, PlotSpec, results_df):
    # Visualize comparison using PlotGrid
    if len(results_df) > 0:
        _plot_specs = []
        _spec1 = PlotSpec(
            data={"x": results_df.index, "y": results_df["accuracy"]},
            plot_type="bar",
            subplot_position=0,
            title="Classification Accuracy",
            color="steelblue",
            kwargs={"x_label": "Method", "y_label": "Accuracy"},
        )
        _plot_specs.append(_spec1)  # Accuracy comparison
        _spec2 = PlotSpec(
            data={"x": results_df.index, "y": results_df["f1"]},
            plot_type="bar",
            subplot_position=1,
            title="F1 Score",
            color="coral",
            kwargs={"x_label": "Method", "y_label": "F1 Score"},
        )
        _plot_specs.append(_spec2)
        _spec3 = PlotSpec(
            data={"x": results_df.index, "y": results_df["time"]},
            plot_type="bar",
            subplot_position=2,
            title="Computation Time",
            color="green",
            kwargs={"x_label": "Method", "y_label": "Time (s)"},
        )
        _plot_specs.append(_spec3)
        _grid = PlotGrid(
            plot_specs=_plot_specs,
            config=PlotConfig(figsize=(15, 5)),
            layout=GridLayoutConfig(rows=1, cols=3),
            backend="matplotlib",
        )
        _fig = _grid.plot()  # F1 score comparison  # Time comparison
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Section 3: Unsupervised Clustering

    We'll apply multiple clustering methods to identify cell types without using labels.
    """
    )
    return


@app.cell
def _(cell_types, cluster_cells, evaluate_clustering, features, np):
    # Determine number of clusters from ground truth
    n_clusters = len(np.unique(cell_types))
    print(f"Number of cell types (clusters): {n_clusters}")
    labels = cluster_cells(
        features, method="kmeans", n_clusters=n_clusters, random_state=42
    )
    # Test individual clusterer
    _metrics = evaluate_clustering(features, labels, true_labels=cell_types)
    print("\nKMeans Results:")
    print(f"  Silhouette Score: {_metrics['silhouette_score']:.3f}")
    print(f"  Adjusted Rand Index: {_metrics.get('adjusted_rand_score', np.nan):.3f}")
    # Evaluate
    print(f"  Homogeneity: {_metrics.get('homogeneity', np.nan):.3f}")
    print(f"  Completeness: {_metrics.get('completeness', np.nan):.3f}")
    return (n_clusters,)


@app.cell
def _(cell_types, compare_clusterers, features, n_clusters, np, pd):
    # Compare all unsupervised clusterers
    print("Comparing all unsupervised clusterers...")
    clustering_results = compare_clusterers(
        features,
        n_clusters=n_clusters,
        true_labels=cell_types,
        methods=["kmeans", "gaussian_mixture", "agglomerative", "spectral", "birch"],
        random_state=42,
    )
    clustering_df = pd.DataFrame(
        {
            method: {
                "silhouette": _metrics.get("silhouette_score", np.nan),
                "ari": _metrics.get("adjusted_rand_score", np.nan),
                "homogeneity": _metrics.get("homogeneity", np.nan),
                "completeness": _metrics.get("completeness", np.nan),
                "time": _metrics.get("time", np.nan),
                "n_clusters_found": _metrics.get("n_clusters_found", np.nan),
            }
            for method, _metrics in clustering_results.items()
            if "error" not in _metrics
        }
    ).T
    print("\nClustering Comparison:")
    # Create results DataFrame
    print(clustering_df.round(3))
    return clustering_df, clustering_results


@app.cell
def _(GridLayoutConfig, PlotConfig, PlotGrid, PlotSpec, clustering_df):
    # Visualize clustering comparison using PlotGrid
    if len(clustering_df) > 0:
        _plot_specs = []
        _spec1 = PlotSpec(
            data={"x": clustering_df.index, "y": clustering_df["silhouette"]},
            plot_type="bar",
            subplot_position=0,
            title="Silhouette Score",
            color="steelblue",
            kwargs={"x_label": "Method", "y_label": "Silhouette Score"},
        )
        _plot_specs.append(_spec1)  # Silhouette score
        _spec2 = PlotSpec(
            data={"x": clustering_df.index, "y": clustering_df["ari"]},
            plot_type="bar",
            subplot_position=1,
            title="Adjusted Rand Index",
            color="coral",
            kwargs={"x_label": "Method", "y_label": "ARI"},
        )
        _plot_specs.append(_spec2)
        _spec3 = PlotSpec(
            data={"x": clustering_df.index, "y": clustering_df["time"]},
            plot_type="bar",
            subplot_position=2,
            title="Computation Time",
            color="green",
            kwargs={"x_label": "Method", "y_label": "Time (s)"},
        )
        _plot_specs.append(_spec3)
        _grid = PlotGrid(
            plot_specs=_plot_specs,
            config=PlotConfig(figsize=(15, 5)),
            layout=GridLayoutConfig(rows=1, cols=3),
            backend="matplotlib",
        )
        _fig = _grid.plot()  # Adjusted Rand Index  # Time comparison
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Section 4: Visualize Clustering Results

    Visualize how different clustering methods partition the feature space.
    """
    )
    return


@app.cell
def _(
    GridLayoutConfig,
    PlotConfig,
    PlotGrid,
    PlotSpec,
    cell_types,
    clustering_results,
    features,
):
    # Use PCA to reduce features to 2D for visualization
    from neural_analysis.embeddings import compute_embedding

    features_2d = compute_embedding(
        features, method="pca", n_components=2, random_state=42
    )
    kmeans_labels = clustering_results["kmeans"]["labels"]
    gmm_labels = clustering_results["gaussian_mixture"]["labels"]
    # Get clustering results for visualization
    _plot_specs = []
    _spec1 = PlotSpec(
        data={"x": features_2d[:, 0], "y": features_2d[:, 1]},
        plot_type="scatter",
        subplot_position=0,
        title="Ground Truth Cell Types",
        color_by=cell_types,
        cmap="Set1",
        marker_size=20,
        alpha=0.6,
        kwargs={"x_label": "PC1", "y_label": "PC2"},
    )
    _plot_specs.append(_spec1)
    # Create visualization
    _spec2 = PlotSpec(
        data={"x": features_2d[:, 0], "y": features_2d[:, 1]},
        plot_type="scatter",
        subplot_position=1,
        title="KMeans Clustering",
        color_by=kmeans_labels,
        cmap="Set1",
        marker_size=20,
        alpha=0.6,
        kwargs={"x_label": "PC1", "y_label": "PC2"},
    )
    _plot_specs.append(_spec2)
    # Ground truth
    _spec3 = PlotSpec(
        data={"x": features_2d[:, 0], "y": features_2d[:, 1]},
        plot_type="scatter",
        subplot_position=2,
        title="Gaussian Mixture Clustering",
        color_by=gmm_labels,
        cmap="Set1",
        marker_size=20,
        alpha=0.6,
        kwargs={"x_label": "PC1", "y_label": "PC2"},
    )
    _plot_specs.append(_spec3)
    _grid = PlotGrid(
        plot_specs=_plot_specs,
        config=PlotConfig(figsize=(15, 5)),
        layout=GridLayoutConfig(rows=1, cols=3),
        backend="matplotlib",
    )
    # KMeans
    # Gaussian Mixture
    _fig = _grid.plot()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Section 5: Cross-Validation Training

    Demonstrate training with cross-validation for robust performance estimation.
    """
    )
    return


@app.cell
def _(cell_types, features, train_classifier):
    # Train classifier with cross-validation
    classifier, cv_scores = train_classifier(
        features, cell_types, method="random_forest", cv=5, random_state=42
    )

    print("Cross-Validation Results:")
    print(f"  Mean Accuracy: {cv_scores['mean']:.3f} ± {cv_scores['std']:.3f}")
    print(f"  Individual CV scores: {[f'{s:.3f}' for s in cv_scores['scores']]}")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Summary

    This notebook demonstrated:
    - Feature extraction from neural activity patterns
    - 9 supervised classification methods
    - 7 unsupervised clustering methods
    - Performance comparison and visualization
    - Cross-validation for robust evaluation

    All methods are ready to use with your neural data!
    """
    )
    return


