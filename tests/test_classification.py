"""Tests for classification and clustering functions."""

import numpy as np

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


class TestExtractCellFeatures:
    """Tests for feature extraction."""

    def test_basic_features(self) -> None:
        """Test basic feature extraction."""
        activity = np.random.rand(100, 50)
        features = extract_cell_features(activity)

        assert features.shape[0] == 50  # n_cells
        assert features.shape[1] > 0  # n_features

    def test_with_metadata(self) -> None:
        """Test feature extraction with metadata."""
        activity, meta = generate_mixed_population_flexible(
            n_samples=500, seed=42, plot=False
        )
        features = extract_cell_features(activity, meta)

        assert features.shape[0] == activity.shape[1]
        assert features.shape[1] > 0

    def test_with_positions(self) -> None:
        """Test feature extraction with positions."""
        activity = np.random.rand(200, 30)
        positions = np.random.rand(200, 2)
        features = extract_cell_features(activity, positions=positions)

        assert features.shape[0] == 30
        # Should have spatial features
        assert features.shape[1] >= 6

    def test_with_head_directions(self) -> None:
        """Test feature extraction with head directions."""
        activity = np.random.rand(200, 30)
        head_directions = np.random.rand(200) * 2 * np.pi
        features = extract_cell_features(activity, head_directions=head_directions)

        assert features.shape[0] == 30
        # Should have directional features
        assert features.shape[1] >= 6


class TestClassifyCells:
    """Tests for supervised classification."""

    def test_random_forest(self) -> None:
        """Test Random Forest classification."""
        activity, meta = generate_mixed_population_flexible(
            n_samples=500, seed=42, plot=False
        )
        features = extract_cell_features(activity, meta)
        cell_types = meta["cell_types"]

        # Split train/test
        n_train = len(features) // 2
        train_features = features[:n_train]
        train_labels = cell_types[:n_train]
        test_features = features[n_train:]
        test_labels = cell_types[n_train:]

        predictions = classify_cells(
            train_features, train_labels, test_features, method="random_forest", random_state=42
        )

        assert len(predictions) == len(test_labels)
        assert all(p in cell_types for p in predictions)

    def test_multiple_methods(self) -> None:
        """Test multiple classification methods."""
        activity, meta = generate_mixed_population_flexible(
            n_samples=500, seed=42, plot=False
        )
        features = extract_cell_features(activity, meta)
        cell_types = meta["cell_types"]

        n_train = len(features) // 2
        train_features = features[:n_train]
        train_labels = cell_types[:n_train]
        test_features = features[n_train:]

        methods = ["random_forest", "svc", "logistic_regression", "knn"]
        for method in methods:
            predictions = classify_cells(
                train_features,
                train_labels,
                test_features,
                method=method,
                random_state=42,
            )
            assert len(predictions) == len(test_features)

    def test_return_proba(self) -> None:
        """Test returning probabilities."""
        activity, meta = generate_mixed_population_flexible(
            n_samples=500, seed=42, plot=False
        )
        features = extract_cell_features(activity, meta)
        cell_types = meta["cell_types"]

        n_train = len(features) // 2
        train_features = features[:n_train]
        train_labels = cell_types[:n_train]
        test_features = features[n_train:]

        predictions, probabilities = classify_cells(
            train_features,
            train_labels,
            test_features,
            method="random_forest",
            return_proba=True,
            random_state=42,
        )

        assert len(predictions) == len(test_features)
        assert probabilities.shape[0] == len(test_features)
        assert probabilities.shape[1] > 0
        assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)


class TestClusterCells:
    """Tests for unsupervised clustering."""

    def test_kmeans(self) -> None:
        """Test KMeans clustering."""
        activity, meta = generate_mixed_population_flexible(
            n_samples=500, seed=42, plot=False
        )
        features = extract_cell_features(activity, meta)

        labels = cluster_cells(features, method="kmeans", n_clusters=4, random_state=42)

        assert len(labels) == len(features)
        assert len(np.unique(labels)) <= 4

    def test_dbscan(self) -> None:
        """Test DBSCAN clustering."""
        activity, meta = generate_mixed_population_flexible(
            n_samples=500, seed=42, plot=False
        )
        features = extract_cell_features(activity, meta)

        labels = cluster_cells(features, method="dbscan", eps=0.5, min_samples=5)

        assert len(labels) == len(features)
        # DBSCAN can have -1 for noise
        assert all(l >= -1 for l in labels)

    def test_gaussian_mixture(self) -> None:
        """Test Gaussian Mixture clustering."""
        activity, meta = generate_mixed_population_flexible(
            n_samples=500, seed=42, plot=False
        )
        features = extract_cell_features(activity, meta)

        labels = cluster_cells(
            features, method="gaussian_mixture", n_clusters=4, random_state=42
        )

        assert len(labels) == len(features)
        assert len(np.unique(labels)) <= 4


class TestEvaluateClassifier:
    """Tests for classifier evaluation."""

    def test_basic_evaluation(self) -> None:
        """Test basic classifier evaluation."""
        y_true = np.array(["A", "B", "A", "B", "A"])
        y_pred = np.array(["A", "B", "A", "B", "B"])

        metrics = evaluate_classifier(y_true, y_pred)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert 0 <= metrics["accuracy"] <= 1

    def test_with_confusion_matrix(self) -> None:
        """Test evaluation with confusion matrix."""
        y_true = np.array(["A", "B", "A", "B"])
        y_pred = np.array(["A", "B", "A", "A"])

        metrics = evaluate_classifier(y_true, y_pred, return_confusion_matrix=True)

        assert "confusion_matrix" in metrics
        assert metrics["confusion_matrix"].shape == (2, 2)


class TestEvaluateClustering:
    """Tests for clustering evaluation."""

    def test_silhouette_score(self) -> None:
        """Test silhouette score computation."""
        features = np.random.rand(100, 10)
        labels = np.random.randint(0, 4, 100)

        metrics = evaluate_clustering(features, labels)

        assert "silhouette_score" in metrics
        assert -1 <= metrics["silhouette_score"] <= 1

    def test_with_true_labels(self) -> None:
        """Test clustering evaluation with true labels."""
        features = np.random.rand(100, 10)
        labels = np.random.randint(0, 4, 100)
        true_labels = np.random.randint(0, 4, 100)

        metrics = evaluate_clustering(features, labels, true_labels)

        assert "silhouette_score" in metrics
        assert "adjusted_rand_score" in metrics
        assert "homogeneity" in metrics
        assert "completeness" in metrics


class TestTrainClassifier:
    """Tests for classifier training with CV."""

    def test_basic_training(self) -> None:
        """Test basic classifier training."""
        activity, meta = generate_mixed_population_flexible(
            n_samples=500, seed=42, plot=False
        )
        features = extract_cell_features(activity, meta)
        cell_types = meta["cell_types"]

        classifier, cv_scores = train_classifier(
            features, cell_types, method="random_forest", cv=3, random_state=42
        )

        assert classifier is not None
        assert "mean" in cv_scores
        assert "std" in cv_scores
        assert 0 <= cv_scores["mean"] <= 1


class TestFitClusterer:
    """Tests for clusterer fitting."""

    def test_basic_fitting(self) -> None:
        """Test basic clusterer fitting."""
        activity, meta = generate_mixed_population_flexible(
            n_samples=500, seed=42, plot=False
        )
        features = extract_cell_features(activity, meta)

        clusterer, labels = fit_clusterer(
            features, method="kmeans", n_clusters=4, random_state=42
        )

        assert clusterer is not None
        assert len(labels) == len(features)


class TestCompareClassifiers:
    """Tests for classifier comparison."""

    def test_compare_all(self) -> None:
        """Test comparing all classifiers."""
        activity, meta = generate_mixed_population_flexible(
            n_samples=500, seed=42, plot=False
        )
        features = extract_cell_features(activity, meta)
        cell_types = meta["cell_types"]

        n_train = len(features) // 2
        train_features = features[:n_train]
        train_labels = cell_types[:n_train]
        test_features = features[n_train:]
        test_labels = cell_types[n_train:]

        results = compare_classifiers(
            train_features,
            train_labels,
            test_features,
            test_labels,
            methods=["random_forest", "svc", "knn"],
            random_state=42,
        )

        assert len(results) > 0
        for _method, metrics in results.items():
            if "error" not in metrics:
                assert "accuracy" in metrics
                assert "time" in metrics


class TestCompareClusterers:
    """Tests for clusterer comparison."""

    def test_compare_all(self) -> None:
        """Test comparing all clusterers."""
        activity, meta = generate_mixed_population_flexible(
            n_samples=500, seed=42, plot=False
        )
        features = extract_cell_features(activity, meta)
        cell_types = meta["cell_types"]

        results = compare_clusterers(
            features,
            n_clusters=4,
            true_labels=cell_types,
            methods=["kmeans", "gaussian_mixture", "agglomerative"],
            random_state=42,
        )

        assert len(results) > 0
        for _method, metrics in results.items():
            if "error" not in metrics:
                assert "silhouette_score" in metrics
                assert "time" in metrics

