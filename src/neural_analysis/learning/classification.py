"""Classification and clustering methods for neural data analysis.

This module provides functions for classifying and clustering neural cells
based on their activity patterns. It includes both supervised classification
and unsupervised clustering methods using standard sklearn implementations.

Supported supervised classifiers:
- RandomForestClassifier: Ensemble method
- SVC: Support Vector Classifier (linear and RBF kernels)
- LogisticRegression: Linear classifier with regularization
- KNeighborsClassifier: Instance-based learning
- GaussianNB: Naive Bayes
- MLPClassifier: Neural network
- GradientBoostingClassifier: Boosting ensemble
- AdaBoostClassifier: Adaptive boosting

Supported unsupervised clusterers:
- KMeans: Partition-based clustering
- DBSCAN: Density-based clustering
- AgglomerativeClustering: Hierarchical clustering
- GaussianMixture: Probabilistic clustering
- SpectralClustering: Graph-based clustering
- Birch: Memory-efficient clustering
- MeanShift: Mode-seeking clustering

Examples:
    Supervised classification:
        >>> from neural_analysis.learning.classification import classify_cells
        >>> from neural_analysis.data.synthetic_data import generate_mixed_population_flexible
        >>> activity, meta = generate_mixed_population_flexible(n_samples=1000)
        >>> features = extract_cell_features(activity, meta)
        >>> predictions = classify_cells(
        ...     features, meta['cell_types'], features,
        ...     method='random_forest'
        ... )

    Unsupervised clustering:
        >>> from neural_analysis.learning.classification import cluster_cells
        >>> labels = cluster_cells(features, method='kmeans', n_clusters=4)
"""

from __future__ import annotations

import logging
import time
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from sklearn.cluster import (
    DBSCAN,
    AgglomerativeClustering,
    Birch,
    KMeans,
    MeanShift,
    SpectralClustering,
)
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    classification_report,
    completeness_score,
    confusion_matrix,
    f1_score,
    homogeneity_score,
    precision_score,
    recall_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from neural_analysis.utils.logging import get_logger, log_calls

logger = get_logger(__name__)

# Type aliases
SupervisedMethod = Literal[
    "random_forest",
    "svc",
    "svc_rbf",
    "logistic_regression",
    "knn",
    "naive_bayes",
    "mlp",
    "gradient_boosting",
    "adaboost",
]

UnsupervisedMethod = Literal[
    "kmeans",
    "dbscan",
    "agglomerative",
    "gaussian_mixture",
    "spectral",
    "birch",
    "mean_shift",
]

__all__ = [
    "classify_cells",
    "cluster_cells",
    "extract_cell_features",
    "compare_classifiers",
    "compare_clusterers",
    "evaluate_classifier",
    "evaluate_clustering",
    "train_classifier",
    "fit_clusterer",
    "SupervisedMethod",
    "UnsupervisedMethod",
]


@log_calls(level=logging.DEBUG)
def extract_cell_features(
    activity: npt.NDArray[np.floating],
    metadata: dict[str, Any] | None = None,
    positions: npt.NDArray[np.floating] | None = None,
    head_directions: npt.NDArray[np.floating] | None = None,
) -> npt.NDArray[np.float64]:
    """Extract features from cell activity patterns for classification.

    Computes various statistical and spatial features from neural activity
    that can be used to classify cell types (place, grid, head_direction, random).

    Args:
        activity: Neural activity matrix, shape (n_samples, n_cells).
        metadata: Optional metadata dictionary from synthetic data generation.
            If provided, extracts positions and head_directions from it.
        positions: Optional position trajectory, shape (n_samples, n_dims).
            If None and metadata provided, uses metadata['positions'].
        head_directions: Optional head direction angles, shape (n_samples,).
            If None and metadata provided, uses metadata['head_directions'].

    Returns:
        features: Feature matrix, shape (n_cells, n_features).
            Features include:
            - Mean firing rate
            - Firing rate std
            - Coefficient of variation
            - Spatial information (if positions available)
            - Periodicity index (if positions available)
            - Directional tuning strength (if head_directions available)
            - Temporal autocorrelation
            - Peak firing rate
            - Sparsity index

    Examples:
        >>> from neural_analysis.data.synthetic_data import generate_place_cells
        >>> activity, meta = generate_place_cells(50, 1000)
        >>> features = extract_cell_features(activity, meta)
        >>> print(f"Features shape: {features.shape}")  # (50, n_features)
    """
    activity = np.asarray(activity, dtype=np.float64)
    n_samples, n_cells = activity.shape

    # Extract positions and head directions from metadata if available
    if metadata is not None:
        if positions is None and "positions" in metadata:
            positions = metadata["positions"]
        if head_directions is None and "head_directions" in metadata:
            head_directions = metadata["head_directions"]

    features_list = []

    # Basic firing statistics
    mean_rate = np.mean(activity, axis=0)
    std_rate = np.std(activity, axis=0)
    cv = std_rate / (mean_rate + 1e-10)  # Coefficient of variation
    peak_rate = np.max(activity, axis=0)

    features_list.extend([mean_rate, std_rate, cv, peak_rate])

    # Sparsity index (fraction of time bins with significant activity)
    threshold = np.percentile(activity, 75, axis=0)
    sparsity = np.mean(activity > threshold[:, None].T, axis=0)

    features_list.append(sparsity)

    # Temporal autocorrelation (lag-1 correlation)
    if n_samples > 1:
        autocorr = np.array(
            [
                np.corrcoef(activity[:-1, i], activity[1:, i])[0, 1]
                if np.std(activity[:, i]) > 0
                else 0.0
                for i in range(n_cells)
            ]
        )
    else:
        autocorr = np.zeros(n_cells)
    features_list.append(autocorr)

    # Spatial information (if positions available)
    if positions is not None:
        positions = np.asarray(positions)
        if positions.ndim == 1:
            positions = positions.reshape(-1, 1)

        # Compute spatial information (mutual information between position and firing)
        spatial_info = np.zeros(n_cells)
        for i in range(n_cells):
            # Bin positions
            n_bins = min(20, int(np.sqrt(n_samples)))
            pos_binned, _ = np.histogramdd(positions, bins=n_bins)
            pos_binned = pos_binned / (pos_binned.sum() + 1e-10)

            # Bin activity
            activity_binned = np.zeros_like(pos_binned)
            for j in range(n_samples):
                bin_idx = tuple(
                    int(min(n_bins - 1, (positions[j, d] - positions[:, d].min()) /
                        (positions[:, d].max() - positions[:, d].min() + 1e-10) * n_bins))
                    for d in range(positions.shape[1])
                )
                if len(bin_idx) == 1:
                    bin_idx = (bin_idx[0],)
                activity_binned[bin_idx] += activity[j, i]

            activity_binned = activity_binned / (activity_binned.sum() + 1e-10)

            # Compute spatial information (simplified)
            mean_rate_pos = activity_binned / (pos_binned + 1e-10)
            spatial_info[i] = np.sum(
                pos_binned * mean_rate_pos * np.log(mean_rate_pos + 1e-10)
            )

        features_list.append(spatial_info)

        # Periodicity index (for grid cells)
        # Compute FFT power at different frequencies
        periodicity = np.zeros(n_cells)
        for i in range(n_cells):
            # Project activity onto position trajectory
            if positions.shape[1] == 1:
                # 1D: use position directly
                pos_norm = (positions[:, 0] - positions[:, 0].min()) / (
                    positions[:, 0].max() - positions[:, 0].min() + 1e-10
                )
                activity_proj = activity[:, i]
            else:
                # 2D+: use distance from origin
                pos_norm = np.linalg.norm(positions - positions.mean(axis=0), axis=1)
                pos_norm = (pos_norm - pos_norm.min()) / (pos_norm.max() - pos_norm.min() + 1e-10)
                activity_proj = activity[:, i]

            # Compute FFT and find dominant frequency
            if len(activity_proj) > 10:
                fft_vals = np.abs(np.fft.rfft(activity_proj))
                if len(fft_vals) > 1:
                    periodicity[i] = np.max(fft_vals[1:]) / (np.mean(fft_vals) + 1e-10)
        features_list.append(periodicity)
    else:
        # Fill with zeros if no positions
        features_list.append(np.zeros(n_cells))
        features_list.append(np.zeros(n_cells))

    # Directional tuning strength (if head directions available)
    if head_directions is not None:
        head_directions = np.asarray(head_directions).ravel()
        # Compute circular variance (1 - |mean(exp(i*angle))|)
        directional_tuning = np.zeros(n_cells)
        for i in range(n_cells):
            # Weight angles by activity
            weights = activity[:, i]
            if weights.sum() > 0:
                complex_mean = np.sum(weights * np.exp(1j * head_directions)) / weights.sum()
                directional_tuning[i] = 1 - np.abs(complex_mean)
            else:
                directional_tuning[i] = 1.0
        features_list.append(directional_tuning)
    else:
        features_list.append(np.zeros(n_cells))

    # Stack all features
    features = np.column_stack(features_list)
    return features.astype(np.float64)


@log_calls(level=logging.DEBUG)
def _get_supervised_classifier(
    method: SupervisedMethod,
    random_state: int | None = None,
    **kwargs: Any,
) -> Any:
    """Get sklearn classifier instance for specified method.

    Args:
        method: Classification method name.
        random_state: Random seed for reproducibility.
        **kwargs: Additional parameters for the classifier.

    Returns:
        classifier: sklearn classifier instance.
    """
    common_params = {"random_state": random_state}
    common_params.update(kwargs)

    match method:
        case "random_forest":
            return RandomForestClassifier(**common_params)
        case "svc":
            return SVC(kernel="linear", probability=True, **common_params)
        case "svc_rbf":
            return SVC(kernel="rbf", probability=True, **common_params)
        case "logistic_regression":
            return LogisticRegression(max_iter=1000, **common_params)
        case "knn":
            return KNeighborsClassifier(**{k: v for k, v in common_params.items() if k != "random_state"})
        case "naive_bayes":
            return GaussianNB(**{k: v for k, v in common_params.items() if k != "random_state"})
        case "mlp":
            return MLPClassifier(max_iter=1000, **common_params)
        case "gradient_boosting":
            return GradientBoostingClassifier(**common_params)
        case "adaboost":
            return AdaBoostClassifier(**{k: v for k, v in common_params.items() if k != "random_state"})
        case _:
            raise ValueError(
                f"Unknown supervised method: {method}. "
                f"Choose from: {', '.join(['random_forest', 'svc', 'svc_rbf', 'logistic_regression', 'knn', 'naive_bayes', 'mlp', 'gradient_boosting', 'adaboost'])}"
            )


@log_calls(level=logging.DEBUG)
def _get_unsupervised_clusterer(
    method: UnsupervisedMethod,
    n_clusters: int | None = None,
    random_state: int | None = None,
    **kwargs: Any,
) -> Any:
    """Get sklearn clusterer instance for specified method.

    Args:
        method: Clustering method name.
        n_clusters: Number of clusters (required for some methods).
        random_state: Random seed for reproducibility.
        **kwargs: Additional parameters for the clusterer.

    Returns:
        clusterer: sklearn clusterer instance.
    """
    common_params = {"random_state": random_state}
    common_params.update(kwargs)

    match method:
        case "kmeans":
            if n_clusters is None:
                raise ValueError("n_clusters required for kmeans")
            return KMeans(n_clusters=n_clusters, **common_params)
        case "dbscan":
            return DBSCAN(**{k: v for k, v in common_params.items() if k != "random_state"})
        case "agglomerative":
            if n_clusters is None:
                raise ValueError("n_clusters required for agglomerative")
            return AgglomerativeClustering(n_clusters=n_clusters, **{k: v for k, v in common_params.items() if k != "random_state"})
        case "gaussian_mixture":
            if n_clusters is None:
                raise ValueError("n_clusters required for gaussian_mixture")
            return GaussianMixture(n_components=n_clusters, **common_params)
        case "spectral":
            if n_clusters is None:
                raise ValueError("n_clusters required for spectral")
            return SpectralClustering(n_clusters=n_clusters, **common_params)
        case "birch":
            if n_clusters is None:
                raise ValueError("n_clusters required for birch")
            return Birch(n_clusters=n_clusters, **{k: v for k, v in common_params.items() if k != "random_state"})
        case "mean_shift":
            return MeanShift(**{k: v for k, v in common_params.items() if k != "random_state"})
        case _:
            raise ValueError(
                f"Unknown unsupervised method: {method}. "
                f"Choose from: {', '.join(['kmeans', 'dbscan', 'agglomerative', 'gaussian_mixture', 'spectral', 'birch', 'mean_shift'])}"
            )


@log_calls(level=logging.DEBUG)
def classify_cells(
    train_features: npt.NDArray[np.floating],
    train_labels: npt.NDArray[Any],
    test_features: npt.NDArray[np.floating],
    method: SupervisedMethod = "random_forest",
    return_proba: bool = False,
    random_state: int | None = None,
    **kwargs: Any,
) -> npt.NDArray[Any] | tuple[npt.NDArray[Any], npt.NDArray[np.float64]]:
    """Classify cells using supervised learning.

    Args:
        train_features: Training feature matrix, shape (n_train, n_features).
        train_labels: Training labels, shape (n_train,).
        test_features: Test feature matrix, shape (n_test, n_features).
        method: Classification method to use.
        return_proba: If True, return class probabilities in addition to predictions.
        random_state: Random seed for reproducibility.
        **kwargs: Additional parameters for the classifier.

    Returns:
        predictions: Predicted labels, shape (n_test,).
        probabilities: (optional) Class probabilities, shape (n_test, n_classes).
            Only returned if return_proba=True.

    Examples:
        >>> from neural_analysis.learning.classification import classify_cells, extract_cell_features
        >>> from neural_analysis.data.synthetic_data import generate_mixed_population_flexible
        >>> activity, meta = generate_mixed_population_flexible(n_samples=1000)
        >>> features = extract_cell_features(activity, meta)
        >>> # Split into train/test
        >>> train_idx = np.arange(len(features) // 2)
        >>> test_idx = np.arange(len(features) // 2, len(features))
        >>> predictions = classify_cells(
        ...     features[train_idx], meta['cell_types'][train_idx],
        ...     features[test_idx], method='random_forest'
        ... )
    """
    train_features = np.asarray(train_features, dtype=np.float64)
    test_features = np.asarray(test_features, dtype=np.float64)
    train_labels = np.asarray(train_labels)

    # Get classifier
    classifier = _get_supervised_classifier(method, random_state=random_state, **kwargs)

    # Train
    classifier.fit(train_features, train_labels)

    # Predict
    predictions = classifier.predict(test_features)
    predictions = np.asarray(predictions)

    if return_proba:
        probabilities = classifier.predict_proba(test_features)
        probabilities = np.asarray(probabilities, dtype=np.float64)
        return predictions, probabilities
    return predictions


@log_calls(level=logging.DEBUG)
def cluster_cells(
    features: npt.NDArray[np.floating],
    method: UnsupervisedMethod = "kmeans",
    n_clusters: int | None = None,
    random_state: int | None = None,
    **kwargs: Any,
) -> npt.NDArray[np.int64]:
    """Cluster cells using unsupervised learning.

    Args:
        features: Feature matrix, shape (n_cells, n_features).
        method: Clustering method to use.
        n_clusters: Number of clusters (required for some methods).
        random_state: Random seed for reproducibility.
        **kwargs: Additional parameters for the clusterer.

    Returns:
        labels: Cluster labels, shape (n_cells,). -1 indicates noise (DBSCAN).

    Examples:
        >>> from neural_analysis.learning.classification import cluster_cells, extract_cell_features
        >>> from neural_analysis.data.synthetic_data import generate_mixed_population_flexible
        >>> activity, meta = generate_mixed_population_flexible(n_samples=1000)
        >>> features = extract_cell_features(activity, meta)
        >>> labels = cluster_cells(features, method='kmeans', n_clusters=4)
    """
    features = np.asarray(features, dtype=np.float64)

    # Get clusterer
    clusterer = _get_unsupervised_clusterer(
        method, n_clusters=n_clusters, random_state=random_state, **kwargs
    )

    # Fit and predict
    if method == "gaussian_mixture":
        clusterer.fit(features)
        labels = clusterer.predict(features)
    else:
        labels = clusterer.fit_predict(features)

    labels = np.asarray(labels, dtype=np.int64)
    return labels


@log_calls(level=logging.DEBUG)
def evaluate_classifier(
    y_true: npt.NDArray[Any],
    y_pred: npt.NDArray[Any],
    return_confusion_matrix: bool = True,
) -> dict[str, Any]:
    """Evaluate classifier performance.

    Args:
        y_true: True labels, shape (n_samples,).
        y_pred: Predicted labels, shape (n_samples,).
        return_confusion_matrix: If True, include confusion matrix in results.

    Returns:
        metrics: Dictionary with:
            - 'accuracy': Overall accuracy
            - 'precision': Precision score (macro-averaged)
            - 'recall': Recall score (macro-averaged)
            - 'f1': F1 score (macro-averaged)
            - 'confusion_matrix': Confusion matrix (if return_confusion_matrix=True)
            - 'classification_report': Text classification report

    Examples:
        >>> from neural_analysis.learning.classification import evaluate_classifier
        >>> metrics = evaluate_classifier(y_true, y_pred)
        >>> print(f"Accuracy: {metrics['accuracy']:.3f}")
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }

    if return_confusion_matrix:
        metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred)

    metrics["classification_report"] = classification_report(y_true, y_pred, zero_division=0)

    return metrics


@log_calls(level=logging.DEBUG)
def evaluate_clustering(
    features: npt.NDArray[np.floating],
    labels: npt.NDArray[np.integer],
    true_labels: npt.NDArray[Any] | None = None,
) -> dict[str, float]:
    """Evaluate clustering performance.

    Args:
        features: Feature matrix, shape (n_samples, n_features).
        labels: Cluster labels, shape (n_samples,).
        true_labels: Optional true labels for external validation metrics.

    Returns:
        metrics: Dictionary with:
            - 'silhouette_score': Silhouette coefficient
            - 'adjusted_rand_score': Adjusted Rand Index (if true_labels provided)
            - 'homogeneity': Homogeneity score (if true_labels provided)
            - 'completeness': Completeness score (if true_labels provided)

    Examples:
        >>> from neural_analysis.learning.classification import evaluate_clustering
        >>> metrics = evaluate_clustering(features, labels, true_labels)
        >>> print(f"Silhouette: {metrics['silhouette_score']:.3f}")
    """
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    metrics: dict[str, float] = {}

    # Silhouette score (always computable)
    if len(np.unique(labels)) > 1:
        metrics["silhouette_score"] = float(
            silhouette_score(features, labels)
        )
    else:
        metrics["silhouette_score"] = -1.0

    # External validation metrics (if true labels provided)
    if true_labels is not None:
        true_labels = np.asarray(true_labels)
        metrics["adjusted_rand_score"] = float(
            adjusted_rand_score(true_labels, labels)
        )
        metrics["homogeneity"] = float(
            homogeneity_score(true_labels, labels)
        )
        metrics["completeness"] = float(
            completeness_score(true_labels, labels)
        )

    return metrics


@log_calls(level=logging.DEBUG)
def train_classifier(
    features: npt.NDArray[np.floating],
    labels: npt.NDArray[Any],
    method: SupervisedMethod = "random_forest",
    cv: int = 5,
    random_state: int | None = None,
    **kwargs: Any,
) -> tuple[Any, dict[str, Any]]:
    """Train classifier with cross-validation.

    Args:
        features: Feature matrix, shape (n_samples, n_features).
        labels: Labels, shape (n_samples,).
        method: Classification method.
        cv: Number of cross-validation folds.
        random_state: Random seed.
        **kwargs: Additional classifier parameters.

    Returns:
        classifier: Trained classifier.
        cv_scores: Dictionary with cross-validation results.

    Examples:
        >>> from neural_analysis.learning.classification import train_classifier
        >>> classifier, cv_scores = train_classifier(features, labels, method='random_forest')
        >>> print(f"CV accuracy: {cv_scores['mean']:.3f} ± {cv_scores['std']:.3f}")
    """
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels)

    classifier = _get_supervised_classifier(method, random_state=random_state, **kwargs)

    # Cross-validation
    cv_scores = cross_val_score(classifier, features, labels, cv=cv, scoring="accuracy")
    cv_results = {
        "mean": float(np.mean(cv_scores)),
        "std": float(np.std(cv_scores)),
        "scores": cv_scores.tolist(),
    }

    # Train on full data
    classifier.fit(features, labels)

    return classifier, cv_results


@log_calls(level=logging.DEBUG)
def fit_clusterer(
    features: npt.NDArray[np.floating],
    method: UnsupervisedMethod = "kmeans",
    n_clusters: int | None = None,
    random_state: int | None = None,
    **kwargs: Any,
) -> tuple[Any, npt.NDArray[np.int64]]:
    """Fit clusterer and return labels.

    Args:
        features: Feature matrix, shape (n_samples, n_features).
        method: Clustering method.
        n_clusters: Number of clusters.
        random_state: Random seed.
        **kwargs: Additional clusterer parameters.

    Returns:
        clusterer: Fitted clusterer.
        labels: Cluster labels.

    Examples:
        >>> from neural_analysis.learning.classification import fit_clusterer
        >>> clusterer, labels = fit_clusterer(features, method='kmeans', n_clusters=4)
    """
    features = np.asarray(features, dtype=np.float64)

    clusterer = _get_unsupervised_clusterer(
        method, n_clusters=n_clusters, random_state=random_state, **kwargs
    )

    if method == "gaussian_mixture":
        clusterer.fit(features)
        labels = clusterer.predict(features)
    else:
        labels = clusterer.fit_predict(features)

    return clusterer, labels.astype(np.int64)


@log_calls(level=logging.DEBUG)
def compare_classifiers(
    train_features: npt.NDArray[np.floating],
    train_labels: npt.NDArray[Any],
    test_features: npt.NDArray[np.floating],
    test_labels: npt.NDArray[Any],
    methods: list[SupervisedMethod] | None = None,
    random_state: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Compare multiple supervised classifiers.

    Args:
        train_features: Training features, shape (n_train, n_features).
        train_labels: Training labels, shape (n_train,).
        test_features: Test features, shape (n_test, n_features).
        test_labels: Test labels, shape (n_test,).
        methods: List of methods to compare. If None, uses all methods.
        random_state: Random seed.

    Returns:
        results: Dictionary mapping method names to evaluation metrics and timing.

    Examples:
        >>> from neural_analysis.learning.classification import compare_classifiers
        >>> results = compare_classifiers(train_features, train_labels, test_features, test_labels)
        >>> for method, metrics in results.items():
        ...     print(f"{method}: {metrics['accuracy']:.3f}")
    """
    if methods is None:
        methods = [
            "random_forest",
            "svc",
            "svc_rbf",
            "logistic_regression",
            "knn",
            "naive_bayes",
            "mlp",
            "gradient_boosting",
            "adaboost",
        ]

    results: dict[str, dict[str, Any]] = {}

    for method in methods:
        start_time = time.time()
        try:
            predictions = classify_cells(
                train_features,
                train_labels,
                test_features,
                method=method,
                random_state=random_state,
            )
            elapsed_time = time.time() - start_time

            metrics = evaluate_classifier(test_labels, predictions)
            metrics["time"] = elapsed_time
            results[method] = metrics
        except Exception as e:
            logger.warning(f"Method {method} failed: {e}")
            results[method] = {"error": str(e)}

    return results


@log_calls(level=logging.DEBUG)
def compare_clusterers(
    features: npt.NDArray[np.floating],
    n_clusters: int | None = None,
    true_labels: npt.NDArray[Any] | None = None,
    methods: list[UnsupervisedMethod] | None = None,
    random_state: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Compare multiple unsupervised clusterers.

    Args:
        features: Feature matrix, shape (n_samples, n_features).
        n_clusters: Number of clusters (for methods that require it).
        true_labels: Optional true labels for external validation.
        methods: List of methods to compare. If None, uses all methods.
        random_state: Random seed.

    Returns:
        results: Dictionary mapping method names to evaluation metrics, labels, and timing.

    Examples:
        >>> from neural_analysis.learning.classification import compare_clusterers
        >>> results = compare_clusterers(features, n_clusters=4, true_labels=cell_types)
        >>> for method, metrics in results.items():
        ...     print(f"{method}: silhouette={metrics['silhouette_score']:.3f}")
    """
    if methods is None:
        methods = [
            "kmeans",
            "dbscan",
            "agglomerative",
            "gaussian_mixture",
            "spectral",
            "birch",
            "mean_shift",
        ]

    results: dict[str, dict[str, Any]] = {}

    for method in methods:
        start_time = time.time()
        try:
            # Some methods don't need n_clusters
            if method in ["dbscan", "mean_shift"]:
                labels = cluster_cells(
                    features, method=method, random_state=random_state
                )
            else:
                if n_clusters is None:
                    logger.warning(f"Skipping {method}: n_clusters required")
                    continue
                labels = cluster_cells(
                    features,
                    method=method,
                    n_clusters=n_clusters,
                    random_state=random_state,
                )

            elapsed_time = time.time() - start_time

            metrics = evaluate_clustering(features, labels, true_labels)
            metrics["time"] = elapsed_time
            metrics["n_clusters_found"] = len(np.unique(labels[labels >= 0]))
            metrics["labels"] = labels
            results[method] = metrics
        except Exception as e:
            logger.warning(f"Method {method} failed: {e}")
            results[method] = {"error": str(e)}

    return results

