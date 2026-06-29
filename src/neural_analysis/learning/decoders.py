import logging
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from .evaluation import compute_classification_metrics, compute_regression_metrics

logger = logging.getLogger(__name__)


def decode(
    embedding_train: npt.NDArray[np.floating],
    embedding_test: npt.NDArray[np.floating],
    labels_train: npt.NDArray[np.floating],
    labels_test: npt.NDArray[np.floating],
    labels_describe_space: bool = False,
    n_neighbors: int | None = None,
    metric: str = "cosine",
    n_folds: int = 5,
    detailed_metrics: bool = False,
    include_cv_stats: bool = False,
    test_outlier_removal: bool = True,
    regression_outlier_removal_threshold: float = 0.004,
    min_train_class_samples: int = 200,
    min_test_class_samples: int = 30,
) -> dict[str, Any]:
    """
    Decodes neural embeddings using k-Nearest Neighbors with automatic k selection.

    Parameters
    ----------
    embedding_train : np.ndarray
        Training embedding data
    embedding_test : np.ndarray
        Testing embedding data
    labels_train : np.ndarray
        Training target labels
    labels_test : np.ndarray
        Testing target labels
    labels_describe_space : bool, optional
        Whether to describe the label space (default: False).
    n_neighbors : int, optional
        Number of neighbors for kNN (default: None, auto-determined via CV)
    metric : str, optional
        Distance metric for kNN (default: "cosine")
    n_folds : int, optional
        Number of folds for cross-validation (default: 5)
    detailed_metrics : bool, optional
        Whether to return detailed per-class metrics (default: False)
    include_cv_stats : bool, optional
        Whether to include cross-validation statistics (default: False)
    test_outlier_removal : bool, optional
        Whether to remove outliers from the test set (default: True)
    regression_outlier_removal_threshold : float, optional
        Threshold for outlier removal for continuous variables (default: 0.004)
    min_train_class_samples : int, optional
        Minimum number of training samples per class (default: 200)
    min_test_class_samples : int, optional
        Minimum number of test samples per class (default: 30)

    Returns
    -------
    Dict[str, Any]
        Dictionary containing decoding performance metrics
    """
    if not all(
        isinstance(x, np.ndarray)
        for x in [embedding_train, embedding_test, labels_train, labels_test]
    ):
        raise ValueError("All input arrays must be numpy arrays")

    # Ensure labels are 1D arrays if they have 1 column or are flattened to work well with indexing
    labels_train = (
        labels_train.ravel()
        if labels_train.ndim == 2 and labels_train.shape[1] == 1
        else labels_train
    )
    labels_test = (
        labels_test.ravel()
        if labels_test.ndim == 2 and labels_test.shape[1] == 1
        else labels_test
    )

    is_regression = (
        np.issubdtype(labels_train.dtype, np.floating)
        if not labels_describe_space
        else True
    )
    knn_class = KNeighborsRegressor if is_regression else KNeighborsClassifier

    # Outlier removal logic
    if test_outlier_removal:
        idx_remove = []
        if is_regression:
            mins = np.min(labels_train, axis=0)
            maxs = np.max(labels_train, axis=0)
            ranges = maxs - mins
            if labels_describe_space:
                area = np.prod(ranges) if hasattr(ranges, "__iter__") else ranges
                min_acceptable_value = np.sqrt(
                    area * regression_outlier_removal_threshold
                )
            else:
                min_acceptable_value = ranges * regression_outlier_removal_threshold

            for k, loc in enumerate(labels_test):
                diff = loc - labels_train
                dist = (
                    np.linalg.norm(loc - labels_train, axis=1)
                    if labels_describe_space
                    else np.abs(diff)
                )
                cl = np.min(dist)
                if cl > min_acceptable_value:
                    idx_remove.append(k)
        else:
            unique_classes, test_counts = np.unique(labels_test, return_counts=True)
            for cl, num_test_samples in zip(unique_classes, test_counts):
                num_train_samples = np.sum(labels_train == cl)
                if (
                    num_train_samples < min_train_class_samples
                    or num_test_samples < min_test_class_samples
                ):
                    idx_remove.extend(np.where(labels_test == cl)[0])

        if len(idx_remove) > 0:
            embedding_test = np.delete(embedding_test, idx_remove, axis=0)
            labels_test = np.delete(labels_test, idx_remove, axis=0)

    if n_neighbors is None:
        max_k = min(embedding_train.shape[0] - 1, 50)
        k_range = np.unique(
            np.logspace(0, np.log10(max_k), num=10, base=10).astype(int)
        )
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        k_scores = []

        for k in k_range:
            knn_model = knn_class(n_neighbors=k, metric=metric)
            fold_scores = []
            for train_idx, val_idx in kf.split(embedding_train):
                knn_model.fit(embedding_train[train_idx], labels_train[train_idx])
                preds = knn_model.predict(embedding_train[val_idx])
                score = (
                    r2_score(labels_train[val_idx], preds)
                    if is_regression
                    else accuracy_score(labels_train[val_idx], preds)
                )
                fold_scores.append(score)
            k_scores.append(np.mean(fold_scores))
        best_k = k_range[np.argmax(k_scores)]
    else:
        best_k = n_neighbors

    knn_model = knn_class(n_neighbors=best_k, metric=metric)

    cv_results = []
    if include_cv_stats:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        for train_idx, val_idx in kf.split(embedding_train):
            knn_model.fit(embedding_train[train_idx], labels_train[train_idx])
            preds = knn_model.predict(embedding_train[val_idx])
            cv_results.append({"true": labels_train[val_idx], "pred": preds})

    knn_model.fit(embedding_train, labels_train)
    test_predictions = knn_model.predict(embedding_test)

    if is_regression:
        results = compute_regression_metrics(
            labels_test,
            test_predictions,
            cv_results if include_cv_stats else None,
            labels_describe_space,
        )
    else:
        results = compute_classification_metrics(
            labels_test,
            test_predictions,
            cv_results if include_cv_stats else None,
            detailed_metrics,
        )

    results["k"] = best_k
    return results
