from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)


def compute_regression_metrics(
    labels_test: npt.NDArray[np.floating],
    test_predictions: npt.NDArray[np.floating],
    cv_results: list[dict[str, npt.NDArray[Any]]] | None = None,
    labels_describe_space: bool = False,
) -> dict[str, Any]:
    """Compute regression performance metrics."""
    metrics = {}

    # Calculate R2 Score
    if len(np.unique(labels_test)) > 1:
        metrics["r2"] = r2_score(labels_test, test_predictions)
    else:
        metrics["r2"] = 0.0

    # Calculate RMSE
    if labels_describe_space:
        rmse = np.mean(np.linalg.norm(labels_test - test_predictions, axis=1))
    else:
        rmse = np.sqrt(mean_squared_error(labels_test, test_predictions))
    metrics["rmse"] = rmse

    if cv_results:
        cv_r2 = []
        cv_rmse = []
        for fold in cv_results:
            t = fold["true"]
            p = fold["pred"]
            if len(np.unique(t)) > 1:
                cv_r2.append(r2_score(t, p))
            else:
                cv_r2.append(0.0)
            if labels_describe_space:
                cv_rmse.append(np.mean(np.linalg.norm(t - p, axis=1)))
            else:
                cv_rmse.append(np.sqrt(mean_squared_error(t, p)))

        metrics["cv_r2_mean"] = np.mean(cv_r2)
        metrics["cv_r2_std"] = np.std(cv_r2)
        metrics["cv_rmse_mean"] = np.mean(cv_rmse)
        metrics["cv_rmse_std"] = np.std(cv_rmse)

    return metrics


def compute_classification_metrics(
    labels_test: npt.NDArray[Any],
    test_predictions: npt.NDArray[Any],
    cv_results: list[dict[str, npt.NDArray[Any]]] | None = None,
    detailed_metrics: bool = False,
) -> dict[str, Any]:
    """Compute classification performance metrics."""
    metrics = {}
    metrics["accuracy"] = accuracy_score(labels_test, test_predictions)

    if detailed_metrics:
        # Multi-class considerations
        avg_type = "weighted" if len(np.unique(labels_test)) > 2 else "binary"
        metrics["f1"] = f1_score(
            labels_test, test_predictions, average=avg_type, zero_division=0
        )
        metrics["precision"] = precision_score(
            labels_test, test_predictions, average=avg_type, zero_division=0
        )
        metrics["recall"] = recall_score(
            labels_test, test_predictions, average=avg_type, zero_division=0
        )

    if cv_results:
        cv_acc = []
        for fold in cv_results:
            cv_acc.append(accuracy_score(fold["true"], fold["pred"]))
        metrics["cv_accuracy_mean"] = np.mean(cv_acc)
        metrics["cv_accuracy_std"] = np.std(cv_acc)

    return metrics
