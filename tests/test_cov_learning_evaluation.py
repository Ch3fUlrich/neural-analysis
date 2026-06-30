"""Tests for neural_analysis.learning.evaluation — targets >= 95% combined coverage."""

from __future__ import annotations

import numpy as np
import pytest

from neural_analysis.learning.evaluation import (
    compute_classification_metrics,
    compute_regression_metrics,
)

# ---------------------------------------------------------------------------
# compute_regression_metrics
# ---------------------------------------------------------------------------


class TestComputeRegressionMetrics:
    """Tests for compute_regression_metrics."""

    # --- R2 branch: len(unique) > 1  (normal case) ---

    def test_r2_computed_when_multiple_unique_labels(self) -> None:
        rng = np.random.default_rng(0)
        y_true = rng.uniform(0, 1, size=50)
        y_pred = y_true + rng.normal(0, 0.05, size=50)
        result = compute_regression_metrics(y_true, y_pred)
        assert "r2" in result
        assert result["r2"] > 0.0  # good predictions → positive R2

    def test_r2_is_zero_when_all_labels_identical(self) -> None:
        # len(unique(labels_test)) == 1  → else branch sets r2 = 0.0
        y_true = np.ones(10)
        y_pred = np.ones(10) * 0.9
        result = compute_regression_metrics(y_true, y_pred)
        assert result["r2"] == 0.0

    # --- RMSE branch: labels_describe_space=False (default) ---

    def test_rmse_scalar_predictions(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 2.1, 2.9, 3.9])
        result = compute_regression_metrics(y_true, y_pred)
        expected_rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        assert np.isclose(result["rmse"], expected_rmse, rtol=1e-6)

    # --- RMSE branch: labels_describe_space=True ---

    def test_rmse_space_labels_2d(self) -> None:
        # labels_describe_space=True → norm over axis=1
        y_true = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        y_pred = np.array([[0.1, 0.0], [0.9, 0.0], [0.0, 0.9]])
        result = compute_regression_metrics(y_true, y_pred, labels_describe_space=True)
        expected_rmse = np.mean(np.linalg.norm(y_true - y_pred, axis=1))
        assert np.isclose(result["rmse"], expected_rmse, rtol=1e-6)

    # --- No cv_results: keys must be absent ---

    def test_no_cv_keys_when_cv_results_none(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        result = compute_regression_metrics(y_true, y_pred)
        for key in ("cv_r2_mean", "cv_r2_std", "cv_rmse_mean", "cv_rmse_std"):
            assert key not in result

    def test_no_cv_keys_when_cv_results_empty_list(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        result = compute_regression_metrics(y_true, y_pred, cv_results=[])
        for key in ("cv_r2_mean", "cv_r2_std", "cv_rmse_mean", "cv_rmse_std"):
            assert key not in result

    # --- cv_results provided ---

    def test_cv_results_scalar(self) -> None:
        rng = np.random.default_rng(0)
        y_true = rng.uniform(0, 1, size=20)
        y_pred = y_true + rng.normal(0, 0.1, size=20)
        cv_results = [
            {
                "true": rng.uniform(0, 1, size=10),
                "pred": rng.uniform(0, 1, size=10),
            },
            {
                "true": rng.uniform(0, 1, size=10),
                "pred": rng.uniform(0, 1, size=10),
            },
        ]
        result = compute_regression_metrics(y_true, y_pred, cv_results=cv_results)
        assert "cv_r2_mean" in result
        assert "cv_r2_std" in result
        assert "cv_rmse_mean" in result
        assert "cv_rmse_std" in result
        # Mean RMSE must be non-negative
        assert result["cv_rmse_mean"] >= 0.0

    def test_cv_results_with_space_labels(self) -> None:
        # labels_describe_space=True path inside the cv loop
        y_true = np.array([[0.0, 0.0], [1.0, 1.0]])
        y_pred = np.array([[0.1, 0.1], [0.9, 0.9]])
        fold1_t = np.array([[0.0, 0.0], [1.0, 1.0]])
        fold1_p = np.array([[0.05, 0.05], [0.95, 0.95]])
        fold2_t = np.array([[0.5, 0.5], [1.5, 1.5]])
        fold2_p = np.array([[0.6, 0.5], [1.4, 1.5]])
        cv_results = [
            {"true": fold1_t, "pred": fold1_p},
            {"true": fold2_t, "pred": fold2_p},
        ]
        result = compute_regression_metrics(
            y_true, y_pred, cv_results=cv_results, labels_describe_space=True
        )
        assert "cv_rmse_mean" in result
        assert result["cv_rmse_mean"] >= 0.0

    def test_cv_results_fold_with_single_unique_label(self) -> None:
        # Inside cv loop: len(unique(t)) == 1 → cv_r2.append(0.0)
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        fold_single = {"true": np.ones(5), "pred": np.ones(5) * 0.9}
        fold_normal = {
            "true": np.array([0.0, 1.0, 2.0]),
            "pred": np.array([0.1, 0.9, 2.1]),
        }
        result = compute_regression_metrics(
            y_true, y_pred, cv_results=[fold_single, fold_normal]
        )
        # first fold contributes r2=0.0; mean should be <=1
        assert result["cv_r2_mean"] <= 1.0

    def test_return_type_is_dict(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        result = compute_regression_metrics(y_true, y_pred)
        assert isinstance(result, dict)

    def test_perfect_predictions_r2_is_one(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = y_true.copy()
        result = compute_regression_metrics(y_true, y_pred)
        assert np.isclose(result["r2"], 1.0)
        assert np.isclose(result["rmse"], 0.0)


# ---------------------------------------------------------------------------
# compute_classification_metrics
# ---------------------------------------------------------------------------


class TestComputeClassificationMetrics:
    """Tests for compute_classification_metrics."""

    # --- basic accuracy ---

    def test_accuracy_perfect(self) -> None:
        labels = np.array([0, 1, 2, 0, 1])
        preds = np.array([0, 1, 2, 0, 1])
        result = compute_classification_metrics(labels, preds)
        assert result["accuracy"] == pytest.approx(1.0)

    def test_accuracy_partial(self) -> None:
        labels = np.array([0, 1, 0, 1])
        preds = np.array([0, 0, 0, 1])
        result = compute_classification_metrics(labels, preds)
        assert result["accuracy"] == pytest.approx(0.75)

    # --- detailed_metrics=False (default): f1/precision/recall absent ---

    def test_no_detailed_keys_by_default(self) -> None:
        labels = np.array([0, 1, 0, 1])
        preds = np.array([0, 1, 0, 1])
        result = compute_classification_metrics(labels, preds)
        for key in ("f1", "precision", "recall"):
            assert key not in result

    # --- detailed_metrics=True: binary case (avg_type = "binary") ---

    def test_detailed_binary(self) -> None:
        labels = np.array([0, 1, 1, 0, 1])
        preds = np.array([0, 1, 0, 0, 1])
        result = compute_classification_metrics(labels, preds, detailed_metrics=True)
        assert "f1" in result
        assert "precision" in result
        assert "recall" in result
        # binary F1: TP=2, FP=0, FN=1 → precision=1.0, recall=2/3
        assert result["precision"] == pytest.approx(1.0)
        assert result["recall"] == pytest.approx(2 / 3, rel=1e-6)

    # --- detailed_metrics=True: multi-class case (avg_type = "weighted") ---

    def test_detailed_multiclass(self) -> None:
        rng = np.random.default_rng(0)
        labels = rng.integers(0, 3, size=30)
        preds = rng.integers(0, 3, size=30)
        result = compute_classification_metrics(labels, preds, detailed_metrics=True)
        assert "f1" in result
        assert "precision" in result
        assert "recall" in result
        # Values must be in [0, 1]
        for key in ("f1", "precision", "recall"):
            assert 0.0 <= result[key] <= 1.0

    # --- cv_results provided ---

    def test_cv_results_classification(self) -> None:
        labels = np.array([0, 1, 0, 1, 0])
        preds = np.array([0, 1, 0, 1, 0])
        cv_results = [
            {"true": np.array([0, 1, 0]), "pred": np.array([0, 1, 1])},
            {"true": np.array([1, 0, 1]), "pred": np.array([1, 0, 1])},
        ]
        result = compute_classification_metrics(labels, preds, cv_results=cv_results)
        assert "cv_accuracy_mean" in result
        assert "cv_accuracy_std" in result
        # fold 0 accuracy = 2/3, fold 1 accuracy = 1.0 → mean = 5/6
        expected_mean = np.mean([2 / 3, 1.0])
        assert result["cv_accuracy_mean"] == pytest.approx(expected_mean, rel=1e-6)

    def test_no_cv_keys_when_cv_results_none(self) -> None:
        labels = np.array([0, 1, 0])
        preds = np.array([0, 1, 0])
        result = compute_classification_metrics(labels, preds)
        assert "cv_accuracy_mean" not in result
        assert "cv_accuracy_std" not in result

    def test_no_cv_keys_when_cv_results_empty(self) -> None:
        labels = np.array([0, 1, 0])
        preds = np.array([0, 1, 0])
        result = compute_classification_metrics(labels, preds, cv_results=[])
        assert "cv_accuracy_mean" not in result
        assert "cv_accuracy_std" not in result

    # --- combined: detailed + cv ---

    def test_detailed_and_cv_together(self) -> None:
        rng = np.random.default_rng(1)
        labels = rng.integers(0, 2, size=20)
        preds = rng.integers(0, 2, size=20)
        cv_results = [
            {
                "true": rng.integers(0, 2, size=10),
                "pred": rng.integers(0, 2, size=10),
            }
        ]
        result = compute_classification_metrics(
            labels, preds, cv_results=cv_results, detailed_metrics=True
        )
        for key in ("accuracy", "f1", "precision", "recall", "cv_accuracy_mean", "cv_accuracy_std"):
            assert key in result

    def test_return_type_is_dict(self) -> None:
        labels = np.array([0, 1])
        preds = np.array([0, 1])
        result = compute_classification_metrics(labels, preds)
        assert isinstance(result, dict)
