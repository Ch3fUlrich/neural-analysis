"""Tests for neural_analysis.learning.decoding module.

Targets >= 95% line+branch coverage of src/neural_analysis/learning/decoding.py.
"""
from __future__ import annotations

import numpy as np
import pytest

from neural_analysis.learning.decoding import (
    compare_highd_lowd_decoding,
    cross_validated_knn_decoder,
    evaluate_decoder,
    knn_decoder,
    population_vector_decoder,
)

# ---------------------------------------------------------------------------
# Shared small arrays used across multiple tests
# ---------------------------------------------------------------------------
RNG = np.random.default_rng(0)

# 20 samples, 5 cells
_ACT = RNG.random((20, 5)).astype(np.float64)
# 2-D positions
_POS_2D = RNG.random((20, 2)).astype(np.float64)
# 1-D labels (scalar per sample)
_POS_1D = RNG.random(20).astype(np.float64)
# 5 cells × 2 dims
_FIELD_CENTERS_2D = RNG.random((5, 2)).astype(np.float64)
# 5 cells × 1 dim
_FIELD_CENTERS_1D = RNG.random((5, 1)).astype(np.float64)


# ===========================================================================
# population_vector_decoder
# ===========================================================================


class TestPopulationVectorDecoder:
    def test_weighted_average_shape_and_dtype(self):
        result = population_vector_decoder(_ACT, _FIELD_CENTERS_2D, method="weighted_average")
        assert result.shape == (20, 2)
        assert result.dtype == np.float64

    def test_weighted_average_values_are_finite(self):
        result = population_vector_decoder(_ACT, _FIELD_CENTERS_2D, method="weighted_average")
        assert np.all(np.isfinite(result))

    def test_weighted_average_1d_centers(self):
        result = population_vector_decoder(_ACT, _FIELD_CENTERS_1D, method="weighted_average")
        assert result.shape == (20, 1)

    def test_weighted_average_zero_activity_row(self):
        """When a row is all zeros the function falls back to field_centers.mean(axis=0)."""
        act = np.zeros((3, 5), dtype=np.float64)
        centers = RNG.random((5, 2))
        result = population_vector_decoder(act, centers, method="weighted_average")
        expected_fallback = centers.mean(axis=0)
        np.testing.assert_allclose(result[0], expected_fallback)
        np.testing.assert_allclose(result[1], expected_fallback)
        np.testing.assert_allclose(result[2], expected_fallback)

    def test_weighted_average_nonzero_row(self):
        """With a single active cell the decoded position equals that cell's center."""
        act = np.zeros((1, 3), dtype=np.float64)
        act[0, 1] = 1.0
        centers = np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 4.0]])
        result = population_vector_decoder(act, centers, method="weighted_average")
        np.testing.assert_allclose(result[0], [1.0, 2.0])

    def test_peak_method_shape(self):
        result = population_vector_decoder(_ACT, _FIELD_CENTERS_2D, method="peak")
        assert result.shape == (20, 2)

    def test_peak_method_selects_max_cell(self):
        """Peak decoder always returns the field center of the most active cell."""
        act = np.zeros((1, 5), dtype=np.float64)
        act[0, 3] = 10.0  # cell 3 is the peak
        centers = RNG.random((5, 2))
        result = population_vector_decoder(act, centers, method="peak")
        np.testing.assert_array_equal(result[0], centers[3])

    def test_peak_method_1d_centers(self):
        result = population_vector_decoder(_ACT, _FIELD_CENTERS_1D, method="peak")
        assert result.shape == (20, 1)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown method"):
            population_vector_decoder(_ACT, _FIELD_CENTERS_2D, method="bad_method")


# ===========================================================================
# knn_decoder
# ===========================================================================


class TestKnnDecoder:
    def test_basic_shape_2d_labels(self):
        train_act = RNG.random((15, 5))
        test_act = RNG.random((5, 5))
        train_labels = RNG.random((15, 2))
        result = knn_decoder(train_act, train_labels, test_act, k=3)
        assert result.shape == (5, 2)
        assert result.dtype == float

    def test_basic_shape_1d_labels(self):
        """1-D labels are reshaped internally; output must match (n_test, 1)."""
        train_act = RNG.random((15, 5))
        test_act = RNG.random((5, 5))
        train_labels = RNG.random(15)
        result = knn_decoder(train_act, train_labels, test_act, k=3)
        assert result.shape == (5, 1)

    def test_uniform_weights(self):
        train_act = RNG.random((15, 5))
        test_act = RNG.random((5, 5))
        train_labels = RNG.random((15, 2))
        result = knn_decoder(train_act, train_labels, test_act, k=3, weights="uniform")
        assert result.shape == (5, 2)

    def test_manhattan_metric(self):
        train_act = RNG.random((15, 5))
        test_act = RNG.random((5, 5))
        train_labels = RNG.random((15, 2))
        result = knn_decoder(train_act, train_labels, test_act, k=3, metric="manhattan")
        assert result.shape == (5, 2)

    def test_output_dtype_is_float(self):
        train_act = RNG.random((15, 5))
        test_act = RNG.random((5, 5))
        train_labels = RNG.random((15, 2))
        result = knn_decoder(train_act, train_labels, test_act)
        assert result.dtype == float


# ===========================================================================
# cross_validated_knn_decoder
# ===========================================================================


class TestCrossValidatedKnnDecoder:
    def test_returns_dict_with_expected_keys(self):
        metrics = cross_validated_knn_decoder(_ACT, _POS_2D, k=3, n_folds=3)
        expected_keys = {
            "r2_scores", "mse_scores", "euclidean_errors",
            "mean_r2", "std_r2", "mean_mse", "std_mse",
            "mean_error", "std_error", "k", "n_folds",
        }
        assert expected_keys.issubset(set(metrics.keys()))

    def test_r2_scores_length_matches_folds(self):
        metrics = cross_validated_knn_decoder(_ACT, _POS_2D, k=3, n_folds=4)
        assert len(metrics["r2_scores"]) == 4

    def test_k_and_n_folds_stored(self):
        metrics = cross_validated_knn_decoder(_ACT, _POS_2D, k=3, n_folds=4)
        assert metrics["k"] == 3
        assert metrics["n_folds"] == 4

    def test_2d_labels_uses_euclidean_error(self):
        """With 2-D labels the euclidean_errors are non-negative."""
        metrics = cross_validated_knn_decoder(_ACT, _POS_2D, k=3, n_folds=3)
        assert np.all(metrics["euclidean_errors"] >= 0)

    def test_1d_labels_uses_absolute_error(self):
        """With 1-D labels the code path uses np.abs instead of np.linalg.norm."""
        metrics = cross_validated_knn_decoder(_ACT, _POS_1D, k=3, n_folds=3)
        assert np.all(metrics["euclidean_errors"] >= 0)

    def test_return_predictions_false_no_key(self):
        metrics = cross_validated_knn_decoder(_ACT, _POS_2D, k=3, n_folds=3, return_predictions=False)
        assert "predictions" not in metrics

    def test_return_predictions_true_has_predictions(self):
        metrics = cross_validated_knn_decoder(
            _ACT, _POS_2D, k=3, n_folds=3, return_predictions=True
        )
        assert "predictions" in metrics
        preds = metrics["predictions"]
        assert isinstance(preds, list)
        assert len(preds) == 3
        # Each element has required keys
        for fold in preds:
            assert "test_idx" in fold
            assert "predictions" in fold
            assert "true_labels" in fold

    def test_mean_r2_is_float(self):
        metrics = cross_validated_knn_decoder(_ACT, _POS_2D, k=3, n_folds=3)
        assert isinstance(float(metrics["mean_r2"]), float)

    def test_uniform_weights(self):
        metrics = cross_validated_knn_decoder(_ACT, _POS_2D, k=3, n_folds=3, weights="uniform")
        assert "mean_r2" in metrics

    def test_custom_metric(self):
        metrics = cross_validated_knn_decoder(_ACT, _POS_2D, k=3, n_folds=3, metric="manhattan")
        assert "mean_mse" in metrics


# ===========================================================================
# compare_highd_lowd_decoding
# ===========================================================================


class TestCompareHighdLowdDecoding:
    def _setup(self):
        rng = np.random.default_rng(1)
        activity = rng.random((20, 8))
        embedding = rng.random((20, 2))
        labels = rng.random((20, 2))
        return activity, embedding, labels

    def test_returns_dict_with_required_keys(self):
        activity, embedding, labels = self._setup()
        result = compare_highd_lowd_decoding(activity, embedding, labels, k=3, n_folds=3)
        expected_keys = {
            "high_d", "low_d", "dimensionality_reduction",
            "n_cells", "n_components", "performance_ratio",
            "error_increase", "information_preserved",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_dimensionality_reduction_string(self):
        activity, embedding, labels = self._setup()
        result = compare_highd_lowd_decoding(activity, embedding, labels, k=3, n_folds=3)
        assert result["dimensionality_reduction"] == "8 → 2"

    def test_n_cells_and_n_components(self):
        activity, embedding, labels = self._setup()
        result = compare_highd_lowd_decoding(activity, embedding, labels, k=3, n_folds=3)
        assert result["n_cells"] == 8
        assert result["n_components"] == 2

    def test_high_d_and_low_d_are_metrics_dicts(self):
        activity, embedding, labels = self._setup()
        result = compare_highd_lowd_decoding(activity, embedding, labels, k=3, n_folds=3)
        assert "mean_r2" in result["high_d"]
        assert "mean_r2" in result["low_d"]

    def test_performance_ratio_is_finite(self):
        activity, embedding, labels = self._setup()
        result = compare_highd_lowd_decoding(activity, embedding, labels, k=3, n_folds=3)
        assert np.isfinite(result["performance_ratio"])

    def test_information_preserved_is_bool(self):
        activity, embedding, labels = self._setup()
        result = compare_highd_lowd_decoding(activity, embedding, labels, k=3, n_folds=3)
        assert isinstance(result["information_preserved"], (bool, np.bool_))

    def test_error_increase_is_numeric(self):
        activity, embedding, labels = self._setup()
        result = compare_highd_lowd_decoding(activity, embedding, labels, k=3, n_folds=3)
        assert np.isfinite(result["error_increase"])


# ===========================================================================
# evaluate_decoder
# ===========================================================================


class TestEvaluateDecoder:
    def _split(self):
        rng = np.random.default_rng(2)
        activity = rng.random((20, 5))
        labels_2d = rng.random((20, 2))
        labels_1d = rng.random(20)
        train_act, test_act = activity[:15], activity[15:]
        train_lab2, test_lab2 = labels_2d[:15], labels_2d[15:]
        train_lab1, test_lab1 = labels_1d[:15], labels_1d[15:]
        return train_act, test_act, train_lab2, test_lab2, train_lab1, test_lab1

    def test_knn_decoder_returns_expected_keys(self):
        train_act, test_act, train_lab2, test_lab2, *_ = self._split()
        metrics = evaluate_decoder(train_act, train_lab2, test_act, test_lab2, decoder="knn", k=3)
        assert set(metrics.keys()) == {"r2_score", "mse", "mean_error", "decoder"}

    def test_knn_decoder_label(self):
        train_act, test_act, train_lab2, test_lab2, *_ = self._split()
        metrics = evaluate_decoder(train_act, train_lab2, test_act, test_lab2, decoder="knn")
        assert metrics["decoder"] == "knn"

    def test_knn_decoder_2d_labels_euclidean_path(self):
        """2-D labels → euclidean error path (test_labels.shape[1] > 1)."""
        train_act, test_act, train_lab2, test_lab2, *_ = self._split()
        metrics = evaluate_decoder(train_act, train_lab2, test_act, test_lab2, decoder="knn", k=3)
        assert metrics["mean_error"] >= 0

    def test_knn_decoder_1d_labels_abs_path(self):
        """1-D labels → absolute error path."""
        train_act, test_act, _, _, train_lab1, test_lab1 = self._split()
        metrics = evaluate_decoder(train_act, train_lab1, test_act, test_lab1, decoder="knn", k=3)
        assert metrics["mean_error"] >= 0

    def test_knn_custom_params(self):
        train_act, test_act, train_lab2, test_lab2, *_ = self._split()
        metrics = evaluate_decoder(
            train_act, train_lab2, test_act, test_lab2,
            decoder="knn", k=2, weights="uniform", metric="manhattan"
        )
        assert np.isfinite(metrics["r2_score"])

    def test_population_vector_weighted_average(self):
        rng = np.random.default_rng(3)
        act = np.abs(rng.random((20, 5)))
        centers = rng.random((5, 2))
        test_act = np.abs(rng.random((5, 5)))
        test_labels = rng.random((5, 2))
        metrics = evaluate_decoder(
            act, centers, test_act, test_labels,
            decoder="population_vector",
            field_centers=centers,
            method="weighted_average",
        )
        assert metrics["decoder"] == "population_vector"
        assert "r2_score" in metrics
        assert "mse" in metrics

    def test_population_vector_peak_method(self):
        rng = np.random.default_rng(4)
        act = np.abs(rng.random((20, 5)))
        centers = rng.random((5, 2))
        test_act = np.abs(rng.random((5, 5)))
        test_labels = rng.random((5, 2))
        metrics = evaluate_decoder(
            act, centers, test_act, test_labels,
            decoder="population_vector",
            field_centers=centers,
            method="peak",
        )
        assert metrics["decoder"] == "population_vector"

    def test_population_vector_missing_field_centers_raises(self):
        train_act, test_act, train_lab2, test_lab2, *_ = self._split()
        with pytest.raises(ValueError, match="field_centers"):
            evaluate_decoder(
                train_act, train_lab2, test_act, test_lab2,
                decoder="population_vector",
                # no field_centers
            )

    def test_unknown_decoder_raises(self):
        train_act, test_act, train_lab2, test_lab2, *_ = self._split()
        with pytest.raises(ValueError, match="Unknown decoder"):
            evaluate_decoder(
                train_act, train_lab2, test_act, test_lab2,
                decoder="svm",
            )

    def test_1d_test_labels_reshaped(self):
        """When test_labels is 1-D it must be reshaped without error, using abs path."""
        rng = np.random.default_rng(5)
        train_act = rng.random((15, 5))
        test_act = rng.random((5, 5))
        train_labels = rng.random(15)
        test_labels = rng.random(5)
        metrics = evaluate_decoder(train_act, train_labels, test_act, test_labels, decoder="knn", k=3)
        assert metrics["mean_error"] >= 0

    def test_r2_score_is_numeric(self):
        train_act, test_act, train_lab2, test_lab2, *_ = self._split()
        metrics = evaluate_decoder(train_act, train_lab2, test_act, test_lab2, decoder="knn", k=3)
        assert isinstance(float(metrics["r2_score"]), float)

    def test_mse_is_non_negative(self):
        train_act, test_act, train_lab2, test_lab2, *_ = self._split()
        metrics = evaluate_decoder(train_act, train_lab2, test_act, test_lab2, decoder="knn", k=3)
        assert metrics["mse"] >= 0
