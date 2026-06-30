"""Coverage tests for neural_analysis.learning.classification.

Targets the uncovered lines/branches from the baseline 94.57% run:
  170->172, 172->175  metadata extraction short-circuits
  202                 n_samples == 1 -> autocorr zeros branch
  267->250, 269->250  FFT periodicity len checks (activity_proj len <= 10)
  291                 directional_tuning when weights.sum() == 0
  574->577            evaluate_classifier without confusion matrix
  709-710             fit_clusterer with gaussian_mixture
  746                 compare_classifiers with methods=None (default)
  809                 compare_clusterers with methods=None (default)
  831-832             compare_clusterers skips method when n_clusters is None
"""
from __future__ import annotations

import numpy as np
import pytest

from neural_analysis.learning.classification import (
    classify_cells,
    cluster_cells,
    compare_classifiers,
    compare_clusterers,
    evaluate_classifier,
    extract_cell_features,
    fit_clusterer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_data(
    n_samples: int = 40,
    n_cells: int = 6,
    n_classes: int = 2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Small, deterministic activity + label arrays."""
    rng = np.random.default_rng(seed)
    activity = rng.random((n_samples, n_cells)).astype(np.float64)
    labels = np.array([f"cls{i % n_classes}" for i in range(n_cells)])
    return activity, labels


# ---------------------------------------------------------------------------
# Lines 170->172  metadata has 'positions' but positions kwarg already set
# (branch NOT taken: positions is not None so assignment is skipped)
# ---------------------------------------------------------------------------

class TestExtractCellFeaturesMetadataBranches:
    """Cover the metadata short-circuit branches at lines 170-173."""

    def test_metadata_positions_already_supplied_kwarg(self) -> None:
        """positions kwarg supplied -> metadata['positions'] branch skipped (170->172 NOT taken)."""
        rng = np.random.default_rng(0)
        n = 30
        activity = rng.random((n, 4))
        meta_positions = rng.random((n, 2)) * 100   # metadata positions (should be ignored)
        kwarg_positions = rng.random((n, 2)) * 5    # explicit kwarg positions

        meta = {"positions": meta_positions}
        # pass positions explicitly so line 170 branch is NOT taken
        features = extract_cell_features(activity, metadata=meta, positions=kwarg_positions)
        assert features.shape == (4, features.shape[1])
        assert features.dtype == np.float64

    def test_metadata_head_directions_already_supplied_kwarg(self) -> None:
        """head_directions kwarg supplied -> metadata['head_directions'] branch skipped (172->175 NOT taken)."""
        rng = np.random.default_rng(0)
        n = 30
        activity = rng.random((n, 4))
        meta_hd = rng.random(n) * 2 * np.pi        # metadata head_dirs (should be ignored)
        kwarg_hd = rng.random(n) * 2 * np.pi       # explicit kwarg

        meta = {"head_directions": meta_hd}
        features = extract_cell_features(activity, metadata=meta, head_directions=kwarg_hd)
        assert features.shape[0] == 4
        assert features.dtype == np.float64

    def test_metadata_without_positions_key(self) -> None:
        """metadata present but has no 'positions' key -> positions stays None (170->172 branch taken but inner assignment NOT taken)."""
        rng = np.random.default_rng(0)
        n = 20
        activity = rng.random((n, 3))
        meta = {"cell_types": np.array(["A", "B", "C"])}  # no 'positions'
        features = extract_cell_features(activity, metadata=meta)
        assert features.shape[0] == 3


# ---------------------------------------------------------------------------
# Line 202  n_samples == 1 -> autocorr = np.zeros(n_cells)
# ---------------------------------------------------------------------------

class TestExtractCellFeaturesNSamplesOne:
    """Cover the else branch at line 202 (n_samples <= 1)."""

    def test_single_sample_no_autocorr(self) -> None:
        """With exactly 1 sample the else branch fills autocorr with zeros."""
        activity = np.array([[0.5, 0.3, 0.8]])  # shape (1, 3)
        features = extract_cell_features(activity)
        # Result should still be a valid float64 matrix
        assert features.shape[0] == 3
        assert features.dtype == np.float64
        # All features should be finite (no NaNs)
        assert np.all(np.isfinite(features))


# ---------------------------------------------------------------------------
# Lines 267->250, 269->250  FFT periodicity branch when activity_proj too short
# The branch at 267 guards `if len(activity_proj) > 10`.
# We need positions present but very few samples (<= 10).
# ---------------------------------------------------------------------------

class TestExtractCellFeaturesPeriodicityShortArray:
    """Cover the FFT short-circuit branches at lines 267-270."""

    def test_few_samples_with_positions_skips_fft(self) -> None:
        """With n_samples <= 10 and positions present, FFT branch is NOT entered."""
        rng = np.random.default_rng(0)
        n = 8   # <= 10: FFT block skipped entirely
        activity = rng.random((n, 3))
        positions = rng.random((n, 2))
        features = extract_cell_features(activity, positions=positions)
        assert features.shape[0] == 3
        assert np.all(np.isfinite(features))

    def test_11_samples_1d_positions_fft_single_freq(self) -> None:
        """With exactly 11 samples 1D positions, FFT is run (len > 10).
        When fft_vals has only 1 element the inner branch 269->250 is NOT taken."""
        rng = np.random.default_rng(1)
        n = 11
        activity = rng.random((n, 2))
        # 1D positions
        positions = rng.random(n)
        features = extract_cell_features(activity, positions=positions)
        assert features.shape[0] == 2
        assert np.all(np.isfinite(features))


# ---------------------------------------------------------------------------
# Line 291  directional_tuning when weights.sum() == 0
# This happens when a cell's entire activity column is exactly zero.
# ---------------------------------------------------------------------------

class TestExtractCellFeaturesDirectionalZeroWeights:
    """Cover line 291: directional_tuning[i] = 1.0 for all-zero activity cells."""

    def test_zero_activity_cell_directional_tuning(self) -> None:
        """A cell with all-zero firing should yield directional_tuning=1.0."""
        rng = np.random.default_rng(0)
        n = 30
        # 3 cells: first two have activity, last one is all zeros
        activity = np.zeros((n, 3))
        activity[:, 0] = rng.random(n)
        activity[:, 1] = rng.random(n)
        # activity[:, 2] stays zero

        head_directions = rng.random(n) * 2 * np.pi
        features = extract_cell_features(activity, head_directions=head_directions)

        assert features.shape[0] == 3
        assert np.all(np.isfinite(features))
        # The last feature column for directional tuning: zero-weight cell gets 1.0
        # directional_tuning is appended last; index is -1
        directional_col = features[:, -1]
        # zero-weight cell (index 2) should have tuning = 1.0
        assert directional_col[2] == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Line 574->577  evaluate_classifier with return_confusion_matrix=False
# ---------------------------------------------------------------------------

class TestEvaluateClassifierNoCM:
    """Cover the False branch of return_confusion_matrix at line 574."""

    def test_evaluate_classifier_no_confusion_matrix(self) -> None:
        """return_confusion_matrix=False means the cm key is absent."""
        y_true = np.array(["A", "B", "A", "B", "A", "B"])
        y_pred = np.array(["A", "B", "A", "A", "A", "B"])
        metrics = evaluate_classifier(y_true, y_pred, return_confusion_matrix=False)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "classification_report" in metrics
        assert "confusion_matrix" not in metrics

        assert metrics["accuracy"] == pytest.approx(5 / 6, rel=1e-6)


# ---------------------------------------------------------------------------
# Lines 709-710  fit_clusterer with gaussian_mixture uses .fit + .predict path
# (cluster_cells gaussian_mixture is already tested; fit_clusterer was not)
# ---------------------------------------------------------------------------

class TestFitClustererGaussianMixture:
    """Cover lines 709-710: fit_clusterer's gaussian_mixture branch."""

    def test_fit_clusterer_gaussian_mixture_returns_labels(self) -> None:
        """fit_clusterer with gaussian_mixture should return fitted model + labels."""
        rng = np.random.default_rng(0)
        n = 40
        features = rng.random((n, 4))

        clusterer, labels = fit_clusterer(
            features, method="gaussian_mixture", n_clusters=2, random_state=0
        )

        assert labels.shape == (n,)
        assert labels.dtype == np.int64
        assert set(labels).issubset({0, 1})
        # Verify the clusterer has been fitted (has means_ attribute)
        assert hasattr(clusterer, "means_")
        assert clusterer.means_.shape == (2, 4)


# ---------------------------------------------------------------------------
# Line 746  compare_classifiers with methods=None (default all-methods path)
# ---------------------------------------------------------------------------

class TestCompareClassifiersDefaultMethods:
    """Cover line 746: compare_classifiers default methods list."""

    def test_compare_classifiers_methods_none_uses_all(self) -> None:
        """Calling compare_classifiers without methods= triggers default list (line 746)."""
        rng = np.random.default_rng(0)
        # Use very small data so all methods are fast
        n_train, n_test, n_feat = 20, 10, 4
        train_features = rng.random((n_train, n_feat))
        test_features = rng.random((n_test, n_feat))
        train_labels = np.array(["A"] * 10 + ["B"] * 10)
        test_labels = np.array(["A"] * 5 + ["B"] * 5)

        results = compare_classifiers(
            train_features,
            train_labels,
            test_features,
            test_labels,
            methods=None,  # triggers line 746
            random_state=0,
        )

        # All 9 default methods should appear in results
        expected = {
            "random_forest", "svc", "svc_rbf", "logistic_regression",
            "knn", "naive_bayes", "mlp", "gradient_boosting", "adaboost",
        }
        assert set(results.keys()) == expected
        # Every successful method must have accuracy key
        for method, metrics in results.items():
            if "error" not in metrics:
                assert "accuracy" in metrics
                assert 0.0 <= metrics["accuracy"] <= 1.0


# ---------------------------------------------------------------------------
# Line 809  compare_clusterers with methods=None (default all-methods path)
# ---------------------------------------------------------------------------

class TestCompareClusterersDefaultMethods:
    """Cover line 809: compare_clusterers default methods list."""

    def test_compare_clusterers_methods_none_uses_all(self) -> None:
        """Calling compare_clusterers without methods= triggers default list (line 809)."""
        rng = np.random.default_rng(0)
        n, n_feat = 50, 4
        features = rng.random((n, n_feat))

        results = compare_clusterers(
            features,
            n_clusters=2,
            methods=None,   # triggers line 809
            random_state=0,
        )

        expected = {"kmeans", "dbscan", "agglomerative", "gaussian_mixture",
                    "spectral", "birch", "mean_shift"}
        assert set(results.keys()) == expected
        # Each result should have silhouette_score or an error key
        for method, metrics in results.items():
            assert "silhouette_score" in metrics or "error" in metrics


# ---------------------------------------------------------------------------
# Lines 831-832  compare_clusterers skips methods that need n_clusters when
#               n_clusters is None  (logger.warning + continue)
# ---------------------------------------------------------------------------

class TestCompareClusterersSkipsNCluster:
    """Cover lines 831-832: compare_clusterers skips k-requiring methods when n_clusters=None."""

    def test_compare_clusterers_skips_kmeans_without_n_clusters(self) -> None:
        """kmeans requires n_clusters; when it is None the method is skipped."""
        rng = np.random.default_rng(0)
        features = rng.random((30, 4))

        # Pass kmeans (needs n_clusters) together with dbscan (does not)
        results = compare_clusterers(
            features,
            n_clusters=None,    # will cause kmeans to be skipped at line 831-832
            methods=["kmeans", "dbscan"],
            random_state=0,
        )

        # dbscan should succeed
        assert "dbscan" in results
        # kmeans should be absent because it was skipped via 'continue'
        assert "kmeans" not in results

    def test_compare_clusterers_all_non_cluster_methods_run_without_n(self) -> None:
        """dbscan and mean_shift must not be skipped even when n_clusters is None."""
        rng = np.random.default_rng(0)
        features = rng.random((30, 4))

        results = compare_clusterers(
            features,
            n_clusters=None,
            methods=["dbscan", "mean_shift"],
            random_state=0,
        )

        assert "dbscan" in results
        assert "mean_shift" in results
        for metrics in results.values():
            # either has a silhouette score or failed with error
            assert "silhouette_score" in metrics or "error" in metrics
