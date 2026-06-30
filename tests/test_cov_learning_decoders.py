"""Tests for neural_analysis.learning.decoders to raise coverage to >= 95%."""

import numpy as np
import pytest

from neural_analysis.learning.decoders import decode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_regression_data(rng, n_train=300, n_test=60, n_features=4):
    """Return small floating-point arrays suitable for regression decoding."""
    X_train = rng.normal(0, 1, size=(n_train, n_features)).astype(np.float64)
    X_test = rng.normal(0, 1, size=(n_test, n_features)).astype(np.float64)
    y_train = rng.uniform(0, 1, size=n_train).astype(np.float64)
    y_test = rng.uniform(0, 1, size=n_test).astype(np.float64)
    return X_train, X_test, y_train, y_test


def _make_classification_data(rng, n_train=300, n_test=60, n_features=4, n_classes=3):
    """Return small integer-labelled arrays suitable for classification decoding."""
    X_train = rng.normal(0, 1, size=(n_train, n_features)).astype(np.float64)
    X_test = rng.normal(0, 1, size=(n_test, n_features)).astype(np.float64)
    # Each class gets equal representation so class-filters don't remove everything
    y_train = np.repeat(np.arange(n_classes), n_train // n_classes).astype(np.int64)
    y_test = np.repeat(np.arange(n_classes), n_test // n_classes).astype(np.int64)
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# 1. Input validation  (line 74)
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_raises_when_embedding_train_not_ndarray(self):
        rng = np.random.default_rng(0)
        X_train, X_test, y_train, y_test = _make_regression_data(rng)
        with pytest.raises(ValueError, match="numpy arrays"):
            decode(
                embedding_train=X_train.tolist(),   # list, not ndarray
                embedding_test=X_test,
                labels_train=y_train,
                labels_test=y_test,
                test_outlier_removal=False,
            )

    def test_raises_when_labels_test_not_ndarray(self):
        rng = np.random.default_rng(0)
        X_train, X_test, y_train, y_test = _make_regression_data(rng)
        with pytest.raises(ValueError, match="numpy arrays"):
            decode(
                embedding_train=X_train,
                embedding_test=X_test,
                labels_train=y_train,
                labels_test=y_test.tolist(),   # list, not ndarray
                test_outlier_removal=False,
            )


# ---------------------------------------------------------------------------
# 2. 2-D labels with one column are ravelled  (lines 77-86)
# ---------------------------------------------------------------------------

class TestLabelRavelling:
    def test_2d_column_labels_train_ravelled_regression(self):
        rng = np.random.default_rng(0)
        X_train, X_test, y_train, y_test = _make_regression_data(rng)
        # Reshape to (n, 1)
        y_train_2d = y_train.reshape(-1, 1)
        results = decode(
            embedding_train=X_train,
            embedding_test=X_test,
            labels_train=y_train_2d,
            labels_test=y_test,
            n_neighbors=3,
            test_outlier_removal=False,
        )
        assert "r2" in results
        assert "rmse" in results
        assert results["k"] == 3

    def test_2d_column_labels_test_ravelled_regression(self):
        rng = np.random.default_rng(0)
        X_train, X_test, y_train, y_test = _make_regression_data(rng)
        y_test_2d = y_test.reshape(-1, 1)
        results = decode(
            embedding_train=X_train,
            embedding_test=X_test,
            labels_train=y_train,
            labels_test=y_test_2d,
            n_neighbors=3,
            test_outlier_removal=False,
        )
        assert "r2" in results

    def test_2d_column_labels_classification(self):
        rng = np.random.default_rng(0)
        X_train, X_test, y_train, y_test = _make_classification_data(rng)
        y_train_2d = y_train.reshape(-1, 1)
        y_test_2d = y_test.reshape(-1, 1)
        results = decode(
            embedding_train=X_train,
            embedding_test=X_test,
            labels_train=y_train_2d,
            labels_test=y_test_2d,
            n_neighbors=3,
            test_outlier_removal=False,
        )
        assert "accuracy" in results


# ---------------------------------------------------------------------------
# 3. Regression outlier removal with test_outlier_removal=True  (lines 97-132)
# ---------------------------------------------------------------------------

class TestRegressionOutlierRemoval:
    def test_regression_outlier_removal_removes_far_points(self):
        """Points far outside the training range are removed; results still valid."""
        rng = np.random.default_rng(0)
        X_train, X_test, y_train, y_test = _make_regression_data(rng)

        # Add one test point whose label is far from all training labels
        extreme_y = np.array([1e6], dtype=np.float64)
        y_test_with_outlier = np.concatenate([y_test, extreme_y])
        X_test_with_outlier = np.vstack([
            X_test,
            rng.normal(0, 1, size=(1, X_train.shape[1])),
        ])

        results = decode(
            embedding_train=X_train,
            embedding_test=X_test_with_outlier,
            labels_train=y_train,
            labels_test=y_test_with_outlier,
            n_neighbors=3,
            test_outlier_removal=True,
        )
        # After outlier removal results should still be valid
        assert isinstance(results, dict)
        assert "r2" in results
        assert "rmse" in results

    def test_regression_outlier_removal_no_removal_when_all_close(self):
        """No outliers means the full test set is used (very high threshold)."""
        rng = np.random.default_rng(0)
        X_train, X_test, y_train, y_test = _make_regression_data(rng)
        # Use a very large threshold so min_acceptable_value is huge → no test
        # point is ever considered an outlier
        results = decode(
            embedding_train=X_train,
            embedding_test=X_test,
            labels_train=y_train,
            labels_test=y_test,
            n_neighbors=3,
            test_outlier_removal=True,
            regression_outlier_removal_threshold=1e10,  # enormous → never remove
        )
        assert "r2" in results

    def test_regression_no_outlier_removal_path(self):
        """Confirm test_outlier_removal=False skips the block entirely."""
        rng = np.random.default_rng(0)
        X_train, X_test, y_train, y_test = _make_regression_data(rng)
        results = decode(
            embedding_train=X_train,
            embedding_test=X_test,
            labels_train=y_train,
            labels_test=y_test,
            n_neighbors=3,
            test_outlier_removal=False,
        )
        assert "rmse" in results
        assert isinstance(results["rmse"], float)


# ---------------------------------------------------------------------------
# 4. labels_describe_space=True  (lines 102-106, 112-114)
# ---------------------------------------------------------------------------

class TestLabelsDescribeSpace:
    def test_2d_space_labels_regression(self):
        """labels_describe_space forces regression with 2-D spatial labels."""
        rng = np.random.default_rng(0)
        n_train, n_test = 200, 40
        X_train = rng.normal(0, 1, size=(n_train, 4))
        X_test = rng.normal(0, 1, size=(n_test, 4))
        # 2-D spatial labels (positions in 2D space)
        y_train = rng.uniform(0, 1, size=(n_train, 2))
        y_test = rng.uniform(0, 1, size=(n_test, 2))

        results = decode(
            embedding_train=X_train,
            embedding_test=X_test,
            labels_train=y_train,
            labels_test=y_test,
            labels_describe_space=True,
            n_neighbors=3,
            test_outlier_removal=True,
            regression_outlier_removal_threshold=0.004,
        )
        assert "rmse" in results
        assert "r2" in results
        assert results["k"] == 3

    def test_2d_space_labels_outlier_removal_all_within_range(self):
        """With spatial labels and outlier removal, nearby points are kept."""
        rng = np.random.default_rng(0)
        n_train, n_test = 200, 40
        X_train = rng.normal(0, 1, size=(n_train, 4))
        X_test = rng.normal(0, 1, size=(n_test, 4))
        y_train = rng.uniform(0.1, 0.9, size=(n_train, 2))
        # Test labels very close to training data → no outlier removal
        y_test = rng.uniform(0.1, 0.9, size=(n_test, 2))

        results = decode(
            embedding_train=X_train,
            embedding_test=X_test,
            labels_train=y_train,
            labels_test=y_test,
            labels_describe_space=True,
            n_neighbors=3,
            test_outlier_removal=True,
            regression_outlier_removal_threshold=0.004,
        )
        assert "rmse" in results
        assert "r2" in results


# ---------------------------------------------------------------------------
# 5. Classification outlier removal  (lines 120-132)
# ---------------------------------------------------------------------------

class TestClassificationOutlierRemoval:
    def test_classification_outlier_removal_removes_rare_class(self):
        """A class with too few training samples is removed from the test set."""
        rng = np.random.default_rng(0)
        n_features = 4
        # Build training set: class 0 has 250 samples (above threshold),
        # class 1 has only 5 samples (below min_train_class_samples=200)
        n_cls0_train, n_cls1_train = 250, 5
        X_cls0 = rng.normal(0, 1, size=(n_cls0_train, n_features))
        X_cls1 = rng.normal(5, 1, size=(n_cls1_train, n_features))
        X_train = np.vstack([X_cls0, X_cls1])
        y_train = np.array([0] * n_cls0_train + [1] * n_cls1_train, dtype=np.int64)

        # Test set: enough samples per class to pass test threshold (>= 30)
        n_cls0_test, n_cls1_test = 40, 35
        X_t0 = rng.normal(0, 1, size=(n_cls0_test, n_features))
        X_t1 = rng.normal(5, 1, size=(n_cls1_test, n_features))
        X_test = np.vstack([X_t0, X_t1])
        y_test = np.array([0] * n_cls0_test + [1] * n_cls1_test, dtype=np.int64)

        results = decode(
            embedding_train=X_train,
            embedding_test=X_test,
            labels_train=y_train,
            labels_test=y_test,
            test_outlier_removal=True,
            n_neighbors=3,
            min_train_class_samples=200,
            min_test_class_samples=30,
        )
        assert "accuracy" in results

    def test_classification_outlier_removal_keeps_all_when_above_threshold(self):
        """No classes removed when all pass both min-sample thresholds."""
        rng = np.random.default_rng(0)
        X_train, X_test, y_train, y_test = _make_classification_data(
            rng, n_train=300, n_test=90, n_classes=3
        )
        results = decode(
            embedding_train=X_train,
            embedding_test=X_test,
            labels_train=y_train,
            labels_test=y_test,
            test_outlier_removal=True,
            n_neighbors=3,
            min_train_class_samples=50,   # 300/3=100 per class -> above 50
            min_test_class_samples=10,    # 90/3=30 per class  -> above 10
        )
        assert "accuracy" in results
        assert isinstance(results["accuracy"], float)


# ---------------------------------------------------------------------------
# 6. n_neighbors provided (line 157) and auto k-selection (lines 134-155)
# ---------------------------------------------------------------------------

class TestNeighborSelection:
    def test_n_neighbors_provided_skips_cv_selection(self):
        """When n_neighbors is given, best_k equals it."""
        rng = np.random.default_rng(0)
        X_train, X_test, y_train, y_test = _make_regression_data(rng)
        results = decode(
            embedding_train=X_train,
            embedding_test=X_test,
            labels_train=y_train,
            labels_test=y_test,
            n_neighbors=7,
            test_outlier_removal=False,
        )
        assert results["k"] == 7

    def test_auto_k_selection_classification(self):
        """n_neighbors=None triggers cross-val k-selection for classification."""
        rng = np.random.default_rng(0)
        X_train, X_test, y_train, y_test = _make_classification_data(
            rng, n_train=300, n_test=60
        )
        results = decode(
            embedding_train=X_train,
            embedding_test=X_test,
            labels_train=y_train,
            labels_test=y_test,
            n_neighbors=None,
            test_outlier_removal=False,
            n_folds=3,
        )
        assert "k" in results
        assert results["k"] >= 1

    def test_auto_k_selection_regression(self):
        """n_neighbors=None triggers cross-val k-selection for regression."""
        rng = np.random.default_rng(0)
        X_train, X_test, y_train, y_test = _make_regression_data(rng)
        results = decode(
            embedding_train=X_train,
            embedding_test=X_test,
            labels_train=y_train,
            labels_test=y_test,
            n_neighbors=None,
            test_outlier_removal=False,
            n_folds=3,
        )
        assert "k" in results
        assert results["k"] >= 1


# ---------------------------------------------------------------------------
# 7. include_cv_stats=True  (lines 162-167)
# ---------------------------------------------------------------------------

class TestCvStats:
    def test_include_cv_stats_regression(self):
        """include_cv_stats adds cv_r2_mean, cv_rmse_mean to results."""
        rng = np.random.default_rng(0)
        X_train, X_test, y_train, y_test = _make_regression_data(rng)
        results = decode(
            embedding_train=X_train,
            embedding_test=X_test,
            labels_train=y_train,
            labels_test=y_test,
            n_neighbors=3,
            test_outlier_removal=False,
            include_cv_stats=True,
            n_folds=3,
        )
        assert "cv_r2_mean" in results
        assert "cv_rmse_mean" in results
        assert "cv_r2_std" in results
        assert isinstance(results["cv_r2_mean"], float)

    def test_include_cv_stats_classification(self):
        """include_cv_stats adds cv_accuracy_mean to classification results."""
        rng = np.random.default_rng(0)
        X_train, X_test, y_train, y_test = _make_classification_data(rng)
        results = decode(
            embedding_train=X_train,
            embedding_test=X_test,
            labels_train=y_train,
            labels_test=y_test,
            n_neighbors=3,
            test_outlier_removal=False,
            include_cv_stats=True,
            n_folds=3,
        )
        assert "cv_accuracy_mean" in results
        assert "cv_accuracy_std" in results
        assert 0.0 <= results["cv_accuracy_mean"] <= 1.0

    def test_no_cv_stats_by_default(self):
        """Without include_cv_stats, cv keys should not be present."""
        rng = np.random.default_rng(0)
        X_train, X_test, y_train, y_test = _make_regression_data(rng)
        results = decode(
            embedding_train=X_train,
            embedding_test=X_test,
            labels_train=y_train,
            labels_test=y_test,
            n_neighbors=3,
            test_outlier_removal=False,
            include_cv_stats=False,
        )
        assert "cv_r2_mean" not in results
        assert "cv_rmse_mean" not in results


# ---------------------------------------------------------------------------
# 8. detailed_metrics=True  (classification)
# ---------------------------------------------------------------------------

class TestDetailedMetrics:
    def test_detailed_metrics_binary_classification(self):
        """detailed_metrics adds f1, precision, recall for binary case."""
        rng = np.random.default_rng(0)
        X_train, X_test, y_train, y_test = _make_classification_data(
            rng, n_train=300, n_test=60, n_classes=2
        )
        results = decode(
            embedding_train=X_train,
            embedding_test=X_test,
            labels_train=y_train,
            labels_test=y_test,
            n_neighbors=3,
            test_outlier_removal=False,
            detailed_metrics=True,
        )
        assert "f1" in results
        assert "precision" in results
        assert "recall" in results
        assert 0.0 <= results["f1"] <= 1.0

    def test_detailed_metrics_multiclass(self):
        """detailed_metrics works for multi-class (weighted average)."""
        rng = np.random.default_rng(0)
        X_train, X_test, y_train, y_test = _make_classification_data(
            rng, n_train=300, n_test=60, n_classes=4
        )
        results = decode(
            embedding_train=X_train,
            embedding_test=X_test,
            labels_train=y_train,
            labels_test=y_test,
            n_neighbors=3,
            test_outlier_removal=False,
            detailed_metrics=True,
        )
        assert "f1" in results
        assert "precision" in results
        assert "recall" in results

    def test_no_detailed_metrics_by_default(self):
        """Without detailed_metrics, f1/precision/recall are absent."""
        rng = np.random.default_rng(0)
        X_train, X_test, y_train, y_test = _make_classification_data(rng)
        results = decode(
            embedding_train=X_train,
            embedding_test=X_test,
            labels_train=y_train,
            labels_test=y_test,
            n_neighbors=3,
            test_outlier_removal=False,
            detailed_metrics=False,
        )
        assert "f1" not in results


# ---------------------------------------------------------------------------
# 9. Combined: cv_stats + labels_describe_space
# ---------------------------------------------------------------------------

class TestCombinedOptions:
    def test_space_labels_with_cv_stats(self):
        """labels_describe_space=True + include_cv_stats=True covers spatial cv-rmse branch."""
        rng = np.random.default_rng(0)
        n_train, n_test = 200, 40
        X_train = rng.normal(0, 1, size=(n_train, 4))
        X_test = rng.normal(0, 1, size=(n_test, 4))
        y_train = rng.uniform(0, 1, size=(n_train, 2))
        y_test = rng.uniform(0, 1, size=(n_test, 2))

        results = decode(
            embedding_train=X_train,
            embedding_test=X_test,
            labels_train=y_train,
            labels_test=y_test,
            labels_describe_space=True,
            n_neighbors=3,
            test_outlier_removal=False,
            include_cv_stats=True,
            n_folds=3,
        )
        assert "cv_rmse_mean" in results
        assert "rmse" in results
        assert isinstance(results["cv_rmse_mean"], float)

    def test_classification_with_outlier_removal_and_cv_and_detailed(self):
        """Comprehensive classification path: outlier removal + cv + detailed metrics."""
        rng = np.random.default_rng(0)
        X_train, X_test, y_train, y_test = _make_classification_data(
            rng, n_train=300, n_test=90, n_classes=3
        )
        results = decode(
            embedding_train=X_train,
            embedding_test=X_test,
            labels_train=y_train,
            labels_test=y_test,
            test_outlier_removal=True,
            n_neighbors=5,
            min_train_class_samples=50,
            min_test_class_samples=10,
            include_cv_stats=True,
            detailed_metrics=True,
            n_folds=3,
        )
        assert "accuracy" in results
        assert "f1" in results
        assert "cv_accuracy_mean" in results

    def test_regression_with_outlier_removal_and_cv(self):
        """Regression + outlier removal + cv stats all together."""
        rng = np.random.default_rng(0)
        X_train, X_test, y_train, y_test = _make_regression_data(
            rng, n_train=300, n_test=60
        )
        results = decode(
            embedding_train=X_train,
            embedding_test=X_test,
            labels_train=y_train,
            labels_test=y_test,
            test_outlier_removal=True,
            n_neighbors=3,
            include_cv_stats=True,
            n_folds=3,
        )
        assert "r2" in results
        assert "cv_r2_mean" in results
