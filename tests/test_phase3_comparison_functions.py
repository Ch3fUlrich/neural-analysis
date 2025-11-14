"""Comprehensive tests for Phase 3 pairwise comparison functions.

Tests compute_within_distances, compute_between_distances, and compute_all_pairs
with various metrics, edge cases, and validation scenarios.
"""

import numpy as np
import pytest

from neural_analysis.metrics import (
    DISTRIBUTION_METRICS,
    POINT_TO_POINT_METRICS,
    SCALAR_METRICS,
    SHAPE_METRICS,
    compute_all_pairs,
    compute_between_distances,
    compute_within_distances,
)


class TestComputeWithinDistances:
    """Test compute_within_distances function."""

    def test_basic_euclidean(self):
        """Test basic within-dataset distance with euclidean metric."""
        data = np.random.randn(50, 10)
        mean_dist = compute_within_distances(data, metric="euclidean")

        assert isinstance(mean_dist, float)
        assert mean_dist > 0

    def test_return_matrix(self):
        """Test that return_matrix=True returns symmetric matrix."""
        data = np.random.randn(30, 5)
        dist_matrix = compute_within_distances(
            data, metric="euclidean", return_matrix=True
        )

        assert isinstance(dist_matrix, np.ndarray)
        assert dist_matrix.shape == (30, 30)
        assert np.allclose(dist_matrix, dist_matrix.T)  # symmetric
        assert np.allclose(np.diag(dist_matrix), 0)  # zero diagonal

    def test_all_point_to_point_metrics(self):
        """Test all point-to-point metrics work."""
        data = np.random.randn(20, 8)

        for metric in POINT_TO_POINT_METRICS:
            if metric == "mahalanobis":
                # Mahalanobis needs covariance
                cov = np.cov(data.T)
                result = compute_within_distances(data, metric=metric, cov=cov)
            else:
                result = compute_within_distances(data, metric=metric)

            assert isinstance(result, float)
            # Cosine returns similarity (can be negative), others are distances
            if metric != "cosine":
                assert result >= 0, f"Metric {metric} returned negative distance"

    def test_reject_distribution_metrics(self):
        """Test that distribution metrics are rejected."""
        data = np.random.randn(30, 5)

        for metric in DISTRIBUTION_METRICS:
            with pytest.raises(ValueError, match="mode='within'"):
                compute_within_distances(data, metric=metric)

    def test_reject_shape_metrics(self):
        """Test that shape metrics are rejected."""
        data = np.random.randn(30, 5)

        for metric in SHAPE_METRICS:
            with pytest.raises(ValueError, match="mode='within'"):
                compute_within_distances(data, metric=metric)

    def test_edge_case_single_sample(self):
        """Test behavior with single sample."""
        data = np.random.randn(1, 5)
        result = compute_within_distances(data, metric="euclidean")

        assert isinstance(result, float)
        assert result == 0.0

    def test_edge_case_two_samples(self):
        """Test with minimal two samples."""
        data = np.array([[0, 0], [1, 1]])
        mean_dist = compute_within_distances(data, metric="euclidean")

        assert isinstance(mean_dist, float)
        assert mean_dist > 0

    def test_1d_data_handling(self):
        """Test that 1D data is reshaped correctly."""
        data = np.array([1, 2, 3, 4, 5])
        result = compute_within_distances(data, metric="euclidean")

        assert isinstance(result, float)
        assert result > 0

    def test_parallel_option(self):
        """Test that parallel option works."""
        data = np.random.randn(50, 10)

        result_parallel = compute_within_distances(
            data, metric="euclidean", parallel=True
        )
        result_serial = compute_within_distances(
            data, metric="euclidean", parallel=False
        )

        # Results should be very close (may differ slightly due to numerics)
        assert np.isclose(result_parallel, result_serial, rtol=1e-10)


class TestComputeBetweenDistances:
    """Test compute_between_distances function."""

    def test_basic_euclidean_scalar(self):
        """Test basic between-dataset distance with euclidean."""
        data1 = np.random.randn(50, 10)
        data2 = np.random.randn(60, 10)
        mean_dist = compute_between_distances(data1, data2, metric="euclidean")

        assert isinstance(mean_dist, float)
        assert mean_dist > 0

    def test_euclidean_matrix(self):
        """Test euclidean with return_matrix=True."""
        data1 = np.random.randn(30, 5)
        data2 = np.random.randn(40, 5)
        dist_matrix = compute_between_distances(
            data1, data2, metric="euclidean", return_matrix=True
        )

        assert isinstance(dist_matrix, np.ndarray)
        assert dist_matrix.shape == (30, 40)
        assert np.all(dist_matrix >= 0)

    def test_all_point_to_point_metrics_scalar(self):
        """Test all point-to-point metrics return scalar by default."""
        data1 = np.random.randn(20, 8)
        data2 = np.random.randn(25, 8)

        for metric in POINT_TO_POINT_METRICS:
            if metric == "mahalanobis":
                cov = np.cov(np.vstack([data1, data2]).T)
                result = compute_between_distances(data1, data2, metric=metric, cov=cov)
            else:
                result = compute_between_distances(data1, data2, metric=metric)

            assert isinstance(result, float)
            # Cosine returns similarity (can be negative), others are distances
            if metric != "cosine":
                assert result >= 0, f"Metric {metric} returned negative distance"

    def test_all_point_to_point_metrics_matrix(self):
        """Test all point-to-point metrics can return matrix."""
        data1 = np.random.randn(15, 6)
        data2 = np.random.randn(20, 6)

        for metric in POINT_TO_POINT_METRICS:
            if metric == "mahalanobis":
                cov = np.cov(np.vstack([data1, data2]).T)
                result = compute_between_distances(
                    data1, data2, metric=metric, return_matrix=True, cov=cov
                )
            else:
                result = compute_between_distances(
                    data1, data2, metric=metric, return_matrix=True
                )

            assert isinstance(result, np.ndarray)
            assert result.shape == (15, 20)

    def test_distribution_metrics_always_scalar(self):
        """Test distribution metrics always return scalar."""
        data1 = np.random.randn(30, 5)
        data2 = np.random.randn(40, 5)

        for metric in DISTRIBUTION_METRICS:
            # Without return_matrix
            result1 = compute_between_distances(data1, data2, metric=metric)
            assert isinstance(result1, float)
            assert result1 >= 0

            # With return_matrix=True (should still return scalar)
            result2 = compute_between_distances(
                data1, data2, metric=metric, return_matrix=True
            )
            assert isinstance(result2, float)
            assert result2 >= 0

    def test_shape_metrics_always_scalar(self):
        """Test shape metrics always return scalar."""
        # Shape metrics need same number of samples
        data1 = np.random.randn(30, 5)
        data2 = np.random.randn(30, 5)

        for metric in SHAPE_METRICS:
            result = compute_between_distances(data1, data2, metric=metric)
            assert isinstance(result, float)
            assert result >= 0

    def test_edge_case_different_feature_dims_fails(self):
        """Test that mismatched feature dimensions fail gracefully."""
        data1 = np.random.randn(20, 5)
        data2 = np.random.randn(20, 10)

        with pytest.raises((ValueError, AssertionError)):
            compute_between_distances(data1, data2, metric="euclidean")

    def test_1d_data_handling(self):
        """Test 1D data is reshaped correctly."""
        data1 = np.array([1, 2, 3])
        data2 = np.array([4, 5, 6])

        result = compute_between_distances(data1, data2, metric="euclidean")
        assert isinstance(result, float)

    def test_symmetry(self):
        """Test that distance is symmetric: d(A,B) ≈ d(B,A)."""
        data1 = np.random.randn(20, 5)
        data2 = np.random.randn(25, 5)

        # Point-to-point metrics may not be exactly symmetric
        # (mean of different matrices) but should be close
        dist_ab = compute_between_distances(data1, data2, metric="euclidean")
        dist_ba = compute_between_distances(data2, data1, metric="euclidean")

        assert np.isclose(dist_ab, dist_ba, rtol=0.1)

        # Distribution metrics should be exactly symmetric
        dist_ab_wass = compute_between_distances(data1, data2, metric="wasserstein")
        dist_ba_wass = compute_between_distances(data2, data1, metric="wasserstein")

        assert np.isclose(dist_ab_wass, dist_ba_wass)


class TestComputeAllPairs:
    """Test compute_all_pairs function."""

    def test_basic_wasserstein(self):
        """Test basic all-pairs comparison with wasserstein."""
        datasets = {
            "A": np.random.randn(30, 5),
            "B": np.random.randn(40, 5),
            "C": np.random.randn(35, 5),
        }

        results = compute_all_pairs(datasets, metric="wasserstein", show_progress=False)

        assert isinstance(results, dict)
        assert len(results) == 3
        for name in ["A", "B", "C"]:
            assert name in results
            assert len(results[name]) == 3
            for inner_name in ["A", "B", "C"]:
                assert inner_name in results[name]
                # Result can be float or np.nan (for invalid comparisons)
                assert isinstance(results[name][inner_name], (float, type(np.nan)))
                # If not nan, should be non-negative
                if not np.isnan(results[name][inner_name]):
                    assert results[name][inner_name] >= 0

    def test_all_scalar_metrics(self):
        """Test all scalar-returning metrics work."""
        # Use equal sample counts for procrustes (required)
        datasets = {
            "X": np.random.randn(25, 4),
            "Y": np.random.randn(25, 4),  # Same count for procrustes
        }

        for metric in SCALAR_METRICS:
            results = compute_all_pairs(datasets, metric=metric, show_progress=False)

            assert isinstance(results, dict)
            assert len(results) == 2
            assert isinstance(results["X"]["Y"], float)

    def test_reject_point_to_point_metrics(self):
        """Test that point-to-point metrics are rejected."""
        datasets = {"A": np.random.randn(20, 5), "B": np.random.randn(25, 5)}

        for metric in POINT_TO_POINT_METRICS:
            with pytest.raises(ValueError, match="mode='all-pairs'"):
                compute_all_pairs(datasets, metric=metric)

    def test_symmetry(self):
        """Test that result is symmetric."""
        datasets = {
            "A": np.random.randn(30, 5),
            "B": np.random.randn(40, 5),
            "C": np.random.randn(35, 5),
        }

        results = compute_all_pairs(datasets, metric="wasserstein", show_progress=False)

        # Check all pairs (handle np.nan values)
        if np.isnan(results["A"]["B"]) and np.isnan(results["B"]["A"]):
            pass  # Both are nan, symmetric
        else:
            assert results["A"]["B"] == results["B"]["A"]
        if np.isnan(results["A"]["C"]) and np.isnan(results["C"]["A"]):
            pass
        else:
            assert results["A"]["C"] == results["C"]["A"]
        if np.isnan(results["B"]["C"]) and np.isnan(results["C"]["B"]):
            pass
        else:
            assert results["B"]["C"] == results["C"]["B"]

    def test_diagonal_elements(self):
        """Test diagonal (self-comparisons) are reasonable."""
        datasets = {
            "A": np.random.randn(30, 5),
            "B": np.random.randn(40, 5),
        }

        results = compute_all_pairs(datasets, metric="wasserstein", show_progress=False)

        # Self-distances should be lower than between-distances (if both are finite)
        if np.isfinite(results["A"]["A"]) and np.isfinite(results["A"]["B"]):
            assert results["A"]["A"] < results["A"]["B"]
        if np.isfinite(results["B"]["B"]) and np.isfinite(results["A"]["B"]):
            assert results["B"]["B"] < results["A"]["B"]

    def test_single_dataset(self):
        """Test with single dataset."""
        datasets = {"X": np.random.randn(50, 5)}

        results = compute_all_pairs(datasets, metric="wasserstein", show_progress=False)

        assert len(results) == 1
        assert "X" in results
        assert len(results["X"]) == 1
        # Result can be float or np.nan
        assert isinstance(results["X"]["X"], (float, type(np.nan)))
        # If not nan, should be non-negative
        if not np.isnan(results["X"]["X"]):
            assert results["X"]["X"] >= 0

    def test_edge_case_small_datasets(self):
        """Test with very small datasets."""
        datasets = {
            "A": np.random.randn(3, 2),
            "B": np.random.randn(3, 2),
        }

        results = compute_all_pairs(datasets, metric="wasserstein", show_progress=False)

        assert isinstance(results, dict)
        assert len(results) == 2

    def test_shape_metrics_diagonal_handling(self):
        """Test that shape metrics handle diagonal correctly."""
        # Shape metrics split dataset for self-comparison
        datasets = {
            "A": np.random.randn(20, 5),
            "B": np.random.randn(20, 5),
        }

        results = compute_all_pairs(datasets, metric="procrustes", show_progress=False)

        # Diagonal should be non-negative
        assert results["A"]["A"] >= 0
        assert results["B"]["B"] >= 0

    def test_progress_bar_option(self):
        """Test that progress bar option works."""
        datasets = {
            "A": np.random.randn(20, 5),
            "B": np.random.randn(25, 5),
        }

        # Should work with progress=True (requires tqdm)
        try:
            results_progress = compute_all_pairs(
                datasets, metric="wasserstein", show_progress=True
            )
            assert isinstance(results_progress, dict)
        except ImportError:
            pass  # tqdm not available, skip

        # Should always work with progress=False
        results_no_progress = compute_all_pairs(
            datasets, metric="wasserstein", show_progress=False
        )
        assert isinstance(results_no_progress, dict)


class TestModeMetricValidation:
    """Test mode-metric compatibility validation."""

    def test_within_mode_accepts_point_to_point(self):
        """Test within mode accepts all point-to-point metrics."""
        data = np.random.randn(20, 5)

        for metric in POINT_TO_POINT_METRICS:
            if metric == "mahalanobis":
                cov = np.cov(data.T)
                result = compute_within_distances(data, metric=metric, cov=cov)
            else:
                result = compute_within_distances(data, metric=metric)
            assert isinstance(result, float)

    def test_within_mode_rejects_distribution(self):
        """Test within mode rejects distribution metrics."""
        data = np.random.randn(20, 5)

        for metric in DISTRIBUTION_METRICS:
            with pytest.raises(ValueError, match="mode='within'"):
                compute_within_distances(data, metric=metric)

    def test_within_mode_rejects_shape(self):
        """Test within mode rejects shape metrics."""
        data = np.random.randn(20, 5)

        for metric in SHAPE_METRICS:
            with pytest.raises(ValueError, match="mode='within'"):
                compute_within_distances(data, metric=metric)

    def test_between_mode_accepts_all(self):
        """Test between mode accepts all metrics."""
        data1 = np.random.randn(20, 5)
        data2 = np.random.randn(20, 5)

        # Point-to-point
        for metric in POINT_TO_POINT_METRICS:
            if metric == "mahalanobis":
                cov = np.cov(np.vstack([data1, data2]).T)
                result = compute_between_distances(data1, data2, metric=metric, cov=cov)
            else:
                result = compute_between_distances(data1, data2, metric=metric)
            assert isinstance(result, float)

        # Distribution
        for metric in DISTRIBUTION_METRICS:
            result = compute_between_distances(data1, data2, metric=metric)
            assert isinstance(result, float)

        # Shape
        for metric in SHAPE_METRICS:
            result = compute_between_distances(data1, data2, metric=metric)
            assert isinstance(result, float)

    def test_all_pairs_accepts_scalar_only(self):
        """Test all-pairs accepts only scalar metrics."""
        datasets = {"A": np.random.randn(20, 5), "B": np.random.randn(20, 5)}

        # Should accept distribution metrics
        for metric in DISTRIBUTION_METRICS:
            results = compute_all_pairs(datasets, metric=metric, show_progress=False)
            assert isinstance(results, dict)

        # Should accept shape metrics
        for metric in SHAPE_METRICS:
            results = compute_all_pairs(datasets, metric=metric, show_progress=False)
            assert isinstance(results, dict)

    def test_all_pairs_rejects_point_to_point(self):
        """Test all-pairs rejects point-to-point metrics."""
        datasets = {"A": np.random.randn(20, 5), "B": np.random.randn(20, 5)}

        for metric in POINT_TO_POINT_METRICS:
            with pytest.raises(ValueError, match="mode='all-pairs'"):
                compute_all_pairs(datasets, metric=metric)


class TestReturnTypes:
    """Test return type consistency."""

    def test_within_return_types(self):
        """Test compute_within_distances return types."""
        data = np.random.randn(30, 5)

        # Default: float
        result = compute_within_distances(data, metric="euclidean")
        assert isinstance(result, float)

        # With return_matrix=False: float
        result = compute_within_distances(data, metric="euclidean", return_matrix=False)
        assert isinstance(result, float)

        # With return_matrix=True: ndarray
        result = compute_within_distances(data, metric="euclidean", return_matrix=True)
        assert isinstance(result, np.ndarray)

    def test_between_point_to_point_return_types(self):
        """Test compute_between_distances return types for point-to-point."""
        data1 = np.random.randn(20, 5)
        data2 = np.random.randn(25, 5)

        # Default: float
        result = compute_between_distances(data1, data2, metric="euclidean")
        assert isinstance(result, float)

        # With return_matrix=True: ndarray
        result = compute_between_distances(
            data1, data2, metric="euclidean", return_matrix=True
        )
        assert isinstance(result, np.ndarray)

    def test_between_distribution_always_float(self):
        """Test distribution metrics always return float."""
        data1 = np.random.randn(20, 5)
        data2 = np.random.randn(25, 5)

        for metric in DISTRIBUTION_METRICS:
            result = compute_between_distances(data1, data2, metric=metric)
            assert isinstance(result, float)

            # Even with return_matrix=True
            result = compute_between_distances(
                data1, data2, metric=metric, return_matrix=True
            )
            assert isinstance(result, float)

    def test_between_shape_always_float(self):
        """Test shape metrics always return float."""
        data1 = np.random.randn(20, 5)
        data2 = np.random.randn(20, 5)

        for metric in SHAPE_METRICS:
            result = compute_between_distances(data1, data2, metric=metric)
            assert isinstance(result, float)

    def test_all_pairs_always_dict(self):
        """Test compute_all_pairs always returns dict."""
        # Use equal sample counts for procrustes (required)
        datasets = {"A": np.random.randn(20, 5), "B": np.random.randn(20, 5)}

        for metric in SCALAR_METRICS:
            result = compute_all_pairs(datasets, metric=metric, show_progress=False)
            assert isinstance(result, dict)
            assert all(isinstance(v, dict) for v in result.values())
            assert all(
                isinstance(vv, float) for v in result.values() for vv in v.values()
            )
