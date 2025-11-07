"""Tests for outlier detection functions."""

from __future__ import annotations

import numpy as np
import pytest

from neural_analysis.metrics import filter_outlier


class TestFilterOutlier:
    """Test suite for filter_outlier function."""

    def test_no_outliers_iqr(self):
        """Test that clean data passes through with IQR method."""
        np.random.seed(42)
        points = np.random.randn(100, 3)
        filtered = filter_outlier(points, method="iqr", threshold=3.0)
        # Most points should remain
        assert filtered.shape[0] >= 85

    def test_extreme_outliers_removed_iqr(self):
        """Test that extreme outliers are removed with IQR."""
        np.random.seed(42)
        clean = np.random.randn(95, 2)
        outliers = np.random.randn(5, 2) * 10  # Large outliers
        points = np.vstack([clean, outliers])
        filtered = filter_outlier(points, method="iqr", threshold=1.5)
        # Should remove most outliers
        assert filtered.shape[0] < points.shape[0]
        assert filtered.shape[0] >= 85

    def test_zscore_method(self):
        """Test Z-score outlier detection."""
        np.random.seed(42)
        clean = np.random.randn(95, 3)
        outliers = np.array([[10, 10, 10], [15, 15, 15]])
        points = np.vstack([clean, outliers])
        filtered = filter_outlier(points, method="zscore", threshold=3.0)
        # Should remove extreme outliers
        assert filtered.shape[0] < points.shape[0]

    def test_lof_method(self):
        """Test Local Outlier Factor method."""
        np.random.seed(42)
        clean = np.random.randn(90, 3)
        outliers = np.random.randn(10, 3) * 5
        points = np.vstack([clean, outliers])
        filtered = filter_outlier(points, method="lof", contamination=0.1)
        # Should remove approximately 10% of points
        assert 80 <= filtered.shape[0] <= 92

    def test_isolation_method(self):
        """Test Isolation Forest method."""
        np.random.seed(42)
        clean = np.random.randn(95, 3)
        outliers = np.random.randn(5, 3) * 8
        points = np.vstack([clean, outliers])
        filtered = filter_outlier(points, method="isolation", contamination=0.05)
        # Should remove approximately 5% of points
        assert 90 <= filtered.shape[0] <= 98

    def test_elliptic_method(self):
        """Test Elliptic Envelope (Mahalanobis) method."""
        np.random.seed(42)
        # Need enough samples for covariance estimation
        clean = np.random.randn(100, 3)
        outliers = np.random.randn(10, 3) * 6
        points = np.vstack([clean, outliers])
        filtered = filter_outlier(points, method="elliptic", contamination=0.1)
        # Should remove outliers
        assert filtered.shape[0] < points.shape[0]

    def test_return_mask(self):
        """Test that return_mask option works."""
        np.random.seed(42)
        points = np.vstack([np.random.randn(95, 3), np.random.randn(5, 3) * 10])
        filtered, mask = filter_outlier(
            points, method="lof", contamination=0.05, return_mask=True
        )

        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert mask.shape == (points.shape[0],)
        assert filtered.shape[0] == mask.sum()
        # Check that filtered points match mask
        np.testing.assert_array_equal(filtered, points[mask])

    def test_too_few_points_returns_all(self):
        """Test that < 10 points returns all points unchanged."""
        points = np.random.randn(5, 3)
        filtered = filter_outlier(points, method="lof")
        np.testing.assert_array_equal(filtered, points)

    def test_1d_data(self):
        """Test with 1D data."""
        np.random.seed(42)
        clean = np.random.randn(95, 1)
        outliers = np.array([[10], [15]])
        points = np.vstack([clean, outliers])
        filtered = filter_outlier(points, method="zscore", threshold=3.0)
        assert filtered.shape[0] < points.shape[0]

    def test_high_dimensional_data(self):
        """Test with high-dimensional data."""
        np.random.seed(42)
        points = np.random.randn(200, 50)
        outliers = np.random.randn(10, 50) * 5
        combined = np.vstack([points, outliers])
        filtered = filter_outlier(combined, method="lof", contamination=0.05)
        # Should detect and remove some outliers
        assert filtered.shape[0] < combined.shape[0]

    def test_constant_feature_zscore(self):
        """Test Z-score with constant feature (std=0)."""
        points = np.array([[1, 5], [1, 6], [1, 7], [1, 100]])
        # First column is constant; second has outlier
        filtered = filter_outlier(points, method="zscore", threshold=2.0)
        # Should keep first column intact, filter second
        assert filtered.shape[0] == 3

    def test_elliptic_insufficient_samples(self):
        """Test elliptic method with n <= d returns all points."""
        # 10 samples, 15 dimensions
        points = np.random.randn(10, 15)
        filtered = filter_outlier(points, method="elliptic", contamination=0.1)
        # Should return all due to insufficient samples for covariance
        assert filtered.shape[0] == points.shape[0]

    def test_invalid_method_raises(self):
        """Test that invalid method raises error."""
        points = np.random.randn(100, 3)
        with pytest.raises(ValueError, match="Unknown method"):
            filter_outlier(points, method="invalid_method")

    def test_preserves_shape(self):
        """Test that feature dimensionality is preserved."""
        np.random.seed(42)
        points = np.random.randn(100, 7)
        filtered = filter_outlier(points, method="lof", contamination=0.1)
        assert filtered.shape[1] == points.shape[1]
