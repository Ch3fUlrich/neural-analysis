"""Tests for distribution comparison functions."""

from __future__ import annotations

import numpy as np
import pytest

from neural_analysis.metrics import compare_distributions, compare_distribution_groups


class TestCompareDistributions:
    """Test suite for compare_distributions function."""

    def test_identical_distributions(self):
        """Test that identical distributions have zero distance (or 1.0 for cosine)."""
        p = np.random.randn(100, 3)
        # Wasserstein should be zero for identical distributions
        dist = compare_distributions(p, p, metric="wasserstein")
        assert dist == pytest.approx(0.0, abs=1e-6)

    def test_shifted_distributions_wasserstein(self):
        """Test Wasserstein distance increases with shift."""
        p1 = np.random.randn(100, 3)
        p2 = p1 + 2.0
        dist = compare_distributions(p1, p2, metric="wasserstein")
        # Distance should be positive and roughly proportional to shift
        assert dist > 1.0

    def test_kolmogorov_smirnov_metric(self):
        """Test K-S statistic on shifted distributions."""
        p1 = np.random.randn(100, 2)
        p2 = np.random.randn(100, 2) + 1.5
        dist = compare_distributions(p1, p2, metric="kolmogorov-smirnov")
        # K-S should be > 0 for shifted distributions
        assert 0.0 < dist <= 1.0

    def test_jensen_shannon_metric(self):
        """Test Jensen-Shannon divergence."""
        np.random.seed(42)
        p1 = np.random.randn(200, 2)
        p2 = np.random.randn(200, 2) + 1.0
        dist = compare_distributions(p1, p2, metric="jensen-shannon")
        # JS divergence should be in [0, 1] (in bits)
        assert 0.0 <= dist <= 1.0

    def test_euclidean_metric(self):
        """Test Euclidean distance between centroids."""
        p1 = np.random.randn(100, 3)
        p2 = p1 + np.array([3, 4, 0])
        dist = compare_distributions(p1, p2, metric="euclidean")
        # Distance between centroids should be ~5.0
        assert dist == pytest.approx(5.0, rel=0.2)

    def test_mahalanobis_metric(self):
        """Test Mahalanobis distance."""
        np.random.seed(42)
        p1 = np.random.randn(100, 3)
        p2 = np.random.randn(100, 3) + 2.0
        dist = compare_distributions(p1, p2, metric="mahalanobis")
        # Should be positive for different distributions
        assert dist > 0

    def test_cosine_metric(self):
        """Test cosine similarity."""
        p1 = np.random.randn(100, 3) + np.array([1, 0, 0])
        p2 = np.random.randn(100, 3) + np.array([2, 0, 0])
        sim = compare_distributions(p1, p2, metric="cosine")
        # Cosine similarity should be high for aligned distributions
        assert 0.8 <= sim <= 1.0

    def test_1d_distributions(self):
        """Test with 1D distributions."""
        p1 = np.random.randn(100)
        p2 = np.random.randn(100) + 1.0
        dist = compare_distributions(p1, p2, metric="wasserstein")
        assert dist > 0

    def test_empty_distribution_returns_nan(self):
        """Test that empty distributions return NaN."""
        p1 = np.array([]).reshape(0, 3)
        p2 = np.random.randn(100, 3)
        dist = compare_distributions(p1, p2, metric="wasserstein")
        assert np.isnan(dist)

    def test_dimension_mismatch_raises(self):
        """Test that mismatched dimensions raise error."""
        p1 = np.random.randn(100, 3)
        p2 = np.random.randn(100, 4)
        with pytest.raises(ValueError, match="Feature dimension mismatch"):
            compare_distributions(p1, p2, metric="wasserstein")

    def test_invalid_metric_raises(self):
        """Test that invalid metric raises error."""
        p1 = np.random.randn(100, 3)
        p2 = np.random.randn(100, 3)
        with pytest.raises(ValueError, match="Unknown metric"):
            compare_distributions(p1, p2, metric="invalid_metric")


class TestCompareDistributionGroups:
    """Test suite for compare_distribution_groups function."""

    def test_between_groups_basic(self):
        """Test between-group comparison."""
        groups = {
            "A": np.random.randn(50, 3),
            "B": np.random.randn(50, 3) + 1.0,
            "C": np.random.randn(50, 3) + 2.0,
        }
        result = compare_distribution_groups(
            groups, compare_type="between", metric="wasserstein"
        )
        
        # Check structure
        assert set(result.keys()) == {"A", "B", "C"}
        assert result["A"].shape == (3,)
        
        # Self-distance should be zero
        assert result["A"][0] == pytest.approx(0.0, abs=1e-6)
        
        # A to B should be less than A to C (B is closer)
        assert result["A"][1] < result["A"][2]

    def test_inside_groups(self):
        """Test within-group variability."""
        np.random.seed(42)
        groups = {
            "tight": np.random.randn(50, 3) * 0.1,  # Low variance
            "loose": np.random.randn(50, 3) * 2.0,  # High variance
        }
        result = compare_distribution_groups(
            groups, compare_type="inside", metric="euclidean"
        )
        
        # Check structure
        assert "mean" in result and "std" in result
        assert result["mean"].shape == (2,)
        
        # Loose group should have higher internal distance
        assert result["mean"][1] > result["mean"][0]

    def test_single_point_group(self):
        """Test with group containing only one point."""
        groups = {
            "single": np.array([[1, 2, 3]]),
            "normal": np.random.randn(50, 3),
        }
        result = compare_distribution_groups(
            groups, compare_type="inside", metric="wasserstein"
        )
        
        # Single-point group should have zero internal distance
        assert result["mean"][0] == pytest.approx(0.0)

    def test_different_metrics(self):
        """Test that different metrics work."""
        groups = {
            "A": np.random.randn(30, 2),
            "B": np.random.randn(30, 2) + 1.0,
        }
        
        for metric in ["wasserstein", "euclidean", "cosine"]:
            result = compare_distribution_groups(
                groups, compare_type="between", metric=metric
            )
            assert "A" in result and "B" in result

    def test_invalid_compare_type_raises(self):
        """Test that invalid compare_type raises error."""
        groups = {"A": np.random.randn(50, 3)}
        with pytest.raises(ValueError, match="Unknown compare_type"):
            compare_distribution_groups(
                groups, compare_type="invalid", metric="wasserstein"
            )

    def test_tuple_keys(self):
        """Test that tuple keys work as group identifiers."""
        groups = {
            (0, 0): np.random.randn(50, 3),
            (0, 1): np.random.randn(50, 3) + 1.0,
            (1, 0): np.random.randn(50, 3) + 2.0,
        }
        result = compare_distribution_groups(
            groups, compare_type="between", metric="euclidean"
        )
        
        assert (0, 0) in result
        assert result[(0, 0)].shape == (3,)
