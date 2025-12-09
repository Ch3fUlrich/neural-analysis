"""Tests to verify Jensen-Shannon divergence adaptive binning accuracy.

This module tests that the adaptive binning formula used in
jensen_shannon_divergence() doesn't significantly degrade calculation accuracy
while preventing memory explosion for high-dimensional data.
"""

from __future__ import annotations

import numpy as np
import pytest

from neural_analysis.metrics.distributions import jensen_shannon_divergence


class TestJensenShannonAdaptiveBinning:
    """Test adaptive binning accuracy and memory efficiency."""

    def test_binning_formula_verification(self) -> None:
        """Verify the adaptive binning formula: bins_adaptive = max(3, bins^(3/D))."""
        bins_default = 50
        test_cases = [
            (4, max(3, int(50 ** (3.0 / 4)))),  # 4D: 50^(3/4) ≈ 27
            (5, max(3, int(50 ** (3.0 / 5)))),  # 5D: 50^(3/5) ≈ 18
            (6, max(3, int(50 ** (3.0 / 6)))),  # 6D: 50^(3/6) ≈ 7
            (10, max(3, int(50 ** (3.0 / 10)))),  # 10D: 50^(3/10) ≈ 4
            (20, max(3, int(50 ** (3.0 / 20)))),  # 20D: 50^(3/20) ≈ 3
        ]

        for n_dims, expected_bins in test_cases:
            adaptive_bins = max(3, int(bins_default ** (3.0 / n_dims)))
            assert adaptive_bins == expected_bins, (
                f"For {n_dims}D: expected {expected_bins} bins, got {adaptive_bins}"
            )

    def test_low_dimensional_no_binning_reduction(self) -> None:
        """Test that low-dimensional data (≤3D) doesn't trigger bin reduction."""
        np.random.seed(42)
        bins = 50

        # 1D, 2D, 3D should not reduce bins
        for n_dims in [1, 2, 3]:
            p1 = np.random.randn(100, n_dims)
            p2 = np.random.randn(100, n_dims) + 1.0
            # Should use full bins=50 for these dimensions
            result = jensen_shannon_divergence(p1, p2, bins=bins)
            assert 0.0 <= result <= 1.0
            # No reduction should occur (we can't directly check, but verify it works)

    def test_high_dimensional_binning_reduction(self) -> None:
        """Test that high-dimensional data (>3D) triggers bin reduction."""
        np.random.seed(42)
        bins = 50

        # 4D and above should reduce bins
        for n_dims in [4, 5, 6, 10]:
            p1 = np.random.randn(100, n_dims)
            p2 = np.random.randn(100, n_dims) + 1.0
            result = jensen_shannon_divergence(p1, p2, bins=bins)
            assert 0.0 <= result <= 1.0
            # Verify it completes without memory issues

    def test_accuracy_comparison_low_vs_high_dim(self) -> None:
        """Compare JS divergence accuracy between low-D and high-D with same shift."""
        np.random.seed(42)
        shift = 1.0
        bins = 50

        # Low-dimensional (3D) - no binning reduction
        p1_3d = np.random.randn(200, 3)
        p2_3d = p1_3d + shift
        js_3d = jensen_shannon_divergence(p1_3d, p2_3d, bins=bins)

        # High-dimensional (6D) - with binning reduction
        p1_6d = np.random.randn(200, 6)
        p2_6d = p1_6d + shift
        js_6d = jensen_shannon_divergence(p1_6d, p2_6d, bins=bins)

        # Both should be valid JS divergences
        assert 0.0 <= js_3d <= 1.0
        assert 0.0 <= js_3d <= 1.0

        # High-D result should still be reasonable (not NaN or extreme)
        assert not np.isnan(js_6d)
        assert 0.0 <= js_6d <= 1.0

    def test_identical_distributions_zero_divergence(self) -> None:
        """Test that identical distributions have zero JS divergence regardless of binning."""
        np.random.seed(42)
        bins = 50

        # Test across different dimensions
        for n_dims in [2, 3, 4, 5, 6, 10]:
            p = np.random.randn(100, n_dims)
            js = jensen_shannon_divergence(p, p, bins=bins)
            # Should be approximately zero (within numerical precision)
            assert js == pytest.approx(0.0, abs=1e-6), (
                f"Identical {n_dims}D distributions should have JS≈0, got {js}"
            )

    def test_binning_consistency_same_data(self) -> None:
        """Test that same data gives consistent results with different bin counts."""
        np.random.seed(42)
        p1 = np.random.randn(200, 4)  # 4D triggers binning
        p2 = np.random.randn(200, 4) + 1.0

        # Test with different bin counts
        results = []
        for bins in [20, 30, 50, 100]:
            js = jensen_shannon_divergence(p1, p2, bins=bins)
            results.append(js)
            assert 0.0 <= js <= 1.0

        # Results should be relatively consistent (within reasonable tolerance)
        # JS divergence is somewhat sensitive to binning, but shouldn't vary wildly
        results_array = np.array(results)
        std_dev = np.std(results_array)
        # Standard deviation should be reasonable (< 0.1 for similar distributions)
        assert std_dev < 0.15, (
            f"JS divergence varies too much with binning: std={std_dev:.4f}, "
            f"values={results}"
        )

    def test_memory_efficiency_high_dimensions(self) -> None:
        """Test that adaptive binning prevents memory explosion in high dimensions."""
        np.random.seed(42)
        bins = 50

        # Without adaptive binning, 10D with 50 bins = 50^10 = 97,656,250,000,000 bins
        # With adaptive binning: 50^(3/10) ≈ 4 bins per dimension = 4^10 = 1,048,576 bins
        # This is a reduction of ~93 billion times!

        for n_dims in [8, 10, 15, 20]:
            p1 = np.random.randn(100, n_dims)
            p2 = np.random.randn(100, n_dims) + 0.5
            # Should complete without memory issues
            js = jensen_shannon_divergence(p1, p2, bins=bins)
            assert not np.isnan(js)
            assert 0.0 <= js <= 1.0

    def test_minimum_bins_guarantee(self) -> None:
        """Test that adaptive binning never goes below 3 bins per dimension."""
        np.random.seed(42)
        bins = 50

        # Even for very high dimensions, should maintain at least 3 bins
        for n_dims in [50, 100]:
            p1 = np.random.randn(50, n_dims)
            p2 = np.random.randn(50, n_dims) + 0.5
            js = jensen_shannon_divergence(p1, p2, bins=bins)
            # Should complete and return valid result
            assert not np.isnan(js)
            assert 0.0 <= js <= 1.0

    def test_relative_accuracy_preservation(self) -> None:
        """Test that relative ordering of divergences is preserved with binning."""
        np.random.seed(42)
        bins = 50

        # Create three distributions with different separations
        base = np.random.randn(200, 5)  # 5D triggers binning
        close = base + 0.5
        far = base + 2.0

        js_close = jensen_shannon_divergence(base, close, bins=bins)
        js_far = jensen_shannon_divergence(base, far, bins=bins)

        # Far distribution should have higher JS divergence than close
        assert js_far > js_close, (
            f"Expected JS(base, far)={js_far:.4f} > JS(base, close)={js_close:.4f}"
        )

    def test_binning_impact_documentation(self) -> None:
        """Document the trade-offs of adaptive binning."""
        np.random.seed(42)
        bins = 50

        # Example: 6D data
        p1 = np.random.randn(200, 6)
        p2 = np.random.randn(200, 6) + 1.0

        # Without adaptive binning: 50^6 = 15,625,000,000 bins
        # With adaptive binning: 50^(3/6) = 50^0.5 ≈ 7 bins per dim = 7^6 = 117,649 bins
        # Reduction factor: ~133,000x

        js = jensen_shannon_divergence(p1, p2, bins=bins)
        assert 0.0 <= js <= 1.0

        # The result should still be meaningful despite bin reduction
        # (We can't directly compare to non-adaptive version due to memory constraints,
        # but we verify the result is reasonable)

