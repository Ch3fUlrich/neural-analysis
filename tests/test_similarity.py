"""
Tests for similarity metrics module.
"""

import numpy as np
import pytest

from neural_analysis.metrics import (
    correlation_matrix,
    cosine_similarity_matrix,
    angular_similarity_matrix,
)


class TestCorrelationMatrix:
    """Tests for correlation_matrix function."""

    def test_pearson_correlation(self):
        """Test Pearson correlation computation."""
        # Create correlated data
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 100)
        y = 0.8 * x + 0.2 * rng.normal(0, 1, 100)
        z = rng.normal(0, 1, 100)
        data = np.column_stack([x, y, z])
        
        corr = correlation_matrix(data, method='pearson')
        
        assert corr.shape == (3, 3)
        assert np.allclose(np.diag(corr), 1.0)  # Diagonal should be 1
        assert corr[0, 1] > 0.7  # x and y should be highly correlated
        assert np.allclose(corr, corr.T)  # Should be symmetric

    def test_spearman_correlation(self):
        """Test Spearman correlation computation."""
        data = np.random.randn(50, 4)
        
        corr = correlation_matrix(data, method='spearman')
        
        assert corr.shape == (4, 4)
        assert np.allclose(np.diag(corr), 1.0)
        assert np.allclose(corr, corr.T)

    def test_kendall_correlation(self):
        """Test Kendall tau correlation computation."""
        data = np.random.randn(30, 3)
        
        corr = correlation_matrix(data, method='kendall')
        
        assert corr.shape == (3, 3)
        assert np.allclose(np.diag(corr), 1.0)
        assert np.allclose(corr, corr.T)

    def test_correlation_bounds(self):
        """Test that correlation values are in [-1, 1]."""
        data = np.random.randn(100, 5)
        
        for method in ['pearson', 'spearman', 'kendall']:
            corr = correlation_matrix(data, method=method)
            assert np.all(corr >= -1.0) and np.all(corr <= 1.0)

    def test_perfect_correlation(self):
        """Test perfect correlation detection."""
        x = np.arange(100)
        y = 2 * x + 5  # Perfect linear relationship
        data = np.column_stack([x, y])
        
        corr = correlation_matrix(data, method='pearson')
        
        assert np.isclose(corr[0, 1], 1.0, atol=1e-10)

    def test_invalid_method_raises(self):
        """Test that invalid method raises error."""
        data = np.random.randn(50, 3)
        
        with pytest.raises(ValueError, match="Unknown method"):
            correlation_matrix(data, method='invalid')

    def test_non_2d_raises(self):
        """Test that non-2D data raises error."""
        data = np.random.rand(100)
        
        with pytest.raises(ValueError, match="must be 2D"):
            correlation_matrix(data)


class TestCosineSimilarityMatrix:
    """Tests for cosine_similarity_matrix function."""

    def test_cosine_similarity_basic(self):
        """Test basic cosine similarity computation."""
        # Create orthogonal vectors
        v1 = np.array([1, 0, 0])
        v2 = np.array([0, 1, 0])
        v3 = np.array([1, 1, 0]) / np.sqrt(2)  # 45 degrees from v1
        data = np.column_stack([v1, v2, v3])
        
        sim = cosine_similarity_matrix(data)
        
        assert sim.shape == (3, 3)
        assert np.allclose(np.diag(sim), 1.0)  # Self-similarity = 1
        assert np.isclose(sim[0, 1], 0.0, atol=1e-10)  # Orthogonal = 0
        assert np.isclose(sim[0, 2], 1/np.sqrt(2), atol=1e-6)  # 45 degrees

    def test_cosine_centered_equals_correlation(self):
        """Test that centered cosine similarity equals Pearson correlation."""
        data = np.random.randn(100, 4)
        
        sim_centered = cosine_similarity_matrix(data, centered=True)
        corr = correlation_matrix(data, method='pearson')
        
        assert np.allclose(sim_centered, corr, atol=1e-10)

    def test_cosine_symmetric(self):
        """Test that cosine similarity matrix is symmetric."""
        data = np.random.randn(50, 5)
        
        sim = cosine_similarity_matrix(data)
        
        assert np.allclose(sim, sim.T)

    def test_cosine_bounds(self):
        """Test that cosine similarity is in [-1, 1]."""
        data = np.random.randn(100, 6)
        
        sim = cosine_similarity_matrix(data)
        
        # Allow small numerical errors
        assert np.all(sim >= -1.0 - 1e-10) and np.all(sim <= 1.0 + 1e-10)

    def test_cosine_zero_vector_handling(self):
        """Test handling of zero vectors."""
        v1 = np.array([1, 2, 3])
        v2 = np.array([0, 0, 0])  # Zero vector
        data = np.column_stack([v1, v2])
        
        # Should not crash, zero vectors are handled
        sim = cosine_similarity_matrix(data)
        
        assert sim.shape == (2, 2)

    def test_non_2d_raises(self):
        """Test that non-2D data raises error."""
        data = np.random.rand(100)
        
        with pytest.raises(ValueError, match="must be 2D"):
            cosine_similarity_matrix(data)


class TestAngularSimilarityMatrix:
    """Tests for angular_similarity_matrix function."""

    def test_angular_similarity_basic(self):
        """Test basic angular similarity computation."""
        # Create vectors at known angles
        v1 = np.array([1, 0])
        v2 = np.array([1, 1]) / np.sqrt(2)  # 45 degrees
        v3 = np.array([0, 1])  # 90 degrees
        data = np.column_stack([v1, v2, v3])
        
        sim = angular_similarity_matrix(data)
        
        assert sim.shape == (3, 3)
        assert np.allclose(np.diag(sim), 1.0)  # Self-similarity = 1
        assert np.isclose(sim[0, 1], 1 - (np.pi/4) / np.pi, atol=1e-6)  # 45°
        assert np.isclose(sim[0, 2], 0.5, atol=1e-6)  # 90°

    def test_angular_bounds(self):
        """Test that angular similarity is in [0, 1]."""
        data = np.random.randn(100, 5)
        
        sim = angular_similarity_matrix(data)
        
        assert np.all(sim >= 0.0) and np.all(sim <= 1.0)

    def test_angular_symmetric(self):
        """Test that angular similarity matrix is symmetric."""
        data = np.random.randn(50, 4)
        
        sim = angular_similarity_matrix(data)
        
        assert np.allclose(sim, sim.T)

    def test_angular_opposite_vectors(self):
        """Test angular similarity for opposite vectors."""
        v1 = np.array([1, 0, 0])
        v2 = np.array([-1, 0, 0])  # Opposite direction
        data = np.column_stack([v1, v2])
        
        sim = angular_similarity_matrix(data)
        
        # Opposite vectors have angle = π, so similarity = 1 - π/π = 0
        assert np.isclose(sim[0, 1], 0.0, atol=1e-6)

    def test_non_2d_raises(self):
        """Test that non-2D data raises error."""
        data = np.random.rand(100)
        
        with pytest.raises(ValueError, match="must be 2D"):
            angular_similarity_matrix(data)


class TestIntegration:
    """Integration tests for similarity metrics."""

    def test_correlation_vs_cosine(self):
        """Test relationship between correlation and centered cosine."""
        data = np.random.randn(100, 4)
        
        corr = correlation_matrix(data, method='pearson')
        cosine_centered = cosine_similarity_matrix(data, centered=True)
        
        # They should be identical
        assert np.allclose(corr, cosine_centered)

    def test_all_methods_return_square_matrix(self):
        """Test that all methods return square symmetric matrices."""
        data = np.random.randn(50, 5)
        
        corr = correlation_matrix(data)
        cosine = cosine_similarity_matrix(data)
        angular = angular_similarity_matrix(data)
        
        for mat in [corr, cosine, angular]:
            assert mat.shape == (5, 5)
            assert np.allclose(mat, mat.T)

    def test_neural_data_simulation(self):
        """Test with simulated neural tuning curves."""
        # Simulate preferred directions (neurons tuned to angles)
        angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        directions = np.linspace(0, 2*np.pi, 100)
        
        # Create tuning curves: cosine tuning
        activity = np.array([
            np.cos(directions - angle) 
            for angle in angles
        ]).T  # shape: (100 directions, 8 neurons)
        
        # Neurons with similar preferred directions should correlate
        corr = correlation_matrix(activity, method='pearson')
        
        # Check that adjacent neurons are more correlated than opposite ones
        assert corr[0, 1] > corr[0, 4]  # Adjacent vs opposite
