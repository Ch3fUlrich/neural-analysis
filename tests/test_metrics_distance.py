"""Tests for distance metrics."""

from __future__ import annotations

import numpy as np
import pytest

from neural_analysis.metrics import (
    cosine_similarity,
    euclidean_distance,
    mahalanobis_distance,
)


class TestEuclideanDistance:
    """Test suite for euclidean_distance function."""

    def test_1d_vectors(self):
        """Test distance between 1D vectors."""
        x = np.array([0, 0])
        y = np.array([3, 4])
        result = euclidean_distance(x, y)
        assert result == pytest.approx(5.0)

    def test_identical_vectors(self):
        """Test distance between identical vectors is zero."""
        x = np.array([1, 2, 3])
        result = euclidean_distance(x, x)
        assert result == pytest.approx(0.0)

    def test_2d_arrays_pairwise(self):
        """Test pairwise distance matrix for 2D arrays."""
        x_data = np.array([[0, 0], [1, 0], [0, 1]])
        y_data = np.array([[0, 0], [2, 0]])
        result = euclidean_distance(x_data, y_data)
        assert result.shape == (3, 2)
        # First point (0,0) to first point (0,0)
        assert result[0, 0] == pytest.approx(0.0)
        # Second point (1,0) to second point (2,0)
        assert result[1, 1] == pytest.approx(1.0)

    def test_negative_values(self):
        """Test with negative coordinates."""
        x = np.array([-1, -1])
        y = np.array([1, 1])
        result = euclidean_distance(x, y)
        expected = np.sqrt(8)
        assert result == pytest.approx(expected)

    def test_high_dimensional(self):
        """Test with high-dimensional vectors."""
        np.random.seed(42)
        x = np.random.randn(100)
        y = np.random.randn(100)
        result = euclidean_distance(x, y)
        expected = np.linalg.norm(x - y)
        assert result == pytest.approx(expected)


class TestMahalanobisDistance:
    """Test suite for mahalanobis_distance function."""

    def test_identity_covariance(self):
        """Test Mahalanobis equals Euclidean when cov is identity."""
        mean = np.array([0, 0])
        cov = np.eye(2)
        x = np.array([1, 1])
        result = mahalanobis_distance(x, mean, cov)
        expected = np.sqrt(2)
        assert result == pytest.approx(expected)

    def test_single_point(self):
        """Test with a single point."""
        mean = np.array([0, 0, 0])
        cov = np.eye(3)
        x = np.array([1, 0, 0])
        result = mahalanobis_distance(x, mean, cov)
        assert result == pytest.approx(1.0)

    def test_multiple_points(self):
        """Test with multiple points (2D input)."""
        mean = np.array([0, 0])
        cov = np.eye(2)
        x_data = np.array([[1, 0], [0, 1], [1, 1]])
        result = mahalanobis_distance(x_data, mean, cov)
        assert result.shape == (3,)
        assert result[0] == pytest.approx(1.0)
        assert result[1] == pytest.approx(1.0)
        assert result[2] == pytest.approx(np.sqrt(2))

    def test_with_inv_cov(self):
        """Test with inverse covariance directly."""
        mean = np.array([0, 0])
        inv_cov = np.eye(2)
        x = np.array([1, 1])
        result = mahalanobis_distance(x, mean, inv_cov=inv_cov)
        assert result == pytest.approx(np.sqrt(2))

    def test_correlated_covariance(self):
        """Test with correlated features."""
        mean = np.array([0, 0])
        cov = np.array([[1, 0.5], [0.5, 1]])
        x = np.array([1, 0])
        result = mahalanobis_distance(x, mean, cov)
        # Should differ from Euclidean due to correlation
        assert result > 0

    def test_missing_covariance_raises(self):
        """Test that missing both cov and inv_cov raises error."""
        mean = np.array([0, 0])
        x = np.array([1, 1])
        with pytest.raises(ValueError, match="Either cov or inv_cov"):
            mahalanobis_distance(x, mean)


class TestCosineSimilarity:
    """Test suite for cosine_similarity function."""

    def test_identical_vectors(self):
        """Test cosine similarity of identical vectors is 1."""
        v = np.array([1, 2, 3])
        result = cosine_similarity(v, v)
        assert result == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        """Test cosine similarity of orthogonal vectors is 0."""
        v1 = np.array([1, 0, 0])
        v2 = np.array([0, 1, 0])
        result = cosine_similarity(v1, v2)
        assert result == pytest.approx(0.0, abs=1e-7)

    def test_opposite_vectors(self):
        """Test cosine similarity of opposite vectors is -1."""
        v1 = np.array([1, 0])
        v2 = np.array([-1, 0])
        result = cosine_similarity(v1, v2)
        assert result == pytest.approx(-1.0)

    def test_45_degree_angle(self):
        """Test cosine similarity at 45 degrees."""
        v1 = np.array([1, 0])
        v2 = np.array([1, 1])
        result = cosine_similarity(v1, v2)
        expected = 1 / np.sqrt(2)
        assert result == pytest.approx(expected)

    def test_scaled_vectors(self):
        """Test that scaling doesn't affect similarity."""
        v1 = np.array([1, 2, 3])
        v2 = np.array([2, 4, 6])
        result = cosine_similarity(v1, v2)
        assert result == pytest.approx(1.0)

    def test_2d_input_flattened(self):
        """Test that 2D input is flattened correctly."""
        v1 = np.array([[1, 0]])
        v2 = np.array([[0, 1]])
        result = cosine_similarity(v1, v2)
        assert result == pytest.approx(0.0, abs=1e-7)
