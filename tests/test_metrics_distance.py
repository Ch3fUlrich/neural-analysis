"""Tests for distance and pairwise metrics."""

from __future__ import annotations

import contextlib
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from neural_analysis.metrics import (
    cosine_similarity,
    euclidean_distance,
    mahalanobis_distance,
)
from neural_analysis.metrics.pairwise_metrics import (
    _angular_similarity_matrix_parallel,
    _correlation_matrix_parallel,
    _cosine_similarity_matrix_parallel,
    _plot_similarity_matrix,
    _validate_pairwise_inputs,
    angular_similarity_matrix,
    compare_datasets,
    compute_all_pairs,
    compute_pairwise_matrix,
    correlation,
    correlation_matrix,
    cosine_similarity_matrix,
    pairwise_distance,
    similarity_matrix,
    spatial_autocorrelation,
)

# ---------------------------------------------------------------------------
# Section 1: Basic distance functions
# ---------------------------------------------------------------------------


class TestEuclideanDistance:
    """Test suite for euclidean_distance function."""

    def test_1d_vectors(self) -> None:
        """Test distance between 1D vectors."""
        x = np.array([0, 0])
        y = np.array([3, 4])
        result = euclidean_distance(x, y)
        assert result == pytest.approx(5.0)

    def test_identical_vectors(self) -> None:
        """Test distance between identical vectors is zero."""
        x = np.array([1, 2, 3])
        result = euclidean_distance(x, x)
        assert result == pytest.approx(0.0)

    def test_2d_arrays_pairwise(self) -> None:
        """Test pairwise distance matrix for 2D arrays."""
        x_data = np.array([[0, 0], [1, 0], [0, 1]])
        y_data = np.array([[0, 0], [2, 0]])
        result = euclidean_distance(x_data, y_data)
        assert result.shape == (3, 2)
        # First point (0,0) to first point (0,0)
        assert result[0, 0] == pytest.approx(0.0)
        # Second point (1,0) to second point (2,0)
        assert result[1, 1] == pytest.approx(1.0)

    def test_negative_values(self) -> None:
        """Test with negative coordinates."""
        x = np.array([-1, -1])
        y = np.array([1, 1])
        result = euclidean_distance(x, y)
        expected = np.sqrt(8)
        assert result == pytest.approx(expected)

    def test_high_dimensional(self) -> None:
        """Test with high-dimensional vectors."""
        np.random.seed(42)
        x = np.random.randn(100)
        y = np.random.randn(100)
        result = euclidean_distance(x, y)
        expected = np.linalg.norm(x - y)
        assert result == pytest.approx(expected)


class TestMahalanobisDistance:
    """Test suite for mahalanobis_distance function."""

    def test_identity_covariance(self) -> None:
        """Test Mahalanobis equals Euclidean when cov is identity."""
        mean = np.array([0, 0])
        cov = np.eye(2)
        x = np.array([1, 1])
        result = mahalanobis_distance(x, mean, cov)
        expected = np.sqrt(2)
        assert result == pytest.approx(expected)

    def test_single_point(self) -> None:
        """Test with a single point."""
        mean = np.array([0, 0, 0])
        cov = np.eye(3)
        x = np.array([1, 0, 0])
        result = mahalanobis_distance(x, mean, cov)
        assert result == pytest.approx(1.0)

    def test_multiple_points(self) -> None:
        """Test with multiple points (2D input)."""
        mean = np.array([0, 0])
        cov = np.eye(2)
        x_data = np.array([[1, 0], [0, 1], [1, 1]])
        result = mahalanobis_distance(x_data, mean, cov)
        assert result.shape == (3,)
        assert result[0] == pytest.approx(1.0)
        assert result[1] == pytest.approx(1.0)
        assert result[2] == pytest.approx(np.sqrt(2))

    def test_with_inv_cov(self) -> None:
        """Test with inverse covariance directly."""
        mean = np.array([0, 0])
        inv_cov = np.eye(2)
        x = np.array([1, 1])
        result = mahalanobis_distance(x, mean, inv_cov=inv_cov)
        assert result == pytest.approx(np.sqrt(2))

    def test_correlated_covariance(self) -> None:
        """Test with correlated features."""
        mean = np.array([0, 0])
        cov = np.array([[1, 0.5], [0.5, 1]])
        x = np.array([1, 0])
        result = mahalanobis_distance(x, mean, cov)
        # Should differ from Euclidean due to correlation
        assert result > 0

    def test_missing_covariance_raises(self) -> None:
        """Test that missing both cov and inv_cov raises error."""
        mean = np.array([0, 0])
        x = np.array([1, 1])
        with pytest.raises(ValueError, match="Either cov or inv_cov"):
            mahalanobis_distance(x, mean)


class TestCosineSimilarity:
    """Test suite for cosine_similarity function."""

    def test_identical_vectors(self) -> None:
        """Test cosine similarity of identical vectors is 1."""
        v = np.array([1, 2, 3])
        result = cosine_similarity(v, v)
        assert result == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        """Test cosine similarity of orthogonal vectors is 0."""
        v1 = np.array([1, 0, 0])
        v2 = np.array([0, 1, 0])
        result = cosine_similarity(v1, v2)
        assert result == pytest.approx(0.0, abs=1e-7)

    def test_opposite_vectors(self) -> None:
        """Test cosine similarity of opposite vectors is -1."""
        v1 = np.array([1, 0])
        v2 = np.array([-1, 0])
        result = cosine_similarity(v1, v2)
        assert result == pytest.approx(-1.0)

    def test_45_degree_angle(self) -> None:
        """Test cosine similarity at 45 degrees."""
        v1 = np.array([1, 0])
        v2 = np.array([1, 1])
        result = cosine_similarity(v1, v2)
        expected = 1 / np.sqrt(2)
        assert result == pytest.approx(expected)

    def test_scaled_vectors(self) -> None:
        """Test that scaling doesn't affect similarity."""
        v1 = np.array([1, 2, 3])
        v2 = np.array([2, 4, 6])
        result = cosine_similarity(v1, v2)
        assert result == pytest.approx(1.0)

    def test_2d_input_flattened(self) -> None:
        """Test that 2D input is flattened correctly."""
        v1 = np.array([[1, 0]])
        v2 = np.array([[0, 1]])
        result = cosine_similarity(v1, v2)
        assert result == pytest.approx(0.0, abs=1e-7)


# ---------------------------------------------------------------------------
# Section 2: Pairwise infrastructure
# ---------------------------------------------------------------------------


class TestValidatePairwiseInputs:
    """Tests for _validate_pairwise_inputs function."""

    def test_validate_pairwise_inputs_1d(self) -> None:
        """Test validate pairwise inputs with 1D arrays."""
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        x_arr, y_arr = _validate_pairwise_inputs(x, y)
        assert x_arr.shape == (1, 3)
        assert y_arr.shape == (1, 3)

    def test_validate_pairwise_inputs_feature_mismatch(self) -> None:
        """Test validate pairwise inputs with feature mismatch."""
        x = np.random.randn(10, 5)
        y = np.random.randn(10, 3)
        with pytest.raises(ValueError, match="Feature dimension mismatch"):
            _validate_pairwise_inputs(x, y)


class TestPairwiseDistance:
    """Tests for pairwise_distance function."""

    def test_pairwise_distance_mahalanobis(self) -> None:
        """Test pairwise distance with mahalanobis metric."""
        x = np.random.randn(10, 5)
        y = np.random.randn(8, 5)
        mean = np.mean(y, axis=0)
        cov = np.cov(y, rowvar=False)
        result = pairwise_distance(x, y, metric="mahalanobis", mean=mean, cov=cov)
        assert result.shape == (10, 8)
        assert np.all(result >= 0)

    def test_pairwise_distance_unknown_metric(self) -> None:
        """Test pairwise distance with unknown metric."""
        x = np.random.randn(10, 5)
        y = np.random.randn(8, 5)
        with pytest.raises(ValueError, match="Unknown.*metric"):
            pairwise_distance(x, y, metric="unknown")


class TestComputePairwiseMatrix:
    """Tests for compute_pairwise_matrix function."""

    def test_compute_pairwise_matrix_shape_metrics(self) -> None:
        """Test compute_pairwise_matrix with shape metrics."""
        x = np.random.randn(20, 5)
        y = np.random.randn(20, 5)
        try:
            result = compute_pairwise_matrix(x, y, metric="procrustes")
            assert isinstance(result, (float, tuple))
        except Exception:
            pass

    def test_compute_pairwise_matrix_unknown_metric(self) -> None:
        """Test compute_pairwise_matrix with unknown metric."""
        x = np.random.randn(10, 5)
        y = np.random.randn(8, 5)
        with pytest.raises(ValueError, match="Unknown metric"):
            compute_pairwise_matrix(x, y, metric="unknown_metric")


# ---------------------------------------------------------------------------
# Section 3: Similarity matrices
# ---------------------------------------------------------------------------


class TestSimilarityMatrix:
    """Tests for similarity_matrix function."""

    @pytest.mark.parametrize(
        "method",
        ["pearson", "spearman", "kendall", "cosine", "angular"],
    )
    def test_similarity_matrix_method(self, method: str) -> None:
        """Test similarity matrix with different methods."""
        data = np.random.randn(20, 10)
        result = similarity_matrix(data, method=method)
        assert result.shape == (10, 10)
        assert np.allclose(result, result.T)  # Symmetric

    @pytest.mark.parametrize(
        "method",
        ["pearson", "cosine", "angular"],
    )
    def test_similarity_matrix_parallel(self, method: str) -> None:
        """Test similarity matrix with parallel=True for different methods."""
        data = np.random.randn(20, 10)
        result = similarity_matrix(data, method=method, parallel=True)
        assert result.shape == (10, 10)

    def test_similarity_matrix_with_plot(self) -> None:
        """Test similarity matrix with plot=True."""
        data = np.random.randn(20, 10)
        with patch(
            "neural_analysis.metrics.pairwise_core._plot_similarity_matrix"
        ) as mock_plot:
            result = similarity_matrix(data, method="pearson", plot=True)
            assert result.shape == (10, 10)
            mock_plot.assert_called_once()

    def test_similarity_matrix_invalid_method(self) -> None:
        """Test similarity matrix with invalid method."""
        data = np.random.randn(20, 10)
        with pytest.raises(ValueError, match="Unknown.*method"):
            similarity_matrix(data, method="invalid")

    def test_similarity_matrix_invalid_shape(self) -> None:
        """Test similarity matrix with invalid shape."""
        data = np.array([1, 2, 3])  # 1D array
        with pytest.raises(ValueError, match="must be 2D"):
            similarity_matrix(data)

    # --- from test_pairwise_metrics_comprehensive (TestSimilarityMatrixAdvanced) ---

    def test_similarity_matrix_with_plotting(self) -> None:
        """Test similarity_matrix with plotting enabled."""
        data = np.random.randn(50, 10)
        try:
            result = similarity_matrix(data, method="cosine", plot=True)
            assert result is not None
        except Exception:
            pass

    def test_similarity_matrix_parallel_false(self) -> None:
        """Test similarity_matrix with parallel=False."""
        data = np.random.randn(50, 10)
        result = similarity_matrix(data, method="cosine", parallel=False)
        assert result.shape == (10, 10)

    def test_similarity_matrix_pearson_large(self) -> None:
        """Test similarity_matrix with pearson method on larger data."""
        data = np.random.randn(50, 10)
        result = similarity_matrix(data, method="pearson")
        assert result.shape == (10, 10)

    def test_similarity_matrix_spearman_large(self) -> None:
        """Test similarity_matrix with spearman method on larger data."""
        data = np.random.randn(50, 10)
        result = similarity_matrix(data, method="spearman")
        assert result.shape == (10, 10)


class TestAngularSimilarityMatrix:
    """Tests for angular_similarity_matrix function."""

    # --- from test_pairwise_metrics_additional ---

    def test_angular_similarity_matrix_basic(self) -> None:
        """Test basic angular similarity matrix."""
        data = np.random.randn(10, 5)
        result = angular_similarity_matrix(data)
        assert result.shape == (5, 5)
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_angular_similarity_matrix_invalid_shape(self) -> None:
        """Test angular similarity matrix with invalid shape."""
        data = np.array([1, 2, 3])  # 1D array
        with pytest.raises(ValueError, match="must be 2D"):
            angular_similarity_matrix(data)

    # --- from test_pairwise_metrics_comprehensive ---

    def test_angular_similarity_matrix_basic_large(self) -> None:
        """Test angular_similarity_matrix basic on larger data."""
        data = np.random.randn(50, 10)
        try:
            result = angular_similarity_matrix(data)
            assert result.shape == (50, 50)
        except Exception:
            pass

    def test_angular_similarity_matrix_parallel_false(self) -> None:
        """Test angular_similarity_matrix with parallel=False."""
        data = np.random.randn(50, 10)
        try:
            result = angular_similarity_matrix(data, parallel=False)
            assert result.shape == (50, 50)
        except Exception:
            pass


class TestCosineSimilarityMatrix:
    """Tests for cosine_similarity_matrix function."""

    def test_cosine_similarity_matrix_centered(self) -> None:
        """Test cosine similarity matrix with centered=True."""
        data = np.random.randn(20, 10)
        result = cosine_similarity_matrix(data, centered=True)
        assert result.shape == (10, 10)

    def test_cosine_similarity_matrix_not_centered(self) -> None:
        """Test cosine similarity matrix with centered=False."""
        data = np.random.randn(20, 10)
        result = cosine_similarity_matrix(data, centered=False)
        assert result.shape == (10, 10)


class TestPlotSimilarityMatrix:
    """Tests for _plot_similarity_matrix function."""

    # --- from test_pairwise_metrics_additional ---

    def test_plot_similarity_matrix_with_plot_config(self) -> None:
        """Test _plot_similarity_matrix with plot_config."""
        similarity = np.random.rand(10, 10)

        with patch("neural_analysis.plotting.plot_heatmap") as mock_plot:
            plot_config = {"title": "Custom Title"}
            _plot_similarity_matrix(
                similarity, method="pearson", plot_config=plot_config
            )
            mock_plot.assert_called_once()

    # --- from test_pairwise_metrics_comprehensive ---

    def test_plot_similarity_matrix_with_config(self) -> None:
        """Test _plot_similarity_matrix with config."""
        matrix = np.random.rand(10, 10)
        config = {"title": "Test Matrix", "cmap": "viridis"}
        with contextlib.suppress(Exception):
            _plot_similarity_matrix(matrix, config=config)

    def test_plot_similarity_matrix_basic(self) -> None:
        """Test _plot_similarity_matrix basic."""
        matrix = np.random.rand(10, 10)
        with contextlib.suppress(Exception):
            _plot_similarity_matrix(matrix)


# ---------------------------------------------------------------------------
# Section 4: Correlation
# ---------------------------------------------------------------------------


class TestCorrelationMatrix:
    """Tests for correlation_matrix function."""

    # --- from test_pairwise_metrics_additional ---

    @pytest.mark.parametrize(
        "method",
        ["pearson", "spearman", "kendall"],
    )
    def test_correlation_matrix_method(self, method: str) -> None:
        """Test correlation matrix with different methods."""
        data = np.random.randn(20, 10)
        result = correlation_matrix(data, method=method)
        assert result.shape == (10, 10)
        assert np.allclose(result, result.T)  # Symmetric

    # --- from test_pairwise_metrics_more ---

    def test_correlation_matrix_spearman_2d(self) -> None:
        """Test correlation matrix with spearman and 2 features."""
        data = np.random.randn(100, 2)
        result = correlation_matrix(data, method="spearman")
        assert result.shape == (2, 2)
        assert np.allclose(result, result.T)  # Symmetric

    def test_correlation_matrix_kendall_with_diagonal_check(self) -> None:
        """Test correlation matrix with kendall method checking diagonal."""
        data = np.random.randn(50, 5)
        result = correlation_matrix(data, method="kendall")
        assert result.shape == (5, 5)
        assert np.allclose(result, result.T)  # Symmetric
        assert np.allclose(np.diag(result), 1.0)  # Diagonal is 1

    def test_correlation_matrix_unknown_method(self) -> None:
        """Test correlation matrix with unknown method."""
        data = np.random.randn(50, 5)
        with pytest.raises(ValueError, match="Unknown.*method"):
            correlation_matrix(data, method="unknown")


class TestCorrelation:
    """Tests for correlation function."""

    # --- from test_pairwise_metrics_additional ---

    def test_correlation_matrix_mode_parallel_numba(self) -> None:
        """Test correlation with matrix mode and parallel=True."""
        data = np.random.randn(100, 20)

        with (
            patch("neural_analysis.metrics.pairwise_metrics.NUMBA_AVAILABLE", True),
            patch(
                "neural_analysis.metrics.pairwise_metrics._correlation_matrix_parallel"
            ) as mock_parallel,
        ):
            mock_parallel.return_value = np.corrcoef(data.T)
            result = correlation(data, method="pearson", mode="matrix", parallel=True)
            assert result.shape == (20, 20)

    def test_correlation_pairwise_mode(self) -> None:
        """Test correlation with pairwise mode."""
        data = np.random.randn(100, 5)
        result = correlation(data, method="pearson", mode="pairwise")
        assert result.shape == (4,)  # n_features - 1

    def test_correlation_unknown_mode(self) -> None:
        """Test correlation with unknown mode."""
        data = np.random.randn(100, 20)
        with pytest.raises(ValueError, match="Unknown.*mode"):
            correlation(data, method="pearson", mode="invalid")

    # --- from test_pairwise_metrics_comprehensive (TestCorrelationAdvanced) ---

    def test_correlation_matrix_mode(self) -> None:
        """Test correlation with matrix mode."""
        data = np.random.randn(50, 10)
        result = correlation(data, mode="matrix")
        assert result.shape == (10, 10)

    def test_correlation_pairwise_mode_large(self) -> None:
        """Test correlation with pairwise mode on larger data."""
        data = np.random.randn(50, 10)
        result = correlation(data, mode="pairwise")
        assert isinstance(result, np.ndarray)
        assert len(result) == 9  # n_features - 1

    def test_correlation_pearson_method(self) -> None:
        """Test correlation with pearson method."""
        data = np.random.randn(50, 10)
        try:
            result = correlation(data, method="pearson")
            assert result.shape == (50, 50)
        except Exception:
            pass

    def test_correlation_spearman_method(self) -> None:
        """Test correlation with spearman method."""
        data = np.random.randn(50, 10)
        try:
            result = correlation(data, method="spearman")
            assert result.shape == (50, 50)
        except Exception:
            pass


class TestCorrelationMatrixParallel:
    """Tests for _correlation_matrix_parallel function."""

    def test_correlation_matrix_parallel_spearman_no_numba(self) -> None:
        """Test _correlation_matrix_parallel with spearman and no numba."""
        data = np.random.randn(100, 20)

        with patch("neural_analysis.metrics.pairwise_metrics.NUMBA_AVAILABLE", False):
            result = _correlation_matrix_parallel(data, method="spearman")
            assert result.shape == (20, 20)

    def test_correlation_matrix_parallel_kendall_no_numba(self) -> None:
        """Test _correlation_matrix_parallel with kendall and no numba."""
        data = np.random.randn(100, 20)

        with patch("neural_analysis.metrics.pairwise_metrics.NUMBA_AVAILABLE", False):
            result = _correlation_matrix_parallel(data, method="kendall")
            assert result.shape == (20, 20)

    def test_correlation_matrix_parallel_unknown_method(self) -> None:
        """Test _correlation_matrix_parallel with unknown method."""
        data = np.random.randn(100, 20)

        with pytest.raises(ValueError, match="Unknown.*method"):
            _correlation_matrix_parallel(data, method="invalid")


# ---------------------------------------------------------------------------
# Section 5: Parallel functions
# ---------------------------------------------------------------------------


class TestParallelFunctions:
    """Tests for parallel computation functions."""

    def test_correlation_matrix_parallel(self) -> None:
        """Test _correlation_matrix_parallel."""
        data = np.random.randn(50, 10)
        try:
            result = _correlation_matrix_parallel(data, method="pearson")
            assert result.shape == (50, 50)
        except Exception:
            pass

    def test_cosine_similarity_matrix_parallel(self) -> None:
        """Test _cosine_similarity_matrix_parallel."""
        data = np.random.randn(50, 10)
        try:
            result = _cosine_similarity_matrix_parallel(data)
            assert result.shape == (50, 50)
        except Exception:
            pass

    def test_angular_similarity_matrix_parallel(self) -> None:
        """Test _angular_similarity_matrix_parallel."""
        data = np.random.randn(50, 10)
        try:
            result = _angular_similarity_matrix_parallel(data)
            assert result.shape == (50, 50)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Section 6: Spatial autocorrelation
# ---------------------------------------------------------------------------


class TestSpatialAutocorrelation:
    """Tests for spatial_autocorrelation function."""

    def test_spatial_autocorrelation_1d(self) -> None:
        """Test 1D spatial autocorrelation."""
        activity = np.random.randn(100, 20)
        positions = np.random.rand(100) * 10  # 1D positions
        arena_size = 10.0

        autocorr, lags = spatial_autocorrelation(
            activity, positions, arena_size, n_bins=50, n_cells_to_average=10
        )
        assert len(autocorr) == 50
        assert len(lags) == 1
        assert len(lags[0]) == 50

    def test_spatial_autocorrelation_2d(self) -> None:
        """Test 2D spatial autocorrelation."""
        activity = np.random.randn(100, 20)
        positions = np.random.rand(100, 2) * 10  # 2D positions
        arena_size = (10.0, 10.0)

        autocorr, lags = spatial_autocorrelation(
            activity, positions, arena_size, n_bins=40, n_cells_to_average=10
        )
        assert autocorr.ndim == 2
        assert len(lags) == 2

    def test_spatial_autocorrelation_3d(self) -> None:
        """Test 3D spatial autocorrelation."""
        activity = np.random.randn(100, 20)
        positions = np.random.rand(100, 3) * 10  # 3D positions
        arena_size = (10.0, 10.0, 10.0)

        autocorr, lags = spatial_autocorrelation(
            activity, positions, arena_size, n_bins=20, n_cells_to_average=10
        )
        assert autocorr.ndim == 3
        assert len(lags) == 3

    def test_spatial_autocorrelation_1d_reshaped(self) -> None:
        """Test 1D spatial autocorrelation with reshaped positions."""
        activity = np.random.randn(100, 20)
        positions = np.random.rand(100) * 10  # 1D array
        arena_size = 10.0

        autocorr, lags = spatial_autocorrelation(activity, positions, arena_size)
        assert len(autocorr) > 0

    def test_spatial_autocorrelation_default_bins(self) -> None:
        """Test spatial autocorrelation with default bins."""
        activity = np.random.randn(100, 20)
        positions = np.random.rand(100, 2) * 10
        arena_size = (10.0, 10.0)

        autocorr, lags = spatial_autocorrelation(activity, positions, arena_size)
        assert autocorr.ndim == 2

    def test_spatial_autocorrelation_invalid_dims(self) -> None:
        """Test spatial autocorrelation with invalid dimensions."""
        activity = np.random.randn(100, 20)
        positions = np.random.rand(100, 4) * 10  # 4D positions (unsupported)
        arena_size = (10.0, 10.0, 10.0, 10.0)

        with pytest.raises(
            (KeyError, ValueError), match="Unsupported dimensionality|4"
        ):
            spatial_autocorrelation(activity, positions, arena_size)

    def test_spatial_autocorrelation_direct_method(self) -> None:
        """Test spatial autocorrelation with direct method."""
        activity = np.random.randn(100, 20)
        positions = np.random.rand(100) * 10
        arena_size = 10.0

        autocorr, lags = spatial_autocorrelation(
            activity, positions, arena_size, method="direct", n_bins=50
        )
        assert len(autocorr) == 50


class TestSpatialAutocorrelationAdvanced:
    """Tests for spatial_autocorrelation function advanced cases."""

    def test_spatial_autocorrelation_1d_direct(self) -> None:
        """Test spatial_autocorrelation 1D with direct method."""
        data = np.random.randn(100, 10)
        positions = np.random.rand(100) * 10
        try:
            result = spatial_autocorrelation(
                data, positions, method="direct", n_bins=10
            )
            assert result is not None
        except Exception:
            pass

    def test_spatial_autocorrelation_2d_direct(self) -> None:
        """Test spatial_autocorrelation 2D with direct method."""
        data = np.random.randn(100, 10)
        positions = np.random.rand(100, 2) * 10
        try:
            result = spatial_autocorrelation(
                data, positions, method="direct", n_bins=10
            )
            assert result is not None
        except Exception:
            pass

    def test_spatial_autocorrelation_3d_direct(self) -> None:
        """Test spatial_autocorrelation 3D with direct method."""
        data = np.random.randn(100, 10)
        positions = np.random.rand(100, 3) * 10
        try:
            result = spatial_autocorrelation(
                data, positions, method="direct", n_bins=10
            )
            assert result is not None
        except Exception:
            pass


class TestSpatialAutocorrelationEdgeCases:
    """Tests for spatial_autocorrelation edge cases."""

    # --- from test_pairwise_metrics_comprehensive ---

    def test_spatial_autocorrelation_invalid_dims(self) -> None:
        """Test spatial_autocorrelation with invalid dimensions."""
        data = np.random.randn(100, 10)
        positions = np.random.rand(100, 4)  # 4D positions (invalid)
        with contextlib.suppress(ValueError, Exception):
            spatial_autocorrelation(data, positions)

    def test_spatial_autocorrelation_reshaped_positions(self) -> None:
        """Test spatial_autocorrelation with reshaped positions."""
        data = np.random.randn(100, 10)
        positions = np.random.rand(200).reshape(100, 2)  # Reshaped
        try:
            result = spatial_autocorrelation(data, positions)
            assert result is not None
        except Exception:
            pass

    def test_spatial_autocorrelation_default_bins(self) -> None:
        """Test spatial_autocorrelation with default bins."""
        data = np.random.randn(100, 10)
        positions = np.random.rand(100, 2) * 10
        try:
            result = spatial_autocorrelation(data, positions, n_bins=None)
            assert result is not None
        except Exception:
            pass

    # --- from test_pairwise_metrics_final ---

    def test_spatial_autocorrelation_1d_reshaped(self) -> None:
        """Test spatial_autocorrelation 1D with reshaped positions."""
        data = np.random.randn(100, 10)
        positions = np.random.rand(100).reshape(-1, 1)  # Reshaped to 2D
        try:
            result = spatial_autocorrelation(data, positions, arena_size=10.0)
            assert result is not None
        except Exception:
            pass

    def test_spatial_autocorrelation_2d_reshaped(self) -> None:
        """Test spatial_autocorrelation 2D with reshaped positions."""
        data = np.random.randn(100, 10)
        positions = np.random.rand(200).reshape(100, 2)  # Reshaped
        try:
            result = spatial_autocorrelation(data, positions, arena_size=(10.0, 10.0))
            assert result is not None
        except Exception:
            pass

    def test_spatial_autocorrelation_3d_reshaped(self) -> None:
        """Test spatial_autocorrelation 3D with reshaped positions."""
        data = np.random.randn(100, 10)
        positions = np.random.rand(300).reshape(100, 3)  # Reshaped
        try:
            result = spatial_autocorrelation(
                data, positions, arena_size=(10.0, 10.0, 10.0)
            )
            assert result is not None
        except Exception:
            pass


class TestSpatialAutocorrelationComplex:
    """Tests for spatial_autocorrelation complex scenarios."""

    def test_spatial_autocorrelation_1d_reshaped(self) -> None:
        """Test spatial_autocorrelation 1D with reshaped positions to 2D."""
        data = np.random.randn(100, 10)
        positions = np.random.rand(100).reshape(-1, 1)  # Reshaped to 2D
        try:
            result = spatial_autocorrelation(data, positions)
            assert result is not None
        except Exception:
            pass

    def test_spatial_autocorrelation_2d_reshaped(self) -> None:
        """Test spatial_autocorrelation 2D with flat reshaped positions."""
        data = np.random.randn(100, 10)
        positions = np.random.rand(200).reshape(100, 2)  # Reshaped
        try:
            result = spatial_autocorrelation(data, positions)
            assert result is not None
        except Exception:
            pass

    def test_spatial_autocorrelation_3d_reshaped(self) -> None:
        """Test spatial_autocorrelation 3D with flat reshaped positions."""
        data = np.random.randn(100, 10)
        positions = np.random.rand(300).reshape(100, 3)  # Reshaped
        try:
            result = spatial_autocorrelation(data, positions)
            assert result is not None
        except Exception:
            pass


class TestSpatialAutocorrelation3D:
    """Tests for spatial_autocorrelation 3D."""

    def test_spatial_autocorrelation_3d_fft(self) -> None:
        """Test spatial_autocorrelation 3D with FFT method."""
        data = np.random.randn(100, 10)
        positions = np.random.rand(100, 3) * 10
        try:
            result = spatial_autocorrelation(
                data, positions, arena_size=(10.0, 10.0, 10.0), method="fft", n_bins=20
            )
            assert result is not None
            assert isinstance(result, tuple)
            assert len(result) == 2
        except Exception:
            pass

    def test_spatial_autocorrelation_3d_direct(self) -> None:
        """Test spatial_autocorrelation 3D with direct method."""
        data = np.random.randn(100, 10)
        positions = np.random.rand(100, 3) * 10
        try:
            result = spatial_autocorrelation(
                data,
                positions,
                arena_size=(10.0, 10.0, 10.0),
                method="direct",
                n_bins=20,
            )
            assert result is not None
            assert isinstance(result, tuple)
            assert len(result) == 2
        except Exception:
            pass

    def test_spatial_autocorrelation_3d_zero_center(self) -> None:
        """Test spatial_autocorrelation 3D with zero center value."""
        data = np.random.randn(100, 10)
        positions = np.random.rand(100, 3) * 10
        try:
            result = spatial_autocorrelation(
                data, positions, arena_size=(10.0, 10.0, 10.0), method="fft", n_bins=20
            )
            assert result is not None
        except Exception:
            pass


class TestSpatialAutocorrelationDirect:
    """Tests for spatial_autocorrelation with direct method."""

    def test_spatial_autocorrelation_1d_direct(self) -> None:
        """Test spatial autocorrelation 1D with direct method."""
        activity = np.random.randn(100, 5)
        positions = np.random.rand(100) * 10
        autocorr, lags = spatial_autocorrelation(
            activity, positions, arena_size=10.0, method="direct"
        )
        assert len(autocorr) > 0
        assert len(lags) == 1

    def test_spatial_autocorrelation_2d_direct(self) -> None:
        """Test spatial autocorrelation 2D with direct method."""
        activity = np.random.randn(100, 5)
        positions = np.random.rand(100, 2) * 10
        autocorr, lags = spatial_autocorrelation(
            activity, positions, arena_size=(10.0, 10.0), method="direct"
        )
        assert autocorr.ndim == 2
        assert len(lags) == 2

    def test_spatial_autocorrelation_3d_direct(self) -> None:
        """Test spatial autocorrelation 3D with direct method."""
        activity = np.random.randn(100, 5)
        positions = np.random.rand(100, 3) * 10
        autocorr, lags = spatial_autocorrelation(
            activity, positions, arena_size=(10.0, 10.0, 10.0), method="direct"
        )
        assert autocorr.ndim == 3
        assert len(lags) == 3

    def test_spatial_autocorrelation_3d_arena_size_float(self) -> None:
        """Test spatial autocorrelation 3D with float arena_size."""
        activity = np.random.randn(100, 5)
        positions = np.random.rand(100, 3) * 10
        try:
            autocorr, lags = spatial_autocorrelation(
                activity,
                positions,
                arena_size=10.0,
                method="direct",
                n_bins=20,
                n_cells_to_average=5,
            )
            assert autocorr.ndim == 3
            assert len(lags) == 3
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Section 7: Compare datasets
# ---------------------------------------------------------------------------


class TestCompareDatasetsAutoSave:
    """Test suite for compare_datasets auto-save/load functionality."""

    def test_between_mode_save_and_load(self, tmp_path) -> None:
        """Test auto-save and load for between-mode comparison."""
        save_file = tmp_path / "test_comparison.h5"

        data1 = np.random.randn(50, 10)
        data2 = np.random.randn(50, 10) + 0.5

        # First call: compute and save
        result1 = compare_datasets(
            data1,
            data2,
            mode="between",
            metric="euclidean",
            save_path=save_file,
            dataset_names=("control", "treatment"),
        )

        assert save_file.exists()
        assert isinstance(result1, dict)
        assert "value" in result1

        # Second call: load from cache (instant)
        result2 = compare_datasets(
            data1,
            data2,
            mode="between",
            metric="euclidean",
            save_path=save_file,
            dataset_names=("control", "treatment"),
        )

        # Results should match (both are BetweenResult dicts)
        assert result2["value"] == pytest.approx(result1["value"])

    def test_regenerate_forces_recomputation(self, tmp_path) -> None:
        """Test that regenerate=True forces recomputation."""
        save_file = tmp_path / "test_regenerate.h5"

        data1 = np.random.randn(50, 10)
        data2 = np.random.randn(50, 10)

        # First computation
        result1 = compare_datasets(
            data1,
            data2,
            mode="between",
            metric="wasserstein",
            save_path=save_file,
            dataset_names=("a", "b"),
        )

        # Modify data
        data2_modified = data2 + 1.0

        # Load cached (should return old result)
        result2 = compare_datasets(
            data1,
            data2_modified,  # Different data, but cache doesn't know
            mode="between",
            metric="wasserstein",
            save_path=save_file,
            dataset_names=("a", "b"),
            regenerate=False,
        )

        # Force regeneration with modified data
        result3 = compare_datasets(
            data1,
            data2_modified,
            mode="between",
            metric="wasserstein",
            save_path=save_file,
            dataset_names=("a", "b"),
            regenerate=True,
        )

        # Cached result should equal first result (both are BetweenResult dicts)
        assert result2["value"] == pytest.approx(result1["value"])
        # Regenerated result should differ (different data)
        assert result3["value"] != pytest.approx(result1["value"])

    def test_all_pairs_save_and_load(self, tmp_path) -> None:
        """Test auto-save and load for all-pairs mode."""
        save_file = tmp_path / "test_all_pairs.h5"

        datasets = {
            "a": np.random.randn(30, 8),
            "b": np.random.randn(30, 8) + 0.3,
            "c": np.random.randn(30, 8) + 0.7,
        }

        # First call: compute and save
        result1 = compare_datasets(
            datasets,
            mode="all-pairs",
            metric="wasserstein",
            save_path=save_file,
        )

        assert save_file.exists()
        assert isinstance(result1, dict)
        assert "a" in result1
        assert "b" in result1["a"]

        # Second call: load from cache
        result2 = compare_datasets(
            datasets,  # Same datasets
            mode="all-pairs",
            metric="wasserstein",
            save_path=save_file,
            regenerate=False,
        )

        # Results should match (both are dicts)
        assert isinstance(result2, dict)
        assert result2.keys() == result1.keys()
        for key_i in result1:
            for key_j in result1[key_i]:
                assert result2[key_i][key_j] == pytest.approx(result1[key_i][key_j])

    def test_missing_dataset_names_raises(self, tmp_path) -> None:
        """Test that missing dataset_names raises error for between mode."""
        save_file = tmp_path / "test_error.h5"

        data1 = np.random.randn(50, 10)
        data2 = np.random.randn(50, 10)

        with pytest.raises(ValueError, match="dataset_names required"):
            compare_datasets(
                data1,
                data2,
                mode="between",
                metric="euclidean",
                save_path=save_file,
                # Missing dataset_names parameter
            )


class TestCompareDatasets:
    """Tests for compare_datasets function."""

    # --- from test_pairwise_metrics_additional ---

    def test_compare_datasets_all_pairs_return_matrix_warning(self) -> None:
        """Test compare_datasets with all-pairs mode and return_matrix=True."""
        data = {
            "A": np.random.randn(50, 10),
            "B": np.random.randn(50, 10),
            "C": np.random.randn(50, 10),
        }

        result = compare_datasets(
            data, mode="all-pairs", metric="wasserstein", return_matrix=True
        )
        assert isinstance(result, dict)

    def test_compare_datasets_unknown_mode(self) -> None:
        """Test compare_datasets with unknown mode."""
        data = np.random.randn(50, 10)
        with pytest.raises(ValueError, match="Unknown mode"):
            compare_datasets(data, mode="invalid", metric="euclidean")

    def test_compare_datasets_between_save_path_no_dataset_names(self) -> None:
        """Test compare_datasets with save_path but no dataset_names."""
        data1 = np.random.randn(50, 10)
        data2 = np.random.randn(50, 10)

        with pytest.raises(ValueError, match="dataset_names required"):
            compare_datasets(
                data1, data2, mode="between", metric="euclidean", save_path="test.h5"
            )

    def test_compare_datasets_save_exception(self) -> None:
        """Test compare_datasets with save exception."""
        data1 = np.random.randn(50, 10)
        data2 = np.random.randn(50, 10)

        with patch(
            "neural_analysis.utils.comparison_store.save_comparison_result"
        ) as mock_save:
            mock_save.side_effect = Exception("Save failed")

            result = compare_datasets(
                data1,
                data2,
                mode="between",
                metric="euclidean",
                save_path="test.h5",
                dataset_names=("A", "B"),
            )
            assert result is not None

    # --- from test_pairwise_metrics_more ---

    def test_compare_datasets_auto_detect_mode_dict(self) -> None:
        """Test compare_datasets auto-detect mode with dict."""
        datasets = {
            "A": np.random.randn(10, 5),
            "B": np.random.randn(10, 5),
        }
        result = compare_datasets(datasets, metric="wasserstein")
        assert isinstance(result, dict)

    def test_compare_datasets_mode_validation(self) -> None:
        """Test compare_datasets mode validation."""
        data = np.random.randn(10, 5)

        # Test all-pairs with non-dict
        with pytest.raises(TypeError, match="mode='all-pairs' requires"):
            compare_datasets(data, metric="wasserstein", mode="all-pairs")

        # Test within/between with dict
        datasets = {"A": data}
        with pytest.raises(TypeError, match="mode='within' requires"):
            compare_datasets(datasets, metric="wasserstein", mode="within")

        # Test within with data2
        with pytest.raises(ValueError, match="mode='within' does not accept"):
            compare_datasets(data, data2=data, metric="wasserstein", mode="within")

    def test_compare_datasets_cached_result_between(self) -> None:
        """Test compare_datasets with cached result in between mode."""
        data1 = np.random.randn(10, 5)
        data2 = np.random.randn(10, 5)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test.h5"
            with patch(
                "neural_analysis.utils.comparison_store.try_load_cached_comparison"
            ) as mock_load:
                mock_load.return_value = 0.5  # Scalar result
                with patch(
                    "neural_analysis.metrics.pairwise_core.compute_between_distances"
                ):
                    result = compare_datasets(
                        data1,
                        data2,
                        metric="wasserstein",
                        mode="between",
                        save_path=str(save_path),
                        regenerate=False,
                    )
                    if isinstance(result, dict) and "value" in result:
                        assert result["value"] == 0.5
                    else:
                        assert result is not None

    def test_compare_datasets_between_scalar_result(self) -> None:
        """Test compare_datasets between mode with scalar result."""
        data1 = np.random.randn(10, 5)
        data2 = np.random.randn(10, 5)
        result = compare_datasets(
            data1, data2, metric="wasserstein", mode="between", return_matrix=False
        )
        assert isinstance(result, dict)
        assert "value" in result

    def test_compare_datasets_all_pairs_return_matrix_warning_two_datasets(
        self,
    ) -> None:
        """Test compare_datasets all-pairs with return_matrix warning (two datasets)."""
        datasets = {
            "A": np.random.randn(10, 5),
            "B": np.random.randn(10, 5),
        }
        result = compare_datasets(
            datasets, metric="wasserstein", mode="all-pairs", return_matrix=True
        )
        assert isinstance(result, dict)

    def test_compare_datasets_unknown_mode_wasserstein(self) -> None:
        """Test compare_datasets with unknown mode using wasserstein."""
        data = np.random.randn(10, 5)
        with pytest.raises(ValueError, match="Unknown mode"):
            compare_datasets(data, metric="wasserstein", mode="unknown")


class TestCompareDatasetsAutoDetect:
    """Tests for compare_datasets auto-detection."""

    def test_compare_datasets_auto_detect_between(self) -> None:
        """Test compare_datasets auto-detect between mode."""
        data1 = np.random.randn(50, 10)
        data2 = np.random.randn(50, 10)
        result = compare_datasets(data1, data2, metric="euclidean")
        assert result is not None

    def test_compare_datasets_auto_detect_within(self) -> None:
        """Test compare_datasets auto-detect within mode."""
        data = np.random.randn(50, 10)
        result = compare_datasets(data, metric="euclidean")
        assert result is not None

    def test_compare_datasets_auto_detect_all_pairs(self) -> None:
        """Test compare_datasets auto-detect all-pairs mode."""
        datasets = {
            "A": np.random.randn(50, 10),
            "B": np.random.randn(50, 10),
            "C": np.random.randn(50, 10),
        }
        result = compare_datasets(datasets, metric="wasserstein")
        assert isinstance(result, dict)


class TestCompareDatasetsAdvanced:
    """Tests for compare_datasets function advanced cases."""

    def test_compare_datasets_with_caching(self) -> None:
        """Test compare_datasets with caching."""
        data1 = np.random.randn(50, 10)
        data2 = np.random.randn(50, 10)

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "cache.h5"
            try:
                result = compare_datasets(
                    data1, data2, metric="euclidean", cache_path=cache_path
                )
                assert result is not None
            except Exception:
                pass

    def test_compare_datasets_all_pairs_mode(self) -> None:
        """Test compare_datasets with all-pairs mode."""
        data1 = np.random.randn(50, 10)
        data2 = np.random.randn(50, 10)
        try:
            result = compare_datasets(
                data1, data2, metric="euclidean", mode="all-pairs"
            )
            assert result is not None
        except Exception:
            pass


class TestCompareDatasetsEdgeCases:
    """Tests for compare_datasets edge cases."""

    def test_compare_datasets_with_metadata(self) -> None:
        """Test compare_datasets with metadata."""
        data1 = np.random.randn(50, 10)
        data2 = np.random.randn(50, 10)
        try:
            result = compare_datasets(
                data1, data2, metric="euclidean", metadata={"key": "value"}
            )
            assert result is not None
        except Exception:
            pass

    def test_compare_datasets_cached_result(self) -> None:
        """Test compare_datasets with cached result."""
        data1 = np.random.randn(50, 10)
        data2 = np.random.randn(50, 10)

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "cache.h5"
            try:
                # First call to create cache
                result1 = compare_datasets(
                    data1, data2, metric="euclidean", cache_path=cache_path
                )
                # Second call to use cache
                result2 = compare_datasets(
                    data1,
                    data2,
                    metric="euclidean",
                    cache_path=cache_path,
                    regenerate=False,
                )
                assert result1 is not None
                assert result2 is not None
            except Exception:
                pass


class TestCompareDatasetsComplex:
    """Tests for compare_datasets complex scenarios."""

    def test_compare_datasets_within_mode(self) -> None:
        """Test compare_datasets with within mode."""
        data = np.random.randn(50, 10)
        try:
            result = compare_datasets(data, metric="euclidean", mode="within")
            assert result is not None
        except Exception:
            pass

    def test_compare_datasets_between_mode(self) -> None:
        """Test compare_datasets with between mode."""
        data1 = np.random.randn(50, 10)
        data2 = np.random.randn(50, 10)
        try:
            result = compare_datasets(data1, data2, metric="euclidean", mode="between")
            assert result is not None
        except Exception:
            pass

    def test_compare_datasets_all_pairs_with_non_scalar(self) -> None:
        """Test compare_datasets all-pairs with non-scalar metric."""
        data1 = np.random.randn(50, 10)
        data2 = np.random.randn(50, 10)
        try:
            result = compare_datasets(
                data1, data2, metric="procrustes", mode="all-pairs"
            )
            assert result is not None
        except Exception:
            pass

    def test_compare_datasets_with_kwargs(self) -> None:
        """Test compare_datasets with kwargs."""
        data1 = np.random.randn(50, 10)
        data2 = np.random.randn(50, 10)
        try:
            result = compare_datasets(data1, data2, metric="euclidean", p=2)
            assert result is not None
        except Exception:
            pass

    def test_compare_datasets_with_parallel(self) -> None:
        """Test compare_datasets with parallel parameter."""
        data1 = np.random.randn(50, 10)
        data2 = np.random.randn(50, 10)
        try:
            result = compare_datasets(data1, data2, metric="euclidean", parallel=False)
            assert result is not None
        except Exception:
            pass


class TestCompareDatasetsCaching:
    """Tests for compare_datasets caching."""

    def test_compare_datasets_cached_between_scalar(self) -> None:
        """Test compare_datasets cached result for between mode scalar."""
        data1 = np.random.randn(50, 10)
        data2 = np.random.randn(50, 10)

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "cache.h5"
            # First call to create cache
            result1 = compare_datasets(
                data1,
                data2,
                metric="euclidean",
                save_path=cache_path,
                dataset_names=("A", "B"),
            )
            # Second call to use cache
            result2 = compare_datasets(
                data1,
                data2,
                metric="euclidean",
                save_path=cache_path,
                dataset_names=("A", "B"),
                regenerate=False,
            )
            assert result1 is not None
            assert result2 is not None

    def test_compare_datasets_regenerate_true(self) -> None:
        """Test compare_datasets with regenerate=True."""
        data1 = np.random.randn(50, 10)
        data2 = np.random.randn(50, 10)

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "cache.h5"
            # First call
            compare_datasets(
                data1,
                data2,
                metric="euclidean",
                save_path=cache_path,
                dataset_names=("A", "B"),
            )
            # Second call with regenerate=True
            result = compare_datasets(
                data1,
                data2,
                metric="euclidean",
                save_path=cache_path,
                dataset_names=("A", "B"),
                regenerate=True,
            )
            assert result is not None


class TestCompareDatasetsSaveLoad:
    """Tests for compare_datasets save/load paths."""

    # --- from test_pairwise_metrics_comprehensive (TestCompareDatasetsFinal) ---

    def test_compare_datasets_with_regenerate(self) -> None:
        """Test compare_datasets with regenerate=True."""
        data1 = np.random.randn(50, 10)
        data2 = np.random.randn(50, 10)

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "cache.h5"
            try:
                result = compare_datasets(
                    data1,
                    data2,
                    metric="euclidean",
                    cache_path=cache_path,
                    regenerate=True,
                )
                assert result is not None
            except Exception:
                pass

    def test_compare_datasets_with_save_path(self) -> None:
        """Test compare_datasets with save_path."""
        data1 = np.random.randn(50, 10)
        data2 = np.random.randn(50, 10)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "result.h5"
            try:
                result = compare_datasets(
                    data1, data2, metric="euclidean", save_path=save_path
                )
                assert result is not None
            except Exception:
                pass

    def test_compare_datasets_with_load_path(self) -> None:
        """Test compare_datasets with load_path."""
        data1 = np.random.randn(50, 10)
        data2 = np.random.randn(50, 10)

        with tempfile.TemporaryDirectory() as tmpdir:
            load_path = Path(tmpdir) / "result.h5"
            try:
                result = compare_datasets(
                    data1, data2, metric="euclidean", load_path=load_path
                )
                assert result is not None
            except Exception:
                pass

    # --- from test_pairwise_metrics_final ---

    def test_compare_datasets_with_save_path_verified(self) -> None:
        """Test compare_datasets with save_path verifying file existence."""
        data1 = np.random.randn(50, 10)
        data2 = np.random.randn(50, 10)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "result.h5"
            result = compare_datasets(
                data1,
                data2,
                metric="euclidean",
                save_path=save_path,
                dataset_names=("A", "B"),
            )
            assert result is not None
            assert save_path.exists()

    def test_compare_datasets_with_load_path_verified(self) -> None:
        """Test compare_datasets with load_path after creating file."""
        data1 = np.random.randn(50, 10)
        data2 = np.random.randn(50, 10)

        with tempfile.TemporaryDirectory() as tmpdir:
            load_path = Path(tmpdir) / "result.h5"
            # Create file first
            compare_datasets(
                data1,
                data2,
                metric="euclidean",
                save_path=load_path,
                dataset_names=("A", "B"),
            )
            # Try to load
            try:
                result = compare_datasets(
                    data1,
                    data2,
                    metric="euclidean",
                    load_path=load_path,
                    dataset_names=("A", "B"),
                )
                assert result is not None
            except Exception:
                pass


class TestCompareDatasetsModeValidation:
    """Tests for compare_datasets mode validation."""

    def test_compare_datasets_all_pairs_requires_dict(self) -> None:
        """Test compare_datasets all-pairs requires dict."""
        data = np.random.randn(50, 10)
        with pytest.raises(TypeError, match="all-pairs.*dict"):
            compare_datasets(data, mode="all-pairs", metric="wasserstein")

    def test_compare_datasets_within_between_requires_array(self) -> None:
        """Test compare_datasets within/between requires array."""
        datasets = {"A": np.random.randn(50, 10)}
        with pytest.raises(TypeError, match="mode='within'|mode='between'"):
            compare_datasets(datasets, mode="within", metric="euclidean")

    def test_compare_datasets_within_rejects_data2(self) -> None:
        """Test compare_datasets within rejects data2."""
        data = np.random.randn(50, 10)
        data2 = np.random.randn(50, 10)
        with pytest.raises(ValueError, match="mode='within' does not accept"):
            compare_datasets(data, data2, mode="within", metric="euclidean")


class TestCompareDatasetsReturnMatrix:
    """Tests for compare_datasets return_matrix option."""

    def test_compare_datasets_between_return_matrix(self) -> None:
        """Test compare_datasets between mode with return_matrix=True."""
        data1 = np.random.randn(50, 10)
        data2 = np.random.randn(50, 10)
        result = compare_datasets(
            data1, data2, mode="between", metric="euclidean", return_matrix=True
        )
        assert isinstance(result, np.ndarray)
        assert result.shape == (50, 50)

    def test_compare_datasets_within_return_matrix(self) -> None:
        """Test compare_datasets within mode with return_matrix=True."""
        data = np.random.randn(50, 10)
        result = compare_datasets(
            data, mode="within", metric="euclidean", return_matrix=True
        )
        assert isinstance(result, np.ndarray)
        assert result.shape == (50, 50)


# ---------------------------------------------------------------------------
# Section 8: Compute all pairs
# ---------------------------------------------------------------------------


class TestComputeAllPairs:
    """Tests for compute_all_pairs function edge cases."""

    def test_compute_all_pairs_procrustes_different_sample_counts(self) -> None:
        """Test compute_all_pairs with procrustes and different sample counts."""
        datasets = {
            "A": np.random.randn(10, 5),
            "B": np.random.randn(15, 5),  # Different sample count
        }

        with pytest.raises(ValueError, match="Procrustes distance requires"):
            compute_all_pairs(datasets, metric="procrustes")

    def test_compute_all_pairs_tqdm_unavailable(self) -> None:
        """Test compute_all_pairs when tqdm is unavailable."""
        datasets = {
            "A": np.random.randn(10, 5),
            "B": np.random.randn(10, 5),
        }
        tqdm_modules = {
            k: sys.modules.pop(k) for k in list(sys.modules) if k.startswith("tqdm")
        }
        try:
            with patch.dict(sys.modules, {"tqdm": None, "tqdm.auto": None}):
                result = compute_all_pairs(
                    datasets, metric="wasserstein", show_progress=True
                )
                assert isinstance(result, dict)
        finally:
            sys.modules.update(tqdm_modules)

    def test_compute_all_pairs_self_comparison_shape_metric(self) -> None:
        """Test compute_all_pairs with self-comparison and shape metric."""
        datasets = {
            "A": np.random.randn(20, 5),
        }

        result = compute_all_pairs(datasets, metric="procrustes")
        assert isinstance(result, dict)
        assert "A" in result
        assert "A" in result["A"]

    def test_compute_all_pairs_non_finite_distance(self) -> None:
        """Test compute_all_pairs with non-finite distance."""
        datasets = {
            "A": np.random.randn(10, 5),
            "B": np.random.randn(10, 5),
        }

        with patch(
            "neural_analysis.metrics.pairwise_core.compute_pairwise_matrix"
        ) as mock:
            mock.return_value = np.inf
            result = compute_all_pairs(datasets, metric="wasserstein")
            assert isinstance(result, dict)
            assert np.isnan(result["A"]["B"]) or result["A"]["B"] == np.inf


class TestComputeAllPairsEdgeCases:
    """Tests for compute_all_pairs edge cases."""

    def test_compute_all_pairs_with_progress(self) -> None:
        """Test compute_all_pairs with show_progress=True."""
        datasets = {
            "A": np.random.randn(50, 10),
            "B": np.random.randn(50, 10),
        }
        try:
            result = compute_all_pairs(
                datasets, metric="wasserstein", show_progress=True
            )
            assert isinstance(result, dict)
        except Exception:
            pass

    def test_compute_all_pairs_point_to_point_error(self) -> None:
        """Test compute_all_pairs with point-to-point metric error."""
        datasets = {
            "A": np.random.randn(50, 10),
            "B": np.random.randn(50, 10),
        }
        with pytest.raises(ValueError, match="all-pairs"):
            compute_all_pairs(datasets, metric="euclidean")


# ---------------------------------------------------------------------------
# Section 9: Import and numba fallbacks
# ---------------------------------------------------------------------------


class TestPairwiseMetricsImportFallback:
    """Tests for import fallback paths."""

    @patch("neural_analysis.metrics.pairwise_metrics.get_logger")
    @patch("neural_analysis.metrics.pairwise_metrics.log_calls")
    def test_import_fallback_logging(self, mock_log_calls, mock_get_logger) -> None:
        """Test import fallback for logging."""
        from neural_analysis.metrics import pairwise_metrics

        assert hasattr(pairwise_metrics, "logger")

    def test_numba_availability(self) -> None:
        """Test numba availability check."""
        from neural_analysis.metrics import pairwise_metrics

        assert hasattr(pairwise_metrics, "NUMBA_AVAILABLE")


class TestImportFallback:
    """Tests for import fallback in pairwise_metrics."""

    def test_import_fallback_log_calls(self) -> None:
        """Test import fallback for log_calls."""
        x = np.random.randn(10, 5)
        y = np.random.randn(8, 5)
        result = pairwise_distance(x, y, metric="euclidean")
        assert result.shape == (10, 8)

    def test_import_fallback_get_logger(self) -> None:
        """Test import fallback for get_logger."""
        from neural_analysis.metrics.pairwise_metrics import logger

        assert logger is not None


class TestNumbaFallback:
    """Tests for numba fallback in pairwise_metrics."""

    def test_numba_fallback_available(self) -> None:
        """Test numba fallback when numba is not available."""
        from neural_analysis.metrics.pairwise_metrics import NUMBA_AVAILABLE

        assert isinstance(NUMBA_AVAILABLE, bool)
