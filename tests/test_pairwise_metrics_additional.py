"""Additional tests for pairwise_metrics module to improve coverage."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from neural_analysis.metrics.pairwise_metrics import (
    angular_similarity_matrix,
    compare_datasets,
    correlation,
    correlation_matrix,
    cosine_similarity_matrix,
    similarity_matrix,
    spatial_autocorrelation,
)


class TestAngularSimilarityMatrix:
    """Tests for angular_similarity_matrix function."""

    def test_angular_similarity_matrix_basic(self) -> None:
        """Test basic angular similarity matrix (covers line 1609)."""
        data = np.random.randn(10, 5)
        result = angular_similarity_matrix(data)
        assert result.shape == (5, 5)
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_angular_similarity_matrix_invalid_shape(self) -> None:
        """Test angular similarity matrix with invalid shape (covers lines 1611-1612)."""
        data = np.array([1, 2, 3])  # 1D array
        with pytest.raises(ValueError, match="must be 2D"):
            angular_similarity_matrix(data)


class TestSimilarityMatrix:
    """Tests for similarity_matrix function."""

    def test_similarity_matrix_pearson(self) -> None:
        """Test similarity matrix with pearson method (covers lines 1640-1644)."""
        data = np.random.randn(20, 10)
        result = similarity_matrix(data, method="pearson")
        assert result.shape == (10, 10)
        assert np.allclose(result, result.T)  # Symmetric

    def test_similarity_matrix_spearman(self) -> None:
        """Test similarity matrix with spearman method."""
        data = np.random.randn(20, 10)
        result = similarity_matrix(data, method="spearman")
        assert result.shape == (10, 10)

    def test_similarity_matrix_kendall(self) -> None:
        """Test similarity matrix with kendall method."""
        data = np.random.randn(20, 10)
        result = similarity_matrix(data, method="kendall")
        assert result.shape == (10, 10)

    def test_similarity_matrix_cosine(self) -> None:
        """Test similarity matrix with cosine method (covers lines 1645-1649)."""
        data = np.random.randn(20, 10)
        result = similarity_matrix(data, method="cosine")
        assert result.shape == (10, 10)

    def test_similarity_matrix_angular(self) -> None:
        """Test similarity matrix with angular method (covers lines 1650-1654)."""
        data = np.random.randn(20, 10)
        result = similarity_matrix(data, method="angular")
        assert result.shape == (10, 10)

    def test_similarity_matrix_parallel_pearson(self) -> None:
        """Test similarity matrix with parallel=True (covers lines 1641-1642)."""
        data = np.random.randn(20, 10)
        result = similarity_matrix(data, method="pearson", parallel=True)
        assert result.shape == (10, 10)

    def test_similarity_matrix_parallel_cosine(self) -> None:
        """Test similarity matrix with parallel=True for cosine (covers lines 1646-1647)."""
        data = np.random.randn(20, 10)
        result = similarity_matrix(data, method="cosine", parallel=True)
        assert result.shape == (10, 10)

    def test_similarity_matrix_parallel_angular(self) -> None:
        """Test similarity matrix with parallel=True for angular (covers lines 1651-1652)."""
        data = np.random.randn(20, 10)
        result = similarity_matrix(data, method="angular", parallel=True)
        assert result.shape == (10, 10)

    def test_similarity_matrix_with_plot(self) -> None:
        """Test similarity matrix with plot=True (covers lines 1658-1659)."""
        data = np.random.randn(20, 10)
        with patch("neural_analysis.metrics.pairwise_metrics._plot_similarity_matrix") as mock_plot:
            result = similarity_matrix(data, method="pearson", plot=True)
            assert result.shape == (10, 10)
            mock_plot.assert_called_once()

    def test_similarity_matrix_invalid_method(self) -> None:
        """Test similarity matrix with invalid method (covers lines 1655-1656)."""
        data = np.random.randn(20, 10)
        with pytest.raises(ValueError, match="Unknown method"):
            similarity_matrix(data, method="invalid")

    def test_similarity_matrix_invalid_shape(self) -> None:
        """Test similarity matrix with invalid shape (covers lines 1631-1632)."""
        data = np.array([1, 2, 3])  # 1D array
        with pytest.raises(ValueError, match="must be 2D"):
            similarity_matrix(data)


class TestSpatialAutocorrelation:
    """Tests for spatial_autocorrelation function."""

    def test_spatial_autocorrelation_1d(self) -> None:
        """Test 1D spatial autocorrelation (covers lines 1812-1831)."""
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
        """Test 2D spatial autocorrelation (covers lines 1832-1840)."""
        activity = np.random.randn(100, 20)
        positions = np.random.rand(100, 2) * 10  # 2D positions
        arena_size = (10.0, 10.0)

        autocorr, lags = spatial_autocorrelation(
            activity, positions, arena_size, n_bins=40, n_cells_to_average=10
        )
        assert autocorr.ndim == 2
        assert len(lags) == 2

    def test_spatial_autocorrelation_3d(self) -> None:
        """Test 3D spatial autocorrelation (covers lines 1841-1849)."""
        activity = np.random.randn(100, 20)
        positions = np.random.rand(100, 3) * 10  # 3D positions
        arena_size = (10.0, 10.0, 10.0)

        autocorr, lags = spatial_autocorrelation(
            activity, positions, arena_size, n_bins=20, n_cells_to_average=10
        )
        assert autocorr.ndim == 3
        assert len(lags) == 3

    def test_spatial_autocorrelation_1d_reshaped(self) -> None:
        """Test 1D spatial autocorrelation with reshaped positions (covers line 1814)."""
        activity = np.random.randn(100, 20)
        positions = np.random.rand(100) * 10  # 1D array
        arena_size = 10.0

        autocorr, lags = spatial_autocorrelation(activity, positions, arena_size)
        assert len(autocorr) > 0

    def test_spatial_autocorrelation_default_bins(self) -> None:
        """Test spatial autocorrelation with default bins (covers lines 1818-1819)."""
        activity = np.random.randn(100, 20)
        positions = np.random.rand(100, 2) * 10
        arena_size = (10.0, 10.0)

        autocorr, lags = spatial_autocorrelation(activity, positions, arena_size)
        assert autocorr.ndim == 2

    def test_spatial_autocorrelation_invalid_dims(self) -> None:
        """Test spatial autocorrelation with invalid dimensions (covers lines 1850-1851)."""
        activity = np.random.randn(100, 20)
        positions = np.random.rand(100, 4) * 10  # 4D positions (unsupported)
        arena_size = (10.0, 10.0, 10.0, 10.0)

        # The error happens at n_bins lookup (KeyError) or ValueError
        with pytest.raises((KeyError, ValueError), match="Unsupported dimensionality|4"):
            spatial_autocorrelation(activity, positions, arena_size)

    def test_spatial_autocorrelation_direct_method(self) -> None:
        """Test spatial autocorrelation with direct method (covers line 1879)."""
        activity = np.random.randn(100, 20)
        positions = np.random.rand(100) * 10
        arena_size = 10.0

        autocorr, lags = spatial_autocorrelation(
            activity, positions, arena_size, method="direct", n_bins=50
        )
        assert len(autocorr) == 50


class TestCorrelationMatrix:
    """Tests for correlation_matrix function."""

    def test_correlation_matrix_pearson(self) -> None:
        """Test correlation matrix with pearson method."""
        data = np.random.randn(20, 10)
        result = correlation_matrix(data, method="pearson")
        assert result.shape == (10, 10)
        assert np.allclose(result, result.T)  # Symmetric

    def test_correlation_matrix_spearman(self) -> None:
        """Test correlation matrix with spearman method."""
        data = np.random.randn(20, 10)
        result = correlation_matrix(data, method="spearman")
        assert result.shape == (10, 10)

    def test_correlation_matrix_kendall(self) -> None:
        """Test correlation matrix with kendall method."""
        data = np.random.randn(20, 10)
        result = correlation_matrix(data, method="kendall")
        assert result.shape == (10, 10)


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


class TestCompareDatasets:
    """Tests for compare_datasets function."""

    def test_compare_datasets_all_pairs_return_matrix_warning(self) -> None:
        """Test compare_datasets with all-pairs mode and return_matrix=True (covers lines 1477-1480)."""
        data = {
            "A": np.random.randn(50, 10),
            "B": np.random.randn(50, 10),
            "C": np.random.randn(50, 10),
        }
        
        # Should warn but still work - use a scalar metric for all-pairs
        result = compare_datasets(
            data, mode="all-pairs", metric="wasserstein", return_matrix=True
        )
        assert isinstance(result, dict)

    def test_compare_datasets_unknown_mode(self) -> None:
        """Test compare_datasets with unknown mode (covers lines 1491-1494)."""
        data = np.random.randn(50, 10)
        with pytest.raises(ValueError, match="Unknown mode"):
            compare_datasets(data, mode="invalid", metric="euclidean")

    def test_compare_datasets_between_save_path_no_dataset_names(self) -> None:
        """Test compare_datasets with save_path but no dataset_names (covers lines 1500-1504)."""
        data1 = np.random.randn(50, 10)
        data2 = np.random.randn(50, 10)
        
        with pytest.raises(ValueError, match="dataset_names required"):
            compare_datasets(
                data1, data2, mode="between", metric="euclidean", save_path="test.h5"
            )

    def test_compare_datasets_save_exception(self) -> None:
        """Test compare_datasets with save exception (covers lines 1521-1523)."""
        data1 = np.random.randn(50, 10)
        data2 = np.random.randn(50, 10)
        
        with patch("neural_analysis.utils.comparison_store.save_comparison_result") as mock_save:
            mock_save.side_effect = Exception("Save failed")
            
            # Should not raise, just log warning
            result = compare_datasets(
                data1,
                data2,
                mode="between",
                metric="euclidean",
                save_path="test.h5",
                dataset_names=("A", "B"),
            )
            assert result is not None


class TestCorrelation:
    """Tests for correlation function."""

    def test_correlation_matrix_mode_parallel_numba(self) -> None:
        """Test correlation with matrix mode and parallel=True (covers lines 1541-1542)."""
        data = np.random.randn(100, 20)
        
        with patch("neural_analysis.metrics.pairwise_metrics.NUMBA_AVAILABLE", True):
            with patch("neural_analysis.metrics.pairwise_metrics._correlation_matrix_parallel") as mock_parallel:
                mock_parallel.return_value = np.corrcoef(data.T)
                result = correlation(data, method="pearson", mode="matrix", parallel=True)
                assert result.shape == (20, 20)

    def test_correlation_pairwise_mode(self) -> None:
        """Test correlation with pairwise mode (covers lines 1545-1551)."""
        data = np.random.randn(100, 5)
        result = correlation(data, method="pearson", mode="pairwise")
        assert result.shape == (4,)  # n_features - 1

    def test_correlation_unknown_mode(self) -> None:
        """Test correlation with unknown mode (covers lines 1552-1553)."""
        data = np.random.randn(100, 20)
        with pytest.raises(ValueError, match="Unknown mode"):
            correlation(data, method="pearson", mode="invalid")


class TestCorrelationMatrixParallel:
    """Tests for _correlation_matrix_parallel function."""

    def test_correlation_matrix_parallel_spearman_no_numba(self) -> None:
        """Test _correlation_matrix_parallel with spearman and no numba (covers lines 1704-1706)."""
        from neural_analysis.metrics.pairwise_metrics import _correlation_matrix_parallel
        
        data = np.random.randn(100, 20)
        
        with patch("neural_analysis.metrics.pairwise_metrics.NUMBA_AVAILABLE", False):
            result = _correlation_matrix_parallel(data, method="spearman")
            assert result.shape == (20, 20)

    def test_correlation_matrix_parallel_kendall_no_numba(self) -> None:
        """Test _correlation_matrix_parallel with kendall and no numba (covers lines 1710-1712)."""
        from neural_analysis.metrics.pairwise_metrics import _correlation_matrix_parallel
        
        data = np.random.randn(100, 20)
        
        with patch("neural_analysis.metrics.pairwise_metrics.NUMBA_AVAILABLE", False):
            result = _correlation_matrix_parallel(data, method="kendall")
            assert result.shape == (20, 20)

    def test_correlation_matrix_parallel_unknown_method(self) -> None:
        """Test _correlation_matrix_parallel with unknown method (covers lines 1713-1714)."""
        from neural_analysis.metrics.pairwise_metrics import _correlation_matrix_parallel
        
        data = np.random.randn(100, 20)
        
        with pytest.raises(ValueError, match="Unknown method"):
            _correlation_matrix_parallel(data, method="invalid")


class TestPlotSimilarityMatrix:
    """Tests for _plot_similarity_matrix function."""

    # Note: test_plot_similarity_matrix_plotting_unavailable is skipped
    # because mocking the import inside the function is complex

    def test_plot_similarity_matrix_with_plot_config(self) -> None:
        """Test _plot_similarity_matrix with plot_config (covers lines 1678-1679)."""
        from neural_analysis.metrics.pairwise_metrics import _plot_similarity_matrix
        
        similarity = np.random.rand(10, 10)
        
        with patch("neural_analysis.plotting.plot_heatmap") as mock_plot:
            plot_config = {"title": "Custom Title"}
            _plot_similarity_matrix(similarity, method="pearson", plot_config=plot_config)
            mock_plot.assert_called_once()

