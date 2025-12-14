"""Additional tests for distributions module to improve coverage."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from neural_analysis.metrics.distributions import (
    _compute_summary_statistics,
    _deserialize_pairs,
    _function_accepts_argument,
    _normalize_metrics_input,
    _progress_iterable,
    _serialize_pairs,
    modify_matrix,
)


class TestProgressIterable:
    """Tests for _progress_iterable function."""

    def test_progress_iterable_disabled(self) -> None:
        """Test progress iterable with enable=False (covers line 100-101)."""
        data = [1, 2, 3, 4, 5]
        result = list(_progress_iterable(data, enable=False))
        assert result == data

    def test_progress_iterable_enabled(self) -> None:
        """Test progress iterable with enable=True (covers lines 102-106)."""
        data = [1, 2, 3, 4, 5]
        # This will either use tqdm if available, or return the iterable directly
        result = list(_progress_iterable(data, enable=True))
        assert result == data


class TestNormalizeMetricsInput:
    """Tests for _normalize_metrics_input function."""

    def test_normalize_metrics_input_sequence(self) -> None:
        """Test normalize metrics input with sequence (covers lines 120-122)."""
        metrics = ["wasserstein", "jensen-shannon"]
        result = _normalize_metrics_input(metrics)
        assert isinstance(result, dict)
        assert "wasserstein" in result
        assert "jensen-shannon" in result

    def test_normalize_metrics_input_mapping(self) -> None:
        """Test normalize metrics input with mapping (covers lines 115-119)."""
        metrics = {"wasserstein": {"param": 1}, "jensen-shannon": {}}
        common_kwargs = {"common": "value"}
        result = _normalize_metrics_input(metrics, common_kwargs=common_kwargs)
        assert isinstance(result, dict)
        assert result["wasserstein"]["param"] == 1
        assert result["wasserstein"]["common"] == "value"

    def test_normalize_metrics_input_empty(self) -> None:
        """Test normalize metrics input with empty sequence (covers lines 120-121)."""
        with pytest.raises(ValueError, match="must contain at least one"):
            _normalize_metrics_input([])


class TestSerializePairs:
    """Tests for _serialize_pairs function."""

    def test_serialize_pairs_empty(self) -> None:
        """Test serialize pairs with empty dict (covers lines 139-140)."""
        result = _serialize_pairs(None)
        assert result == {}

    def test_serialize_pairs_basic(self) -> None:
        """Test serialize pairs with data (covers lines 141-146)."""
        pairs = {(0, 1): 0.5, (1, 2): 0.7, (2, 3): 0.9}
        result = _serialize_pairs(pairs)
        assert "pair_indices" in result
        assert "pair_values" in result
        assert len(result["pair_indices"]) == 3
        assert len(result["pair_values"]) == 3


class TestDeserializePairs:
    """Tests for _deserialize_pairs function."""

    def test_deserialize_pairs_empty(self) -> None:
        """Test deserialize pairs with empty arrays (covers lines 153-154)."""
        result = _deserialize_pairs(None)
        assert result is None

    def test_deserialize_pairs_missing_keys(self) -> None:
        """Test deserialize pairs with missing keys (covers lines 155-156)."""
        arrays = {"other_key": np.array([1, 2, 3])}
        result = _deserialize_pairs(arrays)
        assert result is None

    def test_deserialize_pairs_basic(self) -> None:
        """Test deserialize pairs with valid data (covers lines 157-162)."""
        arrays = {
            "pair_indices": np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64),
            "pair_values": np.array([0.5, 0.7, 0.9], dtype=np.float64),
        }
        result = _deserialize_pairs(arrays)
        assert result == {(0, 1): 0.5, (1, 2): 0.7, (2, 3): 0.9}


class TestFunctionAcceptsArgument:
    """Tests for _function_accepts_argument function."""

    def test_function_accepts_argument_true(self) -> None:
        """Test function accepts argument (covers lines 172-179)."""
        def test_func(a: int, b: int, param: str = "default") -> None:
            pass

        assert _function_accepts_argument(test_func, "param") is True
        assert _function_accepts_argument(test_func, "a") is True

    def test_function_accepts_argument_false(self) -> None:
        """Test function doesn't accept argument."""
        def test_func(a: int, b: int) -> None:
            pass

        assert _function_accepts_argument(test_func, "param") is False

    def test_function_accepts_argument_var_keyword(self) -> None:
        """Test function with **kwargs (covers lines 173-174)."""
        def test_func(a: int, **kwargs: Any) -> None:
            pass

        assert _function_accepts_argument(test_func, "any_param") is True

    def test_function_accepts_argument_invalid_signature(self) -> None:
        """Test function with invalid signature (covers lines 168-171)."""
        # Built-in functions may not have inspectable signatures
        assert _function_accepts_argument(len, "param") is False


class TestComputeSummaryStatistics:
    """Tests for _compute_summary_statistics function."""

    def test_compute_summary_statistics_mean(self) -> None:
        """Test summary statistics with mean (covers lines 557-559)."""
        dists = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _compute_summary_statistics(dists, summary="mean")
        assert result == pytest.approx(3.0)

    def test_compute_summary_statistics_std(self) -> None:
        """Test summary statistics with std (covers lines 560-561)."""
        dists = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _compute_summary_statistics(dists, summary="std")
        assert isinstance(result, float)
        assert result > 0

    def test_compute_summary_statistics_median(self) -> None:
        """Test summary statistics with median (covers lines 562-563)."""
        dists = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _compute_summary_statistics(dists, summary="median")
        assert result == pytest.approx(3.0)

    def test_compute_summary_statistics_all(self) -> None:
        """Test summary statistics with all (covers lines 564-571)."""
        dists = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _compute_summary_statistics(dists, summary="all")
        assert isinstance(result, dict)
        assert "mean" in result
        assert "std" in result
        assert "median" in result
        assert "min" in result
        assert "max" in result

    def test_compute_summary_statistics_invalid(self) -> None:
        """Test summary statistics with invalid summary (covers lines 572-575)."""
        dists = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="Unknown summary"):
            _compute_summary_statistics(dists, summary="invalid")


class TestModifyMatrix:
    """Tests for modify_matrix function."""

    def test_modify_matrix_basic(self) -> None:
        """Test modify_matrix with default parameters."""
        mtx = np.random.randn(50, 10)
        result = modify_matrix(mtx)
        assert result.shape == mtx.shape
        assert not np.allclose(result, mtx)

    def test_modify_matrix_no_whiten(self) -> None:
        """Test modify_matrix without whitening."""
        mtx = np.random.randn(50, 10)
        result = modify_matrix(mtx, whiten=False, normalize=True)
        assert result.shape == mtx.shape

    def test_modify_matrix_no_normalize(self) -> None:
        """Test modify_matrix without normalization."""
        mtx = np.random.randn(50, 10)
        result = modify_matrix(mtx, whiten=True, normalize=False)
        assert result.shape == mtx.shape

    def test_modify_matrix_unit_length_per_column(self) -> None:
        """Test modify_matrix with unit_length_per_column (covers lines 1176-1181)."""
        mtx = np.random.randn(50, 10)
        result = modify_matrix(mtx, whiten=True, unit_length_per_column=True, normalize=False)
        assert result.shape == mtx.shape
        # Check that columns have unit length
        column_norms = np.linalg.norm(result, axis=0)
        np.testing.assert_allclose(column_norms, 1.0, rtol=1e-10)

    def test_modify_matrix_no_scale_variance(self) -> None:
        """Test modify_matrix without scale_variance (covers line 1187)."""
        mtx = np.random.randn(50, 10)
        result = modify_matrix(mtx, whiten=True, scale_variance=False, normalize=True)
        assert result.shape == mtx.shape

    def test_modify_matrix_zero_norm_column(self) -> None:
        """Test modify_matrix with zero norm column (covers line 1180)."""
        mtx = np.random.randn(50, 10)
        mtx[:, 0] = 0  # Zero column
        result = modify_matrix(mtx, whiten=True, unit_length_per_column=True)
        assert result.shape == mtx.shape
        # Should not crash

    def test_modify_matrix_zero_std_column(self) -> None:
        """Test modify_matrix with zero std column (covers line 1185)."""
        mtx = np.random.randn(50, 10)
        mtx[:, 0] = 1.0  # Constant column (zero std after centering)
        result = modify_matrix(mtx, whiten=True, scale_variance=True)
        assert result.shape == mtx.shape

    def test_modify_matrix_zero_frobenius_norm(self) -> None:
        """Test modify_matrix with zero Frobenius norm (covers lines 1191-1193)."""
        mtx = np.zeros((50, 10))
        result = modify_matrix(mtx, whiten=False, normalize=True)
        assert result.shape == mtx.shape
        # Should not crash


class TestDistributionDistance:
    """Tests for distribution_distance function."""

    def test_distribution_distance_within_shape_metric_error(self) -> None:
        """Test distribution_distance with within mode and shape metric (covers lines 687-691)."""
        from neural_analysis.metrics.distributions import distribution_distance
        
        points1 = np.random.randn(50, 10)
        with pytest.raises(ValueError, match="Shape metric.*cannot be used with mode='within'"):
            distribution_distance(points1, mode="within", metric="procrustes")

    def test_distribution_distance_within_distribution_metric_error(self) -> None:
        """Test distribution_distance with within mode and distribution metric (covers lines 693-698)."""
        from neural_analysis.metrics.distributions import distribution_distance
        
        points1 = np.random.randn(50, 10)
        with pytest.raises(ValueError, match="Distribution metric.*cannot be used with mode='within'"):
            distribution_distance(points1, mode="within", metric="wasserstein")

    def test_distribution_distance_within_less_than_2_samples(self) -> None:
        """Test distribution_distance with within mode and <2 samples (covers lines 705-711)."""
        from neural_analysis.metrics.distributions import distribution_distance
        
        points1 = np.random.randn(1, 10)
        result = distribution_distance(points1, mode="within", metric="euclidean")
        assert result == 0.0

    def test_distribution_distance_within_summary_all(self) -> None:
        """Test distribution_distance with within mode and summary='all' (covers lines 707-710)."""
        from neural_analysis.metrics.distributions import distribution_distance
        
        points1 = np.random.randn(1, 10)
        result = distribution_distance(points1, mode="within", metric="euclidean", summary="all")
        assert isinstance(result, dict)
        assert result["mean"] == 0.0
        assert result["std"] == 0.0
        assert result["median"] == 0.0

    def test_distribution_distance_between_shape_metric(self) -> None:
        """Test distribution_distance with between mode and shape metric (covers lines 747-768)."""
        from neural_analysis.metrics.distributions import distribution_distance
        
        points1 = np.random.randn(50, 10)
        points2 = np.random.randn(50, 10)
        
        result = distribution_distance(
            points1, points2, mode="between", metric="procrustes"
        )
        # Should return tuple (distance, pairs)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_distribution_distance_between_wasserstein(self) -> None:
        """Test distribution_distance with between mode and wasserstein (covers lines 773-776)."""
        from neural_analysis.metrics.distributions import distribution_distance
        
        points1 = np.random.randn(50, 10)
        points2 = np.random.randn(50, 10)
        
        result = distribution_distance(
            points1, points2, mode="between", metric="wasserstein"
        )
        assert isinstance(result, float)

    def test_distribution_distance_between_wasserstein_summary_all(self) -> None:
        """Test distribution_distance with between mode, wasserstein, summary='all' (covers lines 787-791)."""
        from neural_analysis.metrics.distributions import distribution_distance
        
        points1 = np.random.randn(50, 10)
        points2 = np.random.randn(50, 10)
        
        result = distribution_distance(
            points1, points2, mode="between", metric="wasserstein", summary="all"
        )
        assert isinstance(result, dict)
        assert "mean" in result
        assert "std" in result
        assert "median" in result

    def test_distribution_distance_between_kolmogorov_smirnov(self) -> None:
        """Test distribution_distance with between mode and kolmogorov-smirnov (covers lines 777-780)."""
        from neural_analysis.metrics.distributions import distribution_distance
        
        points1 = np.random.randn(50, 10)
        points2 = np.random.randn(50, 10)
        
        result = distribution_distance(
            points1, points2, mode="between", metric="kolmogorov-smirnov"
        )
        assert isinstance(result, float)

    def test_distribution_distance_between_jensen_shannon(self) -> None:
        """Test distribution_distance with between mode and jensen-shannon (covers lines 781-784)."""
        from neural_analysis.metrics.distributions import distribution_distance
        
        points1 = np.random.randn(50, 10)
        points2 = np.random.randn(50, 10)
        
        result = distribution_distance(
            points1, points2, mode="between", metric="jensen-shannon"
        )
        assert isinstance(result, float)

    def test_distribution_distance_between_points2_required(self) -> None:
        """Test distribution_distance with between mode but no points2 (covers lines 734-735)."""
        from neural_analysis.metrics.distributions import distribution_distance
        
        points1 = np.random.randn(50, 10)
        with pytest.raises(ValueError, match="points2 is required when mode='between'"):
            distribution_distance(points1, mode="between", metric="euclidean")

    def test_distribution_distance_unknown_mode(self) -> None:
        """Test distribution_distance with unknown mode (covers lines 805-806)."""
        from neural_analysis.metrics.distributions import distribution_distance
        
        points1 = np.random.randn(50, 10)
        with pytest.raises(ValueError, match="Unknown mode"):
            distribution_distance(points1, mode="invalid", metric="euclidean")


class TestShapeDistanceProcrustes:
    """Tests for shape_distance_procrustes function."""

    def test_shape_distance_procrustes_different_shapes(self) -> None:
        """Test shape_distance_procrustes with different shapes (covers lines 1310-1311)."""
        from neural_analysis.metrics.distributions import shape_distance_procrustes
        
        mtx1 = np.random.randn(50, 10)
        mtx2 = np.random.randn(60, 10)
        
        with pytest.raises(ValueError, match="same shape"):
            shape_distance_procrustes(mtx1, mtx2)

    def test_shape_distance_procrustes_no_return_pairs(self) -> None:
        """Test shape_distance_procrustes with return_pairs=False (covers lines 1340-1341)."""
        from neural_analysis.metrics.distributions import shape_distance_procrustes
        
        mtx1 = np.random.randn(50, 10)
        mtx2 = np.random.randn(50, 10)
        
        dist, pairs = shape_distance_procrustes(mtx1, mtx2, return_pairs=False)
        assert isinstance(dist, float)
        assert pairs is None


class TestShapeDistanceSoftMatching:
    """Tests for shape_distance_soft_matching function."""

    def test_shape_distance_soft_matching_subsampling_error(self) -> None:
        """Test shape_distance_soft_matching with subsampling error (covers lines 1830-1833)."""
        from neural_analysis.metrics.distributions import shape_distance_soft_matching
        
        mtx1 = np.random.randn(50, 10)
        mtx2 = np.random.randn(50, 10)
        
        # Test with mismatched subsamples and subsample_axes
        with pytest.raises(ValueError, match="must have the same length"):
            shape_distance_soft_matching(
                mtx1, mtx2, subsamples=[10], subsample_axes=[0, 1]
            )

    def test_shape_distance_soft_matching_auto_subsampling(self) -> None:
        """Test shape_distance_soft_matching with auto subsampling (covers lines 1811-1817)."""
        from neural_analysis.metrics.distributions import shape_distance_soft_matching
        
        # Different neuron counts to trigger auto subsampling
        mtx1 = np.random.randn(100, 10)
        mtx2 = np.random.randn(50, 10)
        
        dist, pairs, meta = shape_distance_soft_matching(mtx1, mtx2)
        assert isinstance(dist, float)
        assert meta.get("auto_subsampling") is True


class TestComparisonResultsToDataFrame:
    """Tests for _comparison_results_to_dataframe function."""

    def test_comparison_results_to_dataframe_basic(self) -> None:
        """Test _comparison_results_to_dataframe basic (covers lines 1864-1906)."""
        from neural_analysis.metrics.distributions import _comparison_results_to_dataframe
        
        results = {
            "key1": {
                "attributes": {"metric": "wasserstein", "dataset_i": "A", "dataset_j": "B"},
                "arrays": {
                    "pair_indices": np.array([[0, 0], [1, 1]]),
                    "pair_values": np.array([0.5, 0.6]),
                },
            },
            "key2": {
                "attributes": {"metric": "euclidean", "dataset_i": "C", "dataset_j": "D"},
            },
        }
        
        df = _comparison_results_to_dataframe(results)
        assert len(df) == 2
        assert "pairs" in df.columns

    def test_comparison_results_to_dataframe_no_arrays(self) -> None:
        """Test _comparison_results_to_dataframe without arrays (covers lines 1901-1902)."""
        from neural_analysis.metrics.distributions import _comparison_results_to_dataframe
        
        results = {
            "key1": {
                "attributes": {"metric": "wasserstein"},
            },
        }
        
        df = _comparison_results_to_dataframe(results)
        assert len(df) == 1
        assert df.iloc[0]["pairs"] is None

    def test_comparison_results_to_dataframe_no_pair_indices(self) -> None:
        """Test _comparison_results_to_dataframe without pair_indices (covers lines 1899-1900)."""
        from neural_analysis.metrics.distributions import _comparison_results_to_dataframe
        
        results = {
            "key1": {
                "attributes": {"metric": "wasserstein"},
                "arrays": {
                    "other": np.array([1, 2, 3]),
                },
            },
        }
        
        df = _comparison_results_to_dataframe(results)
        assert len(df) == 1
        assert df.iloc[0]["pairs"] is None


class TestAlignMtx:
    """Tests for align_mtx function."""

    def test_align_mtx_different_shapes(self) -> None:
        """Test align_mtx with different shapes (covers lines 1230-1231)."""
        from neural_analysis.metrics.distributions import align_mtx
        
        mtx1 = np.random.randn(50, 10)
        mtx2 = np.random.randn(60, 10)
        
        with pytest.raises(ValueError, match="same shape"):
            align_mtx(mtx1, mtx2)

    def test_align_mtx_not_2d(self) -> None:
        """Test align_mtx with non-2D input (covers lines 1232-1233)."""
        from neural_analysis.metrics.distributions import align_mtx
        
        mtx1 = np.random.randn(50, 10, 5)
        mtx2 = np.random.randn(50, 10, 5)
        
        with pytest.raises(ValueError, match="two-dimensional"):
            align_mtx(mtx1, mtx2)

    def test_align_mtx_with_scale(self) -> None:
        """Test align_mtx with scale=True (covers lines 1251-1252)."""
        from neural_analysis.metrics.distributions import align_mtx
        
        mtx1 = np.random.randn(50, 10)
        mtx2 = np.random.randn(50, 10)
        
        result = align_mtx(mtx1, mtx2, rotate=True, scale=True)
        assert result.shape == mtx2.shape

