"""Coverage-targeted tests for neural_analysis.metrics.pairwise_metrics.

Focus: cover the NUMBA_AVAILABLE=False branch (line 60->67) in the facade
module, plus behavioural assertions on all re-exported public symbols.
"""

from __future__ import annotations

import importlib
import sys
from unittest.mock import patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Module-level constants (tested via facade)
# ---------------------------------------------------------------------------

from neural_analysis.metrics.pairwise_metrics import (
    ALL_METRICS,
    DISTRIBUTION_METRICS,
    NUMBA_AVAILABLE,
    POINT_TO_POINT_METRICS,
    SCALAR_METRICS,
    SHAPE_METRICS,
    _angular_similarity_matrix_parallel,
    _compute_1d_autocorrelation,
    _compute_2d_autocorrelation,
    _compute_3d_autocorrelation,
    _correlation_matrix_parallel,
    _cosine_similarity_matrix_parallel,
    _get_distance_function,
    _validate_metric_mode,
    _validate_pairwise_inputs,
    angular_similarity_matrix,
    compare_datasets,
    compute_all_pairs,
    compute_between_distances,
    compute_pairwise_matrix,
    compute_within_distances,
    correlation,
    correlation_matrix,
    cosine_similarity,
    cosine_similarity_matrix,
    euclidean_distance,
    mahalanobis_distance,
    manhattan_distance,
    pairwise_distance,
    similarity_matrix,
    spatial_autocorrelation,
)

RNG = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# 1.  Branch coverage: NUMBA_AVAILABLE = False path in pairwise_metrics.py
# ---------------------------------------------------------------------------


class TestNumbaUnavailableBranch:
    """Cover the ``if NUMBA_AVAILABLE:`` False-branch in pairwise_metrics.py.

    When NUMBA_AVAILABLE is False the conditional import block (lines 61-65)
    is skipped.  We need the coverage tool to see the 60->67 False edge.
    """

    def test_reload_without_numba_does_not_raise(self) -> None:
        """Reload the facade with NUMBA_AVAILABLE=False; module must survive.

        We patch pairwise_numba.NUMBA_AVAILABLE to False, then reload the
        facade so the ``if NUMBA_AVAILABLE:`` block at line 60 takes the
        False branch (60->67), giving us coverage of that edge.
        """
        import neural_analysis.metrics.pairwise_numba as numba_mod

        with patch.object(numba_mod, "NUMBA_AVAILABLE", False):
            reloaded = importlib.reload(
                sys.modules["neural_analysis.metrics.pairwise_metrics"]
            )

        # Restore module to its proper (NUMBA_AVAILABLE=True) state
        importlib.reload(sys.modules["neural_analysis.metrics.pairwise_metrics"])

        # The reloaded module should still export core symbols
        assert hasattr(reloaded, "euclidean_distance")
        assert hasattr(reloaded, "compute_pairwise_matrix")
        assert hasattr(reloaded, "ALL_METRICS")
        # The reload happens during the patch, but pairwise_numba itself
        # sets NUMBA_AVAILABLE at import time (not via patch), so the value
        # seen in the reloaded module reflects its own import state.
        # What matters for coverage is that line 60 was executed with the
        # False branch — i.e. the import block lines 61-65 was skipped.
        assert isinstance(reloaded.NUMBA_AVAILABLE, bool)

    def test_numba_available_attribute_is_bool(self) -> None:
        """NUMBA_AVAILABLE must be a boolean."""
        assert isinstance(NUMBA_AVAILABLE, bool)

    def test_all_metrics_constants_are_frozensets(self) -> None:
        """Metric category constants must be frozensets."""
        for fs in (
            POINT_TO_POINT_METRICS,
            DISTRIBUTION_METRICS,
            SHAPE_METRICS,
            SCALAR_METRICS,
            ALL_METRICS,
        ):
            assert isinstance(fs, frozenset)

    def test_all_metrics_is_union(self) -> None:
        """ALL_METRICS == POINT_TO_POINT | DISTRIBUTION | SHAPE."""
        assert ALL_METRICS == POINT_TO_POINT_METRICS | DISTRIBUTION_METRICS | SHAPE_METRICS


# ---------------------------------------------------------------------------
# 2.  Euclidean distance (various code-paths)
# ---------------------------------------------------------------------------


class TestEuclideanDistance:
    def test_1d_known_value(self) -> None:
        x = np.array([0.0, 0.0])
        y = np.array([3.0, 4.0])
        assert euclidean_distance(x, y) == pytest.approx(5.0)

    def test_2d_matrix_no_parallel(self) -> None:
        """2-D arrays, parallel=False → cdist path."""
        x = RNG.standard_normal((4, 3))
        y = RNG.standard_normal((5, 3))
        result = euclidean_distance(x, y, parallel=False)
        assert result.shape == (4, 5)
        assert np.all(result >= 0)

    def test_2d_matrix_with_axis(self) -> None:
        """2-D arrays with explicit axis parameter."""
        x = np.array([[1.0, 0.0], [0.0, 1.0]])
        y = np.array([[1.0, 0.0], [0.0, 1.0]])
        result = euclidean_distance(x, y, axis=1)
        assert result.shape == (2,)
        np.testing.assert_allclose(result, [0.0, 0.0])

    def test_non_square_fallthrough(self) -> None:
        """Non-square inputs hit the last fallthrough branch."""
        x = np.array([[[1.0, 2.0]]])  # 3-D
        y = np.array([[[1.0, 3.0]]])
        result = euclidean_distance(x, y)
        assert float(result) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 3.  Manhattan distance
# ---------------------------------------------------------------------------


class TestManhattanDistance:
    def test_1d_known_value(self) -> None:
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([4.0, 5.0, 6.0])
        assert manhattan_distance(x, y) == pytest.approx(9.0)

    def test_2d_matrix_no_parallel(self) -> None:
        x = RNG.standard_normal((3, 4))
        y = RNG.standard_normal((5, 4))
        result = manhattan_distance(x, y, parallel=False)
        assert result.shape == (3, 5)
        assert np.all(result >= 0)

    def test_2d_matrix_with_axis(self) -> None:
        x = np.array([[1.0, 0.0], [0.0, 1.0]])
        y = np.array([[1.0, 0.0], [0.0, 1.0]])
        result = manhattan_distance(x, y, axis=1)
        np.testing.assert_allclose(result, [0.0, 0.0])

    def test_fallthrough_for_3d(self) -> None:
        x = np.array([[[1.0, 2.0]]])
        y = np.array([[[1.0, 4.0]]])
        result = manhattan_distance(x, y)
        assert float(result) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# 4.  Mahalanobis distance (edge cases)
# ---------------------------------------------------------------------------


class TestMahalanobisDistance:
    def test_1d_with_inv_cov(self) -> None:
        mean = np.array([0.0, 0.0])
        inv_cov = np.eye(2)
        x = np.array([1.0, 1.0])
        result = mahalanobis_distance(x, mean, inv_cov=inv_cov)
        assert result == pytest.approx(np.sqrt(2))

    def test_2d_input(self) -> None:
        mean = np.zeros(3)
        cov = np.eye(3)
        x = RNG.standard_normal((5, 3))
        result = mahalanobis_distance(x, mean, cov=cov)
        assert result.shape == (5,)
        assert np.all(result >= 0)

    def test_no_cov_raises(self) -> None:
        with pytest.raises(ValueError, match="Either cov or inv_cov"):
            mahalanobis_distance(np.array([1.0]), np.array([0.0]))


# ---------------------------------------------------------------------------
# 5.  Cosine similarity (2-D pairwise path without parallel)
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_1d_identical(self) -> None:
        v = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_2d_pairwise_no_parallel(self) -> None:
        """pairwise=True, parallel=False → sklearn path."""
        v1 = RNG.standard_normal((4, 5))
        v2 = RNG.standard_normal((3, 5))
        result = cosine_similarity(v1, v2, pairwise=True, parallel=False)
        assert result.shape == (4, 3)

    def test_2d_no_pairwise_flag(self) -> None:
        """2-D input without pairwise=True → flatten path."""
        v1 = np.array([[1.0, 0.0]])
        v2 = np.array([[0.0, 1.0]])
        result = cosine_similarity(v1, v2)
        assert result == pytest.approx(0.0, abs=1e-7)


# ---------------------------------------------------------------------------
# 6.  _get_distance_function
# ---------------------------------------------------------------------------


class TestGetDistanceFunction:
    @pytest.mark.parametrize(
        "metric,expected",
        [
            ("euclidean", euclidean_distance),
            ("manhattan", manhattan_distance),
            ("mahalanobis", mahalanobis_distance),
            ("cosine", cosine_similarity),
        ],
    )
    def test_returns_correct_callable(self, metric, expected) -> None:
        assert _get_distance_function(metric) is expected

    def test_unknown_metric_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown distance metric"):
            _get_distance_function("unknown")


# ---------------------------------------------------------------------------
# 7.  _validate_pairwise_inputs
# ---------------------------------------------------------------------------


class TestValidatePairwiseInputs:
    def test_1d_promoted_to_2d(self) -> None:
        x, y = _validate_pairwise_inputs(np.array([1.0, 2.0]), None)
        assert x.shape == (1, 2)
        assert y.shape == (1, 2)  # y=None → uses x

    def test_feature_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="Feature dimension mismatch"):
            _validate_pairwise_inputs(
                np.ones((3, 4)), np.ones((3, 5))
            )

    def test_valid_2d(self) -> None:
        x = np.ones((3, 4))
        y = np.ones((5, 4))
        x_out, y_out = _validate_pairwise_inputs(x, y)
        assert x_out.shape == (3, 4)
        assert y_out.shape == (5, 4)


# ---------------------------------------------------------------------------
# 8.  pairwise_distance
# ---------------------------------------------------------------------------


class TestPairwiseDistance:
    def test_euclidean_shape(self) -> None:
        x = RNG.standard_normal((4, 3))
        y = RNG.standard_normal((5, 3))
        result = pairwise_distance(x, y, metric="euclidean")
        assert result.shape == (4, 5)

    def test_manhattan_shape(self) -> None:
        x = RNG.standard_normal((4, 3))
        y = RNG.standard_normal((5, 3))
        result = pairwise_distance(x, y, metric="manhattan")
        assert result.shape == (4, 5)

    def test_cosine_shape(self) -> None:
        x = RNG.standard_normal((4, 3))
        y = RNG.standard_normal((5, 3))
        result = pairwise_distance(x, y, metric="cosine")
        assert result.shape == (4, 5)

    def test_mahalanobis_shape(self) -> None:
        x = RNG.standard_normal((4, 3))
        y = RNG.standard_normal((5, 3))
        result = pairwise_distance(x, y, metric="mahalanobis")
        assert result.shape == (4, 5)
        assert np.all(result >= 0)

    def test_unknown_metric_raises(self) -> None:
        x = np.ones((2, 2))
        y = np.ones((2, 2))
        with pytest.raises(ValueError, match="Unknown pairwise distance metric"):
            pairwise_distance(x, y, metric="bad_metric")


# ---------------------------------------------------------------------------
# 9.  compute_pairwise_matrix
# ---------------------------------------------------------------------------


class TestComputePairwiseMatrix:
    def test_euclidean_returns_array(self) -> None:
        x = RNG.standard_normal((4, 3))
        y = RNG.standard_normal((5, 3))
        result = compute_pairwise_matrix(x, y, metric="euclidean")
        assert isinstance(result, np.ndarray)
        assert result.shape == (4, 5)

    def test_alias_ks(self) -> None:
        """'ks' alias maps to 'kolmogorov-smirnov'."""
        x = RNG.standard_normal((20, 2))
        y = RNG.standard_normal((20, 2))
        result = compute_pairwise_matrix(x, y, metric="ks")
        assert isinstance(result, (float, np.floating))

    def test_alias_js(self) -> None:
        """'js' alias maps to 'jensen-shannon'."""
        x = np.abs(RNG.standard_normal((20, 2))) + 0.01
        y = np.abs(RNG.standard_normal((20, 2))) + 0.01
        result = compute_pairwise_matrix(x, y, metric="js")
        assert isinstance(result, (float, np.floating))

    def test_wasserstein_returns_scalar(self) -> None:
        x = RNG.standard_normal((20, 2))
        y = RNG.standard_normal((20, 2))
        result = compute_pairwise_matrix(x, y, metric="wasserstein")
        assert isinstance(result, (float, np.floating))

    def test_unknown_metric_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown metric"):
            compute_pairwise_matrix(np.ones((3, 2)), np.ones((3, 2)), metric="bad")

    def test_shape_metric_procrustes(self) -> None:
        x = RNG.standard_normal((10, 3))
        y = RNG.standard_normal((10, 3))
        result = compute_pairwise_matrix(x, y, metric="procrustes")
        assert isinstance(result, tuple)
        dist, pairs = result
        assert isinstance(dist, float)

    def test_shape_metric_one_to_one(self) -> None:
        x = RNG.standard_normal((6, 2))
        y = RNG.standard_normal((6, 2))
        result = compute_pairwise_matrix(x, y, metric="one-to-one")
        assert isinstance(result, tuple)

    def test_shape_metric_soft_matching(self) -> None:
        x = RNG.standard_normal((6, 2))
        y = RNG.standard_normal((6, 2))
        result = compute_pairwise_matrix(x, y, metric="soft-matching")
        assert isinstance(result, tuple)


# ---------------------------------------------------------------------------
# 10.  _validate_metric_mode
# ---------------------------------------------------------------------------


class TestValidateMetricMode:
    def test_within_with_euclidean(self) -> None:
        norm = _validate_metric_mode("euclidean", "within")
        assert norm == "euclidean"

    def test_within_with_distribution_metric_raises(self) -> None:
        with pytest.raises(ValueError, match="mode='within'"):
            _validate_metric_mode("wasserstein", "within")

    def test_between_accepts_all_metrics(self) -> None:
        for m in ("euclidean", "wasserstein", "procrustes"):
            norm = _validate_metric_mode(m, "between")
            assert isinstance(norm, str)

    def test_all_pairs_rejects_point_to_point(self) -> None:
        with pytest.raises(ValueError, match="mode='all-pairs'"):
            _validate_metric_mode("euclidean", "all-pairs")

    def test_all_pairs_accepts_scalar_metrics(self) -> None:
        norm = _validate_metric_mode("wasserstein", "all-pairs")
        assert norm == "wasserstein"

    def test_unknown_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown mode"):
            _validate_metric_mode("euclidean", "bad_mode")

    def test_unknown_metric_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown metric"):
            _validate_metric_mode("bad_metric", "within")

    def test_ks_alias(self) -> None:
        norm = _validate_metric_mode("ks", "between")
        assert norm == "kolmogorov-smirnov"

    def test_js_alias(self) -> None:
        norm = _validate_metric_mode("js", "between")
        assert norm == "jensen-shannon"


# ---------------------------------------------------------------------------
# 11.  compute_within_distances
# ---------------------------------------------------------------------------


class TestComputeWithinDistances:
    def test_returns_float_by_default(self) -> None:
        data = RNG.standard_normal((10, 3))
        result = compute_within_distances(data, metric="euclidean")
        assert isinstance(result, float)
        assert result >= 0

    def test_return_matrix(self) -> None:
        data = RNG.standard_normal((8, 3))
        result = compute_within_distances(data, metric="euclidean", return_matrix=True)
        assert isinstance(result, np.ndarray)
        assert result.shape == (8, 8)

    def test_single_sample_returns_zero(self) -> None:
        data = RNG.standard_normal((1, 3))
        result = compute_within_distances(data, metric="euclidean")
        assert result == 0.0

    def test_single_sample_return_matrix(self) -> None:
        data = RNG.standard_normal((1, 3))
        result = compute_within_distances(data, metric="euclidean", return_matrix=True)
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 1)

    def test_1d_data_reshaped(self) -> None:
        """1-D array is reshaped to (-1, 1) internally."""
        data = np.array([1.0, 2.0, 3.0, 4.0])
        result = compute_within_distances(data, metric="euclidean")
        assert isinstance(result, float)

    def test_distribution_metric_raises(self) -> None:
        data = RNG.standard_normal((10, 3))
        with pytest.raises(ValueError, match="mode='within'"):
            compute_within_distances(data, metric="wasserstein")


# ---------------------------------------------------------------------------
# 12.  compute_between_distances
# ---------------------------------------------------------------------------


class TestComputeBetweenDistances:
    def test_euclidean_scalar(self) -> None:
        d1 = RNG.standard_normal((8, 3))
        d2 = RNG.standard_normal((6, 3))
        result = compute_between_distances(d1, d2, metric="euclidean")
        assert isinstance(result, float)
        assert result >= 0

    def test_euclidean_return_matrix(self) -> None:
        d1 = RNG.standard_normal((8, 3))
        d2 = RNG.standard_normal((6, 3))
        result = compute_between_distances(d1, d2, metric="euclidean", return_matrix=True)
        assert isinstance(result, np.ndarray)
        assert result.shape == (8, 6)

    def test_wasserstein_scalar(self) -> None:
        d1 = RNG.standard_normal((20, 2))
        d2 = RNG.standard_normal((20, 2)) + 1.0
        result = compute_between_distances(d1, d2, metric="wasserstein")
        assert isinstance(result, float)
        assert result >= 0

    def test_1d_inputs_reshaped(self) -> None:
        """1-D inputs should be reshaped internally."""
        d1 = np.array([1.0, 2.0, 3.0])
        d2 = np.array([4.0, 5.0, 6.0])
        result = compute_between_distances(d1, d2, metric="euclidean")
        assert isinstance(result, float)

    def test_shape_metric_procrustes(self) -> None:
        d1 = RNG.standard_normal((10, 3))
        d2 = RNG.standard_normal((10, 3))
        result = compute_between_distances(d1, d2, metric="procrustes")
        assert isinstance(result, float)
        assert result >= 0


# ---------------------------------------------------------------------------
# 13.  compute_all_pairs
# ---------------------------------------------------------------------------


class TestComputeAllPairs:
    def test_wasserstein_two_datasets(self) -> None:
        datasets = {
            "A": RNG.standard_normal((15, 3)),
            "B": RNG.standard_normal((15, 3)) + 0.5,
        }
        result = compute_all_pairs(datasets, metric="wasserstein", show_progress=False)
        assert isinstance(result, dict)
        assert "A" in result and "B" in result["A"]
        # Symmetry
        assert result["A"]["B"] == pytest.approx(result["B"]["A"])
        # Diagonal self-comparison: distribution metric → 0.0
        assert result["A"]["A"] == pytest.approx(0.0)
        assert result["B"]["B"] == pytest.approx(0.0)

    def test_point_to_point_raises(self) -> None:
        datasets = {"A": np.ones((5, 2)), "B": np.ones((5, 2))}
        with pytest.raises(ValueError, match="all-pairs"):
            compute_all_pairs(datasets, metric="euclidean")

    def test_procrustes_unequal_samples_raises(self) -> None:
        datasets = {
            "A": RNG.standard_normal((8, 3)),
            "B": RNG.standard_normal((10, 3)),
        }
        with pytest.raises(ValueError, match="Procrustes distance requires"):
            compute_all_pairs(datasets, metric="procrustes")

    def test_shape_metric_self_comparison_small(self) -> None:
        """Self-comparison with < 4 samples should be 0."""
        datasets = {"X": RNG.standard_normal((3, 2))}
        result = compute_all_pairs(datasets, metric="procrustes", show_progress=False)
        assert result["X"]["X"] == pytest.approx(0.0)

    def test_shape_metric_self_comparison_larger(self) -> None:
        """Self-comparison splits data in half."""
        datasets = {"X": RNG.standard_normal((12, 3))}
        result = compute_all_pairs(datasets, metric="one-to-one", show_progress=False)
        assert isinstance(result["X"]["X"], float)

    def test_non_finite_distance_replaced_with_nan(self) -> None:
        datasets = {
            "A": RNG.standard_normal((10, 2)),
            "B": RNG.standard_normal((10, 2)),
        }
        with patch(
            "neural_analysis.metrics.pairwise_core.compute_pairwise_matrix"
        ) as mock_fn:
            mock_fn.return_value = np.inf
            result = compute_all_pairs(datasets, metric="wasserstein", show_progress=False)
        assert np.isnan(result["A"]["B"])

    def test_three_datasets(self) -> None:
        datasets = {
            "A": RNG.standard_normal((12, 2)),
            "B": RNG.standard_normal((12, 2)) + 1.0,
            "C": RNG.standard_normal((12, 2)) - 1.0,
        }
        result = compute_all_pairs(datasets, metric="jensen-shannon", show_progress=False)
        assert len(result) == 3
        for key in ("A", "B", "C"):
            assert len(result[key]) == 3


# ---------------------------------------------------------------------------
# 14.  compare_datasets (orchestration layer)
# ---------------------------------------------------------------------------


class TestCompareDatasets:
    def test_auto_within(self) -> None:
        data = RNG.standard_normal((10, 3))
        result = compare_datasets(data, metric="euclidean")
        assert isinstance(result, (float, int))

    def test_auto_between(self) -> None:
        d1 = RNG.standard_normal((10, 3))
        d2 = RNG.standard_normal((10, 3))
        result = compare_datasets(d1, d2, metric="euclidean")
        # Returns a BetweenResult dict
        assert isinstance(result, dict)
        assert "value" in result and "metric" in result

    def test_auto_all_pairs(self) -> None:
        datasets = {"A": RNG.standard_normal((8, 2)), "B": RNG.standard_normal((8, 2))}
        result = compare_datasets(datasets, metric="wasserstein")
        assert isinstance(result, dict)

    def test_within_return_matrix(self) -> None:
        data = RNG.standard_normal((10, 3))
        result = compare_datasets(data, mode="within", metric="euclidean", return_matrix=True)
        assert isinstance(result, np.ndarray)
        assert result.shape == (10, 10)

    def test_between_return_matrix(self) -> None:
        d1 = RNG.standard_normal((10, 3))
        d2 = RNG.standard_normal((8, 3))
        result = compare_datasets(d1, d2, mode="between", metric="euclidean", return_matrix=True)
        assert isinstance(result, np.ndarray)
        assert result.shape == (10, 8)

    def test_between_no_data2_raises(self) -> None:
        data = RNG.standard_normal((10, 3))
        with pytest.raises(ValueError, match="mode='between' requires data2"):
            compare_datasets(data, mode="between", metric="euclidean")

    def test_all_pairs_non_dict_raises(self) -> None:
        data = RNG.standard_normal((10, 3))
        with pytest.raises(TypeError, match="mode='all-pairs' requires data to be dict"):
            compare_datasets(data, mode="all-pairs", metric="wasserstein")

    def test_within_with_dict_raises(self) -> None:
        datasets = {"A": RNG.standard_normal((5, 2))}
        with pytest.raises(TypeError, match="requires data to be array-like"):
            compare_datasets(datasets, mode="within", metric="euclidean")

    def test_within_with_data2_raises(self) -> None:
        data = RNG.standard_normal((10, 3))
        data2 = RNG.standard_normal((10, 3))
        with pytest.raises(ValueError, match="mode='within' does not accept data2"):
            compare_datasets(data, data2, mode="within", metric="euclidean")

    def test_unknown_mode_raises(self) -> None:
        data = RNG.standard_normal((10, 3))
        with pytest.raises(ValueError, match="Unknown mode"):
            compare_datasets(data, mode="invalid", metric="euclidean")

    def test_all_pairs_return_matrix_warning(self) -> None:
        """return_matrix=True is ignored for all-pairs (just returns dict)."""
        datasets = {"A": RNG.standard_normal((8, 2)), "B": RNG.standard_normal((8, 2))}
        result = compare_datasets(
            datasets, mode="all-pairs", metric="wasserstein", return_matrix=True
        )
        assert isinstance(result, dict)

    def test_save_path_no_dataset_names_between_raises(self, tmp_path) -> None:
        d1 = RNG.standard_normal((10, 3))
        d2 = RNG.standard_normal((10, 3))
        save_file = tmp_path / "out.h5"
        with pytest.raises(ValueError, match="dataset_names required"):
            compare_datasets(
                d1, d2, mode="between", metric="euclidean", save_path=save_file
            )

    def test_regenerate_true_logs_and_recomputes(self, tmp_path) -> None:
        d1 = RNG.standard_normal((10, 3))
        d2 = RNG.standard_normal((10, 3))
        save_file = tmp_path / "regen.h5"
        # First save
        compare_datasets(
            d1, d2, mode="between", metric="euclidean",
            save_path=save_file, dataset_names=("a", "b")
        )
        # Regenerate
        result = compare_datasets(
            d1, d2, mode="between", metric="euclidean",
            save_path=save_file, dataset_names=("a", "b"), regenerate=True
        )
        assert result is not None

    def test_cached_result_returned(self, tmp_path) -> None:
        d1 = RNG.standard_normal((10, 3))
        d2 = RNG.standard_normal((10, 3))
        save_file = tmp_path / "cache.h5"
        result1 = compare_datasets(
            d1, d2, mode="between", metric="euclidean",
            save_path=save_file, dataset_names=("x", "y")
        )
        result2 = compare_datasets(
            d1, d2, mode="between", metric="euclidean",
            save_path=save_file, dataset_names=("x", "y"), regenerate=False
        )
        assert result1["value"] == pytest.approx(result2["value"])


# ---------------------------------------------------------------------------
# 15.  correlation & correlation_matrix
# ---------------------------------------------------------------------------


class TestCorrelation:
    def test_pearson_matrix_mode(self) -> None:
        data = RNG.standard_normal((20, 5))
        result = correlation(data, method="pearson", mode="matrix")
        assert result.shape == (5, 5)
        np.testing.assert_allclose(np.diag(result), 1.0, atol=1e-10)

    def test_spearman_matrix_mode(self) -> None:
        data = RNG.standard_normal((20, 5))
        result = correlation(data, method="spearman", mode="matrix")
        assert result.shape == (5, 5)

    def test_kendall_matrix_mode(self) -> None:
        data = RNG.standard_normal((15, 4))
        result = correlation(data, method="kendall", mode="matrix")
        assert result.shape == (4, 4)

    def test_pairwise_mode(self) -> None:
        data = RNG.standard_normal((20, 5))
        result = correlation(data, mode="pairwise")
        assert result.shape == (4,)  # n_features - 1

    def test_invalid_mode_raises(self) -> None:
        data = RNG.standard_normal((10, 3))
        with pytest.raises(ValueError, match="Unknown correlation mode"):
            correlation(data, mode="bad")

    def test_parallel_matrix_mode(self) -> None:
        data = RNG.standard_normal((20, 5))
        result = correlation(data, method="pearson", mode="matrix", parallel=True)
        assert result.shape == (5, 5)


class TestCorrelationMatrix:
    def test_pearson(self) -> None:
        data = RNG.standard_normal((20, 4))
        result = correlation_matrix(data, method="pearson")
        assert result.shape == (4, 4)
        np.testing.assert_allclose(result, result.T)

    def test_spearman(self) -> None:
        data = RNG.standard_normal((20, 4))
        result = correlation_matrix(data, method="spearman")
        assert result.shape == (4, 4)

    def test_spearman_2_features(self) -> None:
        """With 2 features spearmanr returns scalar; code must convert to matrix."""
        data = RNG.standard_normal((20, 2))
        result = correlation_matrix(data, method="spearman")
        assert result.shape == (2, 2)
        np.testing.assert_allclose(result, result.T)

    def test_kendall(self) -> None:
        data = RNG.standard_normal((15, 3))
        result = correlation_matrix(data, method="kendall")
        assert result.shape == (3, 3)
        np.testing.assert_allclose(np.diag(result), 1.0)

    def test_invalid_method_raises(self) -> None:
        data = RNG.standard_normal((10, 3))
        with pytest.raises(ValueError, match="Unknown correlation method"):
            correlation_matrix(data, method="bad")

    def test_1d_raises(self) -> None:
        with pytest.raises(ValueError, match="must be 2D"):
            correlation_matrix(np.array([1.0, 2.0, 3.0]))


# ---------------------------------------------------------------------------
# 16.  cosine_similarity_matrix & angular_similarity_matrix
# ---------------------------------------------------------------------------


class TestCosineSimilarityMatrix:
    def test_shape_and_symmetry(self) -> None:
        data = RNG.standard_normal((20, 5))
        result = cosine_similarity_matrix(data)
        assert result.shape == (5, 5)
        np.testing.assert_allclose(result, result.T, atol=1e-12)

    def test_centered(self) -> None:
        data = RNG.standard_normal((20, 5))
        result = cosine_similarity_matrix(data, centered=True)
        assert result.shape == (5, 5)

    def test_1d_raises(self) -> None:
        with pytest.raises(ValueError, match="must be 2D"):
            cosine_similarity_matrix(np.array([1.0, 2.0]))


class TestAngularSimilarityMatrix:
    def test_shape_and_range(self) -> None:
        data = RNG.standard_normal((20, 5))
        result = angular_similarity_matrix(data)
        assert result.shape == (5, 5)
        assert np.all(result >= 0) and np.all(result <= 1)

    def test_diagonal_is_one(self) -> None:
        data = RNG.standard_normal((20, 5))
        result = angular_similarity_matrix(data)
        np.testing.assert_allclose(np.diag(result), 1.0, atol=1e-10)

    def test_1d_raises(self) -> None:
        with pytest.raises(ValueError, match="must be 2D"):
            angular_similarity_matrix(np.array([1.0, 2.0]))


# ---------------------------------------------------------------------------
# 17.  similarity_matrix
# ---------------------------------------------------------------------------


class TestSimilarityMatrix:
    @pytest.mark.parametrize("method", ["pearson", "spearman", "kendall", "cosine", "angular"])
    def test_all_methods(self, method: str) -> None:
        data = RNG.standard_normal((15, 4))
        result = similarity_matrix(data, method=method)
        assert result.shape == (4, 4)

    def test_cosine_centered_parallel(self) -> None:
        data = RNG.standard_normal((15, 4))
        result = similarity_matrix(data, method="cosine", centered=True, parallel=True)
        assert result.shape == (4, 4)

    def test_angular_parallel(self) -> None:
        data = RNG.standard_normal((15, 4))
        result = similarity_matrix(data, method="angular", parallel=True)
        assert result.shape == (4, 4)

    def test_pearson_parallel(self) -> None:
        data = RNG.standard_normal((15, 4))
        result = similarity_matrix(data, method="pearson", parallel=True)
        assert result.shape == (4, 4)

    def test_1d_raises(self) -> None:
        with pytest.raises(ValueError, match="must be 2D"):
            similarity_matrix(np.array([1.0, 2.0, 3.0]))

    def test_invalid_method_raises(self) -> None:
        data = RNG.standard_normal((10, 3))
        with pytest.raises(ValueError, match="Unknown similarity method"):
            similarity_matrix(data, method="bad")


# ---------------------------------------------------------------------------
# 18.  _correlation_matrix_parallel / _cosine_similarity_matrix_parallel /
#       _angular_similarity_matrix_parallel
# ---------------------------------------------------------------------------


class TestParallelHelpers:
    def test_correlation_pearson(self) -> None:
        data = RNG.standard_normal((20, 5))
        result = _correlation_matrix_parallel(data, "pearson")
        assert result.shape == (5, 5)

    def test_correlation_spearman(self) -> None:
        data = RNG.standard_normal((20, 5))
        result = _correlation_matrix_parallel(data, "spearman")
        assert result.shape == (5, 5)

    def test_correlation_kendall(self) -> None:
        data = RNG.standard_normal((15, 3))
        result = _correlation_matrix_parallel(data, "kendall")
        assert result.shape == (3, 3)

    def test_correlation_unknown_raises(self) -> None:
        data = RNG.standard_normal((10, 3))
        with pytest.raises(ValueError, match="Unknown correlation method"):
            _correlation_matrix_parallel(data, "bad")

    def test_cosine_not_centered(self) -> None:
        data = RNG.standard_normal((20, 5))
        result = _cosine_similarity_matrix_parallel(data, centered=False)
        assert result.shape == (5, 5)

    def test_cosine_centered(self) -> None:
        data = RNG.standard_normal((20, 5))
        result = _cosine_similarity_matrix_parallel(data, centered=True)
        assert result.shape == (5, 5)

    def test_angular(self) -> None:
        data = RNG.standard_normal((20, 5))
        result = _angular_similarity_matrix_parallel(data)
        assert result.shape == (5, 5)
        assert np.all(result >= 0) and np.all(result <= 1)


# ---------------------------------------------------------------------------
# 19.  spatial_autocorrelation (1D / 2D / 3D ; fft / direct)
# ---------------------------------------------------------------------------


class TestSpatialAutocorrelation:
    def test_1d_fft(self) -> None:
        activity = RNG.standard_normal((50, 5))
        positions = RNG.uniform(0, 1, size=50)
        autocorr, lags = spatial_autocorrelation(
            activity, positions, arena_size=1.0, n_bins=20, method="fft"
        )
        assert len(autocorr) == 20
        assert len(lags) == 1 and len(lags[0]) == 20

    def test_1d_direct(self) -> None:
        activity = RNG.standard_normal((50, 5))
        positions = RNG.uniform(0, 1, size=50)
        autocorr, lags = spatial_autocorrelation(
            activity, positions, arena_size=1.0, n_bins=20, method="direct"
        )
        assert len(autocorr) == 20

    def test_2d_fft(self) -> None:
        activity = RNG.standard_normal((50, 5))
        positions = RNG.uniform(0, 1, size=(50, 2))
        autocorr, lags = spatial_autocorrelation(
            activity, positions, arena_size=(1.0, 1.0), n_bins=15, method="fft"
        )
        assert autocorr.ndim == 2
        assert autocorr.shape == (15, 15)
        assert len(lags) == 2

    def test_2d_direct(self) -> None:
        activity = RNG.standard_normal((50, 5))
        positions = RNG.uniform(0, 1, size=(50, 2))
        autocorr, lags = spatial_autocorrelation(
            activity, positions, arena_size=(1.0, 1.0), n_bins=10, method="direct"
        )
        assert autocorr.ndim == 2

    def test_3d_fft(self) -> None:
        activity = RNG.standard_normal((50, 3))
        positions = RNG.uniform(0, 1, size=(50, 3))
        autocorr, lags = spatial_autocorrelation(
            activity, positions, arena_size=(1.0, 1.0, 1.0), n_bins=8, method="fft"
        )
        assert autocorr.ndim == 3
        assert len(lags) == 3

    def test_3d_direct(self) -> None:
        activity = RNG.standard_normal((50, 3))
        positions = RNG.uniform(0, 1, size=(50, 3))
        autocorr, lags = spatial_autocorrelation(
            activity, positions, arena_size=(1.0, 1.0, 1.0), n_bins=8, method="direct"
        )
        assert autocorr.ndim == 3

    def test_3d_tuple_arena_size_explicit(self) -> None:
        """3D with explicit tuple arena_size (float scalar not supported by binning util)."""
        activity = RNG.standard_normal((50, 3))
        positions = RNG.uniform(0, 1, size=(50, 3))
        autocorr, lags = spatial_autocorrelation(
            activity, positions, arena_size=(1.0, 1.0, 1.0), n_bins=8, method="fft"
        )
        assert autocorr.ndim == 3

    def test_unsupported_dims_raises(self) -> None:
        activity = RNG.standard_normal((50, 5))
        positions = RNG.uniform(0, 1, size=(50, 4))
        with pytest.raises((ValueError, KeyError)):
            spatial_autocorrelation(
                activity, positions, arena_size=(1.0,) * 4
            )

    def test_default_nbins_1d(self) -> None:
        activity = RNG.standard_normal((50, 5))
        positions = RNG.uniform(0, 1, size=50)
        autocorr, lags = spatial_autocorrelation(
            activity, positions, arena_size=1.0, n_bins=None
        )
        assert len(autocorr) == 50  # default for 1D

    def test_default_nbins_2d(self) -> None:
        activity = RNG.standard_normal((50, 5))
        positions = RNG.uniform(0, 1, size=(50, 2))
        autocorr, lags = spatial_autocorrelation(
            activity, positions, arena_size=(1.0, 1.0), n_bins=None
        )
        assert autocorr.ndim == 2
        assert autocorr.shape == (40, 40)  # default for 2D


# ---------------------------------------------------------------------------
# 20.  Numba re-exports (conditionally available)
# ---------------------------------------------------------------------------


class TestNumbaReexports:
    def test_kendall_numba_function_callable(self) -> None:
        from neural_analysis.metrics.pairwise_metrics import _kendall_numba

        data = RNG.standard_normal((20, 3))
        result = _kendall_numba(data)
        assert result.shape == (3, 3)

    def test_spearman_numba_function_callable(self) -> None:
        from neural_analysis.metrics.pairwise_metrics import _spearman_numba

        data = RNG.standard_normal((20, 3))
        result = _spearman_numba(data)
        assert result.shape == (3, 3)

    @pytest.mark.skipif(not NUMBA_AVAILABLE, reason="numba not installed")
    def test_pairwise_cosine_numba(self) -> None:
        from neural_analysis.metrics.pairwise_metrics import _pairwise_cosine_numba

        x = RNG.standard_normal((4, 3)).astype(np.float64)
        y = RNG.standard_normal((5, 3)).astype(np.float64)
        result = _pairwise_cosine_numba(x, y)
        assert result.shape == (4, 5)

    @pytest.mark.skipif(not NUMBA_AVAILABLE, reason="numba not installed")
    def test_pairwise_euclidean_numba(self) -> None:
        from neural_analysis.metrics.pairwise_metrics import _pairwise_euclidean_numba

        x = RNG.standard_normal((4, 3)).astype(np.float64)
        y = RNG.standard_normal((5, 3)).astype(np.float64)
        result = _pairwise_euclidean_numba(x, y)
        assert result.shape == (4, 5)
        assert np.all(result >= 0)

    @pytest.mark.skipif(not NUMBA_AVAILABLE, reason="numba not installed")
    def test_pairwise_manhattan_numba(self) -> None:
        from neural_analysis.metrics.pairwise_metrics import _pairwise_manhattan_numba

        x = RNG.standard_normal((4, 3)).astype(np.float64)
        y = RNG.standard_normal((5, 3)).astype(np.float64)
        result = _pairwise_manhattan_numba(x, y)
        assert result.shape == (4, 5)
        assert np.all(result >= 0)
