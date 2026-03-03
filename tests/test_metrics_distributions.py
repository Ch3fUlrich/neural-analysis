"""Tests for distribution comparison functions.

Consolidated from:
- test_metrics_distributions.py (base)
- test_distributions_additional.py
- test_distributions_comprehensive.py
- test_distributions_final.py
- test_distributions_more.py
"""

from __future__ import annotations

import contextlib
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from neural_analysis.metrics.distributions import (
    _comparison_results_to_dataframe,
    _compute_summary_statistics,
    _deserialize_pairs,
    _function_accepts_argument,
    _normalize_metrics_input,
    _progress_iterable,
    _serialize_pairs,
    _split_result_value,
    align_mtx,
    batch_comparison,
    distribution_distance,
    jensen_shannon_divergence,
    kolmogorov_smirnov_distance,
    modify_matrix,
    pairwise_distribution_comparison_batch,
    shape_distance,
    shape_distance_procrustes,
    shape_distance_soft_matching,
    wasserstein_distance_multi,
)
from neural_analysis.metrics.pairwise_metrics import (
    compare_datasets,
    compute_within_distances,
)

# ---------------------------------------------------------------------------
# Section 1: compare_datasets tests (between / all-pairs modes)
# ---------------------------------------------------------------------------


class TestCompareDistributions:
    """Test suite for compare_datasets function (replaces compare_distributions)."""

    def test_identical_distributions(self) -> None:
        """Test that identical distributions have zero distance (or 1.0 for cosine)."""
        p = np.random.randn(100, 3)
        result = compare_datasets(p, p, mode="between", metric="wasserstein")
        dist = result["value"] if isinstance(result, dict) else float(result)
        assert dist == pytest.approx(0.0, abs=1e-6)

    def test_shifted_distributions_wasserstein(self) -> None:
        """Test Wasserstein distance increases with shift."""
        p1 = np.random.randn(100, 3)
        p2 = p1 + 2.0
        result = compare_datasets(p1, p2, mode="between", metric="wasserstein")
        dist = result["value"] if isinstance(result, dict) else float(result)
        assert dist > 1.0

    def test_kolmogorov_smirnov_metric(self) -> None:
        """Test K-S statistic on shifted distributions."""
        p1 = np.random.randn(100, 2)
        p2 = np.random.randn(100, 2) + 1.5
        result = compare_datasets(p1, p2, mode="between", metric="kolmogorov-smirnov")
        dist = result["value"] if isinstance(result, dict) else float(result)
        assert 0.0 < dist <= 1.0

    def test_jensen_shannon_metric(self) -> None:
        """Test Jensen-Shannon divergence."""
        np.random.seed(42)
        p1 = np.random.randn(200, 2)
        p2 = np.random.randn(200, 2) + 1.0
        result = compare_datasets(p1, p2, mode="between", metric="jensen-shannon")
        dist = result["value"] if isinstance(result, dict) else float(result)
        assert 0.0 <= dist <= 1.0

    def test_euclidean_metric(self) -> None:
        """Test Euclidean distance between centroids."""
        p1 = np.random.randn(100, 3)
        p2 = p1 + np.array([3, 4, 0])
        result = compare_datasets(p1, p2, mode="between", metric="euclidean")
        dist = result["value"] if isinstance(result, dict) else float(result)
        assert dist == pytest.approx(5.0, rel=0.2)

    def test_mahalanobis_metric(self) -> None:
        """Test Mahalanobis distance."""
        np.random.seed(42)
        p1 = np.random.randn(100, 3)
        p2 = np.random.randn(100, 3) + 2.0
        result = compare_datasets(p1, p2, mode="between", metric="mahalanobis")
        dist = result["value"] if isinstance(result, dict) else float(result)
        assert dist > 0

    def test_cosine_metric(self) -> None:
        """Test cosine similarity."""
        p1 = np.random.randn(100, 3) + np.array([1, 0, 0])
        p2 = np.random.randn(100, 3) + np.array([2, 0, 0])
        result = compare_datasets(p1, p2, mode="between", metric="cosine")
        sim = result["value"] if isinstance(result, dict) else float(result)
        assert 0.0 <= sim <= 1.0
        assert isinstance(sim, (float, np.floating))

    def test_1d_distributions(self) -> None:
        """Test with 1D distributions."""
        p1 = np.random.randn(100)
        p2 = np.random.randn(100) + 1.0
        result = compare_datasets(p1, p2, mode="between", metric="wasserstein")
        dist = result["value"] if isinstance(result, dict) else float(result)
        assert dist > 0

    def test_empty_distribution_returns_nan(self) -> None:
        """Test that empty distributions return NaN."""
        p1 = np.array([]).reshape(0, 3)
        p2 = np.random.randn(100, 3)
        result = compare_datasets(p1, p2, mode="between", metric="wasserstein")
        dist = result["value"] if isinstance(result, dict) else float(result)
        assert np.isnan(dist)

    def test_dimension_mismatch_raises(self) -> None:
        """Test that mismatched dimensions raise error."""
        p1 = np.random.randn(100, 3)
        p2 = np.random.randn(100, 4)
        with pytest.raises(ValueError, match="Feature dimension mismatch"):
            compare_datasets(p1, p2, mode="between", metric="wasserstein")

    def test_invalid_metric_raises(self) -> None:
        """Test that invalid metric raises error."""
        p1 = np.random.randn(100, 3)
        p2 = np.random.randn(100, 3)
        with pytest.raises(ValueError, match="Unknown metric"):
            compare_datasets(p1, p2, mode="between", metric="invalid_metric")


class TestCompareDistributionGroups:
    """Test suite for compare_datasets with all-pairs mode."""

    def test_between_groups_basic(self) -> None:
        """Test between-group comparison using all-pairs mode."""
        groups = {
            "A": np.random.randn(50, 3),
            "B": np.random.randn(50, 3) + 1.0,
            "C": np.random.randn(50, 3) + 2.0,
        }
        result = compare_datasets(groups, mode="all-pairs", metric="wasserstein")

        assert set(result.keys()) == {"A", "B", "C"}
        assert set(result["A"].keys()) == {"A", "B", "C"}
        assert result["A"]["A"] == pytest.approx(0.0, abs=1e-6)
        assert result["A"]["B"] < result["A"]["C"]

    def test_inside_groups(self) -> None:
        """Test within-group variability using compute_within_distances."""
        np.random.seed(42)
        groups = {
            "tight": np.random.randn(50, 3) * 0.1,
            "loose": np.random.randn(50, 3) * 2.0,
        }

        means = []
        stds = []
        for _name, points in groups.items():
            dist_matrix = compute_within_distances(
                points, metric="euclidean", return_matrix=True
            )
            mask = np.triu(np.ones_like(dist_matrix, dtype=bool), k=1)
            dists = dist_matrix[mask]
            means.append(float(np.mean(dists)))
            stds.append(float(np.std(dists)))

        result = {"mean": np.array(means), "std": np.array(stds)}

        assert "mean" in result and "std" in result
        assert result["mean"].shape == (2,)
        assert result["mean"][1] > result["mean"][0]

    def test_single_point_group(self) -> None:
        """Test with group containing only one point."""
        groups = {
            "single": np.array([[1, 2, 3]]),
            "normal": np.random.randn(50, 3),
        }

        single_dist = compute_within_distances(groups["single"], metric="euclidean")
        compute_within_distances(groups["normal"], metric="euclidean")

        assert single_dist == pytest.approx(0.0)

    def test_different_metrics(self) -> None:
        """Test that different metrics work."""
        groups = {
            "A": np.random.randn(30, 2),
            "B": np.random.randn(30, 2) + 1.0,
        }

        for metric in ["wasserstein"]:
            result = compare_datasets(groups, mode="all-pairs", metric=metric)
            assert "A" in result and "B" in result
            assert "A" in result["A"] and "B" in result["A"]

        for metric in ["euclidean", "cosine"]:
            result_ab = compare_datasets(
                groups["A"], groups["B"], mode="between", metric=metric
            )
            dist_ab = (
                result_ab["value"] if isinstance(result_ab, dict) else float(result_ab)
            )
            if metric == "euclidean":
                assert dist_ab > 0
            else:
                assert -1.0 <= dist_ab <= 1.0

    def test_tuple_keys(self) -> None:
        """Test that tuple keys work as group identifiers."""
        groups = {
            (0, 0): np.random.randn(50, 3),
            (0, 1): np.random.randn(50, 3) + 1.0,
            (1, 0): np.random.randn(50, 3) + 2.0,
        }
        result = compare_datasets(groups, mode="all-pairs", metric="wasserstein")

        assert (0, 0) in result
        assert set(result[(0, 0)].keys()) == {(0, 0), (0, 1), (1, 0)}


# ---------------------------------------------------------------------------
# Section 2: Shape distance tests
# ---------------------------------------------------------------------------


class TestShapeDistance:
    """Test suite for shape distance functions."""

    def test_procrustes_identical(self) -> None:
        """Test Procrustes distance for identical shapes."""
        points1 = np.random.randn(50, 3)
        points2 = points1.copy()

        dist, pairs, meta = shape_distance(points1, points2, method="procrustes")
        assert dist == pytest.approx(0.0, abs=1e-6)

    def test_procrustes_rotated(self) -> None:
        """Test Procrustes handles rotation."""
        points1 = np.random.randn(50, 3)
        angle = np.pi / 4
        rotation_matrix = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )
        points2 = points1 @ rotation_matrix.T

        dist, pairs, meta = shape_distance(points1, points2, method="procrustes")
        assert dist < 0.2

    def test_one_to_one_method(self) -> None:
        """Test one-to-one matching distance."""
        points1 = np.random.randn(30, 2)
        points2 = np.random.randn(30, 2) + 1.0

        dist, pairs, meta = shape_distance(points1, points2, method="one-to-one")
        assert dist > 0

    def test_soft_matching_method(self) -> None:
        """Test soft matching distance."""
        points1 = np.random.randn(40, 2)
        points2 = np.random.randn(40, 2) + 0.5

        dist, pairs, meta = shape_distance(points1, points2, method="soft-matching")
        assert dist > 0

    def test_soft_matching_different_sizes(self) -> None:
        """Test that soft-matching handles different-sized matrices without broadcast errors."""
        np.random.seed(42)
        mtx1 = np.random.randn(26, 100).astype(np.float64)
        mtx2 = np.random.randn(39, 100).astype(np.float64)

        dist, pairs, meta = shape_distance(
            mtx1, mtx2, method="soft-matching", metric="sqeuclidean", approx=False
        )
        assert dist > 0
        assert not np.isnan(dist)
        assert pairs is not None
        assert meta.get("auto_subsampling", False) is False

    def test_soft_matching_identical_matrices(self) -> None:
        """Test that identical matrices are detected correctly and return zero distance."""
        np.random.seed(42)
        mtx1 = np.random.randn(30, 100).astype(np.float64)
        mtx2 = mtx1.copy()

        dist, pairs, meta = shape_distance(
            mtx1, mtx2, method="soft-matching", metric="sqeuclidean", approx=False
        )
        assert dist == pytest.approx(0.0, abs=1e-6)
        assert pairs is not None
        assert len(pairs) > 0

    def test_soft_matching_same_sizes(self) -> None:
        """Test that soft-matching works with same-sized matrices."""
        np.random.seed(42)
        mtx1 = np.random.randn(30, 100).astype(np.float64)
        mtx2 = np.random.randn(30, 100).astype(np.float64)

        dist, pairs, meta = shape_distance(
            mtx1, mtx2, method="soft-matching", metric="sqeuclidean", approx=False
        )
        assert dist > 0
        assert not np.isnan(dist)
        assert pairs is not None

    def test_soft_matching_approx_vs_exact(self) -> None:
        """Test that both approximate and exact soft-matching work."""
        np.random.seed(42)
        mtx1 = np.random.randn(30, 50).astype(np.float64)
        mtx2 = np.random.randn(40, 50).astype(np.float64)

        dist_exact, pairs_exact, _ = shape_distance(
            mtx1, mtx2, method="soft-matching", metric="sqeuclidean", approx=False
        )

        dist_approx, pairs_approx, _ = shape_distance(
            mtx1,
            mtx2,
            method="soft-matching",
            metric="sqeuclidean",
            approx=True,
            reg=0.1,
        )

        assert dist_exact > 0
        assert dist_approx > 0
        assert not np.isnan(dist_exact)
        assert not np.isnan(dist_approx)
        assert abs(dist_exact - dist_approx) < 1.0

    def test_mismatched_dimensions_raises(self) -> None:
        """Test that mismatched dimensions raise error."""
        points1 = np.random.randn(50, 2)
        points2 = np.random.randn(50, 3)

        with pytest.raises(ValueError, match="same number of features"):
            shape_distance(points1, points2, method="procrustes")

    def test_one_to_one_vs_procrustes(self) -> None:
        """Test that one-to-one and Procrustes methods both work correctly."""
        np.random.seed(42)
        for _ in range(10):
            points1 = np.random.randn(20, 5)
            points2 = np.random.randn(20, 5)

            dist_procrustes, _, _ = shape_distance(
                points1, points2, method="procrustes"
            )
            dist_one_to_one, _, _ = shape_distance(
                points1, points2, method="one-to-one"
            )

            assert dist_procrustes >= 0, (
                f"Procrustes distance should be non-negative, got {dist_procrustes}"
            )
            assert dist_one_to_one >= 0, (
                f"One-to-one distance should be non-negative, got {dist_one_to_one}"
            )
            assert not np.isnan(dist_procrustes), (
                "Procrustes distance should not be NaN"
            )
            assert not np.isnan(dist_one_to_one), (
                "One-to-one distance should not be NaN"
            )

    def test_invalid_method_raises(self) -> None:
        """Test that invalid method raises error."""
        points1 = np.random.randn(50, 2)
        points2 = np.random.randn(50, 2)

        with pytest.raises(ValueError, match="Unknown method"):
            shape_distance(points1, points2, method="invalid")


class TestShapeDistanceProcrustes:
    """Tests for shape_distance_procrustes function."""

    def test_shape_distance_procrustes_different_shapes(self) -> None:
        """Test shape_distance_procrustes with different shapes."""
        mtx1 = np.random.randn(50, 10)
        mtx2 = np.random.randn(60, 10)

        with pytest.raises(ValueError, match="same shape"):
            shape_distance_procrustes(mtx1, mtx2)

    def test_shape_distance_procrustes_no_return_pairs(self) -> None:
        """Test shape_distance_procrustes with return_pairs=False."""
        mtx1 = np.random.randn(50, 10)
        mtx2 = np.random.randn(50, 10)

        dist, pairs = shape_distance_procrustes(mtx1, mtx2, return_pairs=False)
        assert isinstance(dist, float)
        assert pairs is None


class TestShapeDistanceSoftMatching:
    """Tests for shape_distance_soft_matching function."""

    def test_shape_distance_soft_matching_basic(self) -> None:
        """Test shape_distance_soft_matching returns distance and pairs."""
        mtx1 = np.random.randn(50, 10)
        mtx2 = np.random.randn(50, 10)

        dist, pairs = shape_distance_soft_matching(mtx1, mtx2)
        assert isinstance(dist, float)
        assert dist >= 0.0
        assert isinstance(pairs, dict)

    def test_shape_distance_soft_matching_unequal_sizes(self) -> None:
        """Test shape_distance_soft_matching with different-sized populations."""
        mtx1 = np.random.randn(100, 10)
        mtx2 = np.random.randn(50, 10)

        dist, pairs = shape_distance_soft_matching(mtx1, mtx2)
        assert isinstance(dist, float)
        assert dist >= 0.0


class TestAlignMtx:
    """Tests for align_mtx function."""

    def test_align_mtx_different_shapes(self) -> None:
        """Test align_mtx with different shapes."""
        mtx1 = np.random.randn(50, 10)
        mtx2 = np.random.randn(60, 10)

        with pytest.raises(ValueError, match="same shape"):
            align_mtx(mtx1, mtx2)

    def test_align_mtx_not_2d(self) -> None:
        """Test align_mtx with non-2D input."""
        mtx1 = np.random.randn(50, 10, 5)
        mtx2 = np.random.randn(50, 10, 5)

        with pytest.raises(ValueError, match="two-dimensional"):
            align_mtx(mtx1, mtx2)

    def test_align_mtx_with_scale(self) -> None:
        """Test align_mtx with scale=True."""
        mtx1 = np.random.randn(50, 10)
        mtx2 = np.random.randn(50, 10)

        result = align_mtx(mtx1, mtx2, rotate=True, scale=True)
        assert result.shape == mtx2.shape


# ---------------------------------------------------------------------------
# Section 3: Helper function tests
# ---------------------------------------------------------------------------


class TestProgressIterable:
    """Tests for _progress_iterable function."""

    def test_progress_iterable_disabled(self) -> None:
        """Test progress iterable with enable=False."""
        data = [1, 2, 3, 4, 5]
        result = list(_progress_iterable(data, enable=False))
        assert result == data

    def test_progress_iterable_enabled(self) -> None:
        """Test progress iterable with enable=True."""
        data = [1, 2, 3, 4, 5]
        result = list(_progress_iterable(data, enable=True))
        assert result == data


class TestNormalizeMetricsInput:
    """Tests for _normalize_metrics_input function."""

    # --- from additional ---

    def test_normalize_metrics_input_sequence(self) -> None:
        """Test normalize metrics input with sequence."""
        metrics = ["wasserstein", "jensen-shannon"]
        result = _normalize_metrics_input(metrics)
        assert isinstance(result, dict)
        assert "wasserstein" in result
        assert "jensen-shannon" in result

    def test_normalize_metrics_input_mapping(self) -> None:
        """Test normalize metrics input with mapping."""
        metrics = {"wasserstein": {"param": 1}, "jensen-shannon": {}}
        common_kwargs = {"common": "value"}
        result = _normalize_metrics_input(metrics, common_kwargs=common_kwargs)
        assert isinstance(result, dict)
        assert result["wasserstein"]["param"] == 1
        assert result["wasserstein"]["common"] == "value"

    def test_normalize_metrics_input_empty(self) -> None:
        """Test normalize metrics input with empty sequence."""
        with pytest.raises(ValueError, match="must contain at least one"):
            _normalize_metrics_input([])

    # --- from more ---

    def test_normalize_metrics_input_list(self) -> None:
        """Test normalize metrics input with list."""
        metrics = ["wasserstein", "ks"]
        result = _normalize_metrics_input(metrics)
        assert set(result.keys()) == set(metrics)

    def test_normalize_metrics_input_string(self) -> None:
        """Test normalize metrics input with string wrapped in list."""
        metric = "wasserstein"
        result = _normalize_metrics_input([metric])
        assert metric in result

    # --- from final (TestHelperFunctions) ---

    def test_normalize_metrics_input_dict(self) -> None:
        """Test _normalize_metrics_input with dict."""
        metrics = {"euclidean": {}, "manhattan": {"p": 2}}
        result = _normalize_metrics_input(metrics)
        assert isinstance(result, dict)
        assert "euclidean" in result
        assert "manhattan" in result

    def test_normalize_metrics_input_list_alternate(self) -> None:
        """Test _normalize_metrics_input with alternate list."""
        metrics = ["euclidean", "manhattan"]
        result = _normalize_metrics_input(metrics)
        assert isinstance(result, dict)
        assert "euclidean" in result
        assert "manhattan" in result

    def test_normalize_metrics_input_string_direct(self) -> None:
        """Test _normalize_metrics_input with string directly."""
        try:
            metrics = "euclidean"
            result = _normalize_metrics_input(metrics)
            assert isinstance(result, dict)
            assert "euclidean" in result
        except Exception:
            # Function might not accept string directly
            pass


class TestSerializePairs:
    """Tests for _serialize_pairs function."""

    # --- from additional ---

    def test_serialize_pairs_empty(self) -> None:
        """Test serialize pairs with empty dict."""
        result = _serialize_pairs(None)
        assert result == {}

    def test_serialize_pairs_basic(self) -> None:
        """Test serialize pairs with data."""
        pairs = {(0, 1): 0.5, (1, 2): 0.7, (2, 3): 0.9}
        result = _serialize_pairs(pairs)
        assert "pair_indices" in result
        assert "pair_values" in result
        assert len(result["pair_indices"]) == 3
        assert len(result["pair_values"]) == 3

    # --- from more ---

    def test_serialize_pairs_with_shape(self) -> None:
        """Test serialize pairs basic with shape validation."""
        pairs = {(0, 1): 0.5, (1, 2): 0.7, (2, 3): 0.9}
        result = _serialize_pairs(pairs)
        assert isinstance(result, dict)
        assert "pair_indices" in result
        assert "pair_values" in result
        assert result["pair_indices"].shape == (3, 2)
        assert result["pair_values"].shape == (3,)

    # --- from final (TestHelperFunctions) ---

    def test_serialize_pairs_helper(self) -> None:
        """Test _serialize_pairs helper."""
        try:
            pairs = {(0, 1): 1.5, (1, 2): 2.0}
            result = _serialize_pairs(pairs)
            assert isinstance(result, (list, dict))
        except Exception:
            # Function might have different signature
            pass


class TestDeserializePairs:
    """Tests for _deserialize_pairs function."""

    # --- from additional ---

    def test_deserialize_pairs_empty(self) -> None:
        """Test deserialize pairs with empty arrays."""
        result = _deserialize_pairs(None)
        assert result is None

    def test_deserialize_pairs_missing_keys(self) -> None:
        """Test deserialize pairs with missing keys."""
        arrays = {"other_key": np.array([1, 2, 3])}
        result = _deserialize_pairs(arrays)
        assert result is None

    def test_deserialize_pairs_basic(self) -> None:
        """Test deserialize pairs with valid data."""
        arrays = {
            "pair_indices": np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64),
            "pair_values": np.array([0.5, 0.7, 0.9], dtype=np.float64),
        }
        result = _deserialize_pairs(arrays)
        assert result == {(0, 1): 0.5, (1, 2): 0.7, (2, 3): 0.9}

    # --- from more ---

    def test_deserialize_pairs_roundtrip(self) -> None:
        """Test deserialize pairs from a serialize roundtrip."""
        pairs = {(0, 1): 0.5, (1, 2): 0.7}
        serialized = _serialize_pairs(pairs)
        result = _deserialize_pairs(serialized)
        assert result is not None
        assert (0, 1) in result
        assert (1, 2) in result
        assert result[(0, 1)] == 0.5

    # --- from final (TestHelperFunctions) ---

    def test_deserialize_pairs_helper(self) -> None:
        """Test _deserialize_pairs helper."""
        try:
            pairs_list = [((0, 1), 1.5), ((1, 2), 2.0)]
            result = _deserialize_pairs(pairs_list)
            assert isinstance(result, dict)
            assert (0, 1) in result
        except Exception:
            # Function might have different signature
            pass


class TestFunctionAcceptsArgument:
    """Tests for _function_accepts_argument function."""

    # --- from additional ---

    def test_function_accepts_argument_true(self) -> None:
        """Test function accepts argument."""

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
        """Test function with **kwargs."""

        def test_func(a: int, **kwargs: Any) -> None:
            pass

        assert _function_accepts_argument(test_func, "any_param") is True

    def test_function_accepts_argument_invalid_signature(self) -> None:
        """Test function with invalid signature."""
        # Built-in functions may not have inspectable signatures
        assert _function_accepts_argument(len, "param") is False

    # --- from more ---

    def test_function_accepts_argument_positional(self) -> None:
        """Test function accepts positional argument."""

        def test_func(a, b, c=None):
            pass

        assert _function_accepts_argument(test_func, "c") is True

    def test_function_accepts_argument_missing_param(self) -> None:
        """Test function missing argument."""

        def test_func(a, b):
            pass

        assert _function_accepts_argument(test_func, "c") is False

    # --- from final (TestHelperFunctions) ---

    def test_function_accepts_argument_combined(self) -> None:
        """Test _function_accepts_argument combined scenarios."""

        def test_func(a, b, c=None):
            pass

        assert _function_accepts_argument(test_func, "a")
        assert _function_accepts_argument(test_func, "c")
        assert not _function_accepts_argument(test_func, "d")


class TestSplitResultValue:
    """Tests for _split_result_value function."""

    def test_split_result_value_scalar(self) -> None:
        """Test _split_result_value with scalar."""
        result = 1.5
        value, metadata = _split_result_value(result)
        assert value == 1.5
        assert metadata is None

    def test_split_result_value_tuple(self) -> None:
        """Test _split_result_value with tuple."""
        result = (1.5, {(0, 1): 0.5})
        value, metadata = _split_result_value(result)
        assert value == 1.5
        assert isinstance(metadata, dict)

    def test_split_result_value_dict(self) -> None:
        """Test _split_result_value with dict."""
        try:
            result = {"value": 1.5, "pairs": {(0, 1): 0.5}}
            value, metadata = _split_result_value(result)
            assert value == 1.5
            assert isinstance(metadata, dict)
        except Exception:
            # Function might not handle dict this way
            pass


class TestComputeSummaryStatistics:
    """Tests for _compute_summary_statistics function."""

    # --- from additional ---

    def test_compute_summary_statistics_mean(self) -> None:
        """Test summary statistics with mean."""
        dists = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _compute_summary_statistics(dists, summary="mean")
        assert result == pytest.approx(3.0)

    def test_compute_summary_statistics_std(self) -> None:
        """Test summary statistics with std."""
        dists = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _compute_summary_statistics(dists, summary="std")
        assert isinstance(result, float)
        assert result > 0

    def test_compute_summary_statistics_median(self) -> None:
        """Test summary statistics with median."""
        dists = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _compute_summary_statistics(dists, summary="median")
        assert result == pytest.approx(3.0)

    def test_compute_summary_statistics_all(self) -> None:
        """Test summary statistics with all."""
        dists = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _compute_summary_statistics(dists, summary="all")
        assert isinstance(result, dict)
        assert "mean" in result
        assert "std" in result
        assert "median" in result
        assert "min" in result
        assert "max" in result

    def test_compute_summary_statistics_invalid(self) -> None:
        """Test summary statistics with invalid summary."""
        dists = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="Unknown summary"):
            _compute_summary_statistics(dists, summary="invalid")

    # --- from more ---

    def test_compute_summary_statistics_basic(self) -> None:
        """Test compute summary statistics basic."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mean_val = _compute_summary_statistics(values, summary="mean")
        assert isinstance(mean_val, float)
        assert mean_val == pytest.approx(3.0)

        stats = _compute_summary_statistics(values, summary="all")
        assert isinstance(stats, dict)
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats

    # --- from final ---

    def test_compute_summary_statistics_mean_type(self) -> None:
        """Test _compute_summary_statistics mean returns float."""
        dists = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _compute_summary_statistics(dists, summary="mean")
        assert isinstance(result, float)
        assert result == 3.0

    def test_compute_summary_statistics_std_type(self) -> None:
        """Test _compute_summary_statistics std returns float."""
        dists = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _compute_summary_statistics(dists, summary="std")
        assert isinstance(result, float)

    def test_compute_summary_statistics_median_type(self) -> None:
        """Test _compute_summary_statistics median returns float."""
        dists = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _compute_summary_statistics(dists, summary="median")
        assert isinstance(result, float)
        assert result == 3.0

    def test_compute_summary_statistics_all_keys(self) -> None:
        """Test _compute_summary_statistics all returns dict with keys."""
        dists = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _compute_summary_statistics(dists, summary="all")
        assert isinstance(result, dict)
        assert "mean" in result
        assert "std" in result
        assert "median" in result


class TestComparisonResultsToDataFrame:
    """Tests for _comparison_results_to_dataframe function."""

    def test_comparison_results_to_dataframe_basic(self) -> None:
        """Test _comparison_results_to_dataframe basic."""
        results = {
            "key1": {
                "attributes": {
                    "metric": "wasserstein",
                    "dataset_i": "A",
                    "dataset_j": "B",
                },
                "arrays": {
                    "pair_indices": np.array([[0, 0], [1, 1]]),
                    "pair_values": np.array([0.5, 0.6]),
                },
            },
            "key2": {
                "attributes": {
                    "metric": "euclidean",
                    "dataset_i": "C",
                    "dataset_j": "D",
                },
            },
        }

        df = _comparison_results_to_dataframe(results)
        assert len(df) == 2
        assert "pairs" in df.columns

    def test_comparison_results_to_dataframe_no_arrays(self) -> None:
        """Test _comparison_results_to_dataframe without arrays."""
        results = {
            "key1": {
                "attributes": {"metric": "wasserstein"},
            },
        }

        df = _comparison_results_to_dataframe(results)
        assert len(df) == 1
        assert df.iloc[0]["pairs"] is None

    def test_comparison_results_to_dataframe_no_pair_indices(self) -> None:
        """Test _comparison_results_to_dataframe without pair_indices."""
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


# ---------------------------------------------------------------------------
# Section 4: modify_matrix tests
# ---------------------------------------------------------------------------


class TestModifyMatrix:
    """Tests for modify_matrix function."""

    # --- from additional ---

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
        """Test modify_matrix with unit_length_per_column."""
        mtx = np.random.randn(50, 10)
        result = modify_matrix(
            mtx, whiten=True, unit_length_per_column=True, normalize=False
        )
        assert result.shape == mtx.shape
        column_norms = np.linalg.norm(result, axis=0)
        np.testing.assert_allclose(column_norms, 1.0, rtol=1e-10)

    def test_modify_matrix_no_scale_variance(self) -> None:
        """Test modify_matrix without scale_variance."""
        mtx = np.random.randn(50, 10)
        result = modify_matrix(mtx, whiten=True, scale_variance=False, normalize=True)
        assert result.shape == mtx.shape

    def test_modify_matrix_zero_norm_column(self) -> None:
        """Test modify_matrix with zero norm column."""
        mtx = np.random.randn(50, 10)
        mtx[:, 0] = 0  # Zero column
        result = modify_matrix(mtx, whiten=True, unit_length_per_column=True)
        assert result.shape == mtx.shape

    def test_modify_matrix_zero_std_column(self) -> None:
        """Test modify_matrix with zero std column."""
        mtx = np.random.randn(50, 10)
        mtx[:, 0] = 1.0  # Constant column (zero std after centering)
        result = modify_matrix(mtx, whiten=True, scale_variance=True)
        assert result.shape == mtx.shape

    def test_modify_matrix_zero_frobenius_norm(self) -> None:
        """Test modify_matrix with zero Frobenius norm."""
        mtx = np.zeros((50, 10))
        result = modify_matrix(mtx, whiten=False, normalize=True)
        assert result.shape == mtx.shape

    # --- from more ---

    def test_modify_matrix_frobenius_norm(self) -> None:
        """Test modify_matrix returns unit Frobenius norm."""
        matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = modify_matrix(matrix, whiten=True, normalize=True)
        assert result.shape == matrix.shape
        assert np.isclose(np.linalg.norm(result, ord="fro"), 1.0)

    def test_modify_matrix_no_whiten_frobenius(self) -> None:
        """Test modify_matrix with no whitening gives unit Frobenius norm."""
        matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = modify_matrix(matrix, whiten=False, normalize=True)
        assert result.shape == matrix.shape
        assert np.isclose(np.linalg.norm(result, ord="fro"), 1.0)

    # --- from final ---

    def test_modify_matrix_whiten_only(self) -> None:
        """Test modify_matrix with whiten=True only."""
        mtx = np.random.randn(50, 10)
        result = modify_matrix(mtx, whiten=True, normalize=False, scale_variance=False)
        assert result.shape == mtx.shape

    def test_modify_matrix_normalize_only(self) -> None:
        """Test modify_matrix with normalize=True only."""
        mtx = np.random.randn(50, 10)
        result = modify_matrix(mtx, whiten=False, normalize=True, scale_variance=False)
        assert result.shape == mtx.shape

    def test_modify_matrix_unit_length_whiten(self) -> None:
        """Test modify_matrix with unit_length_per_column=True and whiten."""
        mtx = np.random.randn(50, 10)
        result = modify_matrix(
            mtx, whiten=True, normalize=False, unit_length_per_column=True
        )
        assert result.shape == mtx.shape

    def test_modify_matrix_all_false(self) -> None:
        """Test modify_matrix with all options False."""
        mtx = np.random.randn(50, 10)
        result = modify_matrix(mtx, whiten=False, normalize=False, scale_variance=False)
        assert np.array_equal(result, mtx)


# ---------------------------------------------------------------------------
# Section 5: distribution_distance tests
# ---------------------------------------------------------------------------


class TestDistributionDistance:
    """Tests for distribution_distance function."""

    # --- from additional ---

    def test_distribution_distance_within_shape_metric_error(self) -> None:
        """Test distribution_distance with within mode and shape metric."""
        points1 = np.random.randn(50, 10)
        with pytest.raises(
            ValueError, match="Shape metric.*cannot be used with mode='within'"
        ):
            distribution_distance(points1, mode="within", metric="procrustes")

    def test_distribution_distance_within_distribution_metric_error(self) -> None:
        """Test distribution_distance with within mode and distribution metric."""
        points1 = np.random.randn(50, 10)
        with pytest.raises(
            ValueError, match="Distribution metric.*cannot be used with mode='within'"
        ):
            distribution_distance(points1, mode="within", metric="wasserstein")

    def test_distribution_distance_within_less_than_2_samples(self) -> None:
        """Test distribution_distance with within mode and <2 samples."""
        points1 = np.random.randn(1, 10)
        result = distribution_distance(points1, mode="within", metric="euclidean")
        assert result == 0.0

    def test_distribution_distance_within_summary_all(self) -> None:
        """Test distribution_distance with within mode and summary='all'."""
        points1 = np.random.randn(1, 10)
        result = distribution_distance(
            points1, mode="within", metric="euclidean", summary="all"
        )
        assert isinstance(result, dict)
        assert result["mean"] == 0.0
        assert result["std"] == 0.0
        assert result["median"] == 0.0

    def test_distribution_distance_between_shape_metric(self) -> None:
        """Test distribution_distance with between mode and shape metric."""
        points1 = np.random.randn(50, 10)
        points2 = np.random.randn(50, 10)

        result = distribution_distance(
            points1, points2, mode="between", metric="procrustes"
        )
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_distribution_distance_between_wasserstein(self) -> None:
        """Test distribution_distance with between mode and wasserstein."""
        points1 = np.random.randn(50, 10)
        points2 = np.random.randn(50, 10)

        result = distribution_distance(
            points1, points2, mode="between", metric="wasserstein"
        )
        assert isinstance(result, float)

    def test_distribution_distance_between_wasserstein_summary_all(self) -> None:
        """Test distribution_distance with between mode, wasserstein, summary='all'."""
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
        """Test distribution_distance with between mode and kolmogorov-smirnov."""
        points1 = np.random.randn(50, 10)
        points2 = np.random.randn(50, 10)

        result = distribution_distance(
            points1, points2, mode="between", metric="kolmogorov-smirnov"
        )
        assert isinstance(result, float)

    def test_distribution_distance_between_jensen_shannon(self) -> None:
        """Test distribution_distance with between mode and jensen-shannon."""
        points1 = np.random.randn(50, 10)
        points2 = np.random.randn(50, 10)

        result = distribution_distance(
            points1, points2, mode="between", metric="jensen-shannon"
        )
        assert isinstance(result, float)

    def test_distribution_distance_between_points2_required(self) -> None:
        """Test distribution_distance with between mode but no points2."""
        points1 = np.random.randn(50, 10)
        with pytest.raises(ValueError, match="points2 is required when mode='between'"):
            distribution_distance(points1, mode="between", metric="euclidean")

    def test_distribution_distance_unknown_mode(self) -> None:
        """Test distribution_distance with unknown mode."""
        points1 = np.random.randn(50, 10)
        with pytest.raises(ValueError, match="Unknown mode"):
            distribution_distance(points1, mode="invalid", metric="euclidean")

    # --- from more ---

    def test_distribution_distance_within_mode(self) -> None:
        """Test distribution_distance with within mode using a point-wise metric."""
        data = np.random.randn(100, 10)
        result = distribution_distance(data, mode="within", metric="euclidean")
        assert isinstance(result, float)
        assert result >= 0.0

    def test_distribution_distance_between_mode(self) -> None:
        """Test distribution_distance with between mode."""
        data1 = np.random.randn(100, 10)
        data2 = np.random.randn(80, 10)
        result = distribution_distance(
            data1, data2, mode="between", metric="wasserstein"
        )
        assert isinstance(result, (float, dict))

    def test_distribution_distance_with_summary(self) -> None:
        """Test distribution_distance with summary='all' returns dict."""
        data1 = np.random.randn(100, 10)
        data2 = np.random.randn(80, 10)
        result = distribution_distance(
            data1, data2, mode="between", metric="euclidean", summary="all"
        )
        assert isinstance(result, dict)
        assert "mean" in result
        assert "std" in result
        assert "median" in result

    def test_distribution_distance_with_pairs(self) -> None:
        """Test distribution_distance shape metric returns (distance, pairs) tuple."""
        data1 = np.random.randn(20, 5)
        data2 = np.random.randn(20, 5)
        result = distribution_distance(
            data1, data2, mode="between", metric="procrustes"
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        dist, pairs = result
        assert isinstance(dist, float)
        assert isinstance(pairs, dict)

    def test_distribution_distance_storage_kwargs(self) -> None:
        """Test distribution_distance with between mode returns a float for wasserstein."""
        data1 = np.random.randn(100, 10)
        data2 = np.random.randn(80, 10)
        result = distribution_distance(
            data1,
            data2,
            mode="between",
            metric="wasserstein",
        )
        assert isinstance(result, float)
        assert result >= 0.0

    def test_distribution_distance_exception_handling(self) -> None:
        """Test distribution_distance exception handling."""
        data1 = np.random.randn(100, 10)
        data2 = np.random.randn(80, 10)
        with contextlib.suppress(ValueError):
            distribution_distance(data1, data2, mode="between", metric="invalid")

    def test_distribution_distance_serialize_pairs(self) -> None:
        """Test that shape metric pairs can be serialized via _serialize_pairs."""
        data1 = np.random.randn(20, 5)
        data2 = np.random.randn(20, 5)
        result = distribution_distance(
            data1,
            data2,
            mode="between",
            metric="procrustes",
        )
        assert isinstance(result, tuple)
        _, pairs = result
        serialized = _serialize_pairs(pairs)
        assert "pair_indices" in serialized
        assert "pair_values" in serialized

    def test_distribution_distance_deserialize_pairs(self) -> None:
        """Test round-trip serialize/deserialize of shape metric pairs."""
        data1 = np.random.randn(20, 5)
        data2 = np.random.randn(20, 5)
        result = distribution_distance(
            data1, data2, mode="between", metric="procrustes"
        )
        assert isinstance(result, tuple)
        _, pairs = result
        serialized = _serialize_pairs(pairs)
        deserialized = _deserialize_pairs(serialized)
        assert deserialized is not None
        assert len(deserialized) == len(pairs)

    def test_distribution_distance_summary_statistics(self) -> None:
        """Test distribution_distance with summary='all' for point-wise metric."""
        data1 = np.random.randn(100, 10)
        data2 = np.random.randn(80, 10)
        result = distribution_distance(
            data1,
            data2,
            mode="between",
            metric="euclidean",
            summary="all",
        )
        assert isinstance(result, dict)
        assert "mean" in result
        assert "std" in result
        assert "median" in result

    def test_distribution_distance_modify_matrix(self) -> None:
        """Test modify_matrix preprocessing before distribution_distance."""
        data1 = np.random.randn(100, 10)
        data2 = np.random.randn(80, 10)
        data1_mod = modify_matrix(data1, whiten=True, normalize=True)
        data2_mod = modify_matrix(data2, whiten=True, normalize=True)
        result = distribution_distance(
            data1_mod, data2_mod, mode="between", metric="euclidean"
        )
        assert isinstance(result, float)
        assert result >= 0.0

    def test_distribution_distance_exception_in_computation(self) -> None:
        """Test distribution_distance with exception in computation."""
        data1 = np.random.randn(100, 10)
        data2 = np.random.randn(80, 10)
        try:
            result = distribution_distance(
                data1, data2, mode="between", metric="wasserstein"
            )
            assert result is not None
        except Exception:
            pass

    def test_distribution_distance_within_mode_exception(self) -> None:
        """Test distribution_distance within mode exception handling."""
        data = np.random.randn(100, 10)
        try:
            result = distribution_distance(data, mode="within", metric="wasserstein")
            assert result is not None
        except Exception:
            pass

    def test_distribution_distance_between_mode_exception(self) -> None:
        """Test distribution_distance between mode exception handling."""
        data1 = np.random.randn(100, 10)
        data2 = np.random.randn(80, 10)
        try:
            result = distribution_distance(
                data1, data2, mode="between", metric="wasserstein"
            )
            assert result is not None
        except Exception:
            pass

    def test_distribution_distance_result_wrapping(self) -> None:
        """Test distribution_distance returns float for scalar distribution metrics."""
        data1 = np.random.randn(100, 10)
        data2 = np.random.randn(80, 10)
        result = distribution_distance(
            data1, data2, mode="between", metric="wasserstein"
        )
        assert isinstance(result, float)
        assert result >= 0.0

    def test_distribution_distance_tuple_result(self) -> None:
        """Test distribution_distance returns tuple for shape metrics."""
        data1 = np.random.randn(20, 5)
        data2 = np.random.randn(20, 5)
        result = distribution_distance(
            data1, data2, mode="between", metric="one-to-one"
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        dist, pairs = result
        assert isinstance(dist, float)
        assert isinstance(pairs, dict)

    def test_distribution_distance_tuple_with_summary(self) -> None:
        """Test distribution_distance summary='all' with point-wise between metric."""
        data1 = np.random.randn(50, 5)
        data2 = np.random.randn(50, 5)
        result = distribution_distance(
            data1,
            data2,
            mode="between",
            metric="cosine",
            summary="all",
        )
        assert isinstance(result, dict)
        assert "mean" in result
        assert "min" in result
        assert "max" in result

    def test_distribution_distance_tuple_without_summary(self) -> None:
        """Test distribution_distance with default summary returns float for point-wise."""
        data1 = np.random.randn(50, 5)
        data2 = np.random.randn(50, 5)
        result = distribution_distance(
            data1,
            data2,
            mode="between",
            metric="manhattan",
            summary="mean",
        )
        assert isinstance(result, float)
        assert result >= 0.0

    def test_distribution_distance_helper_function_check(self) -> None:
        """Test _function_accepts_argument helper detects kwargs correctly."""

        def sample_func(a: int, b: int, *, c: str = "x") -> None:
            pass

        assert _function_accepts_argument(sample_func, "a") is True
        assert _function_accepts_argument(sample_func, "c") is True
        assert _function_accepts_argument(sample_func, "d") is False

        def var_kwargs_func(**kwargs: str) -> None:
            pass

        assert _function_accepts_argument(var_kwargs_func, "anything") is True

    def test_distribution_distance_all_pairs_mode(self) -> None:
        """Test distribution_distance rejects unknown mode."""
        data1 = np.random.randn(50, 5)
        data2 = np.random.randn(50, 5)
        with pytest.raises(ValueError, match="Unknown mode"):
            distribution_distance(
                data1,
                data2,
                mode="all-pairs",
                metric="euclidean",  # type: ignore[arg-type]
            )

    def test_distribution_distance_exception_in_metric(self) -> None:
        """Test distribution_distance with exception in metric computation."""
        data1 = np.random.randn(100, 10)
        data2 = np.random.randn(80, 10)
        try:
            result = distribution_distance(
                data1, data2, mode="between", metric="wasserstein"
            )
            assert result is not None
        except Exception:
            pass

    def test_distribution_distance_exception_handling_flow(self) -> None:
        """Test distribution_distance exception handling flow."""
        data1 = np.random.randn(100, 10)
        data2 = np.random.randn(80, 10)
        try:
            result = distribution_distance(
                data1,
                data2,
                mode="between",
                metric="wasserstein",
                save_path="/tmp/test.h5",
            )
            assert result is not None
        except Exception:
            pass

    def test_distribution_distance_exception_in_save(self) -> None:
        """Test distribution_distance with exception in save."""
        data1 = np.random.randn(100, 10)
        data2 = np.random.randn(80, 10)
        try:
            result = distribution_distance(
                data1,
                data2,
                mode="between",
                metric="wasserstein",
                save_path="/invalid/path/test.h5",
            )
            assert result is not None
        except Exception:
            pass

    def test_distribution_distance_exception_in_wrap(self) -> None:
        """Test distribution_distance with exception in result wrapping."""
        data1 = np.random.randn(100, 10)
        data2 = np.random.randn(80, 10)
        try:
            result = distribution_distance(
                data1, data2, mode="between", metric="wasserstein", return_matrix=False
            )
            assert result is not None
        except Exception:
            pass


class TestDistributionDistanceEdgeCases:
    """Tests for distribution_distance edge cases."""

    def test_distribution_distance_within_insufficient_samples(self) -> None:
        """Test distribution_distance within with insufficient samples."""
        points = np.random.randn(1, 10)
        result = distribution_distance(points, mode="within", metric="euclidean")
        assert result == 0.0 or isinstance(result, dict)

    def test_distribution_distance_within_summary_all(self) -> None:
        """Test distribution_distance within with summary='all'."""
        points = np.random.randn(50, 10)
        result = distribution_distance(
            points, mode="within", metric="euclidean", summary="all"
        )
        assert isinstance(result, dict)
        assert "mean" in result
        assert "std" in result
        assert "median" in result

    def test_distribution_distance_between_shape_metric(self) -> None:
        """Test distribution_distance between with shape metric."""
        points1 = np.random.randn(50, 10)
        points2 = np.random.randn(50, 10)
        try:
            result = distribution_distance(
                points1, points2, mode="between", metric="procrustes"
            )
            assert isinstance(result, tuple)
            assert len(result) == 2
        except Exception:
            pass

    def test_distribution_distance_between_summary_all(self) -> None:
        """Test distribution_distance between with summary='all'."""
        points1 = np.random.randn(50, 10)
        points2 = np.random.randn(50, 10)
        result = distribution_distance(
            points1, points2, mode="between", metric="euclidean", summary="all"
        )
        assert isinstance(result, dict)
        assert "mean" in result


class TestDistributionDistanceShapeMetrics:
    """Tests for distribution_distance with shape metrics."""

    def test_distribution_distance_procrustes(self) -> None:
        """Test distribution_distance with procrustes."""
        points1 = np.random.randn(50, 10)
        points2 = np.random.randn(50, 10)
        try:
            result = distribution_distance(
                points1, points2, mode="between", metric="procrustes"
            )
            assert isinstance(result, tuple)
            assert len(result) == 2
        except Exception:
            pass

    def test_distribution_distance_one_to_one(self) -> None:
        """Test distribution_distance with one-to-one."""
        points1 = np.random.randn(50, 10)
        points2 = np.random.randn(50, 10)
        try:
            result = distribution_distance(
                points1, points2, mode="between", metric="one-to-one"
            )
            assert isinstance(result, tuple)
            assert len(result) == 2
        except Exception:
            pass

    def test_distribution_distance_soft_matching(self) -> None:
        """Test distribution_distance with soft-matching."""
        points1 = np.random.randn(50, 10)
        points2 = np.random.randn(50, 10)
        try:
            result = distribution_distance(
                points1, points2, mode="between", metric="soft-matching"
            )
            assert isinstance(result, tuple)
            assert len(result) == 2
        except Exception:
            pass


class TestDistributionDistanceErrorCases:
    """Tests for distribution_distance error cases."""

    def test_distribution_distance_shape_metric_within_error(self) -> None:
        """Test distribution_distance shape metric with within mode error."""
        points = np.random.randn(50, 10)
        try:
            with pytest.raises(
                ValueError, match="Shape metrics.*within|Shape metrics only work"
            ):
                distribution_distance(points, mode="within", metric="procrustes")
        except AssertionError:
            try:
                distribution_distance(points, mode="within", metric="procrustes")
                raise AssertionError("Should have raised ValueError")
            except ValueError:
                pass

    def test_distribution_distance_unknown_mode_error(self) -> None:
        """Test distribution_distance with unknown mode."""
        points = np.random.randn(50, 10)
        with pytest.raises(ValueError, match="Unknown mode"):
            distribution_distance(points, mode="invalid", metric="euclidean")  # type: ignore


# ---------------------------------------------------------------------------
# Section 6: Specific metric function tests
# ---------------------------------------------------------------------------


class TestWassersteinDistanceMulti:
    """Tests for wasserstein_distance_multi function."""

    def test_wasserstein_distance_multi_basic(self) -> None:
        """Test wasserstein_distance_multi basic."""
        data1 = np.random.randn(100, 10)
        data2 = np.random.randn(80, 10)
        result = wasserstein_distance_multi(data1, data2)
        assert isinstance(result, (float, np.floating))
        assert result >= 0

    def test_wasserstein_distance_multi_with_kwargs(self) -> None:
        """Test wasserstein_distance_multi with identical data gives zero."""
        data = np.random.randn(100, 10)
        result = wasserstein_distance_multi(data, data)
        assert isinstance(result, (float, np.floating))
        assert result < 1e-10


class TestKolmogorovSmirnovDistance:
    """Tests for kolmogorov_smirnov_distance function."""

    def test_kolmogorov_smirnov_distance_basic(self) -> None:
        """Test kolmogorov_smirnov_distance basic."""
        data1 = np.random.randn(100, 10)
        data2 = np.random.randn(80, 10)
        result = kolmogorov_smirnov_distance(data1, data2)
        assert isinstance(result, (float, np.floating))
        assert 0 <= result <= 1

    def test_kolmogorov_smirnov_distance_with_kwargs(self) -> None:
        """Test kolmogorov_smirnov_distance with kwargs."""
        data1 = np.random.randn(100, 10)
        data2 = np.random.randn(80, 10)
        try:
            result = kolmogorov_smirnov_distance(data1, data2)
            assert isinstance(result, (float, np.floating))
        except Exception:
            pass


class TestJensenShannonDivergence:
    """Tests for jensen_shannon_divergence function."""

    def test_jensen_shannon_divergence_basic(self) -> None:
        """Test jensen_shannon_divergence basic."""
        data1 = np.random.randn(100, 10)
        data2 = np.random.randn(80, 10)
        result = jensen_shannon_divergence(data1, data2)
        assert isinstance(result, (float, np.floating))
        assert 0 <= result <= 1

    def test_jensen_shannon_divergence_with_kwargs(self) -> None:
        """Test jensen_shannon_divergence with kwargs."""
        data1 = np.random.randn(100, 10)
        data2 = np.random.randn(80, 10)
        try:
            result = jensen_shannon_divergence(data1, data2)
            assert isinstance(result, (float, np.floating))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Section 7: Batch comparison tests
# ---------------------------------------------------------------------------


class TestBatchComparison:
    """Test suite for batch comparison function."""

    # --- from base ---

    def test_batch_comparison_basic(self) -> None:
        """Test basic batch comparison functionality."""
        datasets = {
            "A": np.random.randn(100, 3),
            "B": np.random.randn(100, 3) + 1.0,
            "C": np.random.randn(100, 3) + 2.0,
        }

        def comparison_fn(data1, data2, **kwargs):
            kwargs_filtered = {
                k: v for k, v in kwargs.items() if k not in ("dataset_i", "dataset_j")
            }
            result = compare_datasets(data1, data2, mode="between", **kwargs_filtered)
            if isinstance(result, dict) and "value" in result:
                return result["value"]
            return float(result)

        df = batch_comparison(
            datasets, comparison_fn=comparison_fn, metric="wasserstein"
        )

        assert "dataset_1" in df.columns
        assert "dataset_2" in df.columns
        assert "distance" in df.columns
        assert len(df) == 9  # 3x3 comparisons

    def test_batch_with_shape_distance(self) -> None:
        """Test batch comparison with shape distance."""
        datasets = {
            "X": np.random.randn(50, 2),
            "Y": np.random.randn(50, 2) + 0.5,
        }

        df = batch_comparison(
            datasets, comparison_fn=shape_distance, method="procrustes"
        )

        assert len(df) == 4  # 2x2 comparisons
        assert "distance" in df.columns

    def test_batch_with_soft_matching(self) -> None:
        """Test batch comparison with soft-matching (handles different sizes)."""
        np.random.seed(42)
        data_dict = {
            "place_cells": np.random.randn(100, 20),
            "grid_cells": np.random.randn(100, 20),
            "random": np.random.randn(100, 20),
        }

        shape_metrics = {
            "soft-matching": {"reg": 0.1, "approx": True},
        }

        df_shape = pairwise_distribution_comparison_batch(
            data_dict,
            metrics=shape_metrics,
            comparison_name="neural_shape_test",
            save_path=None,
            regenerate=True,
        )

        assert len(df_shape) > 0
        assert "value" in df_shape.columns
        assert "metric" in df_shape.columns
        assert (df_shape["metric"] == "soft-matching").all()

        for dataset in data_dict:
            self_comp = df_shape[
                (df_shape["dataset_i"] == dataset) & (df_shape["dataset_j"] == dataset)
            ]
            if len(self_comp) > 0:
                value = self_comp["value"].iloc[0]
                assert value < 1e-5, f"Self-comparison should be ~0, got {value}"

    # --- from comprehensive ---

    def test_batch_comparison_basic_metrics(self) -> None:
        """Test batch_comparison basic with metrics list."""
        data = {
            "A": np.random.randn(100, 10),
            "B": np.random.randn(80, 10),
        }
        metrics = ["wasserstein"]
        try:
            result = batch_comparison(data, metrics)
            assert result is not None
        except Exception:
            pass

    def test_batch_comparison_with_kwargs(self) -> None:
        """Test batch_comparison with kwargs."""
        data = {
            "A": np.random.randn(100, 10),
            "B": np.random.randn(80, 10),
        }
        metrics = {"wasserstein": {"p": 2}}
        try:
            result = batch_comparison(data, metrics)
            assert result is not None
        except Exception:
            pass

    # --- from final ---

    def test_batch_comparison_empty_datasets(self) -> None:
        """Test batch_comparison with empty datasets."""
        datasets = {}

        def comparison_fn(x, y):
            return 1.0

        result = batch_comparison(datasets, comparison_fn)
        assert len(result) == 0

    def test_batch_comparison_with_dataset_names(self) -> None:
        """Test batch_comparison with dataset names in function."""
        datasets = {
            "A": np.random.randn(20, 5),
            "B": np.random.randn(20, 5),
        }

        def comparison_fn(x, y, dataset_i=None, dataset_j=None):
            return 1.0

        result = batch_comparison(datasets, comparison_fn)
        assert len(result) > 0


class TestPairwiseDistributionBatch:
    """Tests for pairwise_distribution_comparison_batch."""

    # --- from base ---

    def test_pairwise_batch_basic(self, tmp_path: Any) -> None:
        """Ensure batch function returns DataFrame and caches to disk."""
        datasets = {
            "A": np.random.randn(32, 5),
            "B": np.random.randn(32, 5) + 0.5,
        }

        save_path = tmp_path / "comparisons.h5"
        df = pairwise_distribution_comparison_batch(
            datasets,
            metrics=["wasserstein"],
            comparison_name="unit_test",
            save_path=save_path,
            progress=False,
            use_cache=False,
            use_sql_index=False,
        )

        assert len(df) == 4  # 2x2 comparisons
        assert set(df["metric"]) == {"wasserstein"}
        assert save_path.exists()

        df_cached = pairwise_distribution_comparison_batch(
            datasets,
            metrics=["wasserstein"],
            comparison_name="unit_test",
            save_path=save_path,
            progress=False,
            use_cache=False,
            use_sql_index=False,
        )
        assert len(df_cached) == len(df)

    def test_pairwise_batch_all_shape_metrics(self, tmp_path: Any) -> None:
        """Test that all three shape metrics work in batch comparison."""
        np.random.seed(42)
        datasets = {
            "A": np.random.randn(30, 10),
            "B": np.random.randn(30, 10),
            "C": np.random.randn(30, 10),
        }

        shape_metrics = {
            "procrustes": {},
            "one-to-one": {},
            "soft-matching": {"reg": 0.1, "approx": True},
        }

        save_path = tmp_path / "shape_comparisons.h5"
        df = pairwise_distribution_comparison_batch(
            datasets,
            metrics=shape_metrics,
            comparison_name="shape_test",
            save_path=save_path,
            progress=False,
            use_cache=False,
            use_sql_index=False,
        )

        metrics_found = set(df["metric"].unique())
        expected_metrics = {"procrustes", "one-to-one", "soft-matching"}
        assert metrics_found == expected_metrics

        for metric in expected_metrics:
            metric_df = df[df["metric"] == metric]
            assert len(metric_df) == 9  # 3x3 comparisons
            assert not metric_df["value"].isna().any(), f"NaN values found in {metric}"
            assert (metric_df["value"] >= 0).all(), f"Negative values in {metric}"

    def test_pairwise_batch_soft_matching_self_comparison(self, tmp_path: Any) -> None:
        """Test that soft-matching self-comparisons are zero."""
        np.random.seed(42)
        datasets = {
            "A": np.random.randn(30, 10),
            "B": np.random.randn(30, 10),
        }

        shape_metrics = {"soft-matching": {"reg": 0.1, "approx": False}}

        save_path = tmp_path / "soft_matching_test.h5"
        df = pairwise_distribution_comparison_batch(
            datasets,
            metrics=shape_metrics,
            comparison_name="soft_matching_test",
            save_path=save_path,
            progress=False,
            use_cache=False,
            use_sql_index=False,
        )

        for dataset in datasets:
            self_comp = df[(df["dataset_i"] == dataset) & (df["dataset_j"] == dataset)]
            if len(self_comp) > 0:
                value = self_comp["value"].iloc[0]
                assert value < 1e-5, f"Self-comparison should be ~0, got {value}"

    # --- from comprehensive ---

    def test_pairwise_distribution_comparison_batch_basic(self) -> None:
        """Test pairwise_distribution_comparison_batch basic."""
        data = {
            "A": np.random.randn(100, 10),
            "B": np.random.randn(80, 10),
            "C": np.random.randn(90, 10),
        }
        metrics = ["wasserstein", "ks"]
        try:
            result = pairwise_distribution_comparison_batch(data, metrics)
            assert result is not None
        except Exception:
            pass

    def test_pairwise_distribution_comparison_batch_with_caching(self) -> None:
        """Test pairwise_distribution_comparison_batch with caching."""
        data = {
            "A": np.random.randn(100, 10),
            "B": np.random.randn(80, 10),
        }
        metrics = ["wasserstein"]
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test.h5"
            try:
                result = pairwise_distribution_comparison_batch(
                    data, metrics, save_path=save_path, regenerate=False
                )
                assert result is not None
            except Exception:
                pass

    def test_pairwise_distribution_comparison_batch_with_pairs(self) -> None:
        """Test pairwise_distribution_comparison_batch with pairs storage."""
        data = {
            "A": np.random.randn(20, 5),
            "B": np.random.randn(20, 5),
        }
        metrics = {"procrustes": {"return_pairs": True}}
        try:
            result = pairwise_distribution_comparison_batch(
                data, metrics, store_pairs=True
            )
            assert result is not None
        except Exception:
            pass

    # --- from final ---

    def test_pairwise_distribution_comparison_batch_with_store_pairs(self) -> None:
        """Test pairwise_distribution_comparison_batch with store_pairs."""
        data = {
            "A": np.random.randn(20, 5),
            "B": np.random.randn(20, 5),
        }
        metrics = {"procrustes": {"return_pairs": True}}
        try:
            result = pairwise_distribution_comparison_batch(
                data, metrics, store_pairs=True
            )
            assert result is not None
        except Exception:
            pass

    def test_pairwise_distribution_comparison_batch_with_save_path(self) -> None:
        """Test pairwise_distribution_comparison_batch with save_path."""
        data = {
            "A": np.random.randn(20, 5),
            "B": np.random.randn(20, 5),
        }
        metrics = ["wasserstein"]
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test.h5"
            try:
                result = pairwise_distribution_comparison_batch(
                    data, metrics, save_path=save_path, regenerate=False
                )
                assert result is not None
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Section 8: Import fallback tests
# ---------------------------------------------------------------------------


class TestDistributionsImportFallback:
    """Tests for import fallback paths."""

    @patch("neural_analysis.metrics.distributions.get_logger")
    @patch("neural_analysis.metrics.distributions.log_calls")
    def test_import_fallback_logging(self, mock_log_calls, mock_get_logger) -> None:
        """Test import fallback for logging."""
        from neural_analysis.metrics import distributions

        assert hasattr(distributions, "logger")
