"""Tests for distribution comparison functions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from neural_analysis.metrics.pairwise_metrics import (
    compare_datasets,
    compute_all_pairs,
    compute_within_distances,
)


class TestCompareDistributions:
    """Test suite for compare_datasets function (replaces compare_distributions)."""

    def test_identical_distributions(self) -> None:
        """Test that identical distributions have zero distance (or 1.0 for cosine)."""
        p = np.random.randn(100, 3)
        # Wasserstein should be zero for identical distributions
        result = compare_datasets(p, p, mode="between", metric="wasserstein")
        # Handle BetweenResult dict format
        dist = result["value"] if isinstance(result, dict) else float(result)
        assert dist == pytest.approx(0.0, abs=1e-6)

    def test_shifted_distributions_wasserstein(self) -> None:
        """Test Wasserstein distance increases with shift."""
        p1 = np.random.randn(100, 3)
        p2 = p1 + 2.0
        result = compare_datasets(p1, p2, mode="between", metric="wasserstein")
        dist = result["value"] if isinstance(result, dict) else float(result)
        # Distance should be positive and roughly proportional to shift
        assert dist > 1.0

    def test_kolmogorov_smirnov_metric(self) -> None:
        """Test K-S statistic on shifted distributions."""
        p1 = np.random.randn(100, 2)
        p2 = np.random.randn(100, 2) + 1.5
        result = compare_datasets(p1, p2, mode="between", metric="kolmogorov-smirnov")
        dist = result["value"] if isinstance(result, dict) else float(result)
        # K-S should be > 0 for shifted distributions
        assert 0.0 < dist <= 1.0

    def test_jensen_shannon_metric(self) -> None:
        """Test Jensen-Shannon divergence."""
        np.random.seed(42)
        p1 = np.random.randn(200, 2)
        p2 = np.random.randn(200, 2) + 1.0
        result = compare_datasets(p1, p2, mode="between", metric="jensen-shannon")
        dist = result["value"] if isinstance(result, dict) else float(result)
        # JS divergence should be in [0, 1] (in bits)
        assert 0.0 <= dist <= 1.0

    def test_euclidean_metric(self) -> None:
        """Test Euclidean distance between centroids."""
        p1 = np.random.randn(100, 3)
        p2 = p1 + np.array([3, 4, 0])
        result = compare_datasets(p1, p2, mode="between", metric="euclidean")
        dist = result["value"] if isinstance(result, dict) else float(result)
        # Distance between centroids should be ~5.0
        assert dist == pytest.approx(5.0, rel=0.2)

    def test_mahalanobis_metric(self) -> None:
        """Test Mahalanobis distance."""
        np.random.seed(42)
        p1 = np.random.randn(100, 3)
        p2 = np.random.randn(100, 3) + 2.0
        result = compare_datasets(p1, p2, mode="between", metric="mahalanobis")
        dist = result["value"] if isinstance(result, dict) else float(result)
        # Should be positive for different distributions
        assert dist > 0

    def test_cosine_metric(self) -> None:
        """Test cosine similarity."""
        p1 = np.random.randn(100, 3) + np.array([1, 0, 0])
        p2 = np.random.randn(100, 3) + np.array([2, 0, 0])
        result = compare_datasets(p1, p2, mode="between", metric="cosine")
        sim = result["value"] if isinstance(result, dict) else float(result)
        # Cosine similarity should be between 0 and 1
        # Random distributions won't necessarily have high similarity
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
    """Test suite for compare_datasets with all-pairs mode (replaces compare_distribution_groups)."""

    def test_between_groups_basic(self) -> None:
        """Test between-group comparison using all-pairs mode."""
        groups = {
            "A": np.random.randn(50, 3),
            "B": np.random.randn(50, 3) + 1.0,
            "C": np.random.randn(50, 3) + 2.0,
        }
        result = compare_datasets(groups, mode="all-pairs", metric="wasserstein")

        # Check structure: result is dict[str, dict[str, float]]
        assert set(result.keys()) == {"A", "B", "C"}
        assert set(result["A"].keys()) == {"A", "B", "C"}

        # Self-distance should be zero
        assert result["A"]["A"] == pytest.approx(0.0, abs=1e-6)

        # A to B should be less than A to C (B is closer)
        assert result["A"]["B"] < result["A"]["C"]

    def test_inside_groups(self) -> None:
        """Test within-group variability using compute_within_distances."""
        np.random.seed(42)
        groups = {
            "tight": np.random.randn(50, 3) * 0.1,  # Low variance
            "loose": np.random.randn(50, 3) * 2.0,  # High variance
        }

        # Compute within-group distances for each group
        means = []
        stds = []
        for name, points in groups.items():
            dist_matrix = compute_within_distances(
                points, metric="euclidean", return_matrix=True
            )
            # Extract upper triangle (excluding diagonal)
            mask = np.triu(np.ones_like(dist_matrix, dtype=bool), k=1)
            dists = dist_matrix[mask]
            means.append(float(np.mean(dists)))
            stds.append(float(np.std(dists)))

        result = {"mean": np.array(means), "std": np.array(stds)}

        # Check structure
        assert "mean" in result and "std" in result
        assert result["mean"].shape == (2,)

        # Loose group should have higher internal distance
        assert result["mean"][1] > result["mean"][0]

    def test_single_point_group(self) -> None:
        """Test with group containing only one point."""
        groups = {
            "single": np.array([[1, 2, 3]]),
            "normal": np.random.randn(50, 3),
        }

        # For single-point group, within-distance should be zero
        single_dist = compute_within_distances(groups["single"], metric="euclidean")
        normal_dist = compute_within_distances(groups["normal"], metric="euclidean")

        # Single-point group should have zero internal distance
        assert single_dist == pytest.approx(0.0)

    def test_different_metrics(self) -> None:
        """Test that different metrics work."""
        groups = {
            "A": np.random.randn(30, 2),
            "B": np.random.randn(30, 2) + 1.0,
        }

        # Scalar-returning metrics work with all-pairs mode
        for metric in ["wasserstein"]:
            result = compare_datasets(groups, mode="all-pairs", metric=metric)
            assert "A" in result and "B" in result
            assert "A" in result["A"] and "B" in result["A"]

        # Point-to-point metrics need to use between mode in a loop
        for metric in ["euclidean", "cosine"]:
            # Use compare_datasets with mode="between" for each pair
            result_ab = compare_datasets(
                groups["A"], groups["B"], mode="between", metric=metric
            )
            dist_ab = result_ab["value"] if isinstance(result_ab, dict) else float(result_ab)
            # Euclidean should be positive, cosine can be negative but in [-1, 1]
            if metric == "euclidean":
                assert dist_ab > 0
            else:  # cosine
                assert -1.0 <= dist_ab <= 1.0

    def test_tuple_keys(self) -> None:
        """Test that tuple keys work as group identifiers."""
        groups = {
            (0, 0): np.random.randn(50, 3),
            (0, 1): np.random.randn(50, 3) + 1.0,
            (1, 0): np.random.randn(50, 3) + 2.0,
        }
        # Use scalar-returning metric for all-pairs mode
        result = compare_datasets(groups, mode="all-pairs", metric="wasserstein")

        assert (0, 0) in result
        assert set(result[(0, 0)].keys()) == {(0, 0), (0, 1), (1, 0)}


class TestShapeDistance:
    """Test suite for shape distance functions."""

    def test_procrustes_identical(self) -> None:
        """Test Procrustes distance for identical shapes."""
        from neural_analysis.metrics.distributions import shape_distance

        points1 = np.random.randn(50, 3)
        points2 = points1.copy()

        dist, pairs, meta = shape_distance(points1, points2, method="procrustes")
        assert dist == pytest.approx(0.0, abs=1e-6)

    def test_procrustes_rotated(self) -> None:
        """Test Procrustes handles rotation."""
        from neural_analysis.metrics.distributions import shape_distance

        points1 = np.random.randn(50, 3)
        # Apply rotation
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
        # Should be near zero after alignment (allow some tolerance for numerical precision)
        assert dist < 0.2

    def test_one_to_one_method(self) -> None:
        """Test one-to-one matching distance."""
        from neural_analysis.metrics.distributions import shape_distance

        points1 = np.random.randn(30, 2)
        points2 = np.random.randn(30, 2) + 1.0

        dist, pairs, meta = shape_distance(points1, points2, method="one-to-one")
        assert dist > 0

    def test_soft_matching_method(self) -> None:
        """Test soft matching distance."""
        from neural_analysis.metrics.distributions import shape_distance

        points1 = np.random.randn(40, 2)
        points2 = np.random.randn(40, 2) + 0.5

        dist, pairs, meta = shape_distance(points1, points2, method="soft-matching")
        assert dist > 0

    def test_mismatched_dimensions_raises(self) -> None:
        """Test that mismatched dimensions raise error."""
        from neural_analysis.metrics.distributions import shape_distance

        points1 = np.random.randn(50, 2)
        points2 = np.random.randn(50, 3)

        with pytest.raises(ValueError, match="same number of features"):
            shape_distance(points1, points2, method="procrustes")

    def test_one_to_one_less_than_or_equal_procrustes(self) -> None:
        """Test that one-to-one distance ≤ Procrustes distance.

        This property must hold because one-to-one searches over a larger space:
        {all permutations} × {optimal rotation} ⊇ {identity permutation} × {optimal rotation}
        """
        from neural_analysis.metrics.distributions import shape_distance

        # Test with multiple random datasets
        np.random.seed(42)
        for _ in range(10):
            points1 = np.random.randn(20, 5)
            points2 = np.random.randn(20, 5)

            dist_procrustes, _, _ = shape_distance(points1, points2, method="procrustes")
            dist_one_to_one, _, _ = shape_distance(points1, points2, method="one-to-one")

            # Allow small numerical tolerance
            assert dist_one_to_one <= dist_procrustes * 1.0001, (
                f"One-to-one ({dist_one_to_one:.6f}) should be ≤ "
                f"Procrustes ({dist_procrustes:.6f})"
            )

    def test_invalid_method_raises(self) -> None:
        """Test that invalid method raises error."""
        from neural_analysis.metrics.distributions import shape_distance

        points1 = np.random.randn(50, 2)
        points2 = np.random.randn(50, 2)

        with pytest.raises(ValueError, match="Unknown method"):
            shape_distance(points1, points2, method="invalid")


class TestBatchComparison:
    """Test suite for batch comparison function."""

    def test_batch_comparison_basic(self) -> None:
        """Test basic batch comparison functionality."""
        from neural_analysis.metrics.distributions import batch_comparison

        datasets = {
            "A": np.random.randn(100, 3),
            "B": np.random.randn(100, 3) + 1.0,
            "C": np.random.randn(100, 3) + 2.0,
        }

        # Use compare_datasets as comparison function
        # Filter out dataset_i and dataset_j kwargs that batch_comparison adds
        def comparison_fn(data1, data2, **kwargs):
            # Remove dataset_i and dataset_j as they're not needed for compare_datasets
            kwargs_filtered = {k: v for k, v in kwargs.items() if k not in ("dataset_i", "dataset_j")}
            result = compare_datasets(data1, data2, mode="between", **kwargs_filtered)
            # Extract value from BetweenResult dict if needed
            if isinstance(result, dict) and "value" in result:
                return result["value"]
            return float(result)

        df = batch_comparison(
            datasets, comparison_fn=comparison_fn, metric="wasserstein"
        )

        # Check dataframe structure
        assert "dataset_1" in df.columns
        assert "dataset_2" in df.columns
        assert "distance" in df.columns
        assert len(df) == 9  # 3x3 comparisons

    def test_batch_with_shape_distance(self) -> None:
        """Test batch comparison with shape distance."""
        from neural_analysis.metrics.distributions import (
            batch_comparison,
            shape_distance,
        )

        datasets = {
            "X": np.random.randn(50, 2),
            "Y": np.random.randn(50, 2) + 0.5,
        }

        df = batch_comparison(
            datasets, comparison_fn=shape_distance, method="procrustes"
        )

        assert len(df) == 4  # 2x2 comparisons
        assert "distance" in df.columns


class TestPairwiseDistributionBatch:
    """Tests for pairwise_distribution_comparison_batch."""

    def test_pairwise_batch_basic(self, tmp_path: Any) -> None:
        """Ensure batch function returns DataFrame and caches to disk."""
        from neural_analysis.metrics.distributions import (
            pairwise_distribution_comparison_batch,
        )

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

        # Second call should reuse cached values without errors
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
