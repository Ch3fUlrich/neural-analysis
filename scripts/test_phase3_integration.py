"""Quick integration test for Phase 3 API.

Tests the new comparison functions and HDF5 storage to verify
everything works end-to-end.
"""

import tempfile
from pathlib import Path

import numpy as np

from neural_analysis.metrics import (
    ALL_METRICS,
    DISTRIBUTION_METRICS,
    POINT_TO_POINT_METRICS,
    SCALAR_METRICS,
    SHAPE_METRICS,
    compute_all_pairs,
    compute_between_distances,
    compute_within_distances,
)
from neural_analysis.utils import (
    load_comparison,
    query_comparisons,
    save_comparison,
)


def test_metric_constants():
    """Test that metric category constants are defined correctly."""
    print("Testing metric constants...")

    # Check set sizes
    assert len(POINT_TO_POINT_METRICS) == 4
    assert len(DISTRIBUTION_METRICS) == 3
    assert len(SHAPE_METRICS) == 3
    assert len(SCALAR_METRICS) == 6  # distribution + shape
    assert len(ALL_METRICS) == 10  # all three categories

    # Check no overlap between point-to-point and scalar
    assert len(POINT_TO_POINT_METRICS & SCALAR_METRICS) == 0

    print("✅ Metric constants validated")


def test_compute_within_distances():
    """Test within-dataset distance computation."""
    print("\nTesting compute_within_distances...")

    data = np.random.randn(50, 10)

    # Test summary statistic
    mean_dist = compute_within_distances(data, metric="euclidean")
    assert isinstance(mean_dist, float)
    assert mean_dist > 0

    # Test full matrix
    dist_matrix = compute_within_distances(data, metric="euclidean", return_matrix=True)
    assert isinstance(dist_matrix, np.ndarray)
    assert dist_matrix.shape == (50, 50)
    assert np.allclose(dist_matrix, dist_matrix.T)  # symmetric

    # Test validation: should reject distribution metrics
    try:
        compute_within_distances(data, metric="wasserstein")
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "mode='within'" in str(e)

    print("✅ compute_within_distances working correctly")


def test_compute_between_distances():
    """Test between-dataset distance computation."""
    print("\nTesting compute_between_distances...")

    data1 = np.random.randn(50, 10)
    data2 = np.random.randn(60, 10)
    data3 = np.random.randn(50, 10)  # Same shape for procrustes

    # Test point-to-point metric (summary)
    mean_dist = compute_between_distances(data1, data2, metric="euclidean")
    assert isinstance(mean_dist, float)
    assert mean_dist > 0

    # Test point-to-point metric (matrix)
    dist_matrix = compute_between_distances(
        data1, data2, metric="euclidean", return_matrix=True
    )
    assert isinstance(dist_matrix, np.ndarray)
    assert dist_matrix.shape == (50, 60)

    # Test distribution metric (always scalar)
    wass_dist = compute_between_distances(data1, data2, metric="wasserstein")
    assert isinstance(wass_dist, float)
    assert wass_dist > 0

    # Test shape metric (always scalar, needs same-shaped inputs)
    proc_dist = compute_between_distances(data1, data3, metric="procrustes")
    assert isinstance(proc_dist, float)
    assert proc_dist >= 0

    print("✅ compute_between_distances working correctly")


def test_compute_all_pairs():
    """Test all-pairs comparison."""
    print("\nTesting compute_all_pairs...")

    datasets = {
        "A": np.random.randn(30, 5),
        "B": np.random.randn(40, 5),
        "C": np.random.randn(35, 5),
    }

    # Test with distribution metric
    results = compute_all_pairs(datasets, metric="wasserstein", show_progress=False)

    # Check structure
    assert isinstance(results, dict)
    assert len(results) == 3
    for name in ["A", "B", "C"]:
        assert name in results
        assert len(results[name]) == 3

    # Check symmetry
    assert results["A"]["B"] == results["B"]["A"]
    assert results["A"]["C"] == results["C"]["A"]
    assert results["B"]["C"] == results["C"]["B"]

    # Test validation: should reject point-to-point metrics
    try:
        compute_all_pairs(datasets, metric="euclidean")
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "mode='all-pairs'" in str(e)

    print("✅ compute_all_pairs working correctly")


def test_hdf5_storage():
    """Test HDF5 comparison storage."""
    print("\nTesting HDF5 storage...")

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "test_comparisons.h5"

        # Save scalar comparison
        save_comparison(
            filepath=filepath,
            metric="wasserstein",
            dataset_i="exp1",
            dataset_j="exp2",
            mode="between",
            value=42.5,
            metadata={"n_samples_i": 100, "n_samples_j": 80},
        )

        # Save matrix comparison
        matrix = np.random.randn(10, 15)
        save_comparison(
            filepath=filepath,
            metric="euclidean",
            dataset_i="exp1",
            dataset_j="exp3",
            mode="between",
            value=matrix,
        )

        # Save dict comparison
        all_pairs_result = {
            "A": {"A": 0.0, "B": 1.5, "C": 2.0},
            "B": {"A": 1.5, "B": 0.0, "C": 1.8},
            "C": {"A": 2.0, "B": 1.8, "C": 0.0},
        }
        save_comparison(
            filepath=filepath,
            metric="wasserstein",
            dataset_i="batch",
            dataset_j="batch",
            mode="all-pairs",
            value=all_pairs_result,
        )

        # Load comparisons
        loaded_scalar = load_comparison(filepath, "wasserstein", "exp1", "exp2")
        assert isinstance(loaded_scalar, float)
        assert loaded_scalar == 42.5

        loaded_matrix = load_comparison(filepath, "euclidean", "exp1", "exp3")
        assert isinstance(loaded_matrix, np.ndarray)
        assert loaded_matrix.shape == (10, 15)
        assert np.allclose(loaded_matrix, matrix)

        loaded_dict = load_comparison(filepath, "wasserstein", "batch", "batch")
        assert isinstance(loaded_dict, dict)
        assert loaded_dict["A"]["B"] == 1.5

        # Query comparisons
        df = query_comparisons(filepath, metric="wasserstein")
        assert len(df) == 2  # Two wasserstein comparisons

        df_between = query_comparisons(filepath, mode="between")
        assert len(df_between) == 2  # Two between-mode comparisons

        df_all = query_comparisons(filepath)
        assert len(df_all) == 3  # Total three comparisons

        print("✅ HDF5 storage working correctly")


def main():
    """Run all integration tests."""
    print("=" * 60)
    print("Phase 3 API Integration Tests")
    print("=" * 60)

    test_metric_constants()
    test_compute_within_distances()
    test_compute_between_distances()
    test_compute_all_pairs()
    test_hdf5_storage()

    print("\n" + "=" * 60)
    print("✅ All integration tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
