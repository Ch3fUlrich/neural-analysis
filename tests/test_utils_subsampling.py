"""Tests for subsampling utilities."""

import numpy as np
import pytest

from neural_analysis.utils.subsampling import run_with_subsampling


class TestRunWithSubsampling:
    """Test suite for run_with_subsampling function."""

    def test_basic_subsampling(self) -> None:
        """Test basic subsampling functionality."""

        def euclidean_dist(a: np.ndarray, b: np.ndarray) -> float:
            return float(np.linalg.norm(a - b))

        arr1 = np.random.randn(100, 50)
        arr2 = np.random.randn(100, 50)

        values, meta = run_with_subsampling(
            func=euclidean_dist,
            arrays=(arr1, arr2),
            subsamples=[70],
            subsample_axes=[0],
            repeats=5,
            seed=42,
        )

        assert len(values) == 5
        assert all(isinstance(v, float) for v in values)
        assert "indices" in meta
        assert len(meta["indices"]) == 5
        assert all(isinstance(idx_list, list) for idx_list in meta["indices"])

    def test_different_array_sizes(self) -> None:
        """Test subsampling with arrays of different sizes."""

        def sum_diff(a: np.ndarray, b: np.ndarray) -> float:
            return float(np.sum(a) - np.sum(b))

        arr1 = np.random.randn(100, 50)
        arr2 = np.random.randn(80, 50)  # Different size

        values, meta = run_with_subsampling(
            func=sum_diff,
            arrays=(arr1, arr2),
            subsamples=[70],  # Subsample to 70 rows
            subsample_axes=[0],
            repeats=3,
            seed=42,
        )

        assert len(values) == 3
        # Check that indices are valid
        for idx_list in meta["indices"]:
            assert len(idx_list) == 2  # One per array
            for _arr_idx, indexers in enumerate(idx_list):
                assert isinstance(indexers, dict)
                assert 0 in indexers  # Axis 0
                assert len(indexers[0]) == 70  # Subsample size

    def test_multiple_axes(self) -> None:
        """Test subsampling along multiple axes."""

        def matrix_norm(a: np.ndarray, b: np.ndarray) -> float:
            # Use 2-norm (Euclidean norm) which works for any dimension
            # This is equivalent to Frobenius norm for 2D arrays
            return float(np.linalg.norm(a - b, ord=None))

        arr1 = np.random.randn(100, 50, 10)
        arr2 = np.random.randn(100, 50, 10)

        values, meta = run_with_subsampling(
            func=matrix_norm,
            arrays=(arr1, arr2),
            subsamples=[80, 40],  # Subsample rows and columns
            subsample_axes=[0, 1],
            repeats=3,
            seed=42,
        )

        assert len(values) == 3
        for idx_list in meta["indices"]:
            for _arr_idx, indexers in enumerate(idx_list):
                assert 0 in indexers
                assert 1 in indexers
                assert len(indexers[0]) == 80
                assert len(indexers[1]) == 40

    def test_reproducibility(self) -> None:
        """Test that same seed produces same results."""

        def simple_sum(a: np.ndarray, b: np.ndarray) -> float:
            return float(np.sum(a) + np.sum(b))

        arr1 = np.random.randn(50, 20)
        arr2 = np.random.randn(50, 20)

        values1, _ = run_with_subsampling(
            func=simple_sum,
            arrays=(arr1, arr2),
            subsamples=[30],
            subsample_axes=[0],
            repeats=5,
            seed=42,
        )

        values2, _ = run_with_subsampling(
            func=simple_sum,
            arrays=(arr1, arr2),
            subsamples=[30],
            subsample_axes=[0],
            repeats=5,
            seed=42,
        )

        np.testing.assert_array_equal(values1, values2)

    def test_empty_arrays_raises(self) -> None:
        """Test that empty arrays raise error."""

        def dummy_func(a: np.ndarray, b: np.ndarray) -> float:
            return 0.0

        with pytest.raises(ValueError, match="at least one array"):
            run_with_subsampling(
                func=dummy_func,
                arrays=(),
                subsamples=[10],
                subsample_axes=[0],
                repeats=1,
            )

    def test_mismatched_dimensions_raises(self) -> None:
        """Test that arrays with different dimensions raise error."""

        def dummy_func(a: np.ndarray, b: np.ndarray) -> float:
            return 0.0

        arr1 = np.random.randn(50, 20)
        arr2 = np.random.randn(50, 20, 10)  # Different ndim

        with pytest.raises(ValueError, match="same number of dimensions"):
            run_with_subsampling(
                func=dummy_func,
                arrays=(arr1, arr2),
                subsamples=[30],
                subsample_axes=[0],
                repeats=1,
            )

    def test_mismatched_subsample_params_raises(self) -> None:
        """Test that mismatched subsamples and axes raise error."""

        def dummy_func(a: np.ndarray, b: np.ndarray) -> float:
            return 0.0

        arr1 = np.random.randn(50, 20)
        arr2 = np.random.randn(50, 20)

        with pytest.raises(ValueError, match="same length"):
            run_with_subsampling(
                func=dummy_func,
                arrays=(arr1, arr2),
                subsamples=[30, 15],
                subsample_axes=[0],  # Mismatched length
                repeats=1,
            )

    def test_subsample_size_larger_than_array(self) -> None:
        """Test that subsample size larger than array size is handled."""

        def simple_sum(a: np.ndarray, b: np.ndarray) -> float:
            return float(np.sum(a) + np.sum(b))

        arr1 = np.random.randn(30, 20)
        arr2 = np.random.randn(30, 20)

        # Request subsample of 50, but array only has 30 rows
        values, meta = run_with_subsampling(
            func=simple_sum,
            arrays=(arr1, arr2),
            subsamples=[50],  # Larger than array size
            subsample_axes=[0],
            repeats=3,
            seed=42,
        )

        assert len(values) == 3
        # Should subsample to min(50, 30) = 30
        for idx_list in meta["indices"]:
            for indexers in idx_list:
                assert len(indexers[0]) == 30

    def test_single_array(self) -> None:
        """Test subsampling with a single array."""

        def array_norm(a: np.ndarray) -> float:
            return float(np.linalg.norm(a))

        arr = np.random.randn(100, 50)

        values, meta = run_with_subsampling(
            func=array_norm,
            arrays=(arr,),
            subsamples=[70],
            subsample_axes=[0],
            repeats=3,
            seed=42,
        )

        assert len(values) == 3
        assert len(meta["indices"]) == 3

    def test_three_arrays(self) -> None:
        """Test subsampling with three arrays."""

        def three_way_sum(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
            return float(np.sum(a) + np.sum(b) + np.sum(c))

        arr1 = np.random.randn(100, 50)
        arr2 = np.random.randn(100, 50)
        arr3 = np.random.randn(100, 50)

        values, meta = run_with_subsampling(
            func=three_way_sum,
            arrays=(arr1, arr2, arr3),
            subsamples=[70],
            subsample_axes=[0],
            repeats=3,
            seed=42,
        )

        assert len(values) == 3
        for idx_list in meta["indices"]:
            assert len(idx_list) == 3  # One indexer per array

    def test_empty_subsample_axes(self) -> None:
        """Test with empty subsample_axes (covers line 118 when indexers is empty)."""

        def simple_sum(a: np.ndarray) -> float:
            return float(np.sum(a))

        arr = np.random.randn(100, 50)

        # When both subsamples and subsample_axes are empty lists,
        # indexers will be empty, triggering line 118
        values, meta = run_with_subsampling(
            func=simple_sum,
            arrays=(arr,),
            subsamples=[],  # Empty - no subsampling
            subsample_axes=[],  # Empty - no axes
            repeats=3,
            seed=42,
        )

        assert len(values) == 3
        # All values should be the same since no subsampling occurred
        assert all(v == values[0] for v in values)
