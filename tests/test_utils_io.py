"""Tests for I/O utilities (NumPy and HDF5)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neural_analysis.utils.io import (
    load_array,
    load_hdf5,
    save_array,
    save_hdf5,
    update_array,
)

# ============================================================================
# NumPy I/O Tests
# ============================================================================


class TestSaveLoadArray:
    """Tests for save_array and load_array functions."""

    def test_save_load_single_array_npy(self, tmp_path) -> None:
        """Test saving and loading a single array in .npy format."""
        arr = np.random.rand(10, 5)
        path = tmp_path / "test.npy"

        save_array(path, arr)
        loaded = load_array(path)

        assert loaded is not None
        assert np.allclose(arr, loaded)

    def test_save_load_dict_npz(self, tmp_path) -> None:
        """Test saving and loading multiple arrays in .npz format."""
        data = {
            "embeddings": np.random.rand(100, 2),
            "labels": np.arange(100),
            "scores": np.random.rand(100),
        }
        path = tmp_path / "test.npz"

        save_array(path, data)
        loaded = load_array(path)

        assert loaded is not None
        assert isinstance(loaded, dict)
        assert set(loaded.keys()) == set(data.keys())
        for key in data:
            assert np.allclose(data[key], loaded[key])

    def test_save_array_auto_extension(self, tmp_path) -> None:
        """Test automatic extension assignment based on data type."""
        # Single array → .npy
        arr = np.ones(10)
        path1 = tmp_path / "single"
        save_array(path1, arr)
        assert (tmp_path / "single.npy").exists()

        # Dict → .npz
        data = {"arr1": np.ones(5), "arr2": np.zeros(3)}
        path2 = tmp_path / "multi"
        save_array(path2, data)
        assert (tmp_path / "multi.npz").exists()

    def test_save_array_creates_directories(self, tmp_path) -> None:
        """Test that parent directories are created automatically."""
        deep_path = tmp_path / "level1" / "level2" / "level3" / "data.npy"
        arr = np.ones(5)

        save_array(deep_path, arr)

        assert deep_path.exists()
        loaded = load_array(deep_path)
        assert np.allclose(arr, loaded)

    def test_save_array_overwrite_protection(self, tmp_path) -> None:
        """Test that overwrite protection works."""
        path = tmp_path / "test.npy"
        arr1 = np.ones(5)
        arr2 = np.zeros(5)

        # Save first array
        save_array(path, arr1)

        # Try to save second without overwrite - should raise
        with pytest.raises(ValueError, match="already exists"):
            save_array(path, arr2, allow_overwrite=False)

        # With overwrite should work
        save_array(path, arr2, allow_overwrite=True)
        loaded = load_array(path)
        assert np.allclose(arr2, loaded)


class TestUpdateArray:
    """Tests for update_array function."""

    def test_update_existing_npz(self, tmp_path) -> None:
        """Test updating existing .npz file adds new arrays."""
        path = tmp_path / "test.npz"
        initial_data = {"arr1": np.ones(5)}
        new_data = {"arr2": np.zeros(3)}

        save_array(path, initial_data)
        update_array(path, new_data)

        loaded = load_array(path)
        assert "arr1" in loaded
        assert "arr2" in loaded
        assert np.allclose(loaded["arr1"], np.ones(5))
        assert np.allclose(loaded["arr2"], np.zeros(3))


class TestSaveLoadHDF5:
    """Tests for save_hdf5 and load_hdf5 functions."""

    def test_save_load_dataframe(self, tmp_path) -> None:
        """Test saving and loading a DataFrame."""
        df = pd.DataFrame(
            {
                "item_i": ["A", "B", "C"],
                "item_j": ["X", "Y", "Z"],
                "value": [1.0, 2.0, 3.0],
            }
        )
        labels = ["A", "B", "C", "X", "Y", "Z"]
        path = tmp_path / "test.h5"

        save_hdf5(path, df, labels=labels)
        loaded_df, loaded_labels = load_hdf5(path)

        assert loaded_df is not None
        pd.testing.assert_frame_equal(df, loaded_df)
        assert loaded_labels == labels

    def test_save_load_array(self, tmp_path) -> None:
        """Test saving and loading a numpy array."""
        arr = np.random.rand(50, 10)
        labels = [f"neuron_{i}" for i in range(50)]
        path = tmp_path / "test.h5"

        save_hdf5(path, arr, labels=labels)
        loaded_arr, loaded_labels = load_hdf5(path)

        assert loaded_arr is not None
        assert np.allclose(arr, loaded_arr)
        assert loaded_labels == labels

    def test_load_hdf5_filter_pairs(self, tmp_path) -> None:
        """Test filtering DataFrame by item pairs."""
        df = pd.DataFrame(
            {
                "item_i": ["A", "B", "C", "D"],
                "item_j": ["X", "Y", "Z", "W"],
                "value": [1.0, 2.0, 3.0, 4.0],
            }
        )
        path = tmp_path / "test.h5"

        save_hdf5(path, df)

        # Load with filter
        filter_pairs = [("A", "X"), ("C", "Z")]
        loaded_df, _ = load_hdf5(path, filter_pairs=filter_pairs)

        assert loaded_df is not None
        assert len(loaded_df) == 2
        assert set(loaded_df["item_i"]) == {"A", "C"}

    def test_load_hdf5_missing_file(self, tmp_path) -> None:
        """Test loading non-existent file returns None."""
        path = tmp_path / "nonexistent.h5"
        data, labels = load_hdf5(path)

        assert data is None
        assert labels == []


class TestComparisonBatch:
    """Tests for save_comparison_batch and get_missing_comparisons functions."""

    def test_save_comparison_batch_creates_dataframe(self, tmp_path) -> None:
        """Test that save_comparison_batch creates a DataFrame from result rows."""
        from neural_analysis.utils.io import save_comparison_batch

        result_rows = [
            {
                "dataset_i": "A",
                "dataset_j": "B",
                "metric": "wasserstein",
                "value": 0.5,
                "pairs": None,
            },
            {
                "dataset_i": "C",
                "dataset_j": "D",
                "metric": "procrustes",
                "value": 0.3,
                "pairs": {"0,0": 0.1, "1,1": 0.2},
            },
        ]

        df_results = save_comparison_batch(result_rows, None, None)

        assert len(df_results) == 2
        assert "pairs" in df_results.columns
        assert df_results.loc[0, "pairs"] is None
        assert isinstance(df_results.loc[1, "pairs"], dict)
        assert df_results.loc[1, "pairs"]["0,0"] == 0.1

    def test_save_comparison_batch_appends(self, tmp_path) -> None:
        """Test that save_comparison_batch appends to existing DataFrame."""
        from neural_analysis.utils.io import save_comparison_batch

        initial_rows = [
            {
                "dataset_i": "A",
                "dataset_j": "B",
                "metric": "wasserstein",
                "value": 0.5,
                "pairs": None,
            }
        ]

        df_results = save_comparison_batch(initial_rows, None, None)
        assert len(df_results) == 1

        more_rows = [
            {
                "dataset_i": "C",
                "dataset_j": "D",
                "metric": "kolmogorov_smirnov",
                "value": 0.7,
                "pairs": None,
            }
        ]

        df_results = save_comparison_batch(more_rows, df_results, None)
        assert len(df_results) == 2

    def test_save_comparison_batch_saves_to_file(self, tmp_path) -> None:
        """Test that save_comparison_batch saves to HDF5 file."""
        from neural_analysis.utils.io import save_comparison_batch

        result_rows = [
            {
                "dataset_i": "A",
                "dataset_j": "B",
                "metric": "wasserstein",
                "value": 0.5,
                "pairs": None,
            }
        ]

        save_path = tmp_path / "test_results.h5"
        df_results = save_comparison_batch(result_rows, None, save_path)

        assert save_path.exists()
        loaded_df, _ = load_hdf5(save_path)

        # Compare columns and values separately to handle None comparison
        assert list(df_results.columns) == list(loaded_df.columns)
        assert len(df_results) == len(loaded_df)
        assert df_results["dataset_i"].tolist() == loaded_df["dataset_i"].tolist()
        assert df_results["metric"].tolist() == loaded_df["metric"].tolist()
        assert df_results["value"].tolist() == loaded_df["value"].tolist()

    def test_save_comparison_batch_empty_rows(self, tmp_path) -> None:
        """Test that save_comparison_batch handles empty result rows."""
        from neural_analysis.utils.io import save_comparison_batch

        # With no existing results
        df_results = save_comparison_batch([], None, None)
        assert len(df_results) == 0

        # With existing results
        initial_rows = [
            {
                "dataset_i": "A",
                "dataset_j": "B",
                "metric": "wasserstein",
                "value": 0.5,
                "pairs": None,
            }
        ]
        df_existing = pd.DataFrame(initial_rows)
        df_results = save_comparison_batch([], df_existing, None)
        pd.testing.assert_frame_equal(df_results, df_existing)

    def test_get_missing_comparisons_all_missing(self, tmp_path) -> None:
        """Test get_missing_comparisons when no cache exists."""
        from neural_analysis.utils.io import get_missing_comparisons

        item_pairs = [("A", "B"), ("C", "D"), ("E", "F")]
        metrics_dict = {"wasserstein": {}, "procrustes": {}}

        missing = get_missing_comparisons(item_pairs, metrics_dict, None)

        expected_count = len(item_pairs) * len(metrics_dict)
        assert len(missing) == expected_count
        assert ("A", "B", "wasserstein") in missing
        assert ("C", "D", "procrustes") in missing

    def test_get_missing_comparisons_partial_cache(self, tmp_path) -> None:
        """Test get_missing_comparisons with partial cached results."""
        from neural_analysis.utils.io import get_missing_comparisons

        item_pairs = [("A", "B"), ("C", "D"), ("E", "F")]
        metrics_dict = {"wasserstein": {}, "procrustes": {}}

        # Create partial cache
        df_partial = pd.DataFrame(
            [
                {
                    "dataset_i": "A",
                    "dataset_j": "B",
                    "metric": "wasserstein",
                    "value": 0.5,
                },
                {
                    "dataset_i": "C",
                    "dataset_j": "D",
                    "metric": "wasserstein",
                    "value": 0.3,
                },
            ]
        )

        missing = get_missing_comparisons(item_pairs, metrics_dict, df_partial)

        # Should be missing: A-B procrustes, C-D procrustes, E-F both
        assert len(missing) == 4
        assert ("A", "B", "procrustes") in missing
        assert ("E", "F", "wasserstein") in missing

    def test_get_missing_comparisons_empty_dataframe(self, tmp_path) -> None:
        """Test get_missing_comparisons with empty DataFrame."""
        from neural_analysis.utils.io import get_missing_comparisons

        item_pairs = [("A", "B")]
        metrics_dict = {"wasserstein": {}}

        df_empty = pd.DataFrame()
        missing = get_missing_comparisons(item_pairs, metrics_dict, df_empty)

        assert len(missing) == 1
        assert ("A", "B", "wasserstein") in missing
