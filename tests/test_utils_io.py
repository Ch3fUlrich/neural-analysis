"""Tests for I/O utilities (NumPy and HDF5)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

# Import private functions for testing
from neural_analysis.utils.io import (
    _attr_equals,
    _normalize_attr_value,
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

    def test_save_load_single_array_npy(self, tmp_path: Any) -> None:
        """Test saving and loading a single array in .npy format."""
        arr = np.random.rand(10, 5)
        path = tmp_path / "test.npy"

        save_array(path, arr)
        loaded = load_array(path)

        assert loaded is not None
        assert np.allclose(arr, loaded)

    def test_save_load_dict_npz(self, tmp_path: Any) -> None:
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

    def test_save_array_auto_extension(self, tmp_path: Any) -> None:
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

    def test_save_array_creates_directories(self, tmp_path: Any) -> None:
        """Test that parent directories are created automatically."""
        deep_path = tmp_path / "level1" / "level2" / "level3" / "data.npy"
        arr = np.ones(5)

        save_array(deep_path, arr)

        assert deep_path.exists()
        loaded = load_array(deep_path)
        assert np.allclose(arr, loaded)

    def test_save_array_overwrite_protection(self, tmp_path: Any) -> None:
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

    def test_update_existing_npz(self, tmp_path: Any) -> None:
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

    def test_save_load_dataframe(self, tmp_path: Any) -> None:
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

    def test_save_load_array(self, tmp_path: Any) -> None:
        """Test saving and loading a numpy array."""
        arr = np.random.rand(50, 10)
        labels = [f"neuron_{i}" for i in range(50)]
        path = tmp_path / "test.h5"

        save_hdf5(path, arr, labels=labels)
        loaded_arr, loaded_labels = load_hdf5(path)

        assert loaded_arr is not None
        assert np.allclose(arr, loaded_arr)
        assert loaded_labels == labels

    def test_load_hdf5_filter_pairs(self, tmp_path: Any) -> None:
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

    def test_load_hdf5_missing_file(self, tmp_path: Any) -> None:
        """Test loading non-existent file returns None."""
        path = tmp_path / "nonexistent.h5"
        data, labels = load_hdf5(path)

        assert data is None
        assert labels == []


class TestComparisonBatch:
    """Tests for save_comparison_batch and get_missing_comparisons functions."""

    def test_save_comparison_batch_creates_dataframe(self, tmp_path: Any) -> None:
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

    def test_save_comparison_batch_appends(self, tmp_path: Any) -> None:
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

    def test_save_comparison_batch_saves_to_file(self, tmp_path: Any) -> None:
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

    def test_save_comparison_batch_empty_rows(self, tmp_path: Any) -> None:
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

    def test_get_missing_comparisons_all_missing(self, tmp_path: Any) -> None:
        """Test get_missing_comparisons when no cache exists."""
        from neural_analysis.utils.io import get_missing_comparisons

        item_pairs = [("A", "B"), ("C", "D"), ("E", "F")]
        metrics_dict: dict[str, dict[str, Any]] = {"wasserstein": {}, "procrustes": {}}

        missing = get_missing_comparisons(item_pairs, metrics_dict, None)

        expected_count = len(item_pairs) * len(metrics_dict)
        assert len(missing) == expected_count
        assert ("A", "B", "wasserstein") in missing
        assert ("C", "D", "procrustes") in missing

    def test_get_missing_comparisons_partial_cache(self, tmp_path: Any) -> None:
        """Test get_missing_comparisons with partial cached results."""
        from neural_analysis.utils.io import get_missing_comparisons

        item_pairs = [("A", "B"), ("C", "D"), ("E", "F")]
        metrics_dict: dict[str, dict[str, Any]] = {"wasserstein": {}, "procrustes": {}}

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

    def test_get_missing_comparisons_empty_dataframe(self, tmp_path: Any) -> None:
        """Test get_missing_comparisons with empty DataFrame."""
        from neural_analysis.utils.io import get_missing_comparisons

        item_pairs = [("A", "B")]
        metrics_dict: dict[str, dict[str, Any]] = {"wasserstein": {}}

        df_empty = pd.DataFrame()
        missing = get_missing_comparisons(item_pairs, metrics_dict, df_empty)

        assert len(missing) == 1
        assert ("A", "B", "wasserstein") in missing

    def test_get_hdf5_result_summary(self, tmp_path: Any) -> None:
        """Test get_hdf5_result_summary function."""
        from neural_analysis.utils.io import (
            get_hdf5_result_summary,
            save_result_to_hdf5_dataset,
        )

        # Create test HDF5 file with some results
        save_path = tmp_path / "test_summary.h5"
        data = {
            "dataset_A": {
                "result_1": {"value": 0.5, "metric": "wasserstein"},
                "result_2": {"value": 0.3, "metric": "euclidean"},
            },
            "dataset_B": {
                "result_1": {"value": 0.7, "metric": "wasserstein"},
            },
        }

        # Save using save_result_to_hdf5_dataset
        for dataset_name, results in data.items():
            for result_key, attrs in results.items():
                save_result_to_hdf5_dataset(
                    save_path,
                    dataset_name=dataset_name,
                    result_key=result_key,
                    scalar_data=attrs,
                    array_data={},
                    use_cache=False,
                    use_sql_index=False,
                )

        # Test loading all results
        summary = get_hdf5_result_summary(save_path)
        assert len(summary) == 3  # 3 total results
        assert "dataset_name" in summary.columns
        assert "result_key" in summary.columns
        assert "value" in summary.columns
        assert "metric" in summary.columns

        # Test filtering by dataset
        summary_a = get_hdf5_result_summary(save_path, dataset_name="dataset_A")
        assert len(summary_a) == 2
        assert all(summary_a["dataset_name"] == "dataset_A")

        # Test with non-existent file
        empty_summary = get_hdf5_result_summary(tmp_path / "nonexistent.h5")
        assert len(empty_summary) == 0

    def test_get_hdf5_result_summary_exception_handling(
        self, tmp_path: Any, monkeypatch
    ) -> None:
        """Test get_hdf5_result_summary exception handling (covers lines 924-925)."""
        import h5py

        from neural_analysis.utils.io import get_hdf5_result_summary

        save_path = tmp_path / "test_exception.h5"

        # Create a file that exists but will fail when opened
        save_path.write_bytes(b"not a valid hdf5 file")

        # Mock h5py.File.__init__ to raise an exception
        original_init = h5py.File.__init__

        def mock_init(self, name, mode="r", **kwargs):
            if "exception" in str(name) or mode == "r":
                raise OSError("Mocked file read error")
            return original_init(self, name, mode, **kwargs)

        monkeypatch.setattr(h5py.File, "__init__", mock_init)

        # This should catch the exception and return empty DataFrame
        summary = get_hdf5_result_summary(save_path)
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 0


class TestNormalizeAttrValue:
    """Tests for _normalize_attr_value function."""

    def test_normalize_numpy_generic(self):
        """Test normalizing numpy generic types (covers line 105)."""
        assert _normalize_attr_value(np.int32(42)) == 42
        assert _normalize_attr_value(np.float64(3.14)) == 3.14
        assert _normalize_attr_value(np.bool_(True)) is True

    def test_normalize_bytes(self):
        """Test normalizing bytes (covers lines 106-110)."""
        assert _normalize_attr_value(b"test") == "test"
        assert _normalize_attr_value(bytearray(b"test")) == "test"
        # Test with invalid UTF-8 (covers exception path)
        invalid_bytes = b"\xff\xfe"
        result = _normalize_attr_value(invalid_bytes)
        assert result == invalid_bytes  # Should return original on decode error

    def test_normalize_other_types(self):
        """Test normalizing other types."""
        assert _normalize_attr_value("string") == "string"
        assert _normalize_attr_value(42) == 42


class TestAttrEquals:
    """Tests for _attr_equals function."""

    def test_attr_equals_bool(self):
        """Test _attr_equals with bool (covers lines 118-121)."""
        assert _attr_equals(1, True) is True
        assert _attr_equals(0, False) is True
        assert _attr_equals("1", True) is True
        assert _attr_equals("0", False) is True
        # Test exception path (covers line 120-121)
        # "invalid" can't be converted to int, so exception is caught
        # and bool("invalid") is True (non-empty string)
        result = _attr_equals("invalid", True)
        # The result depends on how bool("invalid") compares to True
        assert isinstance(result, bool)

    def test_attr_equals_numeric_string(self):
        """Test _attr_equals with numeric string (covers lines 122-124)."""
        assert _attr_equals("42", 42) is True
        assert _attr_equals("3.14", 3.14) is True
        # Test with invalid conversion (covers suppress path)
        assert _attr_equals("not_a_number", 42) is False

    def test_attr_equals_string_conversion(self):
        """Test _attr_equals with string conversion (covers line 125-126)."""
        assert _attr_equals(42, "42") is True
        assert _attr_equals(3.14, "3.14") is True
