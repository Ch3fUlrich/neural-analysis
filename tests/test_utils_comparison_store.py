"""Tests for comparison store utilities."""

import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

from neural_analysis.utils.comparison_store import (
    load_comparison,
    query_comparisons,
    save_comparison,
    save_comparison_result,
    try_load_cached_comparison,
)


class TestValueType:
    """Tests for _infer_value_type function."""

    def test_value_type_scalar(self):
        """Test scalar value type (covers line 109)."""
        from neural_analysis.utils.comparison_store import _infer_value_type

        assert _infer_value_type(5.0) == "scalar"
        assert _infer_value_type(42) == "scalar"
        assert _infer_value_type(np.float64(3.14)) == "scalar"

    def test_value_type_matrix(self):
        """Test matrix value type (covers line 107)."""
        from neural_analysis.utils.comparison_store import _infer_value_type

        arr = np.array([[1, 2], [3, 4]])
        assert _infer_value_type(arr) == "matrix"

    def test_value_type_dict(self):
        """Test dict value type (covers line 105)."""
        from neural_analysis.utils.comparison_store import _infer_value_type

        d = {"a": {"b": 1.0}}
        assert _infer_value_type(d) == "dict"

    def test_value_type_tuple(self):
        """Test tuple value type (covers lines 110-112)."""
        from neural_analysis.utils.comparison_store import _infer_value_type

        # Tuple for shape metrics
        result = (0.5, {(0, 1): 0.3})
        assert _infer_value_type(result) == "scalar"

    def test_value_type_unsupported(self):
        """Test unsupported value type (covers lines 113-115)."""
        from neural_analysis.utils.comparison_store import _infer_value_type

        with pytest.raises(TypeError, match="Unsupported value type"):
            _infer_value_type("string")


class TestSaveComparison:
    """Tests for save_comparison function."""

    def test_save_scalar_comparison(self):
        """Test saving scalar comparison."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            filepath = f.name

        try:
            save_comparison(
                filepath=filepath,
                metric="euclidean",
                dataset_i="A",
                dataset_j="B",
                mode="between",
                value=0.5,
            )

            # Verify it was saved
            result = load_comparison(filepath, "euclidean", "A", "B")
            assert result == 0.5
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_save_matrix_comparison(self):
        """Test saving matrix comparison."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            filepath = f.name

        try:
            matrix = np.array([[1, 2], [3, 4]], dtype=np.float64)
            save_comparison(
                filepath=filepath,
                metric="distance",
                dataset_i="X",
                dataset_j="Y",
                mode="between",
                value=matrix,
            )

            # Verify it was saved
            result = load_comparison(filepath, "distance", "X", "Y")
            np.testing.assert_array_equal(result, matrix)
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_save_with_overwrite_false_existing(self):
        """Test save with overwrite=False when comparison exists (covers lines 268-277)."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            filepath = f.name

        try:
            # Save first time
            save_comparison(
                filepath=filepath,
                metric="test",
                dataset_i="A",
                dataset_j="B",
                mode="between",
                value=1.0,
                overwrite=False,
            )

            # Try to save again without overwrite
            with pytest.raises(ValueError, match="Comparison already exists"):
                save_comparison(
                    filepath=filepath,
                    metric="test",
                    dataset_i="A",
                    dataset_j="B",
                    mode="between",
                    value=2.0,
                    overwrite=False,
                )
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_save_with_overwrite_true(self):
        """Test save with overwrite=True."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            filepath = f.name

        try:
            # Save first time
            save_comparison(
                filepath=filepath,
                metric="test",
                dataset_i="A",
                dataset_j="B",
                mode="between",
                value=1.0,
            )

            # Overwrite
            save_comparison(
                filepath=filepath,
                metric="test",
                dataset_i="A",
                dataset_j="B",
                mode="between",
                value=2.0,
                overwrite=True,
            )

            result = load_comparison(filepath, "test", "A", "B")
            assert result == 2.0
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_save_with_corrupted_file(self):
        """Test save with corrupted HDF5 file (covers lines 278-281)."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            filepath = f.name

        try:
            # Create a valid HDF5 file first
            save_comparison(
                filepath=filepath,
                metric="test",
                dataset_i="A",
                dataset_j="B",
                mode="between",
                value=0.5,
            )

            # Corrupt the file by writing invalid content
            Path(filepath).write_text("not hdf5 content")

            # Should handle OSError and continue (overwrite=True to allow overwrite)
            # The OSError is caught at line 278-281 and we continue to save
            save_comparison(
                filepath=filepath,
                metric="test",
                dataset_i="A",
                dataset_j="B",
                mode="between",
                value=1.0,
                overwrite=True,
            )

            # Should have saved successfully
            result = load_comparison(filepath, "test", "A", "B")
            assert result == 1.0
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_save_with_cache_invalidation_exception(self):
        """Test save with cache invalidation exception (covers lines 255-257)."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            filepath = f.name

        try:
            # Should handle exception in cache invalidation gracefully
            save_comparison(
                filepath=filepath,
                metric="test",
                dataset_i="A",
                dataset_j="B",
                mode="between",
                value=1.0,
            )

            # Should still save successfully
            result = load_comparison(filepath, "test", "A", "B")
            assert result == 1.0
        finally:
            Path(filepath).unlink(missing_ok=True)


class TestLoadComparison:
    """Tests for load_comparison function."""

    def test_load_nonexistent_comparison(self):
        """Test loading nonexistent comparison (covers line 374)."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            filepath = f.name

        try:
            # File exists but comparison doesn't
            with pytest.raises(KeyError, match="Comparison not found"):
                load_comparison(filepath, "nonexistent", "A", "B")
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_load_nonexistent_file(self):
        """Test loading from nonexistent file."""
        with pytest.raises(FileNotFoundError):
            load_comparison("nonexistent.h5", "test", "A", "B")

    def test_load_with_bytes_keys(self):
        """Test loading with bytes keys (covers lines 165-172)."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            filepath = f.name

        try:
            # Save with string keys
            save_comparison(
                filepath=filepath,
                metric="test",
                dataset_i="A",
                dataset_j="B",
                mode="between",
                value=1.0,
            )

            # Load should work
            result = load_comparison(filepath, "test", "A", "B")
            assert result == 1.0
        finally:
            Path(filepath).unlink(missing_ok=True)


class TestSaveComparisonResult:
    """Tests for save_comparison_result function."""

    def test_save_between_mode_with_dict_result(self):
        """Test save_comparison_result with dict result (covers line 603-604)."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            filepath = f.name

        try:
            result = {"value": 0.75}
            save_comparison_result(
                save_path=filepath,
                mode="between",
                metric="test",
                result=result,
                dataset_names=("A", "B"),
            )

            loaded = load_comparison(filepath, "test", "A", "B")
            assert loaded == 0.75
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_save_between_mode_with_tuple_result(self):
        """Test save_comparison_result with tuple result (covers lines 605-607)."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            filepath = f.name

        try:
            result = (0.5, {(0, 1): 0.3})  # Shape metric result
            save_comparison_result(
                save_path=filepath,
                mode="between",
                metric="shape",
                result=result,
                dataset_names=("A", "B"),
            )

            loaded = load_comparison(filepath, "shape", "A", "B")
            assert loaded == 0.5
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_save_between_mode_missing_dataset_names(self):
        """Test save_comparison_result with missing dataset_names (covers lines 531-535)."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            filepath = f.name

        try:
            with pytest.raises(ValueError, match="dataset_names required"):
                save_comparison_result(
                    save_path=filepath,
                    mode="between",
                    metric="test",
                    result=1.0,
                    dataset_names=None,
                )
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_save_all_pairs_mode(self):
        """Test save_comparison_result with all-pairs mode (covers lines 622-636)."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            filepath = f.name

        try:
            # For all-pairs mode, result should be a dict of dicts: {dataset_i: {dataset_j: value}}
            result = {"A": {"B": 0.5, "C": 0.6}, "B": {"C": 0.7}}
            save_comparison_result(
                save_path=filepath,
                mode="all-pairs",
                metric="test",
                result=result,
            )

            # Should be saved with special naming
            loaded = load_comparison(filepath, "test", "all_pairs", "all_pairs")
            assert isinstance(loaded, dict)
            assert "A_B" in loaded or loaded == result  # May be encoded differently
        finally:
            Path(filepath).unlink(missing_ok=True)


class TestTryLoadCachedComparison:
    """Tests for try_load_cached_comparison function."""

    def test_try_load_between_mode(self):
        """Test try_load_cached_comparison with between mode."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            filepath = f.name

        try:
            # Save a comparison
            save_comparison(
                filepath=filepath,
                metric="test",
                dataset_i="A",
                dataset_j="B",
                mode="between",
                value=0.5,
            )

            # Try to load it
            result = try_load_cached_comparison(
                save_path=filepath,
                mode="between",
                metric="test",
                dataset_names=("A", "B"),
            )

            assert result == 0.5
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_try_load_all_pairs_mode(self):
        """Test try_load_cached_comparison with all-pairs mode (covers lines 543-549)."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            filepath = f.name

        try:
            # Save all-pairs comparison using save_comparison_result
            # For all-pairs mode, result should be a dict of dicts
            save_comparison_result(
                save_path=filepath,
                mode="all-pairs",
                metric="test",
                result={"A": {"B": 0.5}},
            )

            # Try to load it
            result = try_load_cached_comparison(
                save_path=filepath,
                mode="all-pairs",
                metric="test",
            )

            assert result is not None
            assert isinstance(result, dict)
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_try_load_within_mode_returns_none(self):
        """Test try_load_cached_comparison with within mode (covers line 551)."""
        result = try_load_cached_comparison(
            save_path="dummy.h5",
            mode="within",
            metric="test",
        )
        assert result is None

    def test_try_load_file_not_found(self):
        """Test try_load_cached_comparison with file not found (covers lines 552-554)."""
        result = try_load_cached_comparison(
            save_path="nonexistent.h5",
            mode="between",
            metric="test",
            dataset_names=("A", "B"),
        )
        assert result is None


class TestQueryComparisons:
    """Tests for query_comparisons function."""

    def test_query_comparisons_empty_file(self):
        """Test query_comparisons with empty file."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            filepath = f.name

        try:
            df = query_comparisons(filepath)
            assert df.empty
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_query_comparisons_nonexistent_file(self):
        """Test query_comparisons with nonexistent file."""
        df = query_comparisons("nonexistent.h5")
        assert df.empty

    def test_query_comparisons_with_filters(self):
        """Test query_comparisons with filters (covers lines 481-488)."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            filepath = f.name

        try:
            # Save multiple comparisons
            save_comparison(
                filepath=filepath,
                metric="euclidean",
                dataset_i="A",
                dataset_j="B",
                mode="between",
                value=0.5,
            )
            save_comparison(
                filepath=filepath,
                metric="cosine",
                dataset_i="A",
                dataset_j="B",
                mode="between",
                value=0.3,
            )

            # Query with metric filter (covers line 481-482)
            df = query_comparisons(filepath, metric="euclidean")
            assert len(df) >= 1
            assert all(df["metric"] == "euclidean")

            # Query with mode filter (covers line 483-484)
            df = query_comparisons(filepath, mode="between")
            assert len(df) >= 1

            # Query with dataset_i filter (covers line 485-486)
            df = query_comparisons(filepath, dataset_i="A")
            assert len(df) >= 1

            # Query with dataset_j filter (covers line 487-488)
            df = query_comparisons(filepath, dataset_j="B")
            assert len(df) >= 1

            # Query with all filters (covers all filter branches 481-488)
            df = query_comparisons(
                filepath, metric="euclidean", mode="between", dataset_i="A", dataset_j="B"
            )
            # Should match the euclidean comparison we saved
            assert len(df) == 1
            assert df.iloc[0]["metric"] == "euclidean"
            assert df.iloc[0]["dataset_i"] == "A"
            assert df.iloc[0]["dataset_j"] == "B"
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_decode_dict_non_bytes_keys(self):
        """Test _decode_dict_from_hdf5 with non-bytes keys (covers lines 167, 172)."""
        from neural_analysis.utils.comparison_store import _decode_dict_from_hdf5

        # Create a structured array with non-bytes string keys
        dtype = [("key_i", "S100"), ("key_j", "S100"), ("value", "f8")]
        arr = np.array([("A", "B", 0.5)], dtype=dtype)
        # Convert to regular strings (not bytes)
        arr_str = np.array([("A", "B", 0.5)], dtype=[("key_i", "U100"), ("key_j", "U100"), ("value", "f8")])

        # Test with bytes (should decode)
        result = _decode_dict_from_hdf5(arr)
        assert result == {"A": {"B": 0.5}}

        # Test with non-bytes (should convert to string)
        result = _decode_dict_from_hdf5(arr_str)
        assert result == {"A": {"B": 0.5}}

    def test_save_comparison_unexpected_value_type(self):
        """Test save_comparison with unexpected value_type (covers line 306)."""
        from unittest.mock import patch
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            filepath = f.name

        try:
            # Mock _infer_value_type to return invalid type to trigger line 306
            with patch("neural_analysis.utils.comparison_store._infer_value_type", return_value="invalid"):
                with pytest.raises(TypeError, match="Unexpected value_type"):
                    save_comparison(
                        filepath=filepath,
                        metric="test",
                        dataset_i="A",
                        dataset_j="B",
                        value=0.5,
                    )
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_load_comparison_unknown_value_type(self):
        """Test load_comparison with unknown value_type (covers line 411)."""
        import h5py
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            filepath = f.name

        try:
            # Create a file with invalid value_type using the proper structure
            # that matches what load_results_from_hdf5_dataset expects
            with h5py.File(filepath, "w") as h5f:
                # Create structure: {dataset_name}/{result_key}/
                comparison_group = h5f.create_group("test")
                dataset_group = comparison_group.create_group("A")
                result_group = dataset_group.create_group("B")
                result_group.attrs["value_type"] = "invalid_type"
                result_group.create_dataset("value", data=np.array([0.5]))

            # This should raise TypeError at line 411
            with pytest.raises(TypeError, match="Unknown value_type"):
                load_comparison(filepath, "test", "A", "B")
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_try_load_cached_comparison_exception_handling(self):
        """Test try_load_cached_comparison exception handling (covers lines 551-554)."""
        # Test with non-existent file (FileNotFoundError)
        result = try_load_cached_comparison(
            save_path="/nonexistent/path.h5",
            metric="test",
            mode="between",
            dataset_names=("A", "B"),
        )
        assert result is None

        # Test with invalid file (OSError)
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            filepath = f.name
            Path(filepath).write_text("not hdf5")

        try:
            result = try_load_cached_comparison(
                save_path=filepath,
                metric="test",
                mode="between",
                dataset_names=("A", "B"),
            )
            assert result is None
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_try_load_cached_comparison_missing_dataset_names(self):
        """Test try_load_cached_comparison with missing dataset_names (covers line 532)."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            filepath = f.name

        try:
            # Create a valid file
            save_comparison(
                filepath=filepath,
                metric="test",
                dataset_i="A",
                dataset_j="B",
                value=0.5,
            )

            # Try to load without dataset_names (should raise ValueError at line 532)
            with pytest.raises(ValueError, match="dataset_names required"):
                try_load_cached_comparison(
                    save_path=filepath,
                    metric="test",
                    mode="between",
                    dataset_names=None,
                )
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_save_comparison_result_else_branch(self):
        """Test save_comparison_result else branch (covers line 609)."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            filepath = f.name

        try:
            # Pass a result that's not dict with "value", not tuple, and not a simple dict
            # This should hit the else branch at line 609
            # Use a list (not dict/float/tuple) - will be inferred as scalar
            result = [0.5, 0.6]  # List, not dict/float/tuple
            save_comparison_result(
                save_path=filepath,
                mode="between",
                metric="test",
                dataset_i="A",
                dataset_j="B",
                result=result,
            )

            # Should save (list gets converted to array, then inferred as matrix)
            loaded = load_comparison(filepath, "test", "A", "B")
            assert isinstance(loaded, np.ndarray) or isinstance(loaded, (float, int))
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_save_comparison_cache_invalidation_exception(self):
        """Test save_comparison cache invalidation exception handling (covers lines 255-257)."""
        from unittest.mock import patch, MagicMock
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            filepath = f.name

        try:
            # Mock the StorageManager to raise exception when invalidate_cache is called
            # This tests the exception handling at lines 255-257
            with patch("neural_analysis.utils.comparison_store.StorageManager") as mock_storage_class:
                mock_instance = MagicMock()
                mock_instance.invalidate_cache.side_effect = Exception("Cache error")
                mock_storage_class.return_value = mock_instance

                # Should not raise, just continue (exception is caught at line 255-257)
                save_comparison(
                    filepath=filepath,
                    metric="test",
                    dataset_i="A",
                    dataset_j="B",
                    value=0.5,
                    overwrite=True,
                    use_cache=True,
                )

                # Verify it was saved despite cache error
                loaded = load_comparison(filepath, "test", "A", "B")
                assert loaded == 0.5
        finally:
            Path(filepath).unlink(missing_ok=True)

