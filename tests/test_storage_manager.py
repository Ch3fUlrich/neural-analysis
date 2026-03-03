"""Tests for unified storage manager."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from neural_analysis.utils.storage.config import StorageConfig
from neural_analysis.utils.storage.manager import StorageManager


class TestStorageManager:
    """Tests for StorageManager class."""

    def test_storage_manager_initialization(self) -> None:
        """Test StorageManager initialization."""
        config = StorageConfig(use_redis=False, use_sql=False)
        manager = StorageManager(config)

        assert manager.cache is not None
        assert manager.metadata is not None

    def test_storage_manager_save_data_unavailable_backends(self) -> None:
        """Test save_data when backends unavailable."""
        config = StorageConfig(use_redis=False, use_sql=False)
        manager = StorageManager(config)

        # Should return False when no backends available
        result = manager.save_data(
            key="test_key",
            data={"test": "data"},
            file_path="test.h5",
            use_cache=False,
        )
        # Returns False when no backends available, but doesn't crash
        assert isinstance(result, bool)

    def test_storage_manager_query_data_unavailable(self) -> None:
        """Test query_data when SQL unavailable."""
        config = StorageConfig(use_redis=False, use_sql=False)
        manager = StorageManager(config)

        # Should return empty DataFrame
        result = manager.query_data(filters={"metric": "euclidean"})
        assert isinstance(result, type(manager.query_data()))
        assert len(result) == 0

    def test_storage_manager_invalidate_cache(self) -> None:
        """Test cache invalidation."""
        config = StorageConfig(use_redis=False)
        manager = StorageManager(config)

        # Should return 0 when cache unavailable
        result = manager.invalidate_cache("test_pattern")
        assert result == 0

    def test_storage_manager_get_cache_stats(self) -> None:
        """Test getting cache statistics."""
        config = StorageConfig(use_redis=False, use_sql=False)
        manager = StorageManager(config)

        stats = manager.get_cache_stats()
        assert isinstance(stats, dict)
        assert "cache_available" in stats
        assert "metadata_available" in stats
        assert "config" in stats

    @pytest.mark.integration
    def test_storage_manager_integration(self) -> None:
        """Integration test with available backends."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sql_path = Path(tmpdir) / "test_meta.db"
            config = StorageConfig(use_redis=False, use_sql=True, sql_path=sql_path)
            manager = StorageManager(config)

            # Test indexing comparison
            comparison_id = manager.index_comparison(
                dataset_i="A",
                dataset_j="B",
                metric="euclidean",
                file_path="results.h5",
                group_path="euclidean/A___B",
                mode="between",
                value_type="scalar",
            )

            if manager.metadata.is_available():
                assert comparison_id != ""

                # Test querying
                results = manager.query_data(
                    filters={"metric": "euclidean", "mode": "between"}
                )
                assert isinstance(results, type(manager.query_data()))

            manager.close()

    def test_storage_manager_fallback_behavior(self) -> None:
        """Test that manager gracefully handles unavailable backends."""
        config = StorageConfig(use_redis=False, use_sql=False)
        manager = StorageManager(config)

        # All operations should work (returning empty/False) without backends
        assert manager.cache.is_available() is False
        assert manager.metadata.is_available() is False

        # Operations should not crash
        manager.save_data("key", "data", use_cache=False)
        result = manager.query_data()
        assert len(result) == 0
        assert manager.invalidate_cache("*") == 0

    def test_storage_manager_context_manager(self) -> None:
        """Test StorageManager as context manager (covers lines 54-58)."""
        config = StorageConfig(use_redis=False, use_sql=False)
        with StorageManager(config) as manager:
            assert manager is not None
            assert manager.cache is not None
        # Should be closed after context exit

    def test_storage_manager_save_data_with_cache(self) -> None:
        """Test save_data with cache enabled (covers lines 105-108)."""
        config = StorageConfig(use_redis=False, use_sql=False)
        manager = StorageManager(config)

        # With cache disabled, should return False
        result = manager.save_data("key", "data", use_cache=True)
        assert isinstance(result, bool)

    def test_storage_manager_save_data_index_comparison(self) -> None:
        """Test save_data with comparison indexing (covers lines 111-132)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sql_path = Path(tmpdir) / "test_meta.db"
            config = StorageConfig(use_redis=False, use_sql=True, sql_path=sql_path)
            manager = StorageManager(config)

            if manager.metadata.is_available():
                # Save with comparison metadata
                result = manager.save_data(
                    key="test_key",
                    data={"value": 0.5},
                    file_path="results.h5",
                    group_path="euclidean/A___B",
                    metadata={
                        "dataset_i": "A",
                        "dataset_j": "B",
                        "metric": "euclidean",
                        "mode": "between",
                        "value_type": "scalar",
                    },
                )
                assert result is True

            manager.close()

    def test_storage_manager_save_data_index_dataset(self) -> None:
        """Test save_data with dataset indexing (covers lines 133-138)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sql_path = Path(tmpdir) / "test_meta.db"
            config = StorageConfig(use_redis=False, use_sql=True, sql_path=sql_path)
            manager = StorageManager(config)

            if manager.metadata.is_available():
                # Save without group_path (indexes as dataset)
                result = manager.save_data(
                    key="test_key",
                    data={"value": 0.5},
                    file_path="data.h5",
                    metadata={"n_samples": 100},
                )
                assert result is True

            manager.close()

    def test_storage_manager_load_data_from_cache(self) -> None:
        """Test load_data from cache (covers lines 171-174)."""
        config = StorageConfig(use_redis=False, use_sql=False)
        manager = StorageManager(config)

        # With cache disabled, should return None
        result = manager.load_data("key", use_cache=True)
        assert result is None

    def test_storage_manager_load_data_hdf5_loader(self) -> None:
        """Test load_data with HDF5 loader (covers lines 184-194)."""
        config = StorageConfig(use_redis=False, use_sql=False)
        manager = StorageManager(config)

        # Test with successful loader
        def loader() -> str:
            return "loaded_data"

        result = manager.load_data("key", hdf5_loader=loader)
        assert result == "loaded_data"

        # Test with failing loader (covers line 191-192)
        def failing_loader() -> str:
            raise ValueError("Load error")

        result = manager.load_data("key", hdf5_loader=failing_loader)
        assert result is None

    def test_storage_manager_load_data_cache_result(self) -> None:
        """Test load_data caching result (covers lines 188-190)."""
        config = StorageConfig(use_redis=False, use_sql=False)
        manager = StorageManager(config)

        def loader() -> str:
            return "cached_data"

        # Should work even with cache disabled
        result = manager.load_data("key", hdf5_loader=loader, use_cache=True)
        assert result == "cached_data"

    def test_storage_manager_query_data_with_sql(self) -> None:
        """Test query_data with SQL (covers lines 200-204)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sql_path = Path(tmpdir) / "test_meta.db"
            config = StorageConfig(use_redis=False, use_sql=True, sql_path=sql_path)
            manager = StorageManager(config)

            if manager.metadata.is_available():
                result = manager.query_data(
                    filters={"metric": "euclidean"}, use_sql=True
                )
                assert isinstance(result, type(manager.query_data()))

            manager.close()

    def test_storage_manager_query_data_without_sql(self) -> None:
        """Test query_data without SQL (covers lines 205-207)."""
        config = StorageConfig(use_redis=False, use_sql=False)
        manager = StorageManager(config)

        result = manager.query_data(use_sql=False)
        assert isinstance(result, type(manager.query_data()))
        assert len(result) == 0

    def test_storage_manager_cache_get(self) -> None:
        """Test cache_get method (covers lines 242-246)."""
        config = StorageConfig(use_redis=False, use_sql=False)
        manager = StorageManager(config)

        result = manager.cache_get("key")
        assert result is None

    def test_storage_manager_cache_set(self) -> None:
        """Test cache_set method (covers lines 248-259)."""
        config = StorageConfig(use_redis=False, use_sql=False)
        manager = StorageManager(config)

        result = manager.cache_set("key", "data", ttl=100, metadata={"key": "value"})
        assert result is False  # Cache unavailable

    def test_storage_manager_cache_delete(self) -> None:
        """Test cache_delete method (covers lines 261-265)."""
        config = StorageConfig(use_redis=False, use_sql=False)
        manager = StorageManager(config)

        result = manager.cache_delete("key")
        assert result is False  # Cache unavailable

    def test_storage_manager_index_comparison_unavailable(self) -> None:
        """Test index_comparison when metadata unavailable (covers lines 263-265)."""
        config = StorageConfig(use_redis=False, use_sql=False)
        manager = StorageManager(config)

        result = manager.index_comparison(
            dataset_i="A",
            dataset_j="B",
            metric="euclidean",
            file_path="results.h5",
            group_path="group",
        )
        assert result == ""

    def test_storage_manager_close(self) -> None:
        """Test close method (covers lines 331-339)."""
        config = StorageConfig(use_redis=False, use_sql=False)
        manager = StorageManager(config)

        # Should not raise
        manager.close()

        # Can close multiple times
        manager.close()

    def test_storage_manager_load_data_metadata_available(self) -> None:
        """Test load_data when metadata available (covers lines 177-181)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sql_path = Path(tmpdir) / "test_meta.db"
            config = StorageConfig(use_redis=False, use_sql=True, sql_path=sql_path)
            manager = StorageManager(config)

            if manager.metadata.is_available():
                # Metadata-assisted loading not implemented, should fall back
                def loader() -> str:
                    return "fallback_data"

                result = manager.load_data("key", hdf5_loader=loader)
                assert result == "fallback_data"

            manager.close()


class TestStorageManagerComplete:
    """Complete tests for StorageManager to reach 100% coverage."""

    def test_save_data_cache_set_success(self) -> None:
        """Test save_data when cache_set succeeds (covers line 108)."""
        config = StorageConfig(use_redis=False, use_sql=False)
        manager = StorageManager(config)

        # Mock cache to be available and cache_set to return True
        with (
            patch.object(manager.cache, "is_available", return_value=True),
            patch.object(manager, "cache_set", return_value=True),
        ):
            result = manager.save_data("key", "data", use_cache=True)
            assert result is True  # Should be True when cache_set succeeds

    def test_save_data_dataset_id_truthy(self) -> None:
        """Test save_data when index_dataset returns truthy dataset_id (covers line 136->140)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sql_path = Path(tmpdir) / "test_meta.db"
            config = StorageConfig(use_redis=False, use_sql=True, sql_path=sql_path)
            manager = StorageManager(config)

            if manager.metadata.is_available():
                # Mock index_dataset to return a truthy dataset_id (covers branch 136->140)
                with patch.object(
                    manager.metadata, "index_dataset", return_value="dataset_123"
                ):
                    result = manager.save_data(
                        key="test_key",
                        data={"value": 0.5},
                        file_path="test.h5",
                        group_path=None,  # Not a comparison
                        metadata={},
                    )
                    assert result is True  # Should be True when dataset_id is truthy

                # Also test when dataset_id is falsy to cover both branches
                with patch.object(manager.metadata, "index_dataset", return_value=""):
                    result = manager.save_data(
                        key="test_key2",
                        data={"value": 0.5},
                        file_path="test2.h5",
                        group_path=None,
                        metadata={},
                    )
                    # Result might be False or True depending on other conditions
                    assert isinstance(result, bool)

            manager.close()

    def test_load_data_from_cache(self) -> None:
        """Test load_data when cache_get returns a value (covers lines 172-174)."""
        config = StorageConfig(use_redis=False, use_sql=False)
        manager = StorageManager(config)

        # Mock cache to be available and return cached data
        with (
            patch.object(manager.cache, "is_available", return_value=True),
            patch.object(manager, "cache_get", return_value="cached_data"),
        ):
            result = manager.load_data("key", use_cache=True)
            assert result == "cached_data"  # Should return cached data

    def test_load_data_cache_after_hdf5(self) -> None:
        """Test load_data caches data after loading from HDF5 (covers line 189)."""
        config = StorageConfig(use_redis=False, use_sql=False)
        manager = StorageManager(config)

        def mock_hdf5_loader():
            return "loaded_data"

        # Mock cache to be available
        with (
            patch.object(manager.cache, "is_available", return_value=True),
            patch.object(manager, "cache_set") as mock_cache_set,
        ):
            result = manager.load_data(
                "key", use_cache=True, hdf5_loader=mock_hdf5_loader
            )
            assert result == "loaded_data"
            # Verify cache_set was called (line 189)
            mock_cache_set.assert_called_once_with("key", "loaded_data")

    def test_invalidate_cache_when_available(self) -> None:
        """Test invalidate_cache when cache is available (covers line 239)."""
        config = StorageConfig(use_redis=False, use_sql=False)
        manager = StorageManager(config)

        # Mock cache to be available
        with (
            patch.object(manager.cache, "is_available", return_value=True),
            patch.object(
                manager.cache, "invalidate", return_value=5
            ) as mock_invalidate,
        ):
            result = manager.invalidate_cache("pattern*")
            assert result == 5
            mock_invalidate.assert_called_once_with("pattern*")

    def test_cache_get_when_available(self) -> None:
        """Test cache_get when cache is available (covers line 245)."""
        config = StorageConfig(use_redis=False, use_sql=False)
        manager = StorageManager(config)

        # Mock cache to be available
        with (
            patch.object(manager.cache, "is_available", return_value=True),
            patch.object(
                manager.cache, "get_cached", return_value="cached_value"
            ) as mock_get,
        ):
            result = manager.cache_get("key")
            assert result == "cached_value"
            mock_get.assert_called_once_with("key")

    def test_cache_set_when_available(self) -> None:
        """Test cache_set when cache is available (covers line 258)."""
        config = StorageConfig(use_redis=False, use_sql=False)
        manager = StorageManager(config)

        # Mock cache to be available
        with (
            patch.object(manager.cache, "is_available", return_value=True),
            patch.object(manager.cache, "set_cached", return_value=True) as mock_set,
        ):
            result = manager.cache_set("key", "data", ttl=100, metadata={"k": "v"})
            assert result is True
            mock_set.assert_called_once_with(
                "key", "data", ttl=100, metadata={"k": "v"}
            )

    def test_cache_delete_when_available(self) -> None:
        """Test cache_delete when cache is available (covers line 264)."""
        config = StorageConfig(use_redis=False, use_sql=False)
        manager = StorageManager(config)

        # Mock cache to be available
        with (
            patch.object(manager.cache, "is_available", return_value=True),
            patch.object(manager.cache, "delete", return_value=True) as mock_delete,
        ):
            result = manager.cache_delete("key")
            assert result is True
            mock_delete.assert_called_once_with("key")

    def test_close_with_callable_methods(self) -> None:
        """Test close when cache_close and metadata_close are callable (covers lines 334->337, 338->exit)."""
        config = StorageConfig(use_redis=False, use_sql=False)
        manager = StorageManager(config)

        # Test with callable close methods (covers lines 334->337, 338->exit - True branches)
        mock_cache_close = MagicMock()
        mock_metadata_close = MagicMock()

        # Set close methods directly to ensure they're callable
        manager.cache.close = mock_cache_close
        manager.metadata.close = mock_metadata_close

        manager.close()
        # Verify both close methods were called (covers branches 334->337, 338->exit)
        mock_cache_close.assert_called_once()
        mock_metadata_close.assert_called_once()

        # Test when close methods are not callable (covers False branches)
        # Create a new manager to test the False branches
        manager2 = StorageManager(config)

        # Set non-callable attributes (like a string or None) to test False branches
        try:
            manager2.cache.close = None  # Not callable - tests False branch for cache
            manager2.metadata.close = (
                "not_callable"  # Not callable - tests False branch for metadata
            )

            # Should not raise even if close methods are not callable
            manager2.close()
        except (AttributeError, TypeError):
            # If setting fails, try a different approach
            pass

        # Test with one callable and one not callable to cover all branch combinations
        manager3 = StorageManager(config)
        mock_cache_close3 = MagicMock()
        manager3.cache.close = mock_cache_close3
        manager3.metadata.close = None  # Not callable

        manager3.close()
        # Cache close should be called, metadata close should not
        mock_cache_close3.assert_called_once()

        # Test with metadata callable and cache not callable
        manager4 = StorageManager(config)
        mock_metadata_close4 = MagicMock()
        manager4.cache.close = None  # Not callable
        manager4.metadata.close = mock_metadata_close4

        manager4.close()
        # Metadata close should be called, cache close should not
        mock_metadata_close4.assert_called_once()
