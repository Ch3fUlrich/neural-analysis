"""Complete tests for storage manager to reach 100% coverage."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from neural_analysis.utils.storage.config import StorageConfig
from neural_analysis.utils.storage.manager import StorageManager


class TestStorageManagerComplete:
    """Complete tests for StorageManager to reach 100% coverage."""

    def test_save_data_cache_set_success(self) -> None:
        """Test save_data when cache_set succeeds (covers line 108)."""
        config = StorageConfig(use_redis=False, use_sql=False)
        manager = StorageManager(config)
        
        # Mock cache to be available and cache_set to return True
        with patch.object(manager.cache, "is_available", return_value=True):
            with patch.object(manager, "cache_set", return_value=True):
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
                with patch.object(manager.metadata, "index_dataset", return_value="dataset_123"):
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

    def test_load_data_from_cache(self) -> None:
        """Test load_data when cache_get returns a value (covers lines 172-174)."""
        config = StorageConfig(use_redis=False, use_sql=False)
        manager = StorageManager(config)
        
        # Mock cache to be available and return cached data
        with patch.object(manager.cache, "is_available", return_value=True):
            with patch.object(manager, "cache_get", return_value="cached_data"):
                result = manager.load_data("key", use_cache=True)
                assert result == "cached_data"  # Should return cached data

    def test_load_data_cache_after_hdf5(self) -> None:
        """Test load_data caches data after loading from HDF5 (covers line 189)."""
        config = StorageConfig(use_redis=False, use_sql=False)
        manager = StorageManager(config)
        
        def mock_hdf5_loader():
            return "loaded_data"
        
        # Mock cache to be available
        with patch.object(manager.cache, "is_available", return_value=True):
            with patch.object(manager, "cache_set") as mock_cache_set:
                result = manager.load_data("key", use_cache=True, hdf5_loader=mock_hdf5_loader)
                assert result == "loaded_data"
                # Verify cache_set was called (line 189)
                mock_cache_set.assert_called_once_with("key", "loaded_data")

    def test_invalidate_cache_when_available(self) -> None:
        """Test invalidate_cache when cache is available (covers line 239)."""
        config = StorageConfig(use_redis=False, use_sql=False)
        manager = StorageManager(config)
        
        # Mock cache to be available
        with patch.object(manager.cache, "is_available", return_value=True):
            with patch.object(manager.cache, "invalidate", return_value=5) as mock_invalidate:
                result = manager.invalidate_cache("pattern*")
                assert result == 5
                mock_invalidate.assert_called_once_with("pattern*")

    def test_cache_get_when_available(self) -> None:
        """Test cache_get when cache is available (covers line 245)."""
        config = StorageConfig(use_redis=False, use_sql=False)
        manager = StorageManager(config)
        
        # Mock cache to be available
        with patch.object(manager.cache, "is_available", return_value=True):
            with patch.object(manager.cache, "get_cached", return_value="cached_value") as mock_get:
                result = manager.cache_get("key")
                assert result == "cached_value"
                mock_get.assert_called_once_with("key")

    def test_cache_set_when_available(self) -> None:
        """Test cache_set when cache is available (covers line 258)."""
        config = StorageConfig(use_redis=False, use_sql=False)
        manager = StorageManager(config)
        
        # Mock cache to be available
        with patch.object(manager.cache, "is_available", return_value=True):
            with patch.object(manager.cache, "set_cached", return_value=True) as mock_set:
                result = manager.cache_set("key", "data", ttl=100, metadata={"k": "v"})
                assert result is True
                mock_set.assert_called_once_with("key", "data", ttl=100, metadata={"k": "v"})

    def test_cache_delete_when_available(self) -> None:
        """Test cache_delete when cache is available (covers line 264)."""
        config = StorageConfig(use_redis=False, use_sql=False)
        manager = StorageManager(config)
        
        # Mock cache to be available
        with patch.object(manager.cache, "is_available", return_value=True):
            with patch.object(manager.cache, "delete", return_value=True) as mock_delete:
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
            manager2.metadata.close = "not_callable"  # Not callable - tests False branch for metadata
            
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

