"""Tests for unified storage manager."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
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



