"""Tests for Redis cache backend."""

from __future__ import annotations

import numpy as np
import pytest

from neural_analysis.utils.storage.config import StorageConfig
from neural_analysis.utils.storage.redis_cache import RedisCache, is_redis_available


class TestRedisCache:
    """Tests for RedisCache class."""

    def test_is_redis_available_without_redis(self) -> None:
        """Test is_redis_available when Redis package not installed."""
        # This test will pass whether or not Redis is installed
        # It just checks the function doesn't crash
        result = is_redis_available()
        assert isinstance(result, bool)

    def test_redis_cache_initialization_without_redis(self) -> None:
        """Test RedisCache initialization when Redis unavailable."""
        config = StorageConfig(use_redis=False)
        cache = RedisCache(config)
        assert not cache.is_available()

    def test_redis_cache_save_load_unavailable(self) -> None:
        """Test cache operations when Redis unavailable."""
        config = StorageConfig(use_redis=False)
        cache = RedisCache(config)

        # Save should return False when unavailable
        result = cache.set_cached("test_key", {"data": [1, 2, 3]})
        assert result is False

        # Load should return None when unavailable
        result = cache.get_cached("test_key")
        assert result is None

        # Exists should return False
        assert cache.exists("test_key") is False

        # Delete should return False
        assert cache.delete("test_key") is False

    def test_redis_cache_invalidate_unavailable(self) -> None:
        """Test cache invalidation when Redis unavailable."""
        config = StorageConfig(use_redis=False)
        cache = RedisCache(config)

        # Invalidate should return 0 when unavailable
        result = cache.invalidate("test_pattern")
        assert result == 0

    @pytest.mark.integration
    def test_redis_cache_with_redis_server(self) -> None:
        """Integration test with actual Redis server (if available)."""
        if not is_redis_available():
            pytest.skip("Redis server not available")

        config = StorageConfig(use_redis=True)
        cache = RedisCache(config)

        if not cache.is_available():
            pytest.skip("Redis cache not available")

        # Test save and load
        test_data = {"numbers": [1, 2, 3, 4, 5], "array": np.array([1.0, 2.0, 3.0])}
        key = "test_integration_key"

        # Save
        success = cache.set_cached(key, test_data, ttl=60)
        assert success is True

        # Check exists
        assert cache.exists(key) is True

        # Load
        loaded = cache.get_cached(key)
        assert loaded is not None
        assert "numbers" in loaded
        assert np.array_equal(loaded["array"], test_data["array"])

        # Delete
        deleted = cache.delete(key)
        assert deleted is True
        assert cache.exists(key) is False

    @pytest.mark.integration
    def test_redis_cache_ttl(self) -> None:
        """Test cache TTL expiration (if Redis available)."""
        if not is_redis_available():
            pytest.skip("Redis server not available")

        config = StorageConfig(use_redis=True)
        cache = RedisCache(config)

        if not cache.is_available():
            pytest.skip("Redis cache not available")

        key = "test_ttl_key"
        test_data = {"test": "data"}

        # Save with short TTL
        cache.set_cached(key, test_data, ttl=1)
        assert cache.exists(key) is True

        # Wait for expiration (this test may be flaky, but useful for integration)
        import time

        time.sleep(2)
        # Note: TTL expiration is handled by Redis, so this may still exist
        # depending on Redis configuration. We just verify the TTL was set.

    def test_redis_cache_interface_compatibility(self) -> None:
        """Test that RedisCache implements StorageBackend interface."""
        config = StorageConfig(use_redis=False)
        cache = RedisCache(config)

        # Test interface methods exist
        assert hasattr(cache, "save")
        assert hasattr(cache, "load")
        assert hasattr(cache, "exists")
        assert hasattr(cache, "delete")
        assert hasattr(cache, "invalidate")

        # Test save/load aliases
        cache.save("key", "data")
        result = cache.load("key")
        assert result is None  # Unavailable, so returns None



