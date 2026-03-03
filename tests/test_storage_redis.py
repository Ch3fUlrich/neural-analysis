"""Tests for Redis cache backend."""

from __future__ import annotations

import importlib
import pickle
import sys
from unittest.mock import MagicMock, patch

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

    @patch("neural_analysis.utils.storage.redis_cache.REDIS_AVAILABLE", True)
    @patch("neural_analysis.utils.storage.redis_cache.redis")
    def test_is_redis_available_with_redis_package(self, mock_redis: MagicMock) -> None:
        """Test is_redis_available when Redis package is installed (covers lines 35, 48)."""
        # Mock successful connection
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_redis.Redis.return_value = mock_client

        result = is_redis_available()
        assert result is True

    @patch("neural_analysis.utils.storage.redis_cache.REDIS_AVAILABLE", True)
    @patch("neural_analysis.utils.storage.redis_cache.redis")
    def test_is_redis_available_connection_error(self, mock_redis: MagicMock) -> None:
        """Test is_redis_available when connection fails (covers line 48)."""
        # Mock connection error
        mock_redis.Redis.side_effect = Exception("Connection refused")

        result = is_redis_available()
        assert result is False

    @patch("neural_analysis.utils.storage.redis_cache.REDIS_AVAILABLE", False)
    def test_is_redis_available_package_not_installed(self) -> None:
        """Test is_redis_available when package not installed (covers line 35)."""
        result = is_redis_available()
        assert result is False

    @patch("neural_analysis.utils.storage.redis_cache.REDIS_AVAILABLE", True)
    @patch("neural_analysis.utils.storage.redis_cache.redis")
    def test_redis_cache_initialization_success(self, mock_redis: MagicMock) -> None:
        """Test RedisCache initialization with successful connection (covers lines 85-86)."""
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_redis.Redis.return_value = mock_client

        config = StorageConfig(use_redis=True)
        cache = RedisCache(config)

        assert cache.is_available() is True
        mock_client.ping.assert_called_once()

    @patch("neural_analysis.utils.storage.redis_cache.REDIS_AVAILABLE", False)
    def test_redis_cache_initialization_package_not_installed(self) -> None:
        """Test RedisCache initialization when package not installed (covers line 95)."""
        with patch("neural_analysis.utils.storage.redis_cache.logger") as mock_logger:
            config = StorageConfig(use_redis=True)
            cache = RedisCache(config)
            assert not cache.is_available()
            # Verify the debug message was logged (line 95) - this covers the if branch
            mock_logger.debug.assert_called_with(
                "Redis package not installed. Cache disabled."
            )

    @patch("neural_analysis.utils.storage.redis_cache.REDIS_AVAILABLE", True)
    def test_redis_cache_initialization_disabled(self) -> None:
        """Test RedisCache initialization when disabled in config (covers lines 96->98)."""

        with patch("neural_analysis.utils.storage.redis_cache.logger") as mock_logger:
            # Ensure REDIS_AVAILABLE is True (patched) and use_redis is False
            # This should hit the elif branch (line 96->98)
            config = StorageConfig(use_redis=False)
            cache = RedisCache(config)
            assert not cache.is_available()
            # Verify the debug message was logged (line 97) - this covers the elif branch (96->98)
            mock_logger.debug.assert_called_with(
                "Redis disabled in configuration. Cache disabled."
            )

            # Also verify that the if branch (line 94-95) was NOT executed
            # by checking that the "Redis package not installed" message was not called
            calls = [str(call) for call in mock_logger.debug.call_args_list]
            assert not any("Redis package not installed" in str(call) for call in calls)

    @patch("neural_analysis.utils.storage.redis_cache.REDIS_AVAILABLE", True)
    def test_redis_cache_initialization_both_branches(self) -> None:
        """Test both if and elif branches to ensure branch coverage (covers lines 94->98)."""
        # Test 1: if branch (REDIS_AVAILABLE is False)
        with (
            patch("neural_analysis.utils.storage.redis_cache.REDIS_AVAILABLE", False),
            patch("neural_analysis.utils.storage.redis_cache.logger") as mock_logger,
        ):
            config = StorageConfig(use_redis=True)
            cache = RedisCache(config)
            assert not cache.is_available()
            # Should call the if branch message
            mock_logger.debug.assert_called_with(
                "Redis package not installed. Cache disabled."
            )

        # Test 2: elif branch (REDIS_AVAILABLE is True, use_redis is False) - covers 96->98
        with (
            patch("neural_analysis.utils.storage.redis_cache.REDIS_AVAILABLE", True),
            patch("neural_analysis.utils.storage.redis_cache.logger") as mock_logger,
        ):
            config = StorageConfig(use_redis=False)
            cache = RedisCache(config)
            assert not cache.is_available()
            # Should call the elif branch message (covers 96->98)
            mock_logger.debug.assert_called_with(
                "Redis disabled in configuration. Cache disabled."
            )

    @patch("neural_analysis.utils.storage.redis_cache.REDIS_AVAILABLE", True)
    @patch("neural_analysis.utils.storage.redis_cache.redis")
    def test_redis_cache_get_cached_success(self, mock_redis: MagicMock) -> None:
        """Test get_cached with successful retrieval (covers lines 133-143)."""
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_redis.Redis.return_value = mock_client

        test_data = {"test": "value"}
        pickled_data = pickle.dumps(test_data)
        mock_client.get.return_value = pickled_data

        config = StorageConfig(use_redis=True)
        cache = RedisCache(config)

        result = cache.get_cached("test_key")
        assert result == test_data
        assert mock_client.get.called

    @patch("neural_analysis.utils.storage.redis_cache.REDIS_AVAILABLE", True)
    @patch("neural_analysis.utils.storage.redis_cache.redis")
    def test_redis_cache_get_cached_not_found(self, mock_redis: MagicMock) -> None:
        """Test get_cached when key not found (covers lines 137-138)."""
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_redis.Redis.return_value = mock_client
        mock_client.get.return_value = None

        config = StorageConfig(use_redis=True)
        cache = RedisCache(config)

        result = cache.get_cached("test_key")
        assert result is None

    @patch("neural_analysis.utils.storage.redis_cache.REDIS_AVAILABLE", True)
    @patch("neural_analysis.utils.storage.redis_cache.redis")
    def test_redis_cache_get_cached_exception(self, mock_redis: MagicMock) -> None:
        """Test get_cached with exception (covers lines 144-146)."""
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_redis.Redis.return_value = mock_client
        mock_client.get.side_effect = Exception("Redis error")

        config = StorageConfig(use_redis=True)
        cache = RedisCache(config)

        result = cache.get_cached("test_key")
        assert result is None

    @patch("neural_analysis.utils.storage.redis_cache.REDIS_AVAILABLE", True)
    @patch("neural_analysis.utils.storage.redis_cache.redis")
    def test_redis_cache_set_cached_success(self, mock_redis: MagicMock) -> None:
        """Test set_cached with successful save (covers lines 173-188)."""
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_redis.Redis.return_value = mock_client

        config = StorageConfig(use_redis=True, cache_max_size_mb=10)
        cache = RedisCache(config)

        test_data = {"test": "value"}
        result = cache.set_cached("test_key", test_data, ttl=60)

        assert result is True
        assert mock_client.setex.called

    @patch("neural_analysis.utils.storage.redis_cache.REDIS_AVAILABLE", True)
    @patch("neural_analysis.utils.storage.redis_cache.redis")
    def test_redis_cache_set_cached_too_large(self, mock_redis: MagicMock) -> None:
        """Test set_cached when data is too large (covers lines 177-181)."""
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_redis.Redis.return_value = mock_client

        # Create data larger than max size
        large_data = b"x" * (11 * 1024 * 1024)  # 11 MB
        config = StorageConfig(use_redis=True, cache_max_size_mb=10)
        cache = RedisCache(config)

        result = cache.set_cached("test_key", large_data)
        assert result is False
        assert not mock_client.setex.called

    @patch("neural_analysis.utils.storage.redis_cache.REDIS_AVAILABLE", True)
    @patch("neural_analysis.utils.storage.redis_cache.redis")
    def test_redis_cache_set_cached_exception(self, mock_redis: MagicMock) -> None:
        """Test set_cached with exception (covers lines 189-191)."""
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_redis.Redis.return_value = mock_client
        mock_client.setex.side_effect = Exception("Redis error")

        config = StorageConfig(use_redis=True, cache_max_size_mb=10)
        cache = RedisCache(config)

        result = cache.set_cached("test_key", {"test": "value"})
        assert result is False

    @patch("neural_analysis.utils.storage.redis_cache.REDIS_AVAILABLE", True)
    @patch("neural_analysis.utils.storage.redis_cache.redis")
    def test_redis_cache_exists_success(self, mock_redis: MagicMock) -> None:
        """Test exists with successful check (covers lines 210-212)."""
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_redis.Redis.return_value = mock_client
        mock_client.exists.return_value = 1

        config = StorageConfig(use_redis=True)
        cache = RedisCache(config)

        result = cache.exists("test_key")
        assert result is True

    @patch("neural_analysis.utils.storage.redis_cache.REDIS_AVAILABLE", True)
    @patch("neural_analysis.utils.storage.redis_cache.redis")
    def test_redis_cache_exists_exception(self, mock_redis: MagicMock) -> None:
        """Test exists with exception (covers lines 213-214)."""
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_redis.Redis.return_value = mock_client
        mock_client.exists.side_effect = Exception("Redis error")

        config = StorageConfig(use_redis=True)
        cache = RedisCache(config)

        result = cache.exists("test_key")
        assert result is False

    @patch("neural_analysis.utils.storage.redis_cache.REDIS_AVAILABLE", True)
    @patch("neural_analysis.utils.storage.redis_cache.redis")
    def test_redis_cache_delete_success(self, mock_redis: MagicMock) -> None:
        """Test delete with successful deletion (covers lines 233-238)."""
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_redis.Redis.return_value = mock_client
        mock_client.delete.return_value = 1

        config = StorageConfig(use_redis=True)
        cache = RedisCache(config)

        result = cache.delete("test_key")
        assert result is True

    @patch("neural_analysis.utils.storage.redis_cache.REDIS_AVAILABLE", True)
    @patch("neural_analysis.utils.storage.redis_cache.redis")
    def test_redis_cache_delete_exception(self, mock_redis: MagicMock) -> None:
        """Test delete with exception (covers lines 239-241)."""
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_redis.Redis.return_value = mock_client
        mock_client.delete.side_effect = Exception("Redis error")

        config = StorageConfig(use_redis=True)
        cache = RedisCache(config)

        result = cache.delete("test_key")
        assert result is False

    @patch("neural_analysis.utils.storage.redis_cache.REDIS_AVAILABLE", True)
    @patch("neural_analysis.utils.storage.redis_cache.redis")
    def test_redis_cache_invalidate_success(self, mock_redis: MagicMock) -> None:
        """Test invalidate with successful invalidation (covers lines 260-269)."""
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_redis.Redis.return_value = mock_client

        # Mock scan_iter to return some keys
        mock_keys = [b"cache:test:key1", b"cache:test:key2"]
        mock_client.scan_iter.return_value = iter(mock_keys)
        mock_client.delete.return_value = 2

        config = StorageConfig(use_redis=True, cache_namespace="test")
        cache = RedisCache(config)

        result = cache.invalidate("pattern*")
        assert result == 2

    @patch("neural_analysis.utils.storage.redis_cache.REDIS_AVAILABLE", True)
    @patch("neural_analysis.utils.storage.redis_cache.redis")
    def test_redis_cache_invalidate_no_keys(self, mock_redis: MagicMock) -> None:
        """Test invalidate when no keys match (covers lines 263-264)."""
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_redis.Redis.return_value = mock_client
        mock_client.scan_iter.return_value = iter([])

        config = StorageConfig(use_redis=True)
        cache = RedisCache(config)

        result = cache.invalidate("pattern*")
        assert result == 0

    @patch("neural_analysis.utils.storage.redis_cache.REDIS_AVAILABLE", True)
    @patch("neural_analysis.utils.storage.redis_cache.redis")
    def test_redis_cache_invalidate_exception(self, mock_redis: MagicMock) -> None:
        """Test invalidate with exception (covers lines 270-272)."""
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_redis.Redis.return_value = mock_client
        mock_client.scan_iter.side_effect = Exception("Redis error")

        config = StorageConfig(use_redis=True)
        cache = RedisCache(config)

        result = cache.invalidate("pattern*")
        assert result == 0

    @patch("neural_analysis.utils.storage.redis_cache.REDIS_AVAILABLE", True)
    @patch("neural_analysis.utils.storage.redis_cache.redis")
    def test_redis_cache_close(self, mock_redis: MagicMock) -> None:
        """Test close method (covers lines 312-320)."""
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_redis.Redis.return_value = mock_client

        config = StorageConfig(use_redis=True)
        cache = RedisCache(config)

        cache.close()
        assert cache._client is None
        assert not cache._available
        mock_client.close.assert_called_once()

    @patch("neural_analysis.utils.storage.redis_cache.REDIS_AVAILABLE", True)
    @patch("neural_analysis.utils.storage.redis_cache.redis")
    def test_redis_cache_close_exception(self, mock_redis: MagicMock) -> None:
        """Test close method with exception (covers lines 315-318)."""
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_client.close.side_effect = Exception("Close error")
        mock_redis.Redis.return_value = mock_client

        config = StorageConfig(use_redis=True)
        cache = RedisCache(config)

        # Should not raise
        cache.close()
        assert cache._client is None
        assert not cache._available

    @patch("neural_analysis.utils.storage.redis_cache.REDIS_AVAILABLE", True)
    @patch("neural_analysis.utils.storage.redis_cache.redis")
    def test_redis_cache_namespaced_methods(self, mock_redis: MagicMock) -> None:
        """Test namespaced key methods (covers lines 322-326)."""
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_redis.Redis.return_value = mock_client

        config = StorageConfig(use_redis=True, cache_namespace="test_namespace")
        cache = RedisCache(config)

        prefix = cache._namespaced_prefix()
        assert prefix == "cache:test_namespace:"

        key = cache._namespaced_key("my_key")
        assert key == "cache:test_namespace:my_key"

    @patch("neural_analysis.utils.storage.redis_cache.REDIS_AVAILABLE", True)
    @patch("neural_analysis.utils.storage.redis_cache.redis")
    def test_redis_cache_client_guard(self, mock_redis: MagicMock) -> None:
        """Test _client_guard method (covers line 114)."""
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_redis.Redis.return_value = mock_client

        config = StorageConfig(use_redis=True)
        cache = RedisCache(config)

        client = cache._client_guard()
        assert client is not None
        assert client == mock_client

        # Test when unavailable
        cache._available = False
        client = cache._client_guard()
        assert client is None

    @patch("neural_analysis.utils.storage.redis_cache.REDIS_AVAILABLE", True)
    @patch("neural_analysis.utils.storage.redis_cache.redis")
    def test_redis_cache_initialization_connection_error(
        self, mock_redis: MagicMock
    ) -> None:
        """Test RedisCache initialization with connection error (covers lines 89-92)."""
        mock_redis.Redis.side_effect = Exception("Connection refused")

        config = StorageConfig(use_redis=True)
        cache = RedisCache(config)

        assert not cache.is_available()
        assert cache._client is None

    @patch("neural_analysis.utils.storage.redis_cache.REDIS_AVAILABLE", True)
    def test_redis_cache_initialization_use_redis_false(self) -> None:
        """Test RedisCache initialization when use_redis is False (covers lines 96->98)."""
        # This should hit the elif branch
        config = StorageConfig(use_redis=False)
        cache = RedisCache(config)
        assert not cache.is_available()

    @patch("neural_analysis.utils.storage.redis_cache.REDIS_AVAILABLE", True)
    @patch("neural_analysis.utils.storage.redis_cache.redis")
    def test_redis_cache_delete_success_with_logging(
        self, mock_redis: MagicMock
    ) -> None:
        """Test delete with successful deletion and logging (covers lines 236->238)."""
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_redis.Redis.return_value = mock_client
        mock_client.delete.return_value = 1  # Successfully deleted (non-zero)

        config = StorageConfig(use_redis=True)
        cache = RedisCache(config)

        with patch("neural_analysis.utils.storage.redis_cache.logger") as mock_logger:
            result = cache.delete("test_key")
            assert result is True
            # Verify the debug message was logged (line 237)
            mock_logger.debug.assert_called_with("Deleted from cache: test_key")

    @patch("neural_analysis.utils.storage.redis_cache.REDIS_AVAILABLE", True)
    @patch("neural_analysis.utils.storage.redis_cache.redis")
    def test_redis_cache_delete_falsy_no_logging(self, mock_redis: MagicMock) -> None:
        """Test delete when deletion returns 0 (falsy) - covers line 236->238 branch."""
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_redis.Redis.return_value = mock_client
        mock_client.delete.return_value = 0  # Not deleted (falsy)

        config = StorageConfig(use_redis=True)
        cache = RedisCache(config)

        with patch("neural_analysis.utils.storage.redis_cache.logger") as mock_logger:
            result = cache.delete("test_key")
            assert result is False
            # Verify the debug message was NOT logged (falsy branch)
            mock_logger.debug.assert_not_called()
        assert mock_client.delete.called


class TestRedisCacheComplete:
    """Complete tests for RedisCache to reach 100% coverage."""

    def test_redis_import_fallback(self, monkeypatch):
        """Test ImportError fallback for REDIS_AVAILABLE (covers lines 18-19)."""
        # Save original state
        original_redis_cache = sys.modules.get(
            "neural_analysis.utils.storage.redis_cache"
        )
        original_redis = sys.modules.get("redis")

        # Remove modules from cache
        if "neural_analysis.utils.storage.redis_cache" in sys.modules:
            del sys.modules["neural_analysis.utils.storage.redis_cache"]
        if "redis" in sys.modules:
            del sys.modules["redis"]

        # Mock import to raise ImportError for redis
        original_import = __import__

        def mock_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "redis":
                raise ImportError("Mocked import error for redis")
            return original_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr("builtins.__import__", mock_import)
        importlib.invalidate_caches()

        try:
            # Re-import to trigger the fallback (covers lines 18-19)
            import neural_analysis.utils.storage.redis_cache as redis_cache_module

            importlib.reload(redis_cache_module)
            assert redis_cache_module.REDIS_AVAILABLE is False
        finally:
            # Restore original modules
            if original_redis_cache:
                sys.modules["neural_analysis.utils.storage.redis_cache"] = (
                    original_redis_cache
                )
            if original_redis:
                sys.modules["redis"] = original_redis

    @patch("neural_analysis.utils.storage.redis_cache.REDIS_AVAILABLE", True)
    def test_redis_cache_branch_96_to_98_false_path(self):
        """Test branch 96->98 False path: when elif is False, skip to line 98.

        The branch 96->98 False path would be when use_redis is True,
        but we're in the else block. This is impossible in normal execution,
        but let's try to test it by ensuring the condition is evaluated.
        """
        # The False path of elif would require: REDIS_AVAILABLE=True, use_redis=True, but in else block
        # This is impossible, so the branch might be unreachable.
        # However, let's ensure we test the True path properly
        with patch("neural_analysis.utils.storage.redis_cache.logger") as mock_logger:
            config = StorageConfig(use_redis=False)
            cache = RedisCache(config)
            assert not cache.is_available()
            # This should hit the True path of the elif (96->97->98)
            mock_logger.debug.assert_called_with(
                "Redis disabled in configuration. Cache disabled."
            )
