"""Complete tests for redis_cache.py to reach 100% coverage."""

from __future__ import annotations

import importlib
import sys
from unittest.mock import MagicMock, patch

import pytest

from neural_analysis.utils.storage.config import StorageConfig
from neural_analysis.utils.storage.redis_cache import RedisCache, is_redis_available


class TestRedisCacheComplete:
    """Complete tests for RedisCache to reach 100% coverage."""

    def test_redis_import_fallback(self, monkeypatch):
        """Test ImportError fallback for REDIS_AVAILABLE (covers lines 18-19)."""
        # Save original state
        original_redis_cache = sys.modules.get("neural_analysis.utils.storage.redis_cache")
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
                sys.modules["neural_analysis.utils.storage.redis_cache"] = original_redis_cache
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

