"""Redis cache backend for fast in-memory data access.

Provides caching layer with automatic fallback if Redis is unavailable.
"""

from __future__ import annotations

import pickle
from typing import Any, cast

from neural_analysis.utils.logging import get_logger
from neural_analysis.utils.storage.config import get_config

try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = get_logger(__name__)

__all__ = ["RedisCache", "is_redis_available"]


def is_redis_available() -> bool:
    """Check if Redis is available (package installed and server reachable).

    Returns
    -------
    bool
        True if Redis package is installed and server is reachable
    """
    if not REDIS_AVAILABLE:
        return False

    try:
        config = get_config()
        client = redis.Redis(
            host=config.redis_host,
            port=config.redis_port,
            db=config.redis_db,
            password=config.redis_password if config.redis_password else None,
            socket_connect_timeout=1,  # Fast timeout for availability check
            decode_responses=False,  # We handle binary data
        )
        client.ping()
        return True
    except Exception:
        return False


class RedisCache:
    """Redis cache backend with graceful degradation.

    Automatically falls back to no-op if Redis is unavailable.
    """

    def __init__(self, config: Any | None = None) -> None:
        """Initialize Redis cache backend.

        Parameters
        ----------
        config : StorageConfig, optional
            Configuration instance. If None, uses global config.
        """
        from neural_analysis.utils.storage.config import get_config

        self.config = config or get_config()
        self._client: redis.Redis | None = None
        self._available = False

        if REDIS_AVAILABLE and self.config.use_redis:
            try:
                self._client = redis.Redis(
                    host=self.config.redis_host,
                    port=self.config.redis_port,
                    db=self.config.redis_db,
                    password=self.config.redis_password
                    if self.config.redis_password
                    else None,
                    decode_responses=False,  # We handle binary data
                    socket_connect_timeout=5,
                )
                # Test connection
                self._client.ping()
                self._available = True
                logger.info(
                    f"Redis cache connected: {self.config.redis_host}:{self.config.redis_port}"
                )
            except Exception as e:
                logger.warning(
                    f"Redis cache unavailable: {e}. Continuing without cache."
                )
                self._client = None
                self._available = False
        else:
            if not REDIS_AVAILABLE:
                logger.debug("Redis package not installed. Cache disabled.")
            elif not self.config.use_redis:  # pragma: no branch
                logger.debug("Redis disabled in configuration. Cache disabled.")
            self._available = False

    def is_available(self) -> bool:
        """Check if Redis cache is available.

        Returns
        -------
        bool
            True if Redis is connected and ready
        """
        return self._available and self._client is not None

    def _client_guard(self) -> redis.Redis | None:
        """Return active redis client if available."""
        if not self.is_available():
            return None
        return self._client

    def get_cached(self, key: str) -> Any | None:
        """Get cached data by key.

        Parameters
        ----------
        key : str
            Cache key

        Returns
        -------
        Any or None
            Cached data if found, None otherwise
        """
        client = self._client_guard()
        if client is None:
            return None

        try:
            cache_key = self._namespaced_key(key)
            raw_data: bytes | None = cast("bytes | None", client.get(cache_key))
            data = raw_data
            if data is None:
                return None

            # Deserialize
            value = pickle.loads(data)
            logger.debug(f"Cache hit: {key}")
            return value
        except Exception as e:
            logger.warning(f"Error reading from cache for key {key}: {e}")
            return None

    def set_cached(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Cache data with optional TTL.

        Parameters
        ----------
        key : str
            Cache key
        value : Any
            Data to cache (will be pickled)
        ttl : int, optional
            Time-to-live in seconds. If None, uses config default.
        metadata : dict, optional
            Metadata (not stored, for compatibility with interface)

        Returns
        -------
        bool
            True if caching was successful, False otherwise
        """
        client = self._client_guard()
        if client is None:
            return False

        try:
            # Check size limit
            pickled = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
            size_mb = len(pickled) / (1024 * 1024)
            if size_mb > self.config.cache_max_size_mb:
                logger.warning(
                    f"Data too large for cache ({size_mb:.1f} MB > {self.config.cache_max_size_mb} MB): {key}"
                )
                return False

            cache_key = self._namespaced_key(key)
            ttl_seconds = ttl if ttl is not None else self.config.cache_ttl

            client.setex(cache_key, ttl_seconds, pickled)
            logger.debug(f"Cached: {key} (TTL: {ttl_seconds}s, size: {size_mb:.2f} MB)")
            return True
        except Exception as e:
            logger.warning(f"Error writing to cache for key {key}: {e}")
            return False

    def exists(self, key: str) -> bool:
        """Check if key exists in cache.

        Parameters
        ----------
        key : str
            Cache key to check

        Returns
        -------
        bool
            True if key exists, False otherwise
        """
        client = self._client_guard()
        if client is None:
            return False

        try:
            cache_key = self._namespaced_key(key)
            return bool(client.exists(cache_key))
        except Exception:
            return False

    def delete(self, key: str) -> bool:
        """Delete key from cache.

        Parameters
        ----------
        key : str
            Cache key to delete

        Returns
        -------
        bool
            True if deletion was successful, False otherwise
        """
        client = self._client_guard()
        if client is None:
            return False

        try:
            cache_key = self._namespaced_key(key)
            deleted = client.delete(cache_key)
            if deleted:
                logger.debug(f"Deleted from cache: {key}")
            return bool(deleted)
        except Exception as e:
            logger.warning(f"Error deleting from cache for key {key}: {e}")
            return False

    def invalidate(self, pattern: str) -> int:
        """Invalidate cache entries matching a pattern.

        Parameters
        ----------
        pattern : str
            Pattern to match keys (supports Redis wildcards: *, ?, [abc])

        Returns
        -------
        int
            Number of keys invalidated
        """
        client = self._client_guard()
        if client is None:
            return 0

        try:
            cache_pattern = f"{self._namespaced_prefix()}{pattern}"
            keys = list(client.scan_iter(match=cache_pattern))
            if not keys:
                return 0

            deleted_raw = client.delete(*keys)
            deleted = int(deleted_raw)
            logger.info(
                f"Invalidated {deleted} cache entries matching pattern: {pattern}"
            )
            return deleted
        except Exception as e:
            logger.warning(f"Error invalidating cache pattern {pattern}: {e}")
            return 0

    def save(
        self,
        key: str,
        data: Any,
        metadata: dict[str, Any] | None = None,
        ttl: int | None = None,
    ) -> bool:
        """Save data to cache (alias for set_cached for interface compatibility).

        Parameters
        ----------
        key : str
            Cache key
        data : Any
            Data to cache
        metadata : dict, optional
            Metadata (not stored)
        ttl : int, optional
            Time-to-live in seconds

        Returns
        -------
        bool
            True if save was successful
        """
        return self.set_cached(key, data, ttl=ttl, metadata=metadata)

    def load(self, key: str) -> Any | None:
        """Load data from cache (alias for get_cached for interface compatibility).

        Parameters
        ----------
        key : str
            Cache key

        Returns
        -------
        Any or None
            Cached data if found
        """
        return self.get_cached(key)

    def close(self) -> None:
        """Close redis connection and reset availability."""
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                logger.debug("Failed to close Redis client cleanly", exc_info=True)
        self._client = None
        self._available = False

    def _namespaced_prefix(self) -> str:
        return f"cache:{self.config.cache_namespace}:"

    def _namespaced_key(self, key: str) -> str:
        return f"{self._namespaced_prefix()}{key}"
