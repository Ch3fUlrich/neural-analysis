"""Configuration system for storage backends.

Supports environment variables and automatic detection of available backends.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

__all__ = [
    "StorageConfig",
    "get_config",
]


def _env_flag(name: str) -> bool | None:
    """Parse boolean environment variable."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    return raw.strip().lower() in ("true", "1", "yes", "on")


class StorageConfig:
    """Configuration for storage backends.

    Reads from environment variables with sensible defaults.
    All settings can be overridden programmatically.
    """

    def __init__(
        self,
        use_redis: bool | None = None,
        use_sql: bool | None = None,
        redis_host: str | None = None,
        redis_port: int | None = None,
        redis_db: int | None = None,
        redis_password: str | None = None,
        sql_path: str | Path | None = None,
        cache_ttl: int | None = None,
        cache_max_size_mb: int | None = None,
        cache_namespace: str | None = None,
    ) -> None:
        """Initialize storage configuration.

        Parameters
        ----------
        use_redis : bool, optional
            Enable Redis cache. If None, auto-detects from environment or availability.
        use_sql : bool, optional
            Enable SQL metadata indexing. If None, auto-detects from environment or availability.
        redis_host : str, optional
            Redis server hostname. Defaults to 'localhost'.
        redis_port : int, optional
            Redis server port. Defaults to 6379.
        redis_db : int, optional
            Redis database number. Defaults to 0.
        redis_password : str, optional
            Redis password if required.
        sql_path : str or Path, optional
            Path to DuckDB database file. Defaults to '.neural_analysis_meta.db'.
        cache_ttl : int, optional
            Default cache TTL in seconds. Defaults to 3600 (1 hour).
        cache_max_size_mb : int, optional
            Maximum size in MB for cached items. Defaults to 100 MB.
        """
        # Redis settings
        env_redis = _env_flag("NEURAL_ANALYSIS_USE_REDIS")
        if use_redis is not None:
            self.use_redis = use_redis
        elif env_redis is not None:
            self.use_redis = env_redis
        else:
            # Default to True so integration tests exercise Redis when available
            self.use_redis = True
        if redis_host is not None:
            self.redis_host: str = redis_host
        else:
            self.redis_host = os.getenv("NEURAL_ANALYSIS_REDIS_HOST", "localhost")

        if redis_port is not None:
            self.redis_port: int = redis_port
        else:
            self.redis_port = int(os.getenv("NEURAL_ANALYSIS_REDIS_PORT", "6379"))

        if redis_db is not None:
            self.redis_db: int = redis_db
        else:
            self.redis_db = int(os.getenv("NEURAL_ANALYSIS_REDIS_DB", "0"))

        env_password = os.getenv("NEURAL_ANALYSIS_REDIS_PASSWORD", "")
        if redis_password is not None:
            self.redis_password: str | None = redis_password
        else:
            self.redis_password = env_password or None

        # SQL settings
        env_sql = _env_flag("NEURAL_ANALYSIS_USE_SQL")
        if use_sql is not None:
            self.use_sql = use_sql
        elif env_sql is not None:
            self.use_sql = env_sql
        else:
            # Default to True to keep DuckDB metadata enabled when installed
            self.use_sql = True
        sql_path_env = os.getenv("NEURAL_ANALYSIS_SQL_PATH", ".neural_analysis_meta.db")
        self.sql_path = Path(sql_path) if sql_path else Path(sql_path_env)

        # Cache settings
        self.cache_ttl = cache_ttl or int(
            os.getenv("NEURAL_ANALYSIS_CACHE_TTL", "3600")
        )
        self.cache_max_size_mb = cache_max_size_mb or int(
            os.getenv("NEURAL_ANALYSIS_CACHE_MAX_SIZE_MB", "100")
        )
        self.cache_namespace = cache_namespace or os.getenv(
            "NEURAL_ANALYSIS_CACHE_NAMESPACE", "neural_analysis"
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns
        -------
        dict
            Configuration as dictionary
        """
        return {
            "use_redis": self.use_redis,
            "use_sql": self.use_sql,
            "redis_host": self.redis_host,
            "redis_port": self.redis_port,
            "redis_db": self.redis_db,
            "redis_password": "***" if self.redis_password else None,
            "sql_path": str(self.sql_path),
            "cache_ttl": self.cache_ttl,
            "cache_max_size_mb": self.cache_max_size_mb,
            "cache_namespace": self.cache_namespace,
        }

    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"StorageConfig({self.to_dict()})"


# Global configuration instance
_config: StorageConfig | None = None


def get_config() -> StorageConfig:
    """Get global storage configuration.

    Returns
    -------
    StorageConfig
        Global configuration instance (creates if doesn't exist)
    """
    global _config
    if _config is None:
        _config = StorageConfig()
    return _config


def set_config(config: StorageConfig) -> None:
    """Set global storage configuration.

    Parameters
    ----------
    config : StorageConfig
        Configuration instance to use globally
    """
    global _config
    _config = config

