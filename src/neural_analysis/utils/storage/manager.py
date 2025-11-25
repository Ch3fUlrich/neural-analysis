"""Unified storage manager orchestrating Redis -> SQL -> HDF5 access.

Provides a single interface for multi-layer data access with automatic fallback.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from neural_analysis.utils.logging import get_logger
from neural_analysis.utils.storage.config import get_config
from neural_analysis.utils.storage.redis_cache import RedisCache
from neural_analysis.utils.storage.sql_metadata import SQLMetadata

logger = get_logger(__name__)

__all__ = ["StorageManager"]


class StorageManager:
    """Unified storage manager with multi-layer access pattern.

    Access pattern:
    1. Try Redis cache (if enabled and available)
    2. Try SQL metadata query (if enabled and available)
    3. Fallback to HDF5 direct access (always works)

    All layers gracefully degrade if dependencies are unavailable.
    """

    def __init__(self, config: Any | None = None) -> None:
        """Initialize storage manager.

        Parameters
        ----------
        config : StorageConfig, optional
            Configuration instance. If None, uses global config.
        """
        self.config = config or get_config()
        self.cache = RedisCache(self.config)
        self.metadata = SQLMetadata(self.config)

        logger.info(
            f"Storage manager initialized: "
            f"cache={'enabled' if self.cache.is_available() else 'disabled'}, "
            f"metadata={'enabled' if self.metadata.is_available() else 'disabled'}"
        )

    def __enter__(self) -> "StorageManager":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def save_data(
        self,
        key: str,
        data: Any,
        metadata: dict[str, Any] | None = None,
        file_path: str | Path | None = None,
        group_path: str | None = None,
        use_cache: bool = True,
        cache_ttl: int | None = None,
    ) -> bool:
        """Save data to storage layers.

        Saves to:
        1. Redis cache (if enabled)
        2. SQL metadata index (if enabled and file_path provided)
        3. HDF5 (via io.py functions - caller must handle)

        Parameters
        ----------
        key : str
            Unique identifier for the data
        data : Any
            Data to save
        metadata : dict, optional
            Additional metadata
        file_path : str or Path, optional
            Path to HDF5 file (for metadata indexing)
        group_path : str, optional
            Group path within file (for metadata indexing)
        use_cache : bool, default=True
            Whether to cache in Redis
        cache_ttl : int, optional
            Cache TTL in seconds (overrides config default)

        Returns
        -------
        bool
            True if at least one layer saved successfully
        """
        success = False

        # Cache in Redis
        if use_cache and data is not None and self.cache.is_available():
            if self.cache_set(key, data, ttl=cache_ttl, metadata=metadata):
                success = True

        # Index in SQL metadata
        if file_path and self.metadata.is_available():
            if group_path:
                # Index as comparison if group_path provided
                metadata_dict = metadata or {}
                dataset_i = metadata_dict.get("dataset_i", "")
                dataset_j = metadata_dict.get("dataset_j", "")
                metric = metadata_dict.get("metric", "")
                mode = metadata_dict.get("mode")
                value_type = metadata_dict.get("value_type")

                if dataset_i and dataset_j and metric:
                    self.metadata.index_comparison(
                        dataset_i=dataset_i,
                        dataset_j=dataset_j,
                        metric=metric,
                        file_path=file_path,
                        group_path=group_path,
                        mode=mode,
                        value_type=value_type,
                        metadata=metadata,
                    )
                    success = True
            else:
                # Index as dataset
                dataset_id = self.metadata.index_dataset(file_path, group_path, metadata)
                if dataset_id:
                    success = True
                    logger.debug(f"Indexed in metadata: {key}")

        return success

    def load_data(
        self,
        key: str,
        use_cache: bool = True,
        hdf5_loader: Any | None = None,
    ) -> Any | None:
        """Load data from storage layers.

        Loads from:
        1. Redis cache (if enabled)
        2. SQL metadata query + HDF5 (if enabled)
        3. HDF5 direct (via provided loader)

        Parameters
        ----------
        key : str
            Unique identifier for the data
        use_cache : bool, default=True
            Whether to check cache first
        hdf5_loader : callable, optional
            Function to load from HDF5 if cache/metadata miss.
            Should accept (file_path, group_path) and return data.

        Returns
        -------
        Any or None
            Loaded data if found, None otherwise
        """
        # Try cache first
        if use_cache and self.cache.is_available():
            cached = self.cache_get(key)
            if cached is not None:
                return cached

        # Try SQL metadata + HDF5
        if self.metadata.is_available() and hdf5_loader:
            # Query metadata for file_path and group_path
            logger.debug(
                "Metadata-assisted loading is not yet implemented; falling back to loader"
            )

        # Fallback to HDF5 via provided loader
        if hdf5_loader:
            try:
                data = hdf5_loader()
                # Cache the result for future use
                if use_cache and self.cache.is_available() and data is not None:
                    self.cache_set(key, data)
                return data
            except Exception as e:
                logger.warning(f"Error loading from HDF5 for key {key}: {e}")

        return None

    def query_data(
        self,
        filters: dict[str, Any] | None = None,
        use_sql: bool = True,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """Query data using metadata index.

        Parameters
        ----------
        filters : dict, optional
            Query filters (e.g., {"metric": "euclidean", "mode": "between"})
        use_sql : bool, default=True
            Whether to use SQL metadata for queries
        limit : int, optional
            Maximum number of results

        Returns
        -------
        DataFrame
            Query results with metadata
        """
        if use_sql and self.metadata.is_available():
            return self.metadata.query_comparisons(filters=filters, limit=limit)

        # Fallback: return empty DataFrame
        logger.warning("SQL metadata not available for querying")
        return pd.DataFrame()

    def invalidate_cache(self, pattern: str) -> int:
        """Invalidate cache entries matching a pattern.

        Parameters
        ----------
        pattern : str
            Pattern to match keys (Redis wildcard format)

        Returns
        -------
        int
            Number of keys invalidated
        """
        if self.cache.is_available():
            return self.cache.invalidate(pattern)
        return 0

    def cache_get(self, key: str) -> Any | None:
        """Get cached data by key."""
        if self.cache.is_available():
            return self.cache.get_cached(key)
        return None

    def cache_set(
        self,
        key: str,
        data: Any,
        *,
        ttl: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Save data to cache."""
        if self.cache.is_available():
            return self.cache.set_cached(key, data, ttl=ttl, metadata=metadata)
        return False

    def cache_delete(self, key: str) -> bool:
        """Delete cached entry."""
        if self.cache.is_available():
            return self.cache.delete(key)
        return False

    def index_comparison(
        self,
        dataset_i: str,
        dataset_j: str,
        metric: str,
        file_path: str | Path,
        group_path: str,
        mode: str | None = None,
        value_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Index a comparison result in metadata.

        Parameters
        ----------
        dataset_i : str
            First dataset identifier
        dataset_j : str
            Second dataset identifier
        metric : str
            Metric name
        file_path : str or Path
            Path to HDF5 file
        group_path : str
            Group path within file
        mode : str, optional
            Comparison mode
        value_type : str, optional
            Value type
        metadata : dict, optional
            Additional metadata

        Returns
        -------
        str
            Comparison ID
        """
        if self.metadata.is_available():
            return self.metadata.index_comparison(
                dataset_i=dataset_i,
                dataset_j=dataset_j,
                metric=metric,
                file_path=file_path,
                group_path=group_path,
                mode=mode,
                value_type=value_type,
                metadata=metadata,
            )
        return ""

    def get_cache_stats(self) -> dict[str, Any]:
        """Get statistics about cache usage.

        Returns
        -------
        dict
            Cache statistics (availability, etc.)
        """
        return {
            "cache_available": self.cache.is_available(),
            "metadata_available": self.metadata.is_available(),
            "config": self.config.to_dict(),
        }

    def close(self) -> None:
        """Release cache and metadata resources."""
        cache_close = getattr(self.cache, "close", None)
        if callable(cache_close):
            cache_close()

        metadata_close = getattr(self.metadata, "close", None)
        if callable(metadata_close):
            metadata_close()

