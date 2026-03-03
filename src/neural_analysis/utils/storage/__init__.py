"""Storage backend abstraction for multi-layer data access.

This module provides abstract base classes for storage backends (Redis, SQL, HDF5)
and a unified interface for data access with automatic fallback.

Architecture:
- Layer 1: Redis Cache (fastest, optional) - in-memory caching
- Layer 2: SQL Metadata (fast, optional) - metadata indexing and queries
- Layer 3: HDF5 Storage (persistent, required) - compressed hierarchical storage

All backends gracefully degrade if dependencies are unavailable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "StorageBackend",
    "CacheBackend",
    "MetadataBackend",
    "PersistentBackend",
]


class StorageBackend(ABC):
    """Abstract base class for storage backends.

    Defines the unified interface that all storage backends must implement.
    """

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this storage backend is available.

        Returns
        -------
        bool
            True if backend is available and ready to use, False otherwise.
        """
        pass

    @abstractmethod
    def save(
        self,
        key: str,
        data: Any,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Save data with optional metadata.

        Parameters
        ----------
        key : str
            Unique identifier for the data
        data : Any
            Data to save (numpy array, pandas DataFrame, dict, etc.)
        metadata : dict, optional
            Additional metadata to store with the data

        Returns
        -------
        bool
            True if save was successful, False otherwise
        """
        pass

    @abstractmethod
    def load(self, key: str) -> Any | None:
        """Load data by key.

        Parameters
        ----------
        key : str
            Unique identifier for the data

        Returns
        -------
        Any or None
            Loaded data if found, None otherwise
        """
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if data exists for a given key.

        Parameters
        ----------
        key : str
            Unique identifier to check

        Returns
        -------
        bool
            True if data exists, False otherwise
        """
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete data by key.

        Parameters
        ----------
        key : str
            Unique identifier to delete

        Returns
        -------
        bool
            True if deletion was successful, False otherwise
        """
        pass


class CacheBackend(StorageBackend):
    """Abstract base class for cache backends (e.g., Redis).

    Cache backends support TTL (time-to-live) for automatic expiration.
    """

    @abstractmethod
    def save(
        self,
        key: str,
        data: Any,
        metadata: dict[str, Any] | None = None,
        ttl: int | None = None,
    ) -> bool:
        """Save data to cache with optional TTL.

        Parameters
        ----------
        key : str
            Unique identifier for the data
        data : Any
            Data to cache
        metadata : dict, optional
            Additional metadata (may not be supported by all cache backends)
        ttl : int, optional
            Time-to-live in seconds. If None, uses default TTL.

        Returns
        -------
        bool
            True if save was successful, False otherwise
        """
        pass

    @abstractmethod
    def invalidate(self, pattern: str) -> int:
        """Invalidate cache entries matching a pattern.

        Parameters
        ----------
        pattern : str
            Pattern to match keys (backend-specific format)

        Returns
        -------
        int
            Number of keys invalidated
        """
        pass


class MetadataBackend(StorageBackend):
    """Abstract base class for metadata indexing backends (e.g., SQL).

    Metadata backends support querying and filtering by metadata attributes.
    """

    @abstractmethod
    def query(
        self,
        filters: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """Query metadata with optional filters.

        Parameters
        ----------
        filters : dict, optional
            Dictionary of attribute filters (e.g., {"metric": "euclidean"})
        limit : int, optional
            Maximum number of results to return

        Returns
        -------
        DataFrame
            Query results as a pandas DataFrame
        """
        pass

    @abstractmethod
    def index(
        self,
        key: str,
        metadata: dict[str, Any],
        file_path: str | None = None,
        group_path: str | None = None,
    ) -> bool:
        """Index metadata for a data item.

        Parameters
        ----------
        key : str
            Unique identifier for the data
        metadata : dict
            Metadata attributes to index
        file_path : str, optional
            Path to the data file (for HDF5)
        group_path : str, optional
            Group path within the file (for HDF5)

        Returns
        -------
        bool
            True if indexing was successful, False otherwise
        """
        pass


class PersistentBackend(StorageBackend):
    """Abstract base class for persistent storage backends (e.g., HDF5).

    Persistent backends provide long-term storage with compression and hierarchy.
    """

    @abstractmethod
    def save(
        self,
        key: str,
        data: Any,
        metadata: dict[str, Any] | None = None,
        compression: str = "gzip",
    ) -> bool:
        """Save data to persistent storage with compression.

        Parameters
        ----------
        key : str
            Unique identifier for the data (may include hierarchical path)
        data : Any
            Data to save
        metadata : dict, optional
            Additional metadata to store as attributes
        compression : str, default='gzip'
            Compression algorithm to use

        Returns
        -------
        bool
            True if save was successful, False otherwise
        """
        pass
