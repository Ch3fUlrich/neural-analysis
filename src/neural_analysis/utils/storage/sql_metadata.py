"""SQL metadata indexing backend using DuckDB.

Provides fast metadata queries and indexing for HDF5 datasets.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from neural_analysis.utils.logging import get_logger
from neural_analysis.utils.storage.config import get_config

try:
    import duckdb

    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False

logger = get_logger(__name__)

__all__ = ["SQLMetadata", "is_sql_available"]


def is_sql_available() -> bool:
    """Check if DuckDB is available.

    Returns
    -------
    bool
        True if DuckDB package is installed
    """
    return DUCKDB_AVAILABLE


class SQLMetadata:
    """DuckDB-based metadata indexing backend.

    Provides fast queries for dataset locations, comparisons, and metadata.
    Automatically falls back to no-op if DuckDB is unavailable.
    """

    def __init__(self, config: Any | None = None) -> None:
        """Initialize SQL metadata backend.

        Parameters
        ----------
        config : StorageConfig, optional
            Configuration instance. If None, uses global config.
        """

        self.config = config or get_config()
        self._conn: duckdb.DuckDBPyConnection | None = None
        self._available = False

        if DUCKDB_AVAILABLE and self.config.use_sql:
            try:
                # Ensure parent directory exists
                self.config.sql_path.parent.mkdir(parents=True, exist_ok=True)
                self._conn = duckdb.connect(str(self.config.sql_path))
                self._available = True
                self._initialize_schema()
                logger.info(f"SQL metadata database initialized: {self.config.sql_path}")
            except Exception as e:
                logger.warning(f"SQL metadata unavailable: {e}. Continuing without indexing.")
                self._conn = None
                self._available = False
        else:
            if not DUCKDB_AVAILABLE:
                logger.debug("DuckDB package not installed. Metadata indexing disabled.")
            elif not self.config.use_sql:
                logger.debug("SQL disabled in configuration. Metadata indexing disabled.")
            self._available = False

    def _connection(self) -> duckdb.DuckDBPyConnection | None:
        """Return active DuckDB connection if available."""
        if not self.is_available():
            return None
        return self._conn

    def _initialize_schema(self) -> None:
        """Initialize database schema with required tables."""
        conn = self._connection()
        if conn is None:
            return
        assert conn is not None

        # Datasets table: tracks HDF5 files and their group paths
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS datasets (
                id VARCHAR PRIMARY KEY,
                file_path VARCHAR NOT NULL,
                group_path VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata_json TEXT
            )
            """
        )

        # Comparisons table: tracks comparison results
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS comparisons (
                id VARCHAR PRIMARY KEY,
                dataset_i VARCHAR NOT NULL,
                dataset_j VARCHAR NOT NULL,
                metric VARCHAR NOT NULL,
                mode VARCHAR,
                file_path VARCHAR NOT NULL,
                group_path VARCHAR NOT NULL,
                value_type VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata_json TEXT
            )
            """
        )

        # Chunks table: tracks data chunks for large datasets
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id VARCHAR PRIMARY KEY,
                dataset_id VARCHAR NOT NULL,
                chunk_key VARCHAR NOT NULL,
                file_path VARCHAR NOT NULL,
                offset_start INTEGER,
                offset_end INTEGER,
                size_bytes INTEGER,
                compression VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (dataset_id) REFERENCES datasets(id)
            )
            """
        )

        # Create indices for fast queries
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_comparisons_metric ON comparisons(metric)"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_comparisons_mode ON comparisons(mode)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_comparisons_datasets ON comparisons(dataset_i, dataset_j)"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_datasets_path ON datasets(file_path)")

        logger.debug("Database schema initialized")

    def is_available(self) -> bool:
        """Check if SQL metadata backend is available.

        Returns
        -------
        bool
            True if DuckDB is connected and ready
        """
        return self._available and self._conn is not None

    def index_dataset(
        self,
        file_path: str | Path,
        group_path: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Index a dataset in the metadata database.

        Parameters
        ----------
        file_path : str or Path
            Path to HDF5 file
        group_path : str, optional
            Group path within the file
        metadata : dict, optional
            Additional metadata to store

        Returns
        -------
        str
            Dataset ID (generated from file_path and group_path)
        """
        conn = self._connection()
        if conn is None:
            return ""

        try:
            file_path_str = str(Path(file_path).resolve())
            group_path_str = group_path or ""
            dataset_id = f"{file_path_str}:{group_path_str}"

            metadata_json = json.dumps(metadata) if metadata else None

            conn.execute(
                """
                INSERT OR REPLACE INTO datasets (id, file_path, group_path, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                [dataset_id, file_path_str, group_path_str, metadata_json, datetime.now(UTC)],
            )

            logger.debug(f"Indexed dataset: {dataset_id}")
            return dataset_id
        except Exception as e:
            logger.warning(f"Error indexing dataset {file_path}: {e}")
            return ""

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
        """Index a comparison result.

        Parameters
        ----------
        dataset_i : str
            First dataset identifier
        dataset_j : str
            Second dataset identifier
        metric : str
            Metric name
        file_path : str or Path
            Path to HDF5 file containing the result
        group_path : str
            Group path within the file
        mode : str, optional
            Comparison mode (within, between, all-pairs)
        value_type : str, optional
            Type of value (scalar, matrix, dict)
        metadata : dict, optional
            Additional metadata

        Returns
        -------
        str
            Comparison ID
        """
        conn = self._connection()
        if conn is None:
            return ""

        try:
            comparison_id = f"{dataset_i}:{dataset_j}:{metric}:{mode or 'default'}"
            file_path_str = str(Path(file_path).resolve())
            metadata_json = json.dumps(metadata) if metadata else None

            conn.execute(
                """
                INSERT OR REPLACE INTO comparisons 
                (id, dataset_i, dataset_j, metric, mode, file_path, group_path, value_type, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    comparison_id,
                    dataset_i,
                    dataset_j,
                    metric,
                    mode,
                    file_path_str,
                    group_path,
                    value_type,
                    metadata_json,
                    datetime.now(UTC),
                ],
            )

            logger.debug(f"Indexed comparison: {comparison_id}")
            return comparison_id
        except Exception as e:
            logger.warning(f"Error indexing comparison {dataset_i}/{dataset_j}: {e}")
            return ""

    def query_datasets(
        self, filters: dict[str, Any] | None = None, limit: int | None = None
    ) -> pd.DataFrame:
        """Query datasets with optional filters.

        Parameters
        ----------
        filters : dict, optional
            Dictionary of filters (e.g., {"file_path": "*.h5"})
        limit : int, optional
            Maximum number of results

        Returns
        -------
        DataFrame
            Query results
        """
        conn = self._connection()
        if conn is None:
            return pd.DataFrame()

        try:
            query = "SELECT * FROM datasets WHERE 1=1"
            params: list[Any] = []

            if filters:
                for key, value in filters.items():
                    if key == "file_path" and isinstance(value, str):
                        if "*" in value:
                            pattern = value.replace("*", "%")
                            query += " AND file_path LIKE ?"
                            params.append(pattern)
                        else:
                            query += " AND file_path = ?"
                            params.append(str(Path(value).resolve()))
                    else:
                        query += f" AND {key} = ?"
                        params.append(value)

            query += " ORDER BY created_at DESC"

            if limit:
                query += f" LIMIT {limit}"

            result = conn.execute(query, params).df()

            # Parse metadata_json if present
            if "metadata_json" in result.columns:
                result["metadata"] = result["metadata_json"].apply(
                    lambda x: json.loads(x) if x else {}
                )

            return result
        except Exception as e:
            logger.warning(f"Error querying datasets: {e}")
            return pd.DataFrame()

    def query_comparisons(
        self, filters: dict[str, Any] | None = None, limit: int | None = None
    ) -> pd.DataFrame:
        """Query comparisons with optional filters.

        Parameters
        ----------
        filters : dict, optional
            Dictionary of filters (e.g., {"metric": "euclidean", "mode": "between"})
        limit : int, optional
            Maximum number of results

        Returns
        -------
        DataFrame
            Query results
        """
        conn = self._connection()
        if conn is None:
            return pd.DataFrame()

        try:
            query = "SELECT * FROM comparisons WHERE 1=1"
            params: list[Any] = []

            if filters:
                for key, value in filters.items():
                    query += f" AND {key} = ?"
                    params.append(value)

            query += " ORDER BY created_at DESC"

            if limit:
                query += f" LIMIT {limit}"

            result = conn.execute(query, params).df()

            # Parse metadata_json if present
            if "metadata_json" in result.columns:
                result["metadata"] = result["metadata_json"].apply(
                    lambda x: json.loads(x) if x else {}
                )

            return result
        except Exception as e:
            logger.warning(f"Error querying comparisons: {e}")
            return pd.DataFrame()

    def save(self, key: str, data: Any, metadata: dict[str, Any] | None = None) -> bool:
        """Save operation (not supported for metadata backend).

        Metadata backend only indexes, doesn't store data.

        Parameters
        ----------
        key : str
            Not used
        data : Any
            Not used
        metadata : dict, optional
            Not used

        Returns
        -------
        bool
            Always False (operation not supported)
        """
        return False

    def load(self, key: str) -> Any | None:
        """Load operation (not supported for metadata backend).

        Metadata backend only indexes, doesn't store data.

        Parameters
        ----------
        key : str
            Not used

        Returns
        -------
        None
            Always None (operation not supported)
        """
        return None

    def exists(self, key: str) -> bool:
        """Check if dataset exists in metadata.

        Parameters
        ----------
        key : str
            Dataset ID or file_path:group_path

        Returns
        -------
        bool
            True if dataset is indexed
        """
        conn = self._connection()
        if conn is None:
            return False

        try:
            result = conn.execute(
                "SELECT COUNT(*) as count FROM datasets WHERE id = ?", [key]
            ).fetchone()
            return result[0] > 0 if result else False
        except Exception:
            return False

    def delete(self, key: str) -> bool:
        """Delete dataset from metadata index.

        Parameters
        ----------
        key : str
            Dataset ID to delete

        Returns
        -------
        bool
            True if deletion was successful
        """
        conn = self._connection()
        if conn is None:
            return False

        try:
            conn.execute("DELETE FROM datasets WHERE id = ?", [key])
            logger.debug(f"Deleted dataset from index: {key}")
            return True
        except Exception as e:
            logger.warning(f"Error deleting dataset {key}: {e}")
            return False

    def index(
        self,
        key: str,
        metadata: dict[str, Any],
        file_path: str | None = None,
        group_path: str | None = None,
    ) -> bool:
        """Index metadata (alias for index_dataset for interface compatibility).

        Parameters
        ----------
        key : str
            Dataset identifier
        metadata : dict
            Metadata to index
        file_path : str, optional
            Path to data file
        group_path : str, optional
            Group path within file

        Returns
        -------
        bool
            True if indexing was successful
        """
        if file_path:
            dataset_id = self.index_dataset(file_path, group_path, metadata)
            return bool(dataset_id)
        return False

    def query(
        self, filters: dict[str, Any] | None = None, limit: int | None = None
    ) -> pd.DataFrame:
        """Query metadata (alias for query_comparisons for interface compatibility).

        Parameters
        ----------
        filters : dict, optional
            Query filters
        limit : int, optional
            Maximum results

        Returns
        -------
        DataFrame
            Query results
        """
        return self.query_comparisons(filters=filters, limit=limit)

    def close(self) -> None:
        """Close DuckDB connection if open."""
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                logger.debug("Failed to close DuckDB connection", exc_info=True)
        self._conn = None
        self._available = False

