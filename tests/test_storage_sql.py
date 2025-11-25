"""Tests for SQL metadata backend."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from neural_analysis.utils.storage.config import StorageConfig
from neural_analysis.utils.storage.sql_metadata import SQLMetadata, is_sql_available


class TestSQLMetadata:
    """Tests for SQLMetadata class."""

    def test_is_sql_available(self) -> None:
        """Test is_sql_available function."""
        result = is_sql_available()
        assert isinstance(result, bool)

    def test_sql_metadata_initialization_without_duckdb(self) -> None:
        """Test SQLMetadata initialization when DuckDB unavailable."""
        config = StorageConfig(use_sql=False)
        metadata = SQLMetadata(config)
        assert not metadata.is_available()

    def test_sql_metadata_operations_unavailable(self) -> None:
        """Test metadata operations when DuckDB unavailable."""
        config = StorageConfig(use_sql=False)
        metadata = SQLMetadata(config)

        # Index should return empty string
        result = metadata.index_dataset("test.h5", "group/path", {"key": "value"})
        assert result == ""

        # Query should return empty DataFrame
        result = metadata.query_datasets()
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

        # Exists should return False
        assert metadata.exists("test_id") is False

        # Delete should return False
        assert metadata.delete("test_id") is False

    @pytest.mark.integration
    def test_sql_metadata_with_duckdb(self) -> None:
        """Integration test with DuckDB (if available)."""
        if not is_sql_available():
            pytest.skip("DuckDB not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            sql_path = Path(tmpdir) / "test_meta.db"
            config = StorageConfig(use_sql=True, sql_path=sql_path)
            metadata = SQLMetadata(config)

            if not metadata.is_available():
                pytest.skip("SQL metadata not available")

            # Test indexing dataset
            dataset_id = metadata.index_dataset(
                file_path="test_data.h5",
                group_path="experiments/exp1",
                metadata={"n_samples": 100, "date": "2025-01-01"},
            )
            assert dataset_id != ""
            assert metadata.exists(dataset_id) is True

            # Test querying datasets
            results = metadata.query_datasets()
            assert len(results) > 0
            assert "file_path" in results.columns

            # Test filtering
            filtered = metadata.query_datasets(filters={"file_path": "test_data.h5"})
            assert len(filtered) > 0

            # Test indexing comparison
            comparison_id = metadata.index_comparison(
                dataset_i="dataset_A",
                dataset_j="dataset_B",
                metric="euclidean",
                file_path="results.h5",
                group_path="euclidean/dataset_A___dataset_B",
                mode="between",
                value_type="scalar",
                metadata={"value": 42.5},
            )
            assert comparison_id != ""

            # Test querying comparisons
            comp_results = metadata.query_comparisons(filters={"metric": "euclidean"})
            assert len(comp_results) > 0
            assert "metric" in comp_results.columns

            # Test delete
            deleted = metadata.delete(dataset_id)
            assert deleted is True
            assert metadata.exists(dataset_id) is False

    @pytest.mark.integration
    def test_sql_metadata_schema_initialization(self) -> None:
        """Test that database schema is properly initialized."""
        if not is_sql_available():
            pytest.skip("DuckDB not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            sql_path = Path(tmpdir) / "test_schema.db"
            config = StorageConfig(use_sql=True, sql_path=sql_path)
            metadata = SQLMetadata(config)

            if not metadata.is_available():
                pytest.skip("SQL metadata not available")

            # Schema should be initialized, try inserting data
            dataset_id = metadata.index_dataset("test.h5", None, {})
            assert dataset_id != ""

    def test_sql_metadata_interface_compatibility(self) -> None:
        """Test that SQLMetadata implements StorageBackend interface."""
        config = StorageConfig(use_sql=False)
        metadata = SQLMetadata(config)

        # Test interface methods exist
        assert hasattr(metadata, "save")
        assert hasattr(metadata, "load")
        assert hasattr(metadata, "exists")
        assert hasattr(metadata, "delete")
        assert hasattr(metadata, "query")
        assert hasattr(metadata, "index")

        # Test save/load (not supported, should return False/None)
        assert metadata.save("key", "data") is False
        assert metadata.load("key") is None



