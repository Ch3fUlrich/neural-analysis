"""Tests for storage configuration utilities."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from neural_analysis.utils.storage.config import (
    StorageConfig,
    get_config,
    set_config,
)


class TestEnvFlag:
    """Tests for _env_flag function."""

    def test_env_flag_true_values(self):
        """Test _env_flag with true values (covers line 23)."""
        from neural_analysis.utils.storage.config import _env_flag

        for true_val in ["true", "True", "TRUE", "1", "yes", "Yes", "on", "ON"]:
            with patch.dict(os.environ, {"TEST_FLAG": true_val}):
                result = _env_flag("TEST_FLAG")
                assert result is True

    def test_env_flag_false_values(self):
        """Test _env_flag with false values."""
        from neural_analysis.utils.storage.config import _env_flag

        for false_val in ["false", "False", "0", "no", "off", ""]:
            with patch.dict(os.environ, {"TEST_FLAG": false_val}):
                result = _env_flag("TEST_FLAG")
                if false_val == "":
                    assert result is None
                else:
                    assert result is False

    def test_env_flag_none(self):
        """Test _env_flag when env var is not set (covers line 22)."""
        from neural_analysis.utils.storage.config import _env_flag

        with patch.dict(os.environ, {}, clear=True):
            result = _env_flag("NONEXISTENT_FLAG")
            assert result is None


class TestStorageConfig:
    """Tests for StorageConfig class."""

    def test_default_config(self):
        """Test StorageConfig with default values."""
        config = StorageConfig()
        assert config.use_redis is True
        assert config.use_sql is True
        assert config.redis_host == "localhost"
        assert config.redis_port == 6379
        assert config.redis_db == 0
        assert isinstance(config.sql_path, Path)

    def test_custom_redis_host(self):
        """Test custom redis_host (covers line 79)."""
        config = StorageConfig(redis_host="custom-host")
        assert config.redis_host == "custom-host"

    def test_custom_redis_port(self):
        """Test custom redis_port (covers line 84)."""
        config = StorageConfig(redis_port=6380)
        assert config.redis_port == 6380

    def test_custom_redis_db(self):
        """Test custom redis_db (covers line 89)."""
        config = StorageConfig(redis_db=1)
        assert config.redis_db == 1

    def test_custom_redis_password(self):
        """Test custom redis_password (covers line 95)."""
        config = StorageConfig(redis_password="secret")
        assert config.redis_password == "secret"

    def test_env_redis_flag(self):
        """Test use_redis from environment (covers line 74)."""
        with patch.dict(os.environ, {"NEURAL_ANALYSIS_USE_REDIS": "false"}):
            config = StorageConfig()
            assert config.use_redis is False

    def test_env_sql_flag(self):
        """Test use_sql from environment (covers line 104)."""
        with patch.dict(os.environ, {"NEURAL_ANALYSIS_USE_SQL": "false"}):
            config = StorageConfig()
            assert config.use_sql is False

    def test_env_redis_host(self):
        """Test redis_host from environment."""
        with patch.dict(os.environ, {"NEURAL_ANALYSIS_REDIS_HOST": "env-host"}):
            config = StorageConfig()
            assert config.redis_host == "env-host"

    def test_env_redis_port(self):
        """Test redis_port from environment."""
        with patch.dict(os.environ, {"NEURAL_ANALYSIS_REDIS_PORT": "6380"}):
            config = StorageConfig()
            assert config.redis_port == 6380

    def test_env_redis_db(self):
        """Test redis_db from environment."""
        with patch.dict(os.environ, {"NEURAL_ANALYSIS_REDIS_DB": "2"}):
            config = StorageConfig()
            assert config.redis_db == 2

    def test_env_redis_password(self):
        """Test redis_password from environment."""
        with patch.dict(os.environ, {"NEURAL_ANALYSIS_REDIS_PASSWORD": "env-secret"}):
            config = StorageConfig()
            assert config.redis_password == "env-secret"

    def test_env_sql_path(self):
        """Test sql_path from environment."""
        with patch.dict(os.environ, {"NEURAL_ANALYSIS_SQL_PATH": "/tmp/test.db"}):
            config = StorageConfig()
            assert str(config.sql_path) == "/tmp/test.db"

    def test_custom_sql_path(self):
        """Test custom sql_path."""
        custom_path = Path("/custom/path.db")
        config = StorageConfig(sql_path=custom_path)
        assert config.sql_path == custom_path

    def test_custom_cache_settings(self):
        """Test custom cache settings."""
        config = StorageConfig(cache_ttl=7200, cache_max_size_mb=200, cache_namespace="test")
        assert config.cache_ttl == 7200
        assert config.cache_max_size_mb == 200
        assert config.cache_namespace == "test"

    def test_to_dict(self):
        """Test to_dict method."""
        config = StorageConfig(redis_password="secret")
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict["redis_password"] == "***"  # Should mask password
        assert "use_redis" in config_dict
        assert "use_sql" in config_dict

    def test_repr(self):
        """Test __repr__ method."""
        config = StorageConfig()
        repr_str = repr(config)
        assert "StorageConfig" in repr_str
        assert isinstance(repr_str, str)


class TestGetConfig:
    """Tests for get_config function."""

    def test_get_config_creates_default(self):
        """Test get_config creates default config."""
        # Reset global config
        set_config(None)  # type: ignore
        config = get_config()
        assert isinstance(config, StorageConfig)
        assert config.use_redis is True

    def test_get_config_returns_same_instance(self):
        """Test get_config returns same instance."""
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2


class TestSetConfig:
    """Tests for set_config function."""

    def test_set_config(self):
        """Test set_config (covers line 175)."""
        custom_config = StorageConfig(use_redis=False, use_sql=False)
        set_config(custom_config)
        retrieved = get_config()
        assert retrieved is custom_config
        assert retrieved.use_redis is False
        assert retrieved.use_sql is False

    def test_set_config_none_resets(self):
        """Test set_config with None resets to default."""
        custom_config = StorageConfig(use_redis=False)
        set_config(custom_config)
        set_config(None)  # type: ignore
        config = get_config()
        # Should create new default config
        assert config.use_redis is True



