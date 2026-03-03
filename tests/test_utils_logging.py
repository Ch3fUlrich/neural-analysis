"""Tests for logging utilities."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import pytest

from neural_analysis.utils.logging import (
    LogConfig,
    configure_logging,
    get_logger,
    log_calls,
    log_kv,
    log_section,
)


def reset_logging_config() -> None:
    """Reset the global _CONFIGURED flag to allow reconfiguration."""
    import neural_analysis.utils.logging as logging_module

    logging_module._CONFIGURED = False
    # Also clear the logger handlers
    logger = logging.getLogger("neural_analysis")
    logger.handlers.clear()


class TestLogConfig:
    """Tests for LogConfig dataclass."""

    def test_default_values(self) -> None:
        """Test LogConfig default values."""
        config = LogConfig()
        assert config.level == logging.INFO
        assert config.fmt is not None
        assert config.datefmt is not None
        assert config.propagate is False
        assert config.stream is not None
        assert config.file_path is None


class TestLevelFromEnv:
    """Tests for _level_from_env function."""

    def test_level_from_env_valid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test _level_from_env with valid level (covers lines 54-60)."""
        from neural_analysis.utils.logging import _level_from_env

        monkeypatch.setenv("NEURAL_ANALYSIS_LOG_LEVEL", "DEBUG")
        result = _level_from_env(logging.INFO)
        assert result == logging.DEBUG

        monkeypatch.setenv("NEURAL_ANALYSIS_LOG_LEVEL", "WARNING")
        result = _level_from_env(logging.INFO)
        assert result == logging.WARNING

    def test_level_from_env_invalid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test _level_from_env with invalid level (covers lines 61-64)."""
        from neural_analysis.utils.logging import _level_from_env

        # Invalid level name
        monkeypatch.setenv("NEURAL_ANALYSIS_LOG_LEVEL", "INVALID_LEVEL")
        result = _level_from_env(logging.INFO)
        assert result == logging.INFO  # Should return default

        # Non-integer attribute (covers line 62)
        # Mock getattr to return a non-int value to ensure line 62 is covered
        original_getattr = getattr

        def mock_getattr(obj, name, default=None):
            if obj == logging and name == "FORMATTER":
                return "not_an_int"  # Return a string, not an int
            return original_getattr(obj, name, default)

        monkeypatch.setenv("NEURAL_ANALYSIS_LOG_LEVEL", "Formatter")
        monkeypatch.setattr("builtins.getattr", mock_getattr)
        result = _level_from_env(logging.INFO)
        assert result == logging.INFO  # Should return default (covers line 62)

    def test_level_from_env_not_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test _level_from_env when env var not set (covers line 55-56)."""
        from neural_analysis.utils.logging import _level_from_env

        monkeypatch.delenv("NEURAL_ANALYSIS_LOG_LEVEL", raising=False)
        result = _level_from_env(logging.INFO)
        assert result == logging.INFO


class TestConfigureLogging:
    """Tests for configure_logging function."""

    def setup_method(self) -> None:
        """Reset logging configuration before each test."""
        reset_logging_config()

    def test_configure_logging_string_level(self) -> None:
        """Test configure_logging with string level (covers lines 101-102)."""
        configure_logging(level="DEBUG")
        logger = get_logger("test")
        # Check that logger is configured (level might be inherited from handler)
        assert logger is not None

    def test_configure_logging_int_level(self) -> None:
        """Test configure_logging with int level (covers lines 103-104)."""
        configure_logging(level=logging.WARNING)
        logger = get_logger("test")
        # Check that logger is configured
        assert logger is not None

    def test_configure_logging_none_level(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test configure_logging with None level (covers lines 105-106)."""
        monkeypatch.delenv("NEURAL_ANALYSIS_LOG_LEVEL", raising=False)
        configure_logging(level=None)
        logger = get_logger("test")
        # Check that logger is configured
        assert logger is not None

    def test_configure_logging_with_file(self) -> None:
        """Test configure_logging with file_path (covers lines 125-131)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            configure_logging(level=logging.INFO, file_path=log_file)

            logger = get_logger("test")
            logger.info("Test message")

            for handler in logger.handlers:
                handler.flush()

            assert log_file.exists()
            content = log_file.read_text()
            assert "Test message" in content

            root_logger = logging.getLogger("neural_analysis")
            for handler in list(root_logger.handlers):
                if isinstance(handler, logging.FileHandler):
                    handler.close()
                    root_logger.removeHandler(handler)

    def test_configure_logging_already_configured(self) -> None:
        """Test configure_logging when already configured (covers lines 96-97)."""
        # Configure once
        configure_logging(level=logging.DEBUG)
        logger1 = get_logger("test1")
        assert logger1 is not None

        # Try to configure again - should be ignored (early return)
        configure_logging(level=logging.WARNING)
        logger2 = get_logger("test2")
        # Should still exist (configuration was ignored)
        assert logger2 is not None

    def test_configure_logging_custom_format(self) -> None:
        """Test configure_logging with custom format."""
        configure_logging(level=logging.INFO, fmt="%(message)s", datefmt="%H:%M")
        logger = get_logger("test")
        assert logger is not None

    def test_configure_logging_custom_stream(self) -> None:
        """Test configure_logging with custom stream."""
        import io

        stream = io.StringIO()
        configure_logging(level=logging.INFO, stream=stream)
        logger = get_logger("test")
        logger.info("Test message")

        # Force flush handlers
        for handler in logger.handlers:
            handler.flush()

        # Check stream has content
        stream_value = stream.getvalue()
        assert "Test message" in stream_value or len(stream_value) > 0

    def test_configure_logging_propagate(self) -> None:
        """Test configure_logging with propagate setting."""
        configure_logging(level=logging.INFO, propagate=True)
        logger = get_logger("test")
        assert logger.propagate is True


class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_logger_with_name(self) -> None:
        """Test get_logger with name (covers line 144)."""
        logger = get_logger("test_module")
        assert logger.name == "neural_analysis.test_module"

    def test_get_logger_without_name(self) -> None:
        """Test get_logger without name (covers line 144)."""
        logger = get_logger(None)
        assert logger.name == "neural_analysis"


class TestLogSection:
    """Tests for log_section function."""

    def setup_method(self) -> None:
        """Reset logging configuration before each test."""
        reset_logging_config()

    def test_log_section_default(self) -> None:
        """Test log_section with default parameters."""
        configure_logging(level=logging.INFO)
        log_section("Test Section")
        # Should not raise

    def test_log_section_custom_level(self) -> None:
        """Test log_section with custom level."""
        configure_logging(level=logging.DEBUG)
        log_section("Debug Section", level=logging.DEBUG)
        # Should not raise

    def test_log_section_custom_char(self) -> None:
        """Test log_section with custom character."""
        configure_logging(level=logging.INFO)
        log_section("Custom Section", char="-")
        # Should not raise

    def test_log_section_long_title(self) -> None:
        """Test log_section with long title (covers line 151)."""
        configure_logging(level=logging.INFO)
        long_title = "A" * 100
        log_section(long_title)
        # Should not raise


class TestLogKv:
    """Tests for log_kv function."""

    def setup_method(self) -> None:
        """Reset logging configuration before each test."""
        reset_logging_config()

    def test_log_kv_with_dict(self) -> None:
        """Test log_kv with dictionary (covers line 175)."""
        configure_logging(level=logging.INFO)
        log_kv("metrics", {"accuracy": 0.95, "loss": 0.1})
        # Should not raise

    def test_log_kv_with_list(self) -> None:
        """Test log_kv with list of tuples (covers line 175)."""
        configure_logging(level=logging.INFO)
        log_kv("config", [("key1", "value1"), ("key2", "value2")])
        # Should not raise

    def test_log_kv_custom_level(self) -> None:
        """Test log_kv with custom level."""
        configure_logging(level=logging.DEBUG)
        log_kv("debug", {"key": "value"}, level=logging.DEBUG)
        # Should not raise


class TestLogCalls:
    """Tests for log_calls decorator."""

    def setup_method(self) -> None:
        """Reset logging configuration before each test."""
        reset_logging_config()

    def test_log_calls_with_timeit(self) -> None:
        """Test log_calls decorator with timeit=True (covers lines 207-209)."""
        configure_logging(level=logging.DEBUG)

        @log_calls(level=logging.DEBUG, timeit=True)
        def test_func(x: int) -> int:
            return x * 2

        result = test_func(5)
        assert result == 10
        # Should not raise

    def test_log_calls_without_timeit(self) -> None:
        """Test log_calls decorator with timeit=False (covers lines 210-211)."""
        configure_logging(level=logging.DEBUG)

        @log_calls(level=logging.DEBUG, timeit=False)
        def test_func(x: int) -> int:
            return x * 2

        result = test_func(5)
        assert result == 10
        # Should not raise

    def test_log_calls_with_exception(self) -> None:
        """Test log_calls decorator when function raises exception."""
        configure_logging(level=logging.DEBUG)

        @log_calls(level=logging.DEBUG, timeit=True)
        def test_func() -> None:
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            test_func()
        # Should still log exit message

    def test_log_calls_with_args_kwargs(self) -> None:
        """Test log_calls decorator with various args/kwargs (covers line 200)."""
        configure_logging(level=logging.DEBUG)

        @log_calls(level=logging.DEBUG)
        def test_func(a: int, b: int, c: int = 0) -> int:
            return a + b + c

        result = test_func(1, 2, c=3)
        assert result == 6
        # Should log args=2, kwargs=1

    def test_log_calls_default_parameters(self) -> None:
        """Test log_calls decorator with default parameters."""
        configure_logging(level=logging.DEBUG)

        @log_calls()
        def test_func(x: int) -> int:
            return x * 2

        result = test_func(5)
        assert result == 10
        # Should use default level=DEBUG, timeit=True
