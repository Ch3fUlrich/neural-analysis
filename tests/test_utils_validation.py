"""Tests for validation utilities."""

from __future__ import annotations

import logging
from typing import Any

import pytest

from neural_analysis.utils import do_critical


class TestDoCritical:
    """Test suite for do_critical function."""

    def test_raises_value_error(self) -> None:
        """Test that do_critical raises ValueError with correct message."""
        message = "This is a critical error"
        with pytest.raises(ValueError, match=message):
            do_critical(ValueError, message)

    def test_raises_type_error(self) -> None:
        """Test that do_critical raises TypeError with correct message."""
        message = "Type mismatch detected"
        with pytest.raises(TypeError, match=message):
            do_critical(TypeError, message)

    def test_raises_runtime_error(self) -> None:
        """Test that do_critical raises RuntimeError with correct message."""
        message = "Runtime failure"
        with pytest.raises(RuntimeError, match=message):
            do_critical(RuntimeError, message)

    def test_logs_critical(self, caplog: Any) -> None:
        """Test that do_critical logs at CRITICAL level."""
        message = "Critical issue logged"
        with caplog.at_level(logging.CRITICAL), pytest.raises(ValueError):
            do_critical(ValueError, message)

        # Check that message was logged at CRITICAL level
        assert any(
            record.levelname == "CRITICAL" and message in record.message
            for record in caplog.records
        )

    def test_custom_exception(self) -> None:
        """Test with custom exception class."""

        class CustomError(Exception):
            pass

        message = "Custom error occurred"
        with pytest.raises(CustomError, match=message):
            do_critical(CustomError, message)

    def test_empty_message(self) -> None:
        """Test with empty message string."""
        with pytest.raises(ValueError, match="^$"):
            do_critical(ValueError, "")

    def test_multiline_message(self, caplog: Any) -> None:
        """Test with multiline error message."""
        message = "Line 1\nLine 2\nLine 3"
        with (
            caplog.at_level(logging.CRITICAL),
            pytest.raises(RuntimeError, match="Line 1"),
        ):
            do_critical(RuntimeError, message)

        # Verify full message was logged
        assert any(message in record.message for record in caplog.records)

    def test_import_fallback(self, monkeypatch):
        """Test ImportError fallback for get_logger (covers lines 13-16)."""
        import sys
        import importlib

        # Save original module references
        original_validation = sys.modules.get("neural_analysis.utils.validation")
        original_logging = sys.modules.get("neural_analysis.utils.logging")

        # Remove both modules from cache to force re-import
        if "neural_analysis.utils.validation" in sys.modules:
            del sys.modules["neural_analysis.utils.validation"]
        if "neural_analysis.utils.logging" in sys.modules:
            del sys.modules["neural_analysis.utils.logging"]

        # Temporarily remove the logging module from the path to simulate ImportError
        import neural_analysis.utils

        original_hasattr = hasattr(neural_analysis.utils, "logging")
        if original_hasattr:
            # Temporarily remove the logging attribute
            original_logging_attr = getattr(neural_analysis.utils, "logging", None)
            delattr(neural_analysis.utils, "logging")

        # Mock __import__ to raise ImportError for the logging module
        original_import = __import__

        def mock_import(name, globals=None, locals=None, fromlist=(), level=0):
            # Check if this is an import of neural_analysis.utils.logging
            if name == "neural_analysis.utils.logging" or (
                fromlist and "logging" in fromlist and name == "neural_analysis.utils"
            ):
                raise ImportError("Mocked import error for logging module")
            return original_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr("builtins.__import__", mock_import)

        # Re-import the module to trigger the fallback (covers lines 13-16)
        importlib.invalidate_caches()
        import neural_analysis.utils.validation as validation_module

        # Verify the fallback logger works (lines 15-16)
        logger = validation_module.get_logger("test_module")
        assert logger is not None
        assert isinstance(logger, logging.Logger)
        # The logger name should be "test_module" (or end with it)
        assert logger.name == "test_module" or logger.name.endswith("test_module")

        # Restore original modules
        if original_validation:
            sys.modules["neural_analysis.utils.validation"] = original_validation
        if original_logging:
            sys.modules["neural_analysis.utils.logging"] = original_logging
        if original_hasattr and original_logging_attr:
            setattr(neural_analysis.utils, "logging", original_logging_attr)
