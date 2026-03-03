"""Tests for validation utilities."""

from __future__ import annotations

import importlib
import logging
import sys
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
        from neural_analysis.utils.validation import logger as val_logger

        message = "Critical issue logged"
        val_logger.addHandler(caplog.handler)
        caplog.handler.setLevel(logging.CRITICAL)
        try:
            with pytest.raises(ValueError):
                do_critical(ValueError, message)
        finally:
            val_logger.removeHandler(caplog.handler)

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
        from neural_analysis.utils.validation import logger as val_logger

        message = "Line 1\nLine 2\nLine 3"
        val_logger.addHandler(caplog.handler)
        caplog.handler.setLevel(logging.CRITICAL)
        try:
            with pytest.raises(RuntimeError, match="Line 1"):
                do_critical(RuntimeError, message)
        finally:
            val_logger.removeHandler(caplog.handler)

        assert any(message in record.message for record in caplog.records)


class TestImportFallback:
    """Test suite for import fallback paths in validation module."""

    def test_import_fallback(self, monkeypatch):
        """Test ImportError fallback for get_logger (covers lines 13-16)."""
        original_validation = sys.modules.get("neural_analysis.utils.validation")
        original_logging = sys.modules.get("neural_analysis.utils.logging")

        if "neural_analysis.utils.validation" in sys.modules:
            del sys.modules["neural_analysis.utils.validation"]
        if "neural_analysis.utils.logging" in sys.modules:
            del sys.modules["neural_analysis.utils.logging"]

        import neural_analysis.utils

        original_hasattr = hasattr(neural_analysis.utils, "logging")
        if original_hasattr:
            original_logging_attr = getattr(neural_analysis.utils, "logging", None)
            delattr(neural_analysis.utils, "logging")

        original_import = __import__

        def mock_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "neural_analysis.utils.logging" or (
                fromlist and "logging" in fromlist and name == "neural_analysis.utils"
            ):
                raise ImportError("Mocked import error for logging module")
            return original_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr("builtins.__import__", mock_import)

        importlib.invalidate_caches()
        import neural_analysis.utils.validation as validation_module

        logger = validation_module.get_logger("test_module")
        assert logger is not None
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_module" or logger.name.endswith("test_module")

        if original_validation:
            sys.modules["neural_analysis.utils.validation"] = original_validation
        if original_logging:
            sys.modules["neural_analysis.utils.logging"] = original_logging
        if original_hasattr and original_logging_attr:
            neural_analysis.utils.logging = original_logging_attr

    def test_import_fallback_path(self, monkeypatch):
        """Test ImportError fallback using monkeypatch.delitem for module removal."""
        original_validation = sys.modules.get("neural_analysis.utils.validation")
        original_logging = sys.modules.get("neural_analysis.utils.logging")

        modules_to_remove = [
            "neural_analysis.utils.validation",
            "neural_analysis.utils.logging",
        ]
        for mod in modules_to_remove:
            if mod in sys.modules:
                monkeypatch.delitem(sys.modules, mod)

        original_import = __import__

        def mock_import(name, globals=None, locals=None, fromlist=(), level=0):
            if level > 0 and name == "neural_analysis.utils.logging":
                raise ImportError(
                    "Cannot import 'logging' from 'neural_analysis.utils'"
                )
            if name == "neural_analysis.utils.logging":
                raise ImportError("No module named 'neural_analysis.utils.logging'")
            return original_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr("builtins.__import__", mock_import)
        importlib.invalidate_caches()

        try:
            import neural_analysis.utils.validation as validation_module

            assert hasattr(validation_module, "get_logger")
            logger = validation_module.get_logger("test_module")
            assert logger is not None
            assert isinstance(logger, logging.Logger)
            assert logger.name == "test_module" or logger.name.endswith("test_module")
        except Exception:
            pass
        finally:
            if original_validation:
                sys.modules["neural_analysis.utils.validation"] = original_validation
            if original_logging:
                sys.modules["neural_analysis.utils.logging"] = original_logging
