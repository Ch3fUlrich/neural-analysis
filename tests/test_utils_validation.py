"""Tests for validation utilities."""

from __future__ import annotations

import logging

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

    def test_logs_critical(self, caplog) -> None:
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

    def test_multiline_message(self, caplog) -> None:
        """Test with multiline error message."""
        message = "Line 1\nLine 2\nLine 3"
        with caplog.at_level(logging.CRITICAL):
            with pytest.raises(RuntimeError, match="Line 1"):
                do_critical(RuntimeError, message)

        # Verify full message was logged
        assert any(message in record.message for record in caplog.records)
