"""Validation and error handling utilities for neural_analysis.

This module provides small helpers to standardize error reporting and
critical checks throughout the codebase.
"""

from __future__ import annotations

import logging

try:
    from .logging import get_logger
except ImportError:
    def get_logger(name: str):  # type: ignore
        return logging.getLogger(name)

__all__ = ["do_critical"]

# Module logger
logger = get_logger(__name__)


def do_critical(exc: type[BaseException], message: str) -> None:
    """Log a critical error and raise the provided exception type.

    Parameters
    ----------
    exc : Exception type
        The exception class to raise, e.g., ValueError.
    message : str
        The message to log and raise with the exception.
    """
    logger.critical(message)
    raise exc(message)
