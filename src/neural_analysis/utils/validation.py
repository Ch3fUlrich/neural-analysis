"""Validation and error handling utilities for neural_analysis.

This module provides small helpers to standardize error reporting and
critical checks throughout the codebase.
"""

from __future__ import annotations

import logging
from typing import Type

__all__ = ["do_critical"]


def do_critical(exc: Type[BaseException], message: str) -> None:
    """Log a critical error and raise the provided exception type.

    Parameters
    ----------
    exc : Exception type
        The exception class to raise, e.g., ValueError.
    message : str
        The message to log and raise with the exception.
    """
    logger = logging.getLogger("neural_analysis")
    logger.critical(message)
    raise exc(message)
