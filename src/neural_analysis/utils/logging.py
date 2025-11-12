"""Project-wide logging utilities for neural_analysis.

This module centralizes logging configuration and helper utilities to reduce
print statements and provide consistent, informative logs across the project.

Usage (quick start):
    from neural_analysis.utils.logging import configure_logging, get_logger
    configure_logging(level="INFO")
    log = get_logger(__name__)
    log.info("Hello logging")

Best practices:
- Do not configure the global logging in library imports. Call
  ``configure_logging`` from your app, notebook, or tests.
- Use ``get_logger(__name__)`` inside modules to get a namespaced logger.
- Prefer structured key=value messages for important metrics.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "configure_logging",
    "get_logger",
    "log_section",
    "log_kv",
    "log_calls",
]


_CONFIGURED = False
_LOGGER_NAME = "neural_analysis"


@dataclass
class LogConfig:
    level: int = logging.INFO
    fmt: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    datefmt: str = "%Y-%m-%d %H:%M:%S"
    propagate: bool = False
    stream: Any = sys.stdout
    file_path: Path | None = None


def _level_from_env(default: int) -> int:
    env = os.getenv("NEURAL_ANALYSIS_LOG_LEVEL")
    if not env:
        return default
    try:
        return getattr(logging, env.upper())
    except Exception:
        return default


def configure_logging(
    *,
    level: int | str | None = None,
    fmt: str | None = None,
    datefmt: str | None = None,
    stream: Any | None = None,
    file_path: str | Path | None = None,
    propagate: bool | None = None,
) -> None:
    """Configure project-wide logging for the "neural_analysis" logger.

    Parameters
    ----------
    level : int | str | None
        Log level (e.g., logging.INFO or "INFO"). If None, uses env var
        NEURAL_ANALYSIS_LOG_LEVEL or INFO.
    fmt : str | None
        Log message format string.
    datefmt : str | None
        Datetime format string.
    stream : IO | None
        Stream handler target (default stdout).
    file_path : str | Path | None
        Optional path to a log file to also write logs.
    propagate : bool | None
        Whether child loggers propagate to root. Default False to avoid
        duplicate messages when used in notebooks.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    cfg = LogConfig()
    # Resolve level
    if isinstance(level, str):
        level_val = getattr(logging, level.upper(), logging.INFO)
    elif isinstance(level, int):
        level_val = level
    else:
        level_val = _level_from_env(cfg.level)

    fmt_val = fmt or cfg.fmt
    datefmt_val = datefmt or cfg.datefmt
    stream_val = stream or cfg.stream
    propagate_val = propagate if propagate is not None else cfg.propagate

    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(level_val)
    logger.propagate = propagate_val

    # Clear existing handlers only on our named logger
    logger.handlers.clear()

    stream_handler = logging.StreamHandler(stream_val)
    stream_handler.setLevel(level_val)
    stream_handler.setFormatter(logging.Formatter(fmt_val, datefmt=datefmt_val))
    logger.addHandler(stream_handler)

    if file_path is not None:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setLevel(level_val)
        file_handler.setFormatter(logging.Formatter(fmt_val, datefmt=datefmt_val))
        logger.addHandler(file_handler)

    _CONFIGURED = True


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a namespaced logger under the project logger.

    Examples
    --------
    >>> log = get_logger(__name__)
    >>> log.info("message")
    """
    base = _LOGGER_NAME if name is None else f"{_LOGGER_NAME}.{name}"
    return logging.getLogger(base)


def log_section(title: str, *, level: int = logging.INFO, char: str = "=") -> None:
    """Log a visual section separator with a title."""
    log = get_logger("section")
    line = char * max(60, len(title) + 10)
    log.log(level, line)
    log.log(level, f" {title} ")
    log.log(level, line)


def log_kv(
    prefix: str,
    mapping: Mapping[str, Any] | Iterable[tuple[str, Any]],
    *,
    level: int = logging.INFO,
) -> None:
    """Log key=value pairs in a compact, consistent style.

    Parameters
    ----------
    prefix : str
        A short message prefix, e.g., "metrics" or "config".
    mapping : Mapping or iterable of (key, value)
        Data to render as key=value pairs.
    level : int, default INFO
        Log level to use.
    """
    log = get_logger("kv")
    items = mapping.items() if isinstance(mapping, Mapping) else mapping
    msg = prefix + ": " + ", ".join(f"{k}={v!r}" for k, v in items)
    log.log(level, msg)


def log_calls(
    *, level: int = logging.DEBUG, timeit: bool = True
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to log function entry/exit (and runtime).

    Examples
    --------
    >>> @log_calls(level=logging.INFO)
    ... def my_fn(x):
    ...     return x * 2
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        log = get_logger(func.__module__)

        def wrapper(*args: Any, **kwargs):
            log.log(
                level,
                f"→ {func.__name__}(args=%d, kwargs=%d)" % (len(args), len(kwargs)),
            )
            t0 = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                if timeit:
                    dt = (time.time() - t0) * 1000.0
                    log.log(level, f"← {func.__name__} completed in {dt:.2f} ms")
                else:
                    log.log(level, f"← {func.__name__} done")

        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper.__qualname__ = func.__qualname__
        return wrapper

    return decorator
