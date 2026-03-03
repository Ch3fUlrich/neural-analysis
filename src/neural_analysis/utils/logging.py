"""Project-wide logging utilities for neural_analysis.

This module centralizes logging configuration and helper utilities to reduce
print statements and provide consistent, informative logs across the project.

Supports two modes:

1. **Simple mode** (default): one console handler + optional single log file.
   Activated by calling ``configure_logging()`` or ``configure_logging(file_path=...)``
2. **Multi-file mode**: session-scoped directory with severity-split log files.
   Activated by calling ``configure_logging(log_root="logs")``

Usage (quick start):
    from neural_analysis.utils.logging import configure_logging, get_logger
    configure_logging(level="INFO")
    log = get_logger(__name__)
    log.info("Hello logging")

Multi-file usage:
    from neural_analysis.utils.logging import configure_logging, get_logger, LogFileReference
    log_dir = configure_logging(log_root="logs", capture_warnings=True)
    print(f"Logs -> {log_dir}")
    # Error messages can reference log files:
    # f"See: {LogFileReference.error_log()}"
"""

from __future__ import annotations

import logging
import os
import sys
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

__all__ = [
    "LogFileReference",
    "configure_logging",
    "get_log_dir",
    "get_logger",
    "log_calls",
    "log_kv",
    "log_section",
]


_CONFIGURED = False
_LOGGER_NAME = "neural_analysis"
_LOG_DIR: Path | None = None
_SESSION_ID: str = ""


@dataclass
class LogConfig:
    level: int = logging.INFO
    fmt: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt: str = "%Y-%m-%d %H:%M:%S"
    propagate: bool = False
    stream: Any = sys.stdout
    file_path: Path | None = None
    console_level: int = logging.INFO
    max_bytes_per_file: int = 10 * 1024 * 1024  # 10 MB
    backup_count: int = 5


class LogFileReference:
    """Helper to generate log file references for error messages."""

    @staticmethod
    def error_log() -> str:
        if _LOG_DIR is None:
            return "(logging not configured — call configure_logging(log_root=...))"
        return str(_LOG_DIR / "errors.log")

    @staticmethod
    def debug_log() -> str:
        if _LOG_DIR is None:
            return "(logging not configured — call configure_logging(log_root=...))"
        return str(_LOG_DIR / "all.log")

    @staticmethod
    def session_dir() -> str:
        if _LOG_DIR is None:
            return "(logging not configured)"
        return str(_LOG_DIR)


def get_log_dir() -> Path | None:
    """Return the current session log directory, or ``None`` if not configured."""
    return _LOG_DIR


def _make_session_id() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S")


def _level_from_env(default: int) -> int:
    env = os.getenv("NEURAL_ANALYSIS_LOG_LEVEL")
    if not env:
        return default
    try:
        val = getattr(logging, env.upper())
        if isinstance(val, int):
            return val
        else:
            return default
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
    log_root: str | Path | None = None,
    session_id: str | None = None,
    capture_warnings: bool = False,
    capture_print: bool = False,
    console_level: int | str | None = None,
    max_bytes_per_file: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> Path | None:
    """Configure project-wide logging for the "neural_analysis" logger.

    Two modes:

    * **Simple mode** — ``configure_logging()`` or
      ``configure_logging(file_path="app.log")``
      One console handler plus an optional single log file.

    * **Multi-file mode** — ``configure_logging(log_root="logs")``.
      Creates a session sub-directory with ``all.log``, ``info.log``,
      ``warnings.log``, and ``errors.log``, each with rotating file
      handlers.

    Returns the session log directory (multi-file mode) or ``None``
    (simple mode).
    """
    global _CONFIGURED, _LOG_DIR, _SESSION_ID
    if _CONFIGURED:
        return _LOG_DIR

    cfg = LogConfig()

    # Resolve level
    if isinstance(level, str):
        level_val = getattr(logging, level.upper(), logging.INFO)
    elif isinstance(level, int):
        level_val = level
    else:
        level_val = _level_from_env(cfg.level)

    # Resolve console level
    if isinstance(console_level, str):
        console_level_val: int = getattr(logging, console_level.upper(), logging.INFO)
    elif isinstance(console_level, int):
        console_level_val = console_level
    else:
        console_level_val = cfg.console_level

    fmt_val = fmt or cfg.fmt
    datefmt_val = datefmt or cfg.datefmt
    stream_val = stream or cfg.stream
    propagate_val = propagate if propagate is not None else cfg.propagate

    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(level_val)
    logger.propagate = propagate_val
    logger.handlers.clear()

    formatter = logging.Formatter(fmt_val, datefmt=datefmt_val)

    # Console handler
    console = logging.StreamHandler(stream_val)
    console.setLevel(console_level_val)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # --- Multi-file mode ---
    if log_root is not None:
        _SESSION_ID = session_id or _make_session_id()
        _LOG_DIR = Path(log_root) / _SESSION_ID
        _LOG_DIR.mkdir(parents=True, exist_ok=True)

        file_configs = [
            ("all.log", logging.DEBUG),
            ("info.log", logging.INFO),
            ("warnings.log", logging.WARNING),
            ("errors.log", logging.ERROR),
        ]
        for filename, file_level in file_configs:
            handler = RotatingFileHandler(
                _LOG_DIR / filename,
                maxBytes=max_bytes_per_file,
                backupCount=backup_count,
                encoding="utf-8",
            )
            handler.setLevel(file_level)
            handler.setFormatter(formatter)
            logger.addHandler(handler)

    # --- Simple file mode (legacy) ---
    elif file_path is not None:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setLevel(level_val)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Capture Python warnings into logging
    if capture_warnings:
        logging.captureWarnings(True)
        warnings_logger = logging.getLogger("py.warnings")
        warnings_logger.handlers = logger.handlers.copy()

    # Optional: capture print() to info log
    if capture_print:
        sys.stdout = _PrintCapture(logger, logging.INFO, sys.stdout)
        sys.stderr = _PrintCapture(logger, logging.ERROR, sys.stderr)

    _CONFIGURED = True
    if _LOG_DIR is not None:
        logger.info("Logging session started: %s | Logs: %s", _SESSION_ID, _LOG_DIR)
    return _LOG_DIR


class _PrintCapture:
    """Stream wrapper that tees ``print()`` output to a logger."""

    def __init__(
        self, logger: logging.Logger, level_val: int, original_stream: Any
    ) -> None:
        self._logger = logger
        self._level = level_val
        self._original = original_stream

    def write(self, msg: str) -> int:
        if msg and msg.strip():
            self._logger.log(self._level, msg.rstrip())
        return self._original.write(msg)  # type: ignore[no-any-return]

    def flush(self) -> None:
        self._original.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


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

    from functools import wraps

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        log = get_logger(func.__module__)

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
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

        return wrapper

    return decorator
