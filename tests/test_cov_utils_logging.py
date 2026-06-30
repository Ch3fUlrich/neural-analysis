"""Additional tests for neural_analysis.utils.logging to raise coverage to >= 95%.

Covers:
- LogFileReference.error_log / debug_log / session_dir (with and without _LOG_DIR set)
- get_log_dir()
- _make_session_id()
- configure_logging() with console_level as str / int, log_root (multi-file mode),
  capture_warnings, capture_print, session_id override
- _PrintCapture.write (non-empty, empty/whitespace), flush, __getattr__
"""

from __future__ import annotations

import io
import logging
import sys
from pathlib import Path

import pytest

import neural_analysis.utils.logging as logging_mod
from neural_analysis.utils.logging import (
    LogFileReference,
    configure_logging,
    get_log_dir,
    get_logger,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reset() -> None:
    """Reset the global state so configure_logging can be called fresh."""
    logging_mod._CONFIGURED = False
    logging_mod._LOG_DIR = None
    logging_mod._SESSION_ID = ""
    logger = logging.getLogger("neural_analysis")
    for h in list(logger.handlers):
        h.close()
    logger.handlers.clear()


@pytest.fixture(autouse=True)
def reset_state():
    """Auto-reset before and after every test in this file."""
    _reset()
    yield
    _reset()
    # Restore stdout/stderr in case capture_print was used
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__


# ---------------------------------------------------------------------------
# LogFileReference – _LOG_DIR is None (uncovered: lines 75-77, 81-83, 87-89)
# ---------------------------------------------------------------------------


class TestLogFileReferenceNone:
    """LogFileReference when _LOG_DIR is None (logging not configured)."""

    def test_error_log_no_dir(self) -> None:
        assert logging_mod._LOG_DIR is None
        result = LogFileReference.error_log()
        assert "not configured" in result
        assert "configure_logging" in result

    def test_debug_log_no_dir(self) -> None:
        assert logging_mod._LOG_DIR is None
        result = LogFileReference.debug_log()
        assert "not configured" in result

    def test_session_dir_no_dir(self) -> None:
        assert logging_mod._LOG_DIR is None
        result = LogFileReference.session_dir()
        assert "not configured" in result


# ---------------------------------------------------------------------------
# LogFileReference – _LOG_DIR is set (uncovered: lines 77, 83, 89)
# ---------------------------------------------------------------------------


class TestLogFileReferenceSet:
    """LogFileReference when _LOG_DIR points to a real directory."""

    def test_error_log_with_dir(self, tmp_path: Path) -> None:
        logging_mod._LOG_DIR = tmp_path
        result = LogFileReference.error_log()
        assert result.endswith("errors.log")
        assert str(tmp_path) in result

    def test_debug_log_with_dir(self, tmp_path: Path) -> None:
        logging_mod._LOG_DIR = tmp_path
        result = LogFileReference.debug_log()
        assert result.endswith("all.log")
        assert str(tmp_path) in result

    def test_session_dir_with_dir(self, tmp_path: Path) -> None:
        logging_mod._LOG_DIR = tmp_path
        result = LogFileReference.session_dir()
        assert result == str(tmp_path)


# ---------------------------------------------------------------------------
# get_log_dir (uncovered: line 94)
# ---------------------------------------------------------------------------


class TestGetLogDir:
    def test_returns_none_when_not_configured(self) -> None:
        assert get_log_dir() is None

    def test_returns_path_after_multi_file_configure(self, tmp_path: Path) -> None:
        result = configure_logging(log_root=str(tmp_path), session_id="test-session")
        assert result is not None
        assert get_log_dir() == result
        assert get_log_dir().is_dir()  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# _make_session_id (uncovered: line 98)
# ---------------------------------------------------------------------------


class TestMakeSessionId:
    def test_session_id_format(self) -> None:
        sid = logging_mod._make_session_id()
        # Expect format: YYYY-MM-DD_HH-MM-SS
        import re
        assert re.match(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", sid), repr(sid)

    def test_session_id_is_string(self) -> None:
        assert isinstance(logging_mod._make_session_id(), str)


# ---------------------------------------------------------------------------
# configure_logging – console_level as str and as int (uncovered: lines 163, 165)
# ---------------------------------------------------------------------------


class TestConfigureLoggingConsoleLevelVariants:
    def test_console_level_as_string(self) -> None:
        stream = io.StringIO()
        configure_logging(level=logging.DEBUG, stream=stream, console_level="WARNING")
        logger = logging.getLogger("neural_analysis")
        # The console handler should have been set to WARNING
        assert any(h.level == logging.WARNING for h in logger.handlers), logger.handlers

    def test_console_level_as_int(self) -> None:
        stream = io.StringIO()
        configure_logging(level=logging.DEBUG, stream=stream, console_level=logging.ERROR)
        logger = logging.getLogger("neural_analysis")
        assert any(h.level == logging.ERROR for h in logger.handlers), logger.handlers


# ---------------------------------------------------------------------------
# configure_logging – multi-file mode (uncovered: lines 189-208, 232)
# ---------------------------------------------------------------------------


class TestConfigureLoggingMultiFile:
    def test_log_root_creates_session_dir(self, tmp_path: Path) -> None:
        session_dir = configure_logging(
            log_root=str(tmp_path), session_id="my-session"
        )
        assert session_dir is not None
        assert session_dir.is_dir()
        assert session_dir.name == "my-session"

    def test_log_root_creates_log_files(self, tmp_path: Path) -> None:
        session_dir = configure_logging(log_root=str(tmp_path), session_id="s1")
        assert session_dir is not None
        for fname in ("all.log", "info.log", "warnings.log", "errors.log"):
            assert (session_dir / fname).exists(), f"{fname} missing"

    def test_log_root_auto_session_id(self, tmp_path: Path) -> None:
        """session_id defaults to a timestamp string when not provided."""
        session_dir = configure_logging(log_root=str(tmp_path))
        assert session_dir is not None
        # auto session_id should match YYYY-MM-DD_HH-MM-SS pattern
        import re
        assert re.match(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", session_dir.name), (
            repr(session_dir.name)
        )

    def test_log_root_writes_messages_to_files(self, tmp_path: Path) -> None:
        session_dir = configure_logging(
            log_root=str(tmp_path), session_id="s2", level=logging.DEBUG
        )
        assert session_dir is not None
        log = get_logger("test_multi")
        log.error("unique-error-sentinel")
        log.warning("unique-warning-sentinel")
        log.info("unique-info-sentinel")
        log.debug("unique-debug-sentinel")
        # Flush all handlers
        for h in logging.getLogger("neural_analysis").handlers:
            h.flush()
        all_log = (session_dir / "all.log").read_text(encoding="utf-8")
        assert "unique-debug-sentinel" in all_log
        assert "unique-info-sentinel" in all_log
        assert "unique-warning-sentinel" in all_log
        assert "unique-error-sentinel" in all_log
        errors_log = (session_dir / "errors.log").read_text(encoding="utf-8")
        assert "unique-error-sentinel" in errors_log
        # info sentinel should not appear in errors.log
        assert "unique-info-sentinel" not in errors_log

    def test_log_root_returns_log_dir_on_repeat_call(self, tmp_path: Path) -> None:
        first = configure_logging(log_root=str(tmp_path), session_id="s3")
        second = configure_logging(log_root=str(tmp_path), session_id="other")
        assert first == second  # second call is a no-op

    def test_log_root_session_started_message_in_log(self, tmp_path: Path) -> None:
        """Line 232 — the session-started log.info call."""
        session_dir = configure_logging(log_root=str(tmp_path), session_id="sess-msg")
        assert session_dir is not None
        for h in logging.getLogger("neural_analysis").handlers:
            h.flush()
        all_log = (session_dir / "all.log").read_text(encoding="utf-8")
        assert "Logging session started" in all_log
        assert "sess-msg" in all_log


# ---------------------------------------------------------------------------
# configure_logging – capture_warnings (uncovered: lines 221-223)
# ---------------------------------------------------------------------------


class TestConfigureLoggingCaptureWarnings:
    def test_capture_warnings_true(self) -> None:
        stream = io.StringIO()
        configure_logging(level=logging.DEBUG, stream=stream, capture_warnings=True)
        warnings_logger = logging.getLogger("py.warnings")
        # py.warnings logger must have at least one handler copied from the main logger
        assert len(warnings_logger.handlers) >= 1

    def test_capture_warnings_py_warnings_handlers_are_copies(self) -> None:
        """capture_warnings=True copies the main logger's handlers to py.warnings."""
        stream = io.StringIO()
        configure_logging(level=logging.DEBUG, stream=stream, capture_warnings=True)
        main_handlers = logging.getLogger("neural_analysis").handlers
        warn_handlers = logging.getLogger("py.warnings").handlers
        # py.warnings handlers should be copies — same objects (shallow copy)
        assert len(warn_handlers) == len(main_handlers)
        for wh, mh in zip(warn_handlers, main_handlers):
            assert wh is mh


# ---------------------------------------------------------------------------
# configure_logging – capture_print (uncovered: lines 227-228)
# ---------------------------------------------------------------------------


class TestConfigureLoggingCapturePrint:
    def test_capture_print_replaces_stdout(self) -> None:
        original_stdout = sys.stdout
        stream = io.StringIO()
        configure_logging(level=logging.DEBUG, stream=stream, capture_print=True)
        assert sys.stdout is not original_stdout
        assert isinstance(sys.stdout, logging_mod._PrintCapture)

    def test_capture_print_replaces_stderr(self) -> None:
        original_stderr = sys.stderr
        stream = io.StringIO()
        configure_logging(level=logging.DEBUG, stream=stream, capture_print=True)
        assert sys.stderr is not original_stderr
        assert isinstance(sys.stderr, logging_mod._PrintCapture)


# ---------------------------------------------------------------------------
# _PrintCapture (uncovered: lines 242-244, 247-249, 252, 255)
# ---------------------------------------------------------------------------


class TestPrintCapture:
    def _make_capture(self, level: int = logging.INFO):
        """Return a _PrintCapture wired to a StringIO backend."""
        backing = io.StringIO()
        logger = logging.getLogger("neural_analysis.test_capture")
        logger.setLevel(logging.DEBUG)
        capture = logging_mod._PrintCapture(logger, level, backing)
        return capture, backing, logger

    def test_init_attributes(self) -> None:
        """Lines 242-244: __init__ stores logger, level, original."""
        capture, backing, logger = self._make_capture(logging.WARNING)
        assert capture._logger is logger
        assert capture._level == logging.WARNING
        assert capture._original is backing

    def test_write_non_empty_msg(self) -> None:
        """Lines 247-249: write() calls logger.log for non-empty stripped msg."""
        capture, backing, logger = self._make_capture(logging.INFO)
        handler = logging.StreamHandler(io.StringIO())
        handler.setLevel(logging.DEBUG)
        logger.addHandler(handler)

        n = capture.write("hello from capture")
        assert n == len("hello from capture")
        # Ensure text went to the backing stream too
        assert "hello from capture" in backing.getvalue()

    def test_write_empty_msg_not_logged(self) -> None:
        """write() with empty/whitespace-only string must NOT call logger.log."""
        capture, backing, logger = self._make_capture(logging.INFO)
        logged: list[str] = []

        class _Handler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                logged.append(record.getMessage())

        logger.addHandler(_Handler())
        capture.write("")  # empty
        capture.write("   \n")  # whitespace only
        assert len(logged) == 0, f"Unexpected log calls: {logged}"

    def test_write_whitespace_only_passes_through(self) -> None:
        """Whitespace-only write still goes to the original stream."""
        capture, backing, _ = self._make_capture()
        n = capture.write("\n")
        assert "\n" in backing.getvalue()
        assert n == len("\n")

    def test_flush_delegates_to_original(self) -> None:
        """Line 252: flush() calls self._original.flush()."""
        flushed: list[bool] = []

        class _FakeStream(io.StringIO):
            def flush(self) -> None:  # type: ignore[override]
                flushed.append(True)
                super().flush()

        fake = _FakeStream()
        logger = logging.getLogger("neural_analysis.flush_test")
        capture = logging_mod._PrintCapture(logger, logging.INFO, fake)
        capture.flush()
        assert flushed == [True]

    def test_getattr_delegates_to_original(self) -> None:
        """Line 255: __getattr__ proxies unknown attributes to _original."""
        capture, backing, _ = self._make_capture()
        # StringIO has a `getvalue` method — access it through the capture
        result = capture.getvalue()
        assert result == backing.getvalue()

    def test_getattr_raises_for_missing(self) -> None:
        """__getattr__ raises AttributeError for attributes that don't exist."""
        capture, _, _ = self._make_capture()
        with pytest.raises(AttributeError):
            _ = capture.this_attribute_does_not_exist_anywhere  # type: ignore[attr-defined]
