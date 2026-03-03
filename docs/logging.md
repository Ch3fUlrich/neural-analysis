# Logging Guidelines

Logging provides traceable, structured information about what analyses are doing. This document shows how to use the shared logging utilities.

---

## 1. Do / Don't

| Do                                          | Don't                         |
|--------------------------------------------|-------------------------------|
| Use `get_logger(__name__)` per module      | Use `print()` in library code |
| Use `log_kv` for structured metrics        | Log raw dicts without context |
| Use `log_section` for major phases         | Scatter unrelated log lines   |
| Configure logging once per script/notebook | Reconfigure loggers per call  |

---

## 2. Basic Logging in Code

```python
from neural_analysis.utils import configure_logging, get_logger

configure_logging(level="INFO")
log = get_logger(__name__)

def run_analysis(data):
    log.info("Starting analysis", extra={"n_samples": len(data)})
    ...
    log.info("Finished analysis")
```

Key points:

- Call `configure_logging` once (usually in `if __name__ == "__main__":` or at notebook setup).
- Use a module-level logger.

---

## 3. Structured Metrics Logging

Use `log_kv` to emit key-value pairs in a consistent format.

```python
from neural_analysis.utils import log_kv

log_kv("metrics", {"accuracy": 0.93, "n_trials": 120})
```

Typical uses:

- Final performance metrics.
- Per-epoch or per-iteration summaries.

---

## 4. Flow Logging

Use `log_section` and decorators to mark phases and trace function calls.

```python
from neural_analysis.utils import log_section, log_calls
import logging

@log_calls(level=logging.DEBUG)
def compute_scores(...):
    ...

log_section("Loading data")
# load data
log_section("Computing scores")
compute_scores(...)
```

Benefits:

- Clear high-level phases in logs.
- Debug-level call traces when needed.

---

## 5. Multi-File Session Logging

For production or long-running analyses, use **multi-file mode** to create
session directories with separate log files by severity.

### 5.1 Setup

```python
from neural_analysis.utils import configure_logging

session_dir = configure_logging(
    level="DEBUG",
    log_root="logs",           # Parent directory for sessions
    capture_warnings=True,     # Route Python warnings into logs
    capture_print=True,        # Tee print() output to info log
)
print(f"Logs at: {session_dir}")  # e.g., logs/2025-06-15_14-30-00/
```

This creates a timestamped session directory with 4 rotating log files:

| File | Level | Purpose |
|------|-------|---------|
| `all.log` | DEBUG | Everything (full trace) |
| `info.log` | INFO | Normal operations |
| `warnings.log` | WARNING | Warnings only |
| `errors.log` | ERROR | Errors only |

Each file uses `RotatingFileHandler` (default: 10 MB, 5 backups).

### 5.2 LogConfig Dataclass

Override defaults by creating a `LogConfig`:

```python
from neural_analysis.utils.logging import LogConfig

cfg = LogConfig(
    level=logging.DEBUG,
    console_level=logging.WARNING,    # Quiet console, verbose files
    max_bytes_per_file=50_000_000,    # 50 MB per file
    backup_count=10,
)
```

### 5.3 LogFileReference

Use `LogFileReference` in error messages to point users at the right log:

```python
from neural_analysis.utils.logging import LogFileReference

try:
    run_analysis(data)
except Exception:
    log.error(
        "Analysis failed. See %s for details.",
        LogFileReference.error_log(),
    )
    raise
```

Available methods:
- `LogFileReference.error_log()`  Path to `errors.log`
- `LogFileReference.debug_log()`  Path to `all.log`
- `LogFileReference.session_dir()`  Path to session directory

### 5.4 Print Capture

When `capture_print=True`, `print()` calls are teed into the logger:

- `stdout`  `INFO` level
- `stderr`  `ERROR` level

The original stream still receives the output. This is useful for capturing
third-party library output that uses `print()`.

---

## 6. Environment Variable Override

Set `NEURAL_ANALYSIS_LOG_LEVEL` to override the default level:

```bash
NEURAL_ANALYSIS_LOG_LEVEL=DEBUG uv run python my_script.py
```

---

## 7. Recipes

### 7.1 Simple file logging

```python
configure_logging(level="INFO", file_path="logs/run.log")
```

### 7.2 Multi-file with custom session name

```python
configure_logging(
    log_root="logs",
    session_id="experiment_42",
)
# Creates logs/experiment_42/{all,info,warnings,errors}.log
```

### 7.3 Verbose debugging for a single run

```python
configure_logging(level="DEBUG")
```

### 7.4 Quiet console, full file logs

```python
configure_logging(
    level="DEBUG",
    console_level="WARNING",
    log_root="logs",
)
```
