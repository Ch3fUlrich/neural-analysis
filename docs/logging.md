# `docs/logging.md` (restructured)

```markdown
# Logging Guidelines

Logging provides traceable, structured information about what analyses are doing. This document shows how to use the shared logging utilities.

---

## 1. Do / Don’t

| Do                                          | Don’t                         |
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
- Use a module‑level logger.

---

## 3. Structured Metrics Logging

Use `log_kv` to emit key‑value pairs in a consistent format.

```python
from neural_analysis.utils import log_kv

log_kv("metrics", {"accuracy": 0.93, "n_trials": 120})
```

Typical uses:

- Final performance metrics.
- Per‑epoch or per‑iteration summaries.

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

- Clear high‑level phases in logs.
- Debug‑level call traces when needed.

---

## 5. Recipes

### 5.1 File logging

```python
configure_logging(
    level="INFO",
    file_path="logs/run.log",
)
```

### 5.2 Verbose debugging for a single run

```python
configure_logging(level="DEBUG")
```

Use higher log level (`WARNING` or `ERROR`) for very quiet scripts, and `DEBUG` only when actively debugging.
