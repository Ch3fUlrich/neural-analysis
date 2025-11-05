# Logging Guidelines

Robust logging is critical to understand, reproduce, and debug computations later. This project uses Python's standard `logging` module with lightweight helpers in `neural_analysis.utils.logging`.

## Quick start

```python
from neural_analysis.utils import configure_logging, get_logger, log_kv, log_section

# Configure once at app/notebook startup
configure_logging(level="INFO")  # or pass file_path="logs/run.log"

log = get_logger(__name__)
log_section("Loading data")
log.info("Reading dataset",)
log_kv("config", {"subject": "S01", "session": 3})
```

## Best practices

- Do not call `basicConfig()` in library modules; call `configure_logging()` from the entry script or notebook.
- Use `get_logger(__name__)` per-module to get a namespaced logger under `neural_analysis.*`.
- Prefer structured messages with `log_kv()` for key metrics or configuration snapshots.
- For function tracing, decorate with `@log_calls(level=logging.DEBUG)` to log entry/exit and runtimes.
- Keep messages concise; rely on levels (`DEBUG`, `INFO`, `WARNING`, `ERROR`).

## Environment variables

- `NEURAL_ANALYSIS_LOG_LEVEL` (e.g., `DEBUG`): default level if not passed explicitly.

## API reference

- `configure_logging(level="INFO", file_path=None, fmt=None, datefmt=None, stream=None, propagate=False)`
- `get_logger(name=None)` → `Logger`
- `log_section(title, level=INFO, char="=")`
- `log_kv(prefix, mapping, level=INFO)`
- `log_calls(level=DEBUG, timeit=True)` decorator to trace functions

## Example: file + console logging

```python
from neural_analysis.utils import configure_logging, get_logger
configure_logging(level="INFO", file_path="logs/experiment.log")
log = get_logger("experiment")
log.info("Experiment started")
```

## Migration notes

Legacy code in `/todo` used ad-hoc `global_logger` patterns. The new centralized utilities replace those while remaining dependency-free and configurable. Convert `print()` to `log.info()` and adopt `log_kv()` for metrics to standardize outputs.
