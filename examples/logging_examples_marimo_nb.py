import marimo


__generated_with = "0.18.3"

app = marimo.App(width="full")

@app.cell(hide_code=True)
def __():

    import marimo as mo

    return mo

@app.cell()
def _(mo):
    mo.md(r"""
    # Logging Examples for neural_analysis

    This notebook demonstrates the centralized logging utilities provided by `neural_analysis.utils.logging`. These tools help make analyses reproducible, debuggable, and easier to track.

    ## Key Features:
    - **configure_logging**: Set up logging level, format, file output
    - **get_logger**: Get namespaced loggers per module
    - **log_kv**: Structured key=value logging for metrics
    - **log_section**: Visual section separators
    - **log_calls**: Decorator for automatic function tracing
    """)
    return


@app.cell
def _():
    # Import logging utilities
    import sys
    import numpy as np
    from neural_analysis.utils import (
        configure_logging,
        get_logger,
        log_kv,
        log_section,
        log_calls,
    )

    # Configure logging once at the start
    # Level can be: DEBUG, INFO, WARNING, ERROR, CRITICAL
    configure_logging(level="INFO")
    print("✓ Logging configured at INFO level")
    return configure_logging, get_logger, log_calls, log_kv, log_section, np


@app.cell()
def _(mo):
    mo.md(r"""
    ## Example 1: Basic logging with get_logger

    Each module should get its own logger using `get_logger(__name__)` for proper namespacing.
    """)
    return


@app.cell
def _(get_logger):
    # Get a namespaced logger
    log = get_logger("example.basic")

    # Log at different levels
    log.debug("This debug message won't show (level=INFO)")
    log.info("Loading dataset...")
    log.warning("This is a warning")
    log.error("This is an error (but doesn't raise)")

    print("\n✓ Basic logging complete")
    return (log,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Example 2: Structured logging with log_kv

    Use `log_kv` to log key-value pairs in a consistent format. Great for hyperparameters, metrics, or configuration snapshots.
    """)
    return


@app.cell
def _(log_kv):
    # Log structured metrics
    log_kv("config", {
        "dataset": "neural_recordings_01",
        "n_neurons": 120,
        "n_trials": 500,
        "sampling_rate": 30000,
    })

    # Log performance metrics
    log_kv("metrics", {
        "accuracy": 0.934,
        "loss": 0.127,
        "f1_score": 0.891,
    })

    print("✓ Structured logging complete")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Example 3: Visual sections with log_section

    Use `log_section` to create clear separators between processing phases.
    """)
    return


@app.cell
def _(log, log_section):
    log_section("Phase 1: Data Loading")
    log.info("Reading raw neural data")
    log.info("Loaded 120 neurons × 500 trials")

    log_section("Phase 2: Preprocessing")
    log.info("Filtering signals")
    log.info("Normalizing firing rates")

    log_section("Phase 3: Analysis", char="-")
    log.info("Computing distance metrics")
    log.info("Clustering neurons by similarity")

    print("\n✓ Section logging complete")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Example 4: Function tracing with @log_calls decorator

    The `@log_calls` decorator automatically logs function entry/exit with timing information. Perfect for tracking which functions are called and how long they take.
    """)
    return


@app.cell
def _(log_calls, log_kv, log_section, np):
    import logging
    import time

    # Decorate functions to trace calls
    @log_calls(level=logging.INFO, timeit=True)
    def process_data(n_samples: int) -> np.ndarray:
        """Simulated data processing."""
        time.sleep(0.1)  # Simulate work
        return np.random.randn(n_samples, 10)

    @log_calls(level=logging.INFO)
    def compute_statistics(data: np.ndarray) -> dict:
        """Compute basic statistics."""
        return {
            "mean": float(data.mean()),
            "std": float(data.std()),
            "shape": data.shape,
        }

    # Run traced functions
    log_section("Running traced functions")
    data = process_data(100)
    stats = compute_statistics(data)

    log_kv("results", stats)
    print("\n✓ Function tracing complete")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Example 5: File logging

    You can write logs to a file by passing `file_path` to `configure_logging`. Logs will go to both console and file.
    """)
    return


@app.cell
def _(configure_logging, get_logger, log_kv):
    import tempfile
    from pathlib import Path
    tmpdir = tempfile.TemporaryDirectory()
    # Create temp directory for log file
    log_file = Path(tmpdir.name) / 'analysis.log'
    configure_logging(level='DEBUG', file_path=log_file)
    log_1 = get_logger('example.file')
    # Reconfigure to add file output
    log_1.debug('This debug message now appears (level=DEBUG)')
    log_1.info('Performing analysis step 1')
    log_1.info('Performing analysis step 2')
    log_kv('timing', {'step1': 12.4, 'step2': 8.7})
    print(f'\n📄 Log file written to: {log_file}')
    if log_file.exists():
        print('\nFile contents:')
        print('=' * 60)
    # Read back the log file
        print(log_file.read_text())
        print('=' * 60)
    else:
        print('⚠️  Log file not found (handler may buffer writes)')
        print('   In production, logs are typically flushed on close or periodically.')
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Example 6: Environment variable control

    You can set `NEURAL_ANALYSIS_LOG_LEVEL` environment variable to control the default log level without code changes.

    ```bash
    export NEURAL_ANALYSIS_LOG_LEVEL=DEBUG
    python your_script.py
    ```

    This is useful for debugging in production without modifying code.
    """)
    return


@app.cell
def _():
    import os

    # Demonstrate env var (normally set before running Python)
    os.environ["NEURAL_ANALYSIS_LOG_LEVEL"] = "WARNING"

    # This would now default to WARNING level
    # configure_logging()  # Would use WARNING from env var

    print("✓ Set NEURAL_ANALYSIS_LOG_LEVEL=WARNING")
    print("  (In practice, set this in your shell before running Python)")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Best Practices Summary

    1. **Configure once** at application/notebook startup with `configure_logging()`
    2. **Use get_logger(__name__)** in each module for proper namespacing
    3. **Prefer structured logs** with `log_kv()` for metrics and config
    4. **Add section markers** with `log_section()` for readability
    5. **Trace expensive functions** with `@log_calls()` decorator
    6. **Avoid print()** in library code; use logging instead
    7. **Use appropriate levels**: DEBUG for details, INFO for progress, WARNING/ERROR for issues

    See `docs/logging.md` for complete documentation and API reference.
    """)
    return


