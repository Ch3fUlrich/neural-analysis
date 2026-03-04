# Refactoring Plan: `neural-analysis`

> **Goal:** Improve usability, coding best practices, and reproducibility across the library.
> **Based on:** [research.md](research.md) deep analysis of the entire repository.
> **Date:** 2026-03-03

---

## Table of Contents

- [Refactoring Plan: `neural-analysis`](#refactoring-plan-neural-analysis)
  - [Table of Contents](#table-of-contents)
  - [Executive Summary](#executive-summary)
  - [Phase 0 — Foundation: CI, Versioning, Docker](#phase-0--foundation-ci-versioning-docker)
    - [0.1 Enable Automatic CI Triggers](#01-enable-automatic-ci-triggers)
    - [0.2 Make Quality Gates Blocking](#02-make-quality-gates-blocking)
    - [0.3 Establish a Versioning Scheme](#03-establish-a-versioning-scheme)
    - [0.4 Migrate Docker to UV](#04-migrate-docker-to-uv)
  - [Phase 1 — Code Quality: Decomposition and Patterns](#phase-1--code-quality-decomposition-and-patterns)
    - [1.1 Decompose Large Files](#11-decompose-large-files)
    - [1.2 Renderer Registry Pattern](#12-renderer-registry-pattern)
    - [1.3 Remove Dead Code](#13-remove-dead-code)
    - [1.4 Consistent Return Types](#14-consistent-return-types)
  - [Phase 2 — Usability: API Surface and Developer Experience](#phase-2--usability-api-surface-and-developer-experience)
    - [2.1 Top-Level Convenience Imports](#21-top-level-convenience-imports)
    - [2.2 Unified `run_analysis` Pipeline](#22-unified-run_analysis-pipeline)
    - [2.3 Improve Error Messages](#23-improve-error-messages)
    - [2.4 Configuration Dataclass](#24-configuration-dataclass)
    - [2.5 Progress Reporting](#25-progress-reporting)
    - [2.6 Enhanced Logging System](#26-enhanced-logging-system)
      - [2.6.1 Multi-File Log Architecture](#261-multi-file-log-architecture)
      - [2.6.2 Error Messages with Log References](#262-error-messages-with-log-references)
      - [2.6.3 Validation Helper with Log Links](#263-validation-helper-with-log-links)
      - [2.6.4 Backward Compatibility](#264-backward-compatibility)
      - [2.6.5 Usage Example](#265-usage-example)
  - [Phase 3 — Reproducibility: Seeds, Locks, and Environments](#phase-3--reproducibility-seeds-locks-and-environments)
    - [3.1 Global Seed Management](#31-global-seed-management)
    - [3.2 Lock File Hygiene](#32-lock-file-hygiene)
    - [3.3 Metadata Provenance](#33-metadata-provenance)
    - [3.4 Notebook Validation in CI](#34-notebook-validation-in-ci)
  - [Phase 4 — Testing: Reorganize and Strengthen](#phase-4--testing-reorganize-and-strengthen)
    - [4.1 Consolidate Test Files](#41-consolidate-test-files)
    - [4.2 Parametrize Where Possible](#42-parametrize-where-possible)
    - [4.3 Test Fixtures for Synthetic Data](#43-test-fixtures-for-synthetic-data)
    - [4.4 Coverage Thresholds](#44-coverage-thresholds)
  - [Phase 5 — Documentation: API Docs, Changelog, Releases](#phase-5--documentation-api-docs-changelog-releases)
    - [5.1 Auto-Generate API Reference](#51-auto-generate-api-reference)
    - [5.2 CHANGELOG.md](#52-changelogmd)
    - [5.3 PyPI Publishing Workflow](#53-pypi-publishing-workflow)
  - [Phase 6 — Performance: Profiling and Caching](#phase-6--performance-profiling-and-caching)
    - [6.1 Benchmark Suite](#61-benchmark-suite)
    - [6.2 Smarter Caching Defaults](#62-smarter-caching-defaults)
  - [Implementation Timeline](#implementation-timeline)

---

## Executive Summary

The `neural-analysis` library has a solid architecture — clear module boundaries, a three-layer storage stack, a comprehensive PlotGrid system, and 181+ passing tests. However, several areas need attention to bring the project from "working research code" to "reliable, usable library":

1. **CI runs manually** — No automated PR checks; type errors and format issues are non-blocking.
2. **Large files** — Five files exceed 2,000 lines and need decomposition.
3. **Version 0.0.0** — No versioning, no release process, no changelog.
4. **Docker uses pip** — Inconsistent with the UV-only development model.
5. **Test sprawl** — 65+ test files with `_additional`, `_more`, `_comprehensive` suffixes instead of structured organization.
6. **No reproducibility controls** — No global seed management, no environment hashing, no notebook validation.
7. **Primitive logging** — Current logging writes to a single optional file or stdout. No multi-file log separation, no print/warning capture, and error messages don't reference log files for debugging context.

This plan is structured in 7 phases (0–6), ordered by impact and dependency. Phases 0–1 are prerequisites. Phases 2–6 can be parallelized.

---

## Phase 0 — Foundation: CI, Versioning, Docker

### 0.1 Enable Automatic CI Triggers

**Problem:** CI only runs on manual dispatch — broken code can be merged unchecked.

**Change:** Uncomment the automatic triggers in `.github/workflows/ci.yml`:

```yaml
# .github/workflows/ci.yml
on:
  push:
    branches: [main, migration]
  pull_request:
    branches: [main]
  workflow_dispatch:  # Keep manual option
```

**Impact:** Every PR and push to main/migration triggers lint, type-check, and tests automatically.

---

### 0.2 Make Quality Gates Blocking

**Problem:** `mypy` and `ruff format` run with `continue-on-error: true`, so type errors and formatting issues don't fail the build.

**Change:** Remove `continue-on-error` from quality steps:

```yaml
# Before
- name: Type check
  run: uv run mypy src tests
  continue-on-error: true

# After
- name: Type check
  run: uv run mypy src tests
```

**Prerequisite:** Fix all existing mypy errors first. Run `uv run mypy src tests` and address every issue before making this change.

**Transition approach:** If the mypy cleanup is large, use `--warn-unused-ignores` and a mypy baseline file to gradually ratchet:

```bash
# Generate baseline of existing errors
uv run mypy src tests 2>&1 | tee .mypy_baseline.txt
# In CI, compare new errors against baseline
```

---

### 0.3 Establish a Versioning Scheme

**Problem:** Version is `0.0.0` with no process for releases.

**Change:** Adopt [CalVer](https://calver.org/) or SemVer. Recommended: **SemVer** starting at `0.1.0` (pre-1.0 indicates instability).

```toml
# pyproject.toml
[project]
version = "0.1.0"
```

Use `hatch-vcs` or a manual bump workflow:

```yaml
# .github/workflows/release.yml
name: Release
on:
  push:
    tags: ['v*']
jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
      - run: uv build
      - uses: pypa/gh-action-pypi-publish@release/v1
        with:
          password: ${{ secrets.PYPI_TOKEN }}
```

---

### 0.4 Migrate Docker to UV

**Problem:** Dockerfile runs `pip install` while the project mandates UV-only.

**Change:**

```dockerfile
# Dockerfile
FROM python:3.12-slim

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl redis-server \
    && rm -rf /var/lib/apt/lists/*

# Install UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --all-extras

COPY . .
RUN uv sync --locked --all-extras

EXPOSE 8888 6379
CMD ["uv", "run", "python", "-c", "print('neural-analysis ready')"]
```

This ensures the Docker environment exactly matches the developer experience.

---

## Phase 1 — Code Quality: Decomposition and Patterns

### 1.1 Decompose Large Files

**Problem:** Five files exceed 2,000 lines:

| File | Lines | Proposed Split |
|------|-------|----------------|
| `renderers.py` | 2,785 | `renderers_matplotlib.py` + `renderers_plotly.py` |
| `grid_config.py` | 2,479 | `grid_config.py` (core) + `grid_dispatch.py` (dispatch logic) |
| `synthetic_data.py` | 2,485 | `generators.py` (cell types) + `trajectories_gen.py` (trajectory/manifold) + `datasets.py` (composite helpers) |
| `synthetic_plots.py` | 2,452 | `synthetic_plots_1d.py` + `synthetic_plots_2d.py` + `synthetic_plots_3d.py` |
| `pairwise_metrics.py` | 2,033 | `pairwise_core.py` (dispatcher + helpers) + `pairwise_numba.py` (Numba kernels) |

**Strategy:** Split files while preserving the public API. Keep the original module as a re-export facade:

```python
# src/neural_analysis/plotting/renderers.py (becomes a facade)
"""Renderer functions — re-exported from backend-specific modules."""

from neural_analysis.plotting.renderers_matplotlib import (
    render_line_matplotlib,
    render_scatter_matplotlib,
    # ... all mpl renderers
)
from neural_analysis.plotting.renderers_plotly import (
    render_line_plotly,
    render_scatter_plotly,
    # ... all plotly renderers
)

__all__ = [
    "render_line_matplotlib",
    "render_scatter_matplotlib",
    "render_line_plotly",
    "render_scatter_plotly",
    # ...
]
```

This ensures zero breaking changes for existing imports.

---

### 1.2 Renderer Registry Pattern

**Problem:** 16-branch `if/elif` chains in `_plot_spec_matplotlib` and `_plot_spec_plotly`.

**Change:** Replace with a registry dict:

```python
# src/neural_analysis/plotting/grid_config.py

from typing import Callable, Any
from neural_analysis.plotting.renderers_matplotlib import (
    render_line_matplotlib,
    render_scatter_matplotlib,
    render_histogram_matplotlib,
    render_heatmap_matplotlib,
    render_violin_matplotlib,
    render_box_matplotlib,
    render_bar_matplotlib,
    render_scatter3d_matplotlib,
    render_trajectory_matplotlib,
    render_trajectory3d_matplotlib,
    render_kde_matplotlib,
    render_grouped_scatter_matplotlib,
    render_convex_hull_matplotlib,
    render_boolean_states_matplotlib,
    render_ellipse_matplotlib,
    render_heatmap_walls_matplotlib,
)

MATPLOTLIB_RENDERERS: dict[str, Callable[..., Any]] = {
    "line": render_line_matplotlib,
    "scatter": render_scatter_matplotlib,
    "histogram": render_histogram_matplotlib,
    "heatmap": render_heatmap_matplotlib,
    "violin": render_violin_matplotlib,
    "box": render_box_matplotlib,
    "bar": render_bar_matplotlib,
    "scatter3d": render_scatter3d_matplotlib,
    "trajectory": render_trajectory_matplotlib,
    "trajectory3d": render_trajectory3d_matplotlib,
    "kde": render_kde_matplotlib,
    "grouped_scatter": render_grouped_scatter_matplotlib,
    "convex_hull": render_convex_hull_matplotlib,
    "boolean_states": render_boolean_states_matplotlib,
    "ellipse": render_ellipse_matplotlib,
    "heatmap_walls": render_heatmap_walls_matplotlib,
}

# Similarly for PLOTLY_RENDERERS

def _plot_spec_matplotlib(self, ax, spec: PlotSpec) -> None:
    """Dispatch a PlotSpec to the correct matplotlib renderer."""
    renderer = MATPLOTLIB_RENDERERS.get(spec.plot_type)
    if renderer is None:
        raise ValueError(f"Unknown plot type: {spec.plot_type!r}. "
                        f"Available: {sorted(MATPLOTLIB_RENDERERS)}")
    renderer(ax, spec)
```

**Benefits:** Adding a new plot type = one dict entry + one function. No dispatch chain to modify.

---

### 1.3 Remove Dead Code

**Problem:** `preprocessing.py` is a deprecated 10-line shim.

```python
# Current content of preprocessing.py
"""Deprecated. Use sklearn.preprocessing directly."""
# ... import redirects
```

**Action:** Delete `src/neural_analysis/utils/preprocessing.py` and remove it from `utils/__init__.py` exports. Any existing imports should be redirected at the call site.

---

### 1.4 Consistent Return Types

**Problem:** Some functions return plain arrays, others return `(data, metadata)` tuples, and some return dicts. This inconsistency forces callers to guess.

**Change:** Define result dataclasses for each domain:

```python
# src/neural_analysis/core/results.py (new module)
from dataclasses import dataclass, field
from typing import Any
import numpy as np

@dataclass(frozen=True)
class AnalysisResult:
    """Base result type for all analysis functions."""
    data: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class MetricResult(AnalysisResult):
    """Result from a metric computation."""
    metric_name: str = ""
    value: float = 0.0

@dataclass(frozen=True)
class EmbeddingResult(AnalysisResult):
    """Result from dimensionality reduction."""
    method: str = ""
    explained_variance: np.ndarray | None = None

@dataclass(frozen=True)
class DecodingResult:
    """Result from a decoding evaluation."""
    r_squared: float = 0.0
    mse: float = 0.0
    predictions: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

**Transition:** Introduce these alongside existing tuple returns. Add `@deprecated` warnings on the old tuple API over 2-3 minor versions.

---

## Phase 2 — Usability: API Surface and Developer Experience

### 2.1 Top-Level Convenience Imports

**Problem:** Users must navigate deep import paths: `from neural_analysis.metrics.pairwise_metrics import compute_pairwise_matrix`.

**Change:** Expose key functions at package root:

```python
# src/neural_analysis/__init__.py

# Data generation
from neural_analysis.data import generate_data

# Metrics
from neural_analysis.metrics import (
    compute_pairwise_matrix,
    shape_distance,
    filter_outlier,
)

# Embeddings
from neural_analysis.embeddings import compute_embedding

# Learning
from neural_analysis.learning import (
    cross_validated_knn_decoder,
    compare_classifiers,
    compare_clusterers,
)

# Topology
from neural_analysis.topology import compute_structure_index

# Plotting
from neural_analysis.plotting import PlotGrid, PlotSpec, PlotConfig

__all__ = [
    "generate_data",
    "compute_pairwise_matrix",
    "shape_distance",
    "filter_outlier",
    "compute_embedding",
    "cross_validated_knn_decoder",
    "compare_classifiers",
    "compare_clusterers",
    "compute_structure_index",
    "PlotGrid",
    "PlotSpec",
    "PlotConfig",
]
```

This lets users write `from neural_analysis import generate_data, compute_embedding` instead of navigating submodules.

---

### 2.2 Unified `run_analysis` Pipeline

**Problem:** Common analysis workflows (generate → embed → decode → visualize) require chaining many functions manually.

**Change:** Add a pipeline helper:

```python
# src/neural_analysis/pipeline.py

from dataclasses import dataclass, field
from typing import Any
import numpy as np

from neural_analysis.data import generate_data
from neural_analysis.embeddings import compute_embedding
from neural_analysis.learning.decoding import compare_highd_lowd_decoding
from neural_analysis.topology import compute_structure_index
from neural_analysis.utils.logging import get_logger, log_calls

logger = get_logger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the standard analysis pipeline."""
    dataset_type: str = "place_cells_2d"
    n_cells: int = 100
    n_timesteps: int = 5000
    embedding_methods: list[str] = field(default_factory=lambda: ["pca", "umap"])
    n_components: int = 3
    compute_si: bool = True
    si_n_bins: int = 20
    si_n_neighbors: int = 10
    random_seed: int | None = None


@dataclass
class PipelineResult:
    """Results from a full analysis pipeline run."""
    activity: np.ndarray
    metadata: dict[str, Any]
    embeddings: dict[str, np.ndarray]
    decoding: dict[str, Any] | None = None
    structure_index: dict[str, Any] | None = None


@log_calls(timeit=True)
def run_analysis(config: PipelineConfig | None = None, **kwargs: Any) -> PipelineResult:
    """Run a standard generate → embed → decode → SI pipeline.

    Args:
        config: Pipeline configuration. If None, uses defaults.
        **kwargs: Override any PipelineConfig field.

    Returns:
        PipelineResult with all computed data.
    """
    if config is None:
        config = PipelineConfig(**kwargs)

    # Step 1: Generate data
    activity, metadata = generate_data(
        config.dataset_type,
        n_cells=config.n_cells,
        n_timesteps=config.n_timesteps,
    )
    logger.info("Generated %s: shape %s", config.dataset_type, activity.shape)

    # Step 2: Compute embeddings
    embeddings = {}
    for method in config.embedding_methods:
        emb, emb_meta = compute_embedding(
            activity, method=method, n_components=config.n_components
        )
        embeddings[method] = emb
        logger.info("Embedding %s: shape %s", method, emb.shape)

    # Step 3: Compare decoding
    decoding = None
    if embeddings:
        first_emb = next(iter(embeddings.values()))
        decoding = compare_highd_lowd_decoding(
            activity, first_emb, metadata.get("positions")
        )

    # Step 4: Structure index
    si_result = None
    if config.compute_si and "positions" in metadata:
        si_result = compute_structure_index(
            activity,
            metadata["positions"],
            n_bins=config.si_n_bins,
            n_neighbors=config.si_n_neighbors,
        )

    return PipelineResult(
        activity=activity,
        metadata=metadata,
        embeddings=embeddings,
        decoding=decoding,
        structure_index=si_result,
    )
```

---

### 2.3 Improve Error Messages

**Problem:** Some error messages are generic ("invalid input") without guiding the user to fix the issue.

**Change:** Adopt a consistent error message pattern:

```python
# Pattern: "What happened" + "What was expected" + "What was received"

def compute_pairwise_matrix(x: np.ndarray, y: np.ndarray, metric: str) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError(
            f"Input x must be 2-dimensional (samples × features), "
            f"but got shape {x.shape} with {x.ndim} dimensions."
        )

    valid_metrics = POINT_TO_POINT_METRICS | DISTRIBUTION_METRICS | SHAPE_METRICS
    if metric not in valid_metrics:
        raise ValueError(
            f"Unknown metric {metric!r}. "
            f"Choose from: {sorted(valid_metrics)}"
        )
```

Apply this pattern consistently across all public `match/case` dispatchers and validation code. The goal: every error message tells the user _what went wrong_ and _how to fix it_.

---

### 2.4 Configuration Dataclass

**Problem:** Functions with 10+ keyword arguments are hard to document and reuse.

**Change:** For functions with many parameters, offer a config dataclass as an alternative:

```python
@dataclass
class StructureIndexConfig:
    """Configuration for structure index computation."""
    n_bins: int = 20
    n_neighbors: int = 10
    num_shuffles: int = 100
    use_faiss: bool = True
    save_path: str | None = None
    dataset_name: str = "default"


def compute_structure_index(
    data: np.ndarray,
    labels: np.ndarray,
    config: StructureIndexConfig | None = None,
    **kwargs: Any,
) -> dict:
    """Compute structure index.

    Args:
        data: Neural activity matrix (timesteps × neurons).
        labels: Behavioral labels for each timestep.
        config: Configuration object. Overridden by kwargs.
        **kwargs: Individual parameters (override config fields).
    """
    if config is None:
        config = StructureIndexConfig(**kwargs)
    else:
        # Allow kwargs to override config fields
        for k, v in kwargs.items():
            if hasattr(config, k):
                object.__setattr__(config, k, v)
    ...
```

This preserves backward compatibility (kwargs still work) while enabling config-object workflows.

---

### 2.5 Progress Reporting

**Problem:** Long computations (parameter sweeps, batch comparisons) show progress bars but with inconsistent formatting.

**Change:** Standardize on `tqdm` with a `get_progress_bar` helper:

```python
# src/neural_analysis/utils/progress.py

from tqdm.auto import tqdm
from typing import Iterable, TypeVar

T = TypeVar("T")


def get_progress_bar(
    iterable: Iterable[T],
    desc: str = "",
    total: int | None = None,
    disable: bool = False,
) -> Iterable[T]:
    """Standardized progress bar for long computations.

    Respects NEURAL_ANALYSIS_QUIET env var to disable in CI.
    """
    import os
    quiet = os.environ.get("NEURAL_ANALYSIS_QUIET", "").lower() in ("1", "true")
    return tqdm(
        iterable,
        desc=desc,
        total=total,
        disable=disable or quiet,
        leave=False,
        ncols=100,
    )
```

---

### 2.6 Enhanced Logging System

**Problem:** The current `utils/logging.py` (~216 lines) provides basic single-file logging. It lacks:
- Multi-file log separation (errors vs. warnings vs. debug)
- Automatic `print()` and `warnings.warn()` capture
- Log file path references in error messages (so users can find detailed context)
- Log rotation for long-running sessions
- Session-scoped log directories

When an error occurs, the user sees a traceback but has no pointer to a debug log with the full execution context leading up to the failure.

**Change:** Rewrite `utils/logging.py` to support a multi-file, session-aware logging system.

#### 2.6.1 Multi-File Log Architecture

Create separate log files by severity, plus a combined `all.log`:

```
logs/
└── {session_id}/              # e.g. 2026-03-03_14-30-00
    ├── all.log                # Everything (DEBUG and above)
    ├── errors.log             # ERROR and CRITICAL only
    ├── warnings.log           # WARNING and above
    └── info.log               # INFO and above (no DEBUG noise)
```

```python
# src/neural_analysis/utils/logging.py (enhanced)

from __future__ import annotations

import logging
import os
import sys
import time
import warnings
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

__all__ = [
    "configure_logging",
    "get_logger",
    "get_log_dir",
    "log_section",
    "log_kv",
    "log_calls",
    "LogFileReference",
]

_CONFIGURED = False
_LOGGER_NAME = "neural_analysis"
_LOG_DIR: Path | None = None
_SESSION_ID: str = ""


@dataclass
class LogConfig:
    """Configuration for the multi-file logging system."""
    level: int = logging.DEBUG
    log_root: Path = Path("logs")
    session_id: str = ""  # Auto-generated if empty
    max_bytes_per_file: int = 10 * 1024 * 1024  # 10 MB
    backup_count: int = 5
    capture_warnings: bool = True
    capture_print: bool = False  # Opt-in: redirect print() to info log
    propagate: bool = False
    console_level: int = logging.INFO  # Console shows INFO+, files get DEBUG+
    fmt: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt: str = "%Y-%m-%d %H:%M:%S"


class LogFileReference:
    """Helper to generate log file references for error messages."""

    @staticmethod
    def error_log() -> str:
        """Return the path to the current session's error log."""
        if _LOG_DIR is None:
            return "(logging not configured — call configure_logging())"
        return str(_LOG_DIR / "errors.log")

    @staticmethod
    def debug_log() -> str:
        """Return the path to the current session's full debug log."""
        if _LOG_DIR is None:
            return "(logging not configured — call configure_logging())"
        return str(_LOG_DIR / "all.log")

    @staticmethod
    def session_dir() -> str:
        """Return the current session's log directory."""
        if _LOG_DIR is None:
            return "(logging not configured)"
        return str(_LOG_DIR)


def get_log_dir() -> Path | None:
    """Return the current session log directory, or None if not configured."""
    return _LOG_DIR


def _make_session_id() -> str:
    """Generate a timestamped session ID."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")


def configure_logging(
    *,
    level: int | str | None = None,
    log_root: str | Path | None = None,
    session_id: str | None = None,
    capture_warnings: bool = True,
    capture_print: bool = False,
    console_level: int | str | None = None,
    fmt: str | None = None,
    datefmt: str | None = None,
    max_bytes_per_file: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> Path:
    """Configure multi-file, session-aware logging.

    Creates a session directory under `log_root` with separate log files
    for different severity levels. Returns the session log directory path.

    Parameters
    ----------
    level : int | str | None
        Root log level. Default DEBUG (files capture everything).
    log_root : str | Path | None
        Root directory for logs. Default: ./logs
    session_id : str | None
        Session identifier for the log subdirectory. Auto-generated if None.
    capture_warnings : bool
        Route Python warnings.warn() to the warning log. Default True.
    capture_print : bool
        Redirect print() to the info log. Default False (opt-in).
    console_level : int | str | None
        Minimum level for console output. Default INFO.
    max_bytes_per_file : int
        Max size before log rotation. Default 10 MB.
    backup_count : int
        Number of rotated log backups. Default 5.

    Returns
    -------
    Path
        The session log directory.
    """
    global _CONFIGURED, _LOG_DIR, _SESSION_ID
    if _CONFIGURED:
        return _LOG_DIR  # type: ignore[return-value]

    cfg = LogConfig()

    # Resolve level
    if isinstance(level, str):
        level_val = getattr(logging, level.upper(), logging.DEBUG)
    elif isinstance(level, int):
        level_val = level
    else:
        level_val = _level_from_env(cfg.level)

    # Resolve console level
    if isinstance(console_level, str):
        console_level_val = getattr(logging, console_level.upper(), logging.INFO)
    elif isinstance(console_level, int):
        console_level_val = console_level
    else:
        console_level_val = cfg.console_level

    fmt_val = fmt or cfg.fmt
    datefmt_val = datefmt or cfg.datefmt
    root = Path(log_root) if log_root else cfg.log_root
    _SESSION_ID = session_id or _make_session_id()
    _LOG_DIR = root / _SESSION_ID
    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(level_val)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter(fmt_val, datefmt=datefmt_val)

    # --- Console handler (INFO+ by default) ---
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(console_level_val)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # --- File handlers with rotation ---
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
        # Filter: only write messages at EXACTLY this level and above
        # (errors.log gets ERROR+, warnings.log gets WARNING+, etc.)
        logger.addHandler(handler)

    # --- Capture Python warnings ---
    if capture_warnings:
        logging.captureWarnings(True)
        warnings_logger = logging.getLogger("py.warnings")
        warnings_logger.handlers = logger.handlers.copy()

    # --- Optional: capture print() ---
    if capture_print:
        sys.stdout = _PrintCapture(logger, logging.INFO, sys.stdout)  # type: ignore[assignment]
        sys.stderr = _PrintCapture(logger, logging.ERROR, sys.stderr)  # type: ignore[assignment]

    _CONFIGURED = True
    logger.info(
        "Logging session started: %s  |  Logs: %s",
        _SESSION_ID, _LOG_DIR,
    )
    return _LOG_DIR


class _PrintCapture:
    """Stream wrapper that tees print() output to a logger."""

    def __init__(
        self, logger: logging.Logger, level: int, original_stream: Any
    ) -> None:
        self._logger = logger
        self._level = level
        self._original = original_stream

    def write(self, msg: str) -> int:
        if msg and msg.strip():
            self._logger.log(self._level, msg.rstrip())
        return self._original.write(msg)

    def flush(self) -> None:
        self._original.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)
```

#### 2.6.2 Error Messages with Log References

Every user-facing error should include a reference to the relevant log file:

```python
# Pattern for error messages that reference logs
from neural_analysis.utils.logging import LogFileReference

def compute_pairwise_matrix(x, y, metric):
    if x.ndim != 2:
        raise ValueError(
            f"Input x must be 2-dimensional (samples × features), "
            f"but got shape {x.shape} with {x.ndim} dimensions.\n"
            f"  → Debug log: {LogFileReference.debug_log()}"
        )

    valid_metrics = POINT_TO_POINT_METRICS | DISTRIBUTION_METRICS | SHAPE_METRICS
    if metric not in valid_metrics:
        raise ValueError(
            f"Unknown metric {metric!r}. "
            f"Choose from: {sorted(valid_metrics)}\n"
            f"  → Debug log: {LogFileReference.debug_log()}"
        )
```

For exceptions caught and re-raised internally:

```python
try:
    result = expensive_computation(data)
except Exception as e:
    logger.error(
        "Computation failed for input shape %s: %s",
        data.shape, e, exc_info=True,  # Full traceback in error log
    )
    raise RuntimeError(
        f"Computation failed: {e}\n"
        f"  → Full traceback: {LogFileReference.error_log()}\n"
        f"  → Execution context: {LogFileReference.debug_log()}"
    ) from e
```

#### 2.6.3 Validation Helper with Log Links

Extend `do_critical` in `validation.py` to include log references:

```python
# src/neural_analysis/utils/validation.py (enhanced)

from neural_analysis.utils.logging import get_logger, LogFileReference

logger = get_logger(__name__)


def do_critical(
    exc_type: type[Exception],
    message: str,
    *,
    include_log_ref: bool = True,
) -> None:
    """Log a CRITICAL error and raise an exception with log reference.

    Args:
        exc_type: Exception class to raise.
        message: Error message.
        include_log_ref: If True, append the log file path to the error.
    """
    logger.critical(message)
    if include_log_ref:
        message = (
            f"{message}\n"
            f"  → Error details: {LogFileReference.error_log()}\n"
            f"  → Full context: {LogFileReference.debug_log()}"
        )
    raise exc_type(message)
```

#### 2.6.4 Backward Compatibility

The existing API (`get_logger`, `log_section`, `log_kv`, `@log_calls`) remains unchanged. The only difference:

- `configure_logging()` now **returns a `Path`** (the session directory) and creates file handlers automatically.
- Old single-file usage (`configure_logging(file_path="app.log")`) is still supported as a fallback mode when `log_root` is not specified and `file_path` is given.
- All existing callers of `get_logger(__name__)` work identically — they inherit the multi-file handlers from the parent logger.

#### 2.6.5 Usage Example

```python
from neural_analysis.utils.logging import configure_logging, get_logger, LogFileReference

# At application/notebook startup
log_dir = configure_logging(
    log_root="logs",
    capture_warnings=True,
    capture_print=True,  # Redirect stray print() calls
)
print(f"Logs → {log_dir}")

# In any module
logger = get_logger(__name__)
logger.debug("Detailed computation step")      # → all.log only
logger.info("Processing 100 cells")              # → all.log + info.log
logger.warning("Falling back to pure Python")    # → all.log + info.log + warnings.log
logger.error("Computation failed")               # → all.log + info.log + warnings.log + errors.log

# In error handling
try:
    result = shape_distance(mtx1, mtx2, method="procrustes")
except Exception as e:
    logger.error("Shape distance failed: %s", e, exc_info=True)
    raise RuntimeError(
        f"Shape distance failed: {e}\n"
        f"  → See: {LogFileReference.error_log()}"
    ) from e
```

The log directory structure after a session:

```
logs/2026-03-03_14-30-00/
├── all.log         (4.2 MB — every DEBUG+ message, full context)
├── info.log        (1.1 MB — operational summaries)
├── warnings.log    (12 KB — fallback notices, deprecation warnings)
└── errors.log      (3 KB — only failures, with full tracebacks)
```

**Impact:** Users seeing an error in the console immediately know where to look for the full story. The `errors.log` has the traceback, and `all.log` has the execution context leading up to it.

---

## Phase 3 — Reproducibility: Seeds, Locks, and Environments

### 3.1 Global Seed Management

**Problem:** No centralized random seed management. Each function may or may not accept a seed parameter.

**Change:** Add a seed context manager:

```python
# src/neural_analysis/utils/reproducibility.py

import contextlib
import numpy as np
from typing import Generator


@contextlib.contextmanager
def reproducible(seed: int = 42) -> Generator[np.random.Generator, None, None]:
    """Context manager for reproducible computation.

    Sets numpy, random, and returns a Generator for explicit RNG.

    Usage:
        with reproducible(seed=42) as rng:
            data = rng.normal(0, 1, size=(100, 50))
    """
    import random

    # Save state
    np_state = np.random.get_state()
    py_state = random.getstate()

    try:
        np.random.seed(seed)
        random.seed(seed)
        rng = np.random.default_rng(seed)
        yield rng
    finally:
        # Restore state
        np.random.set_state(np_state)
        random.setstate(py_state)
```

Usage in analysis:

```python
from neural_analysis.utils.reproducibility import reproducible

with reproducible(seed=42) as rng:
    activity, meta = generate_data("place_cells_2d", n_cells=100, rng=rng)
    embedding, _ = compute_embedding(activity, method="umap", random_state=42)
```

---

### 3.2 Lock File Hygiene

**Problem:** The lock file (`uv.lock`) should be committed and verified in CI.

**Change:** Ensure the CI lockfile check is strict:

```yaml
# .github/workflows/ci.yml
- name: Verify lockfile
  run: |
    uv lock --check
    echo "Lockfile is up to date"
```

Add to `CONTRIBUTING.md`:

> After adding or changing dependencies, always run `uv lock` and commit the updated `uv.lock`.

---

### 3.3 Metadata Provenance

**Problem:** HDF5 files don't consistently record the library version, Python version, or environment hash that produced them.

**Change:** Add provenance metadata to every HDF5 save:

```python
# src/neural_analysis/utils/provenance.py

import platform
import sys
from datetime import datetime, timezone
from typing import Any


def get_provenance() -> dict[str, Any]:
    """Generate provenance metadata for the current environment."""
    import neural_analysis

    return {
        "library_version": getattr(neural_analysis, "__version__", "unknown"),
        "python_version": sys.version,
        "platform": platform.platform(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "numpy_version": _get_version("numpy"),
        "scipy_version": _get_version("scipy"),
    }


def _get_version(package: str) -> str:
    try:
        import importlib.metadata
        return importlib.metadata.version(package)
    except Exception:
        return "unknown"
```

Inject into HDF5 saves:

```python
# In io.py save_result_to_hdf5_dataset
from neural_analysis.utils.provenance import get_provenance

def save_result_to_hdf5_dataset(filepath, dataset_name, result_key, **data):
    provenance = get_provenance()
    with h5py.File(filepath, "a") as f:
        grp = f.require_group(f"{dataset_name}/{result_key}")
        # Save provenance as group attributes
        for k, v in provenance.items():
            grp.attrs[f"_provenance_{k}"] = v
        # ... rest of save logic
```

---

### 3.4 Notebook Validation in CI

**Problem:** 17 Marimo notebooks exist but none are executed in CI. They could silently break.

**Change:** Add a notebook validation step:

```yaml
# .github/workflows/ci.yml
- name: Validate notebooks
  run: |
    for nb in examples/*_marimo_nb.py; do
      echo "Validating: $nb"
      uv run python -c "import importlib.util; spec = importlib.util.spec_from_file_location('nb', '$nb'); mod = importlib.util.module_from_spec(spec)"
    done
  continue-on-error: true  # Start non-blocking, tighten later
```

Or use Marimo's built-in validation:

```bash
uv run marimo check examples/  # If marimo supports batch validation
```

---

## Phase 4 — Testing: Reorganize and Strengthen

### 4.1 Consolidate Test Files

**Problem:** 65+ test files, many with `_additional`, `_comprehensive`, `_final`, `_more` suffixes:

```
test_plotting.py
test_plotting_additional.py
test_plotting_comprehensive.py
test_plotting_final.py
test_plotting_more.py
```

**Change:** Merge related files into one file per source module. Use test classes to organize:

```python
# tests/test_plotting.py (consolidated)

import pytest


class TestPlotGridCore:
    """Tests for PlotGrid dispatch and rendering."""

    def test_line_matplotlib(self): ...
    def test_line_plotly(self): ...
    def test_scatter_matplotlib(self): ...


class TestPlotGridLayout:
    """Tests for GridLayoutConfig and auto-sizing."""

    def test_auto_size_grid_small(self): ...
    def test_auto_size_grid_large(self): ...


class TestPlotGridEdgeCases:
    """Edge case tests for PlotGrid."""

    def test_empty_specs_list(self): ...
    def test_single_point_trajectory(self): ...
```

**Process:**
1. List all test files per module.
2. Identify duplicate/overlapping tests.
3. Merge into one file using classes for organization.
4. Verify coverage doesn't drop: `uv run pytest -v --cov --cov-report=term-missing`.

---

### 4.2 Parametrize Where Possible

**Problem:** Repetitive tests that differ only by input values.

**Change:** Use `pytest.mark.parametrize`:

```python
# Before: 7 separate test functions for 7 embedding methods
def test_pca_embedding(self): ...
def test_umap_embedding(self): ...
def test_tsne_embedding(self): ...

# After: 1 parametrized test
@pytest.mark.parametrize("method", ["pca", "umap", "tsne", "mds", "isomap", "lle", "spectral"])
def test_embedding_method(method, sample_data):
    result, meta = compute_embedding(sample_data, method=method, n_components=2)
    assert result.shape == (sample_data.shape[0], 2)
    assert "method" in meta


@pytest.mark.parametrize("backend", ["matplotlib", "plotly"])
@pytest.mark.parametrize("plot_type", ["line", "scatter", "histogram", "heatmap"])
def test_plot_type_renders(backend, plot_type, sample_plot_data):
    spec = PlotSpec(data=sample_plot_data, plot_type=plot_type)
    grid = PlotGrid([spec], config=PlotConfig(), backend=backend)
    fig = grid.plot()
    assert fig is not None
```

---

### 4.3 Test Fixtures for Synthetic Data

**Problem:** Many tests independently generate their own synthetic data, which is slow and inconsistent.

**Change:** Create shared fixtures in `conftest.py`:

```python
# tests/conftest.py

import pytest
import numpy as np
from neural_analysis.data import generate_data


@pytest.fixture(scope="session")
def place_cells_2d():
    """Session-scoped place cell dataset for reuse across tests."""
    activity, metadata = generate_data(
        "place_cells_2d", n_cells=50, n_timesteps=2000
    )
    return activity, metadata


@pytest.fixture(scope="session")
def random_activity():
    """Session-scoped random activity matrix."""
    rng = np.random.default_rng(42)
    return rng.normal(0, 1, size=(1000, 50))


@pytest.fixture(scope="function")
def small_matrix():
    """Small matrix for quick unit tests."""
    return np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
```

---

### 4.4 Coverage Thresholds

**Problem:** No minimum coverage enforced — coverage can silently drop.

**Change:** Add a `--cov-fail-under` threshold:

```toml
# pyproject.toml
[tool.pytest.ini_options]
addopts = "--cov=src --cov-branch --cov-fail-under=70"
```

Start at 70% and increase as coverage improves. Track with Codecov (already configured in CI).

---

## Phase 5 — Documentation: API Docs, Changelog, Releases

### 5.1 Auto-Generate API Reference

**Problem:** `docs/function_registry.md` requires manual updates. It will drift.

**Change:** Use `scripts/generate_function_registry.py` (already exists) as a CI step:

```yaml
# .github/workflows/ci.yml
- name: Verify function registry
  run: |
    uv run python scripts/generate_function_registry.py --check
    # Fails if the registry is out of date
```

Also add Sphinx autodoc for full API reference:

```python
# docs/conf.py
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # Google-style docstrings
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
]

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
```

---

### 5.2 CHANGELOG.md

**Problem:** No changelog exists. Users can't know what changed between versions.

**Change:** Create `CHANGELOG.md` following [Keep a Changelog](https://keepachangelog.com/):

```markdown
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Refactored instruction files for VS Code Copilot integration
- Detailed refactoring plan (plan.md)

### Changed
- (List changes here as they happen)

## [0.1.0] - YYYY-MM-DD

### Added
- Initial modular architecture: data, metrics, embeddings, learning, topology, plotting, utils
- PlotGrid visualization system (16 plot types, dual matplotlib/plotly backend)
- Three-layer storage stack (HDF5 + DuckDB + Redis)
- 181+ passing tests
- 17 Marimo example notebooks
```

---

### 5.3 PyPI Publishing Workflow

**Problem:** No mechanism to publish releases.

**Change:** Add a release workflow triggered by git tags:

```yaml
# .github/workflows/release.yml
name: Publish to PyPI

on:
  push:
    tags: ['v*']

permissions:
  id-token: write  # Trusted publishing

jobs:
  publish:
    runs-on: ubuntu-latest
    environment: pypi
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
      - run: uv build
      - uses: pypa/gh-action-pypi-publish@release/v1
```

Release process:
1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Commit: `git commit -m "chore: release v0.1.0"`
4. Tag: `git tag v0.1.0`
5. Push: `git push origin v0.1.0`

---

## Phase 6 — Performance: Profiling and Caching

### 6.1 Benchmark Suite

**Problem:** No way to track performance regressions.

**Change:** Add a lightweight benchmark script:

```python
# scripts/benchmark.py

import time
import json
import numpy as np
from neural_analysis.data import generate_data
from neural_analysis.metrics import compute_pairwise_matrix, shape_distance
from neural_analysis.embeddings import compute_embedding


def benchmark(name: str, func, *args, repeats: int = 3, **kwargs) -> dict:
    """Run a function multiple times and report timing."""
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        func(*args, **kwargs)
        times.append(time.perf_counter() - start)
    return {
        "name": name,
        "mean_s": np.mean(times),
        "std_s": np.std(times),
        "min_s": np.min(times),
    }


def main():
    results = []

    # Generate test data
    activity, meta = generate_data("place_cells_2d", n_cells=100, n_timesteps=5000)
    positions = meta["positions"]

    # Benchmark: pairwise euclidean
    results.append(benchmark(
        "pairwise_euclidean_100x5000",
        compute_pairwise_matrix, activity, activity, "euclidean"
    ))

    # Benchmark: shape distance (procrustes)
    results.append(benchmark(
        "procrustes_100x5000",
        shape_distance, activity[:2500], activity[2500:], "procrustes"
    ))

    # Benchmark: PCA embedding
    results.append(benchmark(
        "pca_100x5000_to_3d",
        compute_embedding, activity, method="pca", n_components=3
    ))

    # Output
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
```

Run periodically and compare against previous results: `uv run python scripts/benchmark.py > benchmarks/latest.json`.

---

### 6.2 Smarter Caching Defaults

**Problem:** Redis TTL is fixed at 3600s. Some computations are cheap (seconds) while others take hours.

**Change:** Let functions specify their cache TTL based on computation cost:

```python
# In StorageManager
def save_data(
    self,
    key: str,
    data: Any,
    metadata: dict | None = None,
    ttl: int | None = None,  # Override default TTL
    **kwargs,
) -> None:
    """Save data to the storage stack.

    Args:
        ttl: Cache TTL in seconds. None uses default.
            Suggested values:
            - 300 (5 min) for fast computations
            - 3600 (1 hr) for moderate computations
            - 86400 (24 hr) for expensive parameter sweeps
    """
    effective_ttl = ttl or self.config.cache_ttl
    ...
```

---

## Implementation Timeline

| Phase | Est. Effort | Priority | Dependencies |
|-------|-------------|----------|--------------|
| **0: Foundation** | 4-8 hours | **Critical** | None |
| **1: Code Quality** | 16-24 hours | **High** | Phase 0 |
| **2: Usability** | 16-24 hours | **High** | Phase 0 |
| **3: Reproducibility** | 8-12 hours | **Medium** | Phase 0 |
| **4: Testing** | 12-16 hours | **High** | Phase 0 |
| **5: Documentation** | 6-10 hours | **Medium** | Phase 0.3 |
| **6: Performance** | 6-8 hours | **Low** | Phase 1 |

**Total estimated effort:** 68-102 hours

**Recommended execution order:**
1. Phase 0 (CI + versioning + Docker) — do first, unblocks everything
2. Phase 4 (test consolidation) — reduces noise for all other phases
3. Phase 1 (file decomposition) — biggest code quality win
4. Phase 2 (usability) — biggest user-facing improvement
5. Phase 3 (reproducibility) — enables reliable research outputs
6. Phase 5 (docs + releases) — enables community adoption
7. Phase 6 (performance) — optimization after correctness is solid

---

---

*The detailed task list has been moved to [todo.md](todo.md).*

*This plan addresses the weaknesses identified in [research.md](research.md) and builds on the existing architectural strengths of the library.*
