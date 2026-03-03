"""Reproducibility utilities: seed management and environment capture.

Provides a context manager for deterministic computation and helpers
for recording environment provenance in HDF5 files.
"""

from __future__ import annotations

import contextlib
import platform
import random
import sys
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Generator


@contextlib.contextmanager
def reproducible(seed: int = 42) -> Generator[np.random.Generator]:
    """Context manager for reproducible computation.

    Saves and restores NumPy legacy and Python ``random`` state, and
    yields an explicit ``numpy.random.Generator`` for new-style usage.

    Args:
        seed: Seed value used for all RNGs.

    Yields:
        A ``numpy.random.Generator`` instance seeded with *seed*.

    Examples:
        >>> with reproducible(seed=42) as rng:
        ...     data = rng.normal(0, 1, size=(100, 10))
    """
    np_state = np.random.get_state()
    py_state = random.getstate()

    try:
        np.random.seed(seed)
        random.seed(seed)
        rng = np.random.default_rng(seed)
        yield rng
    finally:
        np.random.set_state(np_state)
        random.setstate(py_state)


def get_provenance() -> dict[str, str]:
    """Return a dict of environment provenance metadata.

    Useful for attaching to HDF5 datasets or result files so that the
    exact software versions that produced a result are recorded.
    """
    import neural_analysis

    return {
        "library_version": getattr(neural_analysis, "__version__", "unknown"),
        "python_version": sys.version,
        "platform": platform.platform(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "numpy_version": _get_version("numpy"),
        "scipy_version": _get_version("scipy"),
    }


def _get_version(package: str) -> str:
    try:
        import importlib.metadata

        return importlib.metadata.version(package)
    except Exception:
        return "unknown"
