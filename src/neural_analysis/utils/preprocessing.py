"""Preprocessing utilities (deprecated shim).

This module previously contained custom helpers like ``normalize_01``.
All normalization should now use ``sklearn.preprocessing.minmax_scale`` directly.
This shim preserves backwards-compatibility temporarily and will be removed.
"""

from __future__ import annotations

from warnings import warn
from sklearn.preprocessing import minmax_scale as normalize_01  # type: ignore

__all__ = ["normalize_01"]

warn(
    "neural_analysis.utils.preprocessing.normalize_01 is deprecated; "
    "import and use sklearn.preprocessing.minmax_scale instead.",
    DeprecationWarning,
    stacklevel=2,
)
