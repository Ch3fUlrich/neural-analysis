"""Preprocessing utilities (deprecated shim).

This module previously contained custom helpers like ``minmax_scale``.
All normalization should now use ``sklearn.preprocessing.minmax_scale`` directly.
This shim preserves backwards-compatibility temporarily and will be removed.
"""

from __future__ import annotations
