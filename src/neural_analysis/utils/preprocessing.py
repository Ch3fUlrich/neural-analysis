"""Preprocessing utilities for neural_analysis.

This module centralizes lightweight data preprocessing helpers used across
the project. Functions here should be fast, dependency-light, and
well-tested.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = ["normalize_01"]


def normalize_01(
    data: npt.ArrayLike,
    axis: int | None = None,
    min_val: float | None = None,
    max_val: float | None = None,
) -> np.ndarray:
    """Normalize an array to [0, 1].

    Parameters
    ----------
    data : array-like
        Input values to be normalized.
    axis : int or None, optional
        Axis along which to compute min/max. When None, use global min/max.
    min_val, max_val : float, optional
        Optional precomputed min/max to use for scaling.

    Returns
    -------
    np.ndarray
        Normalized array in [0, 1] (NaNs preserved where denominator is zero).
    """
    arr = np.asarray(data)

    dmin = np.min(arr, axis=axis, keepdims=True) if min_val is None else min_val
    dmax = np.max(arr, axis=axis, keepdims=True) if max_val is None else max_val
    denom = (dmax - dmin)
    # Avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (arr - dmin) / denom
        out = np.where(denom == 0, 0.0, out)
    return out
