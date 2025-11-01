"""Example utilities for neural analysis (small, well-typed sample).

This module is intentionally tiny so CI/dev tooling can exercise linting,
type-checking and unit tests.
"""
from __future__ import annotations

from typing import Sequence


def mean(values: Sequence[float]) -> float:
    """Return the arithmetic mean of a sequence of floats.

    Raises ValueError for empty sequences.
    """
    if not values:
        raise ValueError("mean() requires a non-empty sequence")
    return sum(values) / len(values)


def normalize(values: Sequence[float]) -> list[float]:
    """Return min-max normalized list of values in range [0, 1]."""
    if not values:
        return []
    mn = min(values)
    mx = max(values)
    if mn == mx:
        return [0.0 for _ in values]
    return [(v - mn) / (mx - mn) for v in values]


__all__ = ["mean", "normalize"]
