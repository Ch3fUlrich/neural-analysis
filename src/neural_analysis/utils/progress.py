"""Standardized progress bar helper for long-running computations.

Wraps ``tqdm.auto`` with project-wide defaults and respects the
``NEURAL_ANALYSIS_QUIET`` environment variable for silent CI runs.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from tqdm.auto import tqdm

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator


def get_progress_bar[T](
    iterable: Iterable[T],
    desc: str = "",
    total: int | None = None,
    disable: bool = False,
) -> Iterator[T]:
    """Return a standardised progress bar wrapping *iterable*.

    The bar is automatically disabled when the environment variable
    ``NEURAL_ANALYSIS_QUIET`` is set to ``1`` or ``true``, which is useful
    for CI or batch processing where tqdm output is undesirable.

    Args:
        iterable: Items to iterate over.
        desc: Short description shown next to the bar.
        total: Total expected items (inferred from *iterable* if possible).
        disable: Force-disable the bar regardless of the env var.

    Returns:
        An iterator that yields items from *iterable* while displaying
        progress.
    """
    quiet = os.environ.get("NEURAL_ANALYSIS_QUIET", "").lower() in ("1", "true")
    return tqdm(  # type: ignore[return-value]
        iterable,
        desc=desc,
        total=total,
        disable=disable or quiet,
        leave=False,
        ncols=100,
    )
