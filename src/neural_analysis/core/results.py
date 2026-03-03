"""Result dataclasses for structured function returns.

These types provide a consistent, typed interface for analysis results
across the library. They replace ad-hoc tuple and dict returns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt


@dataclass(frozen=True)
class AnalysisResult:
    """Base result type for all analysis functions."""

    data: npt.NDArray[np.floating[Any]]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MetricResult(AnalysisResult):
    """Result from a metric computation."""

    metric_name: str = ""
    value: float = 0.0
    pairs: dict[tuple[int, int], float] | None = None


@dataclass(frozen=True)
class EmbeddingResult(AnalysisResult):
    """Result from dimensionality reduction."""

    method: str = ""
    n_components: int = 0
    explained_variance: npt.NDArray[np.floating[Any]] | None = None


@dataclass(frozen=True)
class DecodingResult:
    """Result from a decoding evaluation."""

    r_squared: float = 0.0
    mse: float = 0.0
    predictions: npt.NDArray[np.floating[Any]] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
