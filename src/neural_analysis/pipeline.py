"""Unified analysis pipeline: generate -> embed -> decode -> structure index.

Provides a high-level ``run_analysis`` function and supporting dataclasses
for running a standard neural analysis workflow in a single call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from neural_analysis.data.datasets import DatasetType, generate_data
from neural_analysis.embeddings.dimensionality_reduction import (
    EmbeddingMethod,
    compute_embedding,
)
from neural_analysis.learning.decoding import compare_highd_lowd_decoding
from neural_analysis.topology.structure_index import compute_structure_index
from neural_analysis.utils.logging import get_logger, log_calls

logger = get_logger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the standard analysis pipeline."""

    dataset_type: DatasetType = "place_cells"
    n_samples: int = 5000
    n_features: int = 100
    noise: float = 0.0
    embedding_methods: list[EmbeddingMethod] = field(
        default_factory=lambda: ["pca", "umap"]
    )
    n_components: int = 3
    compute_si: bool = True
    si_n_bins: int = 20
    compute_decoding: bool = True
    decoding_k: int = 5
    decoding_n_folds: int = 5
    random_seed: int | None = None


@dataclass
class PipelineResult:
    """Results from a full analysis pipeline run."""

    activity: npt.NDArray[np.float64]
    labels: npt.NDArray[np.float64] | dict[str, Any]
    embeddings: dict[str, npt.NDArray[np.floating]]
    decoding: dict[str, Any] | None = None
    structure_index: dict[str, Any] | None = None


@log_calls(timeit=True)
def run_analysis(
    config: PipelineConfig | None = None,
    **kwargs: Any,
) -> PipelineResult:
    """Run a standard generate -> embed -> decode -> SI pipeline.

    Args:
        config: Pipeline configuration. If ``None``, defaults are used.
        **kwargs: Override any ``PipelineConfig`` field.

    Returns:
        PipelineResult with all computed data.

    Raises:
        ValueError: If dataset_type is invalid or parameters are incompatible.

    Examples:
        >>> result = run_analysis(dataset_type="swiss_roll", n_samples=1000)
        >>> result.activity.shape
        (1000, 3)
        >>> "pca" in result.embeddings
        True
    """
    if config is None:
        config = PipelineConfig(**kwargs)
    else:
        for k, v in kwargs.items():
            if hasattr(config, k):
                object.__setattr__(config, k, v)

    # Step 1: Generate data
    activity, labels = generate_data(
        config.dataset_type,
        n_samples=config.n_samples,
        n_features=config.n_features,
        noise=config.noise,
        seed=config.random_seed,
    )
    logger.info("Generated %s: shape %s", config.dataset_type, activity.shape)

    # Step 2: Compute embeddings
    embeddings: dict[str, npt.NDArray[np.floating]] = {}
    for method in config.embedding_methods:
        emb = compute_embedding(
            activity,
            method=method,
            n_components=config.n_components,
            random_state=config.random_seed,
        )
        embeddings[method] = emb
        logger.info("Embedding %s: shape %s", method, emb.shape)

    # Step 3: Compare decoding (high-d vs low-d)
    decoding = None
    if config.compute_decoding and embeddings:
        first_emb = np.asarray(next(iter(embeddings.values())), dtype=np.float64)
        label_array = _extract_label_array(labels)
        if label_array is not None:
            decoding = compare_highd_lowd_decoding(
                activity,
                first_emb,
                label_array,
                k=config.decoding_k,
                n_folds=config.decoding_n_folds,
            )
            logger.info("Decoding comparison complete")

    # Step 4: Structure index
    si_result = None
    if config.compute_si:
        label_array = _extract_label_array(labels)
        if label_array is not None:
            si_val, si_extras, si_observed, si_shuffled = compute_structure_index(
                activity,
                label_array,
                n_bins=config.si_n_bins,
            )
            si_result = {
                "structure_index": si_val,
                "extras": si_extras,
                "observed": si_observed,
                "shuffled": si_shuffled,
            }
            logger.info("Structure index: %.4f", si_val)

    return PipelineResult(
        activity=activity,
        labels=labels,
        embeddings=embeddings,
        decoding=decoding,
        structure_index=si_result,
    )


def _extract_label_array(
    labels: npt.NDArray[np.float64] | dict[str, Any],
) -> npt.NDArray[np.float64] | None:
    """Extract a 1-D label array from labels or dict metadata."""
    if isinstance(labels, np.ndarray):
        return labels
    if isinstance(labels, dict):
        for key in ("positions", "labels", "angles"):
            val = labels.get(key)
            if val is not None:
                arr = np.asarray(val, dtype=np.float64)
                if arr.ndim == 2:
                    return arr[:, 0]
                return arr
    return None
