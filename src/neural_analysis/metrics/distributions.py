"""Distribution comparison utilities for neural data analysis.

This module provides functions for comparing probability distributions using
various statistical metrics. It includes both pairwise comparisons and
group-based comparisons with optional outlier filtering.

All distance computations delegate to the distance module to avoid code duplication.
"""

from __future__ import annotations

from typing import Literal
import logging

import numpy as np
import numpy.typing as npt

from .distance import (
    euclidean_distance,
    mahalanobis_distance,
    cosine_similarity,
    wasserstein_distance_multi,
    kolmogorov_smirnov_distance,
    jensen_shannon_divergence,
    distribution_distance,
)

try:
    from ..utils.logging import log_calls, get_logger  # type: ignore
except ImportError:
    def log_calls(**kwargs):  # type: ignore
        def decorator(func):  # type: ignore
            return func
        return decorator
    def get_logger(name: str):  # type: ignore
        return logging.getLogger(name)

# Module logger
logger = get_logger(__name__)

__all__ = ["compare_distributions", "compare_distribution_groups"]


@log_calls(level=logging.DEBUG)
def compare_distributions(
    points1: npt.ArrayLike,
    points2: npt.ArrayLike,
    metric: Literal[
        "wasserstein",
        "kolmogorov-smirnov",
        "jensen-shannon",
        "euclidean",
        "mahalanobis",
        "cosine",
    ] = "wasserstein",
) -> float:
    """Compare two point distributions using a specified metric.

    Parameters
    ----------
    points1, points2 : array-like
        Point distributions to compare. Shape: (n_samples, n_features).
    metric : str, default="wasserstein"
        Distance metric to use:
        - "wasserstein": Wasserstein distance (Earth Mover's Distance)
        - "kolmogorov-smirnov": K-S statistic (max over dimensions)
        - "jensen-shannon": Jensen-Shannon divergence (histogram-based)
        - "euclidean": Euclidean distance between distribution centers
        - "mahalanobis": Mahalanobis distance between distributions
        - "cosine": Cosine similarity of mean vectors

    Returns
    -------
    float
        Distance or similarity value. Lower is more similar for distance metrics,
        higher is more similar for cosine (range [0, 1]).

    Notes
    -----
    - Wasserstein: measures "work" to transform one distribution to another.
    - K-S: maximum CDF difference; sensitive to shape/location differences.
    - Jensen-Shannon: symmetric KL divergence variant; bounded in [0, 1].
    - Euclidean: simple center-to-center distance.
    - Mahalanobis: accounts for covariance structure.
    - Cosine: direction similarity; invariant to scale.

    Examples
    --------
    >>> import numpy as np
    >>> p1 = np.random.randn(100, 3)
    >>> p2 = np.random.randn(100, 3) + 1.0
    >>> dist = compare_distributions(p1, p2, metric="wasserstein")
    """
    p1 = np.asarray(points1)
    p2 = np.asarray(points2)

    if p1.size == 0 or p2.size == 0:
        logger.warning("Empty distribution provided, returning NaN")
        return np.nan

    if p1.ndim == 1:
        p1 = p1.reshape(-1, 1)
    if p2.ndim == 1:
        p2 = p2.reshape(-1, 1)

    if p1.shape[1] != p2.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: {p1.shape[1]} vs {p2.shape[1]}"
        )

    logger.info(
        f"Comparing distributions with metric='{metric}': "
        f"p1.shape={p1.shape}, p2.shape={p2.shape}"
    )

    # Use match/case for cleaner metric dispatch
    match metric:
        case "wasserstein":
            result = wasserstein_distance_multi(p1, p2)
        case "kolmogorov-smirnov":
            result = kolmogorov_smirnov_distance(p1, p2)
        case "jensen-shannon":
            result = jensen_shannon_divergence(p1, p2)
        case "euclidean":
            result = float(euclidean_distance(np.mean(p1, axis=0), np.mean(p2, axis=0)))
        case "mahalanobis":
            # Mahalanobis between distribution centroids with pooled covariance
            mean1 = np.mean(p1, axis=0)
            mean2 = np.mean(p2, axis=0)
            cov1 = np.cov(p1, rowvar=False)
            cov2 = np.cov(p2, rowvar=False)
            pooled_cov = 0.5 * (cov1 + cov2) + np.eye(p1.shape[1]) * 1e-6  # regularize
            result = float(mahalanobis_distance(mean1, mean2, cov=pooled_cov))
        case "cosine":
            v1 = np.mean(p1, axis=0)
            v2 = np.mean(p2, axis=0)
            result = float(cosine_similarity(v1, v2))
        case _:
            raise ValueError(
                f"Unknown metric '{metric}'. Choose from: wasserstein, "
                "kolmogorov-smirnov, jensen-shannon, euclidean, mahalanobis, cosine."
            )

    logger.info(f"Distribution comparison result: {result:.6f}")
    return result


def compare_distribution_groups(
    group_vectors: dict[str | tuple, npt.NDArray],
    compare_type: Literal["inside", "between"] = "between",
    metric: str = "wasserstein",
) -> dict[str, npt.NDArray] | dict[str | tuple, npt.NDArray]:
    """Compare distributions within or between groups.

    Parameters
    ----------
    group_vectors : dict
        Dictionary mapping group identifiers to point arrays (n_samples, n_features).
    compare_type : {"inside", "between"}, default="between"
        - "inside": Compare each group to itself (self-similarity).
        - "between": Compare each group to all others.
    metric : str, default="wasserstein"
        Distance metric (see `compare_distributions` for options).

    Returns
    -------
    dict
        If compare_type == "inside":
            Returns {"mean": ndarray, "std": ndarray} with shape (n_groups,).
        If compare_type == "between":
            Returns {group_name: ndarray} where each array has shape (n_groups,)
            containing distances from that group to all others.

    Examples
    --------
    >>> groups = {
    ...     "A": np.random.randn(50, 3),
    ...     "B": np.random.randn(50, 3) + 1.0,
    ...     "C": np.random.randn(50, 3) + 2.0,
    ... }
    >>> similarities = compare_distribution_groups(
    ...     groups, compare_type="between", metric="wasserstein"
    ... )
    >>> similarities["A"]  # distances from A to all groups
    """
    n_groups = len(group_vectors)
    group_names = list(group_vectors.keys())

    logger.info(
        f"Comparing {n_groups} groups with compare_type='{compare_type}', metric='{metric}'"
    )

    # Use match/case for cleaner dispatch
    match compare_type:
        case "inside":
            # Within-group variability using new distance.py functions
            means = np.zeros(n_groups)
            stds = np.zeros(n_groups)

            for idx, (name, points) in enumerate(group_vectors.items()):
                if len(points) < 2:
                    logger.debug(f"Group '{name}' has <2 points, skipping")
                    means[idx] = 0.0
                    stds[idx] = 0.0
                    continue

                # Use unified distribution_distance from distance.py
                stats = distribution_distance(
                    points, mode="within", metric=metric, summary="all"  # type: ignore
                )
                means[idx] = stats["mean"]  # type: ignore
                stds[idx] = stats["std"]  # type: ignore

            logger.info(f"Within-group statistics computed: mean={means}, std={stds}")
            return {"mean": means, "std": stds}

        case "between":
            # Between-group distances
            similarities = {}
            for i, (name_i, group_i) in enumerate(group_vectors.items()):
                dists_to_all = np.zeros(n_groups)
                for j, (name_j, group_j) in enumerate(group_vectors.items()):
                    dist = compare_distributions(group_i, group_j, metric=metric)  # type: ignore
                    dists_to_all[j] = dist
                similarities[name_i] = dists_to_all
                logger.debug(
                    f"Group '{name_i}' distances to all: {dists_to_all}"
                )

            logger.info(f"Between-group distances computed for {n_groups} groups")
            return similarities

        case _:
            raise ValueError(
                f"Unknown compare_type '{compare_type}'. Choose 'inside' or 'between'."
            )
