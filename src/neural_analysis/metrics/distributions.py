"""Distribution comparison utilities for neural data analysis.

This module provides functions for comparing probability distributions using
various statistical metrics. It includes both pairwise comparisons and
group-based comparisons with optional outlier filtering.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.stats import wasserstein_distance, ks_2samp, entropy
from scipy.spatial.distance import cdist

from .distance import euclidean_distance, mahalanobis_distance, cosine_similarity

__all__ = ["compare_distributions", "compare_distribution_groups"]


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
        return np.nan

    if p1.ndim == 1:
        p1 = p1.reshape(-1, 1)
    if p2.ndim == 1:
        p2 = p2.reshape(-1, 1)

    if p1.shape[1] != p2.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: {p1.shape[1]} vs {p2.shape[1]}"
        )

    if metric == "wasserstein":
        # Sum Wasserstein distance over all features
        distances = [
            wasserstein_distance(p1[:, i], p2[:, i]) for i in range(p1.shape[1])
        ]
        return float(np.sum(distances))

    elif metric == "kolmogorov-smirnov":
        # Max K-S statistic over dimensions
        ks_stats = [ks_2samp(p1[:, i], p2[:, i]).statistic for i in range(p1.shape[1])]
        return float(np.max(ks_stats))

    elif metric == "jensen-shannon":
        # Jensen-Shannon divergence via histograms
        hist1, hist2 = _points_to_histogram(p1, p2)
        m = 0.5 * (hist1 + hist2)
        js_div = 0.5 * (entropy(hist1, m) + entropy(hist2, m))
        return float(js_div)

    elif metric == "euclidean":
        return float(euclidean_distance(np.mean(p1, axis=0), np.mean(p2, axis=0)))

    elif metric == "mahalanobis":
        # Mahalanobis between distribution centroids
        mean1 = np.mean(p1, axis=0)
        mean2 = np.mean(p2, axis=0)
        # Use pooled covariance
        cov1 = np.cov(p1, rowvar=False)
        cov2 = np.cov(p2, rowvar=False)
        pooled_cov = 0.5 * (cov1 + cov2) + np.eye(p1.shape[1]) * 1e-6  # regularize
        return float(mahalanobis_distance(mean1, mean2, cov=pooled_cov))

    elif metric == "cosine":
        v1 = np.mean(p1, axis=0)
        v2 = np.mean(p2, axis=0)
        return float(cosine_similarity(v1, v2))

    else:
        raise ValueError(
            f"Unknown metric '{metric}'. Choose from: wasserstein, "
            "kolmogorov-smirnov, jensen-shannon, euclidean, mahalanobis, cosine."
        )


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

    if compare_type == "inside":
        # Within-group variability
        means = np.zeros(n_groups)
        stds = np.zeros(n_groups)

        for idx, (name, points) in enumerate(group_vectors.items()):
            # Pairwise distances within the group
            if len(points) < 2:
                means[idx] = 0.0
                stds[idx] = 0.0
                continue

            # Compute pairwise for first few dimensions if high-dim
            pairwise_dists = []
            for i in range(len(points)):
                for j in range(i + 1, len(points)):
                    dist = compare_distributions(
                        points[i : i + 1], points[j : j + 1], metric=metric
                    )
                    pairwise_dists.append(dist)

            means[idx] = np.mean(pairwise_dists) if pairwise_dists else 0.0
            stds[idx] = np.std(pairwise_dists) if pairwise_dists else 0.0

        return {"mean": means, "std": stds}

    elif compare_type == "between":
        # Between-group distances
        similarities = {}
        for i, (name_i, group_i) in enumerate(group_vectors.items()):
            dists_to_all = np.zeros(n_groups)
            for j, (name_j, group_j) in enumerate(group_vectors.items()):
                dist = compare_distributions(group_i, group_j, metric=metric)
                dists_to_all[j] = dist
            similarities[name_i] = dists_to_all

        return similarities

    else:
        raise ValueError(
            f"Unknown compare_type '{compare_type}'. Choose 'inside' or 'between'."
        )


def _points_to_histogram(
    points1: npt.NDArray, points2: npt.NDArray, bins: int = 50
) -> tuple[npt.NDArray, npt.NDArray]:
    """Convert point clouds to normalized histograms for comparison.

    Parameters
    ----------
    points1, points2 : ndarray
        Point distributions (n_samples, n_features).
    bins : int, default=50
        Number of bins per dimension.

    Returns
    -------
    hist1, hist2 : ndarray
        Flattened, normalized histograms.
    """
    # Determine common bin edges
    all_data = np.vstack([points1, points2])
    ranges = [(all_data[:, i].min(), all_data[:, i].max()) for i in range(all_data.shape[1])]

    # Compute multi-dimensional histograms
    hist1, _ = np.histogramdd(points1, bins=bins, range=ranges)
    hist2, _ = np.histogramdd(points2, bins=bins, range=ranges)

    # Flatten and normalize
    hist1 = hist1.ravel() + 1e-10  # avoid zeros
    hist2 = hist2.ravel() + 1e-10
    hist1 /= hist1.sum()
    hist2 /= hist2.sum()

    return hist1, hist2
