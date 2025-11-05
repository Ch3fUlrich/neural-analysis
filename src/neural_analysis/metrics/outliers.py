"""Outlier detection utilities for neural data analysis.

This module provides functions for detecting and filtering outliers from point
distributions using various statistical methods.
"""

from __future__ import annotations

from typing import Literal
import logging

import numpy as np
import numpy.typing as npt
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor

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

__all__ = ["filter_outlier"]


@log_calls(level=logging.DEBUG)
def filter_outlier(
    points: npt.ArrayLike,
    method: Literal["iqr", "zscore", "isolation", "lof", "elliptic"] = "lof",
    contamination: float = 0.1,
    threshold: float = 3.0,
    return_mask: bool = False,
    parallel: bool = True,
) -> npt.NDArray | tuple[npt.NDArray, npt.NDArray]:
    """Filter outliers from a point distribution.

    Parameters
    ----------
    points : array-like
        Input points, shape (n_samples, n_features).
    method : {"iqr", "zscore", "isolation", "lof", "elliptic"}, default="lof"
        Outlier detection method:
        - "iqr": Interquartile range (per-feature, simple)
        - "zscore": Z-score threshold (per-feature)
        - "isolation": Isolation Forest (scikit-learn)
        - "lof": Local Outlier Factor (density-based, default)
        - "elliptic": Elliptic Envelope (Mahalanobis-based)
    contamination : float, default=0.1
        Expected proportion of outliers (for "isolation", "lof", "elliptic").
    threshold : float, default=3.0
        Threshold for "iqr" (multiplier) and "zscore" (std devs).
    return_mask : bool, default=False
        If True, return (filtered_points, mask). If False, return filtered_points.

    Returns
    -------
    filtered_points : ndarray
        Points with outliers removed, shape (n_inliers, n_features).
    mask : ndarray, optional
        Boolean mask (n_samples,) where True = inlier. Returned if return_mask=True.

    Examples
    --------
    >>> import numpy as np
    >>> points = np.vstack([np.random.randn(95, 3), np.random.randn(5, 3) * 10])
    >>> filtered = filter_outlier(points, method="lof", contamination=0.05)
    >>> filtered.shape[0] < points.shape[0]
    True

    Notes
    -----
    - "lof" is recommended for general use (density-based, works well in low/high dims).
    - "elliptic" assumes Gaussian distribution; good for symmetric clusters.
    - "iqr" and "zscore" are fast but univariate (apply per feature independently).
    - Minimum 10 points required; otherwise returns all points unchanged.
    """
    points_arr = np.asarray(points)
    n, d = points_arr.shape

    logger.info(
        f"Filtering outliers using method='{method}': n_samples={n}, n_features={d}, "
        f"contamination={contamination}, threshold={threshold}, parallel={parallel}"
    )

    if n < 10 and method in {"isolation", "lof", "elliptic"}:
        # Not enough points for model-based or neighborhood methods
        logger.warning(
            f"Not enough samples ({n}<10) for method '{method}', returning all points"
        )
        mask = np.ones(n, dtype=bool)
        return (points_arr, mask) if return_mask else points_arr

    # Dispatch to method-specific helpers
    if method == "iqr":
        mask = _mask_outliers_iqr(points_arr, threshold=threshold)
    elif method == "zscore":
        mask = _mask_outliers_zscore(points_arr, threshold=threshold)
    elif method == "isolation":
        mask = _mask_outliers_isolation(points_arr, contamination=contamination)
    elif method == "lof":
        mask = _mask_outliers_lof(points_arr, contamination=contamination)
    elif method == "elliptic":
        mask = _mask_outliers_elliptic(points_arr, contamination=contamination)
    else:
        raise ValueError(
            f"Unknown method '{method}'. Choose from: iqr, zscore, isolation, lof, elliptic."
        )

    filtered_points = points_arr[mask]
    n_removed = n - filtered_points.shape[0]
    logger.info(
        f"Outlier filtering complete: removed {n_removed}/{n} points ({100*n_removed/n:.1f}%)"
    )

    return (filtered_points, mask) if return_mask else filtered_points


# -----------------------------
# Method-specific helper masks
# -----------------------------

def _mask_outliers_iqr(points: np.ndarray, threshold: float = 1.5) -> np.ndarray:
    """Return boolean mask of inliers using IQR per feature (vectorized)."""
    # Compute Q1 and Q3 per feature
    q1 = np.percentile(points, 25, axis=0)
    q3 = np.percentile(points, 75, axis=0)
    iqr = q3 - q1
    lower = q1 - threshold * iqr
    upper = q3 + threshold * iqr
    # Broadcast and check all features are within bounds
    within = (points >= lower) & (points <= upper)
    return np.all(within, axis=1)


def _mask_outliers_zscore(points: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """Return boolean mask of inliers using robust Z-score (vectorized)."""
    med = np.median(points, axis=0)
    mad = np.median(np.abs(points - med), axis=0)
    # Scale MAD to approximate std for normal dist
    with np.errstate(divide="ignore", invalid="ignore"):
        robust_z = 0.67448975 * (points - med) / mad
    mask_mad = ~np.isnan(robust_z) & ~np.isinf(robust_z)
    # For columns where MAD==0, fall back to std-based z-score
    std = np.std(points, axis=0, ddof=0)
    fallback_cols = mad == 0
    if np.any(fallback_cols):
        with np.errstate(divide="ignore", invalid="ignore"):
            z_std = (points[:, fallback_cols] - np.mean(points[:, fallback_cols], axis=0)) / (
                std[fallback_cols] + 1e-12
            )
        # Build full robust_z combining fallback columns
        robust_z = np.where(
            np.broadcast_to(fallback_cols, robust_z.shape),
            z_std,
            robust_z,
        )
        mask_mad[:, fallback_cols] = True  # std-based values are valid numbers
    z_abs = np.abs(robust_z)
    # Points are inliers if all feature z-scores are below threshold
    return np.all((z_abs < threshold) & mask_mad, axis=1)


def _mask_outliers_isolation(points: np.ndarray, contamination: float) -> np.ndarray:
    """Return inlier mask using Isolation Forest (leverages n_jobs=-1)."""
    from sklearn.ensemble import IsolationForest

    detector = IsolationForest(
        contamination=contamination, random_state=42, n_jobs=-1
    )
    labels = detector.fit_predict(points)
    return labels != -1


def _mask_outliers_lof(points: np.ndarray, contamination: float) -> np.ndarray:
    """Return inlier mask using Local Outlier Factor.

    Note: scikit-learn's LOF does not expose n_jobs; performance relies on
    optimized neighbor searches under the hood.
    """
    n = points.shape[0]
    n_neighbors = min(20, max(2, n - 1))
    detector = LocalOutlierFactor(
        n_neighbors=n_neighbors, contamination=contamination
    )
    labels = detector.fit_predict(points)
    return labels != -1


def _mask_outliers_elliptic(points: np.ndarray, contamination: float) -> np.ndarray:
    """Return inlier mask using Elliptic Envelope; returns all if n <= d."""
    n, d = points.shape
    if n <= d:
        logger.warning(
            f"Not enough samples ({n}<={d}) for elliptic envelope, returning all points"
        )
        return np.ones(n, dtype=bool)
    detector = EllipticEnvelope(
        contamination=contamination, random_state=42, support_fraction=None
    )
    labels = detector.fit_predict(points)
    return labels != -1
