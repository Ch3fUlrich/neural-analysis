"""Outlier detection utilities for neural data analysis.

This module provides functions for detecting and filtering outliers from point
distributions using various statistical methods.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor

__all__ = ["filter_outlier"]


def filter_outlier(
    points: npt.ArrayLike,
    method: Literal["iqr", "zscore", "isolation", "lof", "elliptic"] = "lof",
    contamination: float = 0.1,
    threshold: float = 3.0,
    return_mask: bool = False,
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

    if n < 10 and method in {"isolation", "lof", "elliptic"}:
        # Not enough points for model-based or neighborhood methods
        mask = np.ones(n, dtype=bool)
        return (points_arr, mask) if return_mask else points_arr

    # IQR method (per-feature)
    if method == "iqr":
        mask = np.ones(n, dtype=bool)
        for feat_idx in range(d):
            q1 = np.percentile(points_arr[:, feat_idx], 25)
            q3 = np.percentile(points_arr[:, feat_idx], 75)
            iqr = q3 - q1
            lower = q1 - threshold * iqr
            upper = q3 + threshold * iqr
            mask &= (points_arr[:, feat_idx] >= lower) & (
                points_arr[:, feat_idx] <= upper
            )

    # Z-score method (per-feature)
    elif method == "zscore":
        mask = np.ones(n, dtype=bool)
        for feat_idx in range(d):
            col = points_arr[:, feat_idx]
            # Use a robust z-score (median/MAD) to avoid masking by extreme outliers
            med = np.median(col)
            mad = np.median(np.abs(col - med))
            if mad > 0:
                # Consistent with normal dist: scaled MAD (approx std)
                robust_z = 0.67448975 * (col - med) / mad
                z_scores = np.abs(robust_z)
                mask &= z_scores < threshold
            else:
                # Fall back to standard z-score if MAD==0 but std>0
                std = np.std(col)
                if std > 0:
                    z_scores = np.abs((col - np.mean(col)) / std)
                    mask &= z_scores < threshold
                # If both MAD and std are 0, all values equal -> keep all

    # Isolation Forest (scikit-learn)
    elif method == "isolation":
        from sklearn.ensemble import IsolationForest

        detector = IsolationForest(
            contamination=contamination, random_state=42, n_jobs=-1
        )
        labels = detector.fit_predict(points_arr)
        mask = labels != -1  # 1 = inlier, -1 = outlier

    # Local Outlier Factor (density-based)
    elif method == "lof":
        n_neighbors = min(20, n - 1)
        detector = LocalOutlierFactor(
            n_neighbors=n_neighbors, contamination=contamination
        )
        labels = detector.fit_predict(points_arr)
        mask = labels != -1

    # Elliptic Envelope (Mahalanobis-based)
    elif method == "elliptic":
        if n <= d:
            # Not enough samples for covariance estimation
            mask = np.ones(n, dtype=bool)
        else:
            detector = EllipticEnvelope(
                contamination=contamination, random_state=42, support_fraction=None
            )
            labels = detector.fit_predict(points_arr)
            mask = labels != -1

    else:
        raise ValueError(
            f"Unknown method '{method}'. Choose from: iqr, zscore, isolation, lof, elliptic."
        )

    filtered_points = points_arr[mask]

    if return_mask:
        return filtered_points, mask
    else:
        return filtered_points
