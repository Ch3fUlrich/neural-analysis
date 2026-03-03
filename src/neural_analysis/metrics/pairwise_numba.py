"""Numba-accelerated pairwise computation functions.

Provides JIT-compiled distance, correlation, and similarity functions
using Numba. Falls back to NumPy/SciPy when Numba is unavailable.
"""

from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from scipy.stats import kendalltau, spearmanr

try:
    from neural_analysis.utils.logging import get_logger
except Exception:

    def get_logger(name: str) -> logging.Logger:  # type: ignore[misc]
        return logging.getLogger(name)


logger = get_logger(__name__)

# Optional numba support
try:
    import numba

    NUMBA_AVAILABLE = True
except Exception:
    numba = None
    NUMBA_AVAILABLE = False

# ---------------------------------------------------------------------------
# Numba-accelerated pairwise distance helpers
# ---------------------------------------------------------------------------
try:
    from numba import njit, prange

    @njit(parallel=True, fastmath=True)  # type: ignore[misc]
    def _pairwise_euclidean_numba(
        x_arr: npt.NDArray[np.float64], y_arr: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        n_x, n_feat = x_arr.shape
        n_y, _ = y_arr.shape
        out = np.empty((n_x, n_y), dtype=np.float64)
        for i in prange(n_x):
            for j in range(n_y):
                s = 0.0
                for k in range(n_feat):
                    d = x_arr[i, k] - y_arr[j, k]
                    s += d * d
                out[i, j] = np.sqrt(s)
        return out

    @njit(parallel=True, fastmath=True)  # type: ignore[misc]
    def _pairwise_cosine_numba(
        x_arr: npt.NDArray[np.float64], y_arr: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        n_x, n_feat = x_arr.shape
        n_y, _ = y_arr.shape
        out = np.empty((n_x, n_y), dtype=np.float64)
        for i in prange(n_x):
            norm_x = 0.0
            for k in range(n_feat):
                norm_x += x_arr[i, k] ** 2
            norm_x = np.sqrt(norm_x)

            for j in range(n_y):
                dot_prod = 0.0
                norm_y = 0.0
                for k in range(n_feat):
                    dot_prod += x_arr[i, k] * y_arr[j, k]
                    norm_y += y_arr[j, k] ** 2
                norm_y = np.sqrt(norm_y)

                if norm_x > 0 and norm_y > 0:
                    out[i, j] = dot_prod / (norm_x * norm_y)
                else:
                    out[i, j] = 0.0
        return out

    @njit(parallel=True, fastmath=True)  # type: ignore[misc]
    def _pairwise_manhattan_numba(
        x_arr: npt.NDArray[np.float64], y_arr: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        n_x, n_feat = x_arr.shape
        n_y, _ = y_arr.shape
        out = np.empty((n_x, n_y), dtype=np.float64)
        for i in prange(n_x):
            for j in range(n_y):
                s = 0.0
                for k in range(n_feat):
                    s += abs(x_arr[i, k] - y_arr[j, k])
                out[i, j] = s
        return out

except Exception:
    pass


# ---------------------------------------------------------------------------
# Numba-accelerated ranking and correlation helpers
# ---------------------------------------------------------------------------
if NUMBA_AVAILABLE:

    @numba.jit(nopython=True, parallel=True, cache=True)  # type: ignore[misc]
    def _rank_data_numba(data: npt.NDArray[Any]) -> npt.NDArray[Any]:
        n_samples, n_features = data.shape
        ranks = np.empty_like(data)
        for j in numba.prange(n_features):
            sorted_idx = np.argsort(data[:, j])
            for i, idx in enumerate(sorted_idx):
                ranks[idx, j] = i + 1
        return ranks

    @numba.jit(nopython=True, parallel=True, cache=True)  # type: ignore[misc]
    def _kendall_tau_pairwise(x: npt.NDArray[Any], y: npt.NDArray[Any]) -> float:
        n = len(x)
        concordant = 0
        discordant = 0
        for i in range(n):
            for j in range(i + 1, n):
                sign_x = np.sign(x[j] - x[i])
                sign_y = np.sign(y[j] - y[i])
                prod = sign_x * sign_y
                if prod > 0:
                    concordant += 1
                elif prod < 0:
                    discordant += 1
        total_pairs = n * (n - 1) / 2
        return (concordant - discordant) / total_pairs

    def _spearman_numba(data: npt.NDArray[Any]) -> npt.NDArray[Any]:
        ranks = _rank_data_numba(data)
        return np.corrcoef(ranks.T)

    def _kendall_numba(data: npt.NDArray[Any]) -> npt.NDArray[Any]:
        n_features = data.shape[1]
        corr_matrix = np.eye(n_features)
        for i in numba.prange(n_features):
            for j in range(i + 1, n_features):
                tau = _kendall_tau_pairwise(data[:, i], data[:, j])
                corr_matrix[i, j] = tau
                corr_matrix[j, i] = tau
        return corr_matrix

else:

    def _spearman_numba(data: npt.NDArray[Any]) -> npt.NDArray[Any]:
        corr_matrix, _ = spearmanr(data, axis=0)
        if data.shape[1] == 2 and np.ndim(corr_matrix) == 0:
            corr_matrix = np.array([[1.0, corr_matrix], [corr_matrix, 1.0]])
        return cast("npt.NDArray[Any]", corr_matrix)

    def _kendall_numba(data: npt.NDArray[Any]) -> npt.NDArray[Any]:
        n_features = data.shape[1]
        corr_matrix = np.eye(n_features)
        for i in range(n_features):
            for j in range(i + 1, n_features):
                tau, _ = kendalltau(data[:, i], data[:, j])
                corr_matrix[i, j] = tau
                corr_matrix[j, i] = tau
        return corr_matrix


def _correlation_matrix_parallel(
    data: npt.NDArray[Any], method: str
) -> npt.NDArray[Any]:
    if method == "pearson":
        return np.corrcoef(data.T)
    elif method == "spearman":
        if not NUMBA_AVAILABLE:
            logger.debug("Numba not available, using scipy for Spearman")
        return _spearman_numba(data)
    elif method == "kendall":
        if not NUMBA_AVAILABLE:
            logger.debug("Numba not available, using scipy for Kendall")
        return _kendall_numba(data)
    else:
        raise ValueError(
            f"Unknown correlation method. Expected: 'pearson', 'spearman', "
            f"or 'kendall'. Got: {method!r}"
        )
