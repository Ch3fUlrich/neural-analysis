"""
Similarity and correlation metrics for neural data analysis.

This module provides functions for computing various similarity measures between
data points or features, including correlation coefficients, cosine similarity,
and other similarity metrics commonly used in neural data analysis.

Functions are organized by the type of similarity measure:
- Correlation: Pearson, Spearman, Kendall
- Geometric: Cosine similarity, angular distance
- Matrix operations: Correlation matrices, similarity matrices

The module supports optional numba acceleration for improved performance on
large matrices. When numba is available, parallel computation can be enabled
to significantly speed up correlation calculations.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.stats import kendalltau, spearmanr

# Optional numba acceleration
try:
    import numba

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    numba = None

logger = logging.getLogger(__name__)

if NUMBA_AVAILABLE:
    logger.debug("Numba acceleration available for similarity computations")

__all__ = [
    "correlation_matrix",
    "cosine_similarity_matrix",
    "angular_similarity_matrix",
    "similarity_matrix",
]


def correlation_matrix(
    data: npt.ArrayLike,
    method: Literal["pearson", "spearman", "kendall"] = "pearson",
) -> np.ndarray:
    """
    Compute pairwise correlation matrix between features.

    Parameters
    ----------
    data : array-like, shape (n_samples, n_features)
        Data matrix where each column is a feature/variable.
    method : {"pearson", "spearman", "kendall"}, default="pearson"
        Correlation method to use:
        - "pearson": Linear correlation (assumes normality)
        - "spearman": Rank-based correlation (non-parametric)
        - "kendall": Tau correlation (ordinal associations)

    Returns
    -------
    corr_matrix : ndarray, shape (n_features, n_features)
        Symmetric correlation matrix with values in [-1, 1].
        Diagonal elements are 1.0 (self-correlation).

    Raises
    ------
    ValueError
        If data is not 2D or method is unknown.

    Examples
    --------
    >>> import numpy as np
    >>> # Generate correlated data
    >>> rng = np.random.default_rng(42)
    >>> x = rng.normal(0, 1, 100)
    >>> y = 0.8 * x + 0.2 * rng.normal(0, 1, 100)
    >>> data = np.column_stack([x, y, rng.normal(0, 1, 100)])
    >>>
    >>> # Compute Pearson correlation
    >>> corr = correlation_matrix(data, method='pearson')
    >>> print(f"Correlation between features 0 and 1: {corr[0, 1]:.3f}")

    >>> # Compare with Spearman (rank-based)
    >>> corr_spearman = correlation_matrix(data, method='spearman')

    Notes
    -----
    - Pearson: Measures linear relationships, sensitive to outliers
    - Spearman: Measures monotonic relationships, robust to outliers
    - Kendall: Similar to Spearman but different null hypothesis

    For neural data:
    - Use Pearson for normally distributed firing rates
    - Use Spearman for non-linear tuning curves
    - Use Kendall for ordinal trial rankings
    """
    data_arr = np.asarray(data)
    if data_arr.ndim != 2:
        raise ValueError(f"data must be 2D, got shape {data_arr.shape}")

    logger.info(
        f"Computing {method} correlation matrix for {data_arr.shape[1]} features"
    )

    match method:
        case "pearson":
            # Use numpy's corrcoef for Pearson (faster, well-tested)
            corr_matrix = np.corrcoef(data_arr.T)
        case "spearman":
            # Spearman rank correlation
            corr_matrix, _ = spearmanr(data_arr, axis=0)
            # Handle scalar return for 2 features
            if data_arr.shape[1] == 2 and np.ndim(corr_matrix) == 0:
                corr_matrix = np.array([[1.0, corr_matrix], [corr_matrix, 1.0]])
        case "kendall":
            # Kendall tau correlation (slower, use for small datasets)
            n_features = data_arr.shape[1]
            corr_matrix = np.eye(n_features)
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    tau, _ = kendalltau(data_arr[:, i], data_arr[:, j])
                    corr_matrix[i, j] = tau
                    corr_matrix[j, i] = tau
        case _:
            raise ValueError(
                f"Unknown method '{method}'. Choose 'pearson', 'spearman', or 'kendall'."
            )

    return corr_matrix


def cosine_similarity_matrix(
    data: npt.ArrayLike,
    centered: bool = False,
) -> np.ndarray:
    """
    Compute pairwise cosine similarity matrix between features.

    Cosine similarity measures the cosine of the angle between two vectors,
    ranging from -1 (opposite) to 1 (identical direction).

    Parameters
    ----------
    data : array-like, shape (n_samples, n_features)
        Data matrix where each column is a feature/variable.
    centered : bool, default=False
        If True, center data before computing (equivalent to Pearson correlation).
        If False, compute raw cosine similarity.

    Returns
    -------
    similarity_matrix : ndarray, shape (n_features, n_features)
        Symmetric similarity matrix with values in [-1, 1].
        Diagonal elements are 1.0 (self-similarity).

    Examples
    --------
    >>> import numpy as np
    >>> # Generate sparse firing patterns
    >>> data = np.array([[1, 0, 0, 1],
    ...                  [0, 1, 1, 0],
    ...                  [1, 0, 1, 0]]).T
    >>>
    >>> # Compute cosine similarity
    >>> sim = cosine_similarity_matrix(data)
    >>> print(f"Similarity between patterns 0 and 2: {sim[0, 2]:.3f}")

    Notes
    -----
    - Cosine similarity is scale-invariant (ignores magnitude)
    - Useful for sparse data (e.g., spike counts)
    - If centered=True, equivalent to Pearson correlation
    - More robust than Euclidean distance for high-dimensional data

    For neural data:
    - Use for comparing tuning curve shapes
    - Use for sparse population activity patterns
    - Use centered=False for directional similarity
    """
    data_arr = np.asarray(data)
    if data_arr.ndim != 2:
        raise ValueError(f"data must be 2D, got shape {data_arr.shape}")

    logger.info(
        f"Computing cosine similarity matrix for {data_arr.shape[1]} features "
        f"(centered={centered})"
    )

    # Center data if requested (makes it equivalent to correlation)
    if centered:
        data_arr = data_arr - data_arr.mean(axis=0, keepdims=True)

    # Normalize columns to unit vectors
    norms = np.linalg.norm(data_arr, axis=0, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms == 0, 1, norms)
    data_normalized = data_arr / norms

    # Compute similarity as dot product of normalized vectors
    similarity_matrix = data_normalized.T @ data_normalized

    return similarity_matrix


def angular_similarity_matrix(
    data: npt.ArrayLike,
) -> np.ndarray:
    """
    Compute pairwise angular similarity (1 - angular distance / π).

    Angular distance is the angle between vectors in radians. This function
    converts it to a similarity measure in [0, 1] where 1 = identical direction.

    Parameters
    ----------
    data : array-like, shape (n_samples, n_features)
        Data matrix where each column is a feature/variable.

    Returns
    -------
    similarity_matrix : ndarray, shape (n_features, n_features)
        Symmetric similarity matrix with values in [0, 1].
        - 1.0: vectors point in same direction (angle = 0)
        - 0.5: vectors are orthogonal (angle = π/2)
        - 0.0: vectors point in opposite directions (angle = π)

    Examples
    --------
    >>> import numpy as np
    >>> # Create vectors at different angles
    >>> v1 = np.array([1, 0])
    >>> v2 = np.array([1, 1]) / np.sqrt(2)  # 45 degrees
    >>> v3 = np.array([0, 1])  # 90 degrees
    >>> data = np.column_stack([v1, v2, v3])
    >>>
    >>> sim = angular_similarity_matrix(data.T)
    >>> print(f"Angular similarity (0° vs 45°): {sim[0, 1]:.3f}")
    >>> print(f"Angular similarity (0° vs 90°): {sim[0, 2]:.3f}")

    Notes
    -----
    - Related to cosine similarity: angle = arccos(cosine_similarity)
    - Scale-invariant (only direction matters)
    - Useful for directional tuning curves

    For neural data:
    - Use for preferred direction analysis
    - Use for population activity states on a hypersphere
    - Naturally bounded in [0, 1] unlike cosine similarity
    """
    data_arr = np.asarray(data)
    if data_arr.ndim != 2:
        raise ValueError(f"data must be 2D, got shape {data_arr.shape}")

    logger.info(f"Computing angular similarity matrix for {data_arr.shape[1]} features")

    # Get cosine similarity
    cosine_sim = cosine_similarity_matrix(data_arr, centered=False)

    # Clip to valid range for arccos (numerical stability)
    cosine_sim = np.clip(cosine_sim, -1.0, 1.0)

    # Convert to angular distance, then to similarity
    angular_distance = np.arccos(cosine_sim)
    angular_similarity = 1.0 - (angular_distance / np.pi)

    return angular_similarity


def similarity_matrix(
    data: npt.ArrayLike,
    method: Literal["pearson", "spearman", "kendall", "cosine", "angular"] = "pearson",
    centered: bool = False,
    parallel: bool = False,
    plot: bool = False,
    plot_config: dict | None = None,
) -> np.ndarray:
    """
    Compute pairwise similarity matrix using specified method.

    Unified interface for computing various similarity measures between features.
    Optionally visualizes results as a heatmap.

    Parameters
    ----------
    data : array-like, shape (n_samples, n_features)
        Data matrix where each column is a feature/variable.
    method : {"pearson", "spearman", "kendall", "cosine", "angular"}, default="pearson"
        Similarity method to use:
        - "pearson": Pearson correlation (linear relationships)
        - "spearman": Spearman rank correlation (monotonic relationships)
        - "kendall": Kendall tau correlation (ordinal associations)
        - "cosine": Cosine similarity (directional similarity)
        - "angular": Angular similarity (1 - angle/π, bounded [0,1])
    centered : bool, default=False
        Only used for "cosine" method. If True, center data before computing
        (equivalent to Pearson correlation).
    parallel : bool, default=False
        If True, use parallel/optimized computation for large matrices.
        Uses numba JIT compilation when available.
    plot : bool, default=False
        If True, display similarity matrix as heatmap after computation.
    plot_config : dict, optional
        Configuration parameters for plotting (title, figsize, cmap, etc.).
        Keys can include: 'title', 'figsize', 'cmap', 'backend', etc.

    Returns
    -------
    similarity : ndarray, shape (n_features, n_features)
        Symmetric similarity matrix.
        - Correlation methods: values in [-1, 1]
        - Cosine similarity: values in [-1, 1]
        - Angular similarity: values in [0, 1]

    Raises
    ------
    ValueError
        If data is not 2D or method is unknown.

    Examples
    --------
    >>> import numpy as np
    >>> # Generate correlated neural activity
    >>> rng = np.random.default_rng(42)
    >>> data = rng.normal(0, 1, (100, 10))
    >>>
    >>> # Compute Pearson correlation
    >>> sim = similarity_matrix(data, method='pearson')
    >>>
    >>> # Compute cosine similarity with visualization
    >>> sim = similarity_matrix(data, method='cosine', plot=True,
    ...                         plot_config={'title': 'Neural Similarity'})
    >>>
    >>> # Use parallel computation for large dataset
    >>> large_data = rng.normal(0, 1, (1000, 500))
    >>> sim = similarity_matrix(large_data, method='pearson', parallel=True)

    Notes
    -----
    Method Selection Guide:
    - Use "pearson" for linear relationships in normally distributed data
    - Use "spearman" for non-linear monotonic relationships
    - Use "kendall" for ordinal data or small samples
    - Use "cosine" for directional similarity (scale-invariant)
    - Use "angular" for bounded [0,1] similarity scores

    Parallel computation is recommended for:
    - Large matrices (>500 features)
    - Repeated computations
    - Real-time analysis pipelines

    See Also
    --------
    correlation_matrix : Compute correlation matrices
    cosine_similarity_matrix : Compute cosine similarity
    angular_similarity_matrix : Compute angular similarity
    """
    data_arr = np.asarray(data)
    if data_arr.ndim != 2:
        raise ValueError(f"data must be 2D, got shape {data_arr.shape}")

    logger.info(
        f"Computing {method} similarity matrix for {data_arr.shape[1]} features "
        f"(parallel={parallel})"
    )

    # Dispatch to appropriate method
    match method:
        case "pearson" | "spearman" | "kendall":
            if parallel:
                similarity = _correlation_matrix_parallel(data_arr, method)
            else:
                similarity = correlation_matrix(data_arr, method=method)
        case "cosine":
            if parallel:
                similarity = _cosine_similarity_matrix_parallel(data_arr, centered)
            else:
                similarity = cosine_similarity_matrix(data_arr, centered=centered)
        case "angular":
            if parallel:
                similarity = _angular_similarity_matrix_parallel(data_arr)
            else:
                similarity = angular_similarity_matrix(data_arr)
        case _:
            raise ValueError(
                f"Unknown method '{method}'. Choose from: "
                "'pearson', 'spearman', 'kendall', 'cosine', 'angular'"
            )

    # Plot if requested
    if plot:
        _plot_similarity_matrix(similarity, method, plot_config)

    return similarity


def _plot_similarity_matrix(
    similarity: np.ndarray,
    method: str,
    plot_config: dict | None,
) -> None:
    """
    Plot similarity matrix as heatmap.

    Parameters
    ----------
    similarity : ndarray
        Similarity matrix to plot.
    method : str
        Method name for title.
    plot_config : dict, optional
        Plot configuration parameters.
    """
    try:
        # Lazy import to avoid circular dependency
        from neural_analysis.plotting import PlotConfig, plot_heatmap
    except ImportError:
        logger.warning("Could not import plotting functions. Skipping visualization.")
        return

    # Set defaults and merge with user config
    config_dict = {
        "title": f"{method.capitalize()} Similarity Matrix",
        "figsize": (8, 7),
        "cmap": "RdBu_r" if method in ["pearson", "spearman", "kendall"] else "viridis",
    }

    if plot_config is not None:
        config_dict.update(plot_config)

    config = PlotConfig(**config_dict)

    # Create heatmap
    plot_heatmap(
        similarity,
        config=config,
        show_values=similarity.shape[0] <= 15,  # Only show values for small matrices
        colorbar=True,
        colorbar_label="Similarity",
        aspect="equal",
    )


# =============================================================================
# Parallel Computation Functions
# =============================================================================


def _correlation_matrix_parallel(
    data: np.ndarray,
    method: str,
) -> np.ndarray:
    """
    Compute correlation matrix with parallel optimization.

    For Pearson correlation, uses numpy's highly optimized corrcoef.
    For Spearman and Kendall, uses numba-accelerated implementations if available,
    otherwise falls back to scipy.

    Parameters
    ----------
    data : ndarray
        Data matrix, shape (n_samples, n_features).
    method : str
        Correlation method: 'pearson', 'spearman', or 'kendall'.

    Returns
    -------
    ndarray
        Correlation matrix, shape (n_features, n_features).
    """
    if method == "pearson":
        # NumPy's corrcoef is already highly optimized with BLAS
        return np.corrcoef(data.T)
    elif method == "spearman":
        if NUMBA_AVAILABLE:
            return _spearman_numba(data)
        else:
            logger.debug("Numba not available, using scipy for Spearman correlation")
            return correlation_matrix(data, method="spearman")
    elif method == "kendall":
        if NUMBA_AVAILABLE:
            return _kendall_numba(data)
        else:
            logger.debug("Numba not available, using scipy for Kendall correlation")
            return correlation_matrix(data, method="kendall")
    else:
        raise ValueError(f"Unknown method: {method}")


def _cosine_similarity_matrix_parallel(
    data: np.ndarray,
    centered: bool,
) -> np.ndarray:
    """
    Compute cosine similarity with parallel optimization.

    Uses optimized BLAS operations through numpy. The matrix multiplication
    automatically leverages multi-threaded BLAS libraries (OpenBLAS, MKL)
    when available, providing significant speedup for large matrices.

    Parameters
    ----------
    data : ndarray
        Data matrix, shape (n_samples, n_features).
    centered : bool
        If True, center data before computing similarity.

    Returns
    -------
    ndarray
        Cosine similarity matrix, shape (n_features, n_features).
    """
    # Center if requested
    if centered:
        data = data - data.mean(axis=0, keepdims=True)

    # Normalize columns to unit vectors
    norms = np.linalg.norm(data, axis=0, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
    data_normalized = data / norms

    # Compute similarity via matrix multiplication (uses optimized BLAS)
    return data_normalized.T @ data_normalized


def _angular_similarity_matrix_parallel(
    data: np.ndarray,
) -> np.ndarray:
    """
    Compute angular similarity with parallel optimization.

    Leverages optimized cosine similarity computation and applies
    angular transformation: similarity = 1 - arccos(cosine) / π

    Parameters
    ----------
    data : ndarray
        Data matrix, shape (n_samples, n_features).

    Returns
    -------
    ndarray
        Angular similarity matrix, shape (n_features, n_features).
        Values in [0, 1] where 1 = identical direction.
    """
    cosine_sim = _cosine_similarity_matrix_parallel(data, centered=False)
    cosine_sim = np.clip(cosine_sim, -1.0, 1.0)  # Numerical stability
    angular_distance = np.arccos(cosine_sim)
    return 1.0 - (angular_distance / np.pi)


# =============================================================================
# Numba-Accelerated Implementations
# =============================================================================

if NUMBA_AVAILABLE:

    @numba.jit(nopython=True, parallel=True, cache=True)
    def _rank_data_numba(data: np.ndarray) -> np.ndarray:
        """
        Compute ranks for each feature column using numba acceleration.

        This function ranks the values in each column independently, which is
        required for Spearman correlation. Uses parallel loops for efficiency.

        Parameters
        ----------
        data : ndarray
            Data matrix, shape (n_samples, n_features).

        Returns
        -------
        ndarray
            Ranked data matrix, shape (n_samples, n_features).
            Ranks start from 1 (not 0) following convention.
        """
        n_samples, n_features = data.shape
        ranks = np.empty_like(data)

        for j in numba.prange(n_features):
            # Get indices that would sort the column
            sorted_idx = np.argsort(data[:, j])
            # Assign ranks (1-indexed)
            for i, idx in enumerate(sorted_idx):
                ranks[idx, j] = i + 1

        return ranks

    @numba.jit(nopython=True, parallel=True, cache=True)
    def _kendall_tau_pairwise(x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute Kendall's tau correlation between two arrays.

        Kendall's tau measures ordinal association based on concordant and
        discordant pairs. This implementation uses O(n²) algorithm with
        numba acceleration.

        Parameters
        ----------
        x, y : ndarray
            1D arrays of equal length.

        Returns
        -------
        float
            Kendall's tau correlation coefficient in [-1, 1].
        """
        n = len(x)
        concordant = 0
        discordant = 0

        for i in range(n):
            for j in range(i + 1, n):
                # Sign of differences
                sign_x = np.sign(x[j] - x[i])
                sign_y = np.sign(y[j] - y[i])
                prod = sign_x * sign_y

                if prod > 0:
                    concordant += 1
                elif prod < 0:
                    discordant += 1
                # If prod == 0, it's a tie (not counted)

        # Tau = (concordant - discordant) / total_pairs
        total_pairs = n * (n - 1) / 2
        return (concordant - discordant) / total_pairs

    def _spearman_numba(data: np.ndarray) -> np.ndarray:
        """
        Compute Spearman correlation matrix using numba-accelerated ranking.

        Spearman correlation is Pearson correlation applied to ranks.
        This implementation ranks data in parallel, then uses numpy's
        optimized corrcoef on the ranked data.

        Parameters
        ----------
        data : ndarray
            Data matrix, shape (n_samples, n_features).

        Returns
        -------
        ndarray
            Spearman correlation matrix, shape (n_features, n_features).
        """
        ranks = _rank_data_numba(data)
        return np.corrcoef(ranks.T)

    def _kendall_numba(data: np.ndarray) -> np.ndarray:
        """
        Compute Kendall correlation matrix using numba acceleration.

        Computes pairwise Kendall's tau for all feature pairs.
        Uses parallel loops to compute upper triangle, then mirrors
        to lower triangle.

        Parameters
        ----------
        data : ndarray
            Data matrix, shape (n_samples, n_features).

        Returns
        -------
        ndarray
            Kendall correlation matrix, shape (n_features, n_features).
        """
        n_features = data.shape[1]
        corr_matrix = np.eye(n_features)

        # Compute upper triangle in parallel
        for i in numba.prange(n_features):
            for j in range(i + 1, n_features):
                tau = _kendall_tau_pairwise(data[:, i], data[:, j])
                corr_matrix[i, j] = tau
                corr_matrix[j, i] = tau  # Symmetric

        return corr_matrix

else:
    # Fallback implementations when numba is not available
    def _spearman_numba(data: np.ndarray) -> np.ndarray:
        """
        Fallback Spearman correlation without numba.
        Uses scipy's spearmanr function.
        """
        corr_matrix, _ = spearmanr(data, axis=0)
        # Handle scalar return for 2 features
        if data.shape[1] == 2 and np.ndim(corr_matrix) == 0:
            corr_matrix = np.array([[1.0, corr_matrix], [corr_matrix, 1.0]])
        return corr_matrix

    def _kendall_numba(data: np.ndarray) -> np.ndarray:
        """
        Fallback Kendall correlation without numba.
        Uses the standard (non-parallel) implementation.
        """
        return correlation_matrix(data, method="kendall")
