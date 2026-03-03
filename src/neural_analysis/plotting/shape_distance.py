"""Visualization utilities for shape distance metrics.

This module provides functions to visualize shape distance matrices using
MDS (Multidimensional Scaling) and related embedding techniques.

All distance computation should be done using `compare_datasets` from
`neural_analysis.metrics.pairwise_metrics`. This module only handles visualization.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from sklearn.decomposition import PCA

from neural_analysis.embeddings.dimensionality_reduction import compute_embedding
from neural_analysis.plotting.core import PlotConfig
from neural_analysis.plotting.grid_config import (
    GridLayoutConfig,
    PlotGrid,
    PlotSpec,
)


def embed_mds(
    distance_matrix: npt.NDArray[np.float64], n_components: int = 2, seed: int = 0
) -> npt.NDArray[np.float64]:
    """Embed distance matrix using MDS.

    Uses `compute_embedding` from `embeddings.dimensionality_reduction`.

    Parameters
    ----------
    distance_matrix : ndarray of shape (n_samples, n_samples)
        Pairwise distance matrix.
    n_components : int, default=2
        Number of dimensions for MDS embedding.
    seed : int, default=0
        Random seed for reproducibility.

    Returns
    -------
    embedding : ndarray of shape (n_samples, n_components)
        MDS embedding coordinates.
    """
    embedding = compute_embedding(
        distance_matrix,
        method="mds",
        n_components=n_components,
        metric="precomputed",
        random_state=seed,
    )
    # Ensure float64 dtype for return type
    return embedding.astype(np.float64)


def embed_mds_pca(
    distance_matrix: npt.NDArray[np.float64],
    mds_dim: int = 20,
    pca_dim: int = 2,
    seed: int = 0,
) -> npt.NDArray[np.float64]:
    """Embed distance matrix using MDS followed by PCA.

    Uses `compute_embedding` for MDS, then applies PCA.

    Parameters
    ----------
    distance_matrix : ndarray of shape (n_samples, n_samples)
        Pairwise distance matrix.
    mds_dim : int, default=20
        Number of dimensions for initial MDS embedding.
        Automatically reduced if n_samples < mds_dim.
    pca_dim : int, default=2
        Number of dimensions for PCA reduction.
    seed : int, default=0
        Random seed for reproducibility.

    Returns
    -------
    embedding : ndarray of shape (n_samples, pca_dim)
        MDS+PCA embedding coordinates.
    """
    n_samples = distance_matrix.shape[0]
    # Ensure mds_dim doesn't exceed n_samples
    actual_mds_dim = min(mds_dim, n_samples - 1)  # MDS needs at least n_samples-1
    if actual_mds_dim < mds_dim:
        import warnings

        warnings.warn(
            f"Reducing mds_dim from {mds_dim} to {actual_mds_dim} "
            f"because n_samples={n_samples}",
            UserWarning,
            stacklevel=2,
        )

    Z = embed_mds(distance_matrix, n_components=actual_mds_dim, seed=seed)
    # Ensure pca_dim doesn't exceed the MDS embedding dimension
    actual_pca_dim = min(pca_dim, Z.shape[1])
    pca = PCA(n_components=actual_pca_dim, random_state=seed)
    result = pca.fit_transform(Z)
    # Ensure float64 dtype for return type
    result_float64: npt.NDArray[np.float64] = result.astype(np.float64)
    return result_float64


def plot_shape_distance_mds(
    distance_matrices: dict[str, npt.NDArray[np.float64]] | None = None,
    datasets: list[npt.NDArray[np.float64]] | None = None,
    methods: list[Literal["procrustes", "one-to-one", "soft-matching"]] | None = None,
    labels: npt.NDArray[np.int_] | None = None,
    backend: Literal["matplotlib", "plotly"] = "matplotlib",
    figsize: tuple[float, float] = (12, 12),
    save_path: str | None = None,
    regenerate: bool = False,
    show_progress: bool = True,
    **metric_kwargs: Any,
) -> Any:
    """Plot MDS embeddings for multiple distance matrices using PlotGrid.

    Creates a grid of MDS and MDS+PCA visualizations for each distance matrix,
    with optional cluster-based coloring.

    This function can either:
    1. Accept pre-computed distance matrices (via `distance_matrices`)
    2. Compute distance matrices from datasets using `compare_datasets`
       (via `datasets` and `methods`)

    Parameters
    ----------
    distance_matrices : dict, optional
        Dictionary mapping method names to distance matrices.
        Each matrix should be of shape (n_datasets, n_datasets).
        If provided, `datasets` and `methods` are ignored.
    datasets : list of ndarray, optional
        List of datasets, each of shape (n_neurons_i, n_features).
        Required if `distance_matrices` is not provided.
        Used with `methods` to compute distances via `compare_datasets`.
    methods : list of str, optional
        List of shape distance methods to compute.
        Valid values: "procrustes", "one-to-one", "soft-matching".
        Required if `distance_matrices` is not provided.
        Used with `datasets` to compute distances via `compare_datasets`.
    labels : ndarray of shape (n_datasets,) or None, default=None
        Cluster labels for coloring points. If None, all points are same color.
    backend : {'matplotlib', 'plotly'}, default='matplotlib'
        Plotting backend to use.
    figsize : tuple of float, default=(12, 12)
        Figure size (width, height) in inches.
    save_path : str or None, optional
        Path to HDF5 file for automatic result caching when computing distances.
        Only used when `datasets` and `methods` are provided.
    regenerate : bool, default=False
        Force recomputation even if cached result exists.
        Only used when `datasets` and `methods` are provided.
    show_progress : bool, default=True
        Show progress bar during distance computation.
        Only used when `datasets` and `methods` are provided.
    **metric_kwargs
        Additional keyword arguments passed to `compare_datasets` when computing
        distances (e.g., metric="sqeuclidean", max_neurons=30).
        Only used when `datasets` and `methods` are provided.

    Returns
    -------
    fig
        Figure object from the plotting backend.

    Examples
    --------
    **Using pre-computed distance matrices**:

    >>> from neural_analysis.plotting.shape_distance import plot_shape_distance_mds
    >>> from neural_analysis.metrics.pairwise_metrics import compare_datasets
    >>> datasets = [np.random.randn(50, 10) for _ in range(20)]
    >>> labels = np.random.randint(0, 3, 20)
    >>>
    >>> # Compute distance matrices using compare_datasets
    >>> distance_matrices = {}
    >>> datasets_dict = {str(i): d for i, d in enumerate(datasets)}
    >>> for method in ["procrustes", "one-to-one", "soft-matching"]:
    ...     result = compare_datasets(
    ...         datasets_dict, mode="all-pairs", metric=method, **metric_kwargs
    ...     )
    ...     # Convert dict result to symmetric matrix
    ...     n = len(datasets)
    ...     D = np.zeros((n, n))
    ...     for i in range(n):
    ...         for j in range(n):
    ...             D[i, j] = result[str(i)][str(j)]
    ...     distance_matrices[method] = D
    >>>
    >>> # Plot MDS visualizations
    >>> fig = plot_shape_distance_mds(distance_matrices, labels=labels)

    **Computing distances automatically**:

    >>> from neural_analysis.plotting.shape_distance import plot_shape_distance_mds
    >>> datasets = [np.random.randn(50, 10) for _ in range(20)]
    >>> labels = np.random.randint(0, 3, 20)
    >>>
    >>> # Plot with automatic distance computation
    >>> fig = plot_shape_distance_mds(
    ...     datasets=datasets,
    ...     methods=["procrustes", "one-to-one", "soft-matching"],
    ...     labels=labels,
    ...     save_path="results.h5",
    ...     metric="sqeuclidean",
    ...     max_neurons=30,
    ... )
    """
    from neural_analysis.metrics.pairwise_metrics import compare_datasets

    # Determine if we need to compute distances or use provided matrices
    if distance_matrices is None:
        if datasets is None or methods is None:
            raise ValueError(
                "Either `distance_matrices` must be provided, or both "
                "`datasets` and `methods` must be provided."
            )

        # Compute distance matrices using compare_datasets
        n_datasets = len(datasets)
        datasets_dict = {str(i): d for i, d in enumerate(datasets)}
        distance_matrices = {}

        for method in methods:
            # Compute all-pairs distances
            result = compare_datasets(
                datasets_dict,
                mode="all-pairs",
                metric=method,
                save_path=save_path,
                regenerate=regenerate,
                show_progress=show_progress,
                **metric_kwargs,
            )

            # Convert dict result to symmetric matrix
            dist_mat = np.zeros((n_datasets, n_datasets), dtype=np.float64)
            for i in range(n_datasets):
                for j in range(n_datasets):
                    dist_mat[i, j] = result[str(i)][str(j)]
            distance_matrices[method] = dist_mat
    # Convert dict keys to list (mypy needs explicit type)
    method_names: list[str] = list(distance_matrices.keys())
    n_methods = len(method_names)
    unique_labels = np.unique(labels) if labels is not None else None

    plot_specs = []

    for row, method_name in enumerate(method_names):
        dist_matrix: npt.NDArray[np.float64] = distance_matrices[method_name]

        # MDS 2D
        emb_mds_2 = embed_mds(dist_matrix, n_components=2)
        if labels is not None and unique_labels is not None:
            # Create one spec per cluster for proper coloring
            for lab_idx, lab in enumerate(unique_labels):
                idx = labels == lab
                if idx.sum() > 0:  # pragma: no branch
                    spec = PlotSpec(
                        data={"x": emb_mds_2[idx, 0], "y": emb_mds_2[idx, 1]},
                        plot_type="scatter",
                        subplot_position=row * 2,
                        title=f"{method_name}: MDS (2D)" if lab_idx == 0 else None,
                        label=f"Cluster {lab}" if row == 0 else None,
                        color=f"C{lab % 10}",  # Use matplotlib color cycle
                        marker_size=30,
                        alpha=0.7,
                        equal_aspect=True,
                        kwargs={
                            "x_label": "Dim 1",
                            "y_label": "Dim 2",
                        },
                    )
                    plot_specs.append(spec)
        else:
            spec = PlotSpec(
                data={"x": emb_mds_2[:, 0], "y": emb_mds_2[:, 1]},
                plot_type="scatter",
                subplot_position=row * 2,
                title=f"{method_name}: MDS (2D)",
                color="steelblue",
                marker_size=30,
                alpha=0.7,
                equal_aspect=True,
                kwargs={
                    "x_label": "Dim 1",
                    "y_label": "Dim 2",
                },
            )
            plot_specs.append(spec)

        # MDS 20D + PCA 2D
        emb_mds_pca_2 = embed_mds_pca(dist_matrix, mds_dim=20, pca_dim=2)
        if labels is not None and unique_labels is not None:
            for lab_idx, lab in enumerate(unique_labels):
                idx = labels == lab
                if idx.sum() > 0:  # pragma: no branch
                    spec = PlotSpec(
                        data={"x": emb_mds_pca_2[idx, 0], "y": emb_mds_pca_2[idx, 1]},
                        plot_type="scatter",
                        subplot_position=row * 2 + 1,
                        title=(
                            f"{method_name}: MDS(20D) + PCA(2D)"
                            if lab_idx == 0
                            else None
                        ),
                        label=f"Cluster {lab}" if row == 0 else None,
                        color=f"C{lab % 10}",  # Use matplotlib color cycle
                        marker_size=30,
                        alpha=0.7,
                        equal_aspect=True,
                        kwargs={
                            "x_label": "PC 1",
                            "y_label": "PC 2",
                        },
                    )
                    plot_specs.append(spec)
        else:
            spec = PlotSpec(
                data={"x": emb_mds_pca_2[:, 0], "y": emb_mds_pca_2[:, 1]},
                plot_type="scatter",
                subplot_position=row * 2 + 1,
                title=f"{method_name}: MDS(20D) + PCA(2D)",
                color="steelblue",
                marker_size=30,
                alpha=0.7,
                equal_aspect=True,
                kwargs={
                    "x_label": "PC 1",
                    "y_label": "PC 2",
                },
            )
            plot_specs.append(spec)

    # Convert figsize from float tuple to int tuple for PlotConfig
    figsize_int: tuple[int, int] = (int(figsize[0]), int(figsize[1]))
    grid = PlotGrid(
        plot_specs=plot_specs,
        config=PlotConfig(figsize=figsize_int),
        layout=GridLayoutConfig(rows=n_methods, cols=2),
        backend=backend,
    )

    return grid.plot()
