"""Visualization utilities for shape distance metrics.

This module provides functions to visualize shape distance matrices using
MDS (Multidimensional Scaling) and related embedding techniques.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from sklearn.decomposition import PCA

from neural_analysis.embeddings.dimensionality_reduction import compute_embedding
from neural_analysis.metrics.pairwise_metrics import compute_all_pairs
from neural_analysis.plotting.embeddings import plot_embedding_2d
from neural_analysis.plotting.grid_config import (
    GridLayoutConfig,
    PlotConfig,
    PlotGrid,
    PlotSpec,
)


def compute_pairwise_distance_matrix(
    datasets: list[npt.NDArray[np.float64]],
    method: Literal["procrustes", "one-to-one", "soft-matching"] = "procrustes",
    metric: str = "sqeuclidean",
    max_neurons: int | None = None,
    show_progress: bool = True,
    **method_kwargs: Any,
) -> npt.NDArray[np.float64]:
    """Compute pairwise distance matrix between multiple datasets.

    Uses `compute_all_pairs` from `pairwise_metrics` and converts the result
    to a symmetric distance matrix.

    Parameters
    ----------
    datasets : list of ndarray
        List of datasets, each of shape (n_neurons_i, n_features).
    method : {'procrustes', 'one-to-one', 'soft-matching'}, default='procrustes'
        Shape distance method to use.
    metric : str, default='sqeuclidean'
        Distance metric for one-to-one and soft-matching.
        Passed to shape_distance via metric_kwargs.
    max_neurons : int or None, default=None
        Maximum number of neurons to use per dataset (for speed).
        If None, uses all neurons.
    show_progress : bool, default=True
        Show progress bar during computation.
    **method_kwargs
        Additional keyword arguments for shape_distance (e.g., approx, reg).

    Returns
    -------
    distance_matrix : ndarray of shape (n_datasets, n_datasets)
        Symmetric pairwise distance matrix.

    Notes
    -----
    This function uses `compute_all_pairs` internally, which handles the
    parameter name conflict between the shape method name (passed as `metric`
    to `compute_all_pairs`) and the distance metric (passed as `metric` in
    `metric_kwargs` to `shape_distance`).
    """
    n_datasets = len(datasets)

    # Subsample if needed for speed
    if max_neurons is not None:
        rng = np.random.default_rng(42)
        subsampled_datasets = []
        for d in datasets:
            N, F = d.shape
            if N > max_neurons:
                idx = rng.choice(N, size=max_neurons, replace=False)
                subsampled_datasets.append(d[idx])
            else:
                subsampled_datasets.append(d)
        datasets = subsampled_datasets

    # Convert list to dict for compute_all_pairs
    datasets_dict = {str(i): d for i, d in enumerate(datasets)}

    # Use compute_all_pairs with shape metric
    # Note: compute_all_pairs uses 'metric' for the shape method name,
    # and we pass the distance metric (sqeuclidean, etc.) via metric_kwargs
    # The distance metric parameter name for shape_distance is also 'metric',
    # so we pass it via metric_kwargs
    metric_kwargs = {"metric": metric, **method_kwargs}
    results = compute_all_pairs(
        datasets_dict,
        metric=method,  # Shape method name (procrustes, one-to-one, soft-matching)
        show_progress=show_progress,
        **metric_kwargs,  # Contains metric='sqeuclidean' and other shape_distance kwargs
    )

    # Convert nested dict to symmetric matrix
    D = np.zeros((n_datasets, n_datasets), dtype=np.float64)
    for i in range(n_datasets):
        for j in range(n_datasets):
            D[i, j] = results[str(i)][str(j)]

    return D


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
    return compute_embedding(
        distance_matrix,
        method="mds",
        n_components=n_components,
        metric="precomputed",
        random_state=seed,
    )


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
        )
    
    Z = embed_mds(distance_matrix, n_components=actual_mds_dim, seed=seed)
    # Ensure pca_dim doesn't exceed the MDS embedding dimension
    actual_pca_dim = min(pca_dim, Z.shape[1])
    pca = PCA(n_components=actual_pca_dim, random_state=seed)
    return pca.fit_transform(Z)


def plot_shape_distance_mds(
    distance_matrices: dict[str, npt.NDArray[np.float64]],
    labels: npt.NDArray[np.int_] | None = None,
    backend: Literal["matplotlib", "plotly"] = "matplotlib",
    figsize: tuple[float, float] = (12, 12),
) -> Any:
    """Plot MDS embeddings for multiple distance matrices using PlotGrid.

    Uses `plot_embedding_2d` from `plotting.embeddings` for each embedding.

    Parameters
    ----------
    distance_matrices : dict
        Dictionary mapping method names to distance matrices.
        Each matrix should be of shape (n_datasets, n_datasets).
    labels : ndarray of shape (n_datasets,) or None, default=None
        Cluster labels for coloring points. If None, all points are same color.
    backend : {'matplotlib', 'plotly'}, default='matplotlib'
        Plotting backend to use.
    figsize : tuple of float, default=(12, 12)
        Figure size (width, height) in inches.

    Returns
    -------
    fig
        Figure object from the plotting backend.
    """
    methods = list(distance_matrices.keys())
    n_methods = len(methods)
    unique_labels = np.unique(labels) if labels is not None else None

    plot_specs = []

    for row, method_name in enumerate(methods):
        D = distance_matrices[method_name]

        # MDS 2D
        emb_mds_2 = embed_mds(D, n_components=2)
        if labels is not None and unique_labels is not None:
            # Create one spec per cluster for proper coloring
            for lab_idx, lab in enumerate(unique_labels):
                idx = labels == lab
                if idx.sum() > 0:
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
        emb_mds_pca_2 = embed_mds_pca(D, mds_dim=20, pca_dim=2)
        if labels is not None and unique_labels is not None:
            for lab_idx, lab in enumerate(unique_labels):
                idx = labels == lab
                if idx.sum() > 0:
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

    grid = PlotGrid(
        plot_specs=plot_specs,
        config=PlotConfig(figsize=figsize),
        layout=GridLayoutConfig(rows=n_methods, cols=2),
        backend=backend,
    )

    return grid.plot()

