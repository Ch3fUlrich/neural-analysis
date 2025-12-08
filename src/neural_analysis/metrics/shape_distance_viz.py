"""Visualization utilities for shape distance metrics.

This module provides functions to visualize shape distance matrices using
MDS (Multidimensional Scaling) and related embedding techniques.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from sklearn.decomposition import PCA
from sklearn.manifold import MDS

from neural_analysis.metrics.distributions import modify_matrix, shape_distance
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
    **method_kwargs: Any,
) -> npt.NDArray[np.float64]:
    """Compute pairwise distance matrix between multiple datasets.

    Parameters
    ----------
    datasets : list of ndarray
        List of datasets, each of shape (n_neurons_i, n_features).
    method : {'procrustes', 'one-to-one', 'soft-matching'}, default='procrustes'
        Shape distance method to use.
    metric : str, default='sqeuclidean'
        Distance metric for one-to-one and soft-matching.
    max_neurons : int or None, default=None
        Maximum number of neurons to use per dataset (for speed).
        If None, uses all neurons.
    **method_kwargs
        Additional keyword arguments for shape_distance.

    Returns
    -------
    distance_matrix : ndarray of shape (n_datasets, n_datasets)
        Symmetric pairwise distance matrix.
    """
    n_datasets = len(datasets)
    D = np.zeros((n_datasets, n_datasets), dtype=np.float64)

    # Preprocess all datasets once
    preprocessed = [
        modify_matrix(d, whiten=True, normalize=True, scale_variance=False)
        for d in datasets
    ]

    for i in range(n_datasets):
        Xi = preprocessed[i]
        for j in range(i + 1, n_datasets):
            Yj = preprocessed[j]

            # Subsample if needed for speed
            if max_neurons is not None:
                rng = np.random.default_rng(42)
                Nx, F = Xi.shape
                Ny, Fy = Yj.shape
                if Nx > max_neurons:
                    idx_x = rng.choice(Nx, size=max_neurons, replace=False)
                    Xi = Xi[idx_x]
                if Ny > max_neurons:
                    idx_y = rng.choice(Ny, size=max_neurons, replace=False)
                    Yj = Yj[idx_y]

            dist, _, _ = shape_distance(
                Xi, Yj, method=method, metric=metric, **method_kwargs
            )
            if isinstance(dist, np.ndarray):
                dist = float(np.mean(dist))
            D[i, j] = D[j, i] = float(dist)

    return D


def embed_mds(
    distance_matrix: npt.NDArray[np.float64], n_components: int = 2, seed: int = 0
) -> npt.NDArray[np.float64]:
    """Embed distance matrix using MDS.

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
    mds = MDS(
        n_components=n_components,
        dissimilarity="precomputed",
        random_state=seed,
        n_init=4,
        max_iter=300,
    )
    return mds.fit_transform(distance_matrix)


def embed_mds_pca(
    distance_matrix: npt.NDArray[np.float64],
    mds_dim: int = 20,
    pca_dim: int = 2,
    seed: int = 0,
) -> npt.NDArray[np.float64]:
    """Embed distance matrix using MDS followed by PCA.

    Parameters
    ----------
    distance_matrix : ndarray of shape (n_samples, n_samples)
        Pairwise distance matrix.
    mds_dim : int, default=20
        Number of dimensions for initial MDS embedding.
    pca_dim : int, default=2
        Number of dimensions for PCA reduction.
    seed : int, default=0
        Random seed for reproducibility.

    Returns
    -------
    embedding : ndarray of shape (n_samples, pca_dim)
        MDS+PCA embedding coordinates.
    """
    Z = embed_mds(distance_matrix, n_components=mds_dim, seed=seed)
    pca = PCA(n_components=pca_dim, random_state=seed)
    return pca.fit_transform(Z)


def plot_shape_distance_mds(
    distance_matrices: dict[str, npt.NDArray[np.float64]],
    labels: npt.NDArray[np.int_] | None = None,
    backend: Literal["matplotlib", "plotly"] = "matplotlib",
    figsize: tuple[float, float] = (12, 12),
) -> Any:
    """Plot MDS embeddings for multiple distance matrices using PlotGrid.

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
                        title=f"{method_name}: MDS(20D) + PCA(2D)" if lab_idx == 0 else None,
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

