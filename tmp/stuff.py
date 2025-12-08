import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple
from scipy.linalg import svd
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Import functions from the distributions module
from neural_analysis.metrics.distributions import (
    modify_matrix,
    shape_distance,
)

# Import data generation functions
from neural_analysis.data.synthetic_data import (
    generate_cluster_templates,
    generate_dataset_from_cluster_template,
    generate_shape_distance_datasets,
)

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not available
    def tqdm(iterable, desc=None, total=None, **kwargs):
        if desc:
            print(desc)
        return iterable


# -------------------------
# Data generation (now using functions from synthetic_data.py)
# -------------------------
# Note: Data generation functions have been moved to neural_analysis.data.synthetic_data
# Import them at the top of the file


# -------------------------
# Distance metric wrappers (for compatibility with compute_distance_matrix)
# -------------------------


def procrustes_distance(
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    max_neurons: int = 50,
) -> float:
    """
    Wrapper for shape_distance with method='procrustes' that handles subsampling and returns only distance.
    """
    # Subsample if needed (for speed) - ensure both matrices have same shape
    Nx, F = X.shape
    Ny, Fy = Y.shape
    if F != Fy:
        raise ValueError(f"Matrices must have same number of features: {F} != {Fy}")

    if Nx > max_neurons or Ny > max_neurons:
        N = min(Nx, Ny, max_neurons)
        rng = np.random.default_rng()
        if Nx > N:
            idx_x = rng.choice(Nx, size=N, replace=False)
            X = X[idx_x]
        if Ny > N:
            idx_y = rng.choice(Ny, size=N, replace=False)
            Y = Y[idx_y]

    # Ensure shapes match (should be same after subsampling)
    if X.shape[0] != Y.shape[0]:
        N = min(X.shape[0], Y.shape[0])
        rng = np.random.default_rng()
        if X.shape[0] > N:
            idx_x = rng.choice(X.shape[0], size=N, replace=False)
            X = X[idx_x]
        if Y.shape[0] > N:
            idx_y = rng.choice(Y.shape[0], size=N, replace=False)
            Y = Y[idx_y]

    dist, _, _ = shape_distance(X, Y, method="procrustes")
    return float(dist)


def one_to_one_matching_distance(
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    metric: str = "sqeuclidean",
    max_neurons: int = 50,
) -> float:
    """
    Wrapper for shape_distance with method='one-to-one' that handles subsampling and returns only distance.
    """
    # Subsample if needed (for speed) - ensure both matrices have same shape
    Nx, F = X.shape
    Ny, Fy = Y.shape
    if F != Fy:
        raise ValueError(f"Matrices must have same number of features: {F} != {Fy}")

    if Nx > max_neurons or Ny > max_neurons:
        N = min(Nx, Ny, max_neurons)
        rng = np.random.default_rng()
        if Nx > N:
            idx_x = rng.choice(Nx, size=N, replace=False)
            X = X[idx_x]
        if Ny > N:
            idx_y = rng.choice(Ny, size=N, replace=False)
            Y = Y[idx_y]

    # Ensure shapes match (should be same after subsampling)
    if X.shape[0] != Y.shape[0]:
        N = min(X.shape[0], Y.shape[0])
        rng = np.random.default_rng()
        if X.shape[0] > N:
            idx_x = rng.choice(X.shape[0], size=N, replace=False)
            X = X[idx_x]
        if Y.shape[0] > N:
            idx_y = rng.choice(Y.shape[0], size=N, replace=False)
            Y = Y[idx_y]

    dist, _, _ = shape_distance(X, Y, method="one-to-one", metric=metric)
    return float(dist)


def soft_matching_distance(
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    metric: str = "sqeuclidean",
    reg: float = 1e-3,
    approx: bool = False,
    max_neurons: int = 50,
) -> float:
    """
    Wrapper for shape_distance with method='soft-matching' that handles subsampling and returns only distance.
    """
    # Subsample if needed (for speed)
    n1, n2 = X.shape[0], Y.shape[0]
    if n1 > max_neurons or n2 > max_neurons:
        rng = np.random.default_rng()
        if n1 > max_neurons:
            idx_x = rng.choice(n1, size=max_neurons, replace=False)
            X = X[idx_x]
        if n2 > max_neurons:
            idx_y = rng.choice(n2, size=max_neurons, replace=False)
            Y = Y[idx_y]

    dist, _, _ = shape_distance(
        X, Y, method="soft-matching", metric=metric, approx=approx, reg=reg
    )
    return float(dist)


# -------------------------
# Pairwise distance matrices
# -------------------------


def compute_distance_matrix(
    datasets: List[NDArray[np.float64]],
    metric_func,
    metric_name: str = "distance",
) -> NDArray[np.float64]:
    """
    Compute pairwise distances between K datasets using metric_func(X, Y).
    """
    K = len(datasets)
    D = np.zeros((K, K), dtype=np.float64)

    # Preprocess all datasets once (whiten=True to preserve shape differences)
    preprocessed = [
        modify_matrix(d, whiten=True, normalize=True, scale_variance=False)
        for d in tqdm(datasets, desc=f"Preprocessing for {metric_name}", leave=False)
    ]

    # Total number of pairs: K*(K-1)/2
    total_pairs = K * (K - 1) // 2
    pbar = tqdm(
        total=total_pairs, desc=f"Computing {metric_name} distances", unit="pairs"
    )

    for i in range(K):
        Xi = preprocessed[i]
        for j in range(i + 1, K):
            Yj = preprocessed[j]
            d = metric_func(Xi, Yj)
            D[i, j] = D[j, i] = d
            pbar.update(1)

    pbar.close()
    return D


# -------------------------
# Embedding & plotting
# -------------------------


def embed_mds(D: NDArray[np.float64], n_components: int = 2) -> NDArray[np.float64]:
    mds = MDS(
        n_components=n_components,
        dissimilarity="precomputed",
        random_state=0,
        n_init=4,
        max_iter=300,
    )
    return mds.fit_transform(D)


def embed_mds_pca(
    D: NDArray[np.float64], mds_dim: int = 20, pca_dim: int = 2
) -> NDArray[np.float64]:
    Z = embed_mds(D, n_components=mds_dim)
    pca = PCA(n_components=pca_dim, random_state=0)
    return pca.fit_transform(Z)


def plot_all(
    D_proc: NDArray[np.float64],
    D_oto: NDArray[np.float64],
    D_soft: NDArray[np.float64],
    labels: NDArray[np.int_],
    save_path: str | None = None,
):
    """
    3x2 grid:
      Row 1: Procrustes (MDS-2; MDS-20+PCA-2)
      Row 2: One-to-one
      Row 3: Soft-matching

    Parameters
    ----------
    save_path : str, optional
        If provided, saves the figure to this path instead of showing it.
        If None, attempts to show the figure (may not work in non-interactive environments).
    """
    # Use non-interactive backend to avoid warnings
    import matplotlib

    matplotlib.use("Agg")

    methods = [
        ("Procrustes", D_proc),
        ("One-to-one", D_oto),
        ("Soft matching", D_soft),
    ]

    cmap = plt.get_cmap("tab10")
    unique_labels = np.unique(labels)

    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    fig.tight_layout(pad=3.0)

    for row, (name, D) in enumerate(methods):
        # MDS 2D
        emb_mds_2 = embed_mds(D, n_components=2)
        ax1 = axes[row, 0]
        for lab in unique_labels:
            idx = labels == lab
            ax1.scatter(
                emb_mds_2[idx, 0],
                emb_mds_2[idx, 1],
                color=cmap(lab),
                label=f"Cluster {lab}" if row == 0 else None,
                alpha=0.7,
            )
        ax1.set_title(f"{name}: MDS (2D)")
        ax1.set_xlabel("Dim 1")
        ax1.set_ylabel("Dim 2")

        # MDS 20D + PCA 2D
        emb_mds_pca_2 = embed_mds_pca(D, mds_dim=20, pca_dim=2)
        ax2 = axes[row, 1]
        for lab in unique_labels:
            idx = labels == lab
            ax2.scatter(
                emb_mds_pca_2[idx, 0],
                emb_mds_pca_2[idx, 1],
                color=cmap(lab),
                label=f"Cluster {lab}" if row == 0 else None,
                alpha=0.7,
            )
        ax2.set_title(f"{name}: MDS(20D) + PCA(2D)")
        ax2.set_xlabel("PC 1")
        ax2.set_ylabel("PC 2")

    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper right", title="True clusters")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    else:
        try:
            plt.show()
        except Exception:
            # Fallback: save to default location if show() fails
            default_path = "distance_comparison.png"
            plt.savefig(default_path, dpi=150, bbox_inches="tight")
            print(
                f"Non-interactive environment detected. Figure saved to {default_path}"
            )

    plt.close()


# -------------------------
# Main
# -------------------------


def main():
    # 1. Generate datasets (reduced sizes for faster computation)
    datasets, labels = generate_shape_distance_datasets(
        n_datasets=50,  # Reduced from 100
        n_clusters=5,
        min_neurons=30,  # Reduced from 50
        max_neurons=80,  # Reduced from 200
        n_features=100,  # Reduced from 300
        seed=1,
    )

    # 2. Compute distance matrices
    D_proc = compute_distance_matrix(datasets, procrustes_distance, "Procrustes")

    D_oto = compute_distance_matrix(
        datasets, one_to_one_matching_distance, "One-to-one"
    )

    D_soft = compute_distance_matrix(datasets, soft_matching_distance, "Soft-matching")

    # 3. Plot embeddings
    plot_all(D_proc, D_oto, D_soft, labels, save_path="distance_comparison.png")


if __name__ == "__main__":
    main()
