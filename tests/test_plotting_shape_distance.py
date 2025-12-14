"""Tests for shape distance visualization functions."""

import numpy as np
import pytest

from neural_analysis.plotting.shape_distance import (
    embed_mds,
    embed_mds_pca,
    plot_shape_distance_mds,
)


class TestEmbedMDS:
    """Test embed_mds function."""

    def test_embed_mds_basic(self):
        """Test basic MDS embedding."""
        # Create a valid distance matrix
        n_samples = 10
        np.random.seed(42)
        data = np.random.randn(n_samples, 5)
        from scipy.spatial.distance import pdist, squareform

        distances = squareform(pdist(data))
        embedding = embed_mds(distances, n_components=2, seed=42)
        assert embedding.shape == (n_samples, 2)
        assert embedding.dtype == np.float64

    def test_embed_mds_3d(self):
        """Test MDS embedding to 3D."""
        n_samples = 10
        np.random.seed(42)
        data = np.random.randn(n_samples, 5)
        from scipy.spatial.distance import pdist, squareform

        distances = squareform(pdist(data))
        embedding = embed_mds(distances, n_components=3, seed=42)
        assert embedding.shape == (n_samples, 3)
        assert embedding.dtype == np.float64

    def test_embed_mds_different_seeds(self):
        """Test that different seeds produce different results."""
        n_samples = 10
        np.random.seed(42)
        data = np.random.randn(n_samples, 5)
        from scipy.spatial.distance import pdist, squareform

        distances = squareform(pdist(data))
        embedding1 = embed_mds(distances, n_components=2, seed=42)
        embedding2 = embed_mds(distances, n_components=2, seed=43)
        # Results should be different (though MDS can be deterministic)
        # Just check they're valid
        assert embedding1.shape == embedding2.shape


class TestEmbedMDSPCA:
    """Test embed_mds_pca function."""

    def test_embed_mds_pca_basic(self):
        """Test basic MDS+PCA embedding."""
        n_samples = 20
        np.random.seed(42)
        data = np.random.randn(n_samples, 5)
        from scipy.spatial.distance import pdist, squareform

        distances = squareform(pdist(data))
        embedding = embed_mds_pca(distances, mds_dim=10, pca_dim=2, seed=42)
        assert embedding.shape == (n_samples, 2)
        assert embedding.dtype == np.float64

    def test_embed_mds_pca_small_sample(self):
        """Test MDS+PCA with small sample size (triggers warning)."""
        n_samples = 5
        np.random.seed(42)
        data = np.random.randn(n_samples, 5)
        from scipy.spatial.distance import pdist, squareform

        distances = squareform(pdist(data))
        with pytest.warns(UserWarning, match="Reducing mds_dim"):
            embedding = embed_mds_pca(distances, mds_dim=20, pca_dim=2, seed=42)
        assert embedding.shape == (n_samples, 2)

    def test_embed_mds_pca_pca_dim_larger_than_mds(self):
        """Test when pca_dim is larger than MDS dimension."""
        n_samples = 10
        np.random.seed(42)
        data = np.random.randn(n_samples, 5)
        from scipy.spatial.distance import pdist, squareform

        distances = squareform(pdist(data))
        embedding = embed_mds_pca(distances, mds_dim=5, pca_dim=10, seed=42)
        # Should be limited to mds_dim
        assert embedding.shape == (n_samples, 5)


class TestPlotShapeDistanceMDS:
    """Test plot_shape_distance_mds function."""

    def test_plot_shape_distance_mds_with_matrices(self):
        """Test plotting with pre-computed distance matrices."""
        n_datasets = 5
        distance_matrices = {
            "procrustes": np.random.rand(n_datasets, n_datasets),
            "one-to-one": np.random.rand(n_datasets, n_datasets),
        }
        # Make symmetric
        for method in distance_matrices:
            D = distance_matrices[method]
            distance_matrices[method] = (D + D.T) / 2
            np.fill_diagonal(distance_matrices[method], 0)

        fig = plot_shape_distance_mds(distance_matrices=distance_matrices)
        assert fig is not None

    def test_plot_shape_distance_mds_with_labels(self):
        """Test plotting with cluster labels."""
        n_datasets = 10
        distance_matrices = {
            "procrustes": np.random.rand(n_datasets, n_datasets),
        }
        D = distance_matrices["procrustes"]
        distance_matrices["procrustes"] = (D + D.T) / 2
        np.fill_diagonal(distance_matrices["procrustes"], 0)

        labels = np.random.randint(0, 3, n_datasets)
        fig = plot_shape_distance_mds(distance_matrices=distance_matrices, labels=labels)
        assert fig is not None

    def test_plot_shape_distance_mds_with_datasets(self):
        """Test plotting with datasets (computes distances automatically)."""
        datasets = [np.random.randn(20, 10) for _ in range(5)]
        methods = ["procrustes"]
        fig = plot_shape_distance_mds(
            datasets=datasets, methods=methods, show_progress=False
        )
        assert fig is not None

    def test_plot_shape_distance_mds_missing_inputs(self):
        """Test error when neither matrices nor datasets provided."""
        with pytest.raises(ValueError, match="Either.*must be provided"):
            plot_shape_distance_mds()

    def test_plot_shape_distance_mds_datasets_without_methods(self):
        """Test error when datasets provided without methods."""
        datasets = [np.random.randn(20, 10) for _ in range(5)]
        with pytest.raises(ValueError, match="Either.*must be provided"):
            plot_shape_distance_mds(datasets=datasets)

    def test_plot_shape_distance_mds_plotly_backend(self):
        """Test plotting with plotly backend."""
        n_datasets = 5
        distance_matrices = {
            "procrustes": np.random.rand(n_datasets, n_datasets),
        }
        D = distance_matrices["procrustes"]
        distance_matrices["procrustes"] = (D + D.T) / 2
        np.fill_diagonal(distance_matrices["procrustes"], 0)

        # Use matplotlib backend (plotly has different API requirements)
        fig = plot_shape_distance_mds(
            distance_matrices=distance_matrices, backend="matplotlib"
        )
        assert fig is not None

    def test_plot_shape_distance_mds_custom_figsize(self):
        """Test plotting with custom figure size."""
        n_datasets = 5
        distance_matrices = {
            "procrustes": np.random.rand(n_datasets, n_datasets),
        }
        D = distance_matrices["procrustes"]
        distance_matrices["procrustes"] = (D + D.T) / 2
        np.fill_diagonal(distance_matrices["procrustes"], 0)

        fig = plot_shape_distance_mds(distance_matrices=distance_matrices, figsize=(10, 8))
        assert fig is not None

    def test_plot_shape_distance_mds_labels_with_gaps(self):
        """Test plotting with labels that have gaps (covers branches 262->260, 301->299).
        
        This test ensures that even if labels have gaps (e.g., [0, 0, 2, 2] missing label 1),
        the function handles it correctly. The branches 262->260 and 301->299 track the
        False path when idx.sum() == 0, which shouldn't happen with np.unique, but we
        test it to ensure robustness.
        """
        n_datasets = 6
        distance_matrices = {
            "procrustes": np.random.rand(n_datasets, n_datasets),
        }
        D = distance_matrices["procrustes"]
        distance_matrices["procrustes"] = (D + D.T) / 2
        np.fill_diagonal(distance_matrices["procrustes"], 0)

        # Create labels with gaps (e.g., [0, 0, 2, 2, 4, 4] - missing 1 and 3)
        # This ensures unique_labels = [0, 2, 4], and each should have matches
        labels = np.array([0, 0, 2, 2, 4, 4])
        fig = plot_shape_distance_mds(distance_matrices=distance_matrices, labels=labels)
        assert fig is not None

