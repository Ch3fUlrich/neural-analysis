"""
Tests for embeddings module (dimensionality reduction).
"""

from typing import Any

import numpy as np
import pytest

from neural_analysis.embeddings import (
    compute_embedding,
    compute_multiple_embeddings,
    pca_explained_variance,
)


@pytest.fixture
def sample_data() -> Any:
    """Generate sample high-dimensional data for testing."""
    rng = np.random.default_rng(42)
    # 100 samples, 20 features
    data = rng.normal(0, 1, (100, 20))
    return data


@pytest.fixture
def sample_data_with_labels() -> Any:
    """Generate sample data with category labels."""
    rng = np.random.default_rng(42)
    # 3 clusters, 50 samples each, 20 features
    data = np.vstack(
        [
            rng.normal(0, 1, (50, 20)),
            rng.normal(3, 1, (50, 20)),
            rng.normal(-3, 1, (50, 20)),
        ]
    )
    labels = np.repeat([0, 1, 2], 50)
    return data, labels


class TestComputeEmbedding:
    """Tests for compute_embedding function."""

    def test_pca_2d(self, sample_data: Any) -> None:
        """Test PCA with 2 components."""
        embedding = compute_embedding(sample_data, method="pca", n_components=2)
        assert embedding.shape == (100, 2)
        assert not np.any(np.isnan(embedding))

    def test_pca_3d(self, sample_data: Any) -> None:
        """Test PCA with 3 components."""
        embedding = compute_embedding(sample_data, method="pca", n_components=3)
        assert embedding.shape == (100, 3)

    def test_tsne_2d(self, sample_data: Any) -> None:
        """Test t-SNE with 2 components."""
        embedding = compute_embedding(
            sample_data, method="tsne", n_components=2, random_state=42
        )
        assert embedding.shape == (100, 2)
        assert not np.any(np.isnan(embedding))

    def test_mds_2d(self, sample_data: Any) -> None:
        """Test MDS with 2 components."""
        embedding = compute_embedding(
            sample_data, method="mds", n_components=2, random_state=42
        )
        assert embedding.shape == (100, 2)
        assert not np.any(np.isnan(embedding))

    def test_isomap_2d(self, sample_data: Any) -> None:
        """Test Isomap with 2 components."""
        embedding = compute_embedding(
            sample_data, method="isomap", n_components=2, n_neighbors=10
        )
        assert embedding.shape == (100, 2)
        assert not np.any(np.isnan(embedding))

    def test_lle_2d(self, sample_data: Any) -> None:
        """Test LLE with 2 components."""
        embedding = compute_embedding(
            sample_data, method="lle", n_components=2, n_neighbors=10, random_state=42
        )
        assert embedding.shape == (100, 2)

    def test_spectral_2d(self, sample_data: Any) -> None:
        """Test Spectral Embedding with 2 components."""
        embedding = compute_embedding(
            sample_data,
            method="spectral",
            n_components=2,
            n_neighbors=10,
            random_state=42,
        )
        assert embedding.shape == (100, 2)

    @pytest.mark.skipif(
        not hasattr(
            __import__(
                "neural_analysis.embeddings.dimensionality_reduction",
                fromlist=["UMAP_AVAILABLE"],
            ),
            "UMAP_AVAILABLE",
        )
        or not __import__(
            "neural_analysis.embeddings.dimensionality_reduction",
            fromlist=["UMAP_AVAILABLE"],
        ).UMAP_AVAILABLE,
        reason="UMAP not installed",
    )
    def test_umap_2d(self, sample_data: Any) -> None:
        """Test UMAP with 2 components (if available)."""
        embedding = compute_embedding(
            sample_data, method="umap", n_components=2, n_neighbors=15, random_state=42
        )
        assert embedding.shape == (100, 2)
        assert not np.any(np.isnan(embedding))

    def test_invalid_method(self, sample_data: Any) -> None:
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown method"):
            compute_embedding(sample_data, method="invalid")

    def test_invalid_n_components(self, sample_data: Any) -> None:
        """Test that invalid n_components raises ValueError."""
        with pytest.raises(ValueError):
            compute_embedding(sample_data, method="pca", n_components=0)

        with pytest.raises(ValueError):
            compute_embedding(sample_data, method="pca", n_components=200)

    def test_invalid_data_shape(self) -> None:
        """Test that 1D data raises ValueError."""
        data_1d = np.random.rand(100)
        with pytest.raises(ValueError, match="must be 2D"):
            compute_embedding(data_1d, method="pca")

    def test_reproducibility(self, sample_data: Any) -> None:
        """Test that same random_state gives same results."""
        emb1 = compute_embedding(
            sample_data, method="tsne", n_components=2, random_state=42
        )
        emb2 = compute_embedding(
            sample_data, method="tsne", n_components=2, random_state=42
        )
        np.testing.assert_array_almost_equal(emb1, emb2)

    def test_precomputed_distance_matrix(self, sample_data: Any) -> None:
        """Test MDS with precomputed distance matrix."""
        from scipy.spatial.distance import pdist, squareform

        distances = squareform(pdist(sample_data, metric="euclidean"))

        embedding = compute_embedding(
            distances,
            method="mds",
            n_components=2,
            metric="precomputed",
            random_state=42,
        )
        assert embedding.shape == (100, 2)


class TestComputeMultipleEmbeddings:
    """Tests for compute_multiple_embeddings function."""

    def test_default_methods(self, sample_data: Any) -> None:
        """Test with default methods."""
        embeddings = compute_multiple_embeddings(sample_data, random_state=42)

        # At minimum, should have PCA and t-SNE
        assert "pca" in embeddings
        assert "tsne" in embeddings

        for _method, embedding in embeddings.items():
            assert embedding.shape == (100, 2)

    def test_custom_methods(self, sample_data: Any) -> None:
        """Test with custom method list."""
        methods = ["pca", "mds", "isomap"]
        embeddings = compute_multiple_embeddings(
            sample_data, methods=methods, n_components=2, random_state=42
        )

        assert len(embeddings) == 3
        assert all(method in embeddings for method in methods)

    def test_3d_embeddings(self, sample_data: Any) -> None:
        """Test with 3D embeddings."""
        embeddings = compute_multiple_embeddings(
            sample_data, methods=["pca", "mds"], n_components=3, random_state=42
        )

        for embedding in embeddings.values():
            assert embedding.shape == (100, 3)

    def test_missing_package(self, sample_data: Any) -> None:
        """Test graceful handling of missing packages."""
        # Try to compute UMAP even if not installed - should skip gracefully
        embeddings = compute_multiple_embeddings(
            sample_data, methods=["pca", "umap"], random_state=42
        )

        # Should at least have PCA
        assert "pca" in embeddings


class TestPCAExplainedVariance:
    """Tests for pca_explained_variance function."""

    def test_all_components(self, sample_data: Any) -> None:
        """Test with all components."""
        variance_info = pca_explained_variance(sample_data)

        assert "explained_variance_ratio" in variance_info
        assert "explained_variance" in variance_info
        assert "cumulative_variance_ratio" in variance_info

        # Should sum to approximately 1.0
        assert np.abs(variance_info["explained_variance_ratio"].sum() - 1.0) < 0.01

    def test_limited_components(self, sample_data: Any) -> None:
        """Test with limited number of components."""
        variance_info = pca_explained_variance(sample_data, n_components=5)

        assert len(variance_info["explained_variance_ratio"]) == 5
        assert len(variance_info["explained_variance"]) == 5

    def test_threshold_components(self, sample_data: Any) -> None:
        """Test computation of threshold components."""
        variance_info = pca_explained_variance(sample_data, cumulative=True)

        assert "n_components_90" in variance_info
        assert "n_components_95" in variance_info
        assert "n_components_99" in variance_info

        # 90% threshold should use fewer components than 99%
        assert variance_info["n_components_90"] <= variance_info["n_components_99"]

    def test_no_cumulative(self, sample_data: Any) -> None:
        """Test without cumulative variance."""
        variance_info = pca_explained_variance(sample_data, cumulative=False)

        assert "explained_variance_ratio" in variance_info
        assert "cumulative_variance_ratio" not in variance_info
        assert "n_components_90" not in variance_info

    def test_variance_decreasing(self, sample_data: Any) -> None:
        """Test that variance is monotonically decreasing."""
        variance_info = pca_explained_variance(sample_data)
        var_ratio = variance_info["explained_variance_ratio"]

        # Each component should explain less or equal variance than previous
        assert np.all(np.diff(var_ratio) <= 0)


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_full_workflow(self, sample_data_with_labels: Any) -> None:
        """Test complete workflow: compute and analyze."""
        data, labels = sample_data_with_labels

        # Compute multiple embeddings
        embeddings = compute_multiple_embeddings(
            data, methods=["pca", "mds", "tsne"], n_components=2, random_state=42
        )

        # Analyze PCA variance
        variance_info = pca_explained_variance(data)

        # Verify results
        assert len(embeddings) >= 2
        assert "n_components_90" in variance_info

    def test_comparison_workflow(self, sample_data: Any) -> None:
        """Test workflow for comparing methods."""
        # Compute same data with different methods
        methods = ["pca", "mds", "isomap", "lle"]
        embeddings = compute_multiple_embeddings(
            sample_data, methods=methods, n_components=2, random_state=42
        )

        # Should successfully compute all available methods
        assert len(embeddings) >= 2

        # All embeddings should have same shape
        shapes = [emb.shape for emb in embeddings.values()]
        assert all(shape == (100, 2) for shape in shapes)
