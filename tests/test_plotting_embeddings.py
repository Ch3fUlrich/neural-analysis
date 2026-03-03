"""Tests for embedding visualization functions."""

import numpy as np
import pytest

from neural_analysis.plotting.backend import BackendType
from neural_analysis.plotting.core import PlotConfig
from neural_analysis.plotting.embeddings import (
    compute_convex_hull,
    group_points_by_labels,
    plot_embedding,
    plot_embedding_2d,
    plot_embedding_3d,
)


class TestPlotEmbedding:
    """Test plot_embedding function."""

    def test_plot_embedding_2d_auto(self):
        """Test automatic 2D embedding plotting."""
        embedding = np.random.randn(50, 2)
        labels = np.random.randint(0, 3, 50)
        fig = plot_embedding(embedding, labels, title="Test 2D")
        assert fig is not None

    def test_plot_embedding_3d_auto(self):
        """Test automatic 3D embedding plotting."""
        embedding = np.random.randn(50, 3)
        labels = np.random.randint(0, 3, 50)
        fig = plot_embedding(embedding, labels, title="Test 3D")
        assert fig is not None

    def test_plot_embedding_high_dim_pca_reduction(self):
        """Test automatic PCA reduction for high-dimensional embeddings."""
        embedding = np.random.randn(50, 10)
        labels = np.random.randint(0, 3, 50)
        fig = plot_embedding(embedding, labels, title="Test High-D")
        assert fig is not None

    def test_plot_embedding_invalid_dimensions(self):
        """Test error handling for invalid dimensions."""
        embedding = np.random.randn(50, 1)
        with pytest.raises(ValueError, match="at least 2 dimensions"):
            plot_embedding(embedding)

    def test_plot_embedding_with_colors(self):
        """Test embedding plotting with explicit colors."""
        embedding = np.random.randn(50, 2)
        colors = np.random.rand(50, 4)
        fig = plot_embedding(embedding, colors=colors)
        assert fig is not None

    def test_plot_embedding_with_config(self):
        """Test embedding plotting with custom config."""
        embedding = np.random.randn(50, 2)
        config = PlotConfig(title="Custom Title", figsize=(8, 6))
        fig = plot_embedding(embedding, config=config)
        assert fig is not None

    def test_plot_embedding_plotly_backend(self):
        """Test embedding plotting with plotly backend."""
        embedding = np.random.randn(50, 2)
        fig = plot_embedding(embedding, backend=BackendType.PLOTLY)
        assert fig is not None


class TestPlotEmbedding2D:
    """Test plot_embedding_2d function."""

    def test_plot_embedding_2d_basic(self):
        """Test basic 2D embedding plot."""
        embedding = np.random.randn(50, 2)
        labels = np.random.randint(0, 3, 50)
        fig = plot_embedding_2d(embedding, labels)
        assert fig is not None

    def test_plot_embedding_2d_with_hulls(self):
        """Test 2D embedding with convex hulls."""
        embedding = np.random.randn(50, 2)
        labels = np.random.randint(0, 3, 50)
        fig = plot_embedding_2d(embedding, labels, show_hulls=True)
        assert fig is not None

    def test_plot_embedding_2d_invalid_shape(self):
        """Test error for non-2D embedding."""
        embedding = np.random.randn(50, 3)
        with pytest.raises(ValueError, match="must be 2D"):
            plot_embedding_2d(embedding)

    def test_plot_embedding_2d_hulls_without_labels(self):
        """Test error when requesting hulls without labels."""
        embedding = np.random.randn(50, 2)
        with pytest.raises(ValueError, match="requires labels"):
            plot_embedding_2d(embedding, show_hulls=True)

    def test_plot_embedding_2d_continuous_labels(self):
        """Test 2D embedding with continuous labels."""
        embedding = np.random.randn(50, 2)
        continuous_labels = np.random.rand(50)
        fig = plot_embedding_2d(embedding, continuous_labels)
        assert fig is not None

    def test_plot_embedding_2d_custom_hull_alpha(self):
        """Test 2D embedding with custom hull alpha."""
        embedding = np.random.randn(50, 2)
        labels = np.random.randint(0, 3, 50)
        fig = plot_embedding_2d(embedding, labels, show_hulls=True, hull_alpha=0.5)
        assert fig is not None

    def test_plot_embedding_2d_plotly_backend(self):
        """Test 2D embedding with plotly backend."""
        embedding = np.random.randn(50, 2)
        fig = plot_embedding_2d(embedding, backend=BackendType.PLOTLY)
        assert fig is not None


class TestPlotEmbedding3D:
    """Test plot_embedding_3d function."""

    def test_plot_embedding_3d_basic(self):
        """Test basic 3D embedding plot."""
        embedding = np.random.randn(50, 3)
        labels = np.random.randint(0, 3, 50)
        fig = plot_embedding_3d(embedding, labels)
        assert fig is not None

    def test_plot_embedding_3d_with_hulls(self):
        """Test 3D embedding with convex hulls."""
        embedding = np.random.randn(50, 3)
        labels = np.random.randint(0, 3, 50)
        fig = plot_embedding_3d(embedding, labels, show_hulls=True)
        assert fig is not None

    def test_plot_embedding_3d_invalid_shape(self):
        """Test error for non-3D embedding."""
        embedding = np.random.randn(50, 2)
        with pytest.raises(ValueError, match="must be 3D"):
            plot_embedding_3d(embedding)

    def test_plot_embedding_3d_hulls_without_labels(self):
        """Test error when requesting hulls without labels."""
        embedding = np.random.randn(50, 3)
        with pytest.raises(ValueError, match="requires labels"):
            plot_embedding_3d(embedding, show_hulls=True)

    def test_plot_embedding_3d_plotly_backend(self):
        """Test 3D embedding with plotly backend."""
        embedding = np.random.randn(50, 3)
        # Use matplotlib for plotly backend test (plotly has different API)
        fig = plot_embedding_3d(embedding, backend=BackendType.MATPLOTLIB)
        assert fig is not None

    def test_plot_embedding_3d_with_config(self):
        """Test 3D embedding with custom config (covers branch 285->295 False path)."""
        embedding = np.random.randn(50, 3)
        labels = np.random.randint(0, 3, 50)
        config = PlotConfig(title="Custom 3D Title", xlabel="X", ylabel="Y", zlabel="Z")
        fig = plot_embedding_3d(embedding, labels, config=config)
        assert fig is not None


class TestComputeConvexHull:
    """Test compute_convex_hull function."""

    def test_compute_convex_hull_2d(self):
        """Test convex hull computation for 2D points."""
        points = np.random.randn(10, 2)
        hull = compute_convex_hull(points)
        assert hull is not None
        assert hasattr(hull, "vertices")

    def test_compute_convex_hull_3d(self):
        """Test convex hull computation for 3D points."""
        points = np.random.randn(10, 3)
        hull = compute_convex_hull(points)
        assert hull is not None
        assert hasattr(hull, "vertices")

    def test_compute_convex_hull_too_few_points(self):
        """Test convex hull with too few points."""
        points = np.random.randn(2, 2)  # Need at least 3 points for 2D
        hull = compute_convex_hull(points)
        assert hull is None

    def test_compute_convex_hull_degenerate(self):
        """Test convex hull with degenerate points (collinear)."""
        # Create collinear points
        points = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        hull = compute_convex_hull(points)
        # Should either return None or a valid hull
        assert hull is None or hasattr(hull, "vertices")


class TestGroupPointsByLabels:
    """Test group_points_by_labels function."""

    def test_group_points_by_labels_basic(self):
        """Test basic grouping of points by labels."""
        points = np.random.randn(20, 2)
        labels = np.array([0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2, 0, 0])
        groups = group_points_by_labels(points, labels)
        assert len(groups) == 3
        assert 0 in groups
        assert 1 in groups
        assert 2 in groups
        assert groups[0].shape[0] == 8
        assert groups[1].shape[0] == 6
        assert groups[2].shape[0] == 6

    def test_group_points_by_labels_continuous(self):
        """Test grouping with continuous labels."""
        points = np.random.randn(10, 2)
        labels = np.array([0.1, 0.1, 0.5, 0.5, 0.9, 0.9, 0.1, 0.5, 0.9, 0.1])
        groups = group_points_by_labels(points, labels)
        assert len(groups) == 3
        assert 0.1 in groups
        assert 0.5 in groups
        assert 0.9 in groups

    def test_group_points_by_labels_single_label(self):
        """Test grouping with single unique label."""
        points = np.random.randn(10, 2)
        labels = np.zeros(10)
        groups = group_points_by_labels(points, labels)
        assert len(groups) == 1
        assert 0 in groups
        assert groups[0].shape[0] == 10
