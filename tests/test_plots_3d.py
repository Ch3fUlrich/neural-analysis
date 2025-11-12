"""Tests for 3D plotting functions."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from neural_analysis.plotting import (
    PlotConfig,
    plot_scatter_3d,
    plot_trajectory_3d,
    set_backend,
)


class TestPlotScatter3D:
    """Tests for plot_scatter_3d function."""

    def test_basic_scatter_matplotlib(self) -> None:
        """Test basic 3D scatter plot with matplotlib."""
        x = np.random.randn(50)
        y = np.random.randn(50)
        z = np.random.randn(50)

        config = PlotConfig(show=False, title="Test 3D Scatter")
        ax = plot_scatter_3d(x, y, z, config=config, backend="matplotlib")

        assert ax is not None
        assert hasattr(ax, "get_xlabel")
        plt.close("all")

    def test_scatter_with_color_array(self) -> None:
        """Test scatter with color array."""
        x = np.random.randn(50)
        y = np.random.randn(50)
        z = np.random.randn(50)
        colors = np.random.rand(50)

        config = PlotConfig(show=False)
        ax = plot_scatter_3d(
            x, y, z, colors=colors, cmap="viridis", config=config, backend="matplotlib"
        )

        assert ax is not None
        plt.close("all")

    def test_scatter_mismatched_lengths(self) -> None:
        """Test error on mismatched array lengths."""
        x = np.random.randn(50)
        y = np.random.randn(40)
        z = np.random.randn(50)

        with pytest.raises(ValueError, match="must have same length"):
            plot_scatter_3d(x, y, z, backend="matplotlib")

        plt.close("all")


class TestPlotTrajectory3D:
    """Tests for plot_trajectory_3d function."""

    def test_basic_trajectory_matplotlib(self) -> None:
        """Test basic 3D trajectory with matplotlib."""
        t = np.linspace(0, 4 * np.pi, 100)
        x = np.sin(t)
        y = np.cos(t)
        z = t / 4

        config = PlotConfig(show=False, title="Test 3D Trajectory")
        ax = plot_trajectory_3d(x, y, z, config=config, backend="matplotlib")

        assert ax is not None
        plt.close("all")

    def test_trajectory_color_by_time(self) -> None:
        """Test trajectory with time-based coloring."""
        t = np.linspace(0, 2 * np.pi, 50)
        x = np.sin(t)
        y = np.cos(t)
        z = t

        config = PlotConfig(show=False)
        ax = plot_trajectory_3d(
            x, y, z, color_by="time", cmap="plasma", config=config, backend="matplotlib"
        )

        assert ax is not None
        plt.close("all")

    def test_trajectory_without_points(self) -> None:
        """Test trajectory without scatter points."""
        t = np.linspace(0, 2 * np.pi, 50)
        x = np.sin(t)
        y = np.cos(t)
        z = t

        config = PlotConfig(show=False)
        ax = plot_trajectory_3d(
            x, y, z, show_points=False, config=config, backend="matplotlib"
        )

        assert ax is not None
        plt.close("all")

    def test_trajectory_mismatched_lengths(self) -> None:
        """Test error on mismatched array lengths."""
        x = np.random.randn(50)
        y = np.random.randn(40)
        z = np.random.randn(50)

        with pytest.raises(ValueError, match="must have same length"):
            plot_trajectory_3d(x, y, z, backend="matplotlib")

        plt.close("all")


class TestBackendSelection:
    """Tests for backend selection."""

    def test_scatter_backend_override(self) -> None:
        """Test backend override for scatter."""
        set_backend("matplotlib")

        x = np.random.randn(20)
        y = np.random.randn(20)
        z = np.random.randn(20)

        config = PlotConfig(show=False)
        ax = plot_scatter_3d(x, y, z, config=config, backend="matplotlib")

        assert ax is not None
        plt.close("all")


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_point_scatter(self) -> None:
        """Test scatter with single point."""
        x = np.array([1.0])
        y = np.array([2.0])
        z = np.array([3.0])

        config = PlotConfig(show=False)
        ax = plot_scatter_3d(x, y, z, config=config, backend="matplotlib")

        assert ax is not None
        plt.close("all")

    def test_large_dataset(self) -> None:
        """Test with larger dataset."""
        n = 5000
        x = np.random.randn(n)
        y = np.random.randn(n)
        z = np.random.randn(n)

        config = PlotConfig(show=False)
        ax = plot_scatter_3d(x, y, z, config=config, backend="matplotlib")

        assert ax is not None
        plt.close("all")
