"""Tests for 2D plotting functions."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

# Test both backends
from neural_analysis.plotting import (
    PlotConfig,
    plot_grouped_scatter_2d,
    plot_kde_2d,
    plot_scatter_2d,
    plot_trajectory_2d,
    set_backend,
)

try:
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class TestPlotScatter2D:
    """Tests for plot_scatter_2d function."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample 2D data."""
        np.random.seed(42)
        x = np.random.randn(50)
        y = np.random.randn(50)
        return x, y

    def test_basic_scatter_matplotlib(self, sample_data):
        """Test basic scatter plot with matplotlib."""
        set_backend("matplotlib")
        x, y = sample_data
        ax = plot_scatter_2d(x, y, PlotConfig(show=False))
        assert isinstance(ax, plt.Axes)
        plt.close()

    @pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not available")
    def test_basic_scatter_plotly(self, sample_data):
        """Test basic scatter plot with plotly."""
        x, y = sample_data
        fig = plot_scatter_2d(x, y, backend="plotly", config=PlotConfig(show=False))
        assert isinstance(fig, go.Figure)

    def test_scatter_with_color_array(self, sample_data):
        """Test scatter with color array."""
        set_backend("matplotlib")
        x, y = sample_data
        colors = np.arange(len(x))
        ax = plot_scatter_2d(
            x, y, colors=colors, colorbar=True, config=PlotConfig(show=False)
        )
        assert isinstance(ax, plt.Axes)
        plt.close()

    def test_scatter_with_color_string(self, sample_data):
        """Test scatter with single color."""
        set_backend("matplotlib")
        x, y = sample_data
        ax = plot_scatter_2d(x, y, colors="red", config=PlotConfig(show=False))
        assert isinstance(ax, plt.Axes)
        plt.close()

    def test_scatter_with_variable_sizes(self, sample_data):
        """Test scatter with variable point sizes."""
        set_backend("matplotlib")
        x, y = sample_data
        sizes = np.random.uniform(10, 50, len(x))
        ax = plot_scatter_2d(x, y, sizes=sizes, config=PlotConfig(show=False))
        assert isinstance(ax, plt.Axes)
        plt.close()

    def test_scatter_with_config(self, sample_data):
        """Test scatter with plot configuration."""
        set_backend("matplotlib")
        x, y = sample_data
        config = PlotConfig(
            title="Test Scatter",
            xlabel="X axis",
            ylabel="Y axis",
            show=False,
            grid=True,
        )
        ax = plot_scatter_2d(x, y, config=config)
        assert ax.get_title() == "Test Scatter"
        assert ax.get_xlabel() == "X axis"
        assert ax.get_ylabel() == "Y axis"
        plt.close()

    def test_scatter_mismatched_lengths(self):
        """Test that mismatched x and y lengths raise error."""
        x = np.array([1, 2, 3])
        y = np.array([1, 2])
        with pytest.raises(ValueError, match="x and y must have same length"):
            plot_scatter_2d(x, y)

    def test_scatter_with_colorbar_label(self, sample_data):
        """Test scatter with colorbar label."""
        set_backend("matplotlib")
        x, y = sample_data
        colors = np.arange(len(x))
        ax = plot_scatter_2d(
            x,
            y,
            colors=colors,
            colorbar=True,
            colorbar_label="Values",
            config=PlotConfig(show=False),
        )
        assert isinstance(ax, plt.Axes)
        plt.close()

    def test_scatter_with_custom_alpha(self, sample_data):
        """Test scatter with custom transparency."""
        set_backend("matplotlib")
        x, y = sample_data
        ax = plot_scatter_2d(x, y, alpha=0.3, config=PlotConfig(show=False))
        assert isinstance(ax, plt.Axes)
        plt.close()


class TestPlotTrajectory2D:
    """Tests for plot_trajectory_2d function."""

    @pytest.fixture
    def trajectory_data(self):
        """Generate sample trajectory data."""
        t = np.linspace(0, 4 * np.pi, 100)
        x = np.sin(t)
        y = np.cos(t)
        return x, y

    def test_basic_trajectory_matplotlib(self, trajectory_data):
        """Test basic trajectory plot with matplotlib."""
        set_backend("matplotlib")
        x, y = trajectory_data
        ax = plot_trajectory_2d(x, y, PlotConfig(show=False))
        assert isinstance(ax, plt.Axes)
        plt.close()

    @pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not available")
    def test_basic_trajectory_plotly(self, trajectory_data):
        """Test basic trajectory plot with plotly."""
        x, y = trajectory_data
        fig = plot_trajectory_2d(
            x, y, backend="plotly", config=PlotConfig(show=False)
        )
        assert isinstance(fig, go.Figure)

    def test_trajectory_color_by_time(self, trajectory_data):
        """Test trajectory with time-based coloring."""
        set_backend("matplotlib")
        x, y = trajectory_data
        ax = plot_trajectory_2d(
            x, y, color_by="time", config=PlotConfig(show=False)
        )
        assert isinstance(ax, plt.Axes)
        plt.close()

    def test_trajectory_without_points(self, trajectory_data):
        """Test trajectory without scatter points."""
        set_backend("matplotlib")
        x, y = trajectory_data
        ax = plot_trajectory_2d(
            x, y, show_points=False, config=PlotConfig(show=False)
        )
        assert isinstance(ax, plt.Axes)
        plt.close()

    def test_trajectory_custom_linewidth(self, trajectory_data):
        """Test trajectory with custom linewidth."""
        set_backend("matplotlib")
        x, y = trajectory_data
        ax = plot_trajectory_2d(x, y, linewidth=3, config=PlotConfig(show=False))
        assert isinstance(ax, plt.Axes)
        plt.close()

    def test_trajectory_without_time_color(self, trajectory_data):
        """Test trajectory without time-based coloring."""
        set_backend("matplotlib")
        x, y = trajectory_data
        ax = plot_trajectory_2d(
            x, y, color_by=None, config=PlotConfig(show=False)
        )
        assert isinstance(ax, plt.Axes)
        plt.close()

    def test_trajectory_mismatched_lengths(self):
        """Test that mismatched x and y lengths raise error."""
        x = np.array([1, 2, 3])
        y = np.array([1, 2])
        with pytest.raises(ValueError, match="x and y must have same length"):
            plot_trajectory_2d(x, y)


class TestPlotGroupedScatter2D:
    """Tests for plot_grouped_scatter_2d function."""

    @pytest.fixture
    def grouped_data(self):
        """Generate sample grouped data."""
        np.random.seed(42)
        groups = {
            "A": (np.random.randn(30), np.random.randn(30)),
            "B": (np.random.randn(30) + 3, np.random.randn(30) + 3),
            "C": (np.random.randn(30) - 3, np.random.randn(30) + 3),
        }
        return groups

    def test_basic_grouped_scatter_matplotlib(self, grouped_data):
        """Test basic grouped scatter with matplotlib."""
        set_backend("matplotlib")
        ax = plot_grouped_scatter_2d(grouped_data, PlotConfig(show=False))
        assert isinstance(ax, plt.Axes)
        plt.close()

    @pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not available")
    def test_basic_grouped_scatter_plotly(self, grouped_data):
        """Test basic grouped scatter with plotly."""
        fig = plot_grouped_scatter_2d(
            grouped_data, backend="plotly", config=PlotConfig(show=False)
        )
        assert isinstance(fig, go.Figure)

    def test_grouped_scatter_with_hulls(self, grouped_data):
        """Test grouped scatter with convex hulls."""
        set_backend("matplotlib")
        ax = plot_grouped_scatter_2d(
            grouped_data, show_hulls=True, config=PlotConfig(show=False)
        )
        assert isinstance(ax, plt.Axes)
        plt.close()

    def test_grouped_scatter_custom_colors(self, grouped_data):
        """Test grouped scatter with custom colors."""
        set_backend("matplotlib")
        colors = ["red", "blue", "green"]
        ax = plot_grouped_scatter_2d(
            grouped_data, colors=colors, config=PlotConfig(show=False)
        )
        assert isinstance(ax, plt.Axes)
        plt.close()

    def test_grouped_scatter_empty_dict(self):
        """Test that empty group_data raises error."""
        with pytest.raises(ValueError, match="group_data cannot be empty"):
            plot_grouped_scatter_2d({})

    def test_grouped_scatter_mismatched_group(self):
        """Test that mismatched x,y in a group raises error."""
        groups = {"A": (np.array([1, 2, 3]), np.array([1, 2]))}
        with pytest.raises(ValueError, match="x and y must have same length"):
            plot_grouped_scatter_2d(groups)

    def test_grouped_scatter_small_groups(self):
        """Test grouped scatter with groups too small for hulls."""
        groups = {
            "A": (np.array([1, 2]), np.array([1, 2])),  # Only 2 points
            "B": (np.array([3, 4, 5]), np.array([3, 4, 5])),  # 3 points
        }
        set_backend("matplotlib")
        # Should not raise error, just skip hulls for small groups
        ax = plot_grouped_scatter_2d(
            groups, show_hulls=True, config=PlotConfig(show=False)
        )
        assert isinstance(ax, plt.Axes)
        plt.close()

    def test_grouped_scatter_with_config(self, grouped_data):
        """Test grouped scatter with plot configuration."""
        set_backend("matplotlib")
        config = PlotConfig(
            title="Test Grouped Scatter",
            xlabel="X",
            ylabel="Y",
            show=False,
        )
        ax = plot_grouped_scatter_2d(grouped_data, config=config)
        assert ax.get_title() == "Test Grouped Scatter"
        plt.close()


class TestPlotKDE2D:
    """Tests for plot_kde_2d function."""

    @pytest.fixture
    def kde_data(self):
        """Generate sample data for KDE."""
        np.random.seed(42)
        x = np.random.randn(200)
        y = np.random.randn(200)
        return x, y

    def test_basic_kde_matplotlib(self, kde_data):
        """Test basic KDE plot with matplotlib."""
        set_backend("matplotlib")
        x, y = kde_data
        ax = plot_kde_2d(x, y, PlotConfig(show=False))
        assert isinstance(ax, plt.Axes)
        plt.close()

    @pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not available")
    def test_basic_kde_plotly(self, kde_data):
        """Test basic KDE plot with plotly."""
        x, y = kde_data
        fig = plot_kde_2d(x, y, backend="plotly", config=PlotConfig(show=False))
        assert isinstance(fig, go.Figure)

    def test_kde_with_points(self, kde_data):
        """Test KDE with underlying points shown."""
        set_backend("matplotlib")
        x, y = kde_data
        ax = plot_kde_2d(x, y, show_points=True, config=PlotConfig(show=False))
        assert isinstance(ax, plt.Axes)
        plt.close()

    def test_kde_contour_only(self, kde_data):
        """Test KDE with contours only (no fill)."""
        set_backend("matplotlib")
        x, y = kde_data
        ax = plot_kde_2d(x, y, fill=False, config=PlotConfig(show=False))
        assert isinstance(ax, plt.Axes)
        plt.close()

    def test_kde_custom_levels(self, kde_data):
        """Test KDE with custom number of levels."""
        set_backend("matplotlib")
        x, y = kde_data
        ax = plot_kde_2d(x, y, n_levels=15, config=PlotConfig(show=False))
        assert isinstance(ax, plt.Axes)
        plt.close()

    def test_kde_mismatched_lengths(self):
        """Test that mismatched x and y lengths raise error."""
        x = np.array([1, 2, 3])
        y = np.array([1, 2])
        with pytest.raises(ValueError, match="x and y must have same length"):
            plot_kde_2d(x, y)

    def test_kde_insufficient_points(self):
        """Test that too few points raise error."""
        x = np.array([1])
        y = np.array([1])
        with pytest.raises(ValueError, match="Need at least 2 points for KDE"):
            plot_kde_2d(x, y)


class TestBackendSelection:
    """Tests for backend selection across 2D functions."""

    def test_scatter_backend_override(self):
        """Test backend override for scatter plot."""
        set_backend("matplotlib")  # Set default to matplotlib
        x = np.random.randn(10)
        y = np.random.randn(10)

        # Override with plotly
        if PLOTLY_AVAILABLE:
            fig = plot_scatter_2d(x, y, backend="plotly", config=PlotConfig(show=False))
            assert isinstance(fig, go.Figure)

    def test_invalid_backend_raises_error(self):
        """Test that invalid backend raises error."""
        x = np.random.randn(10)
        y = np.random.randn(10)

        # This should raise a ValueError during BackendType instantiation
        with pytest.raises(ValueError):
            plot_scatter_2d(x, y, backend="invalid")  # type: ignore


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_arrays(self):
        """Test behavior with empty arrays."""
        x = np.array([])
        y = np.array([])
        # Should work but return empty plot
        set_backend("matplotlib")
        ax = plot_scatter_2d(x, y, config=PlotConfig(show=False))
        assert isinstance(ax, plt.Axes)
        plt.close()

    def test_single_point_scatter(self):
        """Test scatter with single point."""
        x = np.array([1.0])
        y = np.array([2.0])
        set_backend("matplotlib")
        ax = plot_scatter_2d(x, y, config=PlotConfig(show=False))
        assert isinstance(ax, plt.Axes)
        plt.close()

    def test_single_point_trajectory(self):
        """Test trajectory with single point."""
        x = np.array([1.0])
        y = np.array([2.0])
        set_backend("matplotlib")
        ax = plot_trajectory_2d(x, y, config=PlotConfig(show=False))
        assert isinstance(ax, plt.Axes)
        plt.close()

    def test_large_dataset(self):
        """Test with large dataset."""
        np.random.seed(42)
        x = np.random.randn(10000)
        y = np.random.randn(10000)
        set_backend("matplotlib")
        ax = plot_scatter_2d(x, y, config=PlotConfig(show=False))
        assert isinstance(ax, plt.Axes)
        plt.close()

    def test_inf_and_nan_values(self):
        """Test handling of inf and nan values."""
        x = np.array([1, 2, np.inf, 4, np.nan])
        y = np.array([1, 2, 3, np.inf, np.nan])
        set_backend("matplotlib")
        # Matplotlib should handle these gracefully
        ax = plot_scatter_2d(x, y, config=PlotConfig(show=False))
        assert isinstance(ax, plt.Axes)
        plt.close()
