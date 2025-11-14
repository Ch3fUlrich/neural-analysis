"""
Tests for 1D plotting functions.

Tests cover:
- Basic line plotting
- Error bands
- Multiple lines
- Loss curves
- Boolean state visualization
- Both matplotlib and plotly backends
"""

from typing import Any

import matplotlib
import numpy as np
import pytest
from matplotlib.axes import Axes

matplotlib.use("Agg")  # Non-interactive backend for testing
import matplotlib.pyplot as plt

from neural_analysis.plotting.backend import set_backend
from neural_analysis.plotting.core import PlotConfig
from neural_analysis.plotting.plots_1d import (
    plot_boolean_states,
    plot_line,
    plot_multiple_lines,
)


@pytest.fixture
def sample_data_1d() -> Any:
    """Generate sample 1D data for testing."""
    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    y = np.sin(x) + np.random.normal(0, 0.1, 100)
    std = np.ones(100) * 0.2
    return x, y, std


@pytest.fixture
def sample_loss_data() -> Any:
    """Generate sample loss data."""
    return np.array([1.5, 1.2, 1.0, 0.8, 0.6, 0.5, 0.45, 0.42, 0.40, 0.39])


@pytest.fixture
def sample_boolean_states() -> Any:
    """Generate sample boolean states."""
    return np.array([True, True, False, False, False, True, True, True, False, True])


class TestPlotLine:
    """Tests for plot_line function."""

    def test_basic_line_plot(self, sample_data_1d: Any) -> None:
        """Test basic line plot creation."""
        x, y, _ = sample_data_1d
        config = PlotConfig(show=False)
        ax = plot_line(y, x=x, config=config, backend="matplotlib")

        assert isinstance(ax, Axes)
        assert len(ax.lines) == 1
        plt.close("all")

    def test_line_plot_with_std(self, sample_data_1d: Any) -> None:
        """Test line plot with error bands."""
        x, y, std = sample_data_1d
        config = PlotConfig(show=False)
        ax = plot_line(y, x=x, std=std, config=config, backend="matplotlib")

        assert isinstance(ax, Axes)
        assert len(ax.lines) == 1
        # Check that fill_between was called (creates a PolyCollection)
        assert len(ax.collections) > 0
        plt.close("all")

    def test_line_plot_with_markers(self, sample_data_1d: Any) -> None:
        """Test line plot with markers."""
        x, y, _ = sample_data_1d
        config = PlotConfig(show=False)
        ax = plot_line(
            y, x=x, config=config, marker="o", markersize=3, backend="matplotlib"
        )

        assert isinstance(ax, Axes)
        plt.close("all")

    def test_line_plot_with_label(self, sample_data_1d: Any) -> None:
        """Test line plot with label and legend."""
        x, y, _ = sample_data_1d
        config = PlotConfig(show=False, legend=True)
        ax = plot_line(y, x=x, config=config, label="Test Data", backend="matplotlib")

        legend = ax.get_legend()
        assert legend is not None
        assert legend.get_texts()[0].get_text() == "Test Data"
        plt.close("all")

    def test_line_plot_no_x(self, sample_data_1d: Any) -> None:
        """Test line plot without explicit x values."""
        _, y, _ = sample_data_1d
        config = PlotConfig(show=False)
        ax = plot_line(y, config=config, backend="matplotlib")

        assert isinstance(ax, Axes)
        # Check that x is range(len(y))
        line = ax.lines[0]
        x_data = line.get_xdata()
        assert len(x_data) == len(y)
        assert x_data[0] == 0
        assert x_data[-1] == len(y) - 1
        plt.close("all")

    def test_line_plot_configuration(self, sample_data_1d: Any) -> None:
        """Test that PlotConfig is applied correctly."""
        _, y, _ = sample_data_1d
        config = PlotConfig(
            title="Test Title",
            xlabel="X Label",
            ylabel="Y Label",
            xlim=(0, 50),
            ylim=(-2, 2),
            grid=True,
            show=False,
        )
        ax = plot_line(y, config=config, backend="matplotlib")

        assert ax.get_title() == "Test Title"
        assert ax.get_xlabel() == "X Label"
        assert ax.get_ylabel() == "Y Label"
        assert ax.get_xlim() == (0, 50)
        assert ax.get_ylim() == (-2, 2)
        plt.close("all")

    def test_line_plot_invalid_data(self) -> None:
        """Test error handling for invalid data."""
        with pytest.raises(ValueError, match="Data must be 1D"):
            data_2d = np.random.rand(10, 5)
            config = PlotConfig(show=False)
            plot_line(data_2d, config=config, backend="matplotlib")

        plt.close("all")

    def test_line_plot_mismatched_x_data(self, sample_data_1d: Any) -> None:
        """Test error handling for mismatched x and data lengths."""
        _, y, _ = sample_data_1d
        x_wrong = np.linspace(0, 10, 50)  # Different length

        with pytest.raises(ValueError, match="x and data must have same length"):
            config = PlotConfig(show=False)
            plot_line(y, x=x_wrong, config=config, backend="matplotlib")

        plt.close("all")

    def test_line_plot_custom_style(self, sample_data_1d: Any) -> None:
        """Test custom line styling."""
        _, y, _ = sample_data_1d
        config = PlotConfig(show=False)
        ax = plot_line(
            y,
            config=config,
            color="red",
            linewidth=3,
            linestyle="--",
            backend="matplotlib",
        )

        line = ax.lines[0]
        assert line.get_color() == "red"
        assert line.get_linewidth() == 3
        assert line.get_linestyle() == "--"
        plt.close("all")


class TestPlotMultipleLines:
    """Tests for plot_multiple_lines function."""

    def test_multiple_lines_basic(self) -> None:
        """Test plotting multiple lines."""
        x = np.linspace(0, 10, 100)
        data_dict = {"sine": np.sin(x), "cosine": np.cos(x), "tangent": np.tan(x)}
        config = PlotConfig(show=False)
        ax = plot_multiple_lines(data_dict, x=x, config=config, backend="matplotlib")

        assert isinstance(ax, Axes)
        assert len(ax.lines) == 3
        plt.close("all")

    def test_multiple_lines_with_colors(self) -> None:
        """Test multiple lines with custom colors."""
        x = np.linspace(0, 10, 100)
        data_dict = {
            "line1": np.sin(x),
            "line2": np.cos(x),
        }
        colors = ["red", "blue"]
        config = PlotConfig(show=False)
        ax = plot_multiple_lines(
            data_dict, x=x, config=config, colors=colors, backend="matplotlib"
        )

        assert ax.lines[0].get_color() == "red"
        assert ax.lines[1].get_color() == "blue"
        plt.close("all")

    def test_multiple_lines_legend(self) -> None:
        """Test legend with multiple lines."""
        x = np.linspace(0, 10, 100)
        data_dict = {
            "Line A": np.sin(x),
            "Line B": np.cos(x),
        }
        config = PlotConfig(show=False, legend=True)
        ax = plot_multiple_lines(data_dict, x=x, config=config, backend="matplotlib")

        legend = ax.get_legend()
        assert legend is not None
        legend_texts = [t.get_text() for t in legend.get_texts()]
        assert "Line A" in legend_texts
        assert "Line B" in legend_texts
        plt.close("all")


class TestPlotBooleanStates:
    """Tests for plot_boolean_states function."""

    def test_basic_boolean_plot(self, sample_boolean_states: Any) -> None:
        """Test basic boolean state visualization."""
        config = PlotConfig(show=False)
        ax = plot_boolean_states(
            sample_boolean_states, config=config, backend="matplotlib"
        )

        assert isinstance(ax, Axes)
        assert ax.get_ylim() == (0, 1)
        plt.close("all")

    def test_boolean_plot_custom_colors(self, sample_boolean_states: Any) -> None:
        """Test boolean plot with custom colors."""
        config = PlotConfig(show=False)
        ax = plot_boolean_states(
            sample_boolean_states,
            config=config,
            true_color="green",
            false_color="red",
            backend="matplotlib",
        )

        assert isinstance(ax, Axes)
        plt.close("all")

    def test_boolean_plot_custom_labels(self, sample_boolean_states: Any) -> None:
        """Test boolean plot with custom labels."""
        config = PlotConfig(show=False, legend=True)
        ax = plot_boolean_states(
            sample_boolean_states,
            config=config,
            true_label="Moving",
            false_label="Stationary",
            backend="matplotlib",
        )

        legend = ax.get_legend()
        assert legend is not None
        legend_texts = [t.get_text() for t in legend.get_texts()]
        assert "Moving" in legend_texts
        assert "Stationary" in legend_texts
        plt.close("all")

    def test_boolean_plot_with_x(self, sample_boolean_states: Any) -> None:
        """Test boolean plot with custom x values."""
        x = np.linspace(0, 100, len(sample_boolean_states))
        config = PlotConfig(show=False)
        ax = plot_boolean_states(
            sample_boolean_states, x=x, config=config, backend="matplotlib"
        )

        assert isinstance(ax, Axes)
        plt.close("all")

    def test_boolean_all_true(self) -> None:
        """Test with all True values."""
        states = np.ones(10, dtype=bool)
        config = PlotConfig(show=False)
        ax = plot_boolean_states(states, config=config, backend="matplotlib")

        assert isinstance(ax, Axes)
        plt.close("all")

    def test_boolean_all_false(self) -> None:
        """Test with all False values."""
        states = np.zeros(10, dtype=bool)
        config = PlotConfig(show=False)
        ax = plot_boolean_states(states, config=config, backend="matplotlib")

        assert isinstance(ax, Axes)
        plt.close("all")


class TestBackendSelection:
    """Tests for backend selection."""

    def test_matplotlib_backend(self, sample_data_1d: Any) -> None:
        """Test explicit matplotlib backend selection."""
        _, y, _ = sample_data_1d
        config = PlotConfig(show=False)
        result = plot_line(y, config=config, backend="matplotlib")

        assert isinstance(result, Axes)
        plt.close("all")

    def test_global_backend_used(self, sample_data_1d: Any) -> None:
        """Test that global backend setting is used."""
        _, y, _ = sample_data_1d
        set_backend("matplotlib")
        config = PlotConfig(show=False)
        result = plot_line(y, config=config)  # No explicit backend

        assert isinstance(result, Axes)
        plt.close("all")


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_data(self) -> None:
        """Test with empty data array."""
        data = np.array([])
        config = PlotConfig(show=False)

        # Should still work, just produce empty plot
        ax = plot_line(data, config=config, backend="matplotlib")
        assert isinstance(ax, Axes)
        plt.close("all")

    def test_single_point(self) -> None:
        """Test with single data point."""
        data = np.array([5.0])
        config = PlotConfig(show=False)
        ax = plot_line(data, config=config, backend="matplotlib")

        assert isinstance(ax, Axes)
        plt.close("all")

    def test_nan_values(self) -> None:
        """Test with NaN values in data."""
        data = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        config = PlotConfig(show=False)
        ax = plot_line(data, config=config, backend="matplotlib")

        # Matplotlib should handle NaN by breaking the line
        assert isinstance(ax, Axes)
        plt.close("all")

    def test_inf_values(self) -> None:
        """Test with infinite values in data."""
        data = np.array([1.0, 2.0, np.inf, 4.0, 5.0])
        config = PlotConfig(show=False)
        ax = plot_line(data, config=config, backend="matplotlib")

        assert isinstance(ax, Axes)
        plt.close("all")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
