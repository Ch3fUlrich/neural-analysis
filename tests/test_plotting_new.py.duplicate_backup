"""
Tests for heatmap and subplot plotting functions.
"""

import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt
import warnings

from neural_analysis.plotting import (
    plot_heatmap,
    create_subplot_grid,
    add_trace_to_subplot,
    PlotConfig,
)
from neural_analysis.plotting.backend import BackendType

# Suppress non-interactive backend warnings in tests
warnings.filterwarnings("ignore", message=".*non-interactive.*")
warnings.filterwarnings("ignore", message=".*FigureCanvasAgg.*")

# Check if plotly is available
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class TestHeatmap:
    """Tests for heatmap plotting functions."""

    def test_plot_heatmap_matplotlib(self):
        """Test basic heatmap with matplotlib."""
        data = np.random.rand(5, 5)
        config = PlotConfig(title="Test Heatmap")
        
        ax = plot_heatmap(
            data,
            config=config,
            x_labels=[f"X{i}" for i in range(5)],
            y_labels=[f"Y{i}" for i in range(5)],
            backend='matplotlib'
        )
        
        assert ax is not None
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close('all')

    @pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not installed")
    def test_plot_heatmap_plotly(self):
        """Test basic heatmap with plotly."""
        data = np.random.rand(5, 5)
        config = PlotConfig(title="Test Heatmap")
        
        fig = plot_heatmap(
            data,
            config=config,
            x_labels=[f"X{i}" for i in range(5)],
            y_labels=[f"Y{i}" for i in range(5)],
            backend='plotly'
        )
        
        assert fig is not None
        assert isinstance(fig, go.Figure)

    def test_plot_heatmap_with_values(self):
        """Test heatmap with value annotations."""
        data = np.array([[1.23, 4.56], [7.89, 0.12]])
        
        ax = plot_heatmap(
            data,
            show_values=True,
            value_format=".1f",
            backend='matplotlib'
        )
        
        assert ax is not None
        plt.close('all')

    def test_heatmap_invalid_method_raises(self):
        """Test that invalid method raises error."""
        # This test is for the old plot_correlation_matrix function
        # which has been removed. Keeping as placeholder for future validation tests.
        pass

    def test_heatmap_non_2d_raises(self):
        """Test that non-2D data raises error."""
        data = np.random.rand(5)
        
        with pytest.raises(ValueError, match="must be 2D"):
            plot_heatmap(data, backend='matplotlib')


class TestSubplots:
    """Tests for subplot creation and management."""

    def test_create_subplot_grid_matplotlib(self):
        """Test basic subplot grid with matplotlib."""
        fig, axes = create_subplot_grid(
            rows=2,
            cols=2,
            config=PlotConfig(title="Test Grid"),
            backend='matplotlib'
        )
        
        assert fig is not None
        assert isinstance(fig, matplotlib.figure.Figure)
        assert len(axes) == 4
        assert all(isinstance(ax, matplotlib.axes.Axes) for ax in axes)
        plt.close('all')

    @pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not installed")
    def test_create_subplot_grid_plotly(self):
        """Test basic subplot grid with plotly."""
        fig = create_subplot_grid(
            rows=2,
            cols=2,
            config=PlotConfig(title="Test Grid"),
            backend='plotly'
        )
        
        assert fig is not None
        assert isinstance(fig, go.Figure)

    @pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not installed")
    def test_create_subplot_grid_with_titles(self):
        """Test subplot grid with titles."""
        titles = ["A", "B", "C", "D"]
        fig = create_subplot_grid(
            rows=2,
            cols=2,
            subplot_titles=titles,
            backend='plotly'
        )
        
        assert fig is not None

    @pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not installed")
    def test_add_trace_to_subplot(self):
        """Test adding traces to subplots."""
        fig = create_subplot_grid(
            rows=1,
            cols=2,
            backend='plotly'
        )
        
        trace = go.Scatter(x=[1, 2, 3], y=[1, 4, 9], mode='markers')
        fig_updated = add_trace_to_subplot(fig, trace, row=1, col=1)
        
        assert fig_updated is not None
        assert len(fig_updated.data) == 1

    @pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not installed")
    def test_add_trace_invalid_fig_raises(self):
        """Test that invalid figure raises error."""
        with pytest.raises(TypeError, match="must be a plotly"):
            add_trace_to_subplot("not a figure", go.Scatter(), row=1, col=1)

    def test_create_subplot_grid_shared_axes_matplotlib(self):
        """Test shared axes in matplotlib."""
        fig, axes = create_subplot_grid(
            rows=2,
            cols=2,
            shared_xaxes=True,
            shared_yaxes=True,
            backend='matplotlib'
        )
        
        assert fig is not None
        assert len(axes) == 4
        plt.close('all')

    @pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not installed")
    def test_create_subplot_grid_shared_axes_plotly(self):
        """Test shared axes in plotly."""
        fig = create_subplot_grid(
            rows=2,
            cols=2,
            shared_xaxes='all',
            shared_yaxes='all',
            backend='plotly'
        )
        
        assert fig is not None

    @pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not installed")
    def test_create_subplot_grid_with_specs(self):
        """Test subplot grid with custom specs."""
        specs = [
            [{'type': 'scatter'}, {'type': 'scatter'}],
            [{'type': 'scatter3d', 'colspan': 2}, None]
        ]
        
        fig = create_subplot_grid(
            rows=2,
            cols=2,
            specs=specs,
            backend='plotly'
        )
        
        assert fig is not None


class TestIntegration:
    """Integration tests combining multiple features."""

    @pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not installed")
    def test_distance_matrix_in_subplot(self):
        """Test plotting heatmap within a subplot."""
        # This tests the interaction between modules
        
        # Generate sample data
        data = np.random.rand(5, 5)
        
        # Create subplot grid
        fig = create_subplot_grid(rows=1, cols=2, backend='plotly')
        
        # Plot heatmap (it creates its own figure, but we test it doesn't crash)
        heatmap_fig = plot_heatmap(
            data,
            backend='plotly'
        )
        
        assert heatmap_fig is not None

    def test_matplotlib_workflow(self):
        """Test complete matplotlib workflow."""
        # Create data
        data = np.random.rand(5, 5)
        
        # Create subplot grid
        fig, axes = create_subplot_grid(rows=1, cols=2, backend='matplotlib')
        
        # Plot heatmap in second position (manual)
        ax_heat = plot_heatmap(
            data,
            config=PlotConfig(title="Heatmap"),
            backend='matplotlib'
        )
        
        assert ax_heat is not None
        plt.close('all')
