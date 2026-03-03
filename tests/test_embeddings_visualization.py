"""Tests for embeddings visualization functions."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from neural_analysis.embeddings.visualization import (
    plot_multiple_embeddings,
    plot_pca_variance,
)


class TestPlotMultipleEmbeddings:
    """Tests for plot_multiple_embeddings function."""

    def test_plot_multiple_embeddings_empty_dict(self) -> None:
        """Test plot_multiple_embeddings with empty embeddings dict (covers line 138-139)."""
        with pytest.raises(ValueError, match="No embeddings provided"):
            plot_multiple_embeddings({})

    def test_plot_multiple_embeddings_invalid_shape(self) -> None:
        """Test plot_multiple_embeddings with invalid embedding shape (covers lines 182-185)."""
        embeddings = {"pca": np.array([1, 2, 3])}  # 1D instead of 2D
        with pytest.raises(ValueError, match="must be 2D"):
            plot_multiple_embeddings(embeddings)

    def test_plot_multiple_embeddings_unsupported_dims(self) -> None:
        """Test plot_multiple_embeddings with unsupported dimensions (covers lines 189-194)."""
        embeddings = {"pca": np.random.randn(10, 4)}  # 4D embedding
        # Should warn and skip this embedding
        with pytest.raises(ValueError, match="No valid embeddings"):
            plot_multiple_embeddings(embeddings)

    @patch("neural_analysis.embeddings.visualization.PlotGrid")
    def test_plot_multiple_embeddings_basic(self, mock_plot_grid: MagicMock) -> None:
        """Test plot_multiple_embeddings with basic 2D embeddings (covers main path)."""
        mock_fig = MagicMock()
        mock_grid_instance = MagicMock()
        mock_grid_instance.plot.return_value = mock_fig
        mock_plot_grid.return_value = mock_grid_instance

        embeddings = {
            "pca": np.random.randn(50, 2),
            "tsne": np.random.randn(50, 2),
        }

        result = plot_multiple_embeddings(embeddings, show_legend=False)
        assert result == mock_fig
        mock_plot_grid.assert_called_once()

    @patch("neural_analysis.embeddings.visualization.PlotGrid")
    def test_plot_multiple_embeddings_with_labels(
        self, mock_plot_grid: MagicMock
    ) -> None:
        """Test plot_multiple_embeddings with labels (covers lines 153-174)."""
        mock_fig = MagicMock()
        mock_grid_instance = MagicMock()
        mock_grid_instance.plot.return_value = mock_fig
        mock_plot_grid.return_value = mock_grid_instance

        embeddings = {"pca": np.random.randn(50, 2)}
        labels = np.array([0, 0, 1, 1] * 12 + [0, 0])  # 50 labels

        result = plot_multiple_embeddings(embeddings, labels=labels)
        assert result == mock_fig
        mock_plot_grid.assert_called_once()

    @patch("neural_analysis.embeddings.visualization.PlotGrid")
    def test_plot_multiple_embeddings_with_colors_list(
        self, mock_plot_grid: MagicMock
    ) -> None:
        """Test plot_multiple_embeddings with color list (covers line 168-169)."""
        mock_fig = MagicMock()
        mock_grid_instance = MagicMock()
        mock_grid_instance.plot.return_value = mock_fig
        mock_plot_grid.return_value = mock_grid_instance

        embeddings = {"pca": np.random.randn(50, 2)}
        labels = np.array([0, 1] * 25)
        colors = ["red", "blue"]

        result = plot_multiple_embeddings(embeddings, labels=labels, colors=colors)
        assert result == mock_fig

    @patch("neural_analysis.embeddings.visualization.PlotGrid")
    def test_plot_multiple_embeddings_with_colors_str(
        self, mock_plot_grid: MagicMock
    ) -> None:
        """Test plot_multiple_embeddings with single color string (covers line 169)."""
        mock_fig = MagicMock()
        mock_grid_instance = MagicMock()
        mock_grid_instance.plot.return_value = mock_fig
        mock_plot_grid.return_value = mock_grid_instance

        embeddings = {"pca": np.random.randn(50, 2)}
        labels = np.array([0, 1] * 25)
        colors = "steelblue"

        result = plot_multiple_embeddings(embeddings, labels=labels, colors=colors)
        assert result == mock_fig

    @patch("neural_analysis.embeddings.visualization.PlotGrid")
    def test_plot_multiple_embeddings_no_labels_custom_colors(
        self, mock_plot_grid: MagicMock
    ) -> None:
        """Test plot_multiple_embeddings without labels but with custom colors (covers branch 172->174 False path)."""
        mock_fig = MagicMock()
        mock_grid_instance = MagicMock()
        mock_grid_instance.plot.return_value = mock_fig
        mock_plot_grid.return_value = mock_grid_instance

        embeddings = {"pca": np.random.randn(50, 2)}
        colors = "red"  # Custom color, not None

        result = plot_multiple_embeddings(embeddings, colors=colors)
        assert result == mock_fig

    @patch("neural_analysis.embeddings.visualization.PlotGrid")
    def test_plot_multiple_embeddings_3d(self, mock_plot_grid: MagicMock) -> None:
        """Test plot_multiple_embeddings with 3D embeddings (covers 3D path)."""
        mock_fig = MagicMock()
        mock_grid_instance = MagicMock()
        mock_grid_instance.plot.return_value = mock_fig
        mock_plot_grid.return_value = mock_grid_instance

        embeddings = {"pca": np.random.randn(50, 3)}

        result = plot_multiple_embeddings(embeddings)
        assert result == mock_fig

    @patch("neural_analysis.embeddings.visualization.PlotGrid")
    def test_plot_multiple_embeddings_grid_layout_1_2(
        self, mock_plot_grid: MagicMock
    ) -> None:
        """Test grid layout for 1-2 embeddings (covers lines 143-144)."""
        mock_fig = MagicMock()
        mock_grid_instance = MagicMock()
        mock_grid_instance.plot.return_value = mock_fig
        mock_plot_grid.return_value = mock_grid_instance

        embeddings = {"pca": np.random.randn(50, 2)}
        plot_multiple_embeddings(embeddings)

        # Check layout was set correctly
        call_args = mock_plot_grid.call_args
        assert call_args is not None
        layout = call_args.kwargs.get("layout")
        assert layout is not None
        assert layout.rows == 1
        assert layout.cols == 1

    @patch("neural_analysis.embeddings.visualization.PlotGrid")
    def test_plot_multiple_embeddings_grid_layout_3_4(
        self, mock_plot_grid: MagicMock
    ) -> None:
        """Test grid layout for 3-4 embeddings (covers lines 145-146)."""
        mock_fig = MagicMock()
        mock_grid_instance = MagicMock()
        mock_grid_instance.plot.return_value = mock_fig
        mock_plot_grid.return_value = mock_grid_instance

        embeddings = {
            "pca": np.random.randn(50, 2),
            "tsne": np.random.randn(50, 2),
            "umap": np.random.randn(50, 2),
        }
        plot_multiple_embeddings(embeddings)

        call_args = mock_plot_grid.call_args
        assert call_args is not None
        layout = call_args.kwargs.get("layout")
        assert layout is not None
        assert layout.rows == 2
        assert layout.cols == 2

    @patch("neural_analysis.embeddings.visualization.PlotGrid")
    def test_plot_multiple_embeddings_grid_layout_5_6(
        self, mock_plot_grid: MagicMock
    ) -> None:
        """Test grid layout for 5-6 embeddings (covers lines 147-148)."""
        mock_fig = MagicMock()
        mock_grid_instance = MagicMock()
        mock_grid_instance.plot.return_value = mock_fig
        mock_plot_grid.return_value = mock_grid_instance

        embeddings = {f"method_{i}": np.random.randn(50, 2) for i in range(5)}
        plot_multiple_embeddings(embeddings)

        call_args = mock_plot_grid.call_args
        assert call_args is not None
        layout = call_args.kwargs.get("layout")
        assert layout is not None
        assert layout.rows == 2
        assert layout.cols == 3

    @patch("neural_analysis.embeddings.visualization.PlotGrid")
    def test_plot_multiple_embeddings_grid_layout_7plus(
        self, mock_plot_grid: MagicMock
    ) -> None:
        """Test grid layout for 7+ embeddings (covers lines 149-150)."""
        mock_fig = MagicMock()
        mock_grid_instance = MagicMock()
        mock_grid_instance.plot.return_value = mock_fig
        mock_plot_grid.return_value = mock_grid_instance

        embeddings = {f"method_{i}": np.random.randn(50, 2) for i in range(7)}
        plot_multiple_embeddings(embeddings)

        call_args = mock_plot_grid.call_args
        assert call_args is not None
        layout = call_args.kwargs.get("layout")
        assert layout is not None
        assert layout.rows == 3


class TestPlotPCAVariance:
    """Tests for plot_pca_variance function."""

    @patch("neural_analysis.embeddings.visualization.PlotGrid")
    def test_plot_pca_variance_basic(self, mock_plot_grid: MagicMock) -> None:
        """Test plot_pca_variance with basic variance info (covers main path)."""
        mock_fig = MagicMock()
        mock_fig.axes = [MagicMock(), MagicMock()]
        mock_grid_instance = MagicMock()
        mock_grid_instance.plot.return_value = mock_fig
        mock_plot_grid.return_value = mock_grid_instance

        variance_info = {
            "explained_variance_ratio": np.array([0.4, 0.3, 0.2, 0.1]),
            "cumulative_variance_ratio": np.array([0.4, 0.7, 0.9, 1.0]),
        }

        result = plot_pca_variance(variance_info, cumulative=True)
        assert result == mock_fig
        mock_plot_grid.assert_called_once()

    @patch("neural_analysis.embeddings.visualization.PlotGrid")
    def test_plot_pca_variance_no_cumulative(self, mock_plot_grid: MagicMock) -> None:
        """Test plot_pca_variance without cumulative plot (covers line 410)."""
        mock_fig = MagicMock()
        mock_grid_instance = MagicMock()
        mock_grid_instance.plot.return_value = mock_fig
        mock_plot_grid.return_value = mock_grid_instance

        variance_info = {
            "explained_variance_ratio": np.array([0.4, 0.3, 0.2, 0.1]),
        }

        result = plot_pca_variance(variance_info, cumulative=False)
        assert result == mock_fig

    @patch("neural_analysis.embeddings.visualization.PlotGrid")
    def test_plot_pca_variance_n_components_to_show(
        self, mock_plot_grid: MagicMock
    ) -> None:
        """Test plot_pca_variance with n_components_to_show (covers lines 352-354)."""
        mock_fig = MagicMock()
        mock_fig.axes = [MagicMock(), MagicMock()]
        mock_grid_instance = MagicMock()
        mock_grid_instance.plot.return_value = mock_fig
        mock_plot_grid.return_value = mock_grid_instance

        variance_info = {
            "explained_variance_ratio": np.array([0.4, 0.3, 0.2, 0.1, 0.05, 0.05]),
            "cumulative_variance_ratio": np.array([0.4, 0.7, 0.9, 1.0, 1.05, 1.1]),
        }

        result = plot_pca_variance(variance_info, n_components_to_show=3)
        assert result == mock_fig

    @patch("neural_analysis.embeddings.visualization.PlotGrid")
    def test_plot_pca_variance_threshold_lines(self, mock_plot_grid: MagicMock) -> None:
        """Test plot_pca_variance with threshold lines (covers lines 431-443)."""
        mock_ax = MagicMock()
        mock_fig = MagicMock()
        mock_fig.axes = [MagicMock(), mock_ax]
        mock_grid_instance = MagicMock()
        mock_grid_instance.plot.return_value = mock_fig
        mock_plot_grid.return_value = mock_grid_instance

        variance_info = {
            "explained_variance_ratio": np.array([0.4, 0.3, 0.2, 0.1]),
            "cumulative_variance_ratio": np.array([0.4, 0.7, 0.9, 1.0]),
        }

        result = plot_pca_variance(
            variance_info,
            cumulative=True,
            threshold_lines=[0.85, 0.95],
            backend="matplotlib",
        )
        assert result == mock_fig
        # Check that axhline was called for each threshold
        assert mock_ax.axhline.call_count == 2
        assert mock_ax.legend.called

    @patch("neural_analysis.embeddings.visualization.PlotGrid")
    def test_plot_pca_variance_threshold_lines_plotly(
        self, mock_plot_grid: MagicMock
    ) -> None:
        """Test plot_pca_variance with threshold lines but plotly backend (covers line 431)."""
        mock_fig = MagicMock()
        mock_grid_instance = MagicMock()
        mock_grid_instance.plot.return_value = mock_fig
        mock_plot_grid.return_value = mock_grid_instance

        variance_info = {
            "explained_variance_ratio": np.array([0.4, 0.3, 0.2, 0.1]),
            "cumulative_variance_ratio": np.array([0.4, 0.7, 0.9, 1.0]),
        }

        result = plot_pca_variance(
            variance_info,
            cumulative=True,
            threshold_lines=[0.85, 0.95],
            backend="plotly",
        )
        assert result == mock_fig
        # Should not add threshold lines for plotly

    @patch("neural_analysis.embeddings.visualization.PlotGrid")
    def test_plot_pca_variance_default_colors(self, mock_plot_grid: MagicMock) -> None:
        """Test plot_pca_variance with default colors (covers lines 342-343)."""
        mock_fig = MagicMock()
        mock_fig.axes = [MagicMock(), MagicMock()]
        mock_grid_instance = MagicMock()
        mock_grid_instance.plot.return_value = mock_fig
        mock_plot_grid.return_value = mock_grid_instance

        variance_info = {
            "explained_variance_ratio": np.array([0.4, 0.3, 0.2, 0.1]),
            "cumulative_variance_ratio": np.array([0.4, 0.7, 0.9, 1.0]),
        }

        result = plot_pca_variance(variance_info, colors=None)
        assert result == mock_fig

    @patch("neural_analysis.embeddings.visualization.PlotGrid")
    def test_plot_pca_variance_custom_colors(self, mock_plot_grid: MagicMock) -> None:
        """Test plot_pca_variance with custom colors (covers branch 342->345 False path)."""
        mock_fig = MagicMock()
        mock_fig.axes = [MagicMock(), MagicMock()]
        mock_grid_instance = MagicMock()
        mock_grid_instance.plot.return_value = mock_fig
        mock_plot_grid.return_value = mock_grid_instance

        variance_info = {
            "explained_variance_ratio": np.array([0.4, 0.3, 0.2, 0.1]),
            "cumulative_variance_ratio": np.array([0.4, 0.7, 0.9, 1.0]),
        }

        result = plot_pca_variance(variance_info, colors=["red", "blue"])
        assert result == mock_fig

    @patch("neural_analysis.embeddings.visualization.PlotGrid")
    def test_plot_pca_variance_default_threshold_lines(
        self, mock_plot_grid: MagicMock
    ) -> None:
        """Test plot_pca_variance with default threshold lines (covers lines 345-346)."""
        mock_ax = MagicMock()
        mock_fig = MagicMock()
        mock_fig.axes = [MagicMock(), mock_ax]
        mock_grid_instance = MagicMock()
        mock_grid_instance.plot.return_value = mock_fig
        mock_plot_grid.return_value = mock_grid_instance

        variance_info = {
            "explained_variance_ratio": np.array([0.4, 0.3, 0.2, 0.1]),
            "cumulative_variance_ratio": np.array([0.4, 0.7, 0.9, 1.0]),
        }

        result = plot_pca_variance(
            variance_info, threshold_lines=None, backend="matplotlib"
        )
        assert result == mock_fig
        # Default thresholds should be [0.90, 0.95]
        assert mock_ax.axhline.call_count == 2

    @patch("neural_analysis.embeddings.visualization.PlotGrid")
    def test_plot_pca_variance_no_cumulative_in_dict(
        self, mock_plot_grid: MagicMock
    ) -> None:
        """Test plot_pca_variance when cumulative_variance_ratio not in dict (covers line 381)."""
        mock_fig = MagicMock()
        mock_grid_instance = MagicMock()
        mock_grid_instance.plot.return_value = mock_fig
        mock_plot_grid.return_value = mock_grid_instance

        variance_info = {
            "explained_variance_ratio": np.array([0.4, 0.3, 0.2, 0.1]),
        }

        result = plot_pca_variance(variance_info, cumulative=True)
        assert result == mock_fig
        # Should only create one plot (individual variance)
