"""Additional tests for plotting renderers module to improve coverage."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from neural_analysis.plotting.renderers import (
    extract_xyz_from_data,
    render_heatmap_walls_matplotlib,
    render_line_plotly,
    render_scatter3d_plotly,
    render_violin_plotly,
)


class TestExtractXYZFromData:
    """Tests for extract_xyz_from_data function."""

    def test_extract_xyz_from_dict(self) -> None:
        """Test extracting x,y,z from dict (covers line 96-97)."""
        data = {"x": [1, 2, 3], "y": [4, 5, 6], "z": [7, 8, 9]}
        x, y, z = extract_xyz_from_data(data)
        np.testing.assert_array_equal(x, [1, 2, 3])
        np.testing.assert_array_equal(y, [4, 5, 6])
        np.testing.assert_array_equal(z, [7, 8, 9])

    def test_extract_xyz_from_dataframe(self) -> None:
        """Test extracting x,y,z from DataFrame (covers lines 98-101)."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
        x, y, z = extract_xyz_from_data(df)
        np.testing.assert_array_equal(x, [1, 2, 3])
        np.testing.assert_array_equal(y, [4, 5, 6])
        np.testing.assert_array_equal(z, [7, 8, 9])

    def test_extract_xyz_from_dataframe_insufficient_columns(self) -> None:
        """Test DataFrame with insufficient columns (covers lines 102-103)."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        with pytest.raises(ValueError, match="at least 3 columns"):
            extract_xyz_from_data(df)

    def test_extract_xyz_from_array(self) -> None:
        """Test extracting x,y,z from array (covers lines 104-105)."""
        data = np.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]])
        x, y, z = extract_xyz_from_data(data)
        np.testing.assert_array_equal(x, [1, 2, 3])
        np.testing.assert_array_equal(y, [4, 5, 6])
        np.testing.assert_array_equal(z, [7, 8, 9])

    def test_extract_xyz_invalid_format(self) -> None:
        """Test invalid data format (covers line 107)."""
        with pytest.raises(ValueError, match="must be dict with 'x','y','z' keys"):
            extract_xyz_from_data([1, 2, 3])


class TestRenderScatter3DPlotly:
    """Tests for render_scatter3d_plotly function."""

    def test_render_scatter3d_plotly_unavailable(self) -> None:
        """Test render_scatter3d_plotly when plotly unavailable (covers lines 370-371)."""
        with patch("neural_analysis.plotting.renderers.PLOTLY_AVAILABLE", False):
            with pytest.raises(ImportError, match="Plotly is required"):
                render_scatter3d_plotly(
                    data=np.array([[1, 2, 3], [4, 5, 6]]),
                )

    def test_render_scatter3d_plotly_invalid_shape(self) -> None:
        """Test render_scatter3d_plotly with invalid shape (covers lines 373-374)."""
        with patch("neural_analysis.plotting.renderers.PLOTLY_AVAILABLE", True):
            with pytest.raises(ValueError, match="3-column data"):
                render_scatter3d_plotly(
                    data=np.array([[1, 2], [3, 4]]),
                )

    def test_render_scatter3d_plotly_with_colors(self) -> None:
        """Test render_scatter3d_plotly with colors array (covers lines 383-389)."""
        with patch("neural_analysis.plotting.renderers.PLOTLY_AVAILABLE", True):
            with patch("neural_analysis.plotting.renderers.go") as mock_go:
                mock_scatter = MagicMock()
                mock_go.Scatter3d.return_value = mock_scatter
                
                data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
                colors = np.array([0.1, 0.5, 0.9])
                
                result = render_scatter3d_plotly(
                    data=data,
                    colors=colors,
                    cmap="viridis",
                    colorbar=True,
                    colorbar_label="Value",
                )
                assert result == mock_scatter
                mock_go.Scatter3d.assert_called_once()

    def test_render_scatter3d_plotly_with_color(self) -> None:
        """Test render_scatter3d_plotly with single color (covers lines 390-391)."""
        with patch("neural_analysis.plotting.renderers.PLOTLY_AVAILABLE", True):
            with patch("neural_analysis.plotting.renderers.go") as mock_go:
                mock_scatter = MagicMock()
                mock_go.Scatter3d.return_value = mock_scatter
                
                data = np.array([[1, 2, 3], [4, 5, 6]])
                
                result = render_scatter3d_plotly(
                    data=data,
                    color="red",
                )
                assert result == mock_scatter


class TestRenderLinePlotly:
    """Tests for render_line_plotly function."""

    def test_render_line_plotly_unavailable(self) -> None:
        """Test render_line_plotly when plotly unavailable (covers lines 603-604)."""
        with patch("neural_analysis.plotting.renderers.PLOTLY_AVAILABLE", False):
            with pytest.raises(ImportError, match="Plotly is required"):
                render_line_plotly(data=np.array([1, 2, 3]))

    def test_render_line_plotly_dict_data(self) -> None:
        """Test render_line_plotly with dict data (covers lines 634-648)."""
        with patch("neural_analysis.plotting.renderers.PLOTLY_AVAILABLE", True):
            with patch("neural_analysis.plotting.renderers.go") as mock_go:
                mock_scatter = MagicMock()
                mock_go.Scatter.return_value = mock_scatter
                
                data = {"x": [1, 2, 3], "y": [4, 5, 6]}
                
                result = render_line_plotly(data=data)
                assert result == mock_scatter
                mock_go.Scatter.assert_called_once()

    def test_render_line_plotly_dict_data_missing_keys(self) -> None:
        """Test render_line_plotly with dict missing keys (covers lines 649-650)."""
        with patch("neural_analysis.plotting.renderers.PLOTLY_AVAILABLE", True):
            with pytest.raises(ValueError, match="must contain 'x' and 'y' keys"):
                render_line_plotly(data={"x": [1, 2, 3]})

    def test_render_line_plotly_1d(self) -> None:
        """Test render_line_plotly with 1D data (covers lines 652-663)."""
        with patch("neural_analysis.plotting.renderers.PLOTLY_AVAILABLE", True):
            with patch("neural_analysis.plotting.renderers.go") as mock_go:
                mock_scatter = MagicMock()
                mock_go.Scatter.return_value = mock_scatter
                
                data = np.array([1, 2, 3, 4, 5])
                
                result = render_line_plotly(data=data)
                assert result == mock_scatter

    def test_render_line_plotly_2d(self) -> None:
        """Test render_line_plotly with 2D data (covers lines 664-676)."""
        with patch("neural_analysis.plotting.renderers.PLOTLY_AVAILABLE", True):
            with patch("neural_analysis.plotting.renderers.go") as mock_go:
                mock_scatter = MagicMock()
                mock_go.Scatter.return_value = mock_scatter
                
                data = np.array([[1, 4], [2, 5], [3, 6]])
                
                result = render_line_plotly(data=data)
                assert result == mock_scatter

    def test_render_line_plotly_multiple_y(self) -> None:
        """Test render_line_plotly with multiple y values (covers lines 677-688)."""
        with patch("neural_analysis.plotting.renderers.PLOTLY_AVAILABLE", True):
            with patch("neural_analysis.plotting.renderers.go") as mock_go:
                mock_scatter = MagicMock()
                mock_go.Scatter.return_value = mock_scatter
                
                data = np.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]])
                
                result = render_line_plotly(data=data)
                assert result == mock_scatter

    def test_render_line_plotly_with_error_y(self) -> None:
        """Test render_line_plotly with error_y (covers lines 624-631)."""
        with patch("neural_analysis.plotting.renderers.PLOTLY_AVAILABLE", True):
            with patch("neural_analysis.plotting.renderers.go") as mock_go:
                mock_scatter = MagicMock()
                mock_go.Scatter.return_value = mock_scatter
                
                data = np.array([1, 2, 3])
                error_y = np.array([0.1, 0.2, 0.1])
                
                result = render_line_plotly(data=data, error_y=error_y)
                assert result == mock_scatter


class TestRenderViolinPlotly:
    """Tests for render_violin_plotly function."""

    def test_render_violin_plotly_unavailable(self) -> None:
        """Test render_violin_plotly when plotly unavailable (covers lines 1498-1499)."""
        with patch("neural_analysis.plotting.renderers.PLOTLY_AVAILABLE", False):
            with pytest.raises(ImportError, match="Plotly is required"):
                render_violin_plotly(data=np.array([1, 2, 3]))

    def test_render_violin_plotly_basic(self) -> None:
        """Test render_violin_plotly basic (covers main path)."""
        with patch("neural_analysis.plotting.renderers.PLOTLY_AVAILABLE", True):
            with patch("neural_analysis.plotting.renderers.go") as mock_go:
                mock_violin = MagicMock()
                mock_go.Violin.return_value = mock_violin
                
                data = np.array([1, 2, 3, 4, 5])
                
                result = render_violin_plotly(data=data)
                assert result == mock_violin
                mock_go.Violin.assert_called_once()


class TestRenderHeatmapWallsMatplotlib:
    """Tests for render_heatmap_walls_matplotlib function."""

    def test_render_heatmap_walls_matplotlib_basic(self) -> None:
        """Test render_heatmap_walls_matplotlib basic (covers lines 919-1031)."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        
        data = {
            "xy": np.random.rand(10, 10),
            "xz": np.random.rand(10, 10),
            "yz": np.random.rand(10, 10),
            "x_centers": np.linspace(0, 10, 10),
            "y_centers": np.linspace(0, 10, 10),
            "z_centers": np.linspace(0, 10, 10),
        }
        
        artists = render_heatmap_walls_matplotlib(
            ax=ax,
            data=data,
            cmap="viridis",
            alpha=0.7,
            colorbar=True,
            colorbar_label="Value",
        )
        assert len(artists) >= 3  # At least 3 surfaces
        plt.close(fig)

    def test_render_heatmap_walls_matplotlib_partial_data(self) -> None:
        """Test render_heatmap_walls_matplotlib with partial data (covers missing walls)."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        
        # Only provide xy wall
        data = {
            "xy": np.random.rand(10, 10),
            "x_centers": np.linspace(0, 10, 10),
            "y_centers": np.linspace(0, 10, 10),
            "z_centers": np.linspace(0, 10, 10),
        }
        
        artists = render_heatmap_walls_matplotlib(
            ax=ax,
            data=data,
            cmap="viridis",
        )
        assert len(artists) >= 1  # At least 1 surface
        plt.close(fig)

    def test_render_heatmap_walls_matplotlib_custom_positions(self) -> None:
        """Test render_heatmap_walls_matplotlib with custom positions (covers lines 945-947)."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        
        data = {
            "xy": np.random.rand(10, 10),
            "xz": np.random.rand(10, 10),
            "yz": np.random.rand(10, 10),
            "x_centers": np.linspace(0, 10, 10),
            "y_centers": np.linspace(0, 10, 10),
            "z_centers": np.linspace(0, 10, 10),
            "xy_position": 5.0,
            "xz_position": 3.0,
            "yz_position": 7.0,
        }
        
        artists = render_heatmap_walls_matplotlib(
            ax=ax,
            data=data,
            cmap="viridis",
        )
        assert len(artists) >= 3
        plt.close(fig)

    def test_render_heatmap_walls_matplotlib_exception_handling(self) -> None:
        """Test render_heatmap_walls_matplotlib exception handling (covers lines 1018-1019, 1028-1029)."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        
        # Data that will cause exception in set_box_aspect
        data = {
            "xy": np.random.rand(10, 10),
            "x_centers": np.array([0]),  # Single value to trigger exception
            "y_centers": np.array([0]),
            "z_centers": np.array([0]),
        }
        
        # Should handle exception gracefully
        artists = render_heatmap_walls_matplotlib(
            ax=ax,
            data=data,
            cmap="viridis",
            colorbar=True,
        )
        assert artists is not None
        plt.close(fig)


