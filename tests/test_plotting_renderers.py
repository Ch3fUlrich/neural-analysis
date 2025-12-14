"""Tests for plotting renderers module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from neural_analysis.plotting.renderers import (
    extract_xy_from_data,
    extract_xyz_from_data,
    render_scatter_matplotlib,
    render_line_matplotlib,
    render_histogram_matplotlib,
    render_heatmap_matplotlib,
    render_bar_matplotlib,
    render_violin_matplotlib,
    render_box_matplotlib,
    render_trajectory_matplotlib,
    render_kde_matplotlib,
    render_convex_hull_matplotlib,
    render_boolean_states_matplotlib,
    render_ellipse_matplotlib,
)


class TestExtractXYFromData:
    """Tests for extract_xy_from_data function."""

    def test_extract_xy_from_dict(self) -> None:
        """Test extracting x,y from dict (covers line 61-62)."""
        data = {"x": [1, 2, 3], "y": [4, 5, 6]}
        x, y = extract_xy_from_data(data)
        np.testing.assert_array_equal(x, [1, 2, 3])
        np.testing.assert_array_equal(y, [4, 5, 6])

    def test_extract_xy_from_dataframe(self) -> None:
        """Test extracting x,y from DataFrame (covers lines 63-66)."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        x, y = extract_xy_from_data(df)
        np.testing.assert_array_equal(x, [1, 2, 3])
        np.testing.assert_array_equal(y, [4, 5, 6])

    def test_extract_xy_from_dataframe_insufficient_columns(self) -> None:
        """Test DataFrame with insufficient columns (covers lines 67-68)."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(ValueError, match="at least 2 columns"):
            extract_xy_from_data(df)

    def test_extract_xy_from_array(self) -> None:
        """Test extracting x,y from 2D array (covers line 69-70)."""
        data = np.array([[1, 4], [2, 5], [3, 6]])
        x, y = extract_xy_from_data(data)
        np.testing.assert_array_equal(x, [1, 2, 3])
        np.testing.assert_array_equal(y, [4, 5, 6])

    def test_extract_xy_invalid_format(self) -> None:
        """Test invalid data format (covers line 72)."""
        with pytest.raises(ValueError, match="must be dict"):
            extract_xy_from_data([1, 2, 3])


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
        """Test extracting x,y,z from 2D array (covers line 104-105)."""
        data = np.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]])
        x, y, z = extract_xyz_from_data(data)
        np.testing.assert_array_equal(x, [1, 2, 3])
        np.testing.assert_array_equal(y, [4, 5, 6])
        np.testing.assert_array_equal(z, [7, 8, 9])

    def test_extract_xyz_invalid_format(self) -> None:
        """Test invalid data format (covers line 107)."""
        with pytest.raises(ValueError, match="must be dict"):
            extract_xyz_from_data([1, 2, 3])


class TestRenderScatterMatplotlib:
    """Tests for render_scatter_matplotlib function."""

    def test_render_scatter_basic(self) -> None:
        """Test basic scatter plot rendering (covers main path)."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        data = np.array([[1, 2], [3, 4], [5, 6]])

        result = render_scatter_matplotlib(ax, data, color="blue")
        assert result is not None
        plt.close(fig)

    def test_render_scatter_with_colors(self) -> None:
        """Test scatter plot with colors array (covers colors path)."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        data = np.array([[1, 2], [3, 4], [5, 6]])
        colors = np.array([0.1, 0.5, 0.9])

        result = render_scatter_matplotlib(ax, data, colors=colors, cmap="viridis")
        assert result is not None
        plt.close(fig)

    def test_render_scatter_with_marker_size(self) -> None:
        """Test scatter plot with marker size (covers marker_size path)."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        data = np.array([[1, 2], [3, 4], [5, 6]])

        result = render_scatter_matplotlib(ax, data, marker_size=50)
        assert result is not None
        plt.close(fig)

    def test_render_scatter_with_kwargs(self) -> None:
        """Test scatter plot with kwargs (covers kwargs.pop paths)."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        data = np.array([[1, 2], [3, 4], [5, 6]])

        result = render_scatter_matplotlib(
            ax, data, x_label="X", y_label="Y", s=30, alpha=0.8
        )
        assert result is not None
        plt.close(fig)

    def test_render_scatter_3d(self) -> None:
        """Test scatter plot on 3D axes (covers 3D path)."""
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        result = render_scatter_matplotlib(ax, data, color="red")
        assert result is not None
        plt.close(fig)


class TestRenderLineMatplotlib:
    """Tests for render_line_matplotlib function."""

    def test_render_line_basic(self) -> None:
        """Test basic line plot rendering."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        data = np.array([[1, 2], [3, 4], [5, 6]])

        result = render_line_matplotlib(ax, data, color="blue")
        assert result is not None
        plt.close(fig)

    def test_render_line_with_marker(self) -> None:
        """Test line plot with marker."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        data = np.array([[1, 2], [3, 4], [5, 6]])

        result = render_line_matplotlib(ax, data, marker="o", markersize=5)
        assert result is not None
        plt.close(fig)


class TestRenderHistogramMatplotlib:
    """Tests for render_histogram_matplotlib function."""

    def test_render_histogram_basic(self) -> None:
        """Test basic histogram rendering."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        result = render_histogram_matplotlib(ax, data, color="blue")
        assert result is not None
        plt.close(fig)


class TestRenderHeatmapMatplotlib:
    """Tests for render_heatmap_matplotlib function."""

    def test_render_heatmap_basic(self) -> None:
        """Test basic heatmap rendering."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        data = np.random.rand(5, 5)

        result = render_heatmap_matplotlib(ax, data)
        assert result is not None
        plt.close(fig)


class TestRenderBarMatplotlib:
    """Tests for render_bar_matplotlib function."""

    def test_render_bar_basic(self) -> None:
        """Test basic bar plot rendering."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        data = np.array([1, 2, 3, 4, 5])

        result = render_bar_matplotlib(ax, data, color="blue")
        assert result is not None
        plt.close(fig)


class TestRenderViolinMatplotlib:
    """Tests for render_violin_matplotlib function."""

    def test_render_violin_basic(self) -> None:
        """Test basic violin plot rendering."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        result = render_violin_matplotlib(ax, data, color="blue")
        assert result is not None
        plt.close(fig)


class TestRenderBoxMatplotlib:
    """Tests for render_box_matplotlib function."""

    def test_render_box_basic(self) -> None:
        """Test basic box plot rendering."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        result = render_box_matplotlib(ax, data, color="blue")
        assert result is not None
        plt.close(fig)


class TestRenderTrajectoryMatplotlib:
    """Tests for render_trajectory_matplotlib function."""

    def test_render_trajectory_basic(self) -> None:
        """Test basic trajectory rendering."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        x = np.array([1, 3, 5, 7])
        y = np.array([2, 4, 6, 8])

        result = render_trajectory_matplotlib(ax, x, y, color="blue")
        assert result is not None
        plt.close(fig)


class TestRenderKDEMatplotlib:
    """Tests for render_kde_matplotlib function."""

    def test_render_kde_basic(self) -> None:
        """Test basic KDE rendering."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        # Create grid for KDE
        xi = np.linspace(0, 10, 20)
        yi = np.linspace(0, 10, 20)
        xi, yi = np.meshgrid(xi, yi)
        zi = np.random.rand(20, 20)  # Density values

        result = render_kde_matplotlib(ax, xi, yi, zi, color="blue")
        assert result is not None
        plt.close(fig)


class TestRenderConvexHullMatplotlib:
    """Tests for render_convex_hull_matplotlib function."""

    def test_render_convex_hull_basic(self) -> None:
        """Test basic convex hull rendering."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        # Convex hull boundary coordinates (closed loop)
        hull_x = np.array([1, 3, 5, 7, 9, 1])  # Closed loop
        hull_y = np.array([2, 4, 6, 8, 10, 2])  # Closed loop

        result = render_convex_hull_matplotlib(ax, hull_x, hull_y, color="blue")
        assert result is not None
        plt.close(fig)


class TestRenderBooleanStatesMatplotlib:
    """Tests for render_boolean_states_matplotlib function."""

    def test_render_boolean_states_basic(self) -> None:
        """Test basic boolean states rendering."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        x = np.array([0, 1, 2, 3, 4])  # Time points
        states = np.array([True, False, True, True, False])

        result = render_boolean_states_matplotlib(ax, x, states)
        assert result is not None
        plt.close(fig)


class TestRenderEllipseMatplotlib:
    """Tests for render_ellipse_matplotlib function."""

    def test_render_ellipse_basic(self) -> None:
        """Test basic ellipse rendering."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        # Ellipse parameters
        centers = np.array([[0, 0]])  # Center coordinates
        widths = np.array([2.0])  # Width
        heights = np.array([1.0])  # Height
        angles = np.array([0.0])  # Rotation angle

        result = render_ellipse_matplotlib(ax, centers, widths, heights, angles, color="blue")
        assert result is not None
        plt.close(fig)

