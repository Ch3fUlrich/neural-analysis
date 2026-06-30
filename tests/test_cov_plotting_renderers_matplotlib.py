"""Tests for neural_analysis.plotting.renderers_matplotlib covering all branches."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ax2d() -> Axes:
    fig, ax = plt.subplots()
    return ax


def _ax3d():
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    return ax


# ---------------------------------------------------------------------------
# render_scatter_matplotlib
# ---------------------------------------------------------------------------

from neural_analysis.plotting.renderers_matplotlib import render_scatter_matplotlib


class TestRenderScatterMatplotlib:
    def teardown_method(self):
        plt.close("all")

    def test_scatter_2d_basic(self):
        ax = _ax2d()
        rng = np.random.default_rng(0)
        data = rng.uniform(0, 1, size=(20, 2))
        sc = render_scatter_matplotlib(ax, data)
        assert sc is not None
        offsets = sc.get_offsets()
        assert offsets.shape == (20, 2)

    def test_scatter_2d_with_color(self):
        ax = _ax2d()
        rng = np.random.default_rng(0)
        data = rng.uniform(size=(10, 2))
        sc = render_scatter_matplotlib(ax, data, color="red", alpha=0.5)
        assert sc is not None

    def test_scatter_2d_with_colors_array(self):
        ax = _ax2d()
        rng = np.random.default_rng(0)
        data = rng.uniform(size=(10, 2))
        colors = rng.uniform(size=10)
        sc = render_scatter_matplotlib(ax, data, colors=colors, cmap="viridis")
        assert sc is not None

    def test_scatter_2d_with_marker_size(self):
        ax = _ax2d()
        data = np.ones((5, 2))
        sc = render_scatter_matplotlib(ax, data, marker_size=50)
        assert sc is not None

    def test_scatter_2d_s_kwarg(self):
        """s kwarg overrides marker_size."""
        ax = _ax2d()
        data = np.ones((5, 2))
        sc = render_scatter_matplotlib(ax, data, s=30)
        assert sc is not None

    def test_scatter_2d_with_label(self):
        ax = _ax2d()
        data = np.ones((5, 2))
        sc = render_scatter_matplotlib(ax, data, label="test")
        assert sc is not None
        assert sc.get_label() == "test"

    def test_scatter_2d_kwargs_popped(self):
        """axis label kwargs are popped silently."""
        ax = _ax2d()
        data = np.ones((5, 2))
        sc = render_scatter_matplotlib(ax, data, x_label="X", y_label="Y", z_label="Z")
        assert sc is not None

    def test_scatter_3d_basic(self):
        ax = _ax3d()
        rng = np.random.default_rng(0)
        data = rng.uniform(size=(10, 3))
        sc = render_scatter_matplotlib(ax, data)
        assert sc is not None

    def test_scatter_invalid_shape_raises(self):
        ax = _ax2d()
        data = np.ones((5, 4))
        with pytest.raises(ValueError, match="2D or 3D"):
            render_scatter_matplotlib(ax, data)

    def test_scatter_2d_non_string_c_param(self):
        """Non-string, non-None c_param (single color) triggers cmap path."""
        ax = _ax2d()
        rng = np.random.default_rng(0)
        data = rng.uniform(size=(5, 2))
        # Pass a numeric list via color — this is unusual but exercises the branch
        sc = render_scatter_matplotlib(ax, data, colors=rng.uniform(size=5))
        assert sc is not None


# ---------------------------------------------------------------------------
# render_line_matplotlib
# ---------------------------------------------------------------------------

from neural_analysis.plotting.renderers_matplotlib import render_line_matplotlib


class TestRenderLineMatplotlib:
    def teardown_method(self):
        plt.close("all")

    def test_line_1d_basic(self):
        ax = _ax2d()
        data = np.linspace(0, 1, 20)
        lines = render_line_matplotlib(ax, data)
        assert len(lines) == 1

    def test_line_2d_xy_columns(self):
        ax = _ax2d()
        data = np.column_stack([np.linspace(0, 1, 10), np.sin(np.linspace(0, 1, 10))])
        lines = render_line_matplotlib(ax, data)
        assert len(lines) == 1

    def test_line_dict_xy(self):
        ax = _ax2d()
        data = {"x": [0, 1, 2], "y": [0, 1, 0]}
        lines = render_line_matplotlib(ax, data)
        assert len(lines) == 1

    def test_line_dict_missing_keys_raises(self):
        ax = _ax2d()
        with pytest.raises(ValueError, match="'x' and 'y' keys"):
            render_line_matplotlib(ax, {"z": [1, 2]})

    def test_line_multi_columns(self):
        """2D data with > 2 columns → multiple lines path."""
        ax = _ax2d()
        rng = np.random.default_rng(0)
        data = rng.uniform(size=(10, 3))
        lines = render_line_matplotlib(ax, data)
        assert len(lines) == 3

    def test_line_multi_columns_with_marker(self):
        ax = _ax2d()
        rng = np.random.default_rng(0)
        data = rng.uniform(size=(5, 2))  # 2 columns but ndim==2 shape[1]==2 → single line
        # Force multi-column path: use 3 columns
        data3 = rng.uniform(size=(5, 3))
        lines = render_line_matplotlib(ax, data3, marker="o", marker_size=5)
        assert len(lines) == 3

    def test_line_with_error_band_no_color(self):
        ax = _ax2d()
        data = np.ones(10)
        error_y = np.ones(10) * 0.1
        lines = render_line_matplotlib(ax, data, error_y=error_y)
        assert len(lines) == 1

    def test_line_with_error_band_with_color(self):
        ax = _ax2d()
        data = np.ones(10)
        error_y = np.ones(10) * 0.1
        lines = render_line_matplotlib(ax, data, color="blue", error_y=error_y)
        assert len(lines) == 1

    def test_line_show_values(self):
        ax = _ax2d()
        data = np.array([1.0, 2.0, 3.0])
        lines = render_line_matplotlib(ax, data, show_values=True, value_format=".1f")
        assert len(lines) == 1
        # Three text annotations added
        texts = [c for c in ax.get_children() if hasattr(c, "get_text")]
        assert len(texts) >= 3

    def test_line_x_labels(self):
        ax = _ax2d()
        data = np.array([1.0, 2.0, 3.0])
        lines = render_line_matplotlib(ax, data, x_labels=["a", "b", "c"])
        assert len(lines) == 1
        ticklabels = [t.get_text() for t in ax.get_xticklabels()]
        assert "a" in ticklabels

    def test_line_with_marker(self):
        ax = _ax2d()
        data = np.linspace(0, 1, 5)
        lines = render_line_matplotlib(ax, data, marker="o", marker_size=8)
        assert len(lines) == 1

    def test_line_with_marker_no_markersize(self):
        ax = _ax2d()
        data = np.linspace(0, 1, 5)
        lines = render_line_matplotlib(ax, data, marker="o")
        assert len(lines) == 1

    def test_line_pops_custom_kwargs(self):
        ax = _ax2d()
        data = np.linspace(0, 1, 5)
        lines = render_line_matplotlib(
            ax, data, false_color="red", true_label="yes", false_label="no",
            x_label="x", y_label="y"
        )
        assert len(lines) == 1

    def test_line_show_values_empty_y(self):
        """show_values with a constant array (max-min == 0) should not crash."""
        ax = _ax2d()
        data = np.array([5.0, 5.0, 5.0])
        lines = render_line_matplotlib(ax, data, show_values=True)
        assert len(lines) == 1


# ---------------------------------------------------------------------------
# render_histogram_matplotlib
# ---------------------------------------------------------------------------

from neural_analysis.plotting.renderers_matplotlib import render_histogram_matplotlib


class TestRenderHistogramMatplotlib:
    def teardown_method(self):
        plt.close("all")

    def test_histogram_basic(self):
        ax = _ax2d()
        rng = np.random.default_rng(0)
        data = rng.normal(0, 1, 100)
        result = render_histogram_matplotlib(ax, data)
        n, bins, patches = result
        assert len(n) == 30  # default bins
        assert len(bins) == 31

    def test_histogram_with_color(self):
        ax = _ax2d()
        data = np.arange(10, dtype=float)
        result = render_histogram_matplotlib(ax, data, color="green", bins=5, alpha=0.5)
        n, bins, patches = result
        assert len(n) == 5

    def test_histogram_with_label(self):
        ax = _ax2d()
        data = np.arange(10, dtype=float)
        result = render_histogram_matplotlib(ax, data, label="hist")
        n, bins, patches = result
        assert patches[0].get_label() == "hist"


# ---------------------------------------------------------------------------
# render_heatmap_matplotlib
# ---------------------------------------------------------------------------

from neural_analysis.plotting.renderers_matplotlib import render_heatmap_matplotlib


class TestRenderHeatmapMatplotlib:
    def teardown_method(self):
        plt.close("all")

    def test_heatmap_basic(self):
        ax = _ax2d()
        data = np.arange(9).reshape(3, 3).astype(float)
        im = render_heatmap_matplotlib(ax, data)
        assert im is not None
        assert im.get_array().shape == (3, 3)

    def test_heatmap_no_colorbar(self):
        ax = _ax2d()
        data = np.ones((4, 4))
        im = render_heatmap_matplotlib(ax, data, colorbar=False)
        assert im is not None

    def test_heatmap_with_colorbar_label(self):
        ax = _ax2d()
        data = np.ones((3, 3))
        im = render_heatmap_matplotlib(ax, data, colorbar=True, colorbar_label="density")
        assert im is not None

    def test_heatmap_x_labels(self):
        ax = _ax2d()
        data = np.ones((2, 3))
        im = render_heatmap_matplotlib(ax, data, colorbar=False, x_labels=["a", "b", "c"])
        assert im is not None
        ticklabels = [t.get_text() for t in ax.get_xticklabels()]
        assert "a" in ticklabels

    def test_heatmap_y_labels(self):
        ax = _ax2d()
        data = np.ones((2, 2))
        im = render_heatmap_matplotlib(ax, data, colorbar=False, y_labels=["r0", "r1"])
        assert im is not None

    def test_heatmap_custom_ticks(self):
        ax = _ax2d()
        data = np.eye(3)
        im = render_heatmap_matplotlib(
            ax, data, colorbar=False,
            set_xticks=[0, 1, 2], set_xticklabels=["A", "B", "C"],
            set_yticks=[0, 1, 2], set_yticklabels=["X", "Y", "Z"],
        )
        assert im is not None
        xlabels = [t.get_text() for t in ax.get_xticklabels()]
        assert "A" in xlabels

    def test_heatmap_show_values(self):
        ax = _ax2d()
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        im = render_heatmap_matplotlib(ax, data, colorbar=False, show_values=True, value_format=".1f")
        assert im is not None
        texts = [c for c in ax.get_children() if hasattr(c, "get_text")]
        # 4 annotations
        assert len(texts) >= 4

    def test_heatmap_set_xlim_ylim(self):
        ax = _ax2d()
        data = np.ones((3, 3))
        im = render_heatmap_matplotlib(
            ax, data, colorbar=False, set_xlim=(0, 2), set_ylim=(0, 2)
        )
        assert im is not None

    def test_heatmap_pops_axis_labels(self):
        ax = _ax2d()
        data = np.ones((3, 3))
        im = render_heatmap_matplotlib(ax, data, colorbar=False, x_label="X", y_label="Y")
        assert im is not None


# ---------------------------------------------------------------------------
# render_heatmap_walls_matplotlib
# ---------------------------------------------------------------------------

from neural_analysis.plotting.renderers_matplotlib import render_heatmap_walls_matplotlib


class TestRenderHeatmapWallsMatplotlib:
    def teardown_method(self):
        plt.close("all")

    def _make_data(self, n=4):
        rng = np.random.default_rng(0)
        c = np.linspace(0, 1, n)
        return {
            "xy": rng.uniform(size=(n, n)),
            "xz": rng.uniform(size=(n, n)),
            "yz": rng.uniform(size=(n, n)),
            "x_centers": c,
            "y_centers": c,
            "z_centers": c,
        }

    def test_walls_basic(self):
        ax = _ax3d()
        data = self._make_data()
        artists = render_heatmap_walls_matplotlib(ax, data)
        assert isinstance(artists, list)
        assert len(artists) >= 3  # 3 surfaces + colorbar

    def test_walls_no_colorbar(self):
        ax = _ax3d()
        data = self._make_data()
        artists = render_heatmap_walls_matplotlib(ax, data, colorbar=False)
        assert isinstance(artists, list)

    def test_walls_with_colorbar_label(self):
        ax = _ax3d()
        data = self._make_data()
        artists = render_heatmap_walls_matplotlib(ax, data, colorbar_label="rate")
        assert isinstance(artists, list)

    def test_walls_missing_projections(self):
        """Missing xy/xz/yz keys are handled gracefully."""
        ax = _ax3d()
        n = 4
        c = np.linspace(0, 1, n)
        data = {
            "xy": None,
            "xz": None,
            "yz": None,
            "x_centers": c,
            "y_centers": c,
            "z_centers": c,
        }
        artists = render_heatmap_walls_matplotlib(ax, data, colorbar=False)
        assert isinstance(artists, list)

    def test_walls_custom_positions(self):
        ax = _ax3d()
        data = self._make_data()
        data["xy_position"] = 0.5
        data["xz_position"] = 0.5
        data["yz_position"] = 0.5
        artists = render_heatmap_walls_matplotlib(ax, data, colorbar=False)
        assert isinstance(artists, list)

    def test_walls_single_center_vals(self):
        """Single-element centers → range = 0 → handled via else path."""
        ax = _ax3d()
        rng = np.random.default_rng(0)
        c = np.array([0.5])
        data = {
            "xy": rng.uniform(size=(1, 1)),
            "xz": rng.uniform(size=(1, 1)),
            "yz": rng.uniform(size=(1, 1)),
            "x_centers": c,
            "y_centers": c,
            "z_centers": c,
        }
        artists = render_heatmap_walls_matplotlib(ax, data, colorbar=False)
        assert isinstance(artists, list)


# ---------------------------------------------------------------------------
# render_bar_matplotlib
# ---------------------------------------------------------------------------

from neural_analysis.plotting.renderers_matplotlib import render_bar_matplotlib


class TestRenderBarMatplotlib:
    def teardown_method(self):
        plt.close("all")

    def test_bar_vertical_basic(self):
        ax = _ax2d()
        data = np.array([1.0, 2.0, 3.0])
        bars = render_bar_matplotlib(ax, data)
        assert bars is not None
        assert len(bars) == 3

    def test_bar_vertical_with_x(self):
        ax = _ax2d()
        data = np.array([1.0, 2.0, 3.0])
        bars = render_bar_matplotlib(ax, data, x=np.array([0, 1, 2]))
        assert len(bars) == 3

    def test_bar_vertical_with_colors(self):
        ax = _ax2d()
        data = np.array([1.0, 2.0])
        bars = render_bar_matplotlib(ax, data, colors=["red", "blue"])
        assert len(bars) == 2

    def test_bar_vertical_with_error_y(self):
        ax = _ax2d()
        data = np.array([1.0, 2.0, 3.0])
        error = np.array([0.1, 0.2, 0.1])
        bars = render_bar_matplotlib(ax, data, error_y=error)
        assert len(bars) == 3

    def test_bar_vertical_show_values(self):
        ax = _ax2d()
        data = np.array([1.0, 2.0, 3.0])
        bars = render_bar_matplotlib(ax, data, show_values=True)
        texts = [c for c in ax.get_children() if hasattr(c, "get_text")]
        assert len(texts) >= 3

    def test_bar_vertical_show_values_with_error_y(self):
        ax = _ax2d()
        data = np.array([1.0, 2.0])
        error = np.array([0.1, 0.2])
        bars = render_bar_matplotlib(ax, data, show_values=True, error_y=error)
        assert len(bars) == 2

    def test_bar_vertical_x_labels(self):
        ax = _ax2d()
        data = np.array([1.0, 2.0, 3.0])
        bars = render_bar_matplotlib(ax, data, x_labels=["a", "b", "c"])
        ticklabels = [t.get_text() for t in ax.get_xticklabels()]
        assert "a" in ticklabels

    def test_bar_horizontal_basic(self):
        ax = _ax2d()
        data = np.array([1.0, 2.0, 3.0])
        bars = render_bar_matplotlib(ax, data, orientation="h")
        assert bars is not None

    def test_bar_horizontal_with_error_x(self):
        ax = _ax2d()
        data = np.array([1.0, 2.0])
        error = np.array([0.1, 0.2])
        bars = render_bar_matplotlib(ax, data, orientation="h", error_x=error)
        assert bars is not None

    def test_bar_horizontal_show_values(self):
        ax = _ax2d()
        data = np.array([1.0, 2.0, 3.0])
        bars = render_bar_matplotlib(ax, data, orientation="h", show_values=True)
        texts = [c for c in ax.get_children() if hasattr(c, "get_text")]
        assert len(texts) >= 3

    def test_bar_horizontal_show_values_with_error_x(self):
        ax = _ax2d()
        data = np.array([1.0, 2.0])
        error = np.array([0.1, 0.2])
        bars = render_bar_matplotlib(ax, data, orientation="h", show_values=True, error_x=error)
        assert bars is not None

    def test_bar_horizontal_x_labels(self):
        ax = _ax2d()
        data = np.array([1.0, 2.0, 3.0])
        bars = render_bar_matplotlib(ax, data, orientation="h", x_labels=["a", "b", "c"])
        ticklabels = [t.get_text() for t in ax.get_yticklabels()]
        assert "a" in ticklabels

    def test_bar_pops_custom_kwargs(self):
        ax = _ax2d()
        data = np.array([1.0, 2.0])
        bars = render_bar_matplotlib(ax, data, x_label="X", y_label="Y")
        assert bars is not None

    def test_bar_horizontal_show_values_no_error_x(self):
        """show_values with no error_x goes through the None branch."""
        ax = _ax2d()
        data = np.array([3.0, 5.0])
        bars = render_bar_matplotlib(ax, data, orientation="h", show_values=True, error_x=None)
        assert bars is not None


# ---------------------------------------------------------------------------
# render_violin_matplotlib
# ---------------------------------------------------------------------------

from neural_analysis.plotting.renderers_matplotlib import render_violin_matplotlib


class TestRenderViolinMatplotlib:
    def teardown_method(self):
        plt.close("all")

    def test_violin_basic(self):
        ax = _ax2d()
        rng = np.random.default_rng(0)
        data = rng.normal(0, 1, 50)
        result = render_violin_matplotlib(ax, data, position=1)
        assert "violin" in result

    def test_violin_with_color(self):
        ax = _ax2d()
        rng = np.random.default_rng(0)
        data = rng.normal(0, 1, 50)
        result = render_violin_matplotlib(ax, data, position=1, color="blue")
        assert "violin" in result

    def test_violin_with_label(self):
        ax = _ax2d()
        rng = np.random.default_rng(0)
        data = rng.normal(0, 1, 50)
        result = render_violin_matplotlib(ax, data, position=1, label="grp")
        assert "legend_handle" in result

    def test_violin_showpoints_false(self):
        ax = _ax2d()
        rng = np.random.default_rng(0)
        data = rng.normal(0, 1, 30)
        result = render_violin_matplotlib(ax, data, showpoints=False)
        assert "points" not in result

    def test_violin_showpoints_true(self):
        ax = _ax2d()
        rng = np.random.default_rng(0)
        data = rng.normal(0, 1, 30)
        result = render_violin_matplotlib(ax, data, showpoints=True)
        assert "points" in result

    def test_violin_no_box_means_medians(self):
        ax = _ax2d()
        rng = np.random.default_rng(0)
        data = rng.normal(0, 1, 30)
        result = render_violin_matplotlib(
            ax, data, showbox=False, showmeans=False, showmedians=False, showpoints=False
        )
        assert "violin" in result

    def test_violin_filters_meanline_kwarg(self):
        ax = _ax2d()
        rng = np.random.default_rng(0)
        data = rng.normal(0, 1, 30)
        # meanline kwarg should be silently filtered
        result = render_violin_matplotlib(ax, data, meanline=True, showpoints=False)
        assert "violin" in result

    def test_violin_no_label(self):
        ax = _ax2d()
        rng = np.random.default_rng(0)
        data = rng.normal(0, 1, 30)
        result = render_violin_matplotlib(ax, data, label=None, showpoints=False)
        assert "legend_handle" not in result


# ---------------------------------------------------------------------------
# render_box_matplotlib
# ---------------------------------------------------------------------------

from neural_analysis.plotting.renderers_matplotlib import render_box_matplotlib


class TestRenderBoxMatplotlib:
    def teardown_method(self):
        plt.close("all")

    def test_box_basic(self):
        ax = _ax2d()
        rng = np.random.default_rng(0)
        data = rng.normal(0, 1, 50)
        bp = render_box_matplotlib(ax, data, position=1)
        assert "boxes" in bp

    def test_box_with_color(self):
        ax = _ax2d()
        rng = np.random.default_rng(0)
        data = rng.normal(0, 1, 50)
        bp = render_box_matplotlib(ax, data, position=1, color="red")
        assert "boxes" in bp

    def test_box_with_label(self):
        ax = _ax2d()
        rng = np.random.default_rng(0)
        data = rng.normal(0, 1, 50)
        bp = render_box_matplotlib(ax, data, position=1, label="grp")
        assert "legend_handle" in bp

    def test_box_no_label(self):
        ax = _ax2d()
        rng = np.random.default_rng(0)
        data = rng.normal(0, 1, 50)
        bp = render_box_matplotlib(ax, data, position=1, label=None)
        assert "legend_handle" not in bp

    def test_box_showpoints_false(self):
        ax = _ax2d()
        rng = np.random.default_rng(0)
        data = rng.normal(0, 1, 30)
        bp = render_box_matplotlib(ax, data, showpoints=False)
        assert "points" not in bp

    def test_box_showpoints_true(self):
        ax = _ax2d()
        rng = np.random.default_rng(0)
        data = rng.normal(0, 1, 30)
        bp = render_box_matplotlib(ax, data, showpoints=True)
        assert "points" in bp

    def test_box_notch(self):
        ax = _ax2d()
        rng = np.random.default_rng(0)
        data = rng.normal(0, 1, 50)
        bp = render_box_matplotlib(ax, data, notch=True)
        assert "boxes" in bp


# ---------------------------------------------------------------------------
# render_trajectory_matplotlib
# ---------------------------------------------------------------------------

from neural_analysis.plotting.renderers_matplotlib import render_trajectory_matplotlib


class TestRenderTrajectoryMatplotlib:
    def teardown_method(self):
        plt.close("all")

    def test_trajectory_basic(self):
        ax = _ax2d()
        x = np.linspace(0, 1, 20)
        y = np.sin(x)
        lc = render_trajectory_matplotlib(ax, x, y)
        assert lc is not None

    def test_trajectory_with_colors(self):
        ax = _ax2d()
        x = np.linspace(0, 1, 10)
        y = np.cos(x)
        colors = np.linspace(0, 1, 10)
        lc = render_trajectory_matplotlib(ax, x, y, colors=colors, colorbar=True)
        assert lc is not None

    def test_trajectory_with_colors_and_colorbar_label(self):
        ax = _ax2d()
        x = np.linspace(0, 1, 10)
        y = np.cos(x)
        colors = np.linspace(0, 1, 10)
        lc = render_trajectory_matplotlib(ax, x, y, colors=colors, colorbar=True, colorbar_label="speed")
        assert lc is not None

    def test_trajectory_no_colorbar(self):
        ax = _ax2d()
        x = np.linspace(0, 1, 10)
        y = np.cos(x)
        colors = np.linspace(0, 1, 10)
        lc = render_trajectory_matplotlib(ax, x, y, colors=colors, colorbar=False)
        assert lc is not None

    def test_trajectory_no_colors_colorbar_skipped(self):
        """colorbar=True but no colors → no colorbar added."""
        ax = _ax2d()
        x = np.linspace(0, 1, 10)
        y = np.cos(x)
        lc = render_trajectory_matplotlib(ax, x, y, colors=None, colorbar=True)
        assert lc is not None

    def test_trajectory_show_points(self):
        ax = _ax2d()
        x = np.linspace(0, 1, 10)
        y = np.cos(x)
        lc = render_trajectory_matplotlib(ax, x, y, show_points=True)
        assert lc is not None

    def test_trajectory_mismatched_lengths_raises(self):
        ax = _ax2d()
        with pytest.raises(ValueError, match="same length"):
            render_trajectory_matplotlib(ax, np.array([0.0, 1.0]), np.array([0.0]))

    def test_trajectory_single_point_with_show_points(self):
        ax = _ax2d()
        result = render_trajectory_matplotlib(ax, np.array([0.5]), np.array([0.5]), show_points=True)
        assert result is ax  # returns ax for single point

    def test_trajectory_single_point_no_show_points(self):
        ax = _ax2d()
        result = render_trajectory_matplotlib(ax, np.array([0.5]), np.array([0.5]), show_points=False)
        assert result is ax

    def test_trajectory_with_label(self):
        ax = _ax2d()
        x = np.linspace(0, 1, 5)
        y = np.zeros(5)
        lc = render_trajectory_matplotlib(ax, x, y, label="traj", colors=None)
        assert lc is not None


# ---------------------------------------------------------------------------
# render_trajectory3d_matplotlib
# ---------------------------------------------------------------------------

from neural_analysis.plotting.renderers_matplotlib import render_trajectory3d_matplotlib


class TestRenderTrajectory3dMatplotlib:
    def teardown_method(self):
        plt.close("all")

    def test_trajectory3d_basic(self):
        ax = _ax3d()
        x = np.linspace(0, 1, 10)
        y = np.cos(x)
        z = np.sin(x)
        lc = render_trajectory3d_matplotlib(ax, x, y, z)
        assert lc is not None

    def test_trajectory3d_with_colors(self):
        ax = _ax3d()
        x = np.linspace(0, 1, 10)
        y = np.cos(x)
        z = np.sin(x)
        colors = np.linspace(0, 1, 10)
        lc = render_trajectory3d_matplotlib(ax, x, y, z, colors=colors, colorbar=True)
        assert lc is not None

    def test_trajectory3d_colorbar_label(self):
        ax = _ax3d()
        x = np.linspace(0, 1, 10)
        y = np.cos(x)
        z = np.sin(x)
        colors = np.linspace(0, 1, 10)
        lc = render_trajectory3d_matplotlib(ax, x, y, z, colors=colors, colorbar=True, colorbar_label="speed")
        assert lc is not None

    def test_trajectory3d_no_colorbar(self):
        ax = _ax3d()
        x = np.linspace(0, 1, 5)
        y = np.zeros(5)
        z = np.zeros(5)
        lc = render_trajectory3d_matplotlib(ax, x, y, z, colorbar=False)
        assert lc is not None

    def test_trajectory3d_show_points_with_colors(self):
        ax = _ax3d()
        x = np.linspace(0, 1, 5)
        y = np.zeros(5)
        z = np.zeros(5)
        colors = np.linspace(0, 1, 5)
        lc = render_trajectory3d_matplotlib(ax, x, y, z, colors=colors, show_points=True, colorbar=False)
        assert lc is not None

    def test_trajectory3d_show_points_no_colors(self):
        ax = _ax3d()
        x = np.linspace(0, 1, 5)
        y = np.zeros(5)
        z = np.zeros(5)
        lc = render_trajectory3d_matplotlib(ax, x, y, z, show_points=True, colorbar=False)
        assert lc is not None

    def test_trajectory3d_mismatched_raises(self):
        ax = _ax3d()
        with pytest.raises(ValueError, match="same length"):
            render_trajectory3d_matplotlib(ax, np.array([0.0, 1.0]), np.array([0.0]), np.array([0.0]))

    def test_trajectory3d_too_few_points_raises(self):
        ax = _ax3d()
        with pytest.raises(ValueError, match="at least 2"):
            render_trajectory3d_matplotlib(ax, np.array([0.0]), np.array([0.0]), np.array([0.0]))


# ---------------------------------------------------------------------------
# render_kde_matplotlib
# ---------------------------------------------------------------------------

from neural_analysis.plotting.renderers_matplotlib import render_kde_matplotlib


class TestRenderKdeMatplotlib:
    def teardown_method(self):
        plt.close("all")

    def _grid(self, n=20):
        x = np.linspace(0, 1, n)
        y = np.linspace(0, 1, n)
        xi, yi = np.meshgrid(x, y)
        zi = np.exp(-((xi - 0.5) ** 2 + (yi - 0.5) ** 2) / 0.1)
        return xi, yi, zi

    def test_kde_fill_basic(self):
        ax = _ax2d()
        xi, yi, zi = self._grid()
        cs = render_kde_matplotlib(ax, xi, yi, zi, fill=True)
        assert cs is not None

    def test_kde_no_fill(self):
        ax = _ax2d()
        xi, yi, zi = self._grid()
        cs = render_kde_matplotlib(ax, xi, yi, zi, fill=False)
        assert cs is not None

    def test_kde_with_colorbar_label(self):
        ax = _ax2d()
        xi, yi, zi = self._grid()
        cs = render_kde_matplotlib(ax, xi, yi, zi, colorbar=True, colorbar_label="density")
        assert cs is not None

    def test_kde_no_colorbar(self):
        ax = _ax2d()
        xi, yi, zi = self._grid()
        cs = render_kde_matplotlib(ax, xi, yi, zi, colorbar=False)
        assert cs is not None


# ---------------------------------------------------------------------------
# render_convex_hull_matplotlib
# ---------------------------------------------------------------------------

from neural_analysis.plotting.renderers_matplotlib import render_convex_hull_matplotlib


class TestRenderConvexHullMatplotlib:
    def teardown_method(self):
        plt.close("all")

    def test_hull_basic(self):
        ax = _ax2d()
        hull_x = np.array([0.0, 1.0, 1.0, 0.0, 0.0])
        hull_y = np.array([0.0, 0.0, 1.0, 1.0, 0.0])
        line = render_convex_hull_matplotlib(ax, hull_x, hull_y)
        assert line is not None
        import matplotlib.lines as mlines
        assert isinstance(line, mlines.Line2D)

    def test_hull_with_fill(self):
        ax = _ax2d()
        hull_x = np.array([0.0, 1.0, 0.5, 0.0])
        hull_y = np.array([0.0, 0.0, 1.0, 0.0])
        line = render_convex_hull_matplotlib(ax, hull_x, hull_y, fill=True, fill_alpha=0.3)
        assert line is not None

    def test_hull_no_fill(self):
        ax = _ax2d()
        hull_x = np.array([0.0, 1.0, 0.5, 0.0])
        hull_y = np.array([0.0, 0.0, 1.0, 0.0])
        line = render_convex_hull_matplotlib(ax, hull_x, hull_y, fill=False)
        assert line is not None

    def test_hull_with_label(self):
        ax = _ax2d()
        hull_x = np.array([0.0, 1.0, 0.5, 0.0])
        hull_y = np.array([0.0, 0.0, 1.0, 0.0])
        line = render_convex_hull_matplotlib(ax, hull_x, hull_y, label="hull")
        assert line.get_label() == "hull"


# ---------------------------------------------------------------------------
# render_boolean_states_matplotlib
# ---------------------------------------------------------------------------

from neural_analysis.plotting.renderers_matplotlib import render_boolean_states_matplotlib


class TestRenderBooleanStatesMatplotlib:
    def teardown_method(self):
        plt.close("all")

    def test_boolean_basic(self):
        ax = _ax2d()
        x = np.linspace(0, 1, 20)
        states = np.array([True] * 5 + [False] * 10 + [True] * 5)
        artists = render_boolean_states_matplotlib(ax, x, states)
        assert isinstance(artists, list)
        assert len(artists) > 0

    def test_boolean_all_true(self):
        ax = _ax2d()
        x = np.linspace(0, 1, 10)
        states = np.ones(10, dtype=bool)
        artists = render_boolean_states_matplotlib(ax, x, states)
        assert isinstance(artists, list)

    def test_boolean_all_false(self):
        ax = _ax2d()
        x = np.linspace(0, 1, 10)
        states = np.zeros(10, dtype=bool)
        artists = render_boolean_states_matplotlib(ax, x, states)
        assert isinstance(artists, list)

    def test_boolean_starts_with_false(self):
        ax = _ax2d()
        x = np.linspace(0, 1, 10)
        states = np.array([False] * 5 + [True] * 5)
        artists = render_boolean_states_matplotlib(ax, x, states)
        assert isinstance(artists, list)

    def test_boolean_ends_with_false(self):
        ax = _ax2d()
        x = np.linspace(0, 1, 10)
        states = np.array([True] * 5 + [False] * 5)
        artists = render_boolean_states_matplotlib(ax, x, states)
        assert isinstance(artists, list)

    def test_boolean_alternating(self):
        ax = _ax2d()
        x = np.linspace(0, 1, 8)
        states = np.array([True, False, True, False, True, False, True, False])
        artists = render_boolean_states_matplotlib(ax, x, states)
        assert isinstance(artists, list)

    def test_boolean_custom_colors(self):
        ax = _ax2d()
        x = np.linspace(0, 1, 10)
        states = np.array([True] * 5 + [False] * 5)
        artists = render_boolean_states_matplotlib(
            ax, x, states, true_color="green", false_color="red"
        )
        assert isinstance(artists, list)


# ---------------------------------------------------------------------------
# render_ellipse_matplotlib
# ---------------------------------------------------------------------------

from neural_analysis.plotting.renderers_matplotlib import render_ellipse_matplotlib


class TestRenderEllipseMatplotlib:
    def teardown_method(self):
        plt.close("all")

    def test_ellipse_1d(self):
        ax = _ax2d()
        centers = np.array([[0.0], [1.0], [2.0]])
        widths = np.array([0.5, 0.5, 0.5])
        heights = np.array([1.0, 1.0, 1.0])
        patches = render_ellipse_matplotlib(ax, centers, widths, heights)
        assert len(patches) == 3
        from matplotlib.patches import Rectangle
        assert all(isinstance(p, Rectangle) for p in patches)

    def test_ellipse_2d(self):
        ax = _ax2d()
        centers = np.array([[0.0, 0.0], [1.0, 1.0]])
        widths = np.array([0.5, 0.8])
        heights = np.array([0.3, 0.6])
        patches = render_ellipse_matplotlib(ax, centers, widths, heights)
        assert len(patches) == 2
        from matplotlib.patches import Ellipse
        assert all(isinstance(p, Ellipse) for p in patches)

    def test_ellipse_2d_with_angles(self):
        ax = _ax2d()
        centers = np.array([[0.0, 0.0], [1.0, 1.0]])
        widths = np.array([0.5, 0.8])
        heights = np.array([0.3, 0.6])
        angles = np.array([0.0, 45.0])
        patches = render_ellipse_matplotlib(ax, centers, widths, heights, angles=angles)
        assert len(patches) == 2

    def test_ellipse_3d(self):
        ax = _ax3d()
        centers = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        widths = np.array([0.5, 0.8])
        heights = np.array([0.3, 0.6])
        patches = render_ellipse_matplotlib(ax, centers, widths, heights)
        assert len(patches) == 2

    def test_ellipse_2d_no_angles(self):
        ax = _ax2d()
        centers = np.array([[0.5, 0.5]])
        widths = np.array([1.0])
        heights = np.array([0.5])
        patches = render_ellipse_matplotlib(ax, centers, widths, heights, angles=None)
        assert len(patches) == 1

    def test_ellipse_with_edgecolor(self):
        ax = _ax2d()
        centers = np.array([[0.0, 0.0]])
        widths = np.array([1.0])
        heights = np.array([0.5])
        patches = render_ellipse_matplotlib(
            ax, centers, widths, heights, edgecolor="black", linewidth=1.5
        )
        assert len(patches) == 1


# ---------------------------------------------------------------------------
# Additional branch-coverage tests
# ---------------------------------------------------------------------------


class TestViolinShowboxOnlyBranches:
    """Cover the individual showbox / showmedians / showmeans branches."""

    def teardown_method(self):
        plt.close("all")

    def test_violin_showbox_only(self):
        ax = _ax2d()
        rng = np.random.default_rng(0)
        data = rng.normal(0, 1, 40)
        result = render_violin_matplotlib(
            ax, data, showbox=True, showmeans=False, showmedians=False, showpoints=False
        )
        assert "violin" in result

    def test_violin_showmedians_only(self):
        ax = _ax2d()
        rng = np.random.default_rng(0)
        data = rng.normal(0, 1, 40)
        result = render_violin_matplotlib(
            ax, data, showbox=False, showmeans=False, showmedians=True, showpoints=False
        )
        assert "violin" in result

    def test_violin_showmeans_only(self):
        ax = _ax2d()
        rng = np.random.default_rng(0)
        data = rng.normal(0, 1, 40)
        result = render_violin_matplotlib(
            ax, data, showbox=False, showmeans=True, showmedians=False, showpoints=False
        )
        assert "violin" in result

    def test_violin_showbox_and_medians_no_means(self):
        ax = _ax2d()
        rng = np.random.default_rng(0)
        data = rng.normal(0, 1, 40)
        result = render_violin_matplotlib(
            ax, data, showbox=True, showmeans=False, showmedians=True, showpoints=False
        )
        assert "violin" in result


class TestLineMultiColumnMarkerNoSize:
    """Cover multi-column path with marker but no marker_size (191->194 branch miss)."""

    def teardown_method(self):
        plt.close("all")

    def test_multi_column_marker_no_size(self):
        """3-column data + marker with no marker_size should not set markersize."""
        ax = _ax2d()
        rng = np.random.default_rng(0)
        data = rng.uniform(size=(5, 3))
        lines = render_line_matplotlib(ax, data, marker="s", marker_size=None)
        assert len(lines) == 3


class TestTrajectoryEmptyRaises:
    """Cover the len(x) < 1 branch (line 1025) in render_trajectory_matplotlib."""

    def teardown_method(self):
        plt.close("all")

    def test_trajectory_empty_raises(self):
        """An empty array (length 0) bypasses the single-point check and hits the guard."""
        ax = _ax2d()
        # len(x)==0 → NOT caught by len==1 guard → falls through to len<1 raise
        with pytest.raises((ValueError, Exception)):
            render_trajectory_matplotlib(ax, np.array([]), np.array([]))


class TestEllipseHigherDims:
    """Cover the 'elif n_dims == 3 not taken' branch (n_dims > 3 passes all conditions silently)."""

    def teardown_method(self):
        plt.close("all")

    def test_ellipse_4d_returns_empty(self):
        """n_dims == 4 falls through all if/elif without adding patches → empty list."""
        ax = _ax2d()
        centers = np.ones((2, 4))
        widths = np.ones(2)
        heights = np.ones(2)
        patches = render_ellipse_matplotlib(ax, centers, widths, heights)
        assert isinstance(patches, list)
        assert len(patches) == 0
