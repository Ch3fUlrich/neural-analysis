"""
Coverage tests for neural_analysis.plotting.renderers.

Target: reach >= 95% combined line+branch coverage of renderers.py.
Uses only matplotlib Agg backend; never writes to the repo tree.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock

from neural_analysis.plotting.renderers import (
    extract_xy_from_data,
    extract_xyz_from_data,
    render_scatter_matplotlib,
    render_scatter_plotly,
    render_scatter3d_plotly,
    render_line_matplotlib,
    render_line_plotly,
    render_histogram_matplotlib,
    render_histogram_plotly,
    render_heatmap_matplotlib,
    render_heatmap_plotly,
    render_heatmap_walls_matplotlib,
    render_bar_matplotlib,
    render_bar_plotly,
    render_violin_matplotlib,
    render_violin_plotly,
    render_box_matplotlib,
    render_box_plotly,
    render_trajectory_matplotlib,
    render_trajectory_plotly,
    render_trajectory3d_matplotlib,
    render_trajectory3d_plotly,
    render_kde_matplotlib,
    render_kde_plotly,
    render_convex_hull_matplotlib,
    render_convex_hull_plotly,
    render_boolean_states_matplotlib,
    render_boolean_states_plotly,
    render_ellipse_matplotlib,
    render_ellipse_plotly,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ax2d():
    fig, ax = plt.subplots()
    return ax


def _ax3d():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    return ax


def _rng():
    return np.random.default_rng(0)


# ===========================================================================
# Lines 30-31: PLOTLY_AVAILABLE = False branch (import guard)
# ===========================================================================

class TestPlotlyImportFallback:
    def teardown_method(self):
        plt.close("all")

    def test_scatter_plotly_unavailable_raises(self):
        """Line 292-293: PLOTLY_AVAILABLE=False → ImportError."""
        with patch("neural_analysis.plotting.renderers.PLOTLY_AVAILABLE", False):
            with pytest.raises(ImportError, match="Plotly is required"):
                render_scatter_plotly(np.ones((5, 2)))

    def test_scatter3d_plotly_unavailable_raises(self):
        """Lines 375-376: PLOTLY_AVAILABLE=False → ImportError."""
        with patch("neural_analysis.plotting.renderers.PLOTLY_AVAILABLE", False):
            with pytest.raises(ImportError, match="Plotly is required"):
                render_scatter3d_plotly(np.ones((5, 3)))

    def test_line_plotly_unavailable_raises(self):
        with patch("neural_analysis.plotting.renderers.PLOTLY_AVAILABLE", False):
            with pytest.raises(ImportError, match="Plotly is required"):
                render_line_plotly(np.array([1.0, 2.0, 3.0]))

    def test_histogram_plotly_unavailable_raises(self):
        with patch("neural_analysis.plotting.renderers.PLOTLY_AVAILABLE", False):
            with pytest.raises(ImportError, match="Plotly is required"):
                render_histogram_plotly(np.array([1.0, 2.0, 3.0]))

    def test_heatmap_plotly_unavailable_raises(self):
        with patch("neural_analysis.plotting.renderers.PLOTLY_AVAILABLE", False):
            with pytest.raises(ImportError, match="Plotly is required"):
                render_heatmap_plotly(np.ones((3, 3)))

    def test_bar_plotly_unavailable_raises(self):
        with patch("neural_analysis.plotting.renderers.PLOTLY_AVAILABLE", False):
            with pytest.raises(ImportError, match="Plotly is required"):
                render_bar_plotly(np.array([1.0, 2.0, 3.0]))

    def test_violin_plotly_unavailable_raises(self):
        with patch("neural_analysis.plotting.renderers.PLOTLY_AVAILABLE", False):
            with pytest.raises(ImportError, match="Plotly is required"):
                render_violin_plotly(np.array([1.0, 2.0, 3.0]))

    def test_box_plotly_unavailable_raises(self):
        """Line 1684: PLOTLY_AVAILABLE=False → ImportError."""
        with patch("neural_analysis.plotting.renderers.PLOTLY_AVAILABLE", False):
            with pytest.raises(ImportError, match="Plotly is required"):
                render_box_plotly(np.array([1.0, 2.0, 3.0]))

    def test_trajectory_plotly_unavailable_raises(self):
        with patch("neural_analysis.plotting.renderers.PLOTLY_AVAILABLE", False):
            with pytest.raises(ImportError, match="Plotly is required"):
                render_trajectory_plotly(np.array([0.0, 1.0]), np.array([0.0, 1.0]))

    def test_trajectory3d_plotly_unavailable_raises(self):
        with patch("neural_analysis.plotting.renderers.PLOTLY_AVAILABLE", False):
            with pytest.raises(ImportError, match="Plotly is required"):
                render_trajectory3d_plotly(
                    np.array([0.0, 1.0]),
                    np.array([0.0, 1.0]),
                    np.array([0.0, 1.0]),
                )

    def test_kde_plotly_unavailable_raises(self):
        with patch("neural_analysis.plotting.renderers.PLOTLY_AVAILABLE", False):
            with pytest.raises(ImportError, match="Plotly is required"):
                render_kde_plotly(np.ones((5, 5)), np.ones((5, 5)), np.ones((5, 5)))

    def test_convex_hull_plotly_unavailable_raises(self):
        """Line 2360: PLOTLY_AVAILABLE=False → ImportError."""
        with patch("neural_analysis.plotting.renderers.PLOTLY_AVAILABLE", False):
            with pytest.raises(ImportError, match="Plotly is required"):
                render_convex_hull_plotly(np.array([0.0, 1.0]), np.array([0.0, 1.0]))

    def test_boolean_states_plotly_unavailable_raises(self):
        with patch("neural_analysis.plotting.renderers.PLOTLY_AVAILABLE", False):
            with pytest.raises(ImportError, match="Plotly is required"):
                render_boolean_states_plotly(np.array([0.0, 1.0]), np.array([True, False]))


# ===========================================================================
# render_scatter_matplotlib – uncovered branches
# ===========================================================================

class TestScatterMatplotlibBranches:
    def teardown_method(self):
        plt.close("all")

    def test_scatter_invalid_shape_raises(self):
        """Line 236: shape[1] not 2 or 3 → ValueError."""
        ax = _ax2d()
        data = np.ones((5, 4))
        with pytest.raises(ValueError, match="2D or 3D"):
            render_scatter_matplotlib(ax, data)

    def test_scatter_3d_with_colors_and_cmap(self):
        """Lines 227-234: 3D scatter with colors array and cmap."""
        ax = _ax3d()
        rng = _rng()
        data = rng.uniform(size=(10, 3))
        colors = rng.uniform(size=10)
        sc = render_scatter_matplotlib(ax, data, colors=colors, cmap="plasma")
        assert sc is not None

    def test_scatter_2d_non_string_c_not_colors(self):
        """Branch: colors is None, c_param not str → cmap branch NOT triggered."""
        ax = _ax2d()
        data = np.ones((5, 2))
        # color=None, colors=None → c_param is None → no cmap added
        sc = render_scatter_matplotlib(ax, data)
        assert sc is not None


# ===========================================================================
# render_scatter_plotly – uncovered branches
# ===========================================================================

class TestScatterPlotlyBranches:
    def teardown_method(self):
        plt.close("all")

    def test_scatter_plotly_invalid_shape_raises(self):
        """Line 295: shape[1] != 2 → ValueError."""
        data = np.ones((5, 3))
        with pytest.raises(ValueError, match="2-column data"):
            render_scatter_plotly(data)

    def test_scatter_plotly_with_sizes(self):
        """Line 300: sizes is not None → uses sizes array."""
        rng = _rng()
        data = rng.uniform(size=(10, 2))
        sizes = rng.uniform(1, 20, size=10)
        trace = render_scatter_plotly(data, sizes=sizes)
        assert trace is not None
        assert trace.marker.size is not None

    def test_scatter_plotly_with_colors_and_cmap_and_colorbar(self):
        """Lines 306-311: colors, cmap, colorbar all set."""
        rng = _rng()
        data = rng.uniform(size=(10, 2))
        colors = rng.uniform(size=10)
        trace = render_scatter_plotly(
            data, colors=colors, cmap="Viridis", colorbar=True, colorbar_label="val"
        )
        assert trace is not None
        # colorscale is expanded by plotly from name to tuple of color stops
        assert trace.marker.colorscale is not None

    def test_scatter_plotly_with_colors_no_cmap_no_colorbar(self):
        """Lines 305-311: colors set, cmap=None, colorbar=False."""
        rng = _rng()
        data = rng.uniform(size=(8, 2))
        colors = rng.uniform(size=8)
        trace = render_scatter_plotly(data, colors=colors, cmap=None, colorbar=False)
        assert trace is not None

    def test_scatter_plotly_with_single_color(self):
        """Line 312-313: color (not colors) branch."""
        rng = _rng()
        data = rng.uniform(size=(8, 2))
        trace = render_scatter_plotly(data, color="blue")
        assert trace is not None
        assert trace.marker.color == "blue"

    def test_scatter_plotly_empty_colors_array(self):
        """Branch: colors is not None but len==0 → falls through to single color."""
        rng = _rng()
        data = rng.uniform(size=(8, 2))
        trace = render_scatter_plotly(data, colors=np.array([]), color="red")
        assert trace is not None


# ===========================================================================
# render_scatter3d_plotly – uncovered branches
# ===========================================================================

class TestScatter3DPlotlyBranches:
    def teardown_method(self):
        plt.close("all")

    def test_scatter3d_invalid_shape_raises(self):
        """Line 378-379: shape[1] != 3 → ValueError."""
        data = np.ones((5, 2))
        with pytest.raises(ValueError, match="3-column data"):
            render_scatter3d_plotly(data)

    def test_scatter3d_with_colors_cmap_colorbar_label(self):
        """Lines 388-394: colors, cmap, colorbar, colorbar_label all set."""
        rng = _rng()
        data = rng.uniform(size=(10, 3))
        colors = rng.uniform(size=10)
        trace = render_scatter3d_plotly(
            data, colors=colors, cmap="Plasma", colorbar=True, colorbar_label="speed"
        )
        assert trace is not None
        # colorscale is expanded by plotly from name to tuple of color stops
        assert trace.marker.colorscale is not None

    def test_scatter3d_with_colors_no_cmap_no_colorbar(self):
        """Branch: colors set, cmap=None, colorbar=False."""
        rng = _rng()
        data = rng.uniform(size=(10, 3))
        colors = rng.uniform(size=10)
        trace = render_scatter3d_plotly(data, colors=colors, colorbar=False)
        assert trace is not None

    def test_scatter3d_with_single_color(self):
        """Lines 395-396: single color (not colors array)."""
        rng = _rng()
        data = rng.uniform(size=(8, 3))
        trace = render_scatter3d_plotly(data, color="green")
        assert trace is not None
        assert trace.marker.color == "green"

    def test_scatter3d_with_sizes_array(self):
        """Line 383: sizes is not None."""
        rng = _rng()
        data = rng.uniform(size=(8, 3))
        sizes = rng.uniform(1, 10, size=8)
        trace = render_scatter3d_plotly(data, sizes=sizes)
        assert trace is not None

    def test_scatter3d_show_points(self):
        """Line 383: show_points=True → marker dict size = point_size."""
        rng = _rng()
        data = rng.uniform(size=(6, 3))
        trace = render_scatter3d_plotly(data, marker_size=5.0)
        assert trace is not None


# ===========================================================================
# render_line_matplotlib – uncovered branches
# ===========================================================================

class TestLineMatplotlibBranches:
    def teardown_method(self):
        plt.close("all")

    def test_line_dict_data(self):
        """Lines 476-478: dict with x and y."""
        ax = _ax2d()
        data = {"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 0.0]}
        lines = render_line_matplotlib(ax, data)
        assert len(lines) == 1

    def test_line_dict_data_missing_keys_raises(self):
        """Line 480: dict missing keys → ValueError."""
        ax = _ax2d()
        with pytest.raises(ValueError, match="'x' and 'y' keys"):
            render_line_matplotlib(ax, {"z": [1, 2]})

    def test_line_1d_data(self):
        """Lines 485-488: 1D data."""
        ax = _ax2d()
        data = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
        lines = render_line_matplotlib(ax, data)
        assert len(lines) == 1

    def test_line_multi_column_no_marker(self):
        """Lines 492-511: multi-column without marker."""
        ax = _ax2d()
        rng = _rng()
        data = rng.uniform(size=(10, 3))
        lines = render_line_matplotlib(ax, data, color="red")
        assert len(lines) == 3

    def test_line_multi_column_with_marker_and_markersize(self):
        """Lines 493-496: multi-column with marker and marker_size."""
        ax = _ax2d()
        rng = _rng()
        data = rng.uniform(size=(5, 3))
        lines = render_line_matplotlib(ax, data, marker="o", marker_size=8)
        assert len(lines) == 3

    def test_line_multi_column_with_marker_no_markersize(self):
        """Line 493-494: multi-column with marker but no marker_size."""
        ax = _ax2d()
        rng = _rng()
        data = rng.uniform(size=(5, 3))
        lines = render_line_matplotlib(ax, data, marker="^")
        assert len(lines) == 3

    def test_line_single_with_marker_and_markersize(self):
        """Lines 515-518: single line with marker and marker_size."""
        ax = _ax2d()
        data = np.array([1.0, 2.0, 3.0])
        lines = render_line_matplotlib(ax, data, marker="s", marker_size=10)
        assert len(lines) == 1

    def test_line_single_with_marker_no_markersize(self):
        """Line 515-516: single line with marker, no marker_size."""
        ax = _ax2d()
        data = np.array([1.0, 2.0, 3.0])
        lines = render_line_matplotlib(ax, data, marker="D")
        assert len(lines) == 1

    def test_line_show_values_nonzero_range(self):
        """Lines 533-544: show_values=True with nonzero y-range."""
        ax = _ax2d()
        data = np.array([1.0, 3.0, 2.0])
        lines = render_line_matplotlib(ax, data, show_values=True, value_format=".2f")
        assert len(lines) == 1
        texts = [c for c in ax.get_children() if hasattr(c, "get_text")]
        assert len(texts) >= 3

    def test_line_show_values_zero_range(self):
        """Line 534: zero y-range → y_range defaults to 1."""
        ax = _ax2d()
        data = np.array([5.0, 5.0, 5.0])
        lines = render_line_matplotlib(ax, data, show_values=True)
        assert len(lines) == 1

    def test_line_x_labels(self):
        """Lines 547-549: x_labels."""
        ax = _ax2d()
        data = np.array([1.0, 2.0, 3.0])
        lines = render_line_matplotlib(ax, data, x_labels=["a", "b", "c"])
        assert len(lines) == 1
        labels = [t.get_text() for t in ax.get_xticklabels()]
        assert "a" in labels

    def test_line_error_y_with_color(self):
        """Lines 552-562: error_y provided, color given."""
        ax = _ax2d()
        data = np.array([1.0, 2.0, 3.0])
        error_y = np.array([0.1, 0.2, 0.1])
        lines = render_line_matplotlib(ax, data, color="red", error_y=error_y)
        assert len(lines) == 1

    def test_line_error_y_no_color(self):
        """Line 554: no color → uses line color from first line."""
        ax = _ax2d()
        data = np.array([1.0, 2.0, 3.0])
        error_y = np.array([0.1, 0.2, 0.1])
        lines = render_line_matplotlib(ax, data, color=None, error_y=error_y)
        assert len(lines) == 1


# ===========================================================================
# render_line_plotly – uncovered branches
# ===========================================================================

class TestLinePlotlyBranches:
    def teardown_method(self):
        plt.close("all")

    def test_line_plotly_linestyle_mapped(self):
        """Line 618-625: linestyle conversion + dash added."""
        trace = render_line_plotly(np.array([1.0, 2.0, 3.0]), linestyle="--")
        assert trace is not None
        assert trace.line.dash == "dash"

    def test_line_plotly_linestyle_dashdot(self):
        trace = render_line_plotly(np.array([1.0, 2.0, 3.0]), linestyle="-.")
        assert trace.line.dash == "dashdot"

    def test_line_plotly_linestyle_dot(self):
        trace = render_line_plotly(np.array([1.0, 2.0, 3.0]), linestyle=":")
        assert trace.line.dash == "dot"

    def test_line_plotly_linestyle_solid(self):
        trace = render_line_plotly(np.array([1.0, 2.0, 3.0]), linestyle="-")
        assert trace.line.dash == "solid"

    def test_line_plotly_no_linestyle(self):
        """Line 618: linestyle=None → dash_style is None → not added."""
        trace = render_line_plotly(np.array([1.0, 2.0, 3.0]), linestyle=None)
        assert trace is not None

    def test_line_plotly_error_y_with_color(self):
        """Lines 629-636: error_y with color."""
        trace = render_line_plotly(
            np.array([1.0, 2.0, 3.0]),
            error_y=np.array([0.1, 0.2, 0.1]),
            color="blue",
        )
        assert trace is not None
        assert trace.error_y.type == "data"

    def test_line_plotly_error_y_no_color(self):
        """Line 635: no color → default error color."""
        trace = render_line_plotly(
            np.array([1.0, 2.0, 3.0]),
            error_y=np.array([0.1, 0.2, 0.1]),
            color=None,
        )
        assert trace is not None

    def test_line_plotly_dict_data(self):
        """Lines 639-653: dict data path."""
        data = {"x": [0.0, 1.0, 2.0], "y": [3.0, 1.0, 2.0]}
        trace = render_line_plotly(data)
        assert trace is not None
        assert len(trace.x) == 3

    def test_line_plotly_dict_missing_keys_raises(self):
        """Lines 654-655: dict missing keys."""
        with pytest.raises(ValueError, match="'x' and 'y' keys"):
            render_line_plotly({"z": [1, 2]})

    def test_line_plotly_1d(self):
        """Lines 657-668: 1D data."""
        trace = render_line_plotly(np.array([1.0, 2.0, 3.0]))
        assert trace is not None
        assert trace.y is not None

    def test_line_plotly_2d(self):
        """Lines 669-681: 2D [x,y] data."""
        data = np.column_stack([np.arange(5.0), np.arange(5.0) ** 2])
        trace = render_line_plotly(data)
        assert trace is not None
        assert len(trace.x) == 5

    def test_line_plotly_multi_column(self):
        """Lines 682-693: multi-column data → first column used."""
        rng = _rng()
        data = rng.uniform(size=(5, 3))
        trace = render_line_plotly(data)
        assert trace is not None
        assert len(trace.y) == 5


# ===========================================================================
# render_heatmap_matplotlib – uncovered branches
# ===========================================================================

class TestHeatmapMatplotlibBranches:
    def teardown_method(self):
        plt.close("all")

    def test_heatmap_colorbar_with_label(self):
        """Lines 843-846: colorbar enabled with label."""
        ax = _ax2d()
        data = np.arange(9).reshape(3, 3).astype(float)
        im = render_heatmap_matplotlib(ax, data, colorbar=True, colorbar_label="rate")
        assert im is not None

    def test_heatmap_colorbar_no_label(self):
        """Line 843: colorbar enabled, no label."""
        ax = _ax2d()
        data = np.ones((3, 3))
        im = render_heatmap_matplotlib(ax, data, colorbar=True)
        assert im is not None

    def test_heatmap_no_colorbar(self):
        """Branch: colorbar=False."""
        ax = _ax2d()
        data = np.ones((3, 3))
        im = render_heatmap_matplotlib(ax, data, colorbar=False)
        assert im is not None

    def test_heatmap_custom_xticks_and_yticks(self):
        """Lines 849-863: set_xticks + set_xticklabels and set_yticks + set_yticklabels."""
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

    def test_heatmap_x_labels_fallback(self):
        """Lines 852-855: x_labels fallback (no set_xticks provided)."""
        ax = _ax2d()
        data = np.ones((2, 3))
        im = render_heatmap_matplotlib(ax, data, colorbar=False, x_labels=["p", "q", "r"])
        assert im is not None
        xlabels = [t.get_text() for t in ax.get_xticklabels()]
        assert "p" in xlabels

    def test_heatmap_y_labels_fallback(self):
        """Lines 860-863: y_labels fallback (no set_yticks provided)."""
        ax = _ax2d()
        data = np.ones((2, 2))
        im = render_heatmap_matplotlib(ax, data, colorbar=False, y_labels=["row0", "row1"])
        assert im is not None
        ylabels = [t.get_text() for t in ax.get_yticklabels()]
        assert "row0" in ylabels

    def test_heatmap_set_xlim_ylim(self):
        """Lines 866-872: set_xlim and set_ylim."""
        ax = _ax2d()
        data = np.ones((3, 3))
        im = render_heatmap_matplotlib(
            ax, data, colorbar=False, set_xlim=(0, 2), set_ylim=(0, 2)
        )
        assert im is not None

    def test_heatmap_show_values(self):
        """Lines 875-885: show_values=True."""
        ax = _ax2d()
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        im = render_heatmap_matplotlib(
            ax, data, colorbar=False, show_values=True, value_format=".1f"
        )
        assert im is not None
        texts = [c for c in ax.get_children() if hasattr(c, "get_text")]
        assert len(texts) >= 4

    def test_heatmap_pops_axis_labels(self):
        """Lines 837-838: x_label/y_label kwargs are popped."""
        ax = _ax2d()
        data = np.ones((3, 3))
        im = render_heatmap_matplotlib(ax, data, colorbar=False, x_label="X", y_label="Y")
        assert im is not None

    def test_heatmap_set_xlim_only(self):
        """Line 866-869: only set_xlim provided."""
        ax = _ax2d()
        data = np.ones((3, 3))
        im = render_heatmap_matplotlib(ax, data, colorbar=False, set_xlim=(0.0, 2.0))
        assert im is not None

    def test_heatmap_set_ylim_only(self):
        """Lines 870-872: only set_ylim provided."""
        ax = _ax2d()
        data = np.ones((3, 3))
        im = render_heatmap_matplotlib(ax, data, colorbar=False, set_ylim=(0.0, 2.0))
        assert im is not None


# ===========================================================================
# render_heatmap_walls_matplotlib – uncovered branches
# ===========================================================================

class TestHeatmapWallsMatplotlibBranches:
    def teardown_method(self):
        plt.close("all")

    def _make_data(self, n=5):
        rng = _rng()
        c = np.linspace(0.0, 1.0, n)
        return {
            "xy": rng.uniform(size=(n, n)),
            "xz": rng.uniform(size=(n, n)),
            "yz": rng.uniform(size=(n, n)),
            "x_centers": c,
            "y_centers": c,
            "z_centers": c,
        }

    def test_walls_default_positions_fallback(self):
        """Lines 951-953: xy_position/xz_position/yz_position not in data → fallback to centers[0]."""
        ax = _ax3d()
        data = self._make_data()
        # Don't provide position keys → they should default to centers[0]
        artists = render_heatmap_walls_matplotlib(ax, data, colorbar=False)
        assert isinstance(artists, list)
        assert len(artists) >= 3

    def test_walls_with_colorbar_label(self):
        """Lines 1030-1037: colorbar=True with colorbar_label."""
        ax = _ax3d()
        data = self._make_data()
        artists = render_heatmap_walls_matplotlib(
            ax, data, colorbar=True, colorbar_label="firing rate"
        )
        assert isinstance(artists, list)

    def test_walls_none_centers_fallback(self):
        """Lines 951-953: centers are None → default xy_position/xz_position/yz_position = 0."""
        ax = _ax3d()
        rng = _rng()
        n = 4
        c = np.linspace(0, 1, n)
        data = {
            "xy": rng.uniform(size=(n, n)),
            "xz": rng.uniform(size=(n, n)),
            "yz": rng.uniform(size=(n, n)),
            "x_centers": None,
            "y_centers": None,
            "z_centers": None,
        }
        # Should not raise - uses fallback 0 for positions
        artists = render_heatmap_walls_matplotlib(ax, data, colorbar=False)
        assert isinstance(artists, list)

    def test_walls_single_center_value(self):
        """Lines 1012-1014: range=0 when single center value."""
        ax = _ax3d()
        rng = _rng()
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

    def test_walls_colorbar_exception_handled(self):
        """Lines 1036-1037: plt.colorbar raises → except Exception: pass."""
        ax = _ax3d()
        data = self._make_data()
        with patch("matplotlib.pyplot.colorbar", side_effect=RuntimeError("colorbar error")):
            # Should NOT raise; exception is caught silently
            artists = render_heatmap_walls_matplotlib(ax, data, colorbar=True)
        assert isinstance(artists, list)


# ===========================================================================
# render_bar_matplotlib – uncovered branches
# ===========================================================================

class TestBarMatplotlibBranches:
    def teardown_method(self):
        plt.close("all")

    def test_bar_horizontal_basic(self):
        """Line 1164-1167: horizontal bars."""
        ax = _ax2d()
        data = np.array([1.0, 2.0, 3.0])
        bars = render_bar_matplotlib(ax, data, orientation="h")
        assert bars is not None

    def test_bar_horizontal_show_values_no_error(self):
        """Lines 1170-1185: show_values=True, error_x=None."""
        ax = _ax2d()
        data = np.array([1.0, 2.0, 3.0])
        bars = render_bar_matplotlib(ax, data, orientation="h", show_values=True)
        assert bars is not None
        texts = [c for c in ax.get_children() if hasattr(c, "get_text")]
        assert len(texts) >= 3

    def test_bar_horizontal_show_values_with_error_x(self):
        """Lines 1172-1177: show_values=True, error_x provided."""
        ax = _ax2d()
        data = np.array([1.0, 2.0, 3.0])
        error_x = np.array([0.1, 0.2, 0.15])
        bars = render_bar_matplotlib(ax, data, orientation="h", show_values=True, error_x=error_x)
        assert bars is not None

    def test_bar_horizontal_x_labels(self):
        """Lines 1188-1190: horizontal x_labels → y-axis ticks."""
        ax = _ax2d()
        data = np.array([1.0, 2.0, 3.0])
        bars = render_bar_matplotlib(ax, data, orientation="h", x_labels=["a", "b", "c"])
        assert bars is not None
        labels = [t.get_text() for t in ax.get_yticklabels()]
        assert "a" in labels

    def test_bar_vertical_show_values_with_error_y(self):
        """Lines 1199-1204: show_values=True, error_y provided."""
        ax = _ax2d()
        data = np.array([1.0, 2.0, 3.0])
        error_y = np.array([0.1, 0.2, 0.15])
        bars = render_bar_matplotlib(ax, data, show_values=True, error_y=error_y)
        assert bars is not None
        texts = [c for c in ax.get_children() if hasattr(c, "get_text")]
        assert len(texts) >= 3

    def test_bar_vertical_x_labels(self):
        """Lines 1215-1217: vertical x_labels → x-axis ticks."""
        ax = _ax2d()
        data = np.array([1.0, 2.0, 3.0])
        bars = render_bar_matplotlib(ax, data, x_labels=["a", "b", "c"])
        labels = [t.get_text() for t in ax.get_xticklabels()]
        assert "a" in labels

    def test_bar_with_provided_x_values(self):
        """Line 1159: x provided as array."""
        ax = _ax2d()
        data = np.array([2.0, 4.0])
        bars = render_bar_matplotlib(ax, data, x=np.array([10.0, 20.0]))
        assert bars is not None
        assert len(bars) == 2


# ===========================================================================
# render_bar_plotly – uncovered branches
# ===========================================================================

class TestBarPlotlyBranches:
    def teardown_method(self):
        plt.close("all")

    def test_bar_plotly_with_error_y(self):
        """Lines 1277-1279: error_y dict built."""
        data = np.array([1.0, 2.0, 3.0])
        error = np.array([0.1, 0.2, 0.15])
        trace = render_bar_plotly(data, error_y=error)
        assert trace is not None
        assert trace.error_y.type == "data"

    def test_bar_plotly_with_error_x(self):
        """Lines 1280: error_x dict built."""
        data = np.array([1.0, 2.0, 3.0])
        error = np.array([0.1, 0.2, 0.15])
        trace = render_bar_plotly(data, error_x=error)
        assert trace is not None
        assert trace.error_x.type == "data"

    def test_bar_plotly_2d_data(self):
        """Line 1284: data.ndim != 1 → uses data[:, 0]."""
        rng = _rng()
        data = rng.uniform(size=(5, 2))
        trace = render_bar_plotly(data)
        assert trace is not None
        assert len(trace.y) == 5

    def test_bar_plotly_with_colors_list(self):
        """Line 1269: colors list → bar_color = colors."""
        data = np.array([1.0, 2.0, 3.0])
        colors = ["red", "blue", "green"]
        trace = render_bar_plotly(data, colors=colors)
        assert trace is not None

    def test_bar_plotly_with_single_color(self):
        """Line 1269: colors=None → bar_color = color."""
        data = np.array([1.0, 2.0, 3.0])
        trace = render_bar_plotly(data, color="steelblue")
        assert trace is not None


# ===========================================================================
# render_violin_matplotlib – uncovered branches
# ===========================================================================

class TestViolinMatplotlibBranches:
    def teardown_method(self):
        plt.close("all")

    def test_violin_showbox_true(self):
        """Lines 1412-1416: showbox=True → vlines drawn."""
        ax = _ax2d()
        rng = _rng()
        data = rng.normal(0, 1, 50)
        result = render_violin_matplotlib(ax, data, showbox=True, showmeans=False, showmedians=False)
        assert "violin" in result

    def test_violin_showmedians_true(self):
        """Lines 1419-1429: showmedians=True → scatter for median."""
        ax = _ax2d()
        rng = _rng()
        data = rng.normal(0, 1, 50)
        result = render_violin_matplotlib(ax, data, showbox=False, showmeans=False, showmedians=True)
        assert "violin" in result

    def test_violin_showmeans_true(self):
        """Lines 1432-1442: showmeans=True → scatter for mean."""
        ax = _ax2d()
        rng = _rng()
        data = rng.normal(0, 1, 50)
        result = render_violin_matplotlib(ax, data, showbox=False, showmeans=True, showmedians=False)
        assert "violin" in result

    def test_violin_showpoints_true_with_color(self):
        """Lines 1447-1453: showpoints=True with color."""
        ax = _ax2d()
        rng = _rng()
        data = rng.normal(0, 1, 40)
        result = render_violin_matplotlib(ax, data, color="blue", showpoints=True)
        assert "points" in result

    def test_violin_showpoints_true_no_color(self):
        """Line 1451: color=None → 'black'."""
        ax = _ax2d()
        rng = _rng()
        data = rng.normal(0, 1, 40)
        result = render_violin_matplotlib(ax, data, color=None, showpoints=True)
        assert "points" in result

    def test_violin_showpoints_false(self):
        """Branch: showpoints=False → no 'points' key."""
        ax = _ax2d()
        rng = _rng()
        data = rng.normal(0, 1, 40)
        result = render_violin_matplotlib(ax, data, showpoints=False)
        assert "points" not in result

    def test_violin_with_label_and_color(self):
        """Lines 1456-1461: label set → legend_handle created."""
        ax = _ax2d()
        rng = _rng()
        data = rng.normal(0, 1, 40)
        result = render_violin_matplotlib(ax, data, color="green", label="group1", showpoints=False)
        assert "legend_handle" in result
        assert result["legend_handle"].get_label() == "group1"

    def test_violin_with_label_no_color(self):
        """Line 1459: label set, no color → uses 'C0'."""
        ax = _ax2d()
        rng = _rng()
        data = rng.normal(0, 1, 40)
        result = render_violin_matplotlib(ax, data, color=None, label="grp", showpoints=False)
        assert "legend_handle" in result

    def test_violin_no_label(self):
        """Branch: label=None → no legend_handle."""
        ax = _ax2d()
        rng = _rng()
        data = rng.normal(0, 1, 40)
        result = render_violin_matplotlib(ax, data, label=None, showpoints=False)
        assert "legend_handle" not in result

    def test_violin_no_box_no_means_no_medians(self):
        """Branch: all False → stats block skipped."""
        ax = _ax2d()
        rng = _rng()
        data = rng.normal(0, 1, 40)
        result = render_violin_matplotlib(
            ax, data, showbox=False, showmeans=False, showmedians=False, showpoints=False
        )
        assert "violin" in result


# ===========================================================================
# render_violin_plotly – uncovered branches
# ===========================================================================

class TestViolinPlotlyBranches:
    def teardown_method(self):
        plt.close("all")

    def test_violin_plotly_meanline_none(self):
        """Line 1510-1511: meanline=None → default dict created."""
        rng = _rng()
        data = rng.normal(0, 1, 30)
        trace = render_violin_plotly(data, meanline=None)
        assert trace is not None

    def test_violin_plotly_meanline_bool_true(self):
        """Lines 1512-1513: meanline=True (bool)."""
        rng = _rng()
        data = rng.normal(0, 1, 30)
        trace = render_violin_plotly(data, meanline=True, color="blue")
        assert trace is not None

    def test_violin_plotly_meanline_bool_false(self):
        """Lines 1512-1513: meanline=False (bool)."""
        rng = _rng()
        data = rng.normal(0, 1, 30)
        trace = render_violin_plotly(data, meanline=False)
        assert trace is not None

    def test_violin_plotly_meanline_dict_complete(self):
        """Lines 1514-1521: meanline is dict with all keys present."""
        rng = _rng()
        data = rng.normal(0, 1, 30)
        trace = render_violin_plotly(
            data, meanline={"visible": True, "color": "red", "width": 3}
        )
        assert trace is not None

    def test_violin_plotly_meanline_dict_missing_visible(self):
        """Line 1516-1517: meanline dict missing 'visible'."""
        rng = _rng()
        data = rng.normal(0, 1, 30)
        trace = render_violin_plotly(data, meanline={"color": "green", "width": 2})
        assert trace is not None

    def test_violin_plotly_meanline_dict_missing_width(self):
        """Lines 1518-1519: meanline dict missing 'width'."""
        rng = _rng()
        data = rng.normal(0, 1, 30)
        trace = render_violin_plotly(data, meanline={"visible": True, "color": "red"})
        assert trace is not None

    def test_violin_plotly_meanline_dict_missing_color(self):
        """Lines 1520-1521: meanline dict missing 'color'."""
        rng = _rng()
        data = rng.normal(0, 1, 30)
        trace = render_violin_plotly(data, meanline={"visible": True, "width": 2})
        assert trace is not None

    def test_violin_plotly_showpoints_false(self):
        """Lines 1530-1532: showpoints=False."""
        rng = _rng()
        data = rng.normal(0, 1, 30)
        trace = render_violin_plotly(data, showpoints=False)
        assert trace is not None

    def test_violin_plotly_showbox_false(self):
        """Branch: showbox=False → box=None."""
        rng = _rng()
        data = rng.normal(0, 1, 30)
        trace = render_violin_plotly(data, showbox=False)
        assert trace is not None

    def test_violin_plotly_meanline_other_type(self):
        """Lines 1514->1525: meanline is not None/bool/dict → all elif conditions fail.
        The meanline value is passed as-is to go.Violin which then rejects it."""
        rng = _rng()
        data = rng.normal(0, 1, 30)
        # Use integer - not None, not bool, not dict → skips all elif branches at 1510/1512/1514
        # go.Violin raises ValueError for invalid meanline type
        with pytest.raises(ValueError):
            render_violin_plotly(data, meanline=42)  # type: ignore[arg-type]


# ===========================================================================
# render_box_matplotlib – uncovered branches
# ===========================================================================

class TestBoxMatplotlibBranches:
    def teardown_method(self):
        plt.close("all")

    def test_box_with_label_and_color(self):
        """Lines 1637-1641: label + color → legend_handle."""
        ax = _ax2d()
        rng = _rng()
        data = rng.normal(0, 1, 50)
        bp = render_box_matplotlib(ax, data, label="grp", color="blue")
        assert "legend_handle" in bp
        assert bp["legend_handle"].get_label() == "grp"

    def test_box_with_label_no_color(self):
        """Line 1640: label set, color=None → uses 'C0'."""
        ax = _ax2d()
        rng = _rng()
        data = rng.normal(0, 1, 50)
        bp = render_box_matplotlib(ax, data, label="grp", color=None)
        assert "legend_handle" in bp

    def test_box_no_label(self):
        """Branch: label=None → no legend_handle."""
        ax = _ax2d()
        rng = _rng()
        data = rng.normal(0, 1, 50)
        bp = render_box_matplotlib(ax, data, label=None)
        assert "legend_handle" not in bp

    def test_box_showpoints_true_with_color(self):
        """Lines 1629-1634: showpoints=True with color."""
        ax = _ax2d()
        rng = _rng()
        data = rng.normal(0, 1, 30)
        bp = render_box_matplotlib(ax, data, showpoints=True, color="red")
        assert "points" in bp

    def test_box_showpoints_true_no_color(self):
        """Line 1632: color=None → 'black'."""
        ax = _ax2d()
        rng = _rng()
        data = rng.normal(0, 1, 30)
        bp = render_box_matplotlib(ax, data, showpoints=True, color=None)
        assert "points" in bp

    def test_box_color_applied_to_elements(self):
        """Lines 1617-1625: color → styles boxes, whiskers, caps, medians."""
        ax = _ax2d()
        rng = _rng()
        data = rng.normal(0, 1, 50)
        bp = render_box_matplotlib(ax, data, color="purple", showpoints=False)
        assert "boxes" in bp
        # Verify color was applied
        assert bp["boxes"][0].get_facecolor() is not None


# ===========================================================================
# render_box_plotly – uncovered branches
# ===========================================================================

class TestBoxPlotlyBranches:
    def teardown_method(self):
        plt.close("all")

    def test_box_plotly_showpoints_true(self):
        """Lines 1688-1692: showpoints=True → boxpoints='all'."""
        rng = _rng()
        data = rng.normal(0, 1, 30)
        trace = render_box_plotly(data, showpoints=True)
        assert trace is not None
        assert trace.boxpoints == "all"

    def test_box_plotly_showpoints_false(self):
        """Lines 1693-1695: showpoints=False."""
        rng = _rng()
        data = rng.normal(0, 1, 30)
        trace = render_box_plotly(data, showpoints=False)
        assert trace is not None
        assert trace.boxpoints == False  # noqa: E712

    def test_box_plotly_notched(self):
        """Parameter: notched=True."""
        rng = _rng()
        data = rng.normal(0, 1, 50)
        trace = render_box_plotly(data, notched=True)
        assert trace is not None
        assert trace.notched == True  # noqa: E712


# ===========================================================================
# render_trajectory_matplotlib – uncovered branches
# ===========================================================================

class TestTrajectoryMatplotlibBranches:
    def teardown_method(self):
        plt.close("all")

    def test_trajectory_mismatched_length_raises(self):
        """Line 1776: x and y lengths differ → ValueError."""
        ax = _ax2d()
        with pytest.raises(ValueError, match="same length"):
            render_trajectory_matplotlib(ax, np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0]))

    def test_trajectory_zero_length_raises(self):
        """Line 1785: empty arrays → len(x) < 1 → ValueError."""
        ax = _ax2d()
        with pytest.raises(ValueError, match="at least 1 point"):
            render_trajectory_matplotlib(ax, np.array([]), np.array([]))

    def test_trajectory_single_point_show_points_true(self):
        """Lines 1780-1782: single point, show_points=True."""
        ax = _ax2d()
        result = render_trajectory_matplotlib(
            ax, np.array([0.5]), np.array([0.5]), show_points=True
        )
        assert result is ax

    def test_trajectory_single_point_show_points_false(self):
        """Lines 1779-1782: single point, show_points=False → just return ax."""
        ax = _ax2d()
        result = render_trajectory_matplotlib(
            ax, np.array([0.5]), np.array([0.5]), show_points=False
        )
        assert result is ax

    def test_trajectory_with_colors_and_colorbar_label(self):
        """Lines 1804-1807: colors + colorbar=True + colorbar_label."""
        ax = _ax2d()
        x = np.linspace(0, 1, 10)
        y = np.sin(x)
        colors = np.linspace(0, 1, 10)
        lc = render_trajectory_matplotlib(
            ax, x, y, colors=colors, colorbar=True, colorbar_label="speed"
        )
        assert lc is not None

    def test_trajectory_with_colors_no_colorbar(self):
        """Branch: colorbar=False, colors provided → no colorbar."""
        ax = _ax2d()
        x = np.linspace(0, 1, 10)
        y = np.cos(x)
        colors = np.linspace(0, 1, 10)
        lc = render_trajectory_matplotlib(ax, x, y, colors=colors, colorbar=False)
        assert lc is not None

    def test_trajectory_no_colors_with_label(self):
        """Lines 1797-1799: no colors → LineCollection with label."""
        ax = _ax2d()
        x = np.linspace(0, 1, 10)
        y = np.zeros(10)
        lc = render_trajectory_matplotlib(ax, x, y, colors=None, label="traj")
        assert lc is not None

    def test_trajectory_show_points_with_colors(self):
        """Line 1811: show_points=True with colors provided."""
        ax = _ax2d()
        x = np.linspace(0, 1, 10)
        y = np.cos(x)
        colors = np.linspace(0, 1, 10)
        lc = render_trajectory_matplotlib(ax, x, y, colors=colors, show_points=True)
        assert lc is not None

    def test_trajectory_show_points_no_colors(self):
        """Line 1811: show_points=True without colors."""
        ax = _ax2d()
        x = np.linspace(0, 1, 8)
        y = np.cos(x)
        lc = render_trajectory_matplotlib(ax, x, y, show_points=True)
        assert lc is not None


# ===========================================================================
# render_trajectory_plotly – uncovered branches
# ===========================================================================

class TestTrajectoryPlotlyBranches:
    def teardown_method(self):
        plt.close("all")

    def test_trajectory_plotly_with_colors_and_show_points(self):
        """Lines 1877-1899: colors + show_points=True."""
        x = np.linspace(0, 1, 10)
        y = np.sin(x)
        colors = np.linspace(0, 1, 10)
        trace = render_trajectory_plotly(x, y, colors=colors, show_points=True)
        assert trace is not None
        assert trace.mode == "lines+markers"

    def test_trajectory_plotly_with_colors_colorbar_label(self):
        """Lines 1892-1894: colorbar=True and colorbar_label set."""
        x = np.linspace(0, 1, 10)
        y = np.sin(x)
        colors = np.linspace(0, 1, 10)
        trace = render_trajectory_plotly(
            x, y, colors=colors, colorbar=True, colorbar_label="speed"
        )
        assert trace is not None

    def test_trajectory_plotly_with_colors_no_colorbar_label(self):
        """Branch: colorbar=True but no label → None."""
        x = np.linspace(0, 1, 10)
        y = np.sin(x)
        colors = np.linspace(0, 1, 10)
        trace = render_trajectory_plotly(x, y, colors=colors, colorbar=True, colorbar_label=None)
        assert trace is not None

    def test_trajectory_plotly_no_colors_show_points_true(self):
        """Lines 1901-1911: no colors, show_points=True."""
        x = np.linspace(0, 1, 10)
        y = np.cos(x)
        trace = render_trajectory_plotly(x, y, show_points=True)
        assert trace is not None
        assert trace.mode == "lines+markers"

    def test_trajectory_plotly_no_colors_show_points_false(self):
        """Lines 1901-1911: no colors, show_points=False."""
        x = np.linspace(0, 1, 10)
        y = np.cos(x)
        trace = render_trajectory_plotly(x, y, show_points=False)
        assert trace is not None
        assert trace.mode == "lines"

    def test_trajectory_plotly_colors_list(self):
        """Line 1879: colors without tolist attr."""
        x = np.linspace(0, 1, 5)
        y = np.zeros(5)
        colors_list = [0.0, 0.25, 0.5, 0.75, 1.0]  # plain list, no tolist
        trace = render_trajectory_plotly(x, y, colors=colors_list)
        assert trace is not None


# ===========================================================================
# render_trajectory3d_plotly – uncovered branches
# ===========================================================================

class TestTrajectory3DPlotlyBranches:
    def teardown_method(self):
        plt.close("all")

    def test_trajectory3d_plotly_with_colors_and_colorbar_label(self):
        """Lines 1975-1985: colors + colorbar_label."""
        x = np.linspace(0, 1, 10)
        y = np.sin(x)
        z = np.cos(x)
        colors = np.linspace(0, 1, 10)
        trace = render_trajectory3d_plotly(
            x, y, z, colors=colors, colorbar=True, colorbar_label="speed"
        )
        assert trace is not None
        assert trace.marker.showscale == True  # noqa: E712

    def test_trajectory3d_plotly_with_colors_no_label(self):
        """Line 1977-1978: colorbar_label=None → empty colorbar_dict."""
        x = np.linspace(0, 1, 8)
        y = np.zeros(8)
        z = np.zeros(8)
        colors = np.linspace(0, 1, 8)
        trace = render_trajectory3d_plotly(x, y, z, colors=colors, colorbar_label=None)
        assert trace is not None

    def test_trajectory3d_plotly_no_colors(self):
        """Branch: colors=None → marker_config not updated."""
        x = np.linspace(0, 1, 8)
        y = np.zeros(8)
        z = np.zeros(8)
        trace = render_trajectory3d_plotly(x, y, z, colors=None)
        assert trace is not None

    def test_trajectory3d_plotly_show_points_true(self):
        """Line 1972-1973: show_points=True."""
        x = np.linspace(0, 1, 8)
        y = np.zeros(8)
        z = np.zeros(8)
        trace = render_trajectory3d_plotly(x, y, z, show_points=True)
        assert trace is not None
        assert trace.mode == "lines+markers"

    def test_trajectory3d_plotly_show_points_false(self):
        """Line 1972-1973: show_points=False → size=0.1."""
        x = np.linspace(0, 1, 8)
        y = np.zeros(8)
        z = np.zeros(8)
        trace = render_trajectory3d_plotly(x, y, z, show_points=False)
        assert trace is not None
        assert trace.mode == "lines"


# ===========================================================================
# render_trajectory3d_matplotlib – uncovered branches
# ===========================================================================

class TestTrajectory3DMatplotlibBranches:
    def teardown_method(self):
        plt.close("all")

    def test_trajectory3d_mismatched_lengths_raises(self):
        """Line 2061: x, y, z lengths differ → ValueError."""
        ax = _ax3d()
        with pytest.raises(ValueError, match="same length"):
            render_trajectory3d_matplotlib(
                ax, np.array([0.0, 1.0]), np.array([0.0]), np.array([0.0, 1.0])
            )

    def test_trajectory3d_too_few_points_raises(self):
        """Line 2065: < 2 points → ValueError."""
        ax = _ax3d()
        with pytest.raises(ValueError, match="at least 2"):
            render_trajectory3d_matplotlib(
                ax, np.array([0.0]), np.array([0.0]), np.array([0.0])
            )

    def test_trajectory3d_no_colors(self):
        """Line 2077: no colors → Line3DCollection without cmap."""
        ax = _ax3d()
        x = np.linspace(0, 1, 5)
        y = np.zeros(5)
        z = np.zeros(5)
        lc = render_trajectory3d_matplotlib(ax, x, y, z)
        assert lc is not None

    def test_trajectory3d_with_colors_and_colorbar_label(self):
        """Lines 2082-2086: colors + colorbar=True + colorbar_label."""
        ax = _ax3d()
        x = np.linspace(0, 1, 5)
        y = np.zeros(5)
        z = np.zeros(5)
        colors = np.linspace(0, 1, 5)
        lc = render_trajectory3d_matplotlib(
            ax, x, y, z, colors=colors, colorbar=True, colorbar_label="speed"
        )
        assert lc is not None

    def test_trajectory3d_with_colors_no_colorbar(self):
        """Branch: colors + colorbar=False."""
        ax = _ax3d()
        x = np.linspace(0, 1, 5)
        y = np.zeros(5)
        z = np.zeros(5)
        colors = np.linspace(0, 1, 5)
        lc = render_trajectory3d_matplotlib(ax, x, y, z, colors=colors, colorbar=False)
        assert lc is not None

    def test_trajectory3d_show_points_with_colors(self):
        """Lines 2088-2091: show_points=True with colors."""
        ax = _ax3d()
        x = np.linspace(0, 1, 5)
        y = np.zeros(5)
        z = np.zeros(5)
        colors = np.linspace(0, 1, 5)
        lc = render_trajectory3d_matplotlib(
            ax, x, y, z, colors=colors, show_points=True, colorbar=False
        )
        assert lc is not None

    def test_trajectory3d_show_points_no_colors(self):
        """Lines 2092: show_points=True without colors."""
        ax = _ax3d()
        x = np.linspace(0, 1, 5)
        y = np.zeros(5)
        z = np.zeros(5)
        lc = render_trajectory3d_matplotlib(ax, x, y, z, show_points=True, colorbar=False)
        assert lc is not None


# ===========================================================================
# render_kde_matplotlib – uncovered branches
# ===========================================================================

class TestKDEMatplotlibBranches:
    def teardown_method(self):
        plt.close("all")

    def _grid(self, n=15):
        x = np.linspace(0, 1, n)
        y = np.linspace(0, 1, n)
        xi, yi = np.meshgrid(x, y)
        zi = np.exp(-((xi - 0.5) ** 2 + (yi - 0.5) ** 2) / 0.1) + 0.01
        return xi, yi, zi

    def test_kde_fill_false(self):
        """Line 2161: fill=False → ax.contour()."""
        ax = _ax2d()
        xi, yi, zi = self._grid()
        cs = render_kde_matplotlib(ax, xi, yi, zi, fill=False)
        assert cs is not None

    def test_kde_fill_true(self):
        """Line 2157: fill=True → ax.contourf()."""
        ax = _ax2d()
        xi, yi, zi = self._grid()
        cs = render_kde_matplotlib(ax, xi, yi, zi, fill=True)
        assert cs is not None

    def test_kde_with_colorbar_label(self):
        """Lines 2165-2168: colorbar=True with colorbar_label."""
        ax = _ax2d()
        xi, yi, zi = self._grid()
        cs = render_kde_matplotlib(ax, xi, yi, zi, colorbar=True, colorbar_label="density")
        assert cs is not None

    def test_kde_colorbar_no_label(self):
        """Line 2165: colorbar=True but no label."""
        ax = _ax2d()
        xi, yi, zi = self._grid()
        cs = render_kde_matplotlib(ax, xi, yi, zi, colorbar=True, colorbar_label=None)
        assert cs is not None

    def test_kde_no_colorbar(self):
        """Branch: colorbar=False."""
        ax = _ax2d()
        xi, yi, zi = self._grid()
        cs = render_kde_matplotlib(ax, xi, yi, zi, colorbar=False)
        assert cs is not None

    def test_kde_fill_false_with_colorbar_label(self):
        """Lines 2161 + 2165-2168: fill=False + colorbar_label."""
        ax = _ax2d()
        xi, yi, zi = self._grid()
        cs = render_kde_matplotlib(ax, xi, yi, zi, fill=False, colorbar=True, colorbar_label="rate")
        assert cs is not None


# ===========================================================================
# render_kde_plotly – uncovered branches
# ===========================================================================

class TestKDEPlotlyBranches:
    def teardown_method(self):
        plt.close("all")

    def _grid(self, n=15):
        x = np.linspace(0, 1, n)
        y = np.linspace(0, 1, n)
        xi, yi = np.meshgrid(x, y)
        zi = np.exp(-((xi - 0.5) ** 2 + (yi - 0.5) ** 2) / 0.1) + 0.01
        return xi, yi, zi

    def test_kde_plotly_basic_2d_grids(self):
        """Lines 2222-2243: 2D xi/yi grids → xi[0,:] and yi[:,0]."""
        xi, yi, zi = self._grid()
        trace = render_kde_plotly(xi, yi, zi)
        assert trace is not None
        assert len(trace.x) == zi.shape[1]

    def test_kde_plotly_1d_arrays(self):
        """Branch: xi.ndim==1 and yi.ndim==1."""
        n = 10
        xi = np.linspace(0, 1, n)
        yi = np.linspace(0, 1, n)
        zi = np.outer(xi, yi)
        trace = render_kde_plotly(xi, yi, zi)
        assert trace is not None

    def test_kde_plotly_with_colorbar_label(self):
        """Line 2240: colorbar_label provided → dict(title=...)."""
        xi, yi, zi = self._grid()
        trace = render_kde_plotly(xi, yi, zi, colorbar_label="density")
        assert trace is not None

    def test_kde_plotly_no_colorbar_label(self):
        """Line 2240: colorbar_label=None → {}."""
        xi, yi, zi = self._grid()
        trace = render_kde_plotly(xi, yi, zi, colorbar_label=None)
        assert trace is not None

    def test_kde_plotly_fill_false(self):
        """fill=False still uses go.Contour (same path)."""
        xi, yi, zi = self._grid()
        trace = render_kde_plotly(xi, yi, zi, fill=False)
        assert trace is not None

    def test_kde_plotly_fill_true(self):
        """fill=True."""
        xi, yi, zi = self._grid()
        trace = render_kde_plotly(xi, yi, zi, fill=True, n_levels=5)
        assert trace is not None


# ===========================================================================
# render_convex_hull_plotly – uncovered branches
# ===========================================================================

class TestConvexHullPlotlyBranches:
    def teardown_method(self):
        plt.close("all")

    def test_hull_plotly_no_fill(self):
        """Line 2362-2363: fill=False → fill_mode='none', fill_color=None."""
        hull_x = np.array([0.0, 1.0, 1.0, 0.0, 0.0])
        hull_y = np.array([0.0, 0.0, 1.0, 1.0, 0.0])
        trace = render_convex_hull_plotly(hull_x, hull_y, fill=False)
        assert trace is not None
        assert trace.fill == "none"

    def test_hull_plotly_with_fill(self):
        """Line 2362-2363: fill=True → fill_mode='toself'."""
        hull_x = np.array([0.0, 1.0, 0.5, 0.0])
        hull_y = np.array([0.0, 0.0, 1.0, 0.0])
        trace = render_convex_hull_plotly(hull_x, hull_y, fill=True, fill_alpha=0.3)
        assert trace is not None
        assert trace.fill == "toself"

    def test_hull_plotly_opacity_fill(self):
        """Line 2373: fill=True → opacity=fill_alpha."""
        hull_x = np.array([0.0, 1.0, 0.5, 0.0])
        hull_y = np.array([0.0, 0.0, 1.0, 0.0])
        trace = render_convex_hull_plotly(hull_x, hull_y, fill=True, alpha=1.0, fill_alpha=0.2)
        assert trace is not None
        assert abs(trace.opacity - 0.2) < 1e-9

    def test_hull_plotly_opacity_no_fill(self):
        """Line 2373: fill=False → opacity=alpha."""
        hull_x = np.array([0.0, 1.0, 0.5, 0.0])
        hull_y = np.array([0.0, 0.0, 1.0, 0.0])
        trace = render_convex_hull_plotly(hull_x, hull_y, fill=False, alpha=0.7)
        assert trace is not None
        assert abs(trace.opacity - 0.7) < 1e-9

    def test_hull_plotly_with_label(self):
        """name=label."""
        hull_x = np.array([0.0, 1.0])
        hull_y = np.array([0.0, 1.0])
        trace = render_convex_hull_plotly(hull_x, hull_y, label="hull1")
        assert trace.name == "hull1"


# ===========================================================================
# render_boolean_states_matplotlib – uncovered branches
# ===========================================================================

class TestBooleanStatesMatplotlibBranches:
    def teardown_method(self):
        plt.close("all")

    def test_boolean_all_true(self):
        """All-True: no false regions."""
        ax = _ax2d()
        x = np.linspace(0, 1, 10)
        states = np.ones(10, dtype=bool)
        artists = render_boolean_states_matplotlib(ax, x, states)
        assert isinstance(artists, list)

    def test_boolean_all_false(self):
        """All-False: no true regions."""
        ax = _ax2d()
        x = np.linspace(0, 1, 10)
        states = np.zeros(10, dtype=bool)
        artists = render_boolean_states_matplotlib(ax, x, states)
        assert isinstance(artists, list)

    def test_boolean_starts_with_false(self):
        """Lines 2445-2446: states[0] is False."""
        ax = _ax2d()
        x = np.linspace(0, 1, 10)
        states = np.array([False] * 5 + [True] * 5)
        artists = render_boolean_states_matplotlib(ax, x, states)
        assert isinstance(artists, list)

    def test_boolean_ends_with_false(self):
        """Lines 2447-2448: states[-1] is False."""
        ax = _ax2d()
        x = np.linspace(0, 1, 10)
        states = np.array([True] * 5 + [False] * 5)
        artists = render_boolean_states_matplotlib(ax, x, states)
        assert isinstance(artists, list)

    def test_boolean_mixed(self):
        """Mixed True/False regions."""
        ax = _ax2d()
        x = np.linspace(0, 1, 20)
        states = np.array([True] * 5 + [False] * 5 + [True] * 5 + [False] * 5)
        artists = render_boolean_states_matplotlib(ax, x, states)
        assert isinstance(artists, list)
        assert len(artists) > 0


# ===========================================================================
# render_boolean_states_plotly – uncovered branches
# ===========================================================================

class TestBooleanStatesPlotlyBranches:
    def teardown_method(self):
        plt.close("all")

    def test_boolean_plotly_all_true_raises_index_error(self):
        """All-True: known bug in render_boolean_states_plotly where false_starts=[n]
        causes an out-of-bounds index when computing x_end. We document this here."""
        x = np.linspace(0, 1, 10)
        states = np.ones(10, dtype=bool)
        # The function has a bug: it computes false false-region boundaries,
        # resulting in an out-of-bounds access. We verify this known bug exists.
        with pytest.raises(IndexError):
            render_boolean_states_plotly(x, states)

    def test_boolean_plotly_all_false(self):
        """All-False: no true region traces, only false region traces."""
        x = np.linspace(0, 1, 10)
        states = np.zeros(10, dtype=bool)
        traces = render_boolean_states_plotly(x, states)
        assert isinstance(traces, list)

    def test_boolean_plotly_starts_with_false(self):
        """Lines 2539-2540: states[0] is False."""
        x = np.linspace(0, 1, 10)
        states = np.array([False] * 5 + [True] * 5)
        traces = render_boolean_states_plotly(x, states)
        assert isinstance(traces, list)

    def test_boolean_plotly_ends_with_false(self):
        """Lines 2541-2542: states[-1] is False."""
        x = np.linspace(0, 1, 10)
        states = np.array([True] * 5 + [False] * 5)
        traces = render_boolean_states_plotly(x, states)
        assert isinstance(traces, list)

    def test_boolean_plotly_end_clamp(self):
        """Line 2551: end >= len(x) → x[-1]."""
        x = np.linspace(0, 1, 10)
        states = np.array([True] * 5 + [False] * 5)
        # When states ends with False the last false_ends hits len(states)
        traces = render_boolean_states_plotly(x, states)
        assert isinstance(traces, list)

    def test_boolean_plotly_mixed(self):
        """Multiple regions: first trace uses label, rest use empty."""
        x = np.linspace(0, 1, 20)
        states = np.array([True] * 5 + [False] * 5 + [True] * 5 + [False] * 5)
        traces = render_boolean_states_plotly(x, states)
        assert isinstance(traces, list)
        assert len(traces) > 0


# ===========================================================================
# render_ellipse_matplotlib – uncovered branches
# ===========================================================================

class TestEllipseMatplotlibBranches:
    def teardown_method(self):
        plt.close("all")

    def test_ellipse_3d(self):
        """Lines 2667-2687: n_dims=3 → ellipsoids via plot_surface."""
        ax = _ax3d()
        centers = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        widths = np.array([0.4, 0.6])
        heights = np.array([0.3, 0.5])
        patches = render_ellipse_matplotlib(ax, centers, widths, heights)
        assert len(patches) == 2

    def test_ellipse_2d_with_angles(self):
        """Line 2650: angles is not None → angle = angles[i]."""
        ax = _ax2d()
        centers = np.array([[0.0, 0.0], [1.0, 1.0]])
        widths = np.array([0.5, 0.8])
        heights = np.array([0.3, 0.6])
        angles = np.array([30.0, 60.0])
        patches = render_ellipse_matplotlib(ax, centers, widths, heights, angles=angles)
        assert len(patches) == 2
        from matplotlib.patches import Ellipse
        assert all(isinstance(p, Ellipse) for p in patches)

    def test_ellipse_2d_no_angles(self):
        """Line 2650: angles=None → angle defaults to 0."""
        ax = _ax2d()
        centers = np.array([[0.5, 0.5]])
        widths = np.array([1.0])
        heights = np.array([0.5])
        patches = render_ellipse_matplotlib(ax, centers, widths, heights, angles=None)
        assert len(patches) == 1

    def test_ellipse_1d(self):
        """Lines 2622-2641: n_dims=1 → Rectangles."""
        ax = _ax2d()
        centers = np.array([[0.0], [1.0], [2.0]])
        widths = np.array([0.5, 0.5, 0.5])
        heights = np.array([1.0, 1.0, 1.0])
        patches = render_ellipse_matplotlib(ax, centers, widths, heights)
        assert len(patches) == 3
        from matplotlib.patches import Rectangle
        assert all(isinstance(p, Rectangle) for p in patches)

    def test_ellipse_4d_returns_empty(self):
        """Lines 2667->2689: n_dims=4 → elif n_dims==3 is False → returns empty."""
        ax = _ax3d()
        centers = np.array([[0.0, 0.0, 0.0, 0.0]])
        widths = np.array([0.5])
        heights = np.array([0.3])
        patches = render_ellipse_matplotlib(ax, centers, widths, heights)
        # n_dims=4: not 1, not 2, not 3 → falls through all branches, returns empty
        assert isinstance(patches, list)
        assert len(patches) == 0


# ===========================================================================
# render_ellipse_plotly – uncovered branches
# ===========================================================================

class TestEllipsePlotlyBranches:
    def teardown_method(self):
        plt.close("all")

    def test_ellipse_plotly_2d_no_angles(self):
        """Lines 2756-2788: 2D ellipses, angles=None → angle_deg=0."""
        centers = np.array([[0.0, 0.0], [1.0, 1.0]])
        widths = np.array([0.5, 0.8])
        heights = np.array([0.3, 0.6])
        shapes = render_ellipse_plotly(centers, widths, heights, angles=None, color="#ff0000")
        assert len(shapes) == 2
        assert all(s["type"] == "path" for s in shapes)

    def test_ellipse_plotly_2d_with_angles(self):
        """Line 2763: angles is not None → angle_deg = angles[i]."""
        centers = np.array([[0.0, 0.0]])
        widths = np.array([0.5])
        heights = np.array([0.3])
        angles = np.array([45.0])
        shapes = render_ellipse_plotly(centers, widths, heights, angles=angles, color="#00ff00")
        assert len(shapes) == 1
        # Path should contain 'M' and 'L'
        assert "M" in shapes[0]["path"]
        assert "L" in shapes[0]["path"]

    def test_ellipse_plotly_1d(self):
        """Lines 2737-2754: 1D → rectangles."""
        centers = np.array([[0.0], [1.0]])
        widths = np.array([0.4, 0.6])
        heights = np.array([1.0, 1.0])
        shapes = render_ellipse_plotly(centers, widths, heights, color="#0000ff")
        assert len(shapes) == 2
        assert all(s["type"] == "rect" for s in shapes)

    def test_ellipse_plotly_4d_returns_empty(self):
        """Lines 2756->2792: n_dims=4 → neither 1D rect nor 2D path → empty list."""
        centers = np.array([[0.0, 0.0, 0.0, 0.0]])
        widths = np.array([0.5])
        heights = np.array([0.3])
        shapes = render_ellipse_plotly(centers, widths, heights, color="#ff0000")
        assert isinstance(shapes, list)
        assert len(shapes) == 0
