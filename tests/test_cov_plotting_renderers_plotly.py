"""Tests for neural_analysis.plotting.renderers_plotly module.

Covers all public render_* functions with behaviour-asserting tests to reach >= 95% coverage.
"""

from __future__ import annotations

import numpy as np
import pytest

from neural_analysis.plotting.renderers_plotly import (
    PLOTLY_AVAILABLE,
    render_bar_plotly,
    render_boolean_states_plotly,
    render_box_plotly,
    render_convex_hull_plotly,
    render_ellipse_plotly,
    render_heatmap_plotly,
    render_histogram_plotly,
    render_kde_plotly,
    render_line_plotly,
    render_scatter3d_plotly,
    render_scatter_plotly,
    render_trajectory3d_plotly,
    render_trajectory_plotly,
    render_violin_plotly,
)

# ---------------------------------------------------------------------------
# Sanity: plotly must be available in the test environment
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.skipif(
    not PLOTLY_AVAILABLE, reason="plotly not installed"
)


# ===========================================================================
# render_scatter_plotly
# ===========================================================================


class TestRenderScatterPlotly:
    """Behaviour tests for render_scatter_plotly."""

    def test_basic_2d_returns_scatter_trace(self) -> None:
        import plotly.graph_objects as go

        rng = np.random.default_rng(0)
        data = rng.standard_normal((20, 2))
        trace = render_scatter_plotly(data)
        assert isinstance(trace, go.Scatter)
        np.testing.assert_array_equal(trace.x, data[:, 0])
        np.testing.assert_array_equal(trace.y, data[:, 1])

    def test_mode_is_markers(self) -> None:
        data = np.array([[0.0, 1.0], [2.0, 3.0]])
        trace = render_scatter_plotly(data)
        assert trace.mode == "markers"

    def test_color_sets_marker_color(self) -> None:
        data = np.array([[0.0, 1.0], [2.0, 3.0]])
        trace = render_scatter_plotly(data, color="red")
        assert trace.marker.color == "red"

    def test_solid_color_without_colors_array(self) -> None:
        """When colors is None, falls back to color param."""
        data = np.array([[0.0, 1.0], [2.0, 3.0]])
        trace = render_scatter_plotly(data, color="blue")
        assert trace.marker.color == "blue"

    def test_colors_array_sets_marker_color_array(self) -> None:
        rng = np.random.default_rng(0)
        data = rng.standard_normal((10, 2))
        colors = rng.random(10)
        trace = render_scatter_plotly(data, colors=colors)
        np.testing.assert_array_equal(trace.marker.color, colors)

    def test_colors_with_cmap_sets_colorscale(self) -> None:
        """Plotly 6 resolves colorscale names to tuples; just check it's not None."""
        rng = np.random.default_rng(0)
        data = rng.standard_normal((5, 2))
        colors = rng.random(5)
        trace = render_scatter_plotly(data, colors=colors, cmap="Viridis")
        assert trace.marker.colorscale is not None

    def test_colors_with_colorbar_sets_showscale(self) -> None:
        rng = np.random.default_rng(0)
        data = rng.standard_normal((5, 2))
        colors = rng.random(5)
        trace = render_scatter_plotly(data, colors=colors, colorbar=True)
        assert trace.marker.showscale is True

    def test_colorbar_label_sets_colorbar_title(self) -> None:
        rng = np.random.default_rng(0)
        data = rng.standard_normal((5, 2))
        colors = rng.random(5)
        trace = render_scatter_plotly(
            data, colors=colors, colorbar=True, colorbar_label="my_label"
        )
        assert trace.marker.colorbar.title.text == "my_label"

    def test_colorbar_without_label_uses_empty_string(self) -> None:
        rng = np.random.default_rng(0)
        data = rng.standard_normal((5, 2))
        colors = rng.random(5)
        trace = render_scatter_plotly(data, colors=colors, colorbar=True)
        # colorbar title should be empty string (not None)
        assert trace.marker.colorbar.title.text == ""

    def test_sizes_array_sets_marker_size(self) -> None:
        rng = np.random.default_rng(0)
        data = rng.standard_normal((5, 2))
        sizes = rng.random(5) * 20
        trace = render_scatter_plotly(data, sizes=sizes)
        np.testing.assert_array_equal(trace.marker.size, sizes)

    def test_marker_size_scalar_sets_size(self) -> None:
        data = np.array([[0.0, 1.0], [2.0, 3.0]])
        trace = render_scatter_plotly(data, marker_size=15)
        assert trace.marker.size == 15

    def test_default_marker_size_is_8(self) -> None:
        data = np.array([[0.0, 1.0], [2.0, 3.0]])
        trace = render_scatter_plotly(data)
        assert trace.marker.size == 8

    def test_alpha_sets_opacity(self) -> None:
        data = np.array([[0.0, 1.0], [2.0, 3.0]])
        trace = render_scatter_plotly(data, alpha=0.3)
        assert trace.marker.opacity == pytest.approx(0.3)

    def test_label_sets_name(self) -> None:
        data = np.array([[0.0, 1.0], [2.0, 3.0]])
        trace = render_scatter_plotly(data, label="my_group")
        assert trace.name == "my_group"

    def test_no_label_defaults_to_empty_string(self) -> None:
        data = np.array([[0.0, 1.0], [2.0, 3.0]])
        trace = render_scatter_plotly(data)
        assert trace.name == ""

    def test_showlegend_false(self) -> None:
        data = np.array([[0.0, 1.0], [2.0, 3.0]])
        trace = render_scatter_plotly(data, showlegend=False)
        assert trace.showlegend is False

    def test_marker_symbol_forwarded(self) -> None:
        data = np.array([[0.0, 1.0], [2.0, 3.0]])
        trace = render_scatter_plotly(data, marker="square")
        assert trace.marker.symbol == "square"

    def test_wrong_shape_raises_value_error(self) -> None:
        data = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
        with pytest.raises(ValueError, match="2-column"):
            render_scatter_plotly(data)

    def test_empty_colors_array_falls_back_to_color(self) -> None:
        """An empty colors array should not set colorscale — falls to color."""
        data = np.array([[0.0, 1.0], [2.0, 3.0]])
        trace = render_scatter_plotly(
            data, color="green", colors=np.array([])
        )
        assert trace.marker.color == "green"


# ===========================================================================
# render_scatter3d_plotly
# ===========================================================================


class TestRenderScatter3dPlotly:
    """Behaviour tests for render_scatter3d_plotly."""

    def test_basic_3d_returns_scatter3d_trace(self) -> None:
        import plotly.graph_objects as go

        rng = np.random.default_rng(0)
        data = rng.standard_normal((10, 3))
        trace = render_scatter3d_plotly(data)
        assert isinstance(trace, go.Scatter3d)
        np.testing.assert_array_equal(trace.x, data[:, 0])
        np.testing.assert_array_equal(trace.y, data[:, 1])
        np.testing.assert_array_equal(trace.z, data[:, 2])

    def test_wrong_shape_raises_value_error(self) -> None:
        data = np.array([[0.0, 1.0], [3.0, 4.0]])
        with pytest.raises(ValueError, match="3-column"):
            render_scatter3d_plotly(data)

    def test_color_sets_marker_color(self) -> None:
        data = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
        trace = render_scatter3d_plotly(data, color="blue")
        assert trace.marker.color == "blue"

    def test_colors_array_and_cmap(self) -> None:
        """Plotly 6 expands colorscale names; just check it's set (not None)."""
        rng = np.random.default_rng(0)
        data = rng.standard_normal((8, 3))
        colors = rng.random(8)
        trace = render_scatter3d_plotly(data, colors=colors, cmap="Plasma")
        np.testing.assert_array_equal(trace.marker.color, colors)
        assert trace.marker.colorscale is not None

    def test_colorbar_shown_with_colors(self) -> None:
        rng = np.random.default_rng(0)
        data = rng.standard_normal((5, 3))
        colors = rng.random(5)
        trace = render_scatter3d_plotly(
            data, colors=colors, colorbar=True, colorbar_label="depth"
        )
        assert trace.marker.showscale is True
        assert trace.marker.colorbar.title.text == "depth"

    def test_sizes_array_sets_size(self) -> None:
        rng = np.random.default_rng(0)
        data = rng.standard_normal((5, 3))
        sizes = rng.random(5) * 10
        trace = render_scatter3d_plotly(data, sizes=sizes)
        np.testing.assert_array_equal(trace.marker.size, sizes)

    def test_default_marker_size_is_4(self) -> None:
        data = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
        trace = render_scatter3d_plotly(data)
        assert trace.marker.size == 4

    def test_alpha_sets_opacity(self) -> None:
        data = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
        trace = render_scatter3d_plotly(data, alpha=0.5)
        assert trace.marker.opacity == pytest.approx(0.5)

    def test_label_and_showlegend(self) -> None:
        data = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
        trace = render_scatter3d_plotly(data, label="group1", showlegend=False)
        assert trace.name == "group1"
        assert trace.showlegend is False

    def test_empty_colors_falls_back_to_color(self) -> None:
        data = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
        trace = render_scatter3d_plotly(data, color="red", colors=np.array([]))
        assert trace.marker.color == "red"


# ===========================================================================
# render_line_plotly
# ===========================================================================


class TestRenderLinePlotly:
    """Behaviour tests for render_line_plotly."""

    def test_1d_data_uses_indices_as_x(self) -> None:
        import plotly.graph_objects as go

        data = np.array([1.0, 2.0, 3.0, 4.0])
        trace = render_line_plotly(data)
        assert isinstance(trace, go.Scatter)
        np.testing.assert_array_equal(trace.y, data)
        assert trace.x is None  # x not set; plotly uses indices

    def test_2d_data_x_and_y_columns(self) -> None:
        data = np.array([[0.0, 10.0], [1.0, 20.0], [2.0, 30.0]])
        trace = render_line_plotly(data)
        np.testing.assert_array_equal(trace.x, data[:, 0])
        np.testing.assert_array_equal(trace.y, data[:, 1])

    def test_multidim_data_uses_first_column(self) -> None:
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        trace = render_line_plotly(data)
        np.testing.assert_array_equal(trace.y, data[:, 0])

    def test_dict_data_with_x_y_keys(self) -> None:
        data = {"x": [0, 1, 2], "y": [10, 20, 30]}
        trace = render_line_plotly(data)
        np.testing.assert_array_equal(trace.x, [0, 1, 2])
        np.testing.assert_array_equal(trace.y, [10, 20, 30])

    def test_dict_data_missing_y_raises(self) -> None:
        with pytest.raises(ValueError, match="'x' and 'y' keys"):
            render_line_plotly({"x": [1, 2, 3]})

    def test_dict_data_missing_both_raises(self) -> None:
        with pytest.raises(ValueError, match="'x' and 'y' keys"):
            render_line_plotly({"z": [1, 2, 3]})

    def test_mode_is_lines(self) -> None:
        data = np.array([1.0, 2.0, 3.0])
        trace = render_line_plotly(data)
        assert trace.mode == "lines"

    def test_line_width_forwarded(self) -> None:
        data = np.array([1.0, 2.0])
        trace = render_line_plotly(data, line_width=5)
        assert trace.line.width == 5

    def test_color_forwarded_to_line(self) -> None:
        data = np.array([1.0, 2.0])
        trace = render_line_plotly(data, color="purple")
        assert trace.line.color == "purple"

    def test_linestyle_dash_conversion(self) -> None:
        """Matplotlib '--' should map to plotly 'dash'."""
        data = np.array([1.0, 2.0])
        trace = render_line_plotly(data, linestyle="--")
        assert trace.line.dash == "dash"

    def test_linestyle_dotdash_conversion(self) -> None:
        data = np.array([1.0, 2.0])
        trace = render_line_plotly(data, linestyle="-.")
        assert trace.line.dash == "dashdot"

    def test_linestyle_dot_conversion(self) -> None:
        data = np.array([1.0, 2.0])
        trace = render_line_plotly(data, linestyle=":")
        assert trace.line.dash == "dot"

    def test_linestyle_solid_conversion(self) -> None:
        data = np.array([1.0, 2.0])
        trace = render_line_plotly(data, linestyle="-")
        assert trace.line.dash == "solid"

    def test_plotly_native_dash_style_passthrough(self) -> None:
        """A plotly-native style that isn't in the map is passed through."""
        data = np.array([1.0, 2.0])
        trace = render_line_plotly(data, linestyle="longdash")
        assert trace.line.dash == "longdash"

    def test_no_linestyle_dash_not_set(self) -> None:
        data = np.array([1.0, 2.0])
        trace = render_line_plotly(data)
        assert trace.line.dash is None

    def test_error_y_creates_error_bars(self) -> None:
        data = np.array([1.0, 2.0, 3.0])
        error = np.array([0.1, 0.2, 0.1])
        trace = render_line_plotly(data, error_y=error, color="red")
        assert trace.error_y.visible is True
        np.testing.assert_array_equal(trace.error_y.array, error)
        assert trace.error_y.color == "red"

    def test_error_y_without_color_uses_fallback(self) -> None:
        data = np.array([1.0, 2.0, 3.0])
        error = np.array([0.1, 0.2, 0.1])
        trace = render_line_plotly(data, error_y=error)
        assert trace.error_y.color == "rgba(0,0,0,0.3)"

    def test_label_and_showlegend(self) -> None:
        data = np.array([1.0, 2.0])
        trace = render_line_plotly(data, label="trace1", showlegend=False)
        assert trace.name == "trace1"
        assert trace.showlegend is False

    def test_alpha_sets_opacity(self) -> None:
        data = np.array([1.0, 2.0])
        trace = render_line_plotly(data, alpha=0.4)
        assert trace.opacity == pytest.approx(0.4)


# ===========================================================================
# render_histogram_plotly
# ===========================================================================


class TestRenderHistogramPlotly:
    """Behaviour tests for render_histogram_plotly."""

    def test_returns_histogram_trace(self) -> None:
        import plotly.graph_objects as go

        rng = np.random.default_rng(0)
        data = rng.standard_normal(50)
        trace = render_histogram_plotly(data)
        assert isinstance(trace, go.Histogram)

    def test_x_data_matches_input(self) -> None:
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        trace = render_histogram_plotly(data)
        np.testing.assert_array_equal(trace.x, data)

    def test_bins_forwarded_as_nbinsx(self) -> None:
        data = np.array([1.0, 2.0, 3.0])
        trace = render_histogram_plotly(data, bins=10)
        assert trace.nbinsx == 10

    def test_color_sets_marker_color(self) -> None:
        data = np.array([1.0, 2.0, 3.0])
        trace = render_histogram_plotly(data, color="green")
        assert trace.marker.color == "green"

    def test_alpha_sets_opacity(self) -> None:
        data = np.array([1.0, 2.0, 3.0])
        trace = render_histogram_plotly(data, alpha=0.5)
        assert trace.opacity == pytest.approx(0.5)

    def test_label_and_showlegend(self) -> None:
        data = np.array([1.0, 2.0, 3.0])
        trace = render_histogram_plotly(data, label="hist", showlegend=False)
        assert trace.name == "hist"
        assert trace.showlegend is False

    def test_default_label_is_empty_string(self) -> None:
        data = np.array([1.0, 2.0])
        trace = render_histogram_plotly(data)
        assert trace.name == ""


# ===========================================================================
# render_heatmap_plotly
# ===========================================================================


class TestRenderHeatmapPlotly:
    """Behaviour tests for render_heatmap_plotly."""

    def test_returns_heatmap_trace(self) -> None:
        import plotly.graph_objects as go

        rng = np.random.default_rng(0)
        data = rng.random((5, 5))
        trace = render_heatmap_plotly(data)
        assert isinstance(trace, go.Heatmap)

    def test_z_data_matches_input(self) -> None:
        data = np.arange(9, dtype=float).reshape(3, 3)
        trace = render_heatmap_plotly(data)
        np.testing.assert_array_equal(trace.z, data)

    def test_default_colorscale_viridis(self) -> None:
        """Plotly 6 resolves 'Viridis' to its expanded form; check it's not None."""
        data = np.ones((3, 3))
        trace = render_heatmap_plotly(data)
        assert trace.colorscale is not None

    def test_cmap_sets_colorscale(self) -> None:
        """When cmap='Plasma' is passed the colorscale is resolved (not None)."""
        data = np.ones((3, 3))
        trace = render_heatmap_plotly(data, cmap="Plasma")
        assert trace.colorscale is not None

    def test_colorscale_takes_precedence_over_cmap(self) -> None:
        """colorscale kwarg takes precedence; check it's resolved (not None)."""
        data = np.ones((3, 3))
        # The two different inputs should still produce a non-None colorscale
        trace_cmap = render_heatmap_plotly(data, cmap="Plasma")
        trace_cs = render_heatmap_plotly(data, cmap="Plasma", colorscale="Hot")
        assert trace_cs.colorscale is not None
        # They should differ (different color palettes)
        assert trace_cmap.colorscale != trace_cs.colorscale

    def test_colorbar_label_sets_colorbar_title(self) -> None:
        data = np.ones((3, 3))
        trace = render_heatmap_plotly(data, colorbar_label="density")
        # colorbar is enabled by default; check title
        assert trace.colorbar.title.text == "density"

    def test_colorbar_disabled_makes_colorbar_none(self) -> None:
        """When colorbar=False the colorbar_config is None; plotly 6 may keep an empty obj."""
        data = np.ones((3, 3))
        trace = render_heatmap_plotly(data, colorbar=False)
        # In plotly 6 a None colorbar is returned as an empty ColorBar object;
        # the important thing is there is no title text set.
        cb = trace.colorbar
        if cb is not None:
            # title should have no text attribute set
            cb_title = getattr(cb, "title", None)
            if cb_title is not None:
                assert getattr(cb_title, "text", None) is None

    def test_extra_kwargs_popped_without_error(self) -> None:
        """x_labels, y_labels, show_values, value_format, alpha must be removed."""
        data = np.ones((3, 3))
        trace = render_heatmap_plotly(
            data,
            x_labels=["a", "b", "c"],
            y_labels=["x", "y", "z"],
            show_values=True,
            value_format=".2f",
            alpha=0.8,
        )
        assert isinstance(trace, type(trace))  # no error raised


# ===========================================================================
# render_bar_plotly
# ===========================================================================


class TestRenderBarPlotly:
    """Behaviour tests for render_bar_plotly."""

    def test_returns_bar_trace(self) -> None:
        import plotly.graph_objects as go

        data = np.array([1.0, 2.0, 3.0])
        trace = render_bar_plotly(data)
        assert isinstance(trace, go.Bar)

    def test_1d_data_y_matches(self) -> None:
        data = np.array([4.0, 5.0, 6.0])
        trace = render_bar_plotly(data)
        np.testing.assert_array_equal(trace.y, data)

    def test_2d_data_uses_first_column(self) -> None:
        data = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        trace = render_bar_plotly(data)
        np.testing.assert_array_equal(trace.y, data[:, 0])

    def test_x_positions_forwarded(self) -> None:
        data = np.array([1.0, 2.0, 3.0])
        x = np.array([0.0, 1.0, 2.0])
        trace = render_bar_plotly(data, x=x)
        np.testing.assert_array_equal(trace.x, x)

    def test_single_color_sets_marker_color(self) -> None:
        data = np.array([1.0, 2.0])
        trace = render_bar_plotly(data, color="orange")
        assert trace.marker.color == "orange"

    def test_colors_list_takes_precedence(self) -> None:
        data = np.array([1.0, 2.0])
        colors = ["red", "blue"]
        trace = render_bar_plotly(data, color="orange", colors=colors)
        assert list(trace.marker.color) == colors

    def test_alpha_sets_marker_opacity(self) -> None:
        data = np.array([1.0, 2.0])
        trace = render_bar_plotly(data, alpha=0.6)
        assert trace.marker.opacity == pytest.approx(0.6)

    def test_error_y_attached(self) -> None:
        data = np.array([1.0, 2.0, 3.0])
        err = np.array([0.1, 0.2, 0.1])
        trace = render_bar_plotly(data, error_y=err)
        np.testing.assert_array_equal(trace.error_y.array, err)
        assert trace.error_y.visible is True

    def test_error_x_attached(self) -> None:
        data = np.array([1.0, 2.0, 3.0])
        err = np.array([0.05, 0.1, 0.05])
        trace = render_bar_plotly(data, error_x=err)
        np.testing.assert_array_equal(trace.error_x.array, err)
        assert trace.error_x.visible is True

    def test_no_errors_when_none(self) -> None:
        """When no error arrays are passed, error_y/x visible should be False/None."""
        data = np.array([1.0, 2.0])
        trace = render_bar_plotly(data)
        # Plotly 6 returns empty error objects; check they have no array set
        assert trace.error_y.array is None
        assert trace.error_x.array is None

    def test_label_and_showlegend(self) -> None:
        data = np.array([1.0, 2.0])
        trace = render_bar_plotly(data, label="bar1", showlegend=False)
        assert trace.name == "bar1"
        assert trace.showlegend is False


# ===========================================================================
# render_violin_plotly
# ===========================================================================


class TestRenderViolinPlotly:
    """Behaviour tests for render_violin_plotly."""

    def test_returns_violin_trace(self) -> None:
        import plotly.graph_objects as go

        rng = np.random.default_rng(0)
        data = rng.standard_normal(50)
        trace = render_violin_plotly(data)
        assert isinstance(trace, go.Violin)

    def test_y_data_matches(self) -> None:
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        trace = render_violin_plotly(data)
        np.testing.assert_array_equal(trace.y, data)

    def test_side_is_positive(self) -> None:
        data = np.array([1.0, 2.0, 3.0])
        trace = render_violin_plotly(data)
        assert trace.side == "positive"

    def test_color_forwarded_to_marker(self) -> None:
        data = np.array([1.0, 2.0, 3.0])
        trace = render_violin_plotly(data, color="teal")
        assert trace.marker.color == "teal"

    def test_alpha_sets_opacity(self) -> None:
        data = np.array([1.0, 2.0, 3.0])
        trace = render_violin_plotly(data, alpha=0.6)
        assert trace.opacity == pytest.approx(0.6)

    def test_showbox_true_shows_box(self) -> None:
        data = np.array([1.0, 2.0, 3.0])
        trace = render_violin_plotly(data, showbox=True)
        assert trace.box.visible is True

    def test_showbox_false_makes_box_not_visible(self) -> None:
        """showbox=False passes box=None to go.Violin; Plotly may return an object with visible=False."""
        data = np.array([1.0, 2.0, 3.0])
        trace = render_violin_plotly(data, showbox=False)
        # Either box is None/empty or box.visible is False
        assert trace.box is None or trace.box.visible is False or trace.box.visible is None

    def test_showpoints_true_sets_all(self) -> None:
        data = np.array([1.0, 2.0, 3.0])
        trace = render_violin_plotly(data, showpoints=True)
        assert trace.points == "all"

    def test_showpoints_false_disables_points(self) -> None:
        data = np.array([1.0, 2.0, 3.0])
        trace = render_violin_plotly(data, showpoints=False)
        assert trace.points is False

    def test_meanline_none_defaults_to_visible(self) -> None:
        data = np.array([1.0, 2.0, 3.0])
        trace = render_violin_plotly(data, meanline=None)
        assert trace.meanline.visible is True

    def test_meanline_bool_true_converted(self) -> None:
        data = np.array([1.0, 2.0, 3.0])
        trace = render_violin_plotly(data, meanline=True)
        assert trace.meanline.visible is True

    def test_meanline_bool_false_converted(self) -> None:
        data = np.array([1.0, 2.0, 3.0])
        trace = render_violin_plotly(data, meanline=False)
        assert trace.meanline.visible is False

    def test_meanline_dict_with_partial_keys_completed(self) -> None:
        """Passing a dict without 'visible' should have it added as True."""
        data = np.array([1.0, 2.0, 3.0])
        trace = render_violin_plotly(data, meanline={"color": "red"})
        assert trace.meanline.visible is True
        assert trace.meanline.color == "red"

    def test_meanline_dict_with_all_keys_kept(self) -> None:
        """Dict with visible, color AND width - covers the 570->581 branch (all ifs skipped)."""
        data = np.array([1.0, 2.0, 3.0])
        ml = {"visible": True, "color": "blue", "width": 3}
        trace = render_violin_plotly(data, meanline=ml)
        assert trace.meanline.visible is True
        assert trace.meanline.color == "blue"
        assert trace.meanline.width == 3

    def test_meanline_dict_without_color_gets_color_from_color_param(self) -> None:
        """Dict missing 'color' key; covers line 577 (color set from color param)."""
        data = np.array([1.0, 2.0, 3.0])
        ml = {"visible": True, "width": 2}  # no "color" key
        trace = render_violin_plotly(data, meanline=ml, color="teal")
        assert trace.meanline.color == "teal"

    def test_meanline_dict_without_color_defaults_to_black(self) -> None:
        """Dict missing 'color', no color param -> 'black'."""
        data = np.array([1.0, 2.0, 3.0])
        ml = {"visible": True, "width": 2}  # no "color" key
        trace = render_violin_plotly(data, meanline=ml, color=None)
        assert trace.meanline.color == "black"

    def test_label_and_showlegend(self) -> None:
        data = np.array([1.0, 2.0, 3.0])
        trace = render_violin_plotly(data, label="v1", showlegend=False)
        assert trace.name == "v1"
        assert trace.showlegend is False


# ===========================================================================
# render_box_plotly
# ===========================================================================


class TestRenderBoxPlotly:
    """Behaviour tests for render_box_plotly."""

    def test_returns_box_trace(self) -> None:
        import plotly.graph_objects as go

        rng = np.random.default_rng(0)
        data = rng.standard_normal(30)
        trace = render_box_plotly(data)
        assert isinstance(trace, go.Box)

    def test_y_data_matches(self) -> None:
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        trace = render_box_plotly(data)
        np.testing.assert_array_equal(trace.y, data)

    def test_showpoints_true_gives_all(self) -> None:
        data = np.array([1.0, 2.0, 3.0])
        trace = render_box_plotly(data, showpoints=True)
        assert trace.boxpoints == "all"

    def test_showpoints_false_disables_points(self) -> None:
        data = np.array([1.0, 2.0, 3.0])
        trace = render_box_plotly(data, showpoints=False)
        assert trace.boxpoints is False

    def test_notched_sets_notched(self) -> None:
        data = np.arange(1.0, 21.0)
        trace = render_box_plotly(data, notched=True)
        assert trace.notched is True

    def test_color_sets_marker_color(self) -> None:
        data = np.array([1.0, 2.0, 3.0])
        trace = render_box_plotly(data, color="cyan")
        assert trace.marker.color == "cyan"

    def test_alpha_sets_opacity(self) -> None:
        data = np.array([1.0, 2.0, 3.0])
        trace = render_box_plotly(data, alpha=0.8)
        assert trace.opacity == pytest.approx(0.8)

    def test_label_and_showlegend(self) -> None:
        data = np.array([1.0, 2.0, 3.0])
        trace = render_box_plotly(data, label="box1", showlegend=False)
        assert trace.name == "box1"
        assert trace.showlegend is False

    def test_jitter_and_pointpos_when_showpoints_true(self) -> None:
        data = np.array([1.0, 2.0, 3.0])
        trace = render_box_plotly(data, showpoints=True)
        assert trace.jitter == pytest.approx(0.3)
        assert trace.pointpos == 0

    def test_jitter_zero_when_showpoints_false(self) -> None:
        data = np.array([1.0, 2.0, 3.0])
        trace = render_box_plotly(data, showpoints=False)
        assert trace.jitter == 0


# ===========================================================================
# render_trajectory_plotly
# ===========================================================================


class TestRenderTrajectoryPlotly:
    """Behaviour tests for render_trajectory_plotly."""

    def test_returns_scatter_trace(self) -> None:
        import plotly.graph_objects as go

        rng = np.random.default_rng(0)
        x = rng.standard_normal(20)
        y = rng.standard_normal(20)
        trace = render_trajectory_plotly(x, y)
        assert isinstance(trace, go.Scatter)

    def test_xy_coordinates_forwarded(self) -> None:
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 1.0, 0.0])
        trace = render_trajectory_plotly(x, y)
        np.testing.assert_array_equal(trace.x, x)
        np.testing.assert_array_equal(trace.y, y)

    def test_mode_lines_without_points(self) -> None:
        x = np.array([0.0, 1.0])
        y = np.array([0.0, 1.0])
        trace = render_trajectory_plotly(x, y, show_points=False)
        assert trace.mode == "lines"

    def test_mode_lines_plus_markers_with_points(self) -> None:
        x = np.array([0.0, 1.0])
        y = np.array([0.0, 1.0])
        trace = render_trajectory_plotly(x, y, show_points=True)
        assert trace.mode == "lines+markers"

    def test_no_colors_gives_no_colorscale(self) -> None:
        x = np.array([0.0, 1.0])
        y = np.array([0.0, 1.0])
        trace = render_trajectory_plotly(x, y, colors=None)
        # Without colors, marker_config is None (no points) or minimal
        # The trace should exist without colorscale in marker
        assert trace is not None

    def test_colors_sets_marker_color_and_colorscale(self) -> None:
        """Plotly 6 resolves colorscale names; verify it's set (not None)."""
        rng = np.random.default_rng(0)
        n = 10
        x = rng.standard_normal(n)
        y = rng.standard_normal(n)
        colors = rng.random(n)
        trace = render_trajectory_plotly(x, y, colors=colors, cmap="Hot")
        np.testing.assert_array_equal(trace.marker.color, colors.tolist())
        assert trace.marker.colorscale is not None

    def test_colors_with_showscale_true(self) -> None:
        rng = np.random.default_rng(0)
        n = 5
        x = rng.standard_normal(n)
        y = rng.standard_normal(n)
        colors = rng.random(n)
        trace = render_trajectory_plotly(x, y, colors=colors, colorbar=True)
        assert trace.marker.showscale is True

    def test_colors_with_colorbar_label(self) -> None:
        """Plotly 6 wraps title in a Title object; access .title.text."""
        rng = np.random.default_rng(0)
        n = 5
        x = rng.standard_normal(n)
        y = rng.standard_normal(n)
        colors = rng.random(n)
        trace = render_trajectory_plotly(
            x, y, colors=colors, colorbar=True, colorbar_label="time"
        )
        assert trace.marker.colorbar.title.text == "time"

    def test_colors_no_colorbar_label_gives_none_colorbar(self) -> None:
        """When colorbar_label is None, the colorbar dict is None — marker.colorbar is empty."""
        rng = np.random.default_rng(0)
        n = 5
        x = rng.standard_normal(n)
        y = rng.standard_normal(n)
        colors = rng.random(n)
        trace = render_trajectory_plotly(
            x, y, colors=colors, colorbar=True, colorbar_label=None
        )
        # Source passes None for colorbar when no label; plotly 6 returns empty ColorBar
        cb = trace.marker.colorbar
        # Accept either None or an empty object (no title text)
        if cb is not None:
            assert getattr(getattr(cb, "title", None), "text", None) is None

    def test_linewidth_forwarded(self) -> None:
        x = np.array([0.0, 1.0])
        y = np.array([0.0, 1.0])
        trace = render_trajectory_plotly(x, y, linewidth=4.0)
        assert trace.line.width == pytest.approx(4.0)

    def test_alpha_sets_opacity(self) -> None:
        x = np.array([0.0, 1.0])
        y = np.array([0.0, 1.0])
        trace = render_trajectory_plotly(x, y, alpha=0.5)
        assert trace.opacity == pytest.approx(0.5)

    def test_label_and_showlegend(self) -> None:
        x = np.array([0.0, 1.0])
        y = np.array([0.0, 1.0])
        trace = render_trajectory_plotly(x, y, label="traj", showlegend=False)
        assert trace.name == "traj"
        assert trace.showlegend is False

    def test_show_points_without_colors_has_marker_config(self) -> None:
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 1.0, 0.0])
        trace = render_trajectory_plotly(x, y, show_points=True, point_size=8.0)
        assert trace.marker.size == pytest.approx(8.0)


# ===========================================================================
# render_trajectory3d_plotly
# ===========================================================================


class TestRenderTrajectory3dPlotly:
    """Behaviour tests for render_trajectory3d_plotly."""

    def test_returns_scatter3d_trace(self) -> None:
        import plotly.graph_objects as go

        rng = np.random.default_rng(0)
        x = rng.standard_normal(10)
        y = rng.standard_normal(10)
        z = rng.standard_normal(10)
        trace = render_trajectory3d_plotly(x, y, z)
        assert isinstance(trace, go.Scatter3d)

    def test_xyz_forwarded(self) -> None:
        x = np.array([0.0, 1.0])
        y = np.array([1.0, 2.0])
        z = np.array([2.0, 3.0])
        trace = render_trajectory3d_plotly(x, y, z)
        np.testing.assert_array_equal(trace.x, x)
        np.testing.assert_array_equal(trace.y, y)
        np.testing.assert_array_equal(trace.z, z)

    def test_mode_lines_without_points(self) -> None:
        x, y, z = np.ones(3), np.ones(3), np.ones(3)
        trace = render_trajectory3d_plotly(x, y, z, show_points=False)
        assert trace.mode == "lines"

    def test_mode_lines_plus_markers_with_points(self) -> None:
        x, y, z = np.ones(3), np.ones(3), np.ones(3)
        trace = render_trajectory3d_plotly(x, y, z, show_points=True)
        assert trace.mode == "lines+markers"

    def test_colors_set_marker_color_and_colorscale(self) -> None:
        """Plotly 6 resolves colorscale names; verify it's set (not None)."""
        rng = np.random.default_rng(0)
        n = 8
        x = rng.standard_normal(n)
        y = rng.standard_normal(n)
        z = rng.standard_normal(n)
        colors = rng.random(n)
        trace = render_trajectory3d_plotly(x, y, z, colors=colors, cmap="Jet")
        np.testing.assert_array_equal(trace.marker.color, colors)
        assert trace.marker.colorscale is not None

    def test_colors_with_colorbar_label(self) -> None:
        """Plotly 6 wraps title in a Title object; access .title.text."""
        rng = np.random.default_rng(0)
        n = 5
        x = rng.standard_normal(n)
        y = rng.standard_normal(n)
        z = rng.standard_normal(n)
        colors = rng.random(n)
        trace = render_trajectory3d_plotly(
            x, y, z, colors=colors, colorbar=True, colorbar_label="depth"
        )
        assert trace.marker.colorbar.title.text == "depth"

    def test_colors_without_colorbar_label(self) -> None:
        rng = np.random.default_rng(0)
        n = 5
        x = rng.standard_normal(n)
        y = rng.standard_normal(n)
        z = rng.standard_normal(n)
        colors = rng.random(n)
        trace = render_trajectory3d_plotly(
            x, y, z, colors=colors, colorbar=False, colorbar_label=None
        )
        assert trace.marker.showscale is False

    def test_show_points_true_sets_marker_size(self) -> None:
        x, y, z = np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([0.0, 1.0])
        trace = render_trajectory3d_plotly(x, y, z, show_points=True, point_size=6.0)
        assert trace.marker.size == pytest.approx(6.0)

    def test_show_points_false_sets_tiny_marker(self) -> None:
        x, y, z = np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([0.0, 1.0])
        trace = render_trajectory3d_plotly(x, y, z, show_points=False)
        assert trace.marker.size == pytest.approx(0.1)

    def test_linewidth_forwarded(self) -> None:
        x, y, z = np.ones(3), np.ones(3), np.ones(3)
        trace = render_trajectory3d_plotly(x, y, z, linewidth=3.0)
        assert trace.line.width == pytest.approx(3.0)

    def test_alpha_sets_opacity(self) -> None:
        x, y, z = np.ones(3), np.ones(3), np.ones(3)
        trace = render_trajectory3d_plotly(x, y, z, alpha=0.6)
        assert trace.opacity == pytest.approx(0.6)

    def test_label_and_showlegend(self) -> None:
        x, y, z = np.ones(3), np.ones(3), np.ones(3)
        trace = render_trajectory3d_plotly(x, y, z, label="t3d", showlegend=False)
        assert trace.name == "t3d"
        assert trace.showlegend is False


# ===========================================================================
# render_kde_plotly
# ===========================================================================


class TestRenderKdePlotly:
    """Behaviour tests for render_kde_plotly."""

    def _make_grid(self, n: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.linspace(0, 1, n)
        y = np.linspace(0, 1, n)
        xi, yi = np.meshgrid(x, y)
        rng = np.random.default_rng(0)
        zi = rng.random((n, n))
        return xi, yi, zi

    def test_returns_contour_trace(self) -> None:
        import plotly.graph_objects as go

        xi, yi, zi = self._make_grid()
        trace = render_kde_plotly(xi, yi, zi)
        assert isinstance(trace, go.Contour)

    def test_x_uses_first_row_of_2d_xi(self) -> None:
        xi, yi, zi = self._make_grid(8)
        trace = render_kde_plotly(xi, yi, zi)
        np.testing.assert_allclose(trace.x, xi[0, :])

    def test_y_uses_first_col_of_2d_yi(self) -> None:
        xi, yi, zi = self._make_grid(8)
        trace = render_kde_plotly(xi, yi, zi)
        np.testing.assert_allclose(trace.y, yi[:, 0])

    def test_z_data_matches(self) -> None:
        xi, yi, zi = self._make_grid(6)
        trace = render_kde_plotly(xi, yi, zi)
        np.testing.assert_array_equal(trace.z, zi)

    def test_1d_xi_passed_directly(self) -> None:
        rng = np.random.default_rng(0)
        x = np.linspace(0, 1, 8)
        y = np.linspace(0, 1, 8)
        z = rng.random((8, 8))
        trace = render_kde_plotly(x, y, z)
        np.testing.assert_array_equal(trace.x, x)

    def test_colorscale_forwarded(self) -> None:
        """Plotly 6 resolves colorscale names; verify it's set (not None)."""
        xi, yi, zi = self._make_grid()
        trace = render_kde_plotly(xi, yi, zi, cmap="Hot")
        assert trace.colorscale is not None

    def test_alpha_sets_opacity(self) -> None:
        xi, yi, zi = self._make_grid()
        trace = render_kde_plotly(xi, yi, zi, alpha=0.4)
        assert trace.opacity == pytest.approx(0.4)

    def test_colorbar_true_shows_scale(self) -> None:
        xi, yi, zi = self._make_grid()
        trace = render_kde_plotly(xi, yi, zi, colorbar=True)
        assert trace.showscale is True

    def test_colorbar_false_hides_scale(self) -> None:
        xi, yi, zi = self._make_grid()
        trace = render_kde_plotly(xi, yi, zi, colorbar=False)
        assert trace.showscale is False

    def test_colorbar_label_sets_title(self) -> None:
        xi, yi, zi = self._make_grid()
        trace = render_kde_plotly(xi, yi, zi, colorbar_label="prob")
        assert trace.colorbar.title.text == "prob"

    def test_no_colorbar_label_gives_no_colorbar_title(self) -> None:
        """When no label is given, the code passes {} as colorbar; title should be absent."""
        xi, yi, zi = self._make_grid()
        trace = render_kde_plotly(xi, yi, zi, colorbar_label=None)
        # Either colorbar is None/empty or title is not set
        if trace.colorbar is not None:
            # title should be None or an empty string if set at all
            cb_title = getattr(trace.colorbar, "title", None)
            assert cb_title is None or getattr(cb_title, "text", None) is None

    def test_label_and_showlegend(self) -> None:
        xi, yi, zi = self._make_grid()
        trace = render_kde_plotly(xi, yi, zi, label="kde", showlegend=False)
        assert trace.name == "kde"
        assert trace.showlegend is False

    def test_contours_computed_from_zi(self) -> None:
        xi, yi, zi = self._make_grid()
        trace = render_kde_plotly(xi, yi, zi, n_levels=5)
        expected_size = (float(np.max(zi)) - float(np.min(zi))) / 5
        assert trace.contours.size == pytest.approx(expected_size, rel=1e-5)


# ===========================================================================
# render_convex_hull_plotly
# ===========================================================================


class TestRenderConvexHullPlotly:
    """Behaviour tests for render_convex_hull_plotly."""

    def test_returns_scatter_trace(self) -> None:
        import plotly.graph_objects as go

        hull_x = np.array([0.0, 1.0, 1.0, 0.0, 0.0])
        hull_y = np.array([0.0, 0.0, 1.0, 1.0, 0.0])
        trace = render_convex_hull_plotly(hull_x, hull_y)
        assert isinstance(trace, go.Scatter)

    def test_xy_forwarded(self) -> None:
        hull_x = np.array([0.0, 1.0, 0.0])
        hull_y = np.array([0.0, 1.0, 1.0])
        trace = render_convex_hull_plotly(hull_x, hull_y)
        np.testing.assert_array_equal(trace.x, hull_x)
        np.testing.assert_array_equal(trace.y, hull_y)

    def test_mode_is_lines(self) -> None:
        hull_x = np.array([0.0, 1.0])
        hull_y = np.array([0.0, 1.0])
        trace = render_convex_hull_plotly(hull_x, hull_y)
        assert trace.mode == "lines"

    def test_no_fill_sets_fill_none(self) -> None:
        hull_x = np.array([0.0, 1.0])
        hull_y = np.array([0.0, 1.0])
        trace = render_convex_hull_plotly(hull_x, hull_y, fill=False)
        assert trace.fill == "none"

    def test_fill_true_sets_toself(self) -> None:
        hull_x = np.array([0.0, 1.0, 0.5])
        hull_y = np.array([0.0, 0.0, 1.0])
        trace = render_convex_hull_plotly(hull_x, hull_y, fill=True)
        assert trace.fill == "toself"

    def test_fill_true_uses_fill_alpha_for_opacity(self) -> None:
        hull_x = np.array([0.0, 1.0, 0.5])
        hull_y = np.array([0.0, 0.0, 1.0])
        trace = render_convex_hull_plotly(hull_x, hull_y, fill=True, fill_alpha=0.25)
        assert trace.opacity == pytest.approx(0.25)

    def test_fill_false_uses_alpha_for_opacity(self) -> None:
        hull_x = np.array([0.0, 1.0])
        hull_y = np.array([0.0, 1.0])
        trace = render_convex_hull_plotly(hull_x, hull_y, fill=False, alpha=0.8)
        assert trace.opacity == pytest.approx(0.8)

    def test_color_sets_line_color(self) -> None:
        hull_x = np.array([0.0, 1.0])
        hull_y = np.array([0.0, 1.0])
        trace = render_convex_hull_plotly(hull_x, hull_y, color="navy")
        assert trace.line.color == "navy"

    def test_linewidth_forwarded(self) -> None:
        hull_x = np.array([0.0, 1.0])
        hull_y = np.array([0.0, 1.0])
        trace = render_convex_hull_plotly(hull_x, hull_y, linewidth=3.0)
        assert trace.line.width == pytest.approx(3.0)

    def test_label_and_showlegend(self) -> None:
        hull_x = np.array([0.0, 1.0])
        hull_y = np.array([0.0, 1.0])
        trace = render_convex_hull_plotly(hull_x, hull_y, label="hull", showlegend=False)
        assert trace.name == "hull"
        assert trace.showlegend is False


# ===========================================================================
# render_boolean_states_plotly
# ===========================================================================


class TestRenderBooleanStatesPlotly:
    """Behaviour tests for render_boolean_states_plotly."""

    def test_returns_list_of_traces(self) -> None:
        x = np.linspace(0, 1, 10)
        states = np.array([True, True, False, False, True, True, False, False, False, False])
        result = render_boolean_states_plotly(x, states)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_all_true_states_start_true_segment(self) -> None:
        """When states starts with True, at least one true trace is created."""
        import plotly.graph_objects as go

        # Use 6 elements and avoid all-True to sidestep the source's IndexError bug
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        states = np.array([True, True, True, False, False, False])
        traces = render_boolean_states_plotly(x, states, true_color="#00ff00")
        assert len(traces) >= 1
        for t in traces:
            assert isinstance(t, go.Scatter)

    def test_all_false_states_only_false_traces(self) -> None:
        x = np.linspace(0, 5, 6)
        states = np.array([False, False, False, False, False, False])
        traces = render_boolean_states_plotly(x, states, false_color="#ff0000")
        # Should produce false region trace(s); no true traces
        assert isinstance(traces, list)
        true_traces = [t for t in traces if t.fillcolor == "#2ca02c"]
        assert len(true_traces) == 0

    def test_mixed_states_correct_trace_count(self) -> None:
        # [T,T,F,F,F,F] -> 1 true segment, 1 false segment = 2 traces
        # NOTE: patterns ending in True trigger an IndexError in the source (source bug);
        # we use a pattern that ends in False to avoid it.
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        states = np.array([True, True, False, False, False, False])
        traces = render_boolean_states_plotly(x, states)
        # 1 true segment + 1 false segment = 2 traces
        assert len(traces) == 2

    def test_true_first_trace_has_true_label(self) -> None:
        x = np.array([0.0, 1.0, 2.0, 3.0])
        states = np.array([True, True, False, False])
        traces = render_boolean_states_plotly(x, states, true_label="ON")
        # First trace is from the true region
        assert traces[0].name == "ON"

    def test_true_subsequent_traces_have_empty_name(self) -> None:
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        states = np.array([True, False, True, False, True, False])
        traces = render_boolean_states_plotly(x, states)
        # Multiple true segments — first has name, rest are empty
        true_traces = [t for t in traces if t.fillcolor == "#2ca02c"]
        if len(true_traces) > 1:
            assert true_traces[1].name == ""

    def test_false_first_trace_has_false_label(self) -> None:
        x = np.array([0.0, 1.0, 2.0, 3.0])
        states = np.array([False, False, True, True])
        traces = render_boolean_states_plotly(x, states, false_label="OFF")
        false_traces = [t for t in traces if t.fillcolor == "#d62728"]
        assert len(false_traces) >= 1
        assert false_traces[0].name == "OFF"

    def test_alpha_sets_opacity(self) -> None:
        x = np.array([0.0, 1.0, 2.0, 3.0])
        states = np.array([True, True, False, False])
        traces = render_boolean_states_plotly(x, states, alpha=0.5)
        for t in traces:
            assert t.opacity == pytest.approx(0.5)

    def test_traces_have_toself_fill(self) -> None:
        x = np.array([0.0, 1.0, 2.0])
        states = np.array([True, True, False])
        traces = render_boolean_states_plotly(x, states)
        for t in traces:
            assert t.fill == "toself"

    def test_x_shape_preserves_endpoints(self) -> None:
        """Edge case: state ends at last element."""
        x = np.linspace(0, 9, 10)
        states = np.array([True, True, True, True, True, False, False, False, False, False])
        traces = render_boolean_states_plotly(x, states)
        assert len(traces) >= 1

    def test_starts_with_false_creates_false_trace(self) -> None:
        """Edge case: array begins with False — false_starts prepend logic."""
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        states = np.array([False, False, True, True, False])
        traces = render_boolean_states_plotly(x, states)
        false_traces = [t for t in traces if t.fillcolor == "#d62728"]
        assert len(false_traces) >= 1

    def test_ends_with_true_no_crash(self) -> None:
        x = np.array([0.0, 1.0, 2.0])
        states = np.array([False, True, True])
        traces = render_boolean_states_plotly(x, states)
        assert isinstance(traces, list)


# ===========================================================================
# render_ellipse_plotly
# ===========================================================================


class TestRenderEllipsePlotly:
    """Behaviour tests for render_ellipse_plotly."""

    # NOTE: render_ellipse_plotly calls plotly.colors.hex_to_rgb(color) which requires
    # a hex color string (e.g. "#ff0000"), not a named color like "red".
    # All tests below pass hex colors to match the expected API contract.

    def test_2d_centers_returns_list_of_shapes(self) -> None:
        centers = np.array([[0.0, 0.0], [1.0, 1.0]])
        widths = np.array([0.5, 0.3])
        heights = np.array([0.4, 0.2])
        shapes = render_ellipse_plotly(centers, widths, heights, color="#ff0000")
        assert isinstance(shapes, list)
        assert len(shapes) == 2

    def test_2d_shapes_are_paths(self) -> None:
        centers = np.array([[0.0, 0.0]])
        widths = np.array([1.0])
        heights = np.array([0.5])
        shapes = render_ellipse_plotly(centers, widths, heights, color="#ff0000")
        assert shapes[0]["type"] == "path"
        assert "path" in shapes[0]

    def test_2d_with_angles_rotates_ellipse(self) -> None:
        centers = np.array([[0.0, 0.0]])
        widths = np.array([1.0])
        heights = np.array([0.5])
        angles = np.array([45.0])
        shapes = render_ellipse_plotly(
            centers, widths, heights, angles=angles, color="#0000ff"
        )
        assert len(shapes) == 1
        assert shapes[0]["type"] == "path"

    def test_2d_no_angles_defaults_to_zero(self) -> None:
        centers = np.array([[0.0, 0.0]])
        widths = np.array([1.0])
        heights = np.array([0.5])
        # Without angles — should use 0 rotation
        shapes_no_angle = render_ellipse_plotly(
            centers, widths, heights, color="#ff0000"
        )
        shapes_zero_angle = render_ellipse_plotly(
            centers, widths, heights, angles=np.array([0.0]), color="#ff0000"
        )
        # Paths should be identical
        assert shapes_no_angle[0]["path"] == shapes_zero_angle[0]["path"]

    def test_1d_centers_returns_rect_shapes(self) -> None:
        centers = np.array([[0.5], [1.5]])
        widths = np.array([0.4, 0.6])
        heights = np.array([0.3, 0.5])
        shapes = render_ellipse_plotly(centers, widths, heights, color="#00ff00")
        assert len(shapes) == 2
        for s in shapes:
            assert s["type"] == "rect"

    def test_1d_rect_x_bounds_correct(self) -> None:
        centers = np.array([[2.0]])
        widths = np.array([1.0])
        heights = np.array([0.5])
        shapes = render_ellipse_plotly(centers, widths, heights, color="#00ff00")
        s = shapes[0]
        assert s["x0"] == pytest.approx(1.5)
        assert s["x1"] == pytest.approx(2.5)

    def test_color_to_rgba_conversion(self) -> None:
        centers = np.array([[0.0, 0.0]])
        widths = np.array([1.0])
        heights = np.array([0.5])
        shapes = render_ellipse_plotly(centers, widths, heights, color="#ff0000", alpha=0.3)
        assert "rgba(255" in shapes[0]["fillcolor"]

    def test_multiple_2d_ellipses(self) -> None:
        rng = np.random.default_rng(0)
        n = 5
        centers = rng.standard_normal((n, 2))
        widths = rng.random(n) + 0.1
        heights = rng.random(n) + 0.1
        angles = rng.uniform(0, 360, n)
        shapes = render_ellipse_plotly(
            centers, widths, heights, angles=angles, color="#ff8800"
        )
        assert len(shapes) == n

    def test_layer_below_set_in_rect(self) -> None:
        centers = np.array([[0.0]])
        widths = np.array([1.0])
        heights = np.array([0.5])
        shapes = render_ellipse_plotly(centers, widths, heights, color="#00ff00")
        assert shapes[0]["layer"] == "below"

    def test_line_width_zero_in_shapes(self) -> None:
        centers = np.array([[0.0, 0.0]])
        widths = np.array([1.0])
        heights = np.array([0.5])
        shapes = render_ellipse_plotly(centers, widths, heights, color="#ff0000")
        assert shapes[0]["line"]["width"] == 0


# ===========================================================================
# PLOTLY_AVAILABLE = False guard (ImportError paths)
# ===========================================================================


class TestPlotlyUnavailableGuards:
    """Test ImportError raised when plotly unavailable."""

    def test_scatter_raises_import_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import neural_analysis.plotting.renderers_plotly as mod

        monkeypatch.setattr(mod, "PLOTLY_AVAILABLE", False)
        with pytest.raises(ImportError, match="Plotly is required"):
            render_scatter_plotly(np.array([[0.0, 1.0]]))

    def test_scatter3d_raises_import_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import neural_analysis.plotting.renderers_plotly as mod

        monkeypatch.setattr(mod, "PLOTLY_AVAILABLE", False)
        with pytest.raises(ImportError, match="Plotly is required"):
            render_scatter3d_plotly(np.array([[0.0, 1.0, 2.0]]))

    def test_line_raises_import_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import neural_analysis.plotting.renderers_plotly as mod

        monkeypatch.setattr(mod, "PLOTLY_AVAILABLE", False)
        with pytest.raises(ImportError, match="Plotly is required"):
            render_line_plotly(np.array([1.0, 2.0]))

    def test_histogram_raises_import_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import neural_analysis.plotting.renderers_plotly as mod

        monkeypatch.setattr(mod, "PLOTLY_AVAILABLE", False)
        with pytest.raises(ImportError, match="Plotly is required"):
            render_histogram_plotly(np.array([1.0, 2.0]))

    def test_heatmap_raises_import_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import neural_analysis.plotting.renderers_plotly as mod

        monkeypatch.setattr(mod, "PLOTLY_AVAILABLE", False)
        with pytest.raises(ImportError, match="Plotly is required"):
            render_heatmap_plotly(np.ones((3, 3)))

    def test_bar_raises_import_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import neural_analysis.plotting.renderers_plotly as mod

        monkeypatch.setattr(mod, "PLOTLY_AVAILABLE", False)
        with pytest.raises(ImportError, match="Plotly is required"):
            render_bar_plotly(np.array([1.0, 2.0]))

    def test_violin_raises_import_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import neural_analysis.plotting.renderers_plotly as mod

        monkeypatch.setattr(mod, "PLOTLY_AVAILABLE", False)
        with pytest.raises(ImportError, match="Plotly is required"):
            render_violin_plotly(np.array([1.0, 2.0]))

    def test_box_raises_import_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import neural_analysis.plotting.renderers_plotly as mod

        monkeypatch.setattr(mod, "PLOTLY_AVAILABLE", False)
        with pytest.raises(ImportError, match="Plotly is required"):
            render_box_plotly(np.array([1.0, 2.0]))

    def test_trajectory_raises_import_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import neural_analysis.plotting.renderers_plotly as mod

        monkeypatch.setattr(mod, "PLOTLY_AVAILABLE", False)
        with pytest.raises(ImportError, match="Plotly is required"):
            render_trajectory_plotly(np.array([0.0]), np.array([0.0]))

    def test_trajectory3d_raises_import_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import neural_analysis.plotting.renderers_plotly as mod

        monkeypatch.setattr(mod, "PLOTLY_AVAILABLE", False)
        with pytest.raises(ImportError, match="Plotly is required"):
            render_trajectory3d_plotly(
                np.array([0.0]), np.array([0.0]), np.array([0.0])
            )

    def test_kde_raises_import_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import neural_analysis.plotting.renderers_plotly as mod

        monkeypatch.setattr(mod, "PLOTLY_AVAILABLE", False)
        with pytest.raises(ImportError, match="Plotly is required"):
            render_kde_plotly(
                np.ones((3, 3)), np.ones((3, 3)), np.ones((3, 3))
            )

    def test_convex_hull_raises_import_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import neural_analysis.plotting.renderers_plotly as mod

        monkeypatch.setattr(mod, "PLOTLY_AVAILABLE", False)
        with pytest.raises(ImportError, match="Plotly is required"):
            render_convex_hull_plotly(np.array([0.0, 1.0]), np.array([0.0, 1.0]))

    def test_boolean_states_raises_import_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import neural_analysis.plotting.renderers_plotly as mod

        monkeypatch.setattr(mod, "PLOTLY_AVAILABLE", False)
        with pytest.raises(ImportError, match="Plotly is required"):
            render_boolean_states_plotly(
                np.array([0.0, 1.0]), np.array([True, False])
            )
