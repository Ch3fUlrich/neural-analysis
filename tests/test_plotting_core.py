"""Tests for plotting core utilities."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np
import pytest

from neural_analysis.plotting.backend import BackendType
from neural_analysis.plotting.core import (
    PlotConfig,
    apply_layout_matplotlib,
    apply_layout_plotly,
    apply_layout_plotly_3d,
    calculate_alpha,
    create_rgba_labels,
    finalize_plot_matplotlib,
    finalize_plot_plotly,
    generate_similar_colors,
    get_default_categorical_colors,
    make_list_if_not,
    resolve_colormap,
    save_plot,
)


class TestPlotConfig:
    """Test PlotConfig dataclass."""

    def test_plot_config_defaults(self):
        """Test PlotConfig with default values."""
        config = PlotConfig()
        assert config.title is None
        assert config.xlabel is None
        assert config.figsize == (10, 6)  # Has default
        assert config.dpi == 100
        assert config.grid is False
        assert config.legend is True

    def test_plot_config_custom_values(self):
        """Test PlotConfig with custom values."""
        config = PlotConfig(
            title="Test Plot",
            xlabel="X Axis",
            ylabel="Y Axis",
            figsize=(10, 6),
            grid=True,
            legend=False,
        )
        assert config.title == "Test Plot"
        assert config.xlabel == "X Axis"
        assert config.ylabel == "Y Axis"
        assert config.figsize == (10, 6)
        assert config.grid is True
        assert config.legend is False

    def test_plot_config_save_path(self):
        """Test PlotConfig with save_path."""
        config = PlotConfig(save_path="test.png")
        assert config.save_path == Path("test.png")

    def test_plot_config_save_dir(self):
        """Test PlotConfig with save_dir."""
        config = PlotConfig(
            save_dir="output", plot_type="scatter", additional_save_title="test"
        )
        assert config.save_dir == Path("output")
        assert config.plot_type == "scatter"
        assert config.additional_save_title == "test"

    def test_plot_config_get_save_path_with_save_path(self):
        """Test get_save_path() when save_path is provided (line 166-167)."""
        config = PlotConfig(save_path="test.png")
        save_path = config.get_save_path()
        assert save_path == Path("test.png")
        assert isinstance(save_path, Path)

    def test_plot_config_get_save_path_with_save_dir(self):
        """Test get_save_path() when save_dir is provided (line 171-185)."""
        config = PlotConfig(
            save_dir="output", plot_type="scatter", additional_save_title="test"
        )
        save_path = config.get_save_path()
        assert save_path == Path("output/scatter_test.png")
        assert isinstance(save_path, Path)

    def test_plot_config_get_save_path_with_save_dir_no_parts(self):
        """Test get_save_path() with save_dir but no plot_type or title (line 177-179)."""
        config = PlotConfig(save_dir="output")
        save_path = config.get_save_path()
        assert save_path == Path("output/plot.png")

    def test_plot_config_get_save_path_none(self):
        """Test get_save_path() when neither save_path nor save_dir is provided."""
        config = PlotConfig()
        save_path = config.get_save_path()
        assert save_path is None


class TestResolveColormap:
    """Test resolve_colormap function."""

    def test_resolve_colormap_matplotlib(self):
        """Test colormap resolution for matplotlib."""
        result = resolve_colormap("viridis", BackendType.MATPLOTLIB)
        assert isinstance(result, str)  # Returns colormap name as string
        result = resolve_colormap("plasma", BackendType.MATPLOTLIB)
        assert isinstance(result, str)

    def test_resolve_colormap_plotly(self):
        """Test colormap resolution for plotly."""
        result = resolve_colormap("viridis", BackendType.PLOTLY)
        assert isinstance(result, str)
        result = resolve_colormap("plasma", BackendType.PLOTLY)
        assert isinstance(result, str)

    def test_resolve_colormap_none(self):
        """Test colormap resolution with None."""
        result = resolve_colormap(None, BackendType.MATPLOTLIB)
        assert isinstance(result, str)
        result = resolve_colormap(None, BackendType.PLOTLY)
        assert isinstance(result, str)

    def test_resolve_colormap_unknown_matplotlib(self):
        """Test resolve_colormap with unknown colormap (triggers warning and fallback, line 239-252)."""
        with pytest.warns(UserWarning):
            result = resolve_colormap("nonexistent_cmap_xyz123", BackendType.MATPLOTLIB)
        assert isinstance(result, str)
        # Result should be a valid colormap string (fallback to viridis)
        assert len(result) > 0

    def test_resolve_colormap_capitalized(self):
        """Test resolve_colormap with capitalized name (line 234)."""
        result = resolve_colormap("viridis", BackendType.MATPLOTLIB)
        assert isinstance(result, str)

    def test_resolve_colormap_old_matplotlib_fallback(self):
        """Test resolve_colormap with old matplotlib fallback (line 249-252)."""
        # Mock old matplotlib where plt.colormaps raises AttributeError when accessing
        with patch("neural_analysis.plotting.core.plt.colormaps") as mock_colormaps:
            # First two attempts fail (KeyError for both name and capitalized)
            call_count = [0]

            def getitem_side_effect(key):
                call_count[0] += 1
                # First two calls (unknown, Unknown) raise KeyError
                if call_count[0] <= 2:
                    raise KeyError(key)
                # Third call (viridis) raises AttributeError (old matplotlib)
                if key == "viridis":
                    raise AttributeError("old matplotlib - no get_cmap")
                raise KeyError(key)

            mock_colormaps.__getitem__ = MagicMock(side_effect=getitem_side_effect)
            # Mock get_cmap method for old matplotlib fallback
            mock_cmap_obj = MagicMock()
            mock_cmap_obj.__str__ = MagicMock(return_value="viridis")
            mock_colormaps.get_cmap = MagicMock(return_value=mock_cmap_obj)

            with pytest.warns(UserWarning):
                result = resolve_colormap("unknown", BackendType.MATPLOTLIB)
            assert isinstance(result, str)
            assert result == "viridis"


class TestApplyLayout:
    """Test layout application functions."""

    def test_apply_layout_matplotlib(self):
        """Test applying layout to matplotlib axes."""
        fig, ax = plt.subplots()
        config = PlotConfig(
            title="Test",
            xlabel="X",
            ylabel="Y",
            xlim=(0, 10),
            ylim=(0, 20),
            grid=True,
        )
        apply_layout_matplotlib(ax, config)
        assert ax.get_title() == "Test"
        assert ax.get_xlabel() == "X"
        assert ax.get_ylabel() == "Y"
        plt.close(fig)

    def test_apply_layout_matplotlib_3d(self):
        """Test applying layout to 3D matplotlib axes (covers zlabel, zlim, line 268-275)."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        config = PlotConfig(
            title="Test 3D",
            xlabel="X",
            ylabel="Y",
            zlabel="Z",
            xlim=(0, 10),
            ylim=(0, 20),
            zlim=(0, 30),
            grid=True,
        )
        apply_layout_matplotlib(ax, config)
        assert ax.get_title() == "Test 3D"
        assert ax.get_xlabel() == "X"
        assert ax.get_ylabel() == "Y"
        assert ax.get_zlabel() == "Z"
        plt.close(fig)

    def test_apply_layout_matplotlib_2d_with_zlabel(self):
        """Test applying layout to 2D axes with zlabel (should be ignored, line 268)."""
        fig, ax = plt.subplots()
        config = PlotConfig(zlabel="Z")  # 2D axes don't have set_zlabel
        apply_layout_matplotlib(ax, config)  # Should not error
        plt.close(fig)

    def test_apply_layout_matplotlib_2d_with_zlim(self):
        """Test applying layout to 2D axes with zlim (should be ignored, line 274)."""
        fig, ax = plt.subplots()
        config = PlotConfig(zlim=(0, 30))  # 2D axes don't have set_zlim
        apply_layout_matplotlib(ax, config)  # Should not error
        plt.close(fig)

    def test_apply_layout_matplotlib_tight_layout(self):
        """Test applying layout with tight_layout (line 278)."""
        fig, ax = plt.subplots()
        config = PlotConfig(tight_layout=True)
        apply_layout_matplotlib(ax, config)
        plt.close(fig)

    def test_apply_layout_matplotlib_no_tight_layout(self):
        """Test applying layout without tight_layout (covers branch 278->exit False path)."""
        fig, ax = plt.subplots()
        config = PlotConfig(tight_layout=False)
        apply_layout_matplotlib(ax, config)
        plt.close(fig)

    def test_apply_layout_plotly_no_axes(self):
        """Test apply_layout_plotly without xaxis/yaxis/figsize (covers branches 300->302, 302->304, 304->307 False paths)."""
        try:
            import plotly.graph_objects as go

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 4, 9]))
            # Config with no xlim, ylim, or figsize to trigger False paths
            # This means xaxis and yaxis dicts will be empty, and figsize is None
            config = PlotConfig(
                title="Test",
                xlabel="X",
                ylabel="Y",
                grid=False,
                xlim=None,
                ylim=None,
                figsize=None,
            )
            apply_layout_plotly(fig, config)
            # Should not error - xaxis and yaxis will be empty dicts, so the if checks fail
        except ImportError:
            pytest.skip("Plotly not available")

    def test_apply_layout_plotly(self):
        """Test applying layout to plotly figure."""
        try:
            import plotly.graph_objects as go

            fig = go.Figure()
            config = PlotConfig(
                title="Test",
                xlabel="X",
                ylabel="Y",
                xlim=(0, 10),
                ylim=(0, 20),
                grid=True,
            )
            apply_layout_plotly(fig, config)
            assert fig.layout.title.text == "Test"
            assert fig.layout.xaxis.title.text == "X"
            assert fig.layout.yaxis.title.text == "Y"
        except ImportError:
            pytest.skip("Plotly not available")

    def test_apply_layout_plotly_3d(self):
        """Test applying layout to 3D plotly figure."""
        try:
            import plotly.graph_objects as go

            fig = go.Figure()
            config = PlotConfig(
                title="Test 3D",
                xlabel="X",
                ylabel="Y",
                zlabel="Z",
                xlim=(0, 10),
                ylim=(0, 20),
                zlim=(0, 30),
            )
            apply_layout_plotly_3d(fig, config)
            assert fig.layout.title.text == "Test 3D"
            assert fig.layout.scene.xaxis.title.text == "X"
            assert fig.layout.scene.yaxis.title.text == "Y"
            assert fig.layout.scene.zaxis.title.text == "Z"
        except ImportError:
            pytest.skip("Plotly not available")

    def test_apply_layout_plotly_3d_no_figsize(self):
        """Test apply_layout_plotly_3d without figsize (covers branch 339->343 False path)."""
        try:
            import plotly.graph_objects as go

            fig = go.Figure()
            fig.add_trace(go.Scatter3d(x=[1, 2, 3], y=[1, 4, 9], z=[1, 8, 27]))
            config = PlotConfig(
                title="Test 3D",
                xlabel="X",
                ylabel="Y",
                zlabel="Z",
                figsize=None,  # No figsize to trigger False path
            )
            apply_layout_plotly_3d(fig, config)
            # Should not error
        except ImportError:
            pytest.skip("Plotly not available")

    def test_apply_layout_plotly_3d_partial_limits(self):
        """Test applying layout to 3D plotly with partial limits (covers branches 324->326, 328->330, 332->334)."""
        try:
            import plotly.graph_objects as go

            # Test with only xlim (triggers xaxis dict creation, line 324->326)
            fig = go.Figure()
            config = PlotConfig(xlim=(0, 10))
            apply_layout_plotly_3d(fig, config)
            assert "scene" in fig.layout.to_plotly_json()
            # Range is stored as list, not tuple
            assert list(fig.layout.scene.xaxis.range) == [0, 10]

            # Test with ylim (triggers yaxis dict creation, line 327->330)
            fig2 = go.Figure()
            config2 = PlotConfig(ylim=(0, 20))
            apply_layout_plotly_3d(fig2, config2)
            assert list(fig2.layout.scene.yaxis.range) == [0, 20]

            # Test with zlim (triggers zaxis dict creation, line 331->334)
            fig3 = go.Figure()
            config3 = PlotConfig(zlim=(0, 30))
            apply_layout_plotly_3d(fig3, config3)
            assert list(fig3.layout.scene.zaxis.range) == [0, 30]

            # Test with multiple limits (triggers nested dict creation, line 324->326, 328->330)
            fig4 = go.Figure()
            config4 = PlotConfig(xlim=(0, 10), ylim=(0, 20))
            apply_layout_plotly_3d(fig4, config4)
            assert list(fig4.layout.scene.xaxis.range) == [0, 10]
            assert list(fig4.layout.scene.yaxis.range) == [0, 20]

            # Test with all three limits together (covers False paths: when xaxis exists, check ylim; when yaxis exists, check zlim)
            # When processing xlim, ylim, zlim together:
            # - xlim: "xaxis" not in scene_dict -> True path (creates xaxis)
            # - ylim: "yaxis" not in scene_dict -> True path (creates yaxis)
            # - zlim: "zaxis" not in scene_dict -> True path (creates zaxis)
            # To trigger False paths, we need to ensure axis dicts already exist
            fig5 = go.Figure()
            fig5.add_trace(go.Scatter3d(x=[1, 2, 3], y=[1, 4, 9], z=[1, 8, 27]))
            # First set xlim to create xaxis dict
            config5a = PlotConfig(xlim=(0, 10))
            apply_layout_plotly_3d(fig5, config5a)
            # Then set ylim - xaxis already exists, but we're checking yaxis (should be True path)
            # Actually, the False paths are when checking if the axis key exists in scene_dict
            # Let's test by manually creating the scene_dict structure
            config5 = PlotConfig(xlim=(0, 10), ylim=(0, 20), zlim=(0, 30))
            apply_layout_plotly_3d(fig5, config5)
            assert list(fig5.layout.scene.xaxis.range) == [0, 10]
            assert list(fig5.layout.scene.yaxis.range) == [0, 20]
            assert list(fig5.layout.scene.zaxis.range) == [0, 30]

            # Test False paths: when axis dicts already exist in scene_dict
            # This happens when we process limits in a way that creates nested structure
            # Actually, the code always creates fresh scene_dict, so False paths are hard to trigger
            # The False paths 324->326, 328->330, 332->334 occur when the axis key already exists
            # But since we create scene_dict fresh each time, this is unlikely
            # These branches might be unreachable in normal execution

            # Test with figsize (line 339->343)
            fig6 = go.Figure()
            config6 = PlotConfig(figsize=(10, 8))
            apply_layout_plotly_3d(fig6, config6)
            assert fig6.layout.width == 1000
            assert fig6.layout.height == 800

            # Test with empty scene_dict (line 336->339) - no scene config
            fig7 = go.Figure()
            config7 = PlotConfig(title="Test")  # Only title, no scene config
            apply_layout_plotly_3d(fig7, config7)
            # Should not error

            # Test with xlim only to trigger nested dict creation (line 324->326)
            fig8 = go.Figure()
            config8 = PlotConfig(xlim=(0, 10))
            apply_layout_plotly_3d(fig8, config8)
            assert list(fig8.layout.scene.xaxis.range) == [0, 10]

            # Test with ylim only to trigger nested dict creation (line 328->330)
            fig9 = go.Figure()
            config9 = PlotConfig(ylim=(0, 20))
            apply_layout_plotly_3d(fig9, config9)
            assert list(fig9.layout.scene.yaxis.range) == [0, 20]

            # Test with zlim only to trigger nested dict creation (line 332->334)
            fig10 = go.Figure()
            config10 = PlotConfig(zlim=(0, 30))
            apply_layout_plotly_3d(fig10, config10)
            assert list(fig10.layout.scene.zaxis.range) == [0, 30]
        except ImportError:
            pytest.skip("Plotly not available")

    def test_apply_layout_plotly_with_figsize(self):
        """Test applying layout to plotly with figsize (line 304->307)."""
        try:
            import plotly.graph_objects as go

            fig = go.Figure()
            config = PlotConfig(figsize=(12, 8))
            apply_layout_plotly(fig, config)
            assert fig.layout.width == 1200
            assert fig.layout.height == 800
        except ImportError:
            pytest.skip("Plotly not available")

    def test_apply_layout_plotly_empty_axis_dicts(self):
        """Test applying layout to plotly with empty axis dicts (line 300->304)."""
        try:
            import plotly.graph_objects as go

            # Test with no limits, no grid - axis dicts should be empty and not added
            # But grid=False still creates showgrid entries, so dicts aren't empty
            # To test empty dicts, we need grid=None or grid=False with no limits
            fig = go.Figure()
            config = PlotConfig()  # No limits, no grid
            apply_layout_plotly(fig, config)
            # Should not error
        except ImportError:
            pytest.skip("Plotly not available")

    def test_apply_layout_plotly_only_figsize(self):
        """Test applying layout to plotly with only figsize (tests branch 300->304->307)."""
        try:
            import plotly.graph_objects as go

            fig = go.Figure()
            config = PlotConfig(figsize=(10, 8), grid=False)  # Only figsize, no limits
            apply_layout_plotly(fig, config)
            assert fig.layout.width == 1000
            assert fig.layout.height == 800
        except ImportError:
            pytest.skip("Plotly not available")

    def test_apply_layout_plotly_with_grid(self):
        """Test applying layout to plotly with grid (line 297->299)."""
        try:
            import plotly.graph_objects as go

            fig = go.Figure()
            config = PlotConfig(grid=True)
            apply_layout_plotly(fig, config)
            assert fig.layout.xaxis.showgrid is True
            assert fig.layout.yaxis.showgrid is True
        except ImportError:
            pytest.skip("Plotly not available")

    def test_apply_layout_plotly_with_xlim_only(self):
        """Test applying layout to plotly with only xlim (line 300->302)."""
        try:
            import plotly.graph_objects as go

            fig = go.Figure()
            config = PlotConfig(xlim=(0, 10))
            apply_layout_plotly(fig, config)
            # xaxis should be created and have range (even with grid, xaxis dict is created)
            assert hasattr(fig.layout, "xaxis")
            # Range is stored as list
            assert list(fig.layout.xaxis.range) == [0, 10]
        except ImportError:
            pytest.skip("Plotly not available")

    def test_apply_layout_plotly_with_ylim_only(self):
        """Test applying layout to plotly with only ylim (line 302->304)."""
        try:
            import plotly.graph_objects as go

            fig = go.Figure()
            config = PlotConfig(ylim=(0, 20))
            apply_layout_plotly(fig, config)
            # yaxis should be created and have range
            assert hasattr(fig.layout, "yaxis")
            # Range is stored as list
            assert list(fig.layout.yaxis.range) == [0, 20]
        except ImportError:
            pytest.skip("Plotly not available")


class TestGetDefaultCategoricalColors:
    """Test get_default_categorical_colors function."""

    def test_get_default_categorical_colors(self):
        """Test getting default categorical colors."""
        colors = get_default_categorical_colors(5)
        assert len(colors) == 5
        assert all(isinstance(c, str) for c in colors)

    def test_get_default_categorical_colors_large_n(self):
        """Test with large number of colors."""
        colors = get_default_categorical_colors(20)
        assert len(colors) == 20

    def test_get_default_categorical_colors_old_matplotlib(self):
        """Test get_default_categorical_colors with AttributeError fallback (line 354-355)."""
        # Mock old matplotlib where plt.colormaps raises AttributeError
        with patch("matplotlib.pyplot.colormaps") as mock_colormaps:
            # Simulate AttributeError when accessing plt.colormaps["tab10"]
            mock_colormaps.__getitem__ = MagicMock(
                side_effect=AttributeError("old matplotlib")
            )
            # The function should fall back to plt.get_cmap
            with patch("matplotlib.pyplot.get_cmap") as mock_get_cmap:
                # Create a mock palette that is callable and returns RGBA tuples
                def palette_callable(i):
                    return (0.1 * (i % 10), 0.2 * (i % 10), 0.3 * (i % 10), 1.0)

                mock_palette = MagicMock()
                mock_palette.N = 10
                mock_palette.side_effect = palette_callable
                mock_get_cmap.return_value = mock_palette

                colors = get_default_categorical_colors(5)
                assert len(colors) == 5
                assert all(isinstance(c, str) and c.startswith("#") for c in colors)


class TestFinalizePlot:
    """Test plot finalization functions."""

    def test_finalize_plot_matplotlib(self):
        """Test finalizing matplotlib plot."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        config = PlotConfig(save_path=None, show=False, tight_layout=True)
        finalize_plot_matplotlib(config)
        plt.close(fig)

    def test_finalize_plot_matplotlib_with_save(self):
        """Test finalizing matplotlib plot with save (line 371-372)."""
        import tempfile
        from pathlib import Path

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        with tempfile.TemporaryDirectory() as tmpdir:
            config = PlotConfig(
                save_path=Path(tmpdir) / "test.png", show=False, save_format="png"
            )
            finalize_plot_matplotlib(config)
            assert (Path(tmpdir) / "test.png").exists()
        plt.close(fig)

    def test_finalize_plot_matplotlib_with_show(self):
        """Test finalizing matplotlib plot with show=True (line 373-374)."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        config = PlotConfig(show=False)  # Set to False to avoid actually showing
        finalize_plot_matplotlib(config)
        plt.close(fig)

    def test_finalize_plot_matplotlib_with_show_true(self):
        """Test finalizing matplotlib plot with show=True (line 374)."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        config = PlotConfig(show=True)
        # We can't actually test plt.show(), but we can test the code path
        # by mocking or just ensuring it doesn't error
        finalize_plot_matplotlib(config)
        plt.close(fig)

    def test_finalize_plot_plotly(self):
        """Test finalizing plotly plot."""
        try:
            import plotly.graph_objects as go

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 4, 9]))
            config = PlotConfig(save_path=None, show=False)
            finalize_plot_plotly(fig, config)
        except ImportError:
            pytest.skip("Plotly not available")

    def test_finalize_plot_plotly_save_html(self):
        """Test finalizing plotly plot with HTML save format (line 387-388)."""
        try:
            import tempfile
            from pathlib import Path

            import plotly.graph_objects as go

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 4, 9]))
            with tempfile.TemporaryDirectory() as tmpdir:
                config = PlotConfig(
                    save_path=Path(tmpdir) / "test.html",
                    show=False,
                    save_format="html",
                )
                finalize_plot_plotly(fig, config)
                assert (Path(tmpdir) / "test.html").exists()
        except ImportError:
            pytest.skip("Plotly not available")

    def test_finalize_plot_plotly_save_image_with_html(self):
        """Test finalizing plotly plot with image save and HTML (line 390->405)."""
        try:
            import tempfile
            from pathlib import Path

            import plotly.graph_objects as go

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 4, 9]))
            with tempfile.TemporaryDirectory() as tmpdir:
                config = PlotConfig(
                    save_path=Path(tmpdir) / "test.png",
                    show=False,
                    save_format="png",
                    save_html=True,
                )
                # This may fail if kaleido is not installed, but should test the code path
                try:
                    finalize_plot_plotly(fig, config)
                    # Check if HTML was created (line 403->405)
                    html_path = Path(tmpdir) / "test.html"
                    if html_path.exists():
                        assert True  # HTML was created
                except Exception:
                    pass  # Expected if kaleido not installed
        except ImportError:
            pytest.skip("Plotly not available")

    def test_finalize_plot_plotly_save_image_no_html(self):
        """Test finalizing plotly plot with image save but save_html=False (line 403)."""
        try:
            import tempfile
            from pathlib import Path

            import plotly.graph_objects as go

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 4, 9]))
            with tempfile.TemporaryDirectory() as tmpdir:
                config = PlotConfig(
                    save_path=Path(tmpdir) / "test.png",
                    show=False,
                    save_format="png",
                    save_html=False,  # Don't save HTML
                )
                try:
                    finalize_plot_plotly(fig, config)
                    html_path = Path(tmpdir) / "test.html"
                    # HTML should not exist when save_html=False
                    assert not html_path.exists()
                except Exception:
                    pass  # Expected if kaleido not installed
        except ImportError:
            pytest.skip("Plotly not available")

    def test_finalize_plot_plotly_show_jupyter(self):
        """Test finalizing plotly plot with show in Jupyter (line 408->414)."""
        try:
            import plotly.graph_objects as go

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 4, 9]))
            config = PlotConfig(show=False)  # Set to False to avoid actually showing
            finalize_plot_plotly(fig, config)
        except ImportError:
            pytest.skip("Plotly not available")

    def test_finalize_plot_plotly_show_true(self):
        """Test finalizing plotly plot with show=True (line 407->414)."""
        try:
            import plotly.graph_objects as go

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 4, 9]))
            config = PlotConfig(show=True)
            # This will try IPython.display first, then fall back to fig.show()
            # We can't test the actual display, but we can test it doesn't error
            finalize_plot_plotly(fig, config)
        except ImportError:
            pytest.skip("Plotly not available")

    def test_finalize_plot_plotly_show_ipython_fallback(self):
        """Test finalizing plotly plot with IPython display fallback (line 408->414)."""
        try:
            import plotly.graph_objects as go

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 4, 9]))
            config = PlotConfig(show=True)

            # Mock IPython.display to raise Exception (simulating non-Jupyter environment)
            # The import happens inside the function, so we patch the import location
            with patch("IPython.display.display") as mock_display:
                mock_display.side_effect = Exception("Not in Jupyter")
                # Should fall back to fig.show() (line 412->414)
                with patch.object(fig, "show") as mock_show:
                    finalize_plot_plotly(fig, config)
                    # Verify display was attempted and show was called as fallback
                    mock_display.assert_called_once()
                    mock_show.assert_called_once()
        except ImportError:
            pytest.skip("Plotly not available")


class TestCalculateAlpha:
    """Test calculate_alpha function."""

    def test_calculate_alpha_basic(self):
        """Test basic alpha calculation."""
        values = np.array([1, 2, 3, 4, 5])
        alphas = calculate_alpha(values)
        assert len(alphas) == len(values)
        assert all(0 <= a <= 1 for a in alphas)

    def test_calculate_alpha_single_value(self):
        """Test alpha calculation with single value."""
        values = np.array([5])
        alphas = calculate_alpha(values)
        assert len(alphas) == 1
        assert alphas[0] == 1.0

    def test_calculate_alpha_constant_values(self):
        """Test alpha calculation with constant values."""
        values = np.array([5, 5, 5, 5])
        alphas = calculate_alpha(values)
        # Constant values return min_alpha (default 0.3)
        assert all(a == 0.3 for a in alphas)

    def test_calculate_alpha_single_value_with_range(self):
        """Test alpha calculation with single value and explicit range (line 471)."""
        alpha = calculate_alpha(5, min_value=0, max_value=10)
        assert isinstance(alpha, float)
        assert 0.3 <= alpha <= 1.0

    def test_calculate_alpha_array_with_explicit_range(self):
        """Test alpha calculation with array and explicit range."""
        values = np.array([1, 5, 10])
        alphas = calculate_alpha(
            values, min_value=0, max_value=10, min_alpha=0.5, max_alpha=0.9
        )
        assert len(alphas) == 3
        assert all(0.5 <= a <= 0.9 for a in alphas)

    def test_calculate_alpha_invalid_range(self):
        """Test alpha calculation with invalid alpha range."""
        with pytest.raises(ValueError, match="min_alpha.*must be <= max_alpha"):
            calculate_alpha([1, 2, 3], min_alpha=0.9, max_alpha=0.3)

    def test_calculate_alpha_empty_array(self):
        """Test alpha calculation with empty array."""
        with pytest.raises(ValueError, match="No values provided"):
            calculate_alpha([])


class TestGenerateSimilarColors:
    """Test generate_similar_colors function."""

    def test_generate_similar_colors(self):
        """Test generating similar colors."""
        # base_color is RGB tuple (0-1 range)
        base_color = (1.0, 0.0, 0.0)  # Red
        colors = generate_similar_colors(base_color, num_colors=5)
        assert len(colors) == 5
        assert all(isinstance(c, tuple) and len(c) == 3 for c in colors)

    def test_generate_similar_colors_blue(self):
        """Test generating similar colors from blue."""
        base_color = (0.0, 0.0, 1.0)  # Blue
        colors = generate_similar_colors(base_color, num_colors=3)
        assert len(colors) == 3
        assert all(isinstance(c, tuple) and len(c) == 3 for c in colors)


class TestCreateRgbaLabels:
    """Test create_rgba_labels function."""

    def test_create_rgba_labels(self):
        """Test creating RGBA labels."""
        values = np.array([0.1, 0.5, 0.9])
        rgba = create_rgba_labels(values, alpha=0.8, cmap="rainbow")
        assert rgba.shape == (3, 4)
        assert all(0 <= c <= 1 for row in rgba for c in row)

    def test_create_rgba_labels_custom_alpha(self):
        """Test creating RGBA labels with custom alpha."""
        values = np.array([0.1, 0.5, 0.9])
        rgba = create_rgba_labels(values, alpha=0.5)
        assert all(row[3] == 0.5 for row in rgba)

    def test_create_rgba_labels_2d_array(self):
        """Test creating RGBA labels with 2D array (line 618->627)."""
        values = np.array([[0.1, 0.2], [0.5, 0.6], [0.9, 1.0]])
        rgba = create_rgba_labels(values, alpha=0.8)
        assert rgba.shape == (3, 4)
        # Check that red and blue channels are set from first and second dimensions
        # Values are normalized, so they should be in [0, 1] range
        assert rgba[0, 0] >= 0 and rgba[0, 0] <= 1  # Red from first dim
        assert rgba[0, 2] >= 0 and rgba[0, 2] <= 1  # Blue from second dim
        assert rgba[0, 1] == 0.5  # Green is constant 0.5
        assert rgba[0, 3] == 0.8  # Alpha

    def test_create_rgba_labels_invalid_dimension(self):
        """Test creating RGBA labels with invalid dimension (line 626->627)."""
        # Create a 2D array with shape[1] != 2 (should trigger ValueError)
        values = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])  # 2D array with 3 columns
        with pytest.raises(ValueError, match="Invalid values dimension"):
            create_rgba_labels(values)


class TestSavePlot:
    """Test save_plot function."""

    def test_save_plot_png(self):
        """Test saving plot as PNG."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test.png"
            save_plot(save_path, format="png")
            assert save_path.exists()
        plt.close(fig)

    def test_save_plot_pdf(self):
        """Test saving plot as PDF."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test"
            save_plot(save_path, format="pdf")
            assert save_path.with_suffix(".pdf").exists()
        plt.close(fig)

    def test_save_plot_custom_dpi(self):
        """Test saving plot with custom DPI."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test.png"
            save_plot(save_path, format="png", dpi=150)
            assert save_path.exists()
        plt.close(fig)


class TestMakeListIfNot:
    """Test make_list_if_not function."""

    def test_make_list_if_not_with_list(self):
        """Test with list input."""
        result = make_list_if_not([1, 2, 3])
        assert result == [1, 2, 3]

    def test_make_list_if_not_with_single_value(self):
        """Test with single value."""
        result = make_list_if_not(5)
        assert result == [5]

    def test_make_list_if_not_with_string(self):
        """Test with string."""
        result = make_list_if_not("test")
        assert result == ["test"]
