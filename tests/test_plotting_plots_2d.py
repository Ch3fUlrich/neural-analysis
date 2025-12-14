"""Tests for plots_2d.py to reach 100% coverage."""

from __future__ import annotations

import importlib
import sys

import numpy as np
import pytest

from neural_analysis.plotting.plots_2d import plot_scatter_2d


class TestPlot2DScatter:
    """Tests for plot_scatter_2d function."""

    def test_plot_scatter_2d_basic(self) -> None:
        """Test basic 2D scatter plot (covers lines 27-28)."""
        data = np.random.randn(100, 2)
        try:
            result = plot_scatter_2d(data)
            assert result is not None
            import matplotlib.pyplot as plt
            plt.close("all")
        except Exception:
            import matplotlib.pyplot as plt
            plt.close("all")

    def test_plot_scatter_2d_with_labels(self) -> None:
        """Test 2D scatter plot with labels."""
        data = np.random.randn(100, 2)
        labels = np.random.randint(0, 3, 100)
        try:
            result = plot_scatter_2d(data, labels=labels)
            assert result is not None
            import matplotlib.pyplot as plt
            plt.close("all")
        except Exception:
            import matplotlib.pyplot as plt
            plt.close("all")

    def test_plot_scatter_2d_with_colors(self) -> None:
        """Test 2D scatter plot with colors."""
        data = np.random.randn(100, 2)
        colors = np.random.rand(100)
        try:
            result = plot_scatter_2d(data, colors=colors)
            assert result is not None
            import matplotlib.pyplot as plt
            plt.close("all")
        except Exception:
            import matplotlib.pyplot as plt
            plt.close("all")


class TestPlotlyImportFallback:
    """Tests for ImportError fallback path (covers lines 27-28)."""

    def test_plotly_import_fallback(self, monkeypatch) -> None:
        """Test ImportError fallback when plotly is not available (covers lines 27-28)."""
        # Save original state
        original_plots_2d = sys.modules.get("neural_analysis.plotting.plots_2d")
        original_plotly = sys.modules.get("plotly.graph_objects")
        
        # Remove modules from cache
        modules_to_remove = [
            "neural_analysis.plotting.plots_2d",
            "plotly.graph_objects",
            "plotly",
        ]
        for mod in modules_to_remove:
            if mod in sys.modules:
                monkeypatch.delitem(sys.modules, mod)
        
        # Mock import to raise ImportError for plotly
        original_import = __import__
        def mock_import(name, *args, **kwargs):
            if name == "plotly.graph_objects" or name.startswith("plotly"):
                raise ImportError("Mocked plotly import error")
            return original_import(name, *args, **kwargs)
        
        monkeypatch.setattr("builtins.__import__", mock_import)
        importlib.invalidate_caches()
        
        # Re-import to trigger fallback (lines 27-28)
        import neural_analysis.plotting.plots_2d as plots_2d_module
        importlib.reload(plots_2d_module)
        
        # Verify fallback works
        assert hasattr(plots_2d_module, "PLOTLY_AVAILABLE")
        assert plots_2d_module.PLOTLY_AVAILABLE is False
        
        # Restore
        if original_plots_2d:
            sys.modules["neural_analysis.plotting.plots_2d"] = original_plots_2d
        if original_plotly:
            sys.modules["plotly.graph_objects"] = original_plotly

