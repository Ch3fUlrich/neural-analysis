"""Tests targeting uncovered lines in neural_analysis.plotting.grid_config.

Missing lines from baseline: 290, 293-294, 297-298, 340, 344
  - Line 290:   GridLayoutConfig.auto_size_grid when both rows and cols are set
  - Lines 293-294: auto_size_grid when only rows is set (cols auto-computed)
  - Lines 297-298: auto_size_grid when only cols is set (rows auto-computed)
  - Line 340:   ColorScheme.get_colors when group_colors is explicitly provided
  - Line 344:   ColorScheme.get_colors when palette is a list (not a palette name)
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from neural_analysis.plotting.grid_config import (
    ColorScheme,
    GridLayoutConfig,
    PlotSpec,
    _convert_data_to_array,
)


# ---------------------------------------------------------------------------
# GridLayoutConfig.auto_size_grid — uncovered branches
# ---------------------------------------------------------------------------


class TestAutoSizeGridBothSet:
    """Line 290: both rows and cols explicitly set — early return."""

    def test_both_rows_and_cols_set_returns_them_unchanged(self) -> None:
        """When rows=2, cols=3 both set, auto_size_grid must return (2, 3)."""
        config = GridLayoutConfig(rows=2, cols=3)
        rows, cols = config.auto_size_grid(n_plots=10)
        assert rows == 2
        assert cols == 3

    def test_both_set_n_plots_ignored(self) -> None:
        """n_plots should not affect the result when both rows and cols are set."""
        config = GridLayoutConfig(rows=1, cols=1)
        rows, cols = config.auto_size_grid(n_plots=999)
        assert rows == 1
        assert cols == 1

    def test_both_set_large_grid(self) -> None:
        """Large explicit grid dimensions are returned as-is."""
        config = GridLayoutConfig(rows=5, cols=7)
        rows, cols = config.auto_size_grid(n_plots=3)
        assert rows == 5
        assert cols == 7


class TestAutoSizeGridOnlyRowsSet:
    """Lines 293-294: only rows is set; cols is auto-calculated."""

    def test_only_rows_set_cols_auto(self) -> None:
        """With rows=2 and 6 plots, cols should be ceil(6/2) = 3."""
        config = GridLayoutConfig(rows=2)
        rows, cols = config.auto_size_grid(n_plots=6)
        assert rows == 2
        assert cols == 3

    def test_only_rows_set_ceil_division(self) -> None:
        """ceil(7/3) = 3 — ensure ceiling division is applied."""
        config = GridLayoutConfig(rows=3)
        rows, cols = config.auto_size_grid(n_plots=7)
        assert rows == 3
        assert cols == int(np.ceil(7 / 3))  # 3

    def test_only_rows_set_single_row(self) -> None:
        """rows=1, all plots in one row."""
        config = GridLayoutConfig(rows=1)
        rows, cols = config.auto_size_grid(n_plots=5)
        assert rows == 1
        assert cols == 5


class TestAutoSizeGridOnlyColsSet:
    """Lines 297-298: only cols is set; rows is auto-calculated."""

    def test_only_cols_set_rows_auto(self) -> None:
        """With cols=3 and 9 plots, rows should be ceil(9/3) = 3."""
        config = GridLayoutConfig(cols=3)
        rows, cols = config.auto_size_grid(n_plots=9)
        assert rows == 3
        assert cols == 3

    def test_only_cols_set_ceil_division(self) -> None:
        """ceil(7/2) = 4 — ensure ceiling division is applied."""
        config = GridLayoutConfig(cols=2)
        rows, cols = config.auto_size_grid(n_plots=7)
        assert rows == int(np.ceil(7 / 2))  # 4
        assert cols == 2

    def test_only_cols_set_single_col(self) -> None:
        """cols=1, all plots stacked in one column."""
        config = GridLayoutConfig(cols=1)
        rows, cols = config.auto_size_grid(n_plots=4)
        assert rows == 4
        assert cols == 1


# ---------------------------------------------------------------------------
# ColorScheme.get_colors — uncovered branches
# ---------------------------------------------------------------------------


class TestColorSchemeGroupColorsProvided:
    """Line 340: group_colors dict explicitly set — returned directly."""

    def test_group_colors_returned_directly(self) -> None:
        """When group_colors is provided, get_colors must return it unchanged."""
        explicit = {"A": "#ff0000", "B": "#00ff00", "C": "#0000ff"}
        scheme = ColorScheme(group_colors=explicit)
        result = scheme.get_colors(["A", "B", "C"])
        assert result is explicit

    def test_group_colors_ignores_palette(self) -> None:
        """group_colors takes precedence regardless of the palette value."""
        explicit = {"X": "red", "Y": "blue"}
        scheme = ColorScheme(palette="viridis", group_colors=explicit)
        result = scheme.get_colors(["X", "Y"])
        assert result == explicit

    def test_group_colors_ignores_groups_argument(self) -> None:
        """Even if groups don't match keys, the dict is returned verbatim."""
        explicit = {"P": "#123456"}
        scheme = ColorScheme(group_colors=explicit)
        result = scheme.get_colors(["Q", "R"])  # mismatched keys
        assert result == explicit

    def test_group_colors_empty_dict(self) -> None:
        """An empty group_colors dict is returned as-is (not fallen through)."""
        scheme = ColorScheme(group_colors={})
        result = scheme.get_colors(["A"])
        assert result == {}


class TestColorSchemePaletteAsList:
    """Line 344: palette is a list — used directly as color source."""

    def test_palette_list_used_directly(self) -> None:
        """When palette is a list, those colors are cycled over groups."""
        palette = ["#aaaaaa", "#bbbbbb", "#cccccc"]
        scheme = ColorScheme(palette=palette)
        result = scheme.get_colors(["G1", "G2", "G3"])
        assert result == {"G1": "#aaaaaa", "G2": "#bbbbbb", "G3": "#cccccc"}

    def test_palette_list_cycling(self) -> None:
        """When there are more groups than palette colors, cycling occurs."""
        palette = ["red", "blue"]
        scheme = ColorScheme(palette=palette)
        result = scheme.get_colors(["A", "B", "C"])
        assert result["A"] == "red"
        assert result["B"] == "blue"
        assert result["C"] == "red"  # wraps around

    def test_palette_list_single_color(self) -> None:
        """A single-element palette list cycles for all groups."""
        scheme = ColorScheme(palette=["purple"])
        result = scheme.get_colors(["X", "Y"])
        assert result["X"] == "purple"
        assert result["Y"] == "purple"

    def test_palette_list_returns_dict(self) -> None:
        """get_colors always returns a dict regardless of palette type."""
        scheme = ColorScheme(palette=["green", "orange"])
        result = scheme.get_colors(["only"])
        assert isinstance(result, dict)
        assert len(result) == 1
        assert result["only"] == "green"


# ---------------------------------------------------------------------------
# Regression / smoke tests for auto_size_grid auto mode (already covered but
# confirm the boundary between branches is correct)
# ---------------------------------------------------------------------------


class TestAutoSizeGridAutoMode:
    """No rows or cols — auto square-ish grid (already covered; confirms correctness)."""

    def test_auto_mode_four_plots(self) -> None:
        """4 plots -> 2x2 grid."""
        config = GridLayoutConfig()
        rows, cols = config.auto_size_grid(n_plots=4)
        assert rows * cols >= 4

    def test_auto_mode_single_plot(self) -> None:
        """1 plot -> 1x1 grid."""
        config = GridLayoutConfig()
        rows, cols = config.auto_size_grid(n_plots=1)
        assert rows == 1
        assert cols == 1
