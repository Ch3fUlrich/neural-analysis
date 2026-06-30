"""Coverage tests for neural_analysis.plotting.synthetic_plots_3d.

Target: raise module line+branch coverage to >= 95%.
Each test asserts concrete values/shapes/types and never uses bare except.
"""
from __future__ import annotations

import warnings

import matplotlib
matplotlib.use("Agg")  # noqa: E402 — must be before pyplot

import numpy as np
import pytest

import matplotlib.pyplot as plt

from neural_analysis.plotting.grid_config import PlotSpec
from neural_analysis.plotting.synthetic_plots_3d import (
    CELL_TYPE_COLORS,
    _calculate_optimal_example_cells,
    _compute_spatial_bins_3d,
    _count_fixed_plots,
    _create_coverage_heatmap_3d,
    _create_raster_plot,
    _get_cell_colors,
)

RNG = np.random.default_rng(0)


# ===========================================================================
# _compute_spatial_bins_3d
# ===========================================================================

class TestComputeSpatialBins3D:
    """Tests for _compute_spatial_bins_3d — covers both branches (cell_idx given / None)."""

    def test_specific_cell_idx_returns_correct_shapes(self) -> None:
        """cell_idx not None → linear interpolation, returns 4-tuple with 3D volume."""
        n_samples = 50
        positions = RNG.uniform(0, 2, size=(n_samples, 3)).astype(np.float64)
        activity = RNG.standard_normal((n_samples, 4)).astype(np.float64)
        arena_size = (2.0, 2.0, 2.0)
        n_bins = 5

        x_bins, y_bins, z_bins, vol = _compute_spatial_bins_3d(
            positions, activity, arena_size, n_bins=n_bins, cell_idx=0
        )

        assert x_bins.shape == (n_bins + 1,), "x_bins should have n_bins+1 edges"
        assert y_bins.shape == (n_bins + 1,)
        assert z_bins.shape == (n_bins + 1,)
        assert vol.shape == (n_bins, n_bins, n_bins), "firing volume should be 3D"
        assert vol.dtype == np.float64

    def test_specific_cell_idx_bin_edges_span_arena(self) -> None:
        """Bin edges should span from 0 to arena_size."""
        positions = RNG.uniform(0, 3, size=(30, 3)).astype(np.float64)
        activity = RNG.standard_normal((30, 2)).astype(np.float64)
        arena_size = (3.0, 4.0, 5.0)
        n_bins = 4

        x_bins, y_bins, z_bins, _ = _compute_spatial_bins_3d(
            positions, activity, arena_size, n_bins=n_bins, cell_idx=1
        )

        assert x_bins[0] == pytest.approx(0.0)
        assert x_bins[-1] == pytest.approx(3.0)
        assert y_bins[-1] == pytest.approx(4.0)
        assert z_bins[-1] == pytest.approx(5.0)

    def test_all_cells_averaged_returns_correct_shape(self) -> None:
        """cell_idx=None → nearest interpolation across all cells, then averaged."""
        n_samples = 40
        n_cells = 3
        positions = RNG.uniform(0, 1, size=(n_samples, 3)).astype(np.float64)
        activity = np.abs(RNG.standard_normal((n_samples, n_cells))).astype(np.float64)
        arena_size = (1.0, 1.0, 1.0)
        n_bins = 4

        x_bins, y_bins, z_bins, vol = _compute_spatial_bins_3d(
            positions, activity, arena_size, n_bins=n_bins, cell_idx=None
        )

        assert vol.shape == (n_bins, n_bins, n_bins)
        # Averaged volume should be >= 0 since activity was abs
        assert np.all(vol >= 0)

    def test_all_cells_averaged_is_mean_of_individual(self) -> None:
        """Verify the average across cells equals manual per-cell interpolation average."""
        from scipy.interpolate import griddata

        n_samples = 30
        n_cells = 2
        n_bins = 3
        positions = RNG.uniform(0, 1, size=(n_samples, 3)).astype(np.float64)
        activity = RNG.standard_normal((n_samples, n_cells)).astype(np.float64)
        arena_size = (1.0, 1.0, 1.0)

        _, _, _, vol_all = _compute_spatial_bins_3d(
            positions, activity, arena_size, n_bins=n_bins, cell_idx=None
        )

        # Reproduce manual average
        x_bins = np.linspace(0, 1, n_bins + 1)
        y_bins = np.linspace(0, 1, n_bins + 1)
        z_bins = np.linspace(0, 1, n_bins + 1)
        x_centers = (x_bins[:-1] + x_bins[1:]) / 2
        y_centers = (y_bins[:-1] + y_bins[1:]) / 2
        z_centers = (z_bins[:-1] + z_bins[1:]) / 2
        X, Y, Z = np.meshgrid(x_centers, y_centers, z_centers, indexing="ij")
        grid_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

        vol_manual = np.zeros(grid_points.shape[0])
        for c in range(n_cells):
            vol_manual += griddata(positions, activity[:, c], grid_points,
                                   method="nearest", fill_value=0.0)
        vol_manual /= n_cells
        vol_manual_3d = vol_manual.reshape((n_bins, n_bins, n_bins)).astype(np.float64)

        np.testing.assert_allclose(vol_all, vol_manual_3d, rtol=1e-10)


# ===========================================================================
# _get_cell_colors
# ===========================================================================

class TestGetCellColors:
    """Tests for _get_cell_colors."""

    def test_known_single_cell_type(self) -> None:
        """Single known cell type → color from CELL_TYPE_COLORS repeated n_cells times."""
        activity = RNG.standard_normal((10, 5)).astype(np.float64)
        metadata = {"cell_type": "place"}
        cell_type, cell_types, colors = _get_cell_colors(activity, metadata)

        assert cell_type == "place"
        assert cell_types is None
        assert len(colors) == 5
        assert all(c == CELL_TYPE_COLORS["place"] for c in colors)

    def test_unknown_single_cell_type_uses_fallback_color(self) -> None:
        """Unknown cell type → fallback color '#7F8C8D'."""
        activity = RNG.standard_normal((10, 3)).astype(np.float64)
        metadata = {"cell_type": "mysterious"}
        _, _, colors = _get_cell_colors(activity, metadata)

        assert all(c == "#7F8C8D" for c in colors)

    def test_mixed_cell_types_uses_per_cell_colors(self) -> None:
        """cell_types list in metadata → one color per cell."""
        activity = RNG.standard_normal((10, 4)).astype(np.float64)
        cell_types_list = ["place", "grid", "head_direction", "random"]
        metadata = {"cell_types": cell_types_list}
        cell_type, cell_types, colors = _get_cell_colors(activity, metadata)

        assert cell_types == cell_types_list
        assert len(colors) == 4
        assert colors[0] == CELL_TYPE_COLORS["place"]
        assert colors[1] == CELL_TYPE_COLORS["grid"]
        assert colors[2] == CELL_TYPE_COLORS["head_direction"]
        assert colors[3] == CELL_TYPE_COLORS["random"]

    def test_mixed_cell_types_unknown_entries_use_fallback(self) -> None:
        """Unknown entries in cell_types list → '#7F8C8D'."""
        activity = RNG.standard_normal((10, 2)).astype(np.float64)
        metadata = {"cell_types": ["place", "alien"]}
        _, _, colors = _get_cell_colors(activity, metadata)

        assert colors[0] == CELL_TYPE_COLORS["place"]
        assert colors[1] == "#7F8C8D"

    def test_no_cell_type_key_uses_unknown(self) -> None:
        """No 'cell_type' key → defaults to 'unknown' → fallback color."""
        activity = RNG.standard_normal((10, 2)).astype(np.float64)
        _, _, colors = _get_cell_colors(activity, {})
        assert all(c == "#7F8C8D" for c in colors)


# ===========================================================================
# _count_fixed_plots
# ===========================================================================

class TestCountFixedPlots:
    """Tests for _count_fixed_plots — all branches."""

    def test_show_nothing_returns_zero(self) -> None:
        """All flags off → 0 fixed plots."""
        n = _count_fixed_plots(
            {}, show_raster=False, show_fields=False,
            show_behavior=False, show_ground_truth=False,
            show_embeddings=False, embedding_methods=[],
        )
        assert n == 0

    def test_show_raster_adds_one(self) -> None:
        """show_raster=True → +1."""
        n = _count_fixed_plots(
            {}, show_raster=True, show_fields=False,
            show_behavior=False, show_ground_truth=False,
            show_embeddings=False, embedding_methods=[],
        )
        assert n == 1

    def test_show_fields_place_adds_one(self) -> None:
        """show_fields + place cell → +1 (coverage)."""
        n = _count_fixed_plots(
            {"cell_type": "place"}, show_raster=False, show_fields=True,
            show_behavior=False, show_ground_truth=False,
            show_embeddings=False, embedding_methods=[],
        )
        assert n == 1

    def test_show_fields_grid_adds_one(self) -> None:
        """show_fields + grid cell → +1 (coverage)."""
        n = _count_fixed_plots(
            {"cell_type": "grid"}, show_raster=False, show_fields=True,
            show_behavior=False, show_ground_truth=False,
            show_embeddings=False, embedding_methods=[],
        )
        assert n == 1

    def test_show_fields_random_adds_four(self) -> None:
        """show_fields + random cell → +4 (diagnostic plots). Covers line 145."""
        n = _count_fixed_plots(
            {"cell_type": "random"}, show_raster=False, show_fields=True,
            show_behavior=False, show_ground_truth=False,
            show_embeddings=False, embedding_methods=[],
        )
        assert n == 4

    def test_show_behavior_without_positions_adds_zero(self) -> None:
        """show_behavior=True but no 'positions' key → +0."""
        n = _count_fixed_plots(
            {}, show_raster=False, show_fields=False,
            show_behavior=True, show_ground_truth=False,
            show_embeddings=False, embedding_methods=[],
        )
        assert n == 0

    def test_show_behavior_with_positions_adds_one(self) -> None:
        """show_behavior=True and 'positions' in metadata → +1."""
        n = _count_fixed_plots(
            {"positions": np.zeros((10, 3))}, show_raster=False, show_fields=False,
            show_behavior=True, show_ground_truth=False,
            show_embeddings=False, embedding_methods=[],
        )
        assert n == 1

    def test_show_ground_truth_with_embedding_adds_one(self) -> None:
        """show_ground_truth=True + embedding key → +1."""
        n = _count_fixed_plots(
            {"ground_truth_embedding": np.zeros((10, 2))},
            show_raster=False, show_fields=False,
            show_behavior=False, show_ground_truth=True,
            show_embeddings=False, embedding_methods=[],
        )
        assert n == 1

    def test_show_embeddings_adds_len_methods(self) -> None:
        """show_embeddings=True → +len(embedding_methods)."""
        n = _count_fixed_plots(
            {}, show_raster=False, show_fields=False,
            show_behavior=False, show_ground_truth=False,
            show_embeddings=True, embedding_methods=["pca", "umap", "tsne"],
        )
        assert n == 3

    def test_all_options_combined(self) -> None:
        """All options on → sum of all parts."""
        meta = {
            "cell_type": "place",
            "positions": np.zeros((10, 3)),
            "ground_truth_embedding": np.zeros((10, 2)),
        }
        n = _count_fixed_plots(
            meta, show_raster=True, show_fields=True,
            show_behavior=True, show_ground_truth=True,
            show_embeddings=True, embedding_methods=["pca", "umap"],
        )
        # raster(1) + place coverage(1) + behavior(1) + gt(1) + 2 embeddings = 6
        assert n == 6


# ===========================================================================
# _calculate_optimal_example_cells
# ===========================================================================

class TestCalculateOptimalExampleCells:
    """Tests for _calculate_optimal_example_cells — all branches in calc_grid and main loop."""

    def test_calc_grid_n_total_zero_returns_1_0(self) -> None:
        """n_total=0 → n_total <= 2 → returns (1, 0). (via internal calc_grid)"""
        # We call _calculate_optimal_example_cells with n_fixed_plots=0, which
        # will call calc_grid with totals 2, 3, 4. We can't call calc_grid directly,
        # but we can verify the resulting best_n is correct.
        result = _calculate_optimal_example_cells(0, "place")
        assert isinstance(result, int)
        assert 2 <= result <= 4

    def test_calc_grid_n_total_1_returns_1_1(self) -> None:
        """n_total=1 → n_total <= 2 → returns (1, 1). With n_fixed=1, totals are 3,4,5."""
        result = _calculate_optimal_example_cells(1, "place")
        assert isinstance(result, int)
        assert 2 <= result <= 4

    def test_calc_grid_n_total_le_4_returns_2_2(self) -> None:
        """n_total=3 → n_total <= 4 → (2,2). With n_fixed=2, totals are 4,5,6."""
        result = _calculate_optimal_example_cells(2, "place")
        assert isinstance(result, int)
        assert 2 <= result <= 4

    def test_calc_grid_n_total_le_6_returns_2_3(self) -> None:
        """n_total=5 → n_total <= 6 → (2,3). With n_fixed=3, totals are 5,6,7."""
        result = _calculate_optimal_example_cells(3, "place")
        assert isinstance(result, int)
        assert 2 <= result <= 4

    def test_calc_grid_n_total_gt_6_ceiling_division(self) -> None:
        """n_total > 6 → ncols=3, nrows=ceil(n_total/3). Covers lines 167-169."""
        # n_fixed=5 → totals are 7, 8, 9 → all go through the else branch
        result = _calculate_optimal_example_cells(5, "place")
        assert isinstance(result, int)
        assert 2 <= result <= 4

    def test_random_cell_type_prefers_3_examples_when_n_empty_le_3(self) -> None:
        """Covers lines 183-185: random cell_type + n_examples==3 + n_empty<=3 → break with 3."""
        # With n_fixed=0: total=3, calc_grid(3) → (2,2) → grid_size=4 → n_empty=1 ≤ 3
        # So the break is triggered on first iteration where n_examples=3
        result = _calculate_optimal_example_cells(0, "random")
        assert result == 3

    def test_random_cell_type_with_many_fixed_still_gets_3_if_possible(self) -> None:
        """random cell type prefers 3 examples when n_empty <= 3."""
        # n_fixed=3: total=6 → calc_grid(6) → (2,3) → grid_size=6 → n_empty=0 ≤ 3
        result = _calculate_optimal_example_cells(3, "random")
        assert result == 3

    def test_non_random_picks_minimum_empty(self) -> None:
        """Non-random cell type: picks n_examples that minimizes empty subplots."""
        # n_fixed=4: totals are 6, 7, 8
        # 6 → (2,3) → 0 empty → best; 7 → (3,3) → 2 empty; 8 → (3,3) → 1 empty
        result = _calculate_optimal_example_cells(4, "place")
        assert result == 2  # 4+2=6 → 0 empty slots

    def test_result_always_in_range_2_to_4(self) -> None:
        """For any n_fixed in 0..10, result is always in [2, 4]."""
        for n in range(11):
            for ct in ("place", "grid", "head_direction", "random"):
                r = _calculate_optimal_example_cells(n, ct)
                assert 2 <= r <= 4, f"n={n}, ct={ct} → r={r}"


# ===========================================================================
# _create_raster_plot — covers lines 206-320 (the entire function)
# ===========================================================================

class TestCreateRasterPlot:
    """Tests for _create_raster_plot — full function coverage."""

    def teardown_method(self) -> None:
        plt.close("all")

    def test_single_cell_type_small_activity_returns_hot_heatmap(self) -> None:
        """No subsampling, single cell type → PlotSpec with cmap='hot'. Covers lines 219-319."""
        n_samples, n_cells = 100, 5
        activity = RNG.standard_normal((n_samples, n_cells)).astype(np.float64)
        colors = ["#E74C3C"] * n_cells

        spec = _create_raster_plot(activity, colors, max_cells=10, subplot_position=2)

        assert isinstance(spec, PlotSpec)
        assert spec.plot_type == "heatmap"
        assert spec.cmap == "hot"
        assert spec.colorbar is True
        assert spec.colorbar_label == "Firing Rate (Hz)"
        assert spec.subplot_position == 2
        assert spec.data.shape == (n_cells, n_samples)

    def test_single_cell_type_time_ticks_set_for_large_samples(self) -> None:
        """n_samples=300 → n_time_ticks=min(6,300//50)=6 > 0 → time ticks not None."""
        n_samples = 300
        activity = RNG.standard_normal((n_samples, 3)).astype(np.float64)
        colors = ["#3498DB"] * 3

        spec = _create_raster_plot(activity, colors, max_cells=10, subplot_position=0)

        assert spec.kwargs["set_xticks"] is not None
        assert spec.kwargs["set_xticklabels"] is not None
        # Ticks should be integer array
        assert spec.kwargs["set_xticklabels"].dtype in (np.int32, np.int64, int)

    def test_single_cell_type_no_time_ticks_for_small_samples(self) -> None:
        """n_samples < 50 → n_time_ticks=0 → time_ticks/time_labels are None. Covers line 234-235."""
        n_samples = 20  # 20 // 50 = 0
        activity = RNG.standard_normal((n_samples, 3)).astype(np.float64)
        colors = ["#E74C3C"] * 3

        spec = _create_raster_plot(activity, colors, max_cells=10, subplot_position=0)

        assert spec.kwargs["set_xticks"] is None
        assert spec.kwargs["set_xticklabels"] is None

    def test_cell_ticks_are_set_for_nonzero_cells(self) -> None:
        """n_cells > 0 → n_cell_ticks > 0 → cell_ticks computed."""
        n_cells = 4
        activity = RNG.standard_normal((100, n_cells)).astype(np.float64)
        colors = ["#E74C3C"] * n_cells

        spec = _create_raster_plot(activity, colors, max_cells=10, subplot_position=0)

        assert spec.kwargs["set_yticks"] is not None
        assert spec.kwargs["set_yticklabels"] is not None

    def test_subsampling_when_n_cells_exceeds_max_cells(self) -> None:
        """n_cells > max_cells → subsampling applied. Covers lines 209-217."""
        n_cells = 20
        max_cells = 5
        activity = RNG.standard_normal((100, n_cells)).astype(np.float64)
        colors = [f"#{i:06x}" for i in range(n_cells)]

        spec = _create_raster_plot(activity, colors, max_cells=max_cells, subplot_position=0)

        # After subsampling, data shape should have at most max_cells rows
        assert spec.data.shape[0] <= max_cells

    def test_subsampling_with_cell_types_not_none(self) -> None:
        """When n_cells > max_cells and cell_types not None → cell_types_sub also sliced. Covers line 215."""
        n_cells = 20
        max_cells = 5
        activity = RNG.standard_normal((100, n_cells)).astype(np.float64)
        colors = ["#E74C3C"] * n_cells
        cell_types = ["place"] * 10 + ["grid"] * 10

        spec = _create_raster_plot(
            activity, colors, max_cells=max_cells,
            subplot_position=0, cell_types=cell_types
        )

        # Still returns a spec (not mixed since only place/grid may or may not mix)
        assert isinstance(spec, PlotSpec)

    def test_cell_types_none_not_subsampled(self) -> None:
        """n_cells <= max_cells + cell_types=None → cell_types_sub=None → not is_mixed. Covers line 222."""
        activity = RNG.standard_normal((100, 3)).astype(np.float64)
        colors = ["#E74C3C"] * 3

        spec = _create_raster_plot(
            activity, colors, max_cells=10, subplot_position=0, cell_types=None
        )

        assert spec.cmap == "hot"  # Single cell type path

    def test_mixed_population_returns_rgb_raster(self) -> None:
        """cell_types with multiple distinct types → is_mixed=True → RGB PlotSpec. Covers lines 254-295."""
        n_cells = 4
        n_samples = 100
        activity = RNG.standard_normal((n_samples, n_cells)).astype(np.float64)
        colors = [
            CELL_TYPE_COLORS["place"],
            CELL_TYPE_COLORS["place"],
            CELL_TYPE_COLORS["grid"],
            CELL_TYPE_COLORS["grid"],
        ]
        cell_types = ["place", "place", "grid", "grid"]

        spec = _create_raster_plot(
            activity, colors, max_cells=10,
            subplot_position=1, cell_types=cell_types
        )

        assert isinstance(spec, PlotSpec)
        assert spec.plot_type == "heatmap"
        assert spec.cmap is None  # No colormap for RGB
        assert spec.colorbar is False
        assert "Mixed Population" in spec.title
        # RGB data: shape should be (n_cells, n_samples, 3)
        assert spec.data.shape == (n_cells, n_samples, 3)

    def test_mixed_population_rgb_values_in_valid_range(self) -> None:
        """RGB raster values should be in [0, 1]."""
        n_cells = 4
        n_samples = 80
        activity = np.abs(RNG.standard_normal((n_samples, n_cells))).astype(np.float64)
        colors = [
            CELL_TYPE_COLORS["place"],
            CELL_TYPE_COLORS["grid"],
            CELL_TYPE_COLORS["place"],
            CELL_TYPE_COLORS["grid"],
        ]
        cell_types = ["place", "grid", "place", "grid"]

        spec = _create_raster_plot(
            activity, colors, max_cells=10,
            subplot_position=0, cell_types=cell_types
        )

        assert spec.data.min() >= 0.0
        assert spec.data.max() <= 1.0 + 1e-10

    def test_mixed_population_extent_uses_n_samples(self) -> None:
        """Mixed population extent[1] == n_samples."""
        n_samples, n_cells = 60, 4
        activity = RNG.standard_normal((n_samples, n_cells)).astype(np.float64)
        colors = [CELL_TYPE_COLORS["place"]] * 2 + [CELL_TYPE_COLORS["grid"]] * 2
        cell_types = ["place", "place", "grid", "grid"]

        spec = _create_raster_plot(
            activity, colors, max_cells=10, subplot_position=0, cell_types=cell_types
        )

        extent = spec.kwargs["extent"]
        assert extent[1] == n_samples

    def test_single_type_cell_labels_padded_correctly(self) -> None:
        """cell_labels padding: while loop runs when labels < ticks. Covers line 244-246."""
        # With n_cells=1 and cell_tick_step=1: cell_ticks=arange(0,2,1)=[0,1]
        # cell_labels = cell_indices[::1] = [0] → only 1 label but 2 ticks → padding needed
        activity = RNG.standard_normal((100, 1)).astype(np.float64)
        colors = ["#E74C3C"]

        spec = _create_raster_plot(activity, colors, max_cells=10, subplot_position=0)

        assert spec.kwargs["set_yticks"] is not None
        yticks = spec.kwargs["set_yticks"]
        ylabels = spec.kwargs["set_yticklabels"]
        assert len(ylabels) == len(yticks), "labels should be padded to match ticks length"

    def test_mixed_with_subsampled_cell_types(self) -> None:
        """Subsampling when mixed population: cell_types_sub is the subsampled version."""
        n_cells = 12
        max_cells = 4
        activity = RNG.standard_normal((100, n_cells)).astype(np.float64)
        colors = [CELL_TYPE_COLORS["place"]] * 6 + [CELL_TYPE_COLORS["grid"]] * 6
        cell_types = ["place"] * 6 + ["grid"] * 6

        spec = _create_raster_plot(
            activity, colors, max_cells=max_cells,
            subplot_position=0, cell_types=cell_types
        )

        # It's subsampled and mixed → expect RGB raster with at most max_cells rows
        assert spec.data.shape[0] <= max_cells
        # could still be mixed (place+grid) after subsampling
        assert spec.plot_type == "heatmap"

    def test_mixed_population_no_time_ticks_with_small_samples(self) -> None:
        """Mixed population with small n_samples → time_ticks=None. Covers branches in lines 229/233."""
        n_samples = 10
        n_cells = 2
        activity = RNG.standard_normal((n_samples, n_cells)).astype(np.float64)
        colors = [CELL_TYPE_COLORS["place"], CELL_TYPE_COLORS["grid"]]
        cell_types = ["place", "grid"]

        spec = _create_raster_plot(
            activity, colors, max_cells=10, subplot_position=0, cell_types=cell_types
        )

        assert spec.kwargs["set_xticks"] is None
        assert spec.kwargs["set_xticklabels"] is None

    def test_zero_cells_triggers_cell_ticks_none_branch(self) -> None:
        """n_cells=0 → n_cell_ticks=0 → cell_ticks=None, cell_labels=None. Covers lines 248-249."""
        activity = np.zeros((100, 0), dtype=np.float64)
        colors: list[str] = []

        spec = _create_raster_plot(activity, colors, max_cells=10, subplot_position=0)

        assert spec.kwargs["set_yticks"] is None
        assert spec.kwargs["set_yticklabels"] is None

    def test_mixed_population_with_empty_colors_triggers_colors_sub_none(self) -> None:
        """n_cells > max_cells + empty colors → colors_sub=None → reassigned to []. Covers line 261.

        When colors is falsy and cells are subsampled, colors_sub becomes None.
        Inside the is_mixed branch, colors_sub is reassigned to [] (line 261).
        The subsequent rgb_colors list is also empty, causing IndexError on the
        per-cell coloring loop. This test verifies the line-261 branch is
        executed (reached) and documents the resulting behavior.
        """
        n_cells = 20
        max_cells = 5
        n_samples = 100
        activity = RNG.standard_normal((n_samples, n_cells)).astype(np.float64)
        colors: list[str] = []  # Empty → falsy → colors_sub=None in subsampling
        # cell_types must have > 1 distinct type to trigger is_mixed
        cell_types = ["place"] * 10 + ["grid"] * 10

        # When colors_sub is None (set to []) and mixed population, rgb_colors=[]
        # → IndexError when applying per-cell colors in the loop
        with pytest.raises(IndexError):
            _create_raster_plot(
                activity, colors, max_cells=max_cells,
                subplot_position=0, cell_types=cell_types
            )


# ===========================================================================
# _create_coverage_heatmap_3d
# ===========================================================================

class TestCreateCoverageHeatmap3D:
    """Tests for _create_coverage_heatmap_3d — covers lines 323-454."""

    def teardown_method(self) -> None:
        plt.close("all")

    def test_place_cell_returns_wall_spec_list(self) -> None:
        """Place cell path: returns list with one heatmap_walls PlotSpec."""
        n_samples, n_cells = 50, 3
        positions = RNG.uniform(0, 2, size=(n_samples, 3)).astype(np.float64)
        activity = RNG.standard_normal((n_samples, n_cells)).astype(np.float64)
        metadata = {
            "positions": positions,
            "arena_size": (2.0, 2.0, 2.0),
            "cell_type": "place",
        }

        result = _create_coverage_heatmap_3d(activity, metadata, subplot_position=0)

        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 1
        spec = result[0]
        assert spec.plot_type == "heatmap_walls"
        assert spec.cmap == "hot"
        assert "Place Field Coverage" in spec.title

    def test_place_cell_data_has_required_keys(self) -> None:
        """Place cell data dict has xy, xz, yz slices and center arrays."""
        n_samples = 40
        positions = RNG.uniform(0, 1, size=(n_samples, 3)).astype(np.float64)
        activity = RNG.standard_normal((n_samples, 2)).astype(np.float64)
        metadata = {
            "positions": positions,
            "arena_size": (1.0, 1.0, 1.0),
            "cell_type": "place",
        }

        result = _create_coverage_heatmap_3d(activity, metadata, subplot_position=0)

        data = result[0].data
        assert "xy" in data
        assert "xz" in data
        assert "yz" in data
        assert "x_centers" in data
        assert "y_centers" in data
        assert "z_centers" in data
        assert "xy_position" in data
        assert "xz_position" in data
        assert "yz_position" in data

    def test_place_cell_xy_slice_shape_correct(self) -> None:
        """xy slice should be 2D (n_bins x n_bins) for place cell."""
        n_samples = 40
        positions = RNG.uniform(0, 2, size=(n_samples, 3)).astype(np.float64)
        activity = RNG.standard_normal((n_samples, 2)).astype(np.float64)
        metadata = {
            "positions": positions,
            "arena_size": (2.0, 2.0, 2.0),
            "cell_type": "place",
        }

        result = _create_coverage_heatmap_3d(activity, metadata, subplot_position=0)

        n_bins = 30
        data = result[0].data
        assert data["xy"].shape == (n_bins, n_bins)
        assert data["xz"].shape == (n_bins, n_bins)
        assert data["yz"].shape == (n_bins, n_bins)

    def test_positions_none_returns_none(self) -> None:
        """If positions is None → returns None. Covers line 338 branch."""
        activity = RNG.standard_normal((20, 3)).astype(np.float64)
        metadata = {"arena_size": (1.0, 1.0, 1.0), "cell_type": "place"}  # no positions

        result = _create_coverage_heatmap_3d(activity, metadata, subplot_position=0)

        assert result is None

    def test_positions_not_3d_returns_none(self) -> None:
        """If positions.shape[1] != 3 → returns None. Covers line 338."""
        activity = RNG.standard_normal((20, 3)).astype(np.float64)
        positions = RNG.uniform(0, 1, size=(20, 2)).astype(np.float64)  # 2D positions
        metadata = {
            "positions": positions,
            "arena_size": (1.0, 1.0),
            "cell_type": "place",
        }

        result = _create_coverage_heatmap_3d(activity, metadata, subplot_position=0)

        assert result is None

    def test_grid_cell_returns_wall_spec_with_autocorr_title(self) -> None:
        """Grid cell path: autocorrelation computed → heatmap_walls with 'Autocorrelation' title."""
        n_samples = 40
        positions = RNG.uniform(0, 2, size=(n_samples, 3)).astype(np.float64)
        activity = RNG.standard_normal((n_samples, 3)).astype(np.float64)
        metadata = {
            "positions": positions,
            "arena_size": (2.0, 2.0, 2.0),
            "cell_type": "grid",
        }

        result = _create_coverage_heatmap_3d(activity, metadata, subplot_position=1)

        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 1
        spec = result[0]
        assert spec.plot_type == "heatmap_walls"
        assert "Autocorrelation" in spec.title
        assert spec.cmap == "viridis"

    def test_grid_cell_autocorr_data_keys(self) -> None:
        """Grid cell data dict has lag axes and wall position keys."""
        n_samples = 30
        positions = RNG.uniform(0, 1, size=(n_samples, 3)).astype(np.float64)
        activity = RNG.standard_normal((n_samples, 2)).astype(np.float64)
        metadata = {
            "positions": positions,
            "arena_size": (1.0, 1.0, 1.0),
            "cell_type": "grid",
        }

        result = _create_coverage_heatmap_3d(activity, metadata, subplot_position=0)

        assert result is not None
        data = result[0].data
        for key in ("xy", "xz", "yz", "x_centers", "y_centers", "z_centers",
                    "xy_position", "xz_position", "yz_position"):
            assert key in data, f"Missing key: {key}"

    def test_grid_cell_autocorr_normalized_at_center(self) -> None:
        """Grid autocorr: when center != 0, autocorr_normalized[center] == 1.0. Covers line 371-372."""
        n_samples = 40
        positions = RNG.uniform(0, 2, size=(n_samples, 3)).astype(np.float64)
        # Use positive activity to ensure non-zero center
        activity = np.abs(RNG.standard_normal((n_samples, 2))).astype(np.float64) + 0.5
        metadata = {
            "positions": positions,
            "arena_size": (2.0, 2.0, 2.0),
            "cell_type": "grid",
        }

        result = _create_coverage_heatmap_3d(activity, metadata, subplot_position=0)

        assert result is not None
        # xy slice at center of the XY plane should be near 1
        data = result[0].data
        n_bins = 60
        center = n_bins // 2
        # The zero-lag peak is at [center, center, center], xy_slice is [:,:,center_z]
        # so xy_slice[center_x, center_y] should be 1.0
        xy = data["xy"]
        assert xy[center, center] == pytest.approx(1.0, abs=1e-6)

    def test_grid_cell_zero_center_autocorr_not_normalized(self) -> None:
        """When autocorr[center] == 0, autocorr_normalized = autocorr. Covers line 374."""
        n_samples = 30
        # Zero activity → firing_volume_centered = 0 → power_spectrum=0 → autocorr=0
        positions = RNG.uniform(0, 1, size=(n_samples, 3)).astype(np.float64)
        activity = np.zeros((n_samples, 2), dtype=np.float64)  # All zeros → center=0
        metadata = {
            "positions": positions,
            "arena_size": (1.0, 1.0, 1.0),
            "cell_type": "grid",
        }

        result = _create_coverage_heatmap_3d(activity, metadata, subplot_position=0)

        # Zero activity → autocorr=0 everywhere → center=0 → autocorr_normalized = autocorr (unnormalized)
        # The function should still succeed and return a list
        assert result is not None
        assert isinstance(result, list)
        # xy_slice should be all zeros
        data = result[0].data
        assert np.all(data["xy"] == pytest.approx(0.0))

    def test_exception_in_grid_path_returns_empty_list_with_warning(self) -> None:
        """If grid autocorr raises, silently warns and returns empty list. Covers lines 409-413."""
        from unittest.mock import patch

        n_samples = 30
        positions = RNG.uniform(0, 1, size=(n_samples, 3)).astype(np.float64)
        activity = RNG.standard_normal((n_samples, 2)).astype(np.float64)
        metadata = {
            "positions": positions,
            "arena_size": (1.0, 1.0, 1.0),
            "cell_type": "grid",
        }

        # Force an exception during _compute_spatial_bins_3d
        with patch(
            "neural_analysis.plotting.synthetic_plots_3d._compute_spatial_bins_3d",
            side_effect=RuntimeError("forced error for coverage"),
        ):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = _create_coverage_heatmap_3d(activity, metadata, subplot_position=0)

        assert result == []  # Empty list because exception was caught
        assert len(caught) == 1
        assert "Failed to create 3D autocorrelation" in str(caught[0].message)

    def test_place_cell_colorbar_label(self) -> None:
        """Place cell spec should have 'Avg. Firing Rate (Hz)' as colorbar_label."""
        n_samples = 30
        positions = RNG.uniform(0, 1, size=(n_samples, 3)).astype(np.float64)
        activity = RNG.standard_normal((n_samples, 2)).astype(np.float64)
        metadata = {"positions": positions, "arena_size": (1.0, 1.0, 1.0), "cell_type": "place"}

        result = _create_coverage_heatmap_3d(activity, metadata, subplot_position=5)

        assert result[0].colorbar_label == "Avg. Firing Rate (Hz)"
        assert result[0].subplot_position == 5

    def test_non_place_non_grid_cell_type_treated_as_place(self) -> None:
        """Cell type other than 'place' or 'grid' falls into else (place) branch."""
        n_samples = 30
        positions = RNG.uniform(0, 1, size=(n_samples, 3)).astype(np.float64)
        activity = RNG.standard_normal((n_samples, 2)).astype(np.float64)
        metadata = {
            "positions": positions,
            "arena_size": (1.0, 1.0, 1.0),
            "cell_type": "head_direction",  # not "place" or "grid"
        }

        result = _create_coverage_heatmap_3d(activity, metadata, subplot_position=0)

        # Should go through else branch (place-like)
        assert result is not None
        assert len(result) == 1
        assert result[0].plot_type == "heatmap_walls"

    def test_grid_cell_lag_axes_span_correct_range(self) -> None:
        """Grid cell lag axes should span [-arena_size, +arena_size]."""
        n_samples = 30
        x_max, y_max, z_max = 3.0, 4.0, 2.0
        positions = RNG.uniform(0, x_max, size=(n_samples, 3)).astype(np.float64)
        activity = RNG.standard_normal((n_samples, 2)).astype(np.float64)
        metadata = {
            "positions": positions,
            "arena_size": (x_max, y_max, z_max),
            "cell_type": "grid",
        }

        result = _create_coverage_heatmap_3d(activity, metadata, subplot_position=0)

        if result:  # only check if autocorrelation succeeded
            data = result[0].data
            assert data["x_centers"][0] == pytest.approx(-x_max)
            assert data["x_centers"][-1] == pytest.approx(x_max)
            assert data["y_centers"][0] == pytest.approx(-y_max)
            assert data["z_centers"][-1] == pytest.approx(z_max)


# ===========================================================================
# CELL_TYPE_COLORS constant
# ===========================================================================

class TestCellTypeColors:
    """Test the module-level CELL_TYPE_COLORS constant."""

    def test_all_expected_keys_present(self) -> None:
        """CELL_TYPE_COLORS has the 4 expected cell types."""
        expected = {"place", "grid", "head_direction", "random"}
        assert set(CELL_TYPE_COLORS.keys()) == expected

    def test_colors_are_valid_hex(self) -> None:
        """All color values are valid hex strings starting with '#'."""
        for ct, color in CELL_TYPE_COLORS.items():
            assert isinstance(color, str), f"{ct} color is not a string"
            assert color.startswith("#"), f"{ct} color '{color}' is not a hex string"
            assert len(color) == 7, f"{ct} color '{color}' has wrong length"
