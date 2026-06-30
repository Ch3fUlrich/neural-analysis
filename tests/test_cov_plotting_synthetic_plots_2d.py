"""Coverage tests for neural_analysis.plotting.synthetic_plots_2d.

Target: raise module line+branch coverage to >= 95%.

Each test asserts concrete return values / shapes / dtypes / numeric results
or expected errors. No bare try/except, no assert True, no assert x is not None.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # noqa: E402

import numpy as np
import pytest

from neural_analysis.plotting.grid_config import PlotSpec
from neural_analysis.plotting.synthetic_plots_2d import (
    _compute_spatial_bins_2d,
    _compute_radial_power_spectrum,
    _create_coverage_heatmap,
    _create_example_cell_heatmaps,
    _create_random_diagnostics,
)

import matplotlib.pyplot as plt

# Seeded RNG for reproducibility
RNG = np.random.default_rng(0)


# ===========================================================================
# _compute_spatial_bins_2d
# ===========================================================================

class TestComputeSpatialBins2D:
    """Test both the tuple and scalar arena_size paths."""

    def test_tuple_arena_size_output_shapes(self) -> None:
        """Tuple arena_size → x_bins, y_bins have n_bins+1 elements; map is (n_bins, n_bins)."""
        n = 50
        positions = RNG.uniform(0, 2, size=(n, 2))
        activity = RNG.standard_normal((n, 4))
        x_bins, y_bins, firing_map = _compute_spatial_bins_2d(
            positions, activity, arena_size=(2.0, 3.0), n_bins=5
        )
        assert x_bins.shape == (6,)
        assert y_bins.shape == (6,)
        assert firing_map.shape == (5, 5)
        assert x_bins.dtype == np.float64
        assert y_bins.dtype == np.float64
        # x_bins edges go from 0 to 2.0
        assert float(x_bins[0]) == pytest.approx(0.0)
        assert float(x_bins[-1]) == pytest.approx(2.0)
        # y_bins edges go from 0 to 3.0
        assert float(y_bins[0]) == pytest.approx(0.0)
        assert float(y_bins[-1]) == pytest.approx(3.0)

    def test_scalar_arena_size_uses_same_for_x_and_y(self) -> None:
        """Scalar arena_size → x_max == y_max == that scalar (line 45 covered)."""
        n = 100
        positions = RNG.uniform(0, 4, size=(n, 2))
        activity = RNG.standard_normal((n, 3))
        x_bins, y_bins, firing_map = _compute_spatial_bins_2d(
            positions, activity, arena_size=4.0, n_bins=4
        )
        assert firing_map.shape == (4, 4)
        # Both axes should end at 4.0
        assert float(x_bins[-1]) == pytest.approx(4.0)
        assert float(y_bins[-1]) == pytest.approx(4.0)
        # x_bins and y_bins should be identical when arena is square
        np.testing.assert_array_almost_equal(x_bins, y_bins)

    def test_cell_idx_selects_single_cell(self) -> None:
        """cell_idx is not None → only that cell's activity binned (line 65)."""
        n = 200
        positions = RNG.uniform(0, 1, size=(n, 2))
        # Make cell 0 always 1.0, cell 1 always 10.0
        activity = np.ones((n, 2))
        activity[:, 1] = 10.0
        _, _, map_cell0 = _compute_spatial_bins_2d(
            positions, activity, arena_size=(1.0, 1.0), n_bins=3, cell_idx=0
        )
        _, _, map_cell1 = _compute_spatial_bins_2d(
            positions, activity, arena_size=(1.0, 1.0), n_bins=3, cell_idx=1
        )
        # Cell 0 map should be ~1.0, cell 1 map should be ~10.0 in visited bins
        assert map_cell0[map_cell0 > 0].mean() == pytest.approx(1.0, abs=1e-6)
        assert map_cell1[map_cell1 > 0].mean() == pytest.approx(10.0, abs=1e-6)

    def test_no_cell_idx_averages_all_cells(self) -> None:
        """cell_idx is None → average across all cells (line 67)."""
        n = 200
        positions = RNG.uniform(0, 1, size=(n, 2))
        activity = np.ones((n, 2))
        activity[:, 0] = 2.0
        activity[:, 1] = 8.0
        _, _, firing_map = _compute_spatial_bins_2d(
            positions, activity, arena_size=(1.0, 1.0), n_bins=3, cell_idx=None
        )
        # Mean of [2.0, 8.0] = 5.0 in visited bins
        visited = firing_map[firing_map > 0]
        assert visited.mean() == pytest.approx(5.0, abs=1e-6)

    def test_empty_bin_stays_zero(self) -> None:
        """Bins with no samples stay at 0.0 (mask.sum() == 0 branch)."""
        # All positions in the lower-left quarter only
        positions = RNG.uniform(0, 0.4, size=(50, 2))
        activity = RNG.standard_normal((50, 2))
        _, _, firing_map = _compute_spatial_bins_2d(
            positions, activity, arena_size=(1.0, 1.0), n_bins=3
        )
        # Bins covering [0.67..1.0] range have no samples → should be 0
        assert firing_map[2, 2] == 0.0


# ===========================================================================
# _compute_radial_power_spectrum
# ===========================================================================

class TestComputeRadialPowerSpectrum:
    """Test radial power spectrum computation."""

    def test_output_length_matches_n_bins(self) -> None:
        """Return array has exactly n_bins elements."""
        ps = RNG.uniform(0, 1, size=(20, 20))
        profile = _compute_radial_power_spectrum(ps, n_bins=10)
        assert profile.shape == (10,)

    def test_non_negative_values(self) -> None:
        """All radial profile values are non-negative for non-negative input."""
        ps = RNG.uniform(0, 1, size=(30, 30))
        profile = _compute_radial_power_spectrum(ps, n_bins=15)
        assert np.all(profile >= 0)

    def test_center_bin_is_max_for_central_peak(self) -> None:
        """Center bin is highest when spectrum has a central peak."""
        ps = np.zeros((11, 11))
        ps[5, 5] = 100.0  # spike at center
        profile = _compute_radial_power_spectrum(ps, n_bins=5)
        # Bin 0 (r==0) should contain the center spike
        assert profile[0] == pytest.approx(100.0)

    def test_rectangular_spectrum(self) -> None:
        """Non-square spectrum: radial distances computed correctly."""
        ps = RNG.uniform(0, 1, size=(10, 20))
        profile = _compute_radial_power_spectrum(ps, n_bins=5)
        assert profile.shape == (5,)
        # Bins with no samples stay 0 (but that is fine – just check shape)
        assert profile.dtype == np.float64


# ===========================================================================
# _create_coverage_heatmap
# ===========================================================================

class TestCreateCoverageHeatmap:
    """Cover the grid-cell path and scalar arena_size path."""

    def test_returns_none_when_positions_is_none(self) -> None:
        """positions=None in metadata → return None."""
        activity = RNG.standard_normal((50, 5))
        metadata = {"cell_type": "place"}
        result = _create_coverage_heatmap(activity, metadata, subplot_position=0)
        assert result is None
        plt.close("all")

    def test_returns_none_when_positions_not_2d(self) -> None:
        """positions with shape[1] != 2 → return None (line 116-117)."""
        activity = RNG.standard_normal((50, 5))
        metadata = {
            "cell_type": "place",
            "positions": RNG.uniform(0, 1, size=(50, 3)),  # 3D, not 2D
        }
        result = _create_coverage_heatmap(activity, metadata, subplot_position=0)
        assert result is None
        plt.close("all")

    def test_place_cell_returns_plotspec_with_hot_cmap(self) -> None:
        """place cell → cmap='hot', title='Place Field Coverage'."""
        n = 100
        positions = RNG.uniform(0, 2, size=(n, 2))
        activity = RNG.standard_normal((n, 5))
        metadata = {
            "cell_type": "place",
            "positions": positions,
            "arena_size": (2.0, 2.0),
        }
        spec = _create_coverage_heatmap(activity, metadata, subplot_position=1)
        assert isinstance(spec, PlotSpec)
        assert spec.plot_type == "heatmap"
        assert spec.cmap == "hot"
        assert spec.title == "Place Field Coverage"
        assert spec.subplot_position == 1
        plt.close("all")

    def test_grid_cell_returns_autocorr_spec(self) -> None:
        """grid cell → cmap='viridis', title='Grid Field Autocorrelation' (lines 135-159)."""
        n = 150
        positions = RNG.uniform(0, 2, size=(n, 2))
        activity = RNG.standard_normal((n, 12))
        metadata = {
            "cell_type": "grid",
            "positions": positions,
            "arena_size": (2.0, 2.0),
        }
        spec = _create_coverage_heatmap(activity, metadata, subplot_position=2)
        assert isinstance(spec, PlotSpec)
        assert spec.plot_type == "heatmap"
        assert spec.cmap == "viridis"
        assert spec.title == "Grid Field Autocorrelation"
        assert spec.colorbar is True
        assert spec.colorbar_label == "Normalized Autocorr"
        assert spec.subplot_position == 2
        plt.close("all")

    def test_scalar_arena_size_place_cell(self) -> None:
        """Scalar arena_size → x_max == y_max (line 123 covered)."""
        n = 100
        positions = RNG.uniform(0, 1, size=(n, 2))
        activity = RNG.standard_normal((n, 5))
        metadata = {
            "cell_type": "place",
            "positions": positions,
            "arena_size": 1.5,  # scalar, not tuple
        }
        spec = _create_coverage_heatmap(activity, metadata, subplot_position=0)
        assert isinstance(spec, PlotSpec)
        # extent should use the scalar value for both x_max and y_max
        extent = spec.kwargs["extent"]
        assert extent[1] == pytest.approx(1.5)
        assert extent[3] == pytest.approx(1.5)
        plt.close("all")

    def test_scalar_arena_size_grid_cell(self) -> None:
        """Scalar arena_size with grid cell → both x_max and y_max equal scalar."""
        n = 150
        positions = RNG.uniform(0, 1, size=(n, 2))
        activity = RNG.standard_normal((n, 12))
        metadata = {
            "cell_type": "grid",
            "positions": positions,
            "arena_size": 1.0,  # scalar
        }
        spec = _create_coverage_heatmap(activity, metadata, subplot_position=0)
        assert isinstance(spec, PlotSpec)
        assert spec.cmap == "viridis"
        plt.close("all")

    def test_grid_cell_extent_uses_lag_axes(self) -> None:
        """grid cell extent is [x_lags[0], x_lags[-1], y_lags[0], y_lags[-1]]."""
        n = 150
        positions = RNG.uniform(0, 2, size=(n, 2))
        activity = RNG.standard_normal((n, 12))
        metadata = {
            "cell_type": "grid",
            "positions": positions,
            "arena_size": (2.0, 2.0),
        }
        spec = _create_coverage_heatmap(activity, metadata, subplot_position=0)
        assert isinstance(spec, PlotSpec)
        extent = spec.kwargs["extent"]
        # extent has 4 values; lags include negative values (autocorrelation)
        assert len(extent) == 4
        # x_lags[0] < x_lags[-1] and y_lags[0] < y_lags[-1]
        assert extent[0] < extent[1]
        assert extent[2] < extent[3]
        plt.close("all")


# ===========================================================================
# _create_example_cell_heatmaps
# ===========================================================================

class TestCreateExampleCellHeatmaps:
    """Cover all branches: positions=None, n_dims=1, 1D bad shape, 1D tuple arena,
    n_dims=2 with tuple and scalar arena_size, positions.shape[1]!=2, and n_dims other."""

    def test_positions_none_returns_empty(self) -> None:
        """positions=None → empty list (line 201-202)."""
        activity = RNG.standard_normal((50, 5))
        metadata = {"n_dims": 2}
        specs = _create_example_cell_heatmaps(activity, metadata, subplot_position=0)
        assert specs == []
        plt.close("all")

    def test_1d_wrong_shape_returns_empty(self) -> None:
        """n_dims==1 but positions.shape[1] != 1 → empty list (line 207-208)."""
        activity = RNG.standard_normal((50, 5))
        metadata = {
            "n_dims": 1,
            "positions": RNG.uniform(0, 1, size=(50, 2)),  # shape[1]==2, not 1
            "arena_size": 1.0,
        }
        specs = _create_example_cell_heatmaps(activity, metadata, subplot_position=0)
        assert specs == []
        plt.close("all")

    def test_1d_1d_array_positions_returns_empty(self) -> None:
        """n_dims==1 but positions.ndim != 2 → empty list (line 207-208)."""
        activity = RNG.standard_normal((50, 5))
        metadata = {
            "n_dims": 1,
            "positions": RNG.uniform(0, 1, size=50),  # ndim==1, not 2
            "arena_size": 1.0,
        }
        specs = _create_example_cell_heatmaps(activity, metadata, subplot_position=0)
        assert specs == []
        plt.close("all")

    def test_1d_tuple_arena_size_extracts_first_element(self) -> None:
        """n_dims==1, tuple arena_size → arena_size = arena_size[0] (line 212)."""
        n = 100
        positions = RNG.uniform(0, 3, size=(n, 1))
        activity = RNG.standard_normal((n, 4))
        metadata = {
            "n_dims": 1,
            "positions": positions,
            "arena_size": (3.0,),  # tuple
            "cell_type": "place",
        }
        specs = _create_example_cell_heatmaps(activity, metadata, subplot_position=0)
        assert len(specs) == 3  # min(3, 4) = 3
        for spec in specs:
            assert spec.plot_type == "line"
        plt.close("all")

    def test_1d_scalar_arena_size_creates_line_plots(self) -> None:
        """n_dims==1, scalar arena_size → 3 line plots with correct structure."""
        n = 100
        positions = RNG.uniform(0, 2, size=(n, 1))
        activity = RNG.standard_normal((n, 5))
        metadata = {
            "n_dims": 1,
            "positions": positions,
            "arena_size": 2.0,
            "cell_type": "place",
        }
        specs = _create_example_cell_heatmaps(activity, metadata, subplot_position=5)
        assert len(specs) == 3
        for i, spec in enumerate(specs):
            assert spec.plot_type == "line"
            assert spec.subplot_position == 5 + i
            assert "x" in spec.data
            assert "y" in spec.data
            assert len(spec.data["x"]) == 50  # n_bins=50
        plt.close("all")

    def test_1d_fewer_cells_than_3(self) -> None:
        """n_dims==1, only 2 cells → 2 specs returned."""
        n = 80
        positions = RNG.uniform(0, 1, size=(n, 1))
        activity = RNG.standard_normal((n, 2))
        metadata = {
            "n_dims": 1,
            "positions": positions,
            "arena_size": 1.0,
            "cell_type": "grid",
        }
        specs = _create_example_cell_heatmaps(activity, metadata, subplot_position=0)
        assert len(specs) == 2
        plt.close("all")

    def test_2d_tuple_arena_size_creates_heatmaps(self) -> None:
        """n_dims==2, tuple arena_size → x_max, y_max extracted (line 248)."""
        n = 100
        positions = RNG.uniform(0, 3, size=(n, 2))
        activity = RNG.standard_normal((n, 4))
        metadata = {
            "n_dims": 2,
            "positions": positions,
            "arena_size": (3.0, 4.0),
            "cell_type": "place",
        }
        specs = _create_example_cell_heatmaps(activity, metadata, subplot_position=0)
        assert len(specs) == 3
        for spec in specs:
            assert spec.plot_type == "heatmap"
            ext = spec.kwargs["extent"]
            assert ext[1] == pytest.approx(3.0)  # x_max
            assert ext[3] == pytest.approx(4.0)  # y_max
        plt.close("all")

    def test_2d_scalar_arena_size_creates_heatmaps(self) -> None:
        """n_dims==2, scalar arena_size → x_max == y_max (line 250)."""
        n = 100
        positions = RNG.uniform(0, 2, size=(n, 2))
        activity = RNG.standard_normal((n, 4))
        metadata = {
            "n_dims": 2,
            "positions": positions,
            "arena_size": 2.5,  # scalar
            "cell_type": "place",
        }
        specs = _create_example_cell_heatmaps(activity, metadata, subplot_position=0)
        assert len(specs) == 3
        for spec in specs:
            assert spec.plot_type == "heatmap"
            ext = spec.kwargs["extent"]
            assert ext[1] == pytest.approx(2.5)
            assert ext[3] == pytest.approx(2.5)
        plt.close("all")

    def test_2d_positions_wrong_shape_returns_empty(self) -> None:
        """n_dims==2 but positions.shape[1] != 2 → empty list (line 240-241)."""
        n = 50
        activity = RNG.standard_normal((n, 4))
        metadata = {
            "n_dims": 2,
            "positions": RNG.uniform(0, 1, size=(n, 3)),  # shape[1]==3
            "arena_size": (1.0, 1.0),
            "cell_type": "place",
        }
        specs = _create_example_cell_heatmaps(activity, metadata, subplot_position=0)
        assert specs == []
        plt.close("all")

    def test_n_dims_other_returns_empty(self) -> None:
        """n_dims != 1 and n_dims != 2 → empty list (branch 238->276)."""
        n = 50
        activity = RNG.standard_normal((n, 4))
        metadata = {
            "n_dims": 3,  # neither 1 nor 2
            "positions": RNG.uniform(0, 1, size=(n, 3)),
            "arena_size": (1.0, 1.0, 1.0),
        }
        specs = _create_example_cell_heatmaps(activity, metadata, subplot_position=0)
        assert specs == []
        plt.close("all")

    def test_2d_heatmap_data_shape(self) -> None:
        """2D heatmap data has shape (n_bins, n_bins) (transposed from rate_map)."""
        n = 100
        positions = RNG.uniform(0, 1, size=(n, 2))
        activity = RNG.standard_normal((n, 3))
        metadata = {
            "n_dims": 2,
            "positions": positions,
            "arena_size": (1.0, 1.0),
            "cell_type": "grid",
        }
        specs = _create_example_cell_heatmaps(activity, metadata, subplot_position=0)
        assert len(specs) == 3
        for spec in specs:
            # data is rate_map.T, shape should be (30, 30) for n_bins=30
            data = np.array(spec.data)
            assert data.shape == (30, 30)
            assert spec.cmap == "viridis"  # grid → viridis
        plt.close("all")

    def test_2d_place_cell_cmap_is_hot(self) -> None:
        """2D place cell → cmap='hot'."""
        n = 80
        positions = RNG.uniform(0, 1, size=(n, 2))
        activity = RNG.standard_normal((n, 3))
        metadata = {
            "n_dims": 2,
            "positions": positions,
            "arena_size": (1.0, 1.0),
            "cell_type": "place",
        }
        specs = _create_example_cell_heatmaps(activity, metadata, subplot_position=0)
        assert len(specs) == 3
        for spec in specs:
            assert spec.cmap == "hot"
        plt.close("all")

    def test_2d_title_reflects_cell_type_and_index(self) -> None:
        """Title contains cell_type and cell index."""
        n = 80
        positions = RNG.uniform(0, 1, size=(n, 2))
        activity = RNG.standard_normal((n, 3))
        metadata = {
            "n_dims": 2,
            "positions": positions,
            "arena_size": (1.0, 1.0),
            "cell_type": "place",
        }
        specs = _create_example_cell_heatmaps(activity, metadata, subplot_position=0)
        assert "Place Cell 0 Heatmap" == specs[0].title
        assert "Place Cell 1 Heatmap" == specs[1].title
        assert "Place Cell 2 Heatmap" == specs[2].title
        plt.close("all")

    def test_2d_subplot_positions_increment(self) -> None:
        """Subplot positions increment by 1 for each example cell."""
        n = 80
        positions = RNG.uniform(0, 1, size=(n, 2))
        activity = RNG.standard_normal((n, 3))
        metadata = {
            "n_dims": 2,
            "positions": positions,
            "arena_size": (1.0, 1.0),
        }
        specs = _create_example_cell_heatmaps(activity, metadata, subplot_position=7)
        assert specs[0].subplot_position == 7
        assert specs[1].subplot_position == 8
        assert specs[2].subplot_position == 9
        plt.close("all")


# ===========================================================================
# _create_random_diagnostics
# ===========================================================================

class TestCreateRandomDiagnostics:
    """Cover the three-diagnostic output with concrete assertions."""

    def test_raises_when_no_positions(self) -> None:
        """positions=None raises ValueError with correct message."""
        activity = RNG.standard_normal((50, 5))
        metadata = {"arena_size": (1.0, 1.0)}
        with pytest.raises(ValueError, match="require 'positions'"):
            _create_random_diagnostics(activity, metadata, subplot_position=0)

    def test_returns_three_specs(self) -> None:
        """Returns exactly 3 PlotSpec objects."""
        n = 150
        positions = RNG.uniform(0, 1, size=(n, 2))
        activity = RNG.standard_normal((n, 15))
        metadata = {
            "positions": positions,
            "arena_size": (1.0, 1.0),
        }
        specs = _create_random_diagnostics(activity, metadata, subplot_position=0)
        assert len(specs) == 3
        plt.close("all")

    def test_all_specs_are_heatmaps(self) -> None:
        """All three specs have plot_type='heatmap'."""
        n = 150
        positions = RNG.uniform(0, 1, size=(n, 2))
        activity = RNG.standard_normal((n, 12))
        metadata = {
            "positions": positions,
            "arena_size": (1.0, 1.0),
        }
        specs = _create_random_diagnostics(activity, metadata, subplot_position=0)
        for spec in specs:
            assert spec.plot_type == "heatmap"
        plt.close("all")

    def test_single_cell_spec_uses_max_variance_cell(self) -> None:
        """First spec's title references the cell with highest variance."""
        n = 200
        n_cells = 8
        activity = np.zeros((n, n_cells))
        # Give cell 3 much higher variance
        rng = np.random.default_rng(42)
        activity[:, 3] = rng.standard_normal(n) * 100
        for c in range(n_cells):
            if c != 3:
                activity[:, c] = rng.standard_normal(n) * 0.01
        positions = rng.uniform(0, 1, size=(n, 2))
        metadata = {
            "positions": positions,
            "arena_size": (1.0, 1.0),
        }
        specs = _create_random_diagnostics(activity, metadata, subplot_position=0)
        # Title of first spec should mention cell 3
        assert "3" in specs[0].title
        plt.close("all")

    def test_single_cell_spec_data_shape(self) -> None:
        """First spec (single cell) has (20, 20) data shape."""
        n = 200
        positions = RNG.uniform(0, 1, size=(n, 2))
        activity = RNG.standard_normal((n, 10))
        metadata = {
            "positions": positions,
            "arena_size": (1.0, 1.0),
        }
        specs = _create_random_diagnostics(activity, metadata, subplot_position=0)
        data = np.array(specs[0].data)
        assert data.shape == (20, 20)
        plt.close("all")

    def test_population_coverage_spec_data_shape(self) -> None:
        """Second spec (population coverage) has (20, 20) data shape."""
        n = 200
        positions = RNG.uniform(0, 1, size=(n, 2))
        activity = RNG.standard_normal((n, 10))
        metadata = {
            "positions": positions,
            "arena_size": (1.0, 1.0),
        }
        specs = _create_random_diagnostics(activity, metadata, subplot_position=0)
        data = np.array(specs[1].data)
        assert data.shape == (20, 20)
        assert specs[1].cmap == "viridis"
        plt.close("all")

    def test_autocorr_spec_has_viridis_cmap(self) -> None:
        """Third spec (autocorrelation) has cmap='viridis'."""
        n = 150
        positions = RNG.uniform(0, 1, size=(n, 2))
        activity = RNG.standard_normal((n, 12))
        metadata = {
            "positions": positions,
            "arena_size": (1.0, 1.0),
        }
        specs = _create_random_diagnostics(activity, metadata, subplot_position=0)
        assert specs[2].cmap == "viridis"
        assert specs[2].colorbar is True
        assert specs[2].colorbar_label == "Normalized Autocorr"
        plt.close("all")

    def test_subplot_positions_correct(self) -> None:
        """Subplot positions: spec[0]=start, spec[1]=start+1, spec[2]=start+1."""
        n = 150
        positions = RNG.uniform(0, 1, size=(n, 2))
        activity = RNG.standard_normal((n, 10))
        metadata = {
            "positions": positions,
            "arena_size": (1.0, 1.0),
        }
        start = 5
        specs = _create_random_diagnostics(activity, metadata, subplot_position=start)
        assert specs[0].subplot_position == start
        assert specs[1].subplot_position == start + 1
        assert specs[2].subplot_position == start + 1
        plt.close("all")

    def test_extent_uses_arena_size(self) -> None:
        """Extent in specs[0] uses arena_size[0] and arena_size[1]."""
        n = 150
        positions = RNG.uniform(0, 2, size=(n, 2))
        activity = RNG.standard_normal((n, 10))
        arena = (2.0, 3.0)
        metadata = {
            "positions": positions,
            "arena_size": arena,
        }
        specs = _create_random_diagnostics(activity, metadata, subplot_position=0)
        ext0 = specs[0].kwargs["extent"]
        assert ext0[1] == pytest.approx(2.0)
        assert ext0[3] == pytest.approx(3.0)
        plt.close("all")

    def test_first_spec_has_hot_cmap(self) -> None:
        """First spec (single cell map) has cmap='hot'."""
        n = 150
        positions = RNG.uniform(0, 1, size=(n, 2))
        activity = RNG.standard_normal((n, 10))
        metadata = {
            "positions": positions,
            "arena_size": (1.0, 1.0),
        }
        specs = _create_random_diagnostics(activity, metadata, subplot_position=0)
        assert specs[0].cmap == "hot"
        plt.close("all")
