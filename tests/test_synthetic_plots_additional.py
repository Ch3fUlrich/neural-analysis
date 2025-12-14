"""Additional tests for synthetic_plots module to improve coverage."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from neural_analysis.plotting.synthetic_plots import (
    _calculate_optimal_example_cells,
    _collect_plot_specs,
    _compute_spatial_bins_1d,
    _compute_spatial_bins_2d,
    _compute_spatial_bins_3d,
    _count_fixed_plots,
    _get_cell_colors,
    plot_synthetic_data,
)


class TestGetCellColors:
    """Tests for _get_cell_colors function."""

    def test_get_cell_colors_single_type(self) -> None:
        """Test get cell colors with single cell type (covers lines 262-263)."""
        activity = np.random.randn(100, 10)
        metadata = {"cell_type": "place"}
        cell_type, cell_types, colors = _get_cell_colors(activity, metadata)
        assert cell_type == "place"
        assert cell_types is None
        assert len(colors) == 10
        assert all(c == "#E74C3C" for c in colors)  # Red for place

    def test_get_cell_colors_mixed_population(self) -> None:
        """Test get cell colors with mixed population (covers lines 258-260)."""
        activity = np.random.randn(100, 10)
        metadata = {"cell_types": ["place", "grid", "head_direction"] * 3 + ["place"]}
        cell_type, cell_types, colors = _get_cell_colors(activity, metadata)
        assert cell_types == ["place", "grid", "head_direction"] * 3 + ["place"]
        assert len(colors) == 10

    def test_get_cell_colors_unknown_type(self) -> None:
        """Test get cell colors with unknown cell type (covers default color)."""
        activity = np.random.randn(100, 10)
        metadata = {"cell_type": "unknown"}
        cell_type, cell_types, colors = _get_cell_colors(activity, metadata)
        assert cell_type == "unknown"
        assert all(c == "#7F8C8D" for c in colors)  # Default gray


class TestCountFixedPlots:
    """Tests for _count_fixed_plots function."""

    def test_count_fixed_plots_all_enabled(self) -> None:
        """Test count fixed plots with all enabled."""
        metadata = {"cell_type": "place", "positions": np.random.randn(100, 2)}
        n = _count_fixed_plots(
            metadata, True, True, True, True, True, ["pca", "umap"]
        )
        assert n > 0

    def test_count_fixed_plots_place_cells(self) -> None:
        """Test count fixed plots for place cells (covers lines 283-284)."""
        metadata = {"cell_type": "place"}
        n = _count_fixed_plots(metadata, False, True, False, False, False, [])
        assert n == 1  # Coverage plot

    def test_count_fixed_plots_grid_cells(self) -> None:
        """Test count fixed plots for grid cells."""
        metadata = {"cell_type": "grid"}
        n = _count_fixed_plots(metadata, False, True, False, False, False, [])
        assert n == 1  # Coverage plot

    def test_count_fixed_plots_random_cells(self) -> None:
        """Test count fixed plots for random cells (covers lines 285-286)."""
        metadata = {"cell_type": "random"}
        n = _count_fixed_plots(metadata, False, True, False, False, False, [])
        assert n == 4  # 4 diagnostic plots

    def test_count_fixed_plots_with_behavior(self) -> None:
        """Test count fixed plots with behavior (covers lines 288-288)."""
        metadata = {"positions": np.random.randn(100, 2)}
        n = _count_fixed_plots(metadata, False, False, True, False, False, [])
        assert n == 1

    def test_count_fixed_plots_with_ground_truth(self) -> None:
        """Test count fixed plots with ground truth (covers lines 290-290)."""
        metadata = {"ground_truth_embedding": np.random.randn(100, 2)}
        n = _count_fixed_plots(metadata, False, False, False, True, False, [])
        assert n == 1

    def test_count_fixed_plots_with_embeddings(self) -> None:
        """Test count fixed plots with embeddings (covers lines 292-292)."""
        metadata = {}
        n = _count_fixed_plots(metadata, False, False, False, False, True, ["pca", "umap"])
        assert n == 2


class TestCalculateOptimalExampleCells:
    """Tests for _calculate_optimal_example_cells function."""

    def test_calculate_optimal_example_cells_small(self) -> None:
        """Test calculate optimal example cells with small n_fixed (covers lines 302-303)."""
        n = _calculate_optimal_example_cells(1, "place")
        assert 2 <= n <= 4

    def test_calculate_optimal_example_cells_medium(self) -> None:
        """Test calculate optimal example cells with medium n_fixed (covers lines 304-305)."""
        n = _calculate_optimal_example_cells(2, "place")
        assert 2 <= n <= 4

    def test_calculate_optimal_example_cells_large(self) -> None:
        """Test calculate optimal example cells with large n_fixed (covers lines 306-311)."""
        n = _calculate_optimal_example_cells(10, "place")
        assert 2 <= n <= 4

    def test_calculate_optimal_example_cells_random(self) -> None:
        """Test calculate optimal example cells for random cells (covers lines 323-327)."""
        n = _calculate_optimal_example_cells(2, "random")
        assert 2 <= n <= 4


class TestComputeSpatialBins1D:
    """Tests for _compute_spatial_bins_1d function."""

    def test_compute_spatial_bins_1d_basic(self) -> None:
        """Test compute spatial bins 1D basic (covers lines 64-83)."""
        positions = np.random.rand(100, 1) * 10
        activity = np.random.randn(100, 5)
        arena_size = 10.0

        bin_centers, binned_rates = _compute_spatial_bins_1d(
            positions, activity, arena_size, n_bins=50
        )
        assert len(bin_centers) == 50
        assert len(binned_rates) == 50

    def test_compute_spatial_bins_1d_with_cell_idx(self) -> None:
        """Test compute spatial bins 1D with cell_idx (covers lines 75-76)."""
        positions = np.random.rand(100, 1) * 10
        activity = np.random.randn(100, 5)
        arena_size = 10.0

        bin_centers, binned_rates = _compute_spatial_bins_1d(
            positions, activity, arena_size, n_bins=50, cell_idx=0
        )
        assert len(bin_centers) == 50
        assert len(binned_rates) == 50

    def test_compute_spatial_bins_1d_tuple_arena(self) -> None:
        """Test compute spatial bins 1D with tuple arena_size (covers line 65)."""
        positions = np.random.rand(100, 1) * 10
        activity = np.random.randn(100, 5)
        arena_size = (10.0,)

        bin_centers, binned_rates = _compute_spatial_bins_1d(
            positions, activity, arena_size, n_bins=50
        )
        assert len(bin_centers) == 50


class TestComputeSpatialBins2D:
    """Tests for _compute_spatial_bins_2d function."""

    def test_compute_spatial_bins_2d_basic(self) -> None:
        """Test compute spatial bins 2D basic."""
        positions = np.random.rand(100, 2) * 10
        activity = np.random.randn(100, 5)
        arena_size = (10.0, 10.0)

        bin_centers_x, bin_centers_y, binned_rates = _compute_spatial_bins_2d(
            positions, activity, arena_size, n_bins=30
        )
        assert binned_rates.shape == (30, 30)

    def test_compute_spatial_bins_2d_with_cell_idx(self) -> None:
        """Test compute spatial bins 2D with cell_idx."""
        positions = np.random.rand(100, 2) * 10
        activity = np.random.randn(100, 5)
        arena_size = (10.0, 10.0)

        bin_centers_x, bin_centers_y, binned_rates = _compute_spatial_bins_2d(
            positions, activity, arena_size, n_bins=30, cell_idx=0
        )
        assert binned_rates.shape == (30, 30)


class TestComputeSpatialBins3D:
    """Tests for _compute_spatial_bins_3d function."""

    def test_compute_spatial_bins_3d_basic(self) -> None:
        """Test compute spatial bins 3D basic."""
        positions = np.random.rand(100, 3) * 10
        activity = np.random.randn(100, 5)
        arena_size = (10.0, 10.0, 10.0)

        x_bins, y_bins, z_bins, firing_volume = _compute_spatial_bins_3d(
            positions, activity, arena_size, n_bins=20
        )
        assert firing_volume.shape == (20, 20, 20)
        assert len(x_bins) == 21  # n_bins + 1
        assert len(y_bins) == 21
        assert len(z_bins) == 21

    def test_compute_spatial_bins_3d_with_cell_idx(self) -> None:
        """Test compute spatial bins 3D with cell_idx."""
        positions = np.random.rand(100, 3) * 10
        activity = np.random.randn(100, 5)
        arena_size = (10.0, 10.0, 10.0)

        x_bins, y_bins, z_bins, firing_volume = _compute_spatial_bins_3d(
            positions, activity, arena_size, n_bins=20, cell_idx=0
        )
        assert firing_volume.shape == (20, 20, 20)


class TestPlotSyntheticData:
    """Tests for plot_synthetic_data function."""

    @patch("neural_analysis.plotting.synthetic_plots.PlotGrid")
    def test_plot_synthetic_data_basic(self, mock_plot_grid: MagicMock) -> None:
        """Test plot_synthetic_data basic (covers main path)."""
        mock_fig = MagicMock()
        mock_axes = MagicMock()
        mock_grid_instance = MagicMock()
        mock_grid_instance.plot.return_value = (mock_fig, mock_axes)  # Tuple for matplotlib
        mock_plot_grid.return_value = mock_grid_instance

        activity = np.random.randn(100, 10)
        metadata = {"cell_type": "place", "positions": np.random.randn(100, 2)}

        result = plot_synthetic_data(activity, metadata, show_raster=False, show_fields=False, backend="matplotlib")
        assert result == mock_fig
        mock_plot_grid.assert_called_once()

    @patch("neural_analysis.plotting.synthetic_plots.PlotGrid")
    def test_plot_synthetic_data_default_embedding_methods(self, mock_plot_grid: MagicMock) -> None:
        """Test plot_synthetic_data with default embedding methods (covers lines 825-826)."""
        mock_fig = MagicMock()
        mock_axes = MagicMock()
        mock_grid_instance = MagicMock()
        mock_grid_instance.plot.return_value = (mock_fig, mock_axes)
        mock_plot_grid.return_value = mock_grid_instance

        activity = np.random.randn(100, 10)
        metadata = {"cell_type": "place"}

        result = plot_synthetic_data(activity, metadata, embedding_methods=None, backend="matplotlib")
        assert result == mock_fig

    @patch("neural_analysis.plotting.synthetic_plots.PlotGrid")
    def test_plot_synthetic_data_mixed_population(self, mock_plot_grid: MagicMock) -> None:
        """Test plot_synthetic_data with mixed population (covers lines 862-870)."""
        mock_fig = MagicMock()
        mock_axes = MagicMock()
        mock_grid_instance = MagicMock()
        mock_grid_instance.plot.return_value = (mock_fig, mock_axes)
        mock_plot_grid.return_value = mock_grid_instance

        activity = np.random.randn(100, 10)
        metadata = {"cell_types": ["place", "grid"] * 5}

        result = plot_synthetic_data(activity, metadata, show_raster=False, show_fields=False, backend="matplotlib")
        assert result == mock_fig

    @patch("neural_analysis.plotting.synthetic_plots.PlotGrid")
    def test_plot_synthetic_data_single_population(self, mock_plot_grid: MagicMock) -> None:
        """Test plot_synthetic_data with single population (covers lines 871-875)."""
        mock_fig = MagicMock()
        mock_axes = MagicMock()
        mock_grid_instance = MagicMock()
        mock_grid_instance.plot.return_value = (mock_fig, mock_axes)
        mock_plot_grid.return_value = mock_grid_instance

        activity = np.random.randn(100, 10)
        metadata = {"cell_type": "place"}

        result = plot_synthetic_data(activity, metadata, show_raster=False, show_fields=False, backend="matplotlib")
        assert result == mock_fig


class TestCreatePlaceFieldPlots:
    """Tests for _create_place_field_plots function."""

    def test_create_place_field_plots_1d(self) -> None:
        """Test _create_place_field_plots with 1D positions (covers lines 1100-1123)."""
        from neural_analysis.plotting.synthetic_plots import _create_place_field_plots
        
        activity = np.random.randn(100, 5)
        positions = np.random.rand(100, 1) * 10
        metadata = {"arena_size": 10.0, "cell_type": "place", "positions": positions, "field_centers": np.array([[5.0]]), "n_dims": 1}
        colors = ["#E74C3C"] * 5
        
        specs = _create_place_field_plots(
            activity, metadata, colors, subplot_position=0, n_examples=3
        )
        assert len(specs) == 3
        assert all(spec.plot_type == "line" for spec in specs)

    def test_create_place_field_plots_2d(self) -> None:
        """Test _create_place_field_plots with 2D positions (covers lines 1125-1156)."""
        from neural_analysis.plotting.synthetic_plots import _create_place_field_plots
        
        activity = np.random.randn(100, 5)
        positions = np.random.rand(100, 2) * 10
        metadata = {"arena_size": (10.0, 10.0), "cell_type": "place", "positions": positions, "field_centers": np.array([[5.0, 5.0]]), "n_dims": 2}
        colors = ["#E74C3C"] * 5
        
        specs = _create_place_field_plots(
            activity, metadata, colors, subplot_position=0, n_examples=3
        )
        assert len(specs) == 3
        assert all(spec.plot_type == "heatmap" for spec in specs)

    def test_create_place_field_plots_3d(self) -> None:
        """Test _create_place_field_plots with 3D positions (covers lines 1158-1189)."""
        from neural_analysis.plotting.synthetic_plots import _create_place_field_plots
        
        activity = np.random.randn(100, 5)
        positions = np.random.rand(100, 3) * 10
        metadata = {"arena_size": (10.0, 10.0, 10.0), "cell_type": "place", "positions": positions, "field_centers": np.array([[5.0, 5.0, 5.0]]), "n_dims": 3}
        colors = ["#E74C3C"] * 5
        
        specs = _create_place_field_plots(
            activity, metadata, colors, subplot_position=0, n_examples=3
        )
        assert len(specs) == 3
        assert all(spec.plot_type == "heatmap" for spec in specs)

    def test_create_place_field_plots_no_positions(self) -> None:
        """Test _create_place_field_plots with no positions (covers lines 1095-1096)."""
        from neural_analysis.plotting.synthetic_plots import _create_place_field_plots
        
        activity = np.random.randn(100, 5)
        metadata = {"cell_type": "place"}
        colors = ["#E74C3C"] * 5
        
        specs = _create_place_field_plots(
            activity, metadata, colors, subplot_position=0, n_examples=3
        )
        assert len(specs) == 0


class TestCreateCoverageHistogram1D:
    """Tests for _create_coverage_histogram_1d function."""

    def test_create_coverage_histogram_1d_basic(self) -> None:
        """Test _create_coverage_histogram_1d basic (covers lines 1278-1316)."""
        from neural_analysis.plotting.synthetic_plots import _create_coverage_histogram_1d
        
        activity = np.random.randn(100, 5)
        positions = np.random.rand(100, 1) * 10
        metadata = {"arena_size": 10.0, "positions": positions}
        
        spec = _create_coverage_histogram_1d(activity, metadata, subplot_position=0)
        assert spec is not None
        assert spec.plot_type == "line"

    def test_create_coverage_histogram_1d_no_positions(self) -> None:
        """Test _create_coverage_histogram_1d with no positions (covers lines 1288-1289)."""
        from neural_analysis.plotting.synthetic_plots import _create_coverage_histogram_1d
        
        activity = np.random.randn(100, 5)
        metadata = {"arena_size": 10.0}
        
        spec = _create_coverage_histogram_1d(activity, metadata, subplot_position=0)
        assert spec is None


class TestCreateCoverageHeatmap3D:
    """Tests for _create_coverage_heatmap_3d function."""

    def test_create_coverage_heatmap_3d_place_cells(self) -> None:
        """Test _create_coverage_heatmap_3d with place cells (covers lines 1502-1539)."""
        from neural_analysis.plotting.synthetic_plots import _create_coverage_heatmap_3d
        
        activity = np.random.randn(100, 5)
        positions = np.random.rand(100, 3) * 10
        metadata = {"arena_size": (10.0, 10.0, 10.0), "cell_type": "place", "positions": positions}
        
        specs = _create_coverage_heatmap_3d(activity, metadata, subplot_position=0)
        assert specs is not None
        assert len(specs) == 1
        assert specs[0].plot_type == "heatmap_walls"

    def test_create_coverage_heatmap_3d_grid_cells(self) -> None:
        """Test _create_coverage_heatmap_3d with grid cells (covers lines 1435-1495)."""
        from neural_analysis.plotting.synthetic_plots import _create_coverage_heatmap_3d
        
        activity = np.random.randn(100, 5)
        positions = np.random.rand(100, 3) * 10
        metadata = {"arena_size": (10.0, 10.0, 10.0), "cell_type": "grid", "positions": positions}
        
        specs = _create_coverage_heatmap_3d(activity, metadata, subplot_position=0)
        assert specs is not None
        assert len(specs) >= 1

    def test_create_coverage_heatmap_3d_no_positions(self) -> None:
        """Test _create_coverage_heatmap_3d with no positions (covers lines 1424-1425)."""
        from neural_analysis.plotting.synthetic_plots import _create_coverage_heatmap_3d
        
        activity = np.random.randn(100, 5)
        metadata = {"arena_size": (10.0, 10.0, 10.0)}
        
        specs = _create_coverage_heatmap_3d(activity, metadata, subplot_position=0)
        assert specs is None


class TestComputeHDTuningCurve:
    """Tests for _compute_hd_tuning_curve function."""

    def test_compute_hd_tuning_curve_basic(self) -> None:
        """Test _compute_hd_tuning_curve basic (covers lines 1667-1711)."""
        from neural_analysis.plotting.synthetic_plots import _compute_hd_tuning_curve
        
        activity = np.random.randn(100, 5)
        head_directions = np.random.rand(100) * 2 * np.pi
        
        angles, rates = _compute_hd_tuning_curve(activity, head_directions, cell_idx=0)
        assert len(angles) > 0
        assert len(rates) == len(angles)

    def test_compute_hd_tuning_curve_negative_angles(self) -> None:
        """Test _compute_hd_tuning_curve with negative angles (covers lines 1691-1692)."""
        from neural_analysis.plotting.synthetic_plots import _compute_hd_tuning_curve
        
        activity = np.random.randn(100, 5)
        head_directions = np.random.rand(100) * 2 * np.pi - np.pi  # Some negative
        
        angles, rates = _compute_hd_tuning_curve(activity, head_directions, cell_idx=0)
        assert len(angles) > 0


class TestCreateHDTuningPlot:
    """Tests for _create_hd_tuning_plot function."""

    def test_create_hd_tuning_plot_basic(self) -> None:
        """Test _create_hd_tuning_plot basic (covers lines 1951-1987)."""
        from neural_analysis.plotting.synthetic_plots import _create_hd_tuning_plot
        
        activity = np.random.randn(100, 5)
        head_directions = np.random.rand(100) * 2 * np.pi
        metadata = {"preferred_angles": np.array([0.5, 1.0, 1.5, 2.0, 2.5]), "head_directions": head_directions}
        colors = ["#2ECC71"] * 5
        
        spec = _create_hd_tuning_plot(activity, metadata, colors, subplot_position=0)
        assert spec is not None
        assert spec.plot_type == "line"

    def test_create_hd_tuning_plot_no_preferred_angles(self) -> None:
        """Test _create_hd_tuning_plot with no preferred_angles (covers lines 1964-1965)."""
        from neural_analysis.plotting.synthetic_plots import _create_hd_tuning_plot
        
        activity = np.random.randn(100, 5)
        metadata = {}
        colors = ["#2ECC71"] * 5
        
        spec = _create_hd_tuning_plot(activity, metadata, colors, subplot_position=0)
        assert spec is None


class TestCreateBehaviorPlot:
    """Tests for _create_behavior_plot function."""

    def test_create_behavior_plot_1d(self) -> None:
        """Test _create_behavior_plot with 1D positions (covers lines 2283-2293)."""
        from neural_analysis.plotting.synthetic_plots import _create_behavior_plot
        
        positions = np.random.rand(100, 1) * 10
        metadata = {}
        
        spec = _create_behavior_plot(positions, metadata, subplot_position=0)
        assert spec is not None
        assert spec.plot_type == "line"

    def test_create_behavior_plot_2d(self) -> None:
        """Test _create_behavior_plot with 2D positions (covers lines 2294-2308)."""
        from neural_analysis.plotting.synthetic_plots import _create_behavior_plot
        
        positions = np.random.rand(100, 2) * 10
        metadata = {}
        
        spec = _create_behavior_plot(positions, metadata, subplot_position=0)
        assert spec is not None
        assert spec.plot_type == "trajectory"

    def test_create_behavior_plot_3d(self) -> None:
        """Test _create_behavior_plot with 3D positions (covers lines 2309-2332)."""
        from neural_analysis.plotting.synthetic_plots import _create_behavior_plot
        
        positions = np.random.rand(100, 3) * 10
        metadata = {}
        
        spec = _create_behavior_plot(positions, metadata, subplot_position=0)
        assert spec is not None
        assert spec.plot_type == "trajectory3d"

    def test_create_behavior_plot_invalid_dims(self) -> None:
        """Test _create_behavior_plot with invalid dimensions (covers lines 2333-2334)."""
        from neural_analysis.plotting.synthetic_plots import _create_behavior_plot
        
        positions = np.random.rand(100, 4) * 10  # 4D (unsupported)
        metadata = {}
        
        spec = _create_behavior_plot(positions, metadata, subplot_position=0)
        assert spec is None


class TestCreateGroundTruthPlot:
    """Tests for _create_ground_truth_plot function."""

    def test_create_ground_truth_plot_2d(self) -> None:
        """Test _create_ground_truth_plot with 2D embedding (covers lines 2352-2363)."""
        from neural_analysis.plotting.synthetic_plots import _create_ground_truth_plot
        
        embedding = np.random.randn(100, 2)
        metadata = {"ground_truth_embedding": embedding}
        colors = ["#E74C3C"] * 100
        
        spec = _create_ground_truth_plot(metadata, subplot_position=0)
        assert spec is not None
        assert spec.plot_type == "scatter"

    def test_create_ground_truth_plot_3d(self) -> None:
        """Test _create_ground_truth_plot with 3D embedding (covers lines 2364-2379)."""
        from neural_analysis.plotting.synthetic_plots import _create_ground_truth_plot
        
        embedding = np.random.randn(100, 3)
        metadata = {"ground_truth_embedding": embedding}
        
        spec = _create_ground_truth_plot(metadata, subplot_position=0)
        assert spec is not None
        assert spec.plot_type == "scatter3d"

    def test_create_ground_truth_plot_no_embedding(self) -> None:
        """Test _create_ground_truth_plot with no embedding (covers lines 2347-2348)."""
        from neural_analysis.plotting.synthetic_plots import _create_ground_truth_plot
        
        metadata = {}
        
        spec = _create_ground_truth_plot(metadata, subplot_position=0)
        assert spec is None

    def test_create_ground_truth_plot_invalid_dims(self) -> None:
        """Test _create_ground_truth_plot with invalid dimensions (covers lines 2380-2381)."""
        from neural_analysis.plotting.synthetic_plots import _create_ground_truth_plot
        
        embedding = np.random.randn(100, 4)  # 4D (unsupported)
        metadata = {"ground_truth_embedding": embedding}
        
        spec = _create_ground_truth_plot(metadata, subplot_position=0)
        assert spec is None


class TestCreateEmbeddingPlots:
    """Tests for _create_embedding_plots function."""

    def test_create_embedding_plots_2d(self) -> None:
        """Test _create_embedding_plots with 2D embeddings (covers lines 2415-2426)."""
        from neural_analysis.plotting.synthetic_plots import _create_embedding_plots
        
        activity = np.random.randn(100, 10)
        metadata = {"cell_type": "place"}
        methods = ["pca", "tsne"]
        
        specs = _create_embedding_plots(
            activity, metadata, methods, n_dims=2, subplot_position=0
        )
        assert len(specs) >= 2
        assert all(spec.plot_type == "scatter" for spec in specs)

    def test_create_embedding_plots_3d(self) -> None:
        """Test _create_embedding_plots with 3D embeddings (covers lines 2427-2442)."""
        from neural_analysis.plotting.synthetic_plots import _create_embedding_plots
        
        activity = np.random.randn(100, 10)
        metadata = {"cell_type": "place"}
        methods = ["pca"]
        
        specs = _create_embedding_plots(
            activity, metadata, methods, n_dims=3, subplot_position=0
        )
        assert len(specs) >= 1
        assert all(spec.plot_type == "scatter3d" for spec in specs)

    def test_create_embedding_plots_invalid_dims(self) -> None:
        """Test _create_embedding_plots with invalid dimensions (covers lines 2443-2444)."""
        from neural_analysis.plotting.synthetic_plots import _create_embedding_plots
        
        activity = np.random.randn(100, 10)
        metadata = {"cell_type": "place"}
        methods = ["pca"]
        
        specs = _create_embedding_plots(
            activity, metadata, methods, n_dims=4, subplot_position=0
        )
        assert len(specs) == 0

    def test_create_embedding_plots_exception(self) -> None:
        """Test _create_embedding_plots with exception (covers lines 2445-2447)."""
        from neural_analysis.plotting.synthetic_plots import _create_embedding_plots
        
        activity = np.random.randn(100, 10)
        metadata = {"cell_type": "place"}
        methods = ["invalid_method"]  # Will cause exception
        
        specs = _create_embedding_plots(
            activity, metadata, methods, n_dims=2, subplot_position=0
        )
        # Should handle exception gracefully
        assert isinstance(specs, list)

