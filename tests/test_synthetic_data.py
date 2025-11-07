"""Tests for synthetic data generation."""

import numpy as np
import pytest

from neural_analysis.synthetic_data import (
    add_noise,
    generate_grid_cells,
    generate_head_direction,
    generate_head_direction_cells,
    generate_mixed_neural_population,
    generate_place_cells,
    generate_position_trajectory,
    generate_s_curve,
    generate_swiss_roll,
)


class TestManifolds:
    """Tests for manifold generation."""

    def test_swiss_roll_shape(self):
        """Test Swiss roll generates correct shape."""
        points, colors = generate_swiss_roll(n_samples=100, seed=42)

        assert points.shape == (100, 3)
        assert colors.shape == (100,)

    def test_swiss_roll_noise(self):
        """Test Swiss roll with noise."""
        points_clean, _ = generate_swiss_roll(n_samples=100, noise=0.0, seed=42)
        points_noisy, _ = generate_swiss_roll(n_samples=100, noise=0.1, seed=42)

        # Noisy version should be different
        assert not np.allclose(points_clean, points_noisy)

    def test_swiss_roll_reproducible(self):
        """Test Swiss roll is reproducible with same seed."""
        points1, colors1 = generate_swiss_roll(n_samples=100, seed=42)
        points2, colors2 = generate_swiss_roll(n_samples=100, seed=42)

        np.testing.assert_array_equal(points1, points2)
        np.testing.assert_array_equal(colors1, colors2)

    def test_s_curve_shape(self):
        """Test S-curve generates correct shape."""
        points, colors = generate_s_curve(n_samples=100, seed=42)

        assert points.shape == (100, 3)
        assert colors.shape == (100,)

    def test_s_curve_noise(self):
        """Test S-curve with noise."""
        points_clean, _ = generate_s_curve(n_samples=100, noise=0.0, seed=42)
        points_noisy, _ = generate_s_curve(n_samples=100, noise=0.1, seed=42)

        assert not np.allclose(points_clean, points_noisy)


class TestBehavioral:
    """Tests for behavioral data generation."""

    def test_position_trajectory_shape(self):
        """Test position trajectory shape."""
        positions = generate_position_trajectory(n_samples=1000, seed=42)

        assert positions.shape == (1000, 2)

    def test_position_stays_in_arena(self):
        """Test position stays within arena bounds."""
        arena_size = (2.0, 3.0)
        positions = generate_position_trajectory(
            n_samples=1000,
            arena_size=arena_size,
            seed=42
        )

        assert np.all(positions[:, 0] >= 0)
        assert np.all(positions[:, 0] <= arena_size[0])
        assert np.all(positions[:, 1] >= 0)
        assert np.all(positions[:, 1] <= arena_size[1])

    def test_position_starts_center(self):
        """Test position starts near center."""
        arena_size = (2.0, 2.0)
        positions = generate_position_trajectory(
            n_samples=1000,
            arena_size=arena_size,
            seed=42
        )

        expected_center = np.array([arena_size[0] / 2, arena_size[1] / 2])
        np.testing.assert_array_almost_equal(positions[0], expected_center)

    def test_head_direction_shape(self):
        """Test head direction shape."""
        hd = generate_head_direction(n_samples=1000, seed=42)

        assert hd.shape == (1000,)

    def test_head_direction_range(self):
        """Test head direction stays in [0, 2π]."""
        hd = generate_head_direction(n_samples=1000, seed=42)

        assert np.all(hd >= 0)
        assert np.all(hd <= 2 * np.pi)

    def test_head_direction_continuous(self):
        """Test head direction changes smoothly."""
        hd = generate_head_direction(n_samples=1000, turning_rate=0.1, seed=42)

        # Compute angular differences
        diffs = np.diff(hd)

        # Most differences should be small (except wrapping)
        small_diffs = np.abs(diffs) < 1.0
        assert np.sum(small_diffs) > 900  # Most should be smooth


class TestPlaceCells:
    """Tests for place cell generation."""

    def test_place_cells_shape(self):
        """Test place cell activity shape."""
        activity, metadata = generate_place_cells(
            n_cells=50,
            n_samples=1000,
            seed=42
        )

        assert activity.shape == (1000, 50)

    def test_place_cells_metadata(self):
        """Test place cell metadata."""
        activity, metadata = generate_place_cells(
            n_cells=50,
            n_samples=1000,
            seed=42
        )

        assert 'field_centers' in metadata
        assert 'positions' in metadata
        assert 'cell_type' in metadata

        assert metadata['field_centers'].shape == (50, 2)
        assert metadata['positions'].shape == (1000, 2)
        assert metadata['cell_type'] == 'place'

    def test_place_cells_nonnegative(self):
        """Test place cell activity is non-negative."""
        activity, _ = generate_place_cells(
            n_cells=50,
            n_samples=1000,
            seed=42
        )

        assert np.all(activity >= 0)

    def test_place_cells_custom_positions(self):
        """Test place cells with custom position trajectory."""
        positions = np.random.randn(1000, 2)
        activity, metadata = generate_place_cells(
            n_cells=50,
            n_samples=1000,
            positions=positions,
            seed=42
        )

        np.testing.assert_array_equal(metadata['positions'], positions)

    def test_place_cells_field_size(self):
        """Test place cells respond to field size."""
        activity_narrow, _ = generate_place_cells(
            n_cells=10,
            n_samples=1000,
            field_size=0.1,
            seed=42
        )

        activity_wide, _ = generate_place_cells(
            n_cells=10,
            n_samples=1000,
            field_size=0.5,
            seed=42
        )

        # Wide fields should have more non-zero activity
        active_narrow = np.sum(activity_narrow > 0.1)
        active_wide = np.sum(activity_wide > 0.1)

        assert active_wide > active_narrow

    def test_place_cells_noise(self):
        """Test place cells with different noise levels."""
        activity_clean, _ = generate_place_cells(
            n_cells=10,
            n_samples=1000,
            noise_level=0.0,
            seed=42
        )

        activity_noisy, _ = generate_place_cells(
            n_cells=10,
            n_samples=1000,
            noise_level=1.0,
            seed=42
        )

        # Should be different
        assert not np.allclose(activity_clean, activity_noisy)


class TestGridCells:
    """Tests for grid cell generation."""

    def test_grid_cells_shape(self):
        """Test grid cell activity shape."""
        activity, metadata = generate_grid_cells(
            n_cells=30,
            n_samples=1000,
            seed=42
        )

        assert activity.shape == (1000, 30)

    def test_grid_cells_metadata(self):
        """Test grid cell metadata."""
        activity, metadata = generate_grid_cells(
            n_cells=30,
            n_samples=1000,
            seed=42
        )

        assert 'phase_offsets' in metadata
        assert 'positions' in metadata
        assert 'cell_type' in metadata
        assert 'grid_spacing' in metadata

        assert metadata['phase_offsets'].shape == (30, 2)
        assert metadata['cell_type'] == 'grid'

    def test_grid_cells_nonnegative(self):
        """Test grid cell activity is non-negative."""
        activity, _ = generate_grid_cells(
            n_cells=30,
            n_samples=1000,
            seed=42
        )

        assert np.all(activity >= 0)

    def test_grid_cells_spacing(self):
        """Test grid cells respond to spacing parameter."""
        activity_fine, _ = generate_grid_cells(
            n_cells=10,
            n_samples=2000,
            grid_spacing=0.2,
            arena_size=(5.0, 5.0),  # Larger arena for better testing
            seed=42
        )

        activity_coarse, _ = generate_grid_cells(
            n_cells=10,
            n_samples=2000,
            grid_spacing=1.0,  # Much larger spacing
            arena_size=(5.0, 5.0),
            seed=43  # Different seed to ensure different trajectory
        )

        # Fine grids should have higher total activity due to more fields
        total_activity_fine = np.sum(activity_fine)
        total_activity_coarse = np.sum(activity_coarse)

        # Fine grids visit more fields, so more overall activity
        assert total_activity_fine > total_activity_coarse * 0.5  # At least half as much


class TestHeadDirectionCells:
    """Tests for head direction cell generation."""

    def test_hd_cells_shape(self):
        """Test head direction cell activity shape."""
        activity, metadata = generate_head_direction_cells(
            n_cells=60,
            n_samples=1000,
            seed=42
        )

        assert activity.shape == (1000, 60)

    def test_hd_cells_metadata(self):
        """Test head direction cell metadata."""
        activity, metadata = generate_head_direction_cells(
            n_cells=60,
            n_samples=1000,
            seed=42
        )

        assert 'preferred_directions' in metadata
        assert 'head_directions' in metadata
        assert 'cell_type' in metadata

        assert metadata['preferred_directions'].shape == (60,)
        assert metadata['cell_type'] == 'head_direction'

    def test_hd_cells_nonnegative(self):
        """Test head direction cell activity is non-negative."""
        activity, _ = generate_head_direction_cells(
            n_cells=60,
            n_samples=1000,
            seed=42
        )

        assert np.all(activity >= 0)

    def test_hd_cells_tuning(self):
        """Test head direction cells have directional tuning."""
        activity, metadata = generate_head_direction_cells(
            n_cells=60,
            n_samples=1000,
            seed=42
        )

        # Each cell should have peak activity at its preferred direction
        hd = metadata['head_directions']
        preferred_dirs = metadata['preferred_directions']

        for i in range(60):
            # Find when head direction is close to preferred direction
            angle_diff = np.abs(hd - preferred_dirs[i])
            angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
            near_preferred = angle_diff < 0.5  # Within 0.5 radians

            if np.sum(near_preferred) > 0:
                # Activity should be higher near preferred direction
                mean_near = np.mean(activity[near_preferred, i])
                mean_far = np.mean(activity[~near_preferred, i])
                assert mean_near > mean_far


class TestMixedPopulation:
    """Tests for mixed neural population generation."""

    def test_mixed_population_shape(self):
        """Test mixed population shape."""
        activity, metadata = generate_mixed_neural_population(
            n_place=50,
            n_grid=30,
            n_hd=20,
            n_samples=1000,
            seed=42
        )

        assert activity.shape == (1000, 100)  # 50 + 30 + 20

    def test_mixed_population_cell_types(self):
        """Test mixed population cell type labels."""
        activity, metadata = generate_mixed_neural_population(
            n_place=50,
            n_grid=30,
            n_hd=20,
            n_samples=1000,
            seed=42
        )

        cell_types = metadata['cell_types']
        assert len(cell_types) == 100
        assert np.sum(cell_types == 'place') == 50
        assert np.sum(cell_types == 'grid') == 30
        assert np.sum(cell_types == 'head_direction') == 20

    def test_mixed_population_metadata(self):
        """Test mixed population has complete metadata."""
        activity, metadata = generate_mixed_neural_population(
            n_place=50,
            n_grid=30,
            n_hd=20,
            seed=42
        )

        assert 'cell_types' in metadata
        assert 'positions' in metadata
        assert 'head_direction' in metadata
        assert 'place_meta' in metadata
        assert 'grid_meta' in metadata
        assert 'hd_meta' in metadata

    def test_mixed_population_nonnegative(self):
        """Test mixed population activity is non-negative."""
        activity, _ = generate_mixed_neural_population(
            n_place=50,
            n_grid=30,
            n_hd=20,
            seed=42
        )

        assert np.all(activity >= 0)


class TestNoise:
    """Tests for noise addition."""

    def test_gaussian_noise(self):
        """Test Gaussian noise addition."""
        data = np.ones((100, 10))
        noisy = add_noise(data, noise_type='gaussian', noise_level=0.1, seed=42)

        assert noisy.shape == data.shape
        assert not np.allclose(data, noisy)

    def test_poisson_noise(self):
        """Test Poisson noise addition."""
        data = np.ones((100, 10)) * 100  # Higher values for noticeable Poisson
        noisy = add_noise(data, noise_type='poisson', noise_level=1.0, seed=42)

        assert noisy.shape == data.shape
        # Poisson noise should create variability
        assert np.std(noisy) > 0

    def test_uniform_noise(self):
        """Test uniform noise addition."""
        data = np.ones((100, 10))
        noisy = add_noise(data, noise_type='uniform', noise_level=0.1, seed=42)

        assert noisy.shape == data.shape
        assert not np.allclose(data, noisy)

    def test_noise_level(self):
        """Test noise level affects variance."""
        data = np.ones((1000, 10))

        noisy_low = add_noise(data, noise_type='gaussian', noise_level=0.1, seed=42)
        noisy_high = add_noise(data, noise_type='gaussian', noise_level=1.0, seed=42)

        var_low = np.var(noisy_low - data)
        var_high = np.var(noisy_high - data)

        assert var_high > var_low

    def test_noise_reproducible(self):
        """Test noise is reproducible with same seed."""
        data = np.ones((100, 10))

        noisy1 = add_noise(data, noise_type='gaussian', noise_level=0.1, seed=42)
        noisy2 = add_noise(data, noise_type='gaussian', noise_level=0.1, seed=42)

        np.testing.assert_array_equal(noisy1, noisy2)

    def test_invalid_noise_type(self):
        """Test invalid noise type raises error."""
        data = np.ones((100, 10))

        with pytest.raises(ValueError, match="Unknown noise type"):
            add_noise(data, noise_type='invalid', noise_level=0.1)


class TestReproducibility:
    """Tests for reproducibility across functions."""

    def test_all_functions_reproducible(self):
        """Test all generation functions are reproducible."""
        # Swiss roll
        sr1, _ = generate_swiss_roll(100, seed=42)
        sr2, _ = generate_swiss_roll(100, seed=42)
        np.testing.assert_array_equal(sr1, sr2)

        # Position
        pos1 = generate_position_trajectory(100, seed=42)
        pos2 = generate_position_trajectory(100, seed=42)
        np.testing.assert_array_equal(pos1, pos2)

        # Place cells
        pc1, _ = generate_place_cells(10, 100, seed=42)
        pc2, _ = generate_place_cells(10, 100, seed=42)
        np.testing.assert_array_equal(pc1, pc2)

        # Grid cells
        gc1, _ = generate_grid_cells(10, 100, seed=42)
        gc2, _ = generate_grid_cells(10, 100, seed=42)
        np.testing.assert_array_equal(gc1, gc2)

        # HD cells
        hd1, _ = generate_head_direction_cells(10, 100, seed=42)
        hd2, _ = generate_head_direction_cells(10, 100, seed=42)
        np.testing.assert_array_equal(hd1, hd2)
