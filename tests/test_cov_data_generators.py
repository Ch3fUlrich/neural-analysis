"""Coverage-boosting tests for neural_analysis.data.generators.

Targets all uncovered lines/branches in generators.py, focusing on:
- _compute_grid_pattern_with_harmonics (2D and 3D paths, lines 64-81)
- generate_place_cells 1D and 3D paths (lines 140-222)
- generate_grid_cells 1D, 3D paths, noise_level=0 branches (lines 314-490)
- generate_head_direction_cells noise=0 branch (lines 595-610)
- generate_random_cells temporal_smoothness=0 branch (lines 668-723)
- map_to_ring and map_to_torus (lines 878-1105)
- generate_mixed_population_flexible (lines 1162-1300)
- generate_cluster_templates and generate_dataset_from_cluster_template (lines 1303-1499)
- generate_shape_distance_datasets (lines 1501-1579)
- add_noise: poisson with noise_level=0 (line 836)
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from neural_analysis.data.generators import (
    _compute_grid_pattern_with_harmonics,
    add_noise,
    generate_cluster_templates,
    generate_dataset_from_cluster_template,
    generate_grid_cells,
    generate_head_direction_cells,
    generate_mixed_neural_population,
    generate_mixed_population_flexible,
    generate_place_cells,
    generate_random_cells,
    generate_shape_distance_datasets,
    map_to_ring,
    map_to_torus,
)


# ---------------------------------------------------------------------------
# _compute_grid_pattern_with_harmonics
# ---------------------------------------------------------------------------


class TestComputeGridPatternWithHarmonics:
    """Tests for the private grid-pattern helper function."""

    def test_2d_hexagonal_returns_correct_shape(self) -> None:
        """2D path (axes provided and n_dims==2) returns (n_samples,)."""
        rng = np.random.default_rng(0)
        positions = rng.uniform(0, 1, size=(50, 2))
        phase_offset = np.array([0.1, 0.1])
        grid_spacing = 0.5
        axis1 = np.array([1.0, 0.0])
        axis2 = np.array([0.5, np.sqrt(3) / 2])
        axes = [axis1, axis2]

        rates = _compute_grid_pattern_with_harmonics(
            positions, phase_offset, grid_spacing, axes=axes
        )
        assert rates.shape == (50,)
        # Values should be finite
        assert np.all(np.isfinite(rates))

    def test_2d_hexagonal_custom_harmonic_weights(self) -> None:
        """Custom harmonic weights affect the output."""
        rng = np.random.default_rng(0)
        positions = rng.uniform(0, 1, size=(30, 2))
        phase_offset = np.array([0.0, 0.0])
        grid_spacing = 0.4
        axes = [np.array([1.0, 0.0]), np.array([0.5, 0.866])]

        rates_default = _compute_grid_pattern_with_harmonics(
            positions, phase_offset, grid_spacing, axes=axes
        )
        rates_custom = _compute_grid_pattern_with_harmonics(
            positions, phase_offset, grid_spacing, axes=axes, harmonic_weights=(1.0,)
        )
        # Different harmonic weights => different output
        assert not np.allclose(rates_default, rates_custom)

    def test_3d_tetrahedral_returns_correct_shape(self) -> None:
        """3D path (axes provided and n_dims==3) returns (n_samples,)."""
        rng = np.random.default_rng(0)
        positions = rng.uniform(0, 1, size=(40, 3))
        phase_offset = np.array([0.1, 0.1, 0.1])
        grid_spacing = 0.5
        axes = [
            np.array([1, 1, 1]) / np.sqrt(3),
            np.array([1, -1, -1]) / np.sqrt(3),
            np.array([-1, 1, -1]) / np.sqrt(3),
            np.array([-1, -1, 1]) / np.sqrt(3),
        ]

        rates = _compute_grid_pattern_with_harmonics(
            positions, phase_offset, grid_spacing, axes=axes
        )
        assert rates.shape == (40,)
        assert np.all(np.isfinite(rates))

    def test_3d_harmonic_weights_affect_output(self) -> None:
        """Different harmonic weights produce different 3D outputs."""
        rng = np.random.default_rng(1)
        positions = rng.uniform(0, 1, size=(20, 3))
        phase_offset = np.zeros(3)
        grid_spacing = 0.4
        axes = [
            np.array([1, 1, 1]) / np.sqrt(3),
            np.array([1, -1, -1]) / np.sqrt(3),
        ]

        r1 = _compute_grid_pattern_with_harmonics(
            positions, phase_offset, grid_spacing, axes=axes, harmonic_weights=(1.0, 0.4, 0.2)
        )
        r2 = _compute_grid_pattern_with_harmonics(
            positions, phase_offset, grid_spacing, axes=axes, harmonic_weights=(1.0,)
        )
        assert not np.allclose(r1, r2)

    def test_1d_no_axes_returns_correct_shape(self) -> None:
        """1D path (axes=None) returns (n_samples,)."""
        rng = np.random.default_rng(0)
        positions = rng.uniform(0, 1, size=(30, 1))
        phase_offset = np.array([0.1])
        grid_spacing = 0.4

        rates = _compute_grid_pattern_with_harmonics(
            positions, phase_offset, grid_spacing, axes=None
        )
        assert rates.shape == (30,)
        assert np.all(np.isfinite(rates))


# ---------------------------------------------------------------------------
# generate_place_cells
# ---------------------------------------------------------------------------


class TestGeneratePlaceCells1D:
    """Tests for 1D place-cell generation (arena_size as float)."""

    def test_1d_arena_size_float(self) -> None:
        """Passing float arena_size triggers 1D code path (lines 139-141, 162, 169-170, 193-198)."""
        activity, meta = generate_place_cells(
            n_cells=5, n_samples=100, arena_size=2.0, seed=0, plot=False
        )
        assert activity.shape == (100, 5)
        assert meta["n_dims"] == 1
        assert meta["field_centers"].shape == (5, 1)
        assert meta["field_radii"].shape == (5, 1)
        assert np.all(activity >= 0)

    def test_1d_arena_size_int(self) -> None:
        """Passing integer arena_size triggers 1D path."""
        activity, meta = generate_place_cells(
            n_cells=3, n_samples=50, arena_size=1, seed=0, plot=False
        )
        assert meta["n_dims"] == 1

    def test_1d_no_noise(self) -> None:
        """1D place cells with noise_level=0 skips noise block."""
        activity, meta = generate_place_cells(
            n_cells=4, n_samples=80, arena_size=1.5, noise_level=0.0, seed=0, plot=False
        )
        assert np.all(activity >= 0)
        assert activity.shape == (80, 4)

    def test_1d_with_1d_positions(self) -> None:
        """Providing a flat 1D positions array triggers reshape (line 154-155)."""
        positions = np.linspace(0, 1, 50)  # 1D flat array
        activity, meta = generate_place_cells(
            n_cells=3, n_samples=50, positions=positions, arena_size=1.0, seed=0, plot=False
        )
        assert meta["n_dims"] == 1
        assert activity.shape == (50, 3)

    def test_1d_metadata_completeness(self) -> None:
        """Metadata has all expected keys for 1D."""
        _, meta = generate_place_cells(
            n_cells=3, n_samples=50, arena_size=1.0, seed=0, plot=False
        )
        for key in ("field_centers", "field_radii", "positions", "cell_type", "n_dims", "arena_size"):
            assert key in meta
        assert meta["cell_type"] == "place"
        assert meta["field_angles"] is None


class TestGeneratePlaceCells3D:
    """Tests for 3D place-cell generation."""

    def test_3d_shape_and_metadata(self) -> None:
        """3D place cells (lines 176-178, 217-222) produce correct shapes."""
        activity, meta = generate_place_cells(
            n_cells=4,
            n_samples=80,
            arena_size=(1.0, 1.0, 0.5),
            seed=0,
            plot=False,
        )
        assert activity.shape == (80, 4)
        assert meta["n_dims"] == 3
        assert meta["field_centers"].shape == (4, 3)
        assert meta["field_radii"].shape == (4, 3)
        assert meta["field_angles"] is None
        assert np.all(activity >= 0)

    def test_3d_no_noise(self) -> None:
        """3D place cells with noise_level=0 (skips noise branch)."""
        activity, _ = generate_place_cells(
            n_cells=3,
            n_samples=60,
            arena_size=(1.0, 1.0, 0.5),
            noise_level=0.0,
            seed=0,
            plot=False,
        )
        assert np.all(activity >= 0)

    def test_3d_provided_positions(self) -> None:
        """3D with provided position array (arena_size must match dims)."""
        rng = np.random.default_rng(0)
        positions = rng.uniform(0, 1, size=(60, 3))
        activity, meta = generate_place_cells(
            n_cells=3, n_samples=60, positions=positions, arena_size=(1.0, 1.0, 0.5), seed=0, plot=False
        )
        assert meta["n_dims"] == 3


class TestGeneratePlaceCells2DExtras:
    """Extra coverage for 2D place cells."""

    def test_2d_with_noise_zero(self) -> None:
        """2D place cells with noise_level=0.0."""
        activity, _ = generate_place_cells(
            n_cells=4, n_samples=80, arena_size=(1.0, 1.0), noise_level=0.0, seed=0, plot=False
        )
        assert np.all(activity >= 0)

    def test_2d_field_angles_set(self) -> None:
        """2D path sets field_angles in metadata."""
        _, meta = generate_place_cells(
            n_cells=5, n_samples=50, arena_size=(1.0, 1.0), seed=0, plot=False
        )
        assert meta["field_angles"] is not None
        assert meta["field_angles"].shape == (5,)


# ---------------------------------------------------------------------------
# generate_grid_cells
# ---------------------------------------------------------------------------


class TestGenerateGridCells1D:
    """Tests for 1D grid-cell generation (lines 313-364)."""

    def test_1d_float_arena(self) -> None:
        """Float arena_size triggers 1D code path."""
        activity, meta = generate_grid_cells(
            n_cells=5, n_samples=100, arena_size=2.0, seed=0, plot=False
        )
        assert activity.shape == (100, 5)
        assert meta["n_dims"] == 1
        assert "phase_offsets" in meta
        assert meta["phase_offsets"].shape == (5,)
        assert np.all(activity >= 0)

    def test_1d_no_noise(self) -> None:
        """1D grid cells with noise_level=0 (skips noise branch, lines 352-360)."""
        activity, _ = generate_grid_cells(
            n_cells=4, n_samples=80, arena_size=1.5, noise_level=0.0, seed=0, plot=False
        )
        assert np.all(activity >= 0)

    def test_1d_1d_position_array(self) -> None:
        """1D position array (ndim==1) is reshaped (line 326-327)."""
        positions = np.linspace(0, 2, 100)
        activity, meta = generate_grid_cells(
            n_cells=3, n_samples=100, positions=positions, arena_size=2.0, seed=0, plot=False
        )
        assert meta["n_dims"] == 1

    def test_1d_metadata_keys(self) -> None:
        """1D grid metadata has correct keys."""
        _, meta = generate_grid_cells(n_cells=3, n_samples=50, arena_size=1.0, seed=0, plot=False)
        for key in ("phase_offsets", "positions", "cell_type", "grid_spacing", "arena_size", "n_dims"):
            assert key in meta
        assert meta["cell_type"] == "grid"


class TestGenerateGridCells2DExtras:
    """Additional 2D grid-cell coverage."""

    def test_2d_no_noise(self) -> None:
        """2D grid cells with noise_level=0 skips noise branch (line 401)."""
        activity, _ = generate_grid_cells(
            n_cells=4, n_samples=80, arena_size=(1.0, 1.0), noise_level=0.0, seed=0, plot=False
        )
        assert np.all(activity >= 0)

    def test_2d_custom_orientation(self) -> None:
        """Non-zero grid_orientation is applied."""
        activity, meta = generate_grid_cells(
            n_cells=4, n_samples=80, arena_size=(1.0, 1.0), grid_orientation=30.0, seed=0, plot=False
        )
        assert activity.shape == (80, 4)
        assert meta["grid_orientation"] == 30.0


class TestGenerateGridCells3D:
    """Tests for 3D grid-cell generation (lines 420-500)."""

    def test_3d_shape_and_metadata(self) -> None:
        """3D grid cells produce correct shape and metadata."""
        activity, meta = generate_grid_cells(
            n_cells=5, n_samples=80, arena_size=(1.0, 1.0, 0.5), seed=0, plot=False
        )
        assert activity.shape == (80, 5)
        assert meta["n_dims"] == 3
        assert "grid_spacings" in meta
        assert meta["grid_spacings"].shape == (5,)
        assert np.all(activity >= 0)

    def test_3d_no_noise(self) -> None:
        """3D grid cells with noise_level=0 (skips Gaussian noise block)."""
        activity, _ = generate_grid_cells(
            n_cells=4, n_samples=60, arena_size=(1.0, 1.0, 0.5), noise_level=0.0, seed=0, plot=False
        )
        assert np.all(activity >= 0)

    def test_3d_nonzero_orientation(self) -> None:
        """3D grid cells with non-zero orientation (triggers rotation_z branch, line 455-460)."""
        activity, meta = generate_grid_cells(
            n_cells=4,
            n_samples=60,
            arena_size=(1.0, 1.0, 0.5),
            grid_orientation=45.0,
            seed=0,
            plot=False,
        )
        assert activity.shape == (60, 4)

    def test_3d_scale_gradient(self) -> None:
        """3D grid cells create dorsal-ventral scale gradient properly."""
        # Use enough cells to trigger the scale hierarchy
        activity, meta = generate_grid_cells(
            n_cells=15, n_samples=60, arena_size=(1.0, 1.0, 0.5), seed=0, plot=False
        )
        # grid_spacings should vary (not all equal)
        assert meta["grid_spacings"].min() < meta["grid_spacings"].max()

    def test_3d_end_idx_less_than_n_cells(self) -> None:
        """n_cells=6 triggers the 'if end_idx < n_cells' branch (line 439).

        cells_per_scale = max(1, 6//5) = 1, so after 5 iterations end_idx=5 < 6.
        """
        activity, meta = generate_grid_cells(
            n_cells=6, n_samples=60, arena_size=(1.0, 1.0, 0.5), seed=0, plot=False
        )
        assert activity.shape == (60, 6)
        assert meta["grid_spacings"].shape == (6,)


# ---------------------------------------------------------------------------
# generate_head_direction_cells
# ---------------------------------------------------------------------------


class TestGenerateHeadDirectionCellsExtras:
    """Extra coverage for head-direction cells."""

    def test_no_noise(self) -> None:
        """noise_level=0 skips Poisson noise (line 581)."""
        activity, meta = generate_head_direction_cells(
            n_cells=5, n_samples=80, noise_level=0.0, seed=0, plot=False
        )
        assert activity.shape == (80, 5)
        assert np.all(activity >= 0)

    def test_provided_head_direction(self) -> None:
        """Custom head_direction array is used directly."""
        rng = np.random.default_rng(0)
        hd = rng.uniform(-np.pi, np.pi, size=80)
        activity, meta = generate_head_direction_cells(
            n_cells=5, n_samples=80, head_direction=hd, seed=0, plot=False
        )
        np.testing.assert_array_equal(meta["head_directions"], hd)

    def test_sampling_rate_stored(self) -> None:
        """sampling_rate is stored in metadata."""
        _, meta = generate_head_direction_cells(
            n_cells=3, n_samples=50, sampling_rate=30.0, seed=0, plot=False
        )
        assert meta["sampling_rate"] == 30.0


# ---------------------------------------------------------------------------
# generate_random_cells
# ---------------------------------------------------------------------------


class TestGenerateRandomCellsExtras:
    """Extra coverage for random cells."""

    def test_no_temporal_smoothing(self) -> None:
        """temporal_smoothness=0 skips EMA block (line 668)."""
        activity, meta = generate_random_cells(
            n_cells=5, n_samples=80, temporal_smoothness=0.0, seed=0, plot=False
        )
        assert activity.shape == (80, 5)
        assert np.all(activity >= 0)
        assert meta["temporal_smoothness"] == 0.0

    def test_metadata_positions_and_hd(self) -> None:
        """Metadata contains synthetic positions and head directions."""
        _, meta = generate_random_cells(n_cells=3, n_samples=50, seed=0, plot=False)
        assert "positions" in meta
        assert meta["positions"].shape == (50, 2)
        assert "head_directions" in meta
        assert meta["head_directions"].shape == (50,)

    def test_custom_arena_size(self) -> None:
        """Custom arena_size is reflected in metadata."""
        _, meta = generate_random_cells(
            n_cells=3, n_samples=50, arena_size=(2.0, 3.0), seed=0, plot=False
        )
        assert meta["arena_size"] == (2.0, 3.0)


# ---------------------------------------------------------------------------
# add_noise
# ---------------------------------------------------------------------------


class TestAddNoiseExtras:
    """Additional coverage for add_noise."""

    def test_poisson_zero_noise_level(self) -> None:
        """noise_level=0 for Poisson returns data_positive unchanged (line 836)."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = add_noise(data, noise_type="poisson", noise_level=0.0, seed=0)
        # Should return data_positive unchanged (no Poisson sampling)
        np.testing.assert_array_equal(result, data)

    def test_poisson_clips_negative(self) -> None:
        """Poisson branch: negative input values are clipped to 0."""
        data = np.array([[-1.0, 2.0], [3.0, -4.0]])
        result = add_noise(data, noise_type="poisson", noise_level=1.0, seed=0)
        # result from Poisson is non-negative
        assert np.all(result >= 0)

    def test_invalid_noise_type_raises(self) -> None:
        """Invalid noise_type raises ValueError with message."""
        data = np.ones((5, 3))
        with pytest.raises(ValueError, match="Unknown noise type"):
            add_noise(data, noise_type="laplace", noise_level=0.1)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# map_to_ring
# ---------------------------------------------------------------------------


class TestMapToRing:
    """Tests for map_to_ring manifold function (lines 878-966)."""

    def teardown_method(self) -> None:
        plt.close("all")

    def test_basic_ring_mapping(self) -> None:
        """map_to_ring returns correct shape with plot=False."""
        rng = np.random.default_rng(0)
        activity = rng.normal(0, 1, size=(50, 10))
        positions = rng.uniform(0, 1, size=(50, 1))

        ring_coords = map_to_ring(activity, positions, plot=False)
        assert ring_coords.shape == (50, 2)

    def test_ring_values_on_unit_circle(self) -> None:
        """Output points lie on the unit circle (cos^2 + sin^2 == 1)."""
        rng = np.random.default_rng(0)
        activity = rng.normal(0, 1, size=(30, 5))
        positions = np.linspace(0, 2, 30)  # 1D array

        ring_coords = map_to_ring(activity, positions, plot=False)
        # Check cos^2 + sin^2 == 1
        norms = ring_coords[:, 0] ** 2 + ring_coords[:, 1] ** 2
        np.testing.assert_allclose(norms, np.ones(30), atol=1e-12)

    def test_ring_2d_positions_flattened(self) -> None:
        """2D positions are flattened to 1D for ring mapping."""
        rng = np.random.default_rng(0)
        activity = rng.normal(0, 1, size=(30, 5))
        positions = rng.uniform(0, 1, size=(30, 1))  # shape (n_samples, 1)

        ring_coords = map_to_ring(activity, positions, plot=False)
        assert ring_coords.shape == (30, 2)

    def test_ring_with_plot(self) -> None:
        """map_to_ring with plot=True executes without error."""
        rng = np.random.default_rng(0)
        activity = rng.normal(0, 1, size=(20, 5))
        positions = np.linspace(0, 1, 20)

        ring_coords = map_to_ring(activity, positions, plot=True)
        assert ring_coords.shape == (20, 2)
        plt.close("all")


# ---------------------------------------------------------------------------
# map_to_torus
# ---------------------------------------------------------------------------


class TestMapToTorus:
    """Tests for map_to_torus manifold function (lines 969-1105)."""

    def teardown_method(self) -> None:
        plt.close("all")

    def test_basic_torus_mapping(self) -> None:
        """map_to_torus returns shape (n_samples, 3)."""
        rng = np.random.default_rng(0)
        activity = rng.normal(0, 1, size=(50, 10))
        positions = rng.uniform(0, 1, size=(50, 2))

        torus_coords = map_to_torus(activity, positions, plot=False)
        assert torus_coords.shape == (50, 3)
        assert np.all(np.isfinite(torus_coords))

    def test_torus_custom_radii(self) -> None:
        """Custom major/minor radii change coordinates."""
        rng = np.random.default_rng(0)
        activity = rng.normal(0, 1, size=(30, 5))
        positions = rng.uniform(0, 1, size=(30, 2))

        t1 = map_to_torus(activity, positions, major_radius=2.0, minor_radius=1.0, plot=False)
        t2 = map_to_torus(activity, positions, major_radius=5.0, minor_radius=0.5, plot=False)
        assert not np.allclose(t1, t2)

    def test_torus_wrong_dims_raises(self) -> None:
        """Passing 1D positions raises ValueError (line 999-1000)."""
        rng = np.random.default_rng(0)
        activity = rng.normal(0, 1, size=(20, 5))
        positions_1d = rng.uniform(0, 1, size=(20, 1))

        with pytest.raises(ValueError, match="Positions must be 2D for torus mapping"):
            map_to_torus(activity, positions_1d, plot=False)

    def test_torus_with_plot(self) -> None:
        """map_to_torus with plot=True runs without error."""
        rng = np.random.default_rng(0)
        activity = rng.normal(0, 1, size=(20, 5))
        positions = rng.uniform(0, 1, size=(20, 2))

        torus_coords = map_to_torus(activity, positions, plot=True)
        assert torus_coords.shape == (20, 3)
        plt.close("all")


# ---------------------------------------------------------------------------
# generate_mixed_population_flexible
# ---------------------------------------------------------------------------


class TestGenerateMixedPopulationFlexible:
    """Tests for generate_mixed_population_flexible (lines 1108-1300)."""

    def test_default_config(self) -> None:
        """Default configuration (cell_config=None) generates all cell types."""
        activity, meta = generate_mixed_population_flexible(
            n_samples=100, seed=0, plot=False
        )
        assert activity.ndim == 2
        assert activity.shape[0] == 100
        total = sum(len(v) for v in meta["cell_indices"].values())
        assert activity.shape[1] == total

    def test_place_only(self) -> None:
        """Single-type config with place cells."""
        config = {"place": {"n_cells": 5, "field_size": 0.2}}
        activity, meta = generate_mixed_population_flexible(
            cell_config=config, n_samples=80, seed=0, plot=False
        )
        assert activity.shape == (80, 5)
        assert "place" in meta["cell_indices"]

    def test_grid_only(self) -> None:
        """Single-type config with grid cells."""
        config = {"grid": {"n_cells": 4, "grid_spacing": 0.4}}
        activity, meta = generate_mixed_population_flexible(
            cell_config=config, n_samples=80, seed=0, plot=False
        )
        assert activity.shape == (80, 4)

    def test_head_direction_only(self) -> None:
        """Single-type config with head_direction cells (needs_hd=True)."""
        config = {"head_direction": {"n_cells": 4, "tuning_width": np.pi / 6}}
        activity, meta = generate_mixed_population_flexible(
            cell_config=config, n_samples=80, seed=0, plot=False
        )
        assert activity.shape == (80, 4)
        assert meta["head_direction"] is not None

    def test_hd_alias(self) -> None:
        """'hd' key is accepted as alias for 'head_direction'."""
        config = {"hd": {"n_cells": 3}}
        activity, meta = generate_mixed_population_flexible(
            cell_config=config, n_samples=60, seed=0, plot=False
        )
        assert activity.shape == (60, 3)
        assert "hd" in meta["cell_indices"]

    def test_random_only(self) -> None:
        """Single-type config with random cells."""
        config = {"random": {"n_cells": 4, "baseline_rate": 2.0}}
        activity, meta = generate_mixed_population_flexible(
            cell_config=config, n_samples=80, seed=0, plot=False
        )
        assert activity.shape == (80, 4)

    def test_unknown_cell_type_raises(self) -> None:
        """Unknown cell type raises ValueError (line 1238-1241)."""
        config = {"bogus_cell": {"n_cells": 3}}
        with pytest.raises(ValueError, match="Unknown cell type"):
            generate_mixed_population_flexible(
                cell_config=config, n_samples=60, seed=0, plot=False
            )

    def test_mixed_all_types(self) -> None:
        """Combined config with all supported cell types."""
        config = {
            "place": {"n_cells": 4},
            "grid": {"n_cells": 3},
            "head_direction": {"n_cells": 3},
            "random": {"n_cells": 2},
        }
        activity, meta = generate_mixed_population_flexible(
            cell_config=config, n_samples=80, seed=0, plot=False
        )
        assert activity.shape == (80, 12)  # 4+3+3+2
        assert "cell_types" in meta
        assert len(meta["cell_types"]) == 12
        assert "preferred_directions" in meta

    def test_metadata_fields(self) -> None:
        """Metadata has all expected top-level keys."""
        config = {"place": {"n_cells": 3}}
        _, meta = generate_mixed_population_flexible(
            cell_config=config, n_samples=60, seed=0, plot=False
        )
        for key in ("cell_types", "cell_indices", "positions", "arena_size", "n_samples"):
            assert key in meta

    def test_no_head_direction_when_not_requested(self) -> None:
        """head_direction is None when no hd cells requested."""
        config = {"place": {"n_cells": 3}, "random": {"n_cells": 2}}
        _, meta = generate_mixed_population_flexible(
            cell_config=config, n_samples=60, seed=0, plot=False
        )
        assert meta["head_direction"] is None

    def test_1d_arena_size(self) -> None:
        """1D arena_size (float) is accepted."""
        config = {"place": {"n_cells": 3}}
        activity, _ = generate_mixed_population_flexible(
            cell_config=config, n_samples=60, arena_size=2.0, seed=0, plot=False
        )
        assert activity.shape == (60, 3)


# ---------------------------------------------------------------------------
# generate_mixed_neural_population (default plot=False path)
# ---------------------------------------------------------------------------


class TestGenerateMixedNeuralPopulation:
    """Tests for generate_mixed_neural_population (lines 726-798)."""

    def test_basic(self) -> None:
        """Basic call with small counts."""
        activity, meta = generate_mixed_neural_population(
            n_place=4, n_grid=3, n_hd=3, n_samples=80, seed=0
        )
        assert activity.shape == (80, 10)
        assert len(meta["cell_types"]) == 10

    def test_cell_type_labels(self) -> None:
        """Cell type labels match requested counts."""
        activity, meta = generate_mixed_neural_population(
            n_place=3, n_grid=2, n_hd=2, n_samples=60, seed=0
        )
        types = meta["cell_types"]
        assert np.sum(types == "place") == 3
        assert np.sum(types == "grid") == 2
        assert np.sum(types == "head_direction") == 2


# ---------------------------------------------------------------------------
# generate_cluster_templates
# ---------------------------------------------------------------------------


class TestGenerateClusterTemplates:
    """Tests for generate_cluster_templates (lines 1303-1379)."""

    def test_output_shape(self) -> None:
        """Output shape is (n_clusters, n_features)."""
        templates = generate_cluster_templates(n_clusters=3, n_features=50, seed=0)
        assert templates.shape == (3, 50)

    def test_dtype(self) -> None:
        """Templates are float64."""
        templates = generate_cluster_templates(n_clusters=2, n_features=20, seed=0)
        assert templates.dtype == np.float64

    def test_different_clusters_differ(self) -> None:
        """Each row in templates is distinct."""
        templates = generate_cluster_templates(n_clusters=4, n_features=30, seed=0)
        for i in range(4):
            for j in range(i + 1, 4):
                assert not np.allclose(templates[i], templates[j])

    def test_reproducible(self) -> None:
        """Same seed produces identical templates."""
        t1 = generate_cluster_templates(n_clusters=3, n_features=20, seed=7)
        t2 = generate_cluster_templates(n_clusters=3, n_features=20, seed=7)
        np.testing.assert_array_equal(t1, t2)

    def test_single_cluster(self) -> None:
        """Single cluster returns shape (1, n_features)."""
        templates = generate_cluster_templates(n_clusters=1, n_features=10, seed=0)
        assert templates.shape == (1, 10)

    def test_cluster_separation_affects_output(self) -> None:
        """Higher separation creates more distinct templates."""
        t_low = generate_cluster_templates(n_clusters=2, n_features=50, cluster_separation=1.0, seed=0)
        t_high = generate_cluster_templates(n_clusters=2, n_features=50, cluster_separation=50.0, seed=0)
        diff_low = np.linalg.norm(t_low[0] - t_low[1])
        diff_high = np.linalg.norm(t_high[0] - t_high[1])
        assert diff_high > diff_low

    def test_seed_none(self) -> None:
        """seed=None runs without error (uses default_rng with (0 + k*1000))."""
        templates = generate_cluster_templates(n_clusters=2, n_features=15, seed=None)
        assert templates.shape == (2, 15)


# ---------------------------------------------------------------------------
# generate_dataset_from_cluster_template
# ---------------------------------------------------------------------------


class TestGenerateDatasetFromClusterTemplate:
    """Tests for generate_dataset_from_cluster_template (lines 1381-1498)."""

    def test_type0_curve_shape(self) -> None:
        """cluster_id % 3 == 0: points along a curve (line 1447-1451)."""
        template = np.zeros(30)
        dataset = generate_dataset_from_cluster_template(
            template, n_neurons=20, cluster_id=0, seed=0
        )
        assert dataset.shape == (20, 30)
        assert dataset.dtype == np.float64

    def test_type1_plane_shape(self) -> None:
        """cluster_id % 3 == 1: points in a plane (lines 1452-1464)."""
        template = np.zeros(30)
        dataset = generate_dataset_from_cluster_template(
            template, n_neurons=25, cluster_id=1, seed=0
        )
        assert dataset.shape == (25, 30)

    def test_type2_spread_shape(self) -> None:
        """cluster_id % 3 == 2: spread distribution (lines 1465-1469)."""
        template = np.zeros(30)
        dataset = generate_dataset_from_cluster_template(
            template, n_neurons=20, cluster_id=2, seed=0
        )
        assert dataset.shape == (20, 30)

    def test_nonzero_template_affects_output(self) -> None:
        """Non-zero template shifts the dataset (line 1486-1490)."""
        template_zero = np.zeros(20)
        template_nonzero = np.ones(20) * 10.0

        d_zero = generate_dataset_from_cluster_template(
            template_zero, n_neurons=15, cluster_id=0, seed=0
        )
        d_nonzero = generate_dataset_from_cluster_template(
            template_nonzero, n_neurons=15, cluster_id=0, seed=0
        )
        assert not np.allclose(d_zero, d_nonzero)

    def test_noise_scale_affects_output(self) -> None:
        """Different noise_scale values produce different output."""
        template = np.zeros(20)
        d_low = generate_dataset_from_cluster_template(
            template, n_neurons=15, cluster_id=0, noise_scale=0.01, seed=0
        )
        d_high = generate_dataset_from_cluster_template(
            template, n_neurons=15, cluster_id=0, noise_scale=5.0, seed=0
        )
        assert not np.allclose(d_low, d_high)

    def test_reproducible(self) -> None:
        """Same seed produces identical dataset."""
        template = np.arange(20, dtype=float)
        d1 = generate_dataset_from_cluster_template(template, n_neurons=15, cluster_id=0, seed=42)
        d2 = generate_dataset_from_cluster_template(template, n_neurons=15, cluster_id=0, seed=42)
        np.testing.assert_array_equal(d1, d2)

    def test_template_norm_zero(self) -> None:
        """Zero template (norm == 0) falls to else branch (line 1489-1490)."""
        template = np.zeros(20)
        dataset = generate_dataset_from_cluster_template(
            template, n_neurons=10, cluster_id=0, seed=0
        )
        assert dataset.shape == (10, 20)

    def test_multiple_cluster_ids_type_coverage(self) -> None:
        """cluster_id 0..5 covers all three geometry types."""
        template = np.ones(20)
        for cid in range(6):
            d = generate_dataset_from_cluster_template(
                template, n_neurons=10, cluster_id=cid, seed=0
            )
            assert d.shape == (10, 20)


# ---------------------------------------------------------------------------
# generate_shape_distance_datasets
# ---------------------------------------------------------------------------


class TestGenerateShapeDistanceDatasets:
    """Tests for generate_shape_distance_datasets (lines 1501-1579)."""

    def test_basic_output(self) -> None:
        """Returns list of arrays and labels array."""
        datasets, labels = generate_shape_distance_datasets(
            n_datasets=6, n_clusters=3, min_neurons=10, max_neurons=20, n_features=30, seed=0
        )
        assert len(datasets) == 6
        assert labels.shape == (6,)
        assert labels.dtype == int

    def test_labels_cycle_through_clusters(self) -> None:
        """Labels cycle through 0..n_clusters-1."""
        datasets, labels = generate_shape_distance_datasets(
            n_datasets=10, n_clusters=5, min_neurons=10, max_neurons=15, n_features=20, seed=0
        )
        for k in range(5):
            assert k in labels

    def test_dataset_shapes(self) -> None:
        """Each dataset has n_features columns and n_neurons in [min, max]."""
        datasets, _ = generate_shape_distance_datasets(
            n_datasets=6, n_clusters=3, min_neurons=10, max_neurons=20, n_features=30, seed=0
        )
        for ds in datasets:
            assert ds.ndim == 2
            assert ds.shape[1] == 30
            assert 10 <= ds.shape[0] <= 20

    def test_reproducible(self) -> None:
        """Same seed produces identical results."""
        d1, l1 = generate_shape_distance_datasets(
            n_datasets=4, n_clusters=2, min_neurons=10, max_neurons=15, n_features=15, seed=7
        )
        d2, l2 = generate_shape_distance_datasets(
            n_datasets=4, n_clusters=2, min_neurons=10, max_neurons=15, n_features=15, seed=7
        )
        np.testing.assert_array_equal(l1, l2)
        for a, b in zip(d1, d2):
            np.testing.assert_array_equal(a, b)

    def test_custom_noise_scale(self) -> None:
        """noise_scale parameter is passed through."""
        d_low, _ = generate_shape_distance_datasets(
            n_datasets=4, n_clusters=2, min_neurons=10, max_neurons=15, n_features=20, noise_scale=0.0, seed=0
        )
        d_high, _ = generate_shape_distance_datasets(
            n_datasets=4, n_clusters=2, min_neurons=10, max_neurons=15, n_features=20, noise_scale=5.0, seed=0
        )
        # Different noise levels should produce different datasets
        for a, b in zip(d_low, d_high):
            assert not np.allclose(a, b)

    def test_cluster_separation(self) -> None:
        """cluster_separation is passed to templates."""
        datasets, labels = generate_shape_distance_datasets(
            n_datasets=4,
            n_clusters=2,
            min_neurons=10,
            max_neurons=15,
            n_features=20,
            cluster_separation=50.0,
            seed=0,
        )
        assert len(datasets) == 4
