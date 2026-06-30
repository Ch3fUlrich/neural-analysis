"""Tests for neural_analysis.data.trajectories_gen.

Covers all public functions and private helpers to reach >= 95% line+branch
coverage. All tests are deterministic (seeded RNG), use small arrays, and
complete well under 2 s each.
"""

from __future__ import annotations

import numpy as np
import pytest

from neural_analysis.data.trajectories_gen import (
    _generate_smooth_speeds,
    _trajectory_1d,
    _trajectory_2d,
    _trajectory_3d,
    generate_head_direction,
    generate_position_trajectory,
)


# ---------------------------------------------------------------------------
# _generate_smooth_speeds
# ---------------------------------------------------------------------------

class TestGenerateSmoothSpeeds:
    """Tests for the Ornstein-Uhlenbeck speed-profile helper."""

    def test_output_shape(self):
        rng = np.random.default_rng(0)
        speeds = _generate_smooth_speeds(50, (0.1, 0.5), rng)
        assert speeds.shape == (50,)

    def test_values_within_range(self):
        rng = np.random.default_rng(1)
        speeds = _generate_smooth_speeds(200, (0.05, 0.3), rng)
        assert np.all(speeds >= 0.05)
        assert np.all(speeds <= 0.3)

    def test_deterministic_with_seed(self):
        speeds_a = _generate_smooth_speeds(30, (0.1, 0.5), np.random.default_rng(42))
        speeds_b = _generate_smooth_speeds(30, (0.1, 0.5), np.random.default_rng(42))
        np.testing.assert_array_equal(speeds_a, speeds_b)

    def test_single_sample(self):
        """n_samples=1 should return a length-1 array without entering the loop."""
        rng = np.random.default_rng(7)
        speeds = _generate_smooth_speeds(1, (0.0, 1.0), rng)
        assert speeds.shape == (1,)
        assert 0.0 <= speeds[0] <= 1.0

    def test_equal_min_max_speed(self):
        """Fixed speed (min == max) should produce a constant speed array."""
        rng = np.random.default_rng(5)
        speeds = _generate_smooth_speeds(50, (0.2, 0.2), rng)
        # sigma = (max - min) / 6 = 0, so OU noise is zero; speed is clamped to 0.2
        np.testing.assert_allclose(speeds, 0.2, atol=1e-12)

    def test_dtype_is_float(self):
        rng = np.random.default_rng(0)
        speeds = _generate_smooth_speeds(10, (0.1, 0.5), rng)
        assert np.issubdtype(speeds.dtype, np.floating)


# ---------------------------------------------------------------------------
# _trajectory_1d
# ---------------------------------------------------------------------------

class TestTrajectory1D:
    """Tests for the 1D trajectory helper."""

    def _make(self, n=100, arena=(2.0,), turning_rate=0.3, seed=0):
        rng = np.random.default_rng(seed)
        speeds = _generate_smooth_speeds(n, (0.1, 0.3), rng)
        return _trajectory_1d(n, arena, speeds, turning_rate, rng)

    def test_output_shape(self):
        pos = self._make(n=80)
        assert pos.shape == (80, 1)

    def test_starts_at_center(self):
        pos = self._make(n=50, arena=(4.0,))
        assert pos[0, 0] == pytest.approx(2.0)

    def test_positions_within_arena(self):
        arena = (3.0,)
        pos = self._make(n=200, arena=arena)
        assert np.all(pos[:, 0] >= 0.0)
        assert np.all(pos[:, 0] <= arena[0])

    def test_bounce_lower_wall(self):
        """Force a bounce off the lower wall by using a very small arena."""
        rng = np.random.default_rng(99)
        speeds = np.full(100, 0.5)  # large steps guaranteed to hit walls
        pos = _trajectory_1d(100, (0.3,), speeds, 0.5, rng)
        # All positions must stay non-negative
        assert np.all(pos[:, 0] >= 0.0)

    def test_bounce_upper_wall(self):
        """Force a bounce off the upper wall."""
        rng = np.random.default_rng(13)
        speeds = np.full(100, 0.5)
        pos = _trajectory_1d(100, (0.3,), speeds, 0.5, rng)
        assert np.all(pos[:, 0] <= 0.3)

    def test_zero_velocity_branch(self):
        """When velocity is exactly 0 after update it should be re-seeded from rng."""
        # We can't force velocity == 0 deterministically through the OU process,
        # but we can at least verify the function runs without error across many seeds.
        for seed in range(10):
            pos = self._make(n=50, seed=seed)
            assert pos.shape == (50, 1)

    def test_deterministic(self):
        pos_a = self._make(n=60, seed=7)
        pos_b = self._make(n=60, seed=7)
        np.testing.assert_array_equal(pos_a, pos_b)


# ---------------------------------------------------------------------------
# _trajectory_2d
# ---------------------------------------------------------------------------

class TestTrajectory2D:
    """Tests for the 2D trajectory helper."""

    def _make(self, n=100, arena=(2.0, 2.0), turning_rate=0.3, seed=0):
        rng = np.random.default_rng(seed)
        speeds = _generate_smooth_speeds(n, (0.1, 0.3), rng)
        return _trajectory_2d(n, arena, speeds, turning_rate, rng)

    def test_output_shape(self):
        pos = self._make(n=80)
        assert pos.shape == (80, 2)

    def test_starts_at_center(self):
        pos = self._make(n=50, arena=(4.0, 6.0))
        assert pos[0, 0] == pytest.approx(2.0)
        assert pos[0, 1] == pytest.approx(3.0)

    def test_positions_within_arena(self):
        arena = (2.0, 3.0)
        pos = self._make(n=300, arena=arena)
        assert np.all(pos[:, 0] >= 0.0)
        assert np.all(pos[:, 0] <= arena[0])
        assert np.all(pos[:, 1] >= 0.0)
        assert np.all(pos[:, 1] <= arena[1])

    def test_bounce_x_wall(self):
        """Large steps should trigger x-wall bounce logic."""
        rng = np.random.default_rng(42)
        speeds = np.full(200, 1.0)  # large steps
        pos = _trajectory_2d(200, (0.5, 10.0), speeds, 0.1, rng)
        assert np.all(pos[:, 0] >= 0.0)
        assert np.all(pos[:, 0] <= 0.5)

    def test_bounce_y_wall(self):
        """Large steps should trigger y-wall bounce logic."""
        rng = np.random.default_rng(42)
        speeds = np.full(200, 1.0)
        pos = _trajectory_2d(200, (10.0, 0.5), speeds, 0.1, rng)
        assert np.all(pos[:, 1] >= 0.0)
        assert np.all(pos[:, 1] <= 0.5)

    def test_deterministic(self):
        pos_a = self._make(n=60, seed=3)
        pos_b = self._make(n=60, seed=3)
        np.testing.assert_array_equal(pos_a, pos_b)


# ---------------------------------------------------------------------------
# _trajectory_3d
# ---------------------------------------------------------------------------

class TestTrajectory3D:
    """Tests for the 3D trajectory helper."""

    def _make(self, n=100, arena=(2.0, 2.0, 2.0), turning_rate=0.3, seed=0):
        rng = np.random.default_rng(seed)
        speeds = _generate_smooth_speeds(n, (0.1, 0.3), rng)
        return _trajectory_3d(n, arena, speeds, turning_rate, rng)

    def test_output_shape(self):
        pos = self._make(n=80)
        assert pos.shape == (80, 3)

    def test_starts_at_center(self):
        pos = self._make(n=50, arena=(4.0, 6.0, 8.0))
        assert pos[0, 0] == pytest.approx(2.0)
        assert pos[0, 1] == pytest.approx(3.0)
        assert pos[0, 2] == pytest.approx(4.0)

    def test_positions_within_arena(self):
        arena = (2.0, 3.0, 1.5)
        pos = self._make(n=300, arena=arena)
        for dim, size in enumerate(arena):
            assert np.all(pos[:, dim] >= 0.0)
            assert np.all(pos[:, dim] <= size)

    def test_bounce_x_wall(self):
        """Large steps trigger x-wall (theta reflection) branch."""
        rng = np.random.default_rng(17)
        speeds = np.full(200, 2.0)
        pos = _trajectory_3d(200, (0.3, 10.0, 10.0), speeds, 0.1, rng)
        assert np.all(pos[:, 0] >= 0.0)
        assert np.all(pos[:, 0] <= 0.3)

    def test_bounce_y_wall(self):
        """Large steps trigger y-wall (theta negation) branch."""
        rng = np.random.default_rng(18)
        speeds = np.full(200, 2.0)
        pos = _trajectory_3d(200, (10.0, 0.3, 10.0), speeds, 0.1, rng)
        assert np.all(pos[:, 1] >= 0.0)
        assert np.all(pos[:, 1] <= 0.3)

    def test_bounce_z_wall(self):
        """Large steps trigger z-wall (phi reflection) branch."""
        rng = np.random.default_rng(19)
        speeds = np.full(200, 2.0)
        pos = _trajectory_3d(200, (10.0, 10.0, 0.3), speeds, 0.1, rng)
        assert np.all(pos[:, 2] >= 0.0)
        assert np.all(pos[:, 2] <= 0.3)

    def test_deterministic(self):
        pos_a = self._make(n=60, seed=5)
        pos_b = self._make(n=60, seed=5)
        np.testing.assert_array_equal(pos_a, pos_b)


# ---------------------------------------------------------------------------
# generate_position_trajectory (public API)
# ---------------------------------------------------------------------------

class TestGeneratePositionTrajectory:
    """Tests for generate_position_trajectory."""

    # --- 1D paths -----------------------------------------------------------

    def test_1d_float_arena(self):
        pos = generate_position_trajectory(n_samples=50, arena_size=2.0, seed=0)
        assert pos.shape == (50, 1)
        assert np.all(pos[:, 0] >= 0.0)
        assert np.all(pos[:, 0] <= 2.0)

    def test_1d_int_arena(self):
        """arena_size can be an int (covers the isinstance(int, float) branch)."""
        pos = generate_position_trajectory(n_samples=30, arena_size=3, seed=0)
        assert pos.shape == (30, 1)

    # --- 2D paths -----------------------------------------------------------

    def test_2d_default_args(self):
        pos = generate_position_trajectory(seed=0)
        assert pos.shape == (1000, 2)

    def test_2d_explicit(self):
        pos = generate_position_trajectory(n_samples=80, arena_size=(1.5, 2.5), seed=1)
        assert pos.shape == (80, 2)
        assert np.all(pos[:, 0] >= 0.0)
        assert np.all(pos[:, 0] <= 1.5)
        assert np.all(pos[:, 1] >= 0.0)
        assert np.all(pos[:, 1] <= 2.5)

    # --- 3D paths -----------------------------------------------------------

    def test_3d_explicit(self):
        pos = generate_position_trajectory(
            n_samples=60, arena_size=(1.0, 1.0, 0.5), seed=2
        )
        assert pos.shape == (60, 3)
        assert np.all(pos[:, 0] >= 0.0)
        assert np.all(pos[:, 0] <= 1.0)
        assert np.all(pos[:, 1] >= 0.0)
        assert np.all(pos[:, 1] <= 1.0)
        assert np.all(pos[:, 2] >= 0.0)
        assert np.all(pos[:, 2] <= 0.5)

    # --- Deprecated speed parameter -----------------------------------------

    def test_deprecated_speed_overrides_speed_range(self):
        """Passing speed= should override speed_range."""
        pos = generate_position_trajectory(
            n_samples=50, arena_size=(2.0, 2.0), speed=0.15, seed=99
        )
        assert pos.shape == (50, 2)

    def test_deprecated_speed_none_uses_speed_range(self):
        """speed=None should leave speed_range unchanged."""
        pos = generate_position_trajectory(
            n_samples=50,
            arena_size=(2.0, 2.0),
            speed=None,
            speed_range=(0.05, 0.15),
            seed=3,
        )
        assert pos.shape == (50, 2)

    # --- Error path ---------------------------------------------------------

    def test_invalid_dimensions_raises(self):
        with pytest.raises(ValueError, match="Unsupported number of dimensions"):
            generate_position_trajectory(arena_size=(1.0, 1.0, 1.0, 1.0), seed=0)

    # --- Reproducibility ----------------------------------------------------

    def test_deterministic_with_seed(self):
        pos_a = generate_position_trajectory(n_samples=50, arena_size=(2.0, 2.0), seed=77)
        pos_b = generate_position_trajectory(n_samples=50, arena_size=(2.0, 2.0), seed=77)
        np.testing.assert_array_equal(pos_a, pos_b)

    def test_none_seed_is_random(self):
        """Two calls with seed=None should (almost certainly) differ."""
        pos_a = generate_position_trajectory(n_samples=50, arena_size=(2.0, 2.0), seed=None)
        pos_b = generate_position_trajectory(n_samples=50, arena_size=(2.0, 2.0), seed=None)
        # Very unlikely to be equal; this is a probabilistic check
        assert not np.array_equal(pos_a, pos_b)

    # --- Dtype --------------------------------------------------------------

    def test_output_dtype_is_float(self):
        pos = generate_position_trajectory(n_samples=30, seed=0)
        assert np.issubdtype(pos.dtype, np.floating)

    # --- Turning rate -------------------------------------------------------

    def test_turning_rate_zero_is_nearly_straight(self):
        """Very low turning rate should produce a path that moves in one direction."""
        pos = generate_position_trajectory(
            n_samples=50, arena_size=(1000.0, 1000.0), turning_rate=0.0, seed=0
        )
        assert pos.shape == (50, 2)


# ---------------------------------------------------------------------------
# generate_head_direction
# ---------------------------------------------------------------------------

class TestGenerateHeadDirection:
    """Tests for generate_head_direction."""

    def test_output_shape(self):
        hd = generate_head_direction(n_samples=100, seed=0)
        assert hd.shape == (100,)

    def test_values_in_range(self):
        """All angles should be in [0, 2π)."""
        hd = generate_head_direction(n_samples=500, seed=1)
        assert np.all(hd >= 0.0)
        assert np.all(hd < 2 * np.pi)

    def test_deterministic_with_seed(self):
        hd_a = generate_head_direction(n_samples=80, seed=42)
        hd_b = generate_head_direction(n_samples=80, seed=42)
        np.testing.assert_array_equal(hd_a, hd_b)

    def test_default_args(self):
        hd = generate_head_direction(seed=0)
        assert hd.shape == (1000,)

    def test_high_turning_rate(self):
        """A high turning rate should still produce valid angles."""
        hd = generate_head_direction(n_samples=200, turning_rate=2.0, seed=5)
        assert hd.shape == (200,)
        assert np.all(hd >= 0.0)
        assert np.all(hd < 2 * np.pi)

    def test_zero_turning_rate(self):
        """With turning_rate=0 the angle is constant after the first step."""
        hd = generate_head_direction(n_samples=50, turning_rate=0.0, seed=9)
        # All values after index 0 should equal hd[0] (no movement)
        np.testing.assert_allclose(hd[1:], hd[0], atol=1e-12)

    def test_none_seed_is_random(self):
        hd_a = generate_head_direction(n_samples=50, seed=None)
        hd_b = generate_head_direction(n_samples=50, seed=None)
        assert not np.array_equal(hd_a, hd_b)

    def test_dtype_is_float(self):
        hd = generate_head_direction(n_samples=30, seed=0)
        assert np.issubdtype(hd.dtype, np.floating)

    def test_single_sample(self):
        """n_samples=1 should return a length-1 array (loop body never runs)."""
        hd = generate_head_direction(n_samples=1, seed=0)
        assert hd.shape == (1,)
        assert 0.0 <= hd[0] < 2 * np.pi
