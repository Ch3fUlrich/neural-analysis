"""Tests for trajectory computation utilities."""

import numpy as np
import pytest

from neural_analysis.utils.trajectories import compute_colors, prepare_trajectory_segments


class TestPrepareTrajectorySegments:
    """Test prepare_trajectory_segments function."""

    def test_prepare_trajectory_segments_2d(self):
        """Test 2D trajectory segment preparation."""
        x = np.array([0, 1, 2, 3, 4])
        y = np.array([0, 1, 0, 1, 0])
        segments = prepare_trajectory_segments(x, y)
        assert segments.shape == (4, 2, 2)
        assert segments[0, 0, 0] == 0
        assert segments[0, 0, 1] == 0
        assert segments[0, 1, 0] == 1
        assert segments[0, 1, 1] == 1

    def test_prepare_trajectory_segments_3d(self):
        """Test 3D trajectory segment preparation."""
        x = np.array([0, 1, 2, 3])
        y = np.array([0, 1, 0, 1])
        z = np.array([0, 0.5, 1, 1.5])
        segments = prepare_trajectory_segments(x, y, z)
        assert segments.shape == (3, 2, 3)
        assert segments[0, 0, 2] == 0
        assert segments[0, 1, 2] == 0.5

    def test_prepare_trajectory_segments_mismatched_lengths_2d(self):
        """Test error for mismatched x and y lengths."""
        x = np.array([0, 1, 2])
        y = np.array([0, 1])
        with pytest.raises(ValueError, match="same length"):
            prepare_trajectory_segments(x, y)

    def test_prepare_trajectory_segments_mismatched_lengths_3d(self):
        """Test error for mismatched x, y, z lengths."""
        x = np.array([0, 1, 2])
        y = np.array([0, 1, 2])
        z = np.array([0, 1])
        with pytest.raises(ValueError, match="same length"):
            prepare_trajectory_segments(x, y, z)

    def test_prepare_trajectory_segments_too_few_points_2d(self):
        """Test error for too few points in 2D."""
        x = np.array([0])
        y = np.array([0])
        with pytest.raises(ValueError, match="at least 2 points"):
            prepare_trajectory_segments(x, y)

    def test_prepare_trajectory_segments_too_few_points_3d(self):
        """Test error for too few points in 3D."""
        x = np.array([0])
        y = np.array([0])
        z = np.array([0])
        with pytest.raises(ValueError, match="at least 2 points"):
            prepare_trajectory_segments(x, y, z)

    def test_prepare_trajectory_segments_single_segment_2d(self):
        """Test 2D trajectory with exactly 2 points."""
        x = np.array([0, 1])
        y = np.array([0, 1])
        segments = prepare_trajectory_segments(x, y)
        assert segments.shape == (1, 2, 2)

    def test_prepare_trajectory_segments_single_segment_3d(self):
        """Test 3D trajectory with exactly 2 points."""
        x = np.array([0, 1])
        y = np.array([0, 1])
        z = np.array([0, 1])
        segments = prepare_trajectory_segments(x, y, z)
        assert segments.shape == (1, 2, 3)


class TestComputeColors:
    """Test compute_colors function."""

    def test_compute_colors_time(self):
        """Test color computation with time method."""
        n_points = 100
        colors = compute_colors(n_points, color_by="time")
        assert len(colors) == n_points
        assert colors[0] == 0
        assert colors[-1] == n_points - 1
        assert np.allclose(colors, np.arange(n_points, dtype=float))

    def test_compute_colors_with_array(self):
        """Test color computation with array input."""
        n_points = 10
        color_array = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        colors = compute_colors(n_points, color_by=color_array)
        assert len(colors) == n_points
        assert np.allclose(colors, color_array)

    def test_compute_colors_invalid_method(self):
        """Test error for unsupported color_by method."""
        with pytest.raises(ValueError, match="Unsupported color_by method"):
            compute_colors(10, color_by="invalid")

    def test_compute_colors_array_length_mismatch(self):
        """Test error for array length mismatch."""
        n_points = 10
        color_array = np.array([0.1, 0.2, 0.3])  # Wrong length
        with pytest.raises(ValueError, match="must match n_points"):
            compute_colors(n_points, color_by=color_array)

    def test_compute_colors_zero_points(self):
        """Test error for zero points."""
        with pytest.raises(ValueError, match="at least 1 point"):
            compute_colors(0, color_by="time")

    def test_compute_colors_single_point(self):
        """Test color computation for single point."""
        colors = compute_colors(1, color_by="time")
        assert len(colors) == 1
        assert colors[0] == 0



