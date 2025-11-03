"""Tests for preprocessing utilities."""

from __future__ import annotations

import numpy as np
import pytest

from neural_analysis.utils import normalize_01


class TestNormalize01:
    """Test suite for normalize_01 function."""

    def test_basic_normalization(self):
        """Test basic [0, 1] normalization."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = normalize_01(data)
        expected = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        np.testing.assert_allclose(result, expected)

    def test_already_normalized(self):
        """Test data already in [0, 1]."""
        data = np.array([0.0, 0.5, 1.0])
        result = normalize_01(data)
        np.testing.assert_allclose(result, data)

    def test_constant_input(self):
        """Test constant array (all same value)."""
        data = np.array([5.0, 5.0, 5.0, 5.0])
        result = normalize_01(data)
        # Should return zeros when range is zero
        expected = np.array([0.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(result, expected)

    def test_negative_values(self):
        """Test normalization with negative values."""
        data = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = normalize_01(data)
        expected = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        np.testing.assert_allclose(result, expected)

    def test_2d_global_normalization(self):
        """Test 2D array with global min/max (axis=None)."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = normalize_01(data)
        expected = np.array([[0.0, 1/3], [2/3, 1.0]])
        np.testing.assert_allclose(result, expected)

    def test_2d_row_normalization(self):
        """Test 2D array normalized per row (axis=1)."""
        data = np.array([[1.0, 5.0], [2.0, 8.0]])
        result = normalize_01(data, axis=1)
        expected = np.array([[0.0, 1.0], [0.0, 1.0]])
        np.testing.assert_allclose(result, expected)

    def test_2d_column_normalization(self):
        """Test 2D array normalized per column (axis=0)."""
        data = np.array([[1.0, 2.0], [3.0, 6.0]])
        result = normalize_01(data, axis=0)
        expected = np.array([[0.0, 0.0], [1.0, 1.0]])
        np.testing.assert_allclose(result, expected)

    def test_custom_min_max(self):
        """Test with custom min/max values."""
        data = np.array([2.0, 4.0, 6.0])
        result = normalize_01(data, min_val=0.0, max_val=10.0)
        expected = np.array([0.2, 0.4, 0.6])
        np.testing.assert_allclose(result, expected)

    def test_single_element(self):
        """Test single-element array."""
        data = np.array([42.0])
        result = normalize_01(data)
        # Single element has zero range
        expected = np.array([0.0])
        np.testing.assert_allclose(result, expected)

    def test_list_input(self):
        """Test that list input is accepted and converted."""
        data = [1.0, 2.0, 3.0]
        result = normalize_01(data)
        expected = np.array([0.0, 0.5, 1.0])
        np.testing.assert_allclose(result, expected)

    def test_preserves_shape(self):
        """Test that output shape matches input shape."""
        data = np.random.randn(10, 5, 3)
        result = normalize_01(data)
        assert result.shape == data.shape

    def test_axis_with_constant_slices(self):
        """Test axis normalization when some slices are constant."""
        data = np.array([[1.0, 1.0], [2.0, 6.0]])
        result = normalize_01(data, axis=0)
        # axis=0 normalizes along columns: col0: [1,2]→[0,1], col1: [1,6]→[0,1]
        expected = np.array([[0.0, 0.0], [1.0, 1.0]])
        np.testing.assert_allclose(result, expected)
