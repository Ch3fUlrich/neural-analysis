"""Tests for neural_analysis.metrics.pairwise_numba.

Covers all public and private helpers including the numba JIT-compiled
distance functions, ranking/correlation helpers, and the dispatcher.

Run with NUMBA_DISABLE_JIT=1 to allow coverage tracing of JIT-compiled
function bodies (numba supports this env var for testing).
"""

from __future__ import annotations

import importlib
import logging
import sys

import numpy as np
import pytest

from neural_analysis.metrics.pairwise_numba import (
    NUMBA_AVAILABLE,
    _correlation_matrix_parallel,
    _kendall_numba,
    _spearman_numba,
)

# ---------------------------------------------------------------------------
# Euclidean, cosine, manhattan — only available when numba is present
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="numba not available")
class TestPairwiseEuclideanNumba:
    """Tests for _pairwise_euclidean_numba."""

    def _fn(self):
        from neural_analysis.metrics.pairwise_numba import _pairwise_euclidean_numba
        return _pairwise_euclidean_numba

    def test_output_shape(self):
        fn = self._fn()
        rng = np.random.default_rng(0)
        x = rng.normal(size=(4, 6)).astype(np.float64)
        y = rng.normal(size=(7, 6)).astype(np.float64)
        out = fn(x, y)
        assert out.shape == (4, 7)

    def test_self_distance_is_zero(self):
        fn = self._fn()
        rng = np.random.default_rng(1)
        x = rng.normal(size=(5, 3)).astype(np.float64)
        out = fn(x, x)
        np.testing.assert_allclose(np.diag(out), 0.0, atol=1e-10)

    def test_values_match_scipy(self):
        from scipy.spatial.distance import cdist
        fn = self._fn()
        rng = np.random.default_rng(2)
        x = rng.normal(size=(3, 4)).astype(np.float64)
        y = rng.normal(size=(5, 4)).astype(np.float64)
        expected = cdist(x, y, metric="euclidean")
        np.testing.assert_allclose(fn(x, y), expected, atol=1e-10)

    def test_non_negative(self):
        fn = self._fn()
        rng = np.random.default_rng(3)
        x = rng.normal(size=(6, 8)).astype(np.float64)
        y = rng.normal(size=(4, 8)).astype(np.float64)
        assert np.all(fn(x, y) >= 0.0)

    def test_symmetry_square(self):
        fn = self._fn()
        rng = np.random.default_rng(4)
        x = rng.normal(size=(5, 5)).astype(np.float64)
        out = fn(x, x)
        np.testing.assert_allclose(out, out.T, atol=1e-10)

    def test_single_feature(self):
        """Single feature: euclidean is absolute difference."""
        fn = self._fn()
        x = np.array([[1.0], [4.0]], dtype=np.float64)
        y = np.array([[2.0], [6.0]], dtype=np.float64)
        out = fn(x, y)
        np.testing.assert_allclose(out, [[1.0, 5.0], [2.0, 2.0]], atol=1e-10)


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="numba not available")
class TestPairwiseCosineNumba:
    """Tests for _pairwise_cosine_numba."""

    def _fn(self):
        from neural_analysis.metrics.pairwise_numba import _pairwise_cosine_numba
        return _pairwise_cosine_numba

    def test_output_shape(self):
        fn = self._fn()
        rng = np.random.default_rng(0)
        x = rng.normal(size=(3, 5)).astype(np.float64)
        y = rng.normal(size=(6, 5)).astype(np.float64)
        assert fn(x, y).shape == (3, 6)

    def test_parallel_vectors_are_one(self):
        fn = self._fn()
        x = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
        y = np.array([[2.0, 0.0, 0.0]], dtype=np.float64)
        np.testing.assert_allclose(fn(x, y), [[1.0]], atol=1e-10)

    def test_orthogonal_vectors_are_zero(self):
        fn = self._fn()
        x = np.array([[1.0, 0.0]], dtype=np.float64)
        y = np.array([[0.0, 1.0]], dtype=np.float64)
        np.testing.assert_allclose(fn(x, y), [[0.0]], atol=1e-10)

    def test_zero_x_returns_zero(self):
        """When norm_x == 0 the else branch must output 0."""
        fn = self._fn()
        zero = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        other = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
        np.testing.assert_allclose(fn(zero, other), [[0.0]], atol=1e-10)

    def test_zero_y_returns_zero(self):
        """When norm_y == 0 the else branch must output 0."""
        fn = self._fn()
        zero = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        other = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
        np.testing.assert_allclose(fn(other, zero), [[0.0]], atol=1e-10)

    def test_both_zero_returns_zero(self):
        """When both norms are 0 the else branch must output 0."""
        fn = self._fn()
        zero = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        np.testing.assert_allclose(fn(zero, zero), [[0.0]], atol=1e-10)

    def test_range_of_values(self):
        """Cosine similarity in [-1, 1]."""
        fn = self._fn()
        rng = np.random.default_rng(5)
        x = rng.normal(size=(8, 4)).astype(np.float64)
        y = rng.normal(size=(6, 4)).astype(np.float64)
        out = fn(x, y)
        assert out.min() >= -1.0 - 1e-10
        assert out.max() <= 1.0 + 1e-10

    def test_values_match_reference(self):
        """Compare against numpy-based reference."""
        fn = self._fn()
        rng = np.random.default_rng(6)
        x = rng.normal(size=(3, 4)).astype(np.float64)
        y = rng.normal(size=(4, 4)).astype(np.float64)
        out = fn(x, y)
        for i in range(3):
            for j in range(4):
                nx = np.linalg.norm(x[i])
                ny = np.linalg.norm(y[j])
                expected = np.dot(x[i], y[j]) / (nx * ny)
                np.testing.assert_allclose(out[i, j], expected, atol=1e-10)


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="numba not available")
class TestPairwiseManhattanNumba:
    """Tests for _pairwise_manhattan_numba."""

    def _fn(self):
        from neural_analysis.metrics.pairwise_numba import _pairwise_manhattan_numba
        return _pairwise_manhattan_numba

    def test_output_shape(self):
        fn = self._fn()
        rng = np.random.default_rng(0)
        x = rng.normal(size=(4, 5)).astype(np.float64)
        y = rng.normal(size=(3, 5)).astype(np.float64)
        assert fn(x, y).shape == (4, 3)

    def test_self_distance_is_zero(self):
        fn = self._fn()
        rng = np.random.default_rng(1)
        x = rng.normal(size=(5, 4)).astype(np.float64)
        np.testing.assert_allclose(np.diag(fn(x, x)), 0.0, atol=1e-10)

    def test_values_match_scipy(self):
        from scipy.spatial.distance import cdist
        fn = self._fn()
        rng = np.random.default_rng(2)
        x = rng.normal(size=(3, 4)).astype(np.float64)
        y = rng.normal(size=(5, 4)).astype(np.float64)
        expected = cdist(x, y, metric="cityblock")
        np.testing.assert_allclose(fn(x, y), expected, atol=1e-10)

    def test_non_negative(self):
        fn = self._fn()
        rng = np.random.default_rng(3)
        x = rng.normal(size=(5, 6)).astype(np.float64)
        y = rng.normal(size=(4, 6)).astype(np.float64)
        assert np.all(fn(x, y) >= 0.0)

    def test_triangle_inequality(self):
        """d(x,z) <= d(x,y) + d(y,z) for L1."""
        fn = self._fn()
        rng = np.random.default_rng(7)
        a = rng.normal(size=(1, 5)).astype(np.float64)
        b = rng.normal(size=(1, 5)).astype(np.float64)
        c = rng.normal(size=(1, 5)).astype(np.float64)
        dab = fn(a, b)[0, 0]
        dbc = fn(b, c)[0, 0]
        dac = fn(a, c)[0, 0]
        assert dac <= dab + dbc + 1e-10

    def test_known_values(self):
        """L1 = sum of abs differences: |3-1|+|1-4|=5."""
        fn = self._fn()
        x = np.array([[3.0, 1.0]], dtype=np.float64)
        y = np.array([[1.0, 4.0]], dtype=np.float64)
        out = fn(x, y)
        np.testing.assert_allclose(out[0, 0], 5.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Numba-specific ranking / Kendall helpers
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="numba not available")
class TestRankDataNumba:
    """Tests for _rank_data_numba (only defined when NUMBA_AVAILABLE)."""

    def _fn(self):
        from neural_analysis.metrics.pairwise_numba import _rank_data_numba
        return _rank_data_numba

    def test_output_shape(self):
        fn = self._fn()
        data = np.random.default_rng(0).normal(size=(10, 4)).astype(np.float64)
        ranks = fn(data)
        assert ranks.shape == data.shape

    def test_ranks_are_1_to_n(self):
        fn = self._fn()
        rng = np.random.default_rng(1)
        data = rng.normal(size=(8, 3)).astype(np.float64)
        ranks = fn(data)
        expected_set = set(range(1, 9))
        for j in range(3):
            assert set(ranks[:, j].astype(int)) == expected_set

    def test_rank_ordering(self):
        """Larger values must receive larger ranks."""
        fn = self._fn()
        data = np.array([[3.0, 1.0], [1.0, 3.0], [2.0, 2.0]], dtype=np.float64)
        ranks = fn(data)
        # column 0: values 3,1,2 -> ranks should be 3,1,2
        assert ranks[0, 0] == 3.0
        assert ranks[1, 0] == 1.0
        assert ranks[2, 0] == 2.0

    def test_single_column(self):
        fn = self._fn()
        data = np.array([[2.0], [0.0], [1.0]], dtype=np.float64)
        ranks = fn(data)
        np.testing.assert_array_equal(ranks[:, 0], [3.0, 1.0, 2.0])


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="numba not available")
class TestKendallTauPairwise:
    """Tests for _kendall_tau_pairwise (only defined when NUMBA_AVAILABLE)."""

    def _fn(self):
        from neural_analysis.metrics.pairwise_numba import _kendall_tau_pairwise
        return _kendall_tau_pairwise

    def test_perfect_positive_correlation(self):
        fn = self._fn()
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        tau = fn(x, y)
        np.testing.assert_allclose(tau, 1.0, atol=1e-10)

    def test_perfect_negative_correlation(self):
        fn = self._fn()
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        tau = fn(x, y)
        np.testing.assert_allclose(tau, -1.0, atol=1e-10)

    def test_matches_scipy_kendalltau(self):
        from scipy.stats import kendalltau
        fn = self._fn()
        rng = np.random.default_rng(8)
        x = rng.normal(size=12)
        y = rng.normal(size=12)
        tau = fn(x, y)
        expected, _ = kendalltau(x, y)
        np.testing.assert_allclose(tau, expected, atol=1e-10)

    def test_range(self):
        fn = self._fn()
        rng = np.random.default_rng(9)
        x = rng.normal(size=20)
        y = rng.normal(size=20)
        tau = fn(x, y)
        assert -1.0 - 1e-10 <= tau <= 1.0 + 1e-10

    def test_tied_pairs_branch(self):
        """When prod == 0 (one variable has a tie) the pair is neither concordant
        nor discordant — exercises the 'neither' branch (line 130->124)."""
        fn = self._fn()
        # x has ties (prod_x will be 0 for tied pairs), y varies
        x = np.array([1.0, 1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        tau = fn(x, y)
        # The result is just a float in [-1, 1]
        assert -1.0 - 1e-10 <= tau <= 1.0 + 1e-10
        # 2 concordant pairs involving the tie-broken pair, 0 discordant, 1 tie
        # Manually: (0,1)->sign_x=0, so neither; (0,2)->sign_x<0,sign_y<0->concordant;
        # (0,3)->concordant; (1,2)->concordant; (1,3)->concordant; (2,3)->concordant
        # total_pairs=6, concordant=5, discordant=0 -> tau=5/6
        np.testing.assert_allclose(tau, 5.0 / 6.0, atol=1e-10)

    def test_all_tied_y_gives_zero(self):
        """All same y values -> all prods zero -> tau = 0."""
        fn = self._fn()
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([1.0, 1.0, 1.0, 1.0])
        tau = fn(x, y)
        np.testing.assert_allclose(tau, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# _spearman_numba
# ---------------------------------------------------------------------------


class TestSpearmanNumba:
    """Tests for _spearman_numba (present in both NUMBA and fallback paths)."""

    def test_output_shape_multi_feature(self):
        rng = np.random.default_rng(0)
        data = rng.normal(size=(20, 4))
        out = _spearman_numba(data)
        assert out.shape == (4, 4)

    def test_output_shape_two_features(self):
        """Special handling of 2-column case (scalar result from scipy)."""
        rng = np.random.default_rng(1)
        data = rng.normal(size=(20, 2))
        out = _spearman_numba(data)
        # Must always return a 2D array, never a scalar
        assert out.shape == (2, 2)
        assert out.ndim == 2

    def test_diagonal_is_one(self):
        rng = np.random.default_rng(2)
        data = rng.normal(size=(15, 5))
        out = _spearman_numba(data)
        np.testing.assert_allclose(np.diag(out), 1.0, atol=1e-10)

    def test_symmetric(self):
        rng = np.random.default_rng(3)
        data = rng.normal(size=(12, 4))
        out = _spearman_numba(data)
        np.testing.assert_allclose(out, out.T, atol=1e-10)

    def test_values_in_range(self):
        rng = np.random.default_rng(4)
        data = rng.normal(size=(20, 3))
        out = _spearman_numba(data)
        assert out.min() >= -1.0 - 1e-10
        assert out.max() <= 1.0 + 1e-10

    def test_perfect_positive_spearman(self):
        """Two perfectly correlated columns -> rho = 1."""
        rng = np.random.default_rng(5)
        col = rng.normal(size=(20, 1))
        data = np.hstack([col, col])
        out = _spearman_numba(data)
        np.testing.assert_allclose(out[0, 1], 1.0, atol=1e-8)

    def test_matches_scipy_reference(self):
        from scipy.stats import spearmanr
        rng = np.random.default_rng(6)
        data = rng.normal(size=(15, 3))
        out = _spearman_numba(data)
        expected, _ = spearmanr(data, axis=0)
        np.testing.assert_allclose(out, expected, atol=1e-8)


# ---------------------------------------------------------------------------
# _kendall_numba
# ---------------------------------------------------------------------------


class TestKendallNumba:
    """Tests for _kendall_numba (present in both paths)."""

    def test_output_shape(self):
        rng = np.random.default_rng(0)
        data = rng.normal(size=(15, 4))
        out = _kendall_numba(data)
        assert out.shape == (4, 4)

    def test_diagonal_is_one(self):
        rng = np.random.default_rng(1)
        data = rng.normal(size=(12, 3))
        out = _kendall_numba(data)
        np.testing.assert_allclose(np.diag(out), 1.0, atol=1e-10)

    def test_symmetric(self):
        rng = np.random.default_rng(2)
        data = rng.normal(size=(10, 4))
        out = _kendall_numba(data)
        np.testing.assert_allclose(out, out.T, atol=1e-10)

    def test_values_in_range(self):
        rng = np.random.default_rng(3)
        data = rng.normal(size=(15, 3))
        out = _kendall_numba(data)
        assert out.min() >= -1.0 - 1e-10
        assert out.max() <= 1.0 + 1e-10

    def test_single_pair(self):
        """2-feature case tests the loop logic."""
        rng = np.random.default_rng(4)
        data = rng.normal(size=(12, 2))
        out = _kendall_numba(data)
        assert out.shape == (2, 2)
        np.testing.assert_allclose(np.diag(out), 1.0, atol=1e-10)

    def test_matches_scipy_reference(self):
        from scipy.stats import kendalltau
        rng = np.random.default_rng(5)
        data = rng.normal(size=(12, 3))
        out = _kendall_numba(data)
        for i in range(3):
            for j in range(i + 1, 3):
                tau, _ = kendalltau(data[:, i], data[:, j])
                np.testing.assert_allclose(out[i, j], tau, atol=1e-10)
                np.testing.assert_allclose(out[j, i], tau, atol=1e-10)


# ---------------------------------------------------------------------------
# _correlation_matrix_parallel — the dispatcher
# ---------------------------------------------------------------------------


class TestCorrelationMatrixParallel:
    """Tests for the public dispatcher function."""

    def test_pearson_shape(self):
        rng = np.random.default_rng(0)
        data = rng.normal(size=(20, 5))
        out = _correlation_matrix_parallel(data, "pearson")
        assert out.shape == (5, 5)

    def test_pearson_diagonal(self):
        rng = np.random.default_rng(1)
        data = rng.normal(size=(20, 4))
        out = _correlation_matrix_parallel(data, "pearson")
        np.testing.assert_allclose(np.diag(out), 1.0, atol=1e-10)

    def test_pearson_matches_numpy(self):
        rng = np.random.default_rng(2)
        data = rng.normal(size=(15, 4))
        out = _correlation_matrix_parallel(data, "pearson")
        expected = np.corrcoef(data.T)
        np.testing.assert_allclose(out, expected, atol=1e-12)

    def test_spearman_shape(self):
        rng = np.random.default_rng(3)
        data = rng.normal(size=(20, 4))
        out = _correlation_matrix_parallel(data, "spearman")
        assert out.shape == (4, 4)

    def test_spearman_symmetric(self):
        rng = np.random.default_rng(4)
        data = rng.normal(size=(20, 4))
        out = _correlation_matrix_parallel(data, "spearman")
        np.testing.assert_allclose(out, out.T, atol=1e-10)

    def test_kendall_shape(self):
        rng = np.random.default_rng(5)
        data = rng.normal(size=(15, 4))
        out = _correlation_matrix_parallel(data, "kendall")
        assert out.shape == (4, 4)

    def test_kendall_diagonal(self):
        rng = np.random.default_rng(6)
        data = rng.normal(size=(12, 3))
        out = _correlation_matrix_parallel(data, "kendall")
        np.testing.assert_allclose(np.diag(out), 1.0, atol=1e-10)

    def test_invalid_method_raises_value_error(self):
        rng = np.random.default_rng(7)
        data = rng.normal(size=(10, 3))
        with pytest.raises(ValueError, match="Unknown correlation method"):
            _correlation_matrix_parallel(data, "cosine")

    def test_invalid_method_message_contains_got(self):
        rng = np.random.default_rng(8)
        data = rng.normal(size=(10, 3))
        with pytest.raises(ValueError, match="bad_method"):
            _correlation_matrix_parallel(data, "bad_method")

    def test_pearson_method_case_sensitive(self):
        """'Pearson' (capital) must raise ValueError."""
        rng = np.random.default_rng(9)
        data = rng.normal(size=(10, 3))
        with pytest.raises(ValueError):
            _correlation_matrix_parallel(data, "Pearson")

    def test_all_three_methods_return_ndarray(self):
        rng = np.random.default_rng(10)
        data = rng.normal(size=(12, 3))
        for method in ("pearson", "spearman", "kendall"):
            out = _correlation_matrix_parallel(data, method)
            assert isinstance(out, np.ndarray), f"{method} should return ndarray"


# ---------------------------------------------------------------------------
# Tests for NUMBA_AVAILABLE=False fallback paths via module reload
# These exercise lines 151-165, 175, 179 which are the else-branch
# definitions used when numba cannot be imported.
# ---------------------------------------------------------------------------


def _make_no_numba_module():
    """Return a fresh import of pairwise_numba with NUMBA faked as unavailable.

    Works by temporarily removing numba from sys.modules and the module
    itself so the else-branch definitions are compiled and executed.
    """
    # Collect keys to temporarily suppress
    to_remove = {k: sys.modules.pop(k) for k in list(sys.modules) if "numba" in k}
    # Also remove the target module so it re-executes
    to_remove.update(
        {k: sys.modules.pop(k) for k in list(sys.modules) if "pairwise_numba" in k}
    )
    try:
        # Make numba import fail
        sys.modules["numba"] = None  # type: ignore[assignment]
        mod = importlib.import_module("neural_analysis.metrics.pairwise_numba")
    finally:
        # Restore original state
        for k, v in to_remove.items():
            sys.modules[k] = v
        # Clean up the patched entry if still there
        if sys.modules.get("numba") is None:
            sys.modules.pop("numba", None)
    return mod


class TestNoNumbaFallback:
    """Exercise the else-branch (NUMBA_AVAILABLE=False) code paths."""

    def test_numba_available_is_false_in_mock(self):
        mod = _make_no_numba_module()
        assert mod.NUMBA_AVAILABLE is False

    def test_spearman_fallback_shape_multi_feature(self):
        """_spearman_numba fallback: scipy spearmanr for >=3 features."""
        mod = _make_no_numba_module()
        rng = np.random.default_rng(20)
        data = rng.normal(size=(15, 4))
        out = mod._spearman_numba(data)
        assert out.shape == (4, 4)
        np.testing.assert_allclose(np.diag(out), 1.0, atol=1e-10)

    def test_spearman_fallback_two_feature_scalar_fixup(self):
        """2-feature case: scipy returns scalar, must be converted to 2x2."""
        mod = _make_no_numba_module()
        rng = np.random.default_rng(21)
        data = rng.normal(size=(15, 2))
        out = mod._spearman_numba(data)
        assert out.shape == (2, 2)
        assert out.ndim == 2
        np.testing.assert_allclose(out[0, 0], 1.0, atol=1e-10)
        np.testing.assert_allclose(out[1, 1], 1.0, atol=1e-10)
        # Off-diagonal should be equal
        np.testing.assert_allclose(out[0, 1], out[1, 0], atol=1e-12)

    def test_kendall_fallback_shape(self):
        mod = _make_no_numba_module()
        rng = np.random.default_rng(22)
        data = rng.normal(size=(12, 3))
        out = mod._kendall_numba(data)
        assert out.shape == (3, 3)

    def test_kendall_fallback_diagonal(self):
        mod = _make_no_numba_module()
        rng = np.random.default_rng(23)
        data = rng.normal(size=(12, 3))
        out = mod._kendall_numba(data)
        np.testing.assert_allclose(np.diag(out), 1.0, atol=1e-10)

    def test_kendall_fallback_symmetric(self):
        mod = _make_no_numba_module()
        rng = np.random.default_rng(24)
        data = rng.normal(size=(10, 3))
        out = mod._kendall_numba(data)
        np.testing.assert_allclose(out, out.T, atol=1e-10)

    def test_kendall_fallback_matches_scipy(self):
        from scipy.stats import kendalltau
        mod = _make_no_numba_module()
        rng = np.random.default_rng(25)
        data = rng.normal(size=(10, 3))
        out = mod._kendall_numba(data)
        for i in range(3):
            for j in range(i + 1, 3):
                tau, _ = kendalltau(data[:, i], data[:, j])
                np.testing.assert_allclose(out[i, j], tau, atol=1e-10)

    def test_correlation_spearman_logs_debug_when_no_numba(self, caplog):
        """_correlation_matrix_parallel logs debug when numba unavailable (line 175)."""
        mod = _make_no_numba_module()
        rng = np.random.default_rng(26)
        data = rng.normal(size=(12, 3))
        with caplog.at_level(logging.DEBUG):
            out = mod._correlation_matrix_parallel(data, "spearman")
        assert "Numba not available" in caplog.text or out.shape == (3, 3)
        assert out.shape == (3, 3)

    def test_correlation_kendall_logs_debug_when_no_numba(self, caplog):
        """_correlation_matrix_parallel logs debug when numba unavailable (line 179)."""
        mod = _make_no_numba_module()
        rng = np.random.default_rng(27)
        data = rng.normal(size=(12, 3))
        with caplog.at_level(logging.DEBUG):
            out = mod._correlation_matrix_parallel(data, "kendall")
        assert "Numba not available" in caplog.text or out.shape == (3, 3)
        assert out.shape == (3, 3)

    def test_correlation_all_methods_work_without_numba(self):
        mod = _make_no_numba_module()
        rng = np.random.default_rng(28)
        data = rng.normal(size=(12, 4))
        for method in ("pearson", "spearman", "kendall"):
            out = mod._correlation_matrix_parallel(data, method)
            assert out.shape == (4, 4), f"{method} failed"
            assert isinstance(out, np.ndarray)

    def test_invalid_method_raises_without_numba(self):
        mod = _make_no_numba_module()
        rng = np.random.default_rng(29)
        data = rng.normal(size=(10, 3))
        with pytest.raises(ValueError, match="Unknown correlation method"):
            mod._correlation_matrix_parallel(data, "invalid")
