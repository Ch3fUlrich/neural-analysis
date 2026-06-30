"""Coverage tests for neural_analysis.utils.signal_processing.filters."""

from __future__ import annotations

import numpy as np

from neural_analysis.utils.signal_processing.filters import (
    butter_lowpass,
    butter_lowpass_filter,
    may_butter_lowpass_filter,
)


def test_butter_lowpass_returns_order2_coeffs() -> None:
    b, a = butter_lowpass(cutoff=2.0, fs=20.0, order=2)
    assert len(b) == 3 and len(a) == 3


def test_butter_lowpass_clamps_cutoff_at_or_above_nyquist() -> None:
    # cutoff >= nyquist -> normal_cutoff >= 1 -> clamped to 0.999999 (line 22)
    b, a = butter_lowpass(cutoff=15.0, fs=20.0, order=2)
    assert np.all(np.isfinite(b)) and np.all(np.isfinite(a))


def test_butter_lowpass_filter_accepts_list_cutoff() -> None:
    rng = np.random.default_rng(0)
    data = rng.normal(size=200)
    # list cutoff -> uses cutoff[0] (line 46)
    y = butter_lowpass_filter(data, cutoff=[2.0], fs=20.0, order=2)
    assert y.shape == data.shape
    assert np.all(np.isfinite(y))


def test_may_butter_smooths_2d_data() -> None:
    rng = np.random.default_rng(0)
    data = rng.normal(size=(200, 2))
    out = may_butter_lowpass_filter(data, smooth=True, cutoff=2.0, fps=20.0)
    assert out.shape == (200, 2)
    # smoothing should reduce variance relative to the noisy input
    assert out.var() <= data.var()


def test_may_butter_reshapes_1d_input() -> None:
    rng = np.random.default_rng(0)
    data = rng.normal(size=128)
    out = may_butter_lowpass_filter(data, smooth=True, cutoff=2.0, fps=20.0)
    assert out.shape == (128, 1)


def test_may_butter_no_fps_skips_smoothing() -> None:
    rng = np.random.default_rng(0)
    data = rng.normal(size=(100, 1))
    # fps == 0 -> "smoothing not possible" branch (line 87); data returned unchanged
    out = may_butter_lowpass_filter(data, smooth=True, cutoff=2.0, fps=0)
    assert np.array_equal(out, data)


from unittest.mock import patch

def test_may_butter_cutoff_none_estimates_from_psd() -> None:
    rng = np.random.default_rng(0)
    data = rng.normal(size=(100, 2))
    
    with patch("neural_analysis.utils.signal_processing.spectrum.powersspectraldensity") as mock_psd:
        # mock returns (freqs, psd, estimated_cutoff, info)
        mock_psd.return_value = (None, None, 5.0, None)
        
        # fps is None, defaults to 20.0
        out = may_butter_lowpass_filter(data, smooth=True, cutoff=None, fps=None)
        
        mock_psd.assert_called_once()
        # Verify fps defaults to 20.0
        assert mock_psd.call_args[1]["fps"] == 20.0
        assert out.shape == (100, 2)


def test_may_butter_cutoff_none_fps_provided() -> None:
    rng = np.random.default_rng(0)
    data = rng.normal(size=(100, 2))
    
    with patch("neural_analysis.utils.signal_processing.spectrum.powersspectraldensity") as mock_psd:
        mock_psd.return_value = (None, None, 5.0, None)
        out = may_butter_lowpass_filter(data, smooth=True, cutoff=None, fps=30.0)
        
        mock_psd.assert_called_once()
        assert mock_psd.call_args[1]["fps"] == 30.0
        assert out.shape == (100, 2)


def test_may_butter_no_smooth() -> None:
    rng = np.random.default_rng(0)
    data = rng.normal(size=(100, 2))
    out = may_butter_lowpass_filter(data, smooth=False)
    assert np.array_equal(out, data)
    assert out is not data


def test_may_butter_cutoff_list_matches_channels() -> None:
    rng = np.random.default_rng(0)
    data = rng.normal(size=(100, 2))
    out = may_butter_lowpass_filter(data, smooth=True, cutoff=[2.0, 3.0], fps=20.0)
    assert out.shape == (100, 2)


def test_may_butter_cutoff_list_mismatches_channels() -> None:
    rng = np.random.default_rng(0)
    data = rng.normal(size=(100, 2))
    out = may_butter_lowpass_filter(data, smooth=True, cutoff=[2.0], fps=20.0)
    assert out.shape == (100, 2)


def test_may_butter_cutoff_ndarray() -> None:
    rng = np.random.default_rng(0)
    data = rng.normal(size=(100, 2))
    out = may_butter_lowpass_filter(data, smooth=True, cutoff=np.array([2.0, 3.0]), fps=20.0)
    assert out.shape == (100, 2)
