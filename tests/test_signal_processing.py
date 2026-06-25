import numpy as np
from neural_analysis.utils.signal_processing.filters import butter_lowpass_filter

def test_butter_lowpass_filter():
    # Simple test for butter_lowpass_filter
    fs = 100
    t = np.linspace(0, 1, fs, endpoint=False)
    # 5 Hz + 20 Hz signal
    data = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 20 * t)

    filtered_data = butter_lowpass_filter(data, cutoff=10, fs=fs, order=2)

    # 5 Hz amplitude should be preserved, 20 Hz should be attenuated
    fft_original = np.abs(np.fft.fft(data))
    fft_filtered = np.abs(np.fft.fft(filtered_data))

    idx_5hz = 5
    idx_20hz = 20

    assert fft_filtered[idx_5hz] > fft_filtered[idx_20hz]
    assert fft_original[idx_20hz] > fft_filtered[idx_20hz]

from neural_analysis.utils.signal_processing.spectrum import fastfouriertransform, powersspectraldensity

def test_fastfouriertransform():
    fs = 100
    t = np.linspace(0, 1, fs, endpoint=False)
    data = np.sin(2 * np.pi * 10 * t)  # 10 Hz signal

    freq, fft_vals = fastfouriertransform(data, fs)

    assert len(freq) == len(fft_vals)
    # The max amplitude should be at 10 Hz
    max_idx = np.argmax(fft_vals[:, 0])
    assert freq[max_idx] == 10.0

def test_powersspectraldensity():
    fs = 100
    t = np.linspace(0, 1, fs, endpoint=False)
    data = np.sin(2 * np.pi * 10 * t) + 0.5 * np.random.randn(fs)

    freqs, psd, cutoff_freq, snr = powersspectraldensity(data, fs, cutoff=0.99)

    assert len(freqs) == psd.shape[1]
    assert len(cutoff_freq) == 1
    assert len(snr) == 1
    # 10 Hz should have high power
    idx_10hz = np.argmin(np.abs(freqs - 10.0))
    assert psd[0, idx_10hz] > np.mean(psd[0])


from neural_analysis.utils.signal_processing.filters import may_butter_lowpass_filter
from neural_analysis.utils.signal_processing.spectrum import fft_psd

def test_may_butter_lowpass_filter():
    fs = 100
    t = np.linspace(0, 1, fs, endpoint=False)
    data = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 20 * t)

    # Simple lowpass apply
    filtered = may_butter_lowpass_filter(data, smooth=True, cutoff=10, fps=fs)
    assert filtered.shape == (100, 1)

def test_fft_psd():
    fs = 100
    t = np.linspace(0, 1, fs, endpoint=False)
    data = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 20 * t)

    (freq, fft_vals), (freqs, psd, cutoff_freq) = fft_psd(data, fps=fs, )

    assert len(freq) > 0
    assert len(freqs) > 0
    assert len(cutoff_freq) == 1

def test_may_butter_lowpass_filter_with_cutoff_none():
    from neural_analysis.utils.signal_processing.filters import may_butter_lowpass_filter
    fs = 100
    t = np.linspace(0, 1, fs, endpoint=False)
    data = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 20 * t)

    # Simple lowpass apply with None to trigger powersspectraldensity estimation
    filtered = may_butter_lowpass_filter(data, smooth=True, cutoff=None, fps=fs)
    assert filtered.shape == (100, 1)
