import numpy as np
from scipy.signal import welch


def fastfouriertransform(
    data: np.ndarray, fps: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the Fast Fourier Transform (FFT) of the input data.

    Args:
        data (np.ndarray): Input data to be transformed with shape (n_samples, n_features).
        fps (float): Sampling frequency of the data.

    Returns:
        tuple: Frequencies and FFT values.
    """
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    n = len(data)
    freq = np.fft.fftfreq(n, d=1 / fps)[: n // 2]  # Positive frequencies
    fft_vals = np.abs(np.fft.fft(data, axis=0))[: n // 2] / n  # Normalized amplitude

    return freq, fft_vals

def powersspectraldensity(
    data: np.ndarray,
    fps: float,
    cutoff: float = 0.99,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the Power Spectral Density (PSD) of the input freq.

    Args:
        data (np.ndarray): Input data to be transformed.
        fps (float): Sampling frequency of the data.
        cutoff (float): Percentage of cumulative power to determine the cutoff frequency (default: 0.99).

    Returns:
        tuple: Frequencies, PSD values, cutoff frequencies, and SNR values.
    """
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    n = data.shape[0]
    if data.shape[0] > data.shape[1]:
        data = data.T

    freqs, psd = welch(data, fs=fps, nperseg=min(n, 256))

    # suggested cutoff frequency
    psd_sum = np.sum(psd, axis=1)
    cumulative_power = np.cumsum(psd, axis=1) / psd_sum[:, np.newaxis]
    # Find the first frequency where cumulative power exceeds the cutoff for each channel
    cutoff_idx = np.argmax(cumulative_power >= cutoff, axis=1)
    cutoff_freq = freqs[cutoff_idx]

    # estimate SNR
    snr = np.zeros_like(cutoff_freq)
    for i, cf in enumerate(cutoff_freq):
        # Calculate signal and noise power
        signal_freq_range = (0, cf)
        noise_freq_range = (cf, freqs[-1])
        signal_mask = (freqs >= signal_freq_range[0]) & (freqs <= signal_freq_range[1])
        noise_mask = (freqs >= noise_freq_range[0]) & (freqs <= noise_freq_range[1])
        signal_power = np.sum(psd[i, signal_mask])
        noise_power = np.sum(psd[i, noise_mask])
        snr[i] = (
            10 * np.log10(signal_power / noise_power)
            if noise_power > 0
            else float("inf")
        )

    return freqs, psd, cutoff_freq, snr

def fft_psd(
    data: np.ndarray,
    fps: float,
    cutoff: float = 0.95,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Computes the FFT and PSD of the input data.

    Parameters:
        data (np.ndarray): Input data to be transformed with shape (n_samples, n_features).
        fps (float): Sampling frequency of the data.
        cutoff (float): Percentage of cumulative power to determine the cutoff frequency.

    Returns:
        tuple: (freq, fft_vals), (freqs, psd, cutoff_freq)
    """
    freq, fft_vals = fastfouriertransform(data, fps)
    freqs, psd, cutoff_freq, snr = powersspectraldensity(
        data, fps, cutoff=cutoff
    )

    return (freq, fft_vals), (freqs, psd, cutoff_freq)
