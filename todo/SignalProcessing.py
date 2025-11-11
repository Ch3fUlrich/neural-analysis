from typing import Union, List, Optional
import numpy as np
from scipy.signal import butter, filtfilt, welch
import matplotlib.pyplot as plt

from Visualizer import *
from Helper import *
from scipy.signal import welch
from scipy.signal import savgol_filter


def butter_lowpass(cutoff: float, fs: float, order: int = 2):
    """
    Design a lowpass Butterworth filter.

    Parameters:
        cutoff (float): The cutoff frequency in Hertz.
        fs (float): The sampling frequency in Hertz.
        order (int, optional): The filter order. Defaults to 5.

    Returns:
        b (array-like): Numerator (zeros) coefficients of the filter.
        a (array-like): Denominator (poles) coefficients of the filter.
    """

    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    if normal_cutoff >= 1:
        normal_cutoff = 0.999999
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


def butter_lowpass_filter(
    data, cutoff: Union[float, List[float]], fs: float, order: int = 2
):
    """
    Apply a lowpass Butterworth filter to the input data.

    Parameters:
        data (array-like): The input data to filter.
        cutoff (float): The cutoff frequency in Hertz.
        fs (float): The sampling frequency in Hertz.
        order (int, optional): The filter order. Defaults to 5.

    Returns:
        y (array-like): The filtered output data.
    """
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def may_butter_lowpass_filter(
    data,
    smooth: bool = True,
    cutoff_percentage: float = 0.999,
    cutoff: Union[float, List[float]] = 2,
    fps: float = None,
    order: int = 2,
):
    if data.ndim == 1:
        data = data.reshape(-1, 1)  # Ensure data is 2D
    data_smoothed = data
    if smooth:
        if cutoff is None:
            (freq, fft_vals), (freqs, psd, cutoff) = fft_psd(
                data, 20, plot=False, cutoff=cutoff_percentage
            )
            global_logger.debug(f"Found cutoff frequency cutoff for channels: {cutoff}")
        if not fps or fps == 0:
            global_logger.debug("No fps provided smoothing not possible")
        else:
            global_logger.debug(
                f"Applying Butterworth Lowpass filter with cutoff={cutoff}, fps={fps}, order={order}"
            )
            for i in range(data.shape[1]):
                if is_array_like(cutoff):
                    if len(cutoff) == data.shape[1]:
                        cf = cutoff[i]
                    else:
                        cf = cutoff[0]
                else:
                    cf = cutoff
                # get min and max of data
                min_d = np.min(data[:, i])
                max_d = np.max(data[:, i])
                data_smoothed[:, i] = butter_lowpass_filter(
                    data[:, i], cutoff=cf, fs=fps, order=order
                )
                # Ensure the filtered data is within the original range
                data_smoothed[:, i] = np.clip(data_smoothed[:, i], min_d, max_d)
    return data_smoothed


def fastfouriertransform(
    data: np.ndarray, fps: float, plot: bool = False, ax: Optional[plt.Axes] = None
):
    """
    Computes the Fast Fourier Transform (FFT) of the input data and optionally plots the results.

    Args:
        data (np.ndarray): Input data to be transformed with shape (n_samples, n_features).
        fps (float): Sampling frequency of the data.
        plot (bool): Whether to plot the FFT results.
        ax (Optional[plt.Axes]): Axis to plot on. If None, a new figure is created.

    Returns:
        tuple: Frequencies and FFT values.
    """
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    n = len(data)
    freq = np.fft.fftfreq(n, d=1 / fps)[: n // 2]  # Positive frequencies
    fft_vals = np.abs(np.fft.fft(data, axis=0))[: n // 2] / n  # Normalized amplitude

    if plot or ax is not None:
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
        else:
            fig = ax.figure

        ax.plot(freq, fft_vals)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Amplitude")
        ax.set_title("Fast Fourier Transformation")
        ax.grid(alpha=0.2)
        if plot:
            plt.tight_layout()
            plt.show()

    return freq, fft_vals


def powersspectraldensity(
    data: np.ndarray,
    fps: float,
    plot: bool = False,
    ax: Optional[plt.Axes] = None,
    cutoff: float = 0.99,
):
    """
    Computes the Power Spectral Density (PSD) of the input freq and optionally plots the results.

    Args:
        freq (np.ndarray): Input freq to be transformed.
        fps (float): Sampling frequency of the freq.
        plot (bool): Whether to plot the PSD results.
        ax (Optional[plt.Axes]): Axis to plot on. If None, a new figure is created.
        cutoff (float): Percentage of cumulative power to determine the cutoff frequency (default: 0.9).

    Returns:
        tuple: Frequencies and PSD values.
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

    if plot or ax is not None:
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
        else:
            fig = ax.figure

        for i in range(psd.shape[0]):
            ax.semilogy(freqs, psd[i, :], label=f"Channel {i+1}")

            # get color from line above
            color = ax.get_lines()[-1].get_color()

            # Add cutoff frequency line
            ax.axvline(
                cutoff_freq[i],
                color=color,
                linestyle="--",
                label=f"Cutoff Frequency Channel {i+1}: {cutoff_freq[i]:.2f} Hz (SNR: {snr[i]:.2f} dB)",
            )

        # Configure plot
        ax.legend()
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power (dB)")
        ax.set_title("Power Spectral Density")
        ax.grid(alpha=0.2)

        if plot:
            plt.tight_layout()
            plt.show()

    return freqs, psd, cutoff_freq, snr


def fft_psd(
    data: np.ndarray,
    fps: float,
    plot: bool = True,
    cutoff: float = 0.95,
    figsize: tuple = (20, 10),
):
    """
    Computes the FFT and PSD of the input data and optionally plots the results.

    Parameters:
        data (np.ndarray): Input data to be transformed with shape (n_samples, n_features).
        fps (float): Sampling frequency of the data.
        plot (bool): Whether to plot the FFT and PSD results.
        cutoff (float): Percentage of cumulative power to determine the cutoff frequency (default: 0.9).
        figsize (tuple): Size of the figure for plotting.

    Returns:
        tuple: Frequencies and FFT values, Frequencies and PSD values, Cutoff frequencies, SNR values.

    Example:
        >>> data = np.random.randn(1000, 2)
        >>> fps = 100
        >>> (fft_freq, fft_vals), (psd_freqs, psd_vals, cutoff_freqs) = fft_psd(data, fps, plot=True, cutoff=0.95)
    """
    if plot:
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=figsize)
        # remove outer lines of full plot not axis
    else:
        ax = np.full((2, 2), None)

    ax1 = ax[0, 0]  # Raw data
    ax2 = ax[1, 0]  # Filtered data
    ax3 = ax[0, 1]  # FFT plot
    ax4 = ax[1, 1]  # PSD plot

    freq, fft_vals = fastfouriertransform(data, fps, ax=ax3)
    freqs, psd, cutoff_freq, snr = powersspectraldensity(
        data, fps, ax=ax4, cutoff=cutoff
    )

    if plot:
        for i in range(len(cutoff_freq)):
            # get color from last line
            color = ax3.get_lines()[-(i + i)].get_color()
            ax3.axvline(
                cutoff_freq[i],
                color=color,
                linestyle="--",
                label=f"Cutoff Frequency Channel {i+1}: {cutoff_freq[i]:.2f} Hz (SNR: {snr[i]:.2f} dB)",
            )
            ax3.legend()

        # plot data
        plot_line(
            np.linalg.norm(data, axis=1) if data.ndim > 1 else data,
            # xlabel="Frames",
            ylabel="Amplitude",
            title="Input Data",
            ax=ax1,
            legend=False,
        )
        # plot filtered data
        filtered_data = may_butter_lowpass_filter(
            data,
            smooth=True,
            cutoff=cutoff_freq,
            fps=fps,
            order=2,
        )

        # filtered_data = savgol_filter(
        #     filtered_data.flatten(), window_length=fps, polyorder=3
        # )

        plot_line(
            (
                np.linalg.norm(filtered_data, axis=1)
                if filtered_data.ndim > 1
                else filtered_data
            ),
            xlabel="Frames",
            ylabel="Amplitude",
            title="Filtered Data",
            ax=ax2,
            legend=False,
        )

        fontsize = Vizualizer.auto_fontsize(fig) * 0.8
        fig.suptitle("FFT and PSD Analysis", fontsize=fontsize)

        set_share_axes(ax[:, 0], sharex=True, sharey=True)
        set_share_axes(ax[:, 1], sharex=True)

    return (freq, fft_vals), (freqs, psd, cutoff_freq)
