
import numpy as np
from scipy.signal import butter, filtfilt


def butter_lowpass(cutoff: float, fs: float, order: int = 2):
    """
    Design a lowpass Butterworth filter.

    Parameters:
        cutoff (float): The cutoff frequency in Hertz.
        fs (float): The sampling frequency in Hertz.
        order (int, optional): The filter order. Defaults to 2.

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
    data: np.ndarray, cutoff: float | list[float], fs: float, order: int = 2
) -> np.ndarray:
    """
    Apply a lowpass Butterworth filter to the input data.

    Parameters:
        data (array-like): The input data to filter.
        cutoff (float): The cutoff frequency in Hertz.
        fs (float): The sampling frequency in Hertz.
        order (int, optional): The filter order. Defaults to 2.

    Returns:
        y (array-like): The filtered output data.
    """
    if isinstance(cutoff, list):
        cutoff = cutoff[0]
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def may_butter_lowpass_filter(
    data: np.ndarray,
    smooth: bool = True,
    cutoff_percentage: float = 0.999,
    cutoff: float | list[float] | None = 2.0,
    fps: float | None = None,
    order: int = 2,
) -> np.ndarray:
    """
    Conditionally apply a Butterworth lowpass filter.
    If cutoff is None, it can estimate cutoff from PSD (requires `powersspectraldensity` integration).
    """
    import logging

    from .spectrum import powersspectraldensity

    logger = logging.getLogger(__name__)

    if data.ndim == 1:
        data = data.reshape(-1, 1)  # Ensure data is 2D

    # Copy data to avoid mutating original
    data_smoothed = np.copy(data)

    if smooth:
        if cutoff is None:
            if fps is None:
                fps = 20.0 # Default if not provided but needed for PSD
            _, _, estimated_cutoff, _ = powersspectraldensity(
                data, fps=fps, cutoff=cutoff_percentage
            )
            cutoff = estimated_cutoff
            logger.debug(f"Found cutoff frequency cutoff for channels: {cutoff}")

        if not fps or fps == 0:
            logger.debug("No fps provided smoothing not possible")
        else:
            logger.debug(
                f"Applying Butterworth Lowpass filter with cutoff={cutoff}, fps={fps}, order={order}"
            )
            for i in range(data.shape[1]):
                if isinstance(cutoff, (list, np.ndarray)):
                    cf = cutoff[i] if len(cutoff) == data.shape[1] else cutoff[0]
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
