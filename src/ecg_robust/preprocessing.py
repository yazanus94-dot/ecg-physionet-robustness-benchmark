"""Fourth-order Butterworth bandpass filtering of ECG signals (0.5–40 Hz).

Filtering is zero-phase: second-order sections (SOS) with
``scipy.signal.sosfiltfilt`` (forward-backward, no time shift).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

DEFAULT_LOW_HZ = 0.5
DEFAULT_HIGH_HZ = 40.0
DEFAULT_ORDER = 4


def butter_bandpass_sos(
    fs: float,
    low: float = DEFAULT_LOW_HZ,
    high: float = DEFAULT_HIGH_HZ,
    order: int = DEFAULT_ORDER,
) -> np.ndarray:
    """SOS coefficients of the 4th-order Butterworth bandpass filter."""
    nyq = fs / 2.0
    low_n = low / nyq
    high_n = high / nyq
    if not (0 < low_n < high_n < 1):
        raise ValueError(f"invalid bandpass band: [{low_n}, {high_n}] of Nyquist")
    sos = butter(order, [low_n, high_n], btype="bandpass", output="sos")
    return sos


def bandpass_filter(
    x: np.ndarray,
    fs: float,
    low: float = DEFAULT_LOW_HZ,
    high: float = DEFAULT_HIGH_HZ,
    order: int = DEFAULT_ORDER,
) -> np.ndarray:
    """Zero-phase bandpass filtering: ``y[n] = BandpassFilter(x[n])``.

    Uses ``sosfiltfilt`` (SOS coefficients, forward-backward pass).
    """
    sos = butter_bandpass_sos(fs, low, high, order)
    return sosfiltfilt(sos, np.asarray(x, dtype=float), axis=0)


def normalize_segment(segment: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Standardize a heartbeat segment with training-fitted parameters.

    ``segment_norm[i] = (segment[i] - mean) / (std + eps)`` — ``mean`` and
    ``std`` must be fitted on the training data only.
    """
    seg = np.asarray(segment, dtype=float)
    return (seg - mean) / (std + 1e-12)
