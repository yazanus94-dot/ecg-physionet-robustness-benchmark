"""A scientifically different denoiser that should PASS the grader.

Uses a causal-style moving-average baseline estimate and light uniform
smoothing - not the reference Butterworth filter.
"""

import numpy as np
from scipy.ndimage import uniform_filter1d


def denoise(signal, fs):
    array = np.asarray(signal, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("signal must be non-empty and 1-D")
    if not np.isfinite(array).all():
        raise ValueError("non-finite samples")
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("fs must be positive")
    if array.size <= 27:  # shorter than a typical zero-phase filter padlen
        raise ValueError("too short")
    window = max(3, int(0.4 * fs))
    cumulative = np.r_[0.0, np.cumsum(array)]
    baseline = np.empty_like(array)
    for i in range(array.size):
        lo = max(0, i - window + 1)
        baseline[i] = (cumulative[i + 1] - cumulative[lo]) / (i - lo + 1)
    detrended = array - baseline
    return uniform_filter1d(detrended, size=max(3, int(0.02 * fs)), mode="nearest")
