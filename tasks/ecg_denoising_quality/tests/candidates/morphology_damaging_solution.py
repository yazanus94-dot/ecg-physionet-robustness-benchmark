"""Invalid candidate: aggressive smoothing may raise SNR but destroys QRS.

Should FAIL on morphology / peak-timing criteria.
"""

import numpy as np


def denoise(signal, fs):
    array = np.asarray(signal, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("invalid signal")
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("invalid fs")
    width = int(0.30 * fs)
    if array.size <= width:
        raise ValueError("too short")
    return np.convolve(array, np.ones(width) / width, mode="same")
