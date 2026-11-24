"""Reference solution for the ECG denoising and signal-quality task.

Self-contained: NumPy/SciPy only. All signals are synthetic and deterministic;
no PhysioNet download or hidden label is used by the grader.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt


def make_fixture(fs: int = 360, seconds: int = 10, snr_db: float = 0.0) -> dict:
    """Return a deterministic clean/noisy ECG pair with R-peak positions."""
    rng = np.random.default_rng(20261124)
    n = fs * seconds
    t = np.arange(n) / fs
    r_peaks = np.arange(int(0.6 * fs), n - int(0.6 * fs), int(0.8 * fs))

    clean = 0.06 * np.sin(2 * np.pi * 0.25 * t) + 0.01 * rng.normal(size=n)
    for r in r_peaks:
        local = np.arange(-int(0.12 * fs), int(0.18 * fs) + 1)
        idx = r + local
        valid = (idx >= 0) & (idx < n)
        idx = idx[valid]
        local = local[valid]

        def gauss(
            offset: np.ndarray, mu: float, sigma: float, amp: float
        ) -> np.ndarray:
            return amp * np.exp(-0.5 * ((offset - mu) / sigma) ** 2)

        clean[idx] += (
            gauss(local, -0.045 * fs, 0.012 * fs, -0.12)
            + gauss(local, 0, 0.008 * fs, 1.0)
            + gauss(local, 0.035 * fs, 0.018 * fs, -0.25)
            + gauss(local, 0.12 * fs, 0.035 * fs, 0.18)
        )

    # Predominantly out-of-band corruption: baseline wander (<0.5 Hz),
    # 50 Hz powerline interference and a small in-band white component.
    # A correct bandpass can therefore produce a large SNR gain.
    noise = (
        1.4 * np.sin(2 * np.pi * 0.15 * t)
        + 0.7 * np.sin(2 * np.pi * 50.0 * t)
        + 0.02 * rng.normal(size=n)
    )
    signal_power = np.mean(clean**2)
    noise_power = np.mean(noise**2)
    scale = np.sqrt(signal_power / (noise_power * 10 ** (snr_db / 10)))
    noisy = clean + scale * noise
    return {
        "fs": fs,
        "clean": clean,
        "noisy": noisy,
        "r_peaks": r_peaks,
        "snr_db": snr_db,
    }


def denoise(signal: np.ndarray, fs: float) -> np.ndarray:
    """Remove baseline wander, powerline interference and out-of-band noise.

    Uses a zero-phase 4th-order Butterworth bandpass (0.5-40 Hz) implemented
    as second-order sections. Invalid or too-short inputs raise ValueError;
    no samples are silently invented or dropped.
    """
    array = np.asarray(signal, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("signal must be a non-empty 1-D array")
    if not np.isfinite(array).all():
        raise ValueError("signal contains non-finite samples")
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("fs must be a positive finite number")
    if 40.0 >= fs / 2:
        raise ValueError("40 Hz high-pass cutoff must be below Nyquist")

    sos = butter(4, [0.5, 40.0], btype="bandpass", fs=float(fs), output="sos")
    padlen = 3 * (2 * sos.shape[0] + 1)
    if array.size <= padlen:
        raise ValueError(
            f"signal too short for zero-phase filtering: need > {padlen} samples"
        )
    return sosfiltfilt(sos, array)
