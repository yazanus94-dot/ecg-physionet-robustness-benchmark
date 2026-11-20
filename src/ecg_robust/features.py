"""Scientific feature extraction for heartbeat segments.

Every feature documents its physiological or mathematical interpretation.
Features computed per segment:

Basic statistics
----------------
- ``mean``: arithmetic mean of the segment (mV).
- ``std``: population standard deviation of the segment (mV).
- ``min`` / ``max``: minimum and maximum amplitude (mV).
- ``energy``: signal energy ``E = sum(x[n]**2)`` (mV^2).
- ``peak_amplitude``: maximum absolute amplitude ``max(|x|)`` (mV).
- ``duration_s``: waveform duration ``n / fs`` (s).

Morphology
----------
- ``rise_time_s``: time from the segment start to the first point reaching
  the first-half maximum amplitude (s).
- ``fall_time_s``: time from the second-half maximum crossing to the
  segment end (s).
- ``zero_crossings``: number of sign changes in the segment.

RR information
--------------
- ``rr_s``: RR interval ``RR_i = (r_i - r_{i-1}) / fs`` (s).
- ``rr_ratio``: RR ratio ``RR_i / RR_{i-1}`` (dimensionless; zero
  intervals are guarded against).

Frequency domain
----------------
- ``spectral_centroid``: ``sum(f * |X(f)|) / sum(|X(f)|)`` (Hz) of the
  segment's DFT (FFT of the zero-mean-adjusted segment).
- ``spectral_flatness``: ``exp(mean(ln|X|)) / mean(|X|)`` in [0, 1]
  (energy distribution across frequencies; 1 = flat).
- ``fft_peak_frequency``: frequency of the largest DFT magnitude (Hz).
- ``fft_total_energy``: ``sum(|X(f)|^2)`` of the DFT (mV^2-Hz).

Statistical
-----------
- ``skew``: skewness of the segment samples (Fisher, scipy default).
- ``kurtosis``: kurtosis of the segment samples (Fisher, scipy default).
- ``p5``, ``p25``, ``p50``, ``p75``, ``p95``: percentiles of the segment
  samples (5th, 25th, median, 75th, 95th) (mV).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis, skew


def extract_features(
    segment: np.ndarray,
    fs: float,
    rr: float | None = None,
    rr_ratio: float | None = None,
) -> dict[str, float]:
    """Compute the feature dict for one heartbeat segment (see module doc)."""
    seg = np.asarray(segment, dtype=float)
    n = len(seg)
    if n < 4:
        raise ValueError("segment too short for feature extraction")
    if not np.all(np.isfinite(seg)):
        raise ValueError("segment contains non-finite values (missing/invalid)")

    half = n // 2
    baseline = float(np.mean(seg[: max(1, int(0.05 * n))]))

    # morphological: rise to first-half max, fall from second-half max
    rise_s = 0.0
    if half > 0:
        target_rise = float(seg[:half].max())
        idx = int(
            np.argmax(seg[:half] >= target_rise - 1e-9 * (abs(target_rise) + 1.0))
        )
        rise_s = idx / fs
    fall_s = 0.0
    if half < n:
        target_fall = float(seg[half:].max())
        idx2 = half + int(
            np.argmax(seg[half:] >= target_fall - 1e-9 * (abs(target_fall) + 1.0))
        )
        fall_s = (n - idx2) / fs
    n_zc = int(np.sum(np.signbit(seg[1:]) != np.signbit(seg[:-1])))

    # frequency-domain features (rFFT of the zero-mean segment)
    seg_z = seg - baseline
    X = np.abs(np.fft.rfft(seg_z))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    total = float(X.sum())
    centroid = float(np.sum(freqs * X) / total) if total > 0 else 0.0
    mean_x = float(X.mean())
    flatness = (
        float(np.exp(np.mean(np.log(X + 1e-300))) / mean_x) if mean_x > 0 else 0.0
    )
    f_peak = float(freqs[int(np.argmax(X))]) if len(freqs) else 0.0

    feats = {
        "mean": float(seg.mean()),
        "std": float(seg.std()),
        "min": float(seg.min()),
        "max": float(seg.max()),
        "energy": float(np.sum(seg**2)),
        "peak_amplitude": float(np.max(np.abs(seg))),
        "duration_s": float(n / fs),
        "rise_time_s": float(rise_s),
        "fall_time_s": float(fall_s),
        "zero_crossings": n_zc,
        "spectral_centroid": centroid,
        "spectral_flatness": flatness,
        "fft_peak_frequency": f_peak,
        "fft_total_energy": float(np.sum(X**2)),
        "skew": float(skew(seg, bias=False)),
        "kurtosis": float(kurtosis(seg, bias=False)),
    }
    for p in (5, 25, 50, 75, 95):
        feats[f"p{p}"] = float(np.percentile(seg, p))
    if rr is not None:
        feats["rr_s"] = float(rr)
    if rr_ratio is not None:
        feats["rr_ratio"] = float(rr_ratio)
    return feats
