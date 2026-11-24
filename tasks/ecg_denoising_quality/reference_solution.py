"""Reference solution for the ECG denoising and signal-quality task.

Self-contained: NumPy/SciPy only. Every fixture is synthetic and deterministic;
no PhysioNet data, hidden labels or network access is used. The reference is an
*educational example*, not the only scientifically valid implementation.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

SCENARIOS = (
    "baseline_wander",
    "powerline_50hz",
    "broadband_low_snr",
    "high_snr",
    "short_recording",
    "low_sampling_rate",
    "boundary_peaks",
)


def _morphology(
    fs: int, seconds: int, kind: str, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    n = fs * seconds
    interval = int(0.78 * fs) if kind == "B" else int(0.85 * fs)
    start = int(0.45 * fs) if kind == "B" else int(0.6 * fs)
    end = n - int(0.45 * fs)
    r_peaks = np.arange(start, max(start + 1, end), interval)
    clean = 0.03 * rng.normal(size=n)
    for r in r_peaks:
        local = np.arange(-int(0.12 * fs), int(0.20 * fs) + 1)
        idx = r + local
        valid = (idx >= 0) & (idx < n)
        idx, local = idx[valid], local[valid]
        if kind == "B":
            p, q, s, tw = -0.10, -0.22, -0.30, 0.16
            pw, qw, sw, twd = 0.016, 0.009, 0.022, 0.045
        else:
            p, q, s, tw = -0.12, -0.18, -0.25, 0.20
            pw, qw, sw, twd = 0.014, 0.008, 0.020, 0.038

        def bump(offset: np.ndarray, mu: float, sigma: float, amp: float) -> np.ndarray:
            return amp * np.exp(-0.5 * ((offset - mu * fs) / (sigma * fs)) ** 2)

        clean[idx] += (
            bump(local, p, pw, 0.10)
            + bump(local, q, qw, q)
            + bump(local, 0.0, 0.008, 1.0)
            + bump(local, s, sw, s)
            + bump(local, tw, twd, tw)
        )
    return clean, r_peaks


def make_fixture(
    scenario: str = "default",
    fs: int | None = None,
    seconds: int | None = None,
    snr_db: float | None = None,
) -> dict:
    """Return a deterministic clean/noisy ECG pair for the requested scenario.

    ``make_fixture()`` keeps the original default signature for compatibility.
    ``make_fixture(scenario=...)`` exposes the additional corruption regimes.
    """
    if scenario == "default":
        scenario = "powerline_50hz"
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario: {scenario}")

    presets = {
        "baseline_wander": {
            "fs": 360,
            "seconds": 12,
            "snr_db": 4.0,
            "morph": "A",
            "wander": 1.6,
            "line": 0.0,
            "white": 0.05,
        },
        "powerline_50hz": {
            "fs": 360,
            "seconds": 10,
            "snr_db": 4.0,
            "morph": "B",
            "wander": 1.2,
            "line": 0.8,
            "white": 0.05,
        },
        "broadband_low_snr": {
            "fs": 360,
            "seconds": 9,
            "snr_db": 1.5,
            "morph": "A",
            "wander": 0.4,
            "line": 0.0,
            "white": 1.0,
        },
        "high_snr": {
            "fs": 360,
            "seconds": 10,
            "snr_db": 12.0,
            "morph": "B",
            "wander": 1.0,
            "line": 0.6,
            "white": 0.08,
        },
        "short_recording": {
            "fs": 360,
            "seconds": 3,
            "snr_db": 6.0,
            "morph": "A",
            "wander": 0.8,
            "line": 0.4,
            "white": 0.06,
        },
        "low_sampling_rate": {
            "fs": 250,
            "seconds": 10,
            "snr_db": 5.0,
            "morph": "A",
            "wander": 1.0,
            "line": 0.7,
            "white": 0.06,
        },
        "boundary_peaks": {
            "fs": 360,
            "seconds": 6,
            "snr_db": 5.0,
            "morph": "B",
            "wander": 1.0,
            "line": 0.5,
            "white": 0.06,
        },
    }
    cfg = dict(presets[scenario])
    if fs is not None:
        cfg["fs"] = fs
    if seconds is not None:
        cfg["seconds"] = seconds
    if snr_db is not None:
        cfg["snr_db"] = snr_db

    rate, length = int(cfg["fs"]), int(cfg["seconds"])
    rng = np.random.default_rng(20261124 + len(scenario) * 7)
    clean, r_peaks = _morphology(rate, length, cfg["morph"], rng)
    if scenario == "boundary_peaks":
        r_peaks = np.unique(
            np.clip(
                np.concatenate(
                    [
                        r_peaks,
                        [int(0.18 * rate), length * rate - int(0.18 * rate)],
                    ]
                ),
                0,
                length * rate - 1,
            )
        ).astype(int)
        r_peaks.sort()

    t = np.arange(length * rate) / rate
    line_hz = min(50.0, rate / 2 - 10.0)
    noise = cfg["wander"] * np.sin(2 * np.pi * 0.17 * t)
    if cfg["line"]:
        noise = noise + cfg["line"] * np.sin(2 * np.pi * line_hz * t)
    noise = noise + cfg["white"] * rng.normal(size=len(t))
    scale = np.sqrt(
        np.mean(clean**2) / (np.mean(noise**2) * 10 ** (cfg["snr_db"] / 10))
    )
    noisy = clean + scale * noise

    return {
        "name": scenario,
        "fs": rate,
        "clean": clean,
        "noisy": noisy,
        "r_peaks": r_peaks,
        "snr_db": float(cfg["snr_db"]),
        "morphology": cfg["morph"],
    }


def make_scenarios() -> list[dict]:
    """All deterministic corruption scenarios used by the grader."""
    return [make_fixture(name) for name in SCENARIOS]


def denoise(signal: np.ndarray, fs: float) -> np.ndarray:
    """Remove baseline wander, powerline interference and out-of-band noise.

    Zero-phase 4th-order Butterworth bandpass (0.5-40 Hz) via second-order
    sections. Invalid or too-short inputs raise ValueError; no samples are
    silently invented or dropped.
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
