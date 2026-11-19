"""Heartbeat segmentation around reference annotation positions.

Per the specification: ``X_i = x[r_i - 0.25*fs : r_i + 0.45*fs]`` — a 250 ms
window before and a 450 ms window after each annotated R position ``r_i``.
Segments truncated at the signal boundaries are handled; invalid (too
short) segments are dropped.
"""

from __future__ import annotations

import numpy as np

BEFORE_S = 0.25  # seconds before r_i
AFTER_S = 0.45   # seconds after r_i
MIN_LENGTH = 3   # minimum segment length (samples) to keep a beat


def extract_segments(
    x: np.ndarray,
    fs: float,
    r_positions: list[int] | np.ndarray,
    before: float = BEFORE_S,
    after: float = AFTER_S,
) -> list[np.ndarray]:
    """Extract heartbeat segments around annotated positions ``r_i``.

    Returns a list of 1-D numpy arrays (the filtered heartbeat segments),
    in the order of the annotations. Segments at the signal edges are
    truncated; degenerate segments are dropped.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    n_pre = int(round(before * fs))
    n_post = int(round(after * fs))
    segments: list[np.ndarray] = []
    for r in r_positions:
        start = max(0, int(r) - n_pre)
        stop = min(int(r) + n_post, n)
        seg = x[start:stop]
        if len(seg) >= MIN_LENGTH:
            segments.append(seg)
    return segments


def rr_intervals(
    r_positions: list[int] | np.ndarray, fs: float
) -> np.ndarray:
    """RR intervals in seconds: ``RR_i = (r_i - r_{i-1}) / fs``."""
    r = np.asarray(r_positions, dtype=float)
    if len(r) < 2:
        return np.array([], dtype=float)
    return np.diff(r) / float(fs)


def rr_ratios(rr: np.ndarray) -> np.ndarray:
    """RR ratios: ``RR_i / RR_{i-1}`` (zero intervals guarded against)."""
    rr = np.asarray(rr, dtype=float)
    if len(rr) < 2:
        return np.array([], dtype=float)
    denom = rr[:-1]
    out = np.zeros(len(rr) - 1, dtype=float)
    for i in range(len(rr) - 1):
        d = denom[i]
        out[i] = rr[i + 1] / d if abs(d) > 1e-12 else 0.0
    return out
