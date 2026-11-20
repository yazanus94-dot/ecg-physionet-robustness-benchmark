"""Heartbeat segmentation around reference annotation positions.

Per the specification: ``X_i = x[r_i - 0.25*fs : r_i + 0.45*fs]`` — a 250 ms
window before and a 450 ms window after each annotated R position ``r_i``.
Segments truncated at the signal boundaries are handled; invalid (too
short) segments are dropped.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

BEFORE_S = 0.25  # seconds before r_i
AFTER_S = 0.45  # seconds after r_i
MIN_LENGTH = 3  # minimum segment length (samples) to keep a beat


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
    if fs is None or fs <= 0:
        raise ValueError(f"invalid sampling frequency: {fs}")
    x = np.asarray(x, dtype=float)
    n = len(x)
    n_pre = round(before * fs)
    n_post = round(after * fs)
    segments: list[np.ndarray] = []
    for r in r_positions:
        start = max(0, int(r) - n_pre)
        stop = min(int(r) + n_post, n)
        seg = x[start:stop]
        if len(seg) >= MIN_LENGTH:
            segments.append(seg)
    return segments


class Beat(NamedTuple):
    """A heartbeat segment aligned to its source annotation.

    - ``segment``: the extracted waveform.
    - ``r_sample``: the original R annotation position.
    - ``aami_class``: AAMI class when the annotation was mapped (else ``None``).
    - ``index``: the original annotation index in the annotation list.
    - ``symbol``: the raw MITDB annotation symbol.
    - ``status``: ``"mapped"`` | ``"excluded"`` | ``"unknown"``.
    """

    segment: np.ndarray
    r_sample: int
    aami_class: str | None
    # "index" is overridden by `tuple.index` (a method); required by spec #1.
    index: int  # type: ignore[assignment]
    symbol: str
    status: str


def extract_beats(
    x: np.ndarray,
    fs: float,
    annotation_records,
    before: float = BEFORE_S,
    after: float = AFTER_S,
) -> list[Beat]:
    """Extract heartbeat segments aligned to annotation records.

    One :class:`Beat` per retained annotation; every beat carries the
    original annotation ``index``/``sample``/``symbol``/``class``/``status``,
    so dropped segments never shift alignment (spec #1).

    A beat is retained only when its segment is valid and its R position is
    strictly greater than the previous retained beat's position. Duplicate or
    out-of-order annotations cannot define a beat sequence or a non-negative
    RR interval, so they are dropped as invalid beats. Segments truncated at
    the signal boundaries are kept (possibly shorter than ``n_pre + n_post``).
    """
    if fs is None or fs <= 0:
        raise ValueError(f"invalid sampling frequency: {fs}")
    x = np.asarray(x, dtype=float)
    n = len(x)
    n_pre = round(before * fs)
    n_post = round(after * fs)
    beats: list[Beat] = []
    prev_r: int | None = None
    for rec in annotation_records:
        start = max(0, int(rec.sample) - n_pre)
        stop = min(int(rec.sample) + n_post, n)
        seg = x[start:stop]
        if len(seg) >= MIN_LENGTH and len(seg) <= n_pre + n_post:
            r = int(rec.sample)
            if prev_r is None or r > prev_r:
                beats.append(
                    Beat(
                        seg,
                        rec.sample,
                        rec.aami_class,
                        rec.index,
                        rec.symbol,
                        rec.status,
                    )
                )
                prev_r = r
    return beats


def rr_features_for_valid_beats(beats, fs) -> list[tuple[float | None, float | None]]:
    """RR interval and ratio for each beat in the valid (mapped) beat sequence.

    Convention (master engineering spec, Phase 1):

    - ``beats`` is the sequence of valid (mapped) beats in annotation order
      (non-beat annotations are excluded from the sequence, so RR features
      correspond to the beat's previous *valid* beat, never to an excluded
      or unknown annotation's source position).
    - ``r_seq[j] = beats[j].r_sample`` is the R position of the j-th valid
      beat.
    - ``RR[k] = (r_seq[k+1] - r_seq[k]) / fs`` for k = 0..n-2.
    - For the beat at valid position i (0-based):

      - ``rr_s = RR[i-1] = (r_seq[i] - r_seq[i-1]) / fs`` when ``i >= 1``;
        ``None`` for the first beat.
      - ``rr_ratio = RR[i-1] / RR[i-2]`` when ``i >= 2`` (current-to-previous
        ratio); ``None`` when unavailable.

    Spec example: valid positions r0, r1, r2, r3 →
    ``RR[0] = (r1-r0)/fs, RR[1] = (r2-r1)/fs, RR[2] = (r3-r2)/fs``; beat 1
    gets ``RR[0]``, beat 2 gets ``RR[1]``, beat 3 gets ``RR[2]``, beat 0 gets
    ``None`` (no preceding interval).

    Returns a list aligned with ``beats``: ``(rr_s, rr_ratio)`` tuples of
    floats or ``None`` (documented missing values).
    """
    r_seq = [b.r_sample for b in beats]
    d = (
        np.diff(np.asarray(r_seq, dtype=float))
        if len(r_seq) > 1
        else np.array([], dtype=float)
    )
    out: list[tuple[float | None, float | None]] = []
    for j in range(len(beats)):
        rr_s = float(d[j - 1] / fs) if (1 <= j and j - 1 < len(d)) else None
        rr_ratio = (
            float(d[j - 1] / d[j - 2])
            if (2 <= j and j - 2 < len(d) and d[j - 2] != 0)
            else None
        )
        out.append((rr_s, rr_ratio))
    return out


def rr_intervals(r_positions: list[int] | np.ndarray, fs: float) -> np.ndarray:
    """RR intervals in seconds: ``RR_i = (r_i - r_{i-1}) / fs``.

    ``rrs = np.diff(r)/fs`` — ``rrs[k] = (R_(k+1) - R_k)/fs``.
    """
    if fs is None or fs <= 0:
        raise ValueError(f"invalid sampling frequency: {fs}")
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
