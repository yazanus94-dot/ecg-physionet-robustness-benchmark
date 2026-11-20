"""Tests for heartbeat segmentation and RR interval math."""

import numpy as np

from ecg_robust.segmentation import extract_segments, rr_intervals, rr_ratios


def _x():
    return np.arange(10000, dtype=float)


def test_segment_length_standard_window():
    fs = 360.0
    r = [3600]  # 10 s
    segs = extract_segments(_x(), fs, r)
    assert len(segs) == 1
    assert len(segs[0]) == round(0.25 * fs) + round(0.45 * fs)  # 252


def test_segment_content_matches_r_window():
    fs = 360.0
    x = _x()
    r = [3600]
    seg = extract_segments(x, fs, r)[0]
    start = int(r[0]) - round(0.25 * fs)
    assert np.allclose(seg, x[start:start + len(seg)])


def test_segment_at_boundary_is_clamped():
    fs = 360.0
    segs = extract_segments(_x(), fs, [50, 9900])
    assert len(segs[0]) >= 3
    assert len(segs[1]) >= 3


def test_segments_drop_too_short():
    # a beat near the signal end produces a truncated/degenerate segment
    fs = 360.0
    segs = extract_segments(_x(), fs, [5])
    assert all(len(s) >= 3 for s in segs)


def test_rr_intervals_formula():
    # RR_i = (r_i - r_{i-1}) / fs  →  [360/360, 1080/360]
    assert np.allclose(rr_intervals([0, 360, 1080], 360.0), [1.0, 2.0])


def test_rr_intervals_empty_for_single_r():
    assert len(rr_intervals([360], 360.0)) == 0


def test_rr_ratios():
    assert np.allclose(rr_ratios(np.array([1.0, 2.0, 1.0])), [2.0, 0.5])


def test_rr_ratios_zero_guard():
    ratios = rr_ratios(np.array([1.0, 0.0, 1.0]))
    assert len(ratios) == 2
    assert ratios[1] == 0.0  # guarded division


def test_extract_beats_alignment_preserved():
    # Spec #1 (acceptance test): mixed annotation sequence; every retained
    # segment keeps its original annotation index, position and AAMI label.
    from ecg_robust.annotations import annotate_annotations
    from ecg_robust.segmentation import extract_beats
    fs = 360.0
    codes = ["N", "T", "A", "V", "?", "Q"]
    samples = [3600, 7200, 10800, 14400, 18000, 21600]
    recs = annotate_annotations(codes, samples=samples)
    x = np.arange(int(4.5 * 360 * 1000), dtype=float)  # long signal
    beats = extract_beats(x, fs, recs)
    assert len(beats) == len(recs)
    for beat, rec in zip(beats, recs):
        assert beat.index == rec.index
        assert beat.r_sample == rec.sample
        assert beat.symbol == rec.symbol
        assert beat.aami_class == rec.aami_class
        assert beat.status == rec.status
        # segment is x[r_i - 0.25*fs : r_i + 0.45*fs] (clamped at 0)
        assert np.allclose(beat.segment, x[beat.r_sample - 90: beat.r_sample - 90 + len(beat.segment)])


def test_extract_beats_dropped_segments_do_not_shift_alignment():
    from ecg_robust.annotations import annotate_annotations
    from ecg_robust.segmentation import extract_beats
    fs = 360.0
    recs = annotate_annotations(["N", "N"], samples=[3600, 7200])
    # signal long enough for the first beat only -> second segment dropped
    x = np.arange(3600 + 90 + 162, dtype=float)
    beats = extract_beats(x, fs, recs)
    assert len(beats) == 1
    assert beats[0].index == 0
    assert beats[0].r_sample == 3600
