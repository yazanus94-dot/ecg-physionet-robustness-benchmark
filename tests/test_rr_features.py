"""Tests for the RR interval / RR ratio indexing convention (spec #1).

Convention (master engineering spec, Phase 1):

- ``beats`` is the sequence of valid (mapped) beats in annotation order.
- ``r_seq[j] = beats[j].r_sample`` (R positions of valid beats).
- ``RR[k] = (r_seq[k+1] - r_seq[k]) / fs`` for k = 0..n-2.
- Beat j (0-based over valid beats):
  - ``rr_s = RR[j-1] = (r_seq[j] - r_seq[j-1]) / fs`` when ``j >= 1``;
    ``None`` for the first beat.
  - ``rr_ratio = RR[j-1] / RR[j-2]`` when ``j >= 2``; ``None`` otherwise.

Non-beat annotations are excluded from the beat sequence — RR features
correspond to the beat's previous *valid* beat, never to an excluded or
unknown annotation's source position.
"""

import numpy as np
import pytest

from ecg_robust.annotations import annotate_annotations
from ecg_robust.segmentation import extract_beats, rr_features_for_valid_beats


def _beats(r_positions, symbols, fs=360.0, x_len=None):
    annot_recs = annotate_annotations(symbols, samples=r_positions)
    x = np.zeros(x_len if x_len is not None else (max(r_positions) + 1000))
    return extract_beats(x, fs, annot_recs)


def test_spec_example_four_valid_positions():
    # Spec example: valid positions r0, r1, r2, r3 ->
    # RR[0]=(r1-r0)/fs, RR[1]=(r2-r1)/fs, RR[2]=(r3-r2)/fs.
    r = [1000, 2000, 3000, 4000]
    beats = _beats(r, ["N", "N", "N", "N"])
    rr = rr_features_for_valid_beats(beats, 360.0)
    assert len(rr) == 4
    # beat 0: no preceding interval
    assert rr[0] == (None, None)
    # beat 1: RR[0] = (r1-r0)/fs; no ratio (RR[-1] missing)
    assert rr[1] == ((r[1] - r[0]) / 360.0, None)
    # beat 2: RR[1] = (r2-r1)/fs; ratio = RR[1]/RR[0]
    assert rr[2][0] == pytest.approx((r[2] - r[1]) / 360.0)
    assert rr[2][1] == pytest.approx((r[2] - r[1]) / (r[1] - r[0]))
    # beat 3: RR[2] = (r3-r2)/fs; ratio = RR[2]/RR[1]
    assert rr[3][0] == pytest.approx((r[3] - r[2]) / 360.0)
    assert rr[3][1] == pytest.approx((r[3] - r[2]) / (r[2] - r[1]))


def test_first_and_second_heartbeat_missing_values():
    beats = _beats([1000, 2000, 3000], ["N", "N", "N"])
    rr = rr_features_for_valid_beats(beats, 360.0)
    assert rr[0] == (None, None)  # no preceding interval
    assert rr[1] == (pytest.approx((2000 - 1000) / 360.0), None)  # RR but no ratio
    assert rr[2][0] == pytest.approx((3000 - 2000) / 360.0)
    assert rr[2][1] == pytest.approx(1.0)  # RR[2]/RR[1]


def test_excluded_and_unknown_between_mapped_beats():
    # Excluded annotation ('T') and unknown ('c') between mapped beats must
    # not shift RR mapping: beat 2's RR spans the excluded annotation.
    r = [1000, 2000, 3000, 4000, 5000]
    symbols = ["N", "T", "N", "c", "N"]
    beats = _beats(r, symbols)
    valid = [b for b in beats if b.aami_class is not None]
    assert [b.r_sample for b in valid] == [1000, 3000, 5000]
    rr = rr_features_for_valid_beats(valid, 360.0)
    assert rr[0] == (None, None)
    assert rr[1][0] == pytest.approx((3000 - 1000) / 360.0)  # spans excluded 'T'
    assert rr[2][0] == pytest.approx((5000 - 3000) / 360.0)  # spans unknown 'c'
    assert rr[2][1] == pytest.approx((5000 - 3000) / (3000 - 1000))


def test_annotation_at_signal_beginning():
    beats = _beats([0, 1000, 2000], ["N", "N", "N"])
    rr = rr_features_for_valid_beats(beats, 360.0)
    assert rr[0] == (None, None)
    assert rr[1][0] == pytest.approx((1000 - 0) / 360.0)


def test_annotation_at_signal_end():
    beats = _beats([19500, 19900], ["N", "N"])
    rr = rr_features_for_valid_beats(beats, 360.0)
    assert rr[1][0] == pytest.approx((19900 - 19500) / 360.0)
    assert rr[0] == (None, None)


def test_segments_dropped_at_signal_boundaries():
    # Third annotation's segment is beyond the signal end -> dropped; the
    # retained beats keep correct original annotation indices.
    fs = 360.0
    annot_recs = annotate_annotations(["N", "N", "N"], samples=[1000, 2000, 9000])
    x = np.zeros(3600)
    beats = extract_beats(x, fs, annot_recs)
    assert [b.index for b in beats] == [0, 1]  # annotation 2 dropped
    rr = rr_features_for_valid_beats(beats, fs)
    assert rr[1][0] == pytest.approx((2000 - 1000) / 360.0)


def test_mapping_back_to_original_annotation_index():
    r = [1000, 2000, 3000, 4000]
    symbols = ["N", "N", "N", "N"]
    annot_recs = annotate_annotations(symbols, samples=r)
    x = np.zeros(20000)
    beats = extract_beats(x, 360.0, annot_recs)
    for j, beat in enumerate(beats):
        assert beat.index == j
        assert beat.r_sample == r[j]
        assert beat.symbol == symbols[j]
        assert beat.aami_class == "N"
    rr = rr_features_for_valid_beats(beats, 360.0)
    assert rr[1][0] == pytest.approx((r[1] - r[0]) / 360.0)


def test_extract_beats_drops_duplicate_r_positions():
    # Duplicate R positions cannot define a beat sequence: the second
    # annotation at the same position is dropped as an invalid beat; retained
    # beats keep their original annotation indices and order (Phase 1).
    fs = 360.0
    annot_recs = annotate_annotations(["N", "N", "N"], samples=[1000, 1000, 3000])
    beats = extract_beats(np.zeros(20000), fs, annot_recs)
    assert [b.r_sample for b in beats] == [1000, 3000]
    assert [b.index for b in beats] == [0, 2]
    rr = rr_features_for_valid_beats(beats, fs)
    assert rr[1][0] == pytest.approx((3000 - 1000) / 360.0)


def test_extract_beats_drops_out_of_order_r_positions():
    # An R position that does not follow the previous retained beat is
    # dropped (beat sequences must be strictly increasing).
    fs = 360.0
    annot_recs = annotate_annotations(["N", "N", "N"], samples=[3000, 1000, 2000])
    beats = extract_beats(np.zeros(20000), fs, annot_recs)
    assert [b.r_sample for b in beats] == [3000]
