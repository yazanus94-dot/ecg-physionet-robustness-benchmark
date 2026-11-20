"""Tests for AAMI annotation mapping."""

import pytest

from ecg_robust.annotations import (
    AAMI_MAP,
    CLASSES,
    annotate_annotations,
    annotation_report,
    classify_annotations,
)


def test_normal_codes_map_to_n():
    mapped, _excl, _unk = classify_annotations(["N", "L", "R", "e", "j"])
    assert mapped == ["N"] * 5


def test_supraventricular_codes_map_to_s():
    mapped, _excl, _unk = classify_annotations(["A", "a", "J", "S"])
    assert mapped == ["S"] * 4


def test_v_and_f_map():
    mapped, _e, _u = classify_annotations(["V", "E"])
    assert mapped == ["V", "V"]
    mapped, _e, _u = classify_annotations(["F"])
    assert mapped == ["F"]


def test_q_codes_map():
    mapped, _e, _u = classify_annotations(["/", "f", "Q"])
    assert mapped == ["Q", "Q", "Q"]


def test_excluded_known_mitdb_symbols():
    # T, P, U, x, ? are all part of the MITDB database but excluded by the AAMI map
    _m, excluded, unknown = classify_annotations(["T", "P", "U", "x", "?"])
    assert sum(excluded.values()) == 5
    assert sum(unknown.values()) == 0


def test_unknown_codes_are_reported():
    # 'c' is not part of the MITDB database at all
    mapped, excluded, unknown = classify_annotations(["c"])
    assert mapped == []
    assert sum(excluded.values()) == 0
    assert sum(unknown.values()) == 1


def test_annotation_report_keys():
    mapped, excl, unk = classify_annotations(["N", "T", "c"])
    rep = annotation_report(mapped, excl, unk)
    assert set(rep) == {"class_counts", "total_mapped", "excluded", "unknown"}
    assert rep["class_counts"]["N"] == 1
    assert sum(rep["excluded"].values()) == 1
    assert sum(rep["unknown"].values()) == 1
    assert CLASSES == ["N", "S", "V", "F", "Q"]


def test_aami_map_has_five_classes():
    assert sorted(set(AAMI_MAP.values())) == ["F", "N", "Q", "S", "V"]
    assert len(AAMI_MAP) == 15


def test_pipeline_total_excluded_formula():
    # pipeline.py computes totals by summing the report dicts — sanity check the math
    rep = annotation_report(["N", "N"], {"T": 2}, {"x": 0})
    assert sum(int(v) for v in rep["excluded"].values()) == 2
    assert sum(int(v) for v in rep["unknown"].values()) == 0


@pytest.mark.parametrize(
    "code,cls",
    [
        ("N", "N"),
        ("L", "N"),
        ("R", "N"),
        ("e", "N"),
        ("j", "N"),
        ("A", "S"),
        ("a", "S"),
        ("J", "S"),
        ("S", "S"),
        ("V", "V"),
        ("E", "V"),
        ("F", "F"),
        ("/", "Q"),
        ("f", "Q"),
        ("Q", "Q"),
    ],
)
def test_every_mapped_code(code, cls):
    mapped, _e, _u = classify_annotations([code])
    assert mapped == [cls]


# ---------------------------------------------------------------------------
# Phase 4 -- input validation for annotations (spec #4 of master spec)
# ---------------------------------------------------------------------------


def test_annotate_annotations_matching_arrays_ok():
    recs = annotate_annotations(["N", "L", "R"], samples=[10, 20, 30])
    assert [r.index for r in recs] == [0, 1, 2]
    assert [r.sample for r in recs] == [10, 20, 30]
    assert [r.status for r in recs] == ["mapped", "mapped", "mapped"]


def test_annotate_annotations_empty_arrays_ok():
    recs = annotate_annotations([], samples=[])
    assert recs == []
    recs = annotate_annotations([])
    assert recs == []


def test_annotate_annotations_length_mismatch_raises():
    from ecg_robust.annotations import annotate_annotations

    with pytest.raises(ValueError, match="same length"):
        annotate_annotations(["N", "N"], samples=[0])
    with pytest.raises(ValueError, match="same length"):
        annotate_annotations(["N"], samples=[0, 1, 2])


def test_annotate_annotations_negative_position_raises():
    from ecg_robust.annotations import annotate_annotations

    with pytest.raises(ValueError, match="negative"):
        annotate_annotations(["N"], samples=[-1])


def test_annotate_annotations_non_integer_position_raises():
    from ecg_robust.annotations import annotate_annotations

    with pytest.raises(ValueError, match="integer"):
        annotate_annotations(["N"], samples=[0.5])
    with pytest.raises(ValueError, match="integer"):
        annotate_annotations(["N"], samples=["abc"])


def test_annotate_annotations_preserves_original_index():
    from ecg_robust.annotations import annotate_annotations

    recs = annotate_annotations(["N", "T", "N"], samples=[100, 200, 300])
    assert [r.index for r in recs] == [0, 1, 2]
    assert [r.symbol for r in recs] == ["N", "T", "N"]
    assert recs[2].aami_class == "N"


def test_annotate_annotations_unknown_and_excluded_statuses():
    from ecg_robust.annotations import annotate_annotations

    recs = annotate_annotations(["c", "T", "N"], samples=[5, 10, 15])
    assert recs[0].status == "unknown" and recs[0].aami_class is None
    assert recs[1].status == "excluded" and recs[1].aami_class is None
    assert recs[2].status == "mapped" and recs[2].aami_class == "N"


def test_annotate_annotations_length_mismatch_does_not_silently_truncate():
    # Python zip() silently stops at the shorter input -- the validation
    # must raise instead (spec #4: "Do not silently drop annotation entries").
    from ecg_robust.annotations import annotate_annotations

    symbols = ["N", "N", "N", "N"]
    with pytest.raises(ValueError):
        annotate_annotations(symbols, samples=[0, 1, 2])


def test_annotate_annotations_statuses_and_alignment():
    from ecg_robust.annotations import annotate_annotations

    symbols = ["N", "L", "T", "A", "c", "Q"]
    samples = [0, 360, 720, 1080, 1440, 1800]
    recs = annotate_annotations(symbols, samples=samples)
    assert [r.status for r in recs] == [
        "mapped",
        "mapped",
        "excluded",
        "mapped",
        "unknown",
        "mapped",
    ]
    assert [r.aami_class for r in recs] == ["N", "N", None, "S", None, "Q"]
    assert [r.index for r in recs] == [0, 1, 2, 3, 4, 5]
    assert [r.sample for r in recs] == samples
    assert [r.symbol for r in recs] == symbols


def test_annotate_annotations_no_shift_on_mixed_sequence():
    # Spec #1: excluded/unknown symbols appear explicitly without shifting
    # subsequent labels; every record keeps its original index/sample/symbol.
    from ecg_robust.annotations import annotate_annotations, classify_annotations

    symbols = ["N", "T", "A", "V", "c", "L"]
    recs = annotate_annotations(symbols, samples=[10, 20, 30, 40, 50, 60])
    for i, r in enumerate(recs):
        assert r.index == i
        assert r.symbol == symbols[i]
        assert r.sample == [10, 20, 30, 40, 50, 60][i]
    mapped, excluded, unknown = classify_annotations(symbols)
    assert mapped == ["N", "S", "V", "N"]  # mapped symbols only, in original order
    assert sum(excluded.values()) == 1 and excluded["T"] == 1
    assert sum(unknown.values()) == 1 and unknown["c"] == 1


def test_annotate_annotations_signal_length_bounds():
    from ecg_robust.annotations import annotate_annotations

    # In-bounds positions are accepted when the signal length is available.
    recs = annotate_annotations(["N", "N"], samples=[0, 99], signal_length=100)
    assert [r.sample for r in recs] == [0, 99]
    with pytest.raises(ValueError, match="outside the signal bounds"):
        annotate_annotations(["N"], samples=[100], signal_length=100)
    with pytest.raises(ValueError, match="outside the signal bounds"):
        annotate_annotations(["N"], samples=[200], signal_length=100)


def test_annotate_annotations_integral_float_positions_accepted():
    from ecg_robust.annotations import annotate_annotations

    recs = annotate_annotations(["N", "A"], samples=[10.0, 20.0])
    assert [r.sample for r in recs] == [10, 20]


def test_annotate_annotations_numpy_integer_positions_accepted():
    import numpy as np

    from ecg_robust.annotations import annotate_annotations

    recs = annotate_annotations(["N", "N"], samples=np.array([5, 15], dtype=np.int64))
    assert [r.sample for r in recs] == [5, 15]
    assert [r.index for r in recs] == [0, 1]
