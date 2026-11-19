"""Tests for AAMI annotation mapping."""

import pytest

from ecg_robust.annotations import AAMI_MAP, CLASSES, annotation_report, classify_annotations


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


@pytest.mark.parametrize("code,cls", [("N","N"),("L","N"),("R","N"),("e","N"),("j","N"),
                                     ("A","S"),("a","S"),("J","S"),("S","S"),
                                     ("V","V"),("E","V"),("F","F"),
                                     ("/","Q"),("f","Q"),("Q","Q")])
def test_every_mapped_code(code, cls):
    mapped, _e, _u = classify_annotations([code])
    assert mapped == [cls]
