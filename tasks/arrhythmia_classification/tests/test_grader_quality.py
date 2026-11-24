"""Prove the grader rejects invalid patient-independent benchmarks."""

from pathlib import Path

import grader

TASK = Path(__file__).resolve().parents[1]
CANDIDATES = Path(__file__).resolve().parent / "candidates"


def _grade(path):
    return grader.grade(Path(path))


def _failed(report):
    return {name for name, item in report["checks"].items() if not item["passed"]}


def test_reference_passes():
    report = _grade(TASK / "reference_solution.py")
    assert report["passed"], _failed(report)


def test_valid_alternative_passes():
    report = _grade(CANDIDATES / "valid_solution.py")
    assert report["passed"], _failed(report)


def test_starter_fails():
    report = _grade(TASK / "starter_solution.py")
    assert not report["passed"]
    assert "runs" in _failed(report)


def test_all_rows_training_is_rejected():
    report = _grade(CANDIDATES / "all_rows_solution.py")
    assert not report["passed"]
    assert "predictions_from_disjoint_folds" in _failed(report)


def test_fabricated_fold_metadata_is_rejected():
    report = _grade(CANDIDATES / "fabricated_metadata_solution.py")
    assert not report["passed"]
    assert "fold_metadata_exact" in _failed(report)


def test_constant_predictor_is_rejected():
    report = _grade(CANDIDATES / "constant_solution.py")
    assert not report["passed"]
    assert {"balanced_accuracy", "f1_macro"} & _failed(report)


def test_label_leakage_is_rejected():
    report = _grade(CANDIDATES / "label_leak_solution.py")
    assert not report["passed"]
    assert "test_labels_unused" in _failed(report)


def test_incomplete_oof_is_rejected():
    report = _grade(CANDIDATES / "incomplete_oof_solution.py")
    assert not report["passed"]
    assert "predictions_from_disjoint_folds" in _failed(report)


def test_bad_input_handling_is_rejected():
    report = _grade(CANDIDATES / "invalid_handling_solution.py")
    assert not report["passed"]
    assert any(name.startswith("invalid_") for name in _failed(report))


def test_fabricated_metrics_are_rejected():
    report = _grade(CANDIDATES / "fabricated_metrics_solution.py")
    assert not report["passed"]
    assert "metrics_not_fabricated" in _failed(report)
