"""Grader quality tests: prove it distinguishes valid from invalid solutions."""

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


def test_starter_fails():
    report = _grade(TASK / "starter_solution.py")
    assert not report["passed"]
    assert "run_baseline_wander" in _failed(report)


def test_valid_alternative_passes_without_copying_reference():
    report = _grade(CANDIDATES / "alternative_solution.py")
    assert report["passed"], _failed(report)


def test_unchanged_noisy_signal_fails_quality():
    report = _grade(CANDIDATES / "passthrough_solution.py")
    assert not report["passed"]
    assert "median_snr_gain" in _failed(report)


def test_morphology_damage_fails_despite_any_snr_claim():
    report = _grade(CANDIDATES / "morphology_damaging_solution.py")
    assert not report["passed"]
    failures = _failed(report)
    assert {"mean_correlation", "no_catastrophic_morphology"} & failures


def test_invalid_input_handling_is_enforced():
    report = _grade(CANDIDATES / "passthrough_solution.py")
    failures = _failed(report)
    assert any(name.startswith("invalid_") for name in failures)


def test_extreme_values_are_checked():
    report = _grade(TASK / "reference_solution.py")
    assert report["checks"]["extreme_stability"]["passed"]
