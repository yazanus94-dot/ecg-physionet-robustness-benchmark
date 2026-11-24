"""Public regression tests for the denoising reference solution."""

import json
from pathlib import Path

import numpy as np
from grader import evaluate_output
from reference_solution import denoise, make_fixture

EXPECTED = Path(__file__).resolve().parents[1] / "expected_output.json"


def test_reference_meets_expected_metrics():
    fixture = make_fixture()
    output = denoise(fixture["noisy"], fixture["fs"])
    metrics = evaluate_output(
        fixture["clean"], fixture["noisy"], output, fixture["fs"], fixture["r_peaks"]
    )
    expected = json.loads(EXPECTED.read_text(encoding="utf-8"))["metrics"]
    for key, value in expected.items():
        assert abs(metrics[key] - value) <= 0.02
    assert metrics["snr_improvement_db"] >= 4.0
    assert metrics["correlation"] >= 0.90
    assert metrics["peak_f1"] >= 0.85


def test_fixture_is_deterministic():
    a, b = make_fixture(), make_fixture()
    assert np.array_equal(a["clean"], b["clean"])
    assert np.array_equal(a["noisy"], b["noisy"])
