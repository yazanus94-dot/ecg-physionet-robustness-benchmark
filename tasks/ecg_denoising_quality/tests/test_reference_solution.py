"""Public regression tests for the denoising reference solution and grader."""

import json
from pathlib import Path

import numpy as np
from grader import aggregate, evaluate_output, match_peaks_one_to_one
from reference_solution import SCENARIOS, denoise, make_fixture, make_scenarios

HERE = Path(__file__).resolve().parents[1]
EXPECTED = HERE / "expected_output.json"


def test_reference_meets_quality_criteria_on_every_scenario():
    for fixture in make_scenarios():
        output = denoise(fixture["noisy"], fixture["fs"])
        assert output.shape == fixture["noisy"].shape
        assert np.isfinite(output).all()
    metrics = {}
    for name in SCENARIOS:
        f = make_fixture(name)
        out = denoise(f["noisy"], f["fs"])
        m = evaluate_output(f["clean"], f["noisy"], out, f["fs"], f["r_peaks"])
        m["n_samples"] = len(out)
        metrics[name] = m
    agg = aggregate(metrics)
    assert agg["median_snr_gain_db"] >= 1.5
    assert agg["mean_correlation"] >= 0.85
    assert agg["weighted_peak_f1"] >= 0.80


def test_expected_output_is_reproducible_and_informational():
    payload = json.loads(EXPECTED.read_text(encoding="utf-8"))
    assert set(payload["scenarios"]) == set(SCENARIOS)
    assert payload["note"].startswith("Reference metrics only")
    for name in SCENARIOS:
        f = make_fixture(name)
        out = denoise(f["noisy"], f["fs"])
        m = evaluate_output(f["clean"], f["noisy"], out, f["fs"], f["r_peaks"])
        stored = payload["scenarios"][name]
        for key in ("snr_gain_db", "correlation", "peak_f1"):
            assert abs(m[key] - stored[key]) <= 0.02


def test_fixtures_are_deterministic():
    for name in SCENARIOS:
        a, b = make_fixture(name), make_fixture(name)
        assert np.array_equal(a["clean"], b["clean"])
        assert np.array_equal(a["noisy"], b["noisy"])


def test_one_to_one_peak_matching_counts_each_detection_once():
    detected = np.array([100, 105, 300])
    reference = np.array([102, 200, 302])
    count, matched = match_peaks_one_to_one(detected, reference, 10)
    assert count == 2
    assert sorted(matched.tolist()) == [0, 2]
