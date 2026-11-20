"""Tests for the ECG-ROBUST reference solution (task `tasks/ecg_arrhythmia_robustness`).

Determinism, alignment, AAMI mapping, expected-output consistency.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from reference_solution import make_fixture, process_record, solve_fixture

EXPECTED = Path(__file__).resolve().parents[1] / "expected_output.json"


def test_make_fixture_shape():
    f = make_fixture()
    assert len(f["x"]) == int(f["fs"] * 12)
    assert len(f["r_positions"]) == len(f["symbols"]) == 8
    assert set(f["symbols"]) <= {"N", "S", "V", "F", "Q"}
    assert np.all(np.diff(f["r_positions"]) > 0)


def test_process_record_alignment():
    f = make_fixture()
    df = process_record(f["x"], f["fs"], f["r_positions"], f["symbols"])
    assert len(df) == 8
    for i, (_, row) in enumerate(df.iterrows()):
        assert row["index"] == i
        assert row["r_sample"] == f["r_positions"][i]
        assert row["symbol"] == f["symbols"][i]
        assert row["aami_class"] == {"N": "N", "S": "S", "V": "V", "F": "F", "Q": "Q"}[f["symbols"][i]]
        assert row["status"] == "mapped"
        assert np.isfinite(row["mean"])
        if i >= 1:
            assert row["rr_s"] is not None and np.isclose(row["rr_s"], (f["r_positions"][i] - f["r_positions"][i - 1]) / f["fs"])
        if i >= 2:
            assert row["rr_ratio"] is not None
        # core feature columns are finite for all beats
        assert df[["mean", "std", "energy", "peak_amplitude",
                   "spectral_centroid", "skew"]].notna().all().all()


def test_solve_fixture_deterministic():
    f = make_fixture()
    out1 = solve_fixture(f, model_name="random_forest")
    out2 = solve_fixture(f, model_name="random_forest")
    assert out1["predictions"] == out2["predictions"]
    assert out1["truth"] == out2["truth"]


def test_solve_fixture_all_models_run():
    f = make_fixture()
    for name in ["dummy", "logistic_regression", "random_forest",
                 "hist_gradient_boosting", "svm_rbf"]:
        out = solve_fixture(f, model_name=name)
        assert len(out["predictions"]) == len(out["beats"]) == 8
        assert out["metrics"]["accuracy"] >= 0.0
        assert all(math.isfinite(out["metrics"][k]) for k in
                   ["accuracy", "f1_macro", "f1_weighted", "balanced_accuracy"])


def test_expected_output_matches_solution():
    if not EXPECTED.exists():
        pytest.skip("expected_output.json not generated yet")
    expected = json.loads(EXPECTED.read_text())
    out = solve_fixture(make_fixture())
    assert out["predictions"] == expected["predictions"]
    assert out["truth"] == expected["truth"]
