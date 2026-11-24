"""Public regression tests for the arrhythmia classification reference."""

import json
from pathlib import Path

import numpy as np
from reference_solution import (
    FEATURE_COLUMNS,
    fit_predict,
    make_fixture,
    patient_independent_oof,
)

EXPECTED = Path(__file__).resolve().parents[1] / "expected_output.json"


def test_reference_metrics_match_expected():
    frame = make_fixture()
    result = patient_independent_oof(
        frame[list(FEATURE_COLUMNS)],
        frame["label"].to_numpy(),
        frame["patient_id"].to_numpy(),
    )
    expected = json.loads(EXPECTED.read_text(encoding="utf-8"))["metrics"]
    for key, value in expected.items():
        assert abs(result["metrics"][key] - value) <= 0.03
    assert result["metrics"]["balanced_accuracy"] >= 0.80


def test_every_fold_is_patient_independent():
    frame = make_fixture()
    result = patient_independent_oof(
        frame[list(FEATURE_COLUMNS)],
        frame["label"].to_numpy(),
        frame["patient_id"].to_numpy(),
    )
    assert len(result["fold_groups"]) == 4
    for fold in result["fold_groups"]:
        assert not set(fold["train"]) & set(fold["test"])


def test_fit_predict_uses_only_training_rows():
    frame = make_fixture()
    train = frame[frame.patient_id < "P006"]
    test = frame[frame.patient_id >= "P006"]
    a = fit_predict(
        train[list(FEATURE_COLUMNS)],
        train["label"].to_numpy(),
        test[list(FEATURE_COLUMNS)],
    )
    noisy_test = test[list(FEATURE_COLUMNS)].copy()
    noisy_test["label"] = np.random.default_rng(0).permutation(test["label"].to_numpy())
    b = fit_predict(train[list(FEATURE_COLUMNS)], train["label"].to_numpy(), noisy_test)
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
