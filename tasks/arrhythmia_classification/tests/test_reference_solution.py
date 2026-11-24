"""Public regression tests for the classification reference and contract."""

import json
from pathlib import Path

import numpy as np
from reference_solution import (
    FEATURE_COLUMNS,
    compute_metrics,
    expected_folds,
    fit_predict,
    make_fixture,
    patient_independent_oof,
)

HERE = Path(__file__).resolve().parents[1]
EXPECTED = HERE / "expected_output.json"


def _data():
    frame = make_fixture()
    return (
        frame[list(FEATURE_COLUMNS)],
        frame["label"].to_numpy(),
        frame["patient_id"].to_numpy(),
    )


def test_reference_metrics_match_expected():
    X, y, g = _data()
    result = patient_independent_oof(X, y, g)
    expected = json.loads(EXPECTED.read_text(encoding="utf-8"))
    for key, value in expected["metrics"].items():
        assert abs(result["metrics"][key] - value) <= 0.02
    assert result["metrics"]["balanced_accuracy"] >= 0.65
    assert (
        result["metrics"]["balanced_accuracy"]
        >= expected["dummy_metrics"]["balanced_accuracy"] + 0.15
    )


def test_fixture_has_realistic_imperfection():
    frame = make_fixture()
    assert frame["label"].nunique() == 4
    assert frame["label"].value_counts().max() / len(frame) > 0.4  # imbalance
    assert frame[list(FEATURE_COLUMNS)].isna().sum().sum() > 0  # missing values
    assert frame["patient_id"].nunique() >= 18


def test_canonical_folds_are_patient_disjoint_and_exhaustive():
    X, y, g = _data()
    folds = expected_folds(g, y)
    assert len(folds) == 4
    seen = []
    for train_idx, test_idx in folds:
        assert not set(g[train_idx]) & set(g[test_idx])
        seen.extend(test_idx.tolist())
    assert sorted(seen) == list(range(len(X)))


def test_oof_delegates_to_fit_predict():
    X, y, g = _data()
    expected = patient_independent_oof(X, y, g)
    assembled = np.empty(len(y), dtype=object)
    for train_idx, test_idx in expected_folds(g, y):
        assembled[test_idx] = fit_predict(
            X.iloc[train_idx], y[train_idx], X.iloc[test_idx]
        )
    np.testing.assert_array_equal(
        np.asarray(expected["predictions"], dtype=str),
        np.asarray(assembled, dtype=str),
    )
    assert expected["metrics"] == compute_metrics(y, np.asarray(assembled, dtype=str))


def test_fixture_is_deterministic():
    a, b = make_fixture(), make_fixture()
    assert a.equals(b)
