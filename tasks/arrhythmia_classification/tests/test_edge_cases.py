"""Adversarial tests: invalid inputs, single-class folds and missing values."""

import numpy as np
import pytest
from reference_solution import (
    FEATURE_COLUMNS,
    fit_predict,
    make_fixture,
    patient_independent_oof,
)


def _arrays(frame):
    return (
        frame[list(FEATURE_COLUMNS)],
        frame["label"].to_numpy(),
        frame["patient_id"].to_numpy(),
    )


@pytest.mark.parametrize("case", ["empty", "mismatch", "single_class", "bad_groups"])
def test_invalid_inputs_raise_value_error(case):
    frame = make_fixture()
    X, y, g = _arrays(frame)
    if case == "empty":
        X, y, g = X.iloc[:0], y[:0], g[:0]
    elif case == "mismatch":
        y, g = y[:-1], g[:-1]
    elif case == "single_class":
        mask = y == "N"
        X, y, g = X[mask], y[mask], g[mask]
    elif case == "bad_groups":
        g = g[:-1]
    with pytest.raises(ValueError):
        patient_independent_oof(X, y, g)


def test_unsupported_label_raises():
    frame = make_fixture()
    X, y, g = _arrays(frame)
    y = y.copy()
    y[0] = "X"
    with pytest.raises(ValueError, match="unsupported"):
        patient_independent_oof(X, y, g)


def test_nan_features_are_imputed_without_leakage():
    frame = make_fixture()
    X, y, g = _arrays(frame)
    train = g < "P006"
    X_missing = X.copy()
    X_missing.loc[X_missing.index[:4], "qrs_amp"] = np.nan
    preds = fit_predict(X_missing[train], y[train], X_missing[~train])
    assert len(preds) == int((~train).sum())
    assert set(np.asarray(preds)) <= set(np.unique(y))


def test_predictions_are_deterministic():
    frame = make_fixture()
    X, y, g = _arrays(frame)
    a = patient_independent_oof(X, y, g)
    b = patient_independent_oof(X, y, g)
    assert a["predictions"] == b["predictions"]
