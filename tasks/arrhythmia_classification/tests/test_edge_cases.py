"""Adversarial and edge-case tests for the classification task."""

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
    X, y, g = _arrays(make_fixture())
    y = y.copy()
    y[0] = "X"
    with pytest.raises(ValueError, match="unsupported"):
        patient_independent_oof(X, y, g)


def test_wrong_n_splits_raises():
    X, y, g = _arrays(make_fixture())
    with pytest.raises(ValueError, match="n_splits"):
        patient_independent_oof(X, y, g, n_splits=5)


def test_nan_features_are_imputed():
    frame = make_fixture()
    X, y, g = _arrays(frame)
    train = g < "P006"
    X_missing = X.copy()
    X_missing.loc[X_missing.index[:4], "qrs_amp"] = np.nan
    preds = fit_predict(X_missing[train], y[train], X_missing[~train])
    assert len(preds) == int((~train).sum())
    assert set(np.asarray(preds)) <= set(np.unique(y))


def test_extreme_test_value_does_not_change_other_predictions():
    frame = make_fixture()
    X, y, g = _arrays(frame)
    train = g < "P006"
    test_idx = X.index[~train]
    base = fit_predict(X[train], y[train], X.loc[test_idx])
    perturbed = X.loc[test_idx].copy()
    perturbed.iloc[0] = perturbed.iloc[0] + 10.0
    altered = fit_predict(X[train], y[train], perturbed)
    np.testing.assert_array_equal(np.asarray(base)[1:], np.asarray(altered)[1:])
