"""Tests for model construction and training."""

import numpy as np

from ecg_robust.models import build_models


def _toy_data(n=60, n_feat=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_feat))
    y = np.array(["N", "S", "V"] * (n // 3))
    return X, y


def test_build_models_has_five_primary_models():
    models = build_models(random_state=42)
    assert set(models) == {
        "dummy",
        "logistic_regression",
        "random_forest",
        "hist_gradient_boosting",
        "svm_rbf",
    }


def test_dummy_most_frequent_baseline():
    X, y = _toy_data()
    model = build_models()["dummy"]
    model.fit(X, y)
    pred = model.predict(X)
    assert set(np.unique(pred)) == {"N"}  # N is the majority


def test_dummy_respects_random_state():
    X, y = _toy_data()
    m1 = build_models(random_state=7)["dummy"].fit(X, y)
    m2 = build_models(random_state=7)["dummy"].fit(X, y)
    assert np.array_equal(m1.predict(X), m2.predict(X))


def test_random_forest_trains_and_predicts():
    X, y = _toy_data()
    model = build_models()["random_forest"].fit(X, y)
    pred = model.predict(X)
    assert len(pred) == len(X)
    assert set(np.unique(pred)) <= set(y)


def test_svm_rbf_trains_and_predicts():
    X, y = _toy_data()
    model = build_models()["svm_rbf"].fit(X, y)
    pred = model.predict(X)
    assert len(pred) == len(X)
    assert set(np.unique(pred)) <= set(y)


def test_models_reproducible_with_fixed_seed():
    X, y = _toy_data()
    p1 = build_models(random_state=13)["svm_rbf"].fit(X, y)
    p2 = build_models(random_state=13)["svm_rbf"].fit(X, y)
    assert np.array_equal(p1.predict(X), p2.predict(X))
    r1 = build_models(random_state=13)["random_forest"].fit(X, y)
    r2 = build_models(random_state=13)["random_forest"].fit(X, y)
    assert np.array_equal(r1.predict(X), r2.predict(X))


def test_random_forest_has_feature_importances():
    X, y = _toy_data()
    model = build_models()["random_forest"].fit(X, y)
    assert model.feature_importances_.shape == (X.shape[1],)


def test_logistic_regression_trains_and_predicts():
    X, y = _toy_data()
    model = build_models()["logistic_regression"].fit(X, y)
    pred = model.predict(X)
    assert len(pred) == len(X)
    assert set(np.unique(pred)) <= set(y)


def test_hist_gradient_boosting_trains_and_predicts():
    X, y = _toy_data()
    model = build_models()["hist_gradient_boosting"].fit(X, y)
    pred = model.predict(X)
    assert len(pred) == len(X)
    assert set(np.unique(pred)) <= set(y)


def test_all_five_primary_models_present():
    assert set(build_models()) == {
        "dummy",
        "logistic_regression",
        "random_forest",
        "hist_gradient_boosting",
        "svm_rbf",
    }
