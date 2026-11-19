"""Reproducibility tests: fixed seeds must give identical outputs."""

import numpy as np

from ecg_robust.models import build_models


def _toy(n=90, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    y = np.array(["N", "S", "V", "F"] * (n // 4) + ["N"] * (n % 4))
    return X, y


def test_svm_reproducible_with_same_seed():
    X, y = _toy()
    m1 = build_models(random_state=7)["svm_rbf"].fit(X, y)
    m2 = build_models(random_state=7)["svm_rbf"].fit(X, y)
    assert np.array_equal(m1.predict(X), m2.predict(X))


def test_random_forest_reproducible_with_same_seed():
    X, y = _toy()
    m1 = build_models(random_state=7)["random_forest"].fit(X, y)
    m2 = build_models(random_state=7)["random_forest"].fit(X, y)
    assert np.array_equal(m1.predict(X), m2.predict(X))


def test_different_seeds_may_differ_svm():
    X, y = _toy()
    p_a = build_models(random_state=7)["svm_rbf"].fit(X, y)
    p_b = build_models(random_state=8)["svm_rbf"].fit(X, y)
    assert p_a.n_support_ is not None and p_b.n_support_ is not None
