"""Tests for the NSTDB noise experiment helpers."""

import numpy as np
import pandas as pd
import pytest

from ecg_robust.annotations import CLASSES
from ecg_robust.dataset import NSTDB_DIR
from ecg_robust.features import extract_features
from ecg_robust.models import build_models
from ecg_robust.noise import add_noise, estimate_snr_db, run_noise_sweep, scale_noise


def test_estimate_snr_db_of_identical_signals_is_zero():
    x = np.random.default_rng(0).normal(size=500)
    assert np.isclose(estimate_snr_db(x, x), 0.0)


def test_estimate_snr_db_is_finite():
    sig = np.random.default_rng(0).normal(size=500)
    noise = np.random.default_rng(1).normal(size=500)
    assert np.isfinite(estimate_snr_db(sig, noise))


def test_estimate_snr_db_zero_noise_is_inf():
    assert estimate_snr_db(np.ones(10), np.zeros(10)) == float("inf")


def test_scale_noise_hits_target_snr():
    rng = np.random.default_rng(2)
    sig = rng.normal(size=1000)
    noise = rng.normal(size=1000)
    scaled = scale_noise(noise, 12.0, sig)
    assert np.isclose(estimate_snr_db(sig, scaled), 12.0, atol=1e-4)


def test_add_noise_returns_same_length_and_achieved_snr():
    rng = np.random.default_rng(3)
    sig = rng.normal(size=252)
    noise = rng.normal(size=252)
    noisy, achieved = add_noise(sig, noise, 6.0)
    assert len(noisy) == len(sig)
    assert np.isclose(achieved, 6.0, atol=1e-4)
    assert np.isclose(estimate_snr_db(sig, noisy - sig), 6.0, atol=1e-4)


def test_run_noise_sweep_produces_expected_columns():
    rng = np.random.default_rng(4)
    clean = [rng.normal(size=252) for _ in range(40)]
    labels = list(np.array(CLASSES)[[rng.integers(0, 5) for _ in range(40)]])
    noise_realizations = [rng.normal(size=800) for _ in range(3)]
    models = build_models(random_state=42)
    X = pd.DataFrame([extract_features(s, 360.0) for s in clean])
    for model in models.values():
        model.fit(X, labels)
    out = run_noise_sweep(
        models, clean, labels, noise_realizations, fs=360, snrs=[24.0, 6.0]
    )
    assert {
        "snr_db",
        "model",
        "accuracy",
        "macro_f1",
        "sensitivity_macro",
        "balanced_accuracy",
    } <= set(out.columns)
    assert set(out["snr_db"]) == {24.0, 6.0}
    assert set(out["model"]) == set(models)
    assert ((out["accuracy"] >= 0) & (out["accuracy"] <= 1)).all()


def test_run_noise_sweep_empty_noise_list_returns_clean_features():
    rng = np.random.default_rng(5)
    clean = [rng.normal(size=252) for _ in range(10)]
    labels = list(np.array(CLASSES)[[rng.integers(0, 5) for _ in range(10)]])
    models = build_models(random_state=42)
    X = pd.DataFrame([extract_features(s, 360.0) for s in clean])
    for model in models.values():
        model.fit(X, labels)
    out = run_noise_sweep(models, clean, labels, [], fs=360, snrs=[24.0])
    assert len(out) == len(models)


def test_extract_nstdb_noise_requires_download():
    if NSTDB_DIR.exists():
        pytest.skip("NSTDB present; download-dependent test not applicable")
    from ecg_robust.noise import extract_nstdb_noise

    with pytest.raises(FileNotFoundError):
        extract_nstdb_noise("118", 360.0)


def test_snrs_default_levels():
    from ecg_robust.noise import SNRS_DB

    assert SNRS_DB == [24.0, 18.0, 12.0, 6.0, 0.0, -6.0]
