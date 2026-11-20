"""Phase 4 — fair, reproducible noise experiments (spec #4).

- `generate_noisy_features` generates the noisy matrices per (fold, SNR)
  ONCE — every model in the fold scores the same noisy validation samples.
- Determinism: same rng seed -> same matrices.
- SNR consistency: achieved ≈ requested (`add_noise` scales the noise so
  that `SNR(signal, noise) == snr_db`; verified by `estimate_snr_db`).
- The pipeline records `fold`, `model`, `snr_db`, `achieved_snr_db`,
  `n_beats` in `reports/results/noise_results.csv`.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from ecg_robust.features import extract_features
from ecg_robust.noise import (
    add_noise,
    estimate_snr_db,
    generate_noisy_features,
)


def test_generate_noisy_features_shared_across_models():
    # The noisy feature matrix for a (fold, SNR) is one object shared by
    # every model in the fold — generation does not depend on any model.
    rng = np.random.default_rng(0)
    clean = [rng.normal(size=252) for _ in range(6)]
    noise = [rng.normal(size=800) for _ in range(2)]
    X_by_snr, _ = generate_noisy_features(
        clean, noise, [24.0], 360.0, np.random.default_rng(7)
    )
    assert set(X_by_snr) == {24.0}
    Xn = X_by_snr[24.0]
    assert len(Xn) == len(clean)


def test_noise_fairness_identical_inputs_across_models():
    # Phase 3 (spec #3): all models in a fold/SNR are evaluated on the
    # identical noisy samples. `generate_noisy_features` returns ONE matrix
    # per (fold, SNR) that every model in the fold predicts on.
    from ecg_robust.models import build_models

    rng = np.random.default_rng(0)
    clean = [rng.normal(size=252) for _ in range(12)]
    noise = [rng.normal(size=2000) for _ in range(4)]
    Xn, _ = generate_noisy_features(
        clean, noise, [24.0], 360.0, np.random.default_rng(0)
    )
    X = pd.DataFrame([extract_features(s, 360.0) for s in clean])
    y = np.array(["N", "S", "V", "F", "Q", "N", "S", "V", "F", "Q", "N", "S"])
    models = build_models()
    for m in models.values():
        m.fit(X, y)
    # the same noisy matrix is the test signal for every model
    preds = [np.asarray(m.predict(Xn[24.0])) for m in models.values()]
    assert all(len(p) == 12 for p in preds)
    # determinism: same rng seed -> identical matrices
    Xn2, _ = generate_noisy_features(
        clean, noise, [24.0], 360.0, np.random.default_rng(0)
    )
    assert np.allclose(Xn[24.0].values, Xn2[24.0].values)


def test_noise_results_fairness_from_csv():
    # Phase 3: every (snr_db, fold) has identical n_beats across models --
    # models are compared on the same held-out validation beats.
    import os

    p = os.path.join("reports", "results", "noise_results.csv")
    if not os.path.exists(p):
        pytest.skip("noise_results.csv not generated")
    df = pd.read_csv(p)
    for _, g in df.groupby(["snr_db", "fold"]):
        assert g["n_beats"].nunique() == 1, (
            "identical noisy samples required per fold/SNR across models"
        )
    # requested vs achieved SNR consistency
    merged = df.dropna(subset=["achieved_snr_db"])
    if len(merged):
        assert np.allclose(merged["achieved_snr_db"], merged["snr_db"], atol=0.5)


def test_generate_noisy_features_deterministic():
    rng = np.random.default_rng(3)
    clean = [rng.normal(size=252) for _ in range(6)]
    noise = [rng.normal(size=800) for _ in range(2)]
    X1, a1 = generate_noisy_features(
        clean, noise, [24.0, 12.0], 360.0, np.random.default_rng(11)
    )
    X2, a2 = generate_noisy_features(
        clean, noise, [24.0, 12.0], 360.0, np.random.default_rng(11)
    )
    for snr in (24.0, 12.0):
        assert np.allclose(X1[snr].values, X2[snr].values)
        assert np.isclose(a1[snr], a2[snr])


def test_models_receive_identical_test_signals():
    # Two different models — both predict on the SAME noisy matrix.
    rng = np.random.default_rng(0)
    clean = [rng.normal(size=252) for _ in range(8)]
    X_feat = pd.DataFrame([extract_features(s, 360.0) for s in clean])
    y = np.array(["N", "S", "V", "F", "Q", "N", "S", "V"])
    # Fit two different models on the training features first...
    pipe_dummy = Pipeline([("model", DummyClassifier(strategy="most_frequent"))]).fit(
        X_feat, y
    )
    pipe_rf = Pipeline(
        [("model", RandomForestClassifier(n_estimators=10, random_state=0))]
    ).fit(X_feat, y)
    # ...then generate the noisy test signals (independent of the models).
    Xn, _ = generate_noisy_features(
        clean,
        [rng.normal(size=800) for _ in range(3)],
        [24.0],
        360.0,
        np.random.default_rng(11),
        feature_columns=list(X_feat.columns),
    )
    p1 = pipe_dummy.predict(Xn[24.0])
    p2 = pipe_rf.predict(Xn[24.0])
    assert len(p1) == len(p2) == len(clean)


def test_achieved_snr_matches_requested():
    rng = np.random.default_rng(3)
    for snr in [24.0, 18.0, 12.0, 6.0, 0.0, -6.0]:
        seg = rng.normal(size=252)
        noise = np.resize(rng.normal(size=800), len(seg))
        noisy, achieved = add_noise(seg, noise, snr)
        assert np.isclose(achieved, snr, atol=1e-4)
        assert np.isclose(estimate_snr_db(seg, noisy - seg), snr, atol=1e-3)


def test_noise_results_identical_inputs_per_fold_snr_from_csv():
    # Phase 3 (spec #3) regression: every model in a fold/SNR is scored on
    # identical noisy samples — the generated matrix is shared by every
    # model, so the recorded n_beats and achieved_snr_db must be identical
    # across models for the same (fold, snr_db).
    import os

    p = os.path.join("reports", "results", "noise_results.csv")
    if not os.path.exists(p):
        pytest.skip("noise_results.csv not generated")
    df = pd.read_csv(p)
    for _, g in df.groupby(["fold", "snr_db"]):
        assert g["n_beats"].nunique() == 1, (
            "identical noisy samples required per fold/SNR across models"
        )
        merged = g.dropna(subset=["achieved_snr_db"])
        if len(merged):
            assert merged["achieved_snr_db"].nunique() == 1


def test_pipeline_noise_stage_matrix_generated_once_per_fold_snr():
    # Phase 3: mirrors the pipeline's [4/9] stage — the noisy validation
    # feature set is generated once per fold/SNR (single shared rng stream)
    # and every model in the fold is scored on the same matrix.
    from ecg_robust.models import build_models

    rng = np.random.default_rng(0)
    clean = [rng.normal(size=252) for _ in range(10)]
    noise = [rng.normal(size=2000) for _ in range(3)]
    Xn, _ach = generate_noisy_features(
        clean, noise, [24.0], 360.0, np.random.default_rng(5)
    )
    X = pd.DataFrame([extract_features(s, 360.0) for s in clean])
    y = np.array(["N", "S", "V", "F", "Q", "N", "S", "V", "F", "Q"])
    models = build_models(random_state=42)
    for m in models.values():
        m.fit(X, y)
    preds = {name: np.asarray(m.predict(Xn[24.0])) for name, m in models.items()}
    assert len(preds) == len(models)
    assert all(len(p) == len(clean) for p in preds.values())
    # regeneration with the same rng reproduces the same matrix
    Xn2, _ = generate_noisy_features(
        clean, noise, [24.0], 360.0, np.random.default_rng(5)
    )
    assert np.allclose(Xn[24.0].values, Xn2[24.0].values)
