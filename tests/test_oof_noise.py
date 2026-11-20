"""Spec #3: out-of-fold noise evaluation and reserved-record handling."""

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from ecg_robust.features import extract_features
from ecg_robust.noise import noisy_heldout_metrics
from ecg_robust.pipeline import reserved_records_for_nstdb


def test_noisy_heldout_metrics_shape_and_values():
    # Out-of-fold noise evaluation: a fitted pipe scores noisy versions of
    # its held-out validation beats (spec #3).
    rng = np.random.default_rng(3)
    clean = [rng.normal(size=252) for _ in range(10)]
    y = np.array(["N", "S", "V", "F", "Q", "N", "S", "V", "F", "Q"])
    X = pd.DataFrame([extract_features(s, 360.0) for s in clean])
    pipe = Pipeline([("model", DummyClassifier(strategy="most_frequent"))]).fit(X, y)
    df = noisy_heldout_metrics(
        pipe, y[:8], clean[:8],
        [rng.normal(size=4000) for _ in range(2)], [24.0, 12.0], 360.0,
        feature_columns=list(X.columns),
    )
    assert set(df.columns) >= {
        "snr_db", "n_beats", "accuracy", "macro_f1",
        "sensitivity_macro", "balanced_accuracy",
    }
    assert sorted(df["snr_db"].tolist()) == [12.0, 24.0]
    assert int(df["n_beats"].sum()) == 16
    assert ((df["accuracy"] >= 0) & (df["accuracy"] <= 1)).all()
    assert ((df["macro_f1"] >= 0) & (df["macro_f1"] <= 1)).all()


def test_noisy_heldout_metrics_deterministic():
    rng = np.random.default_rng(11)
    clean = [rng.normal(size=252) for _ in range(6)]
    y = np.array(["N", "S", "V", "F", "Q", "N"])
    X = pd.DataFrame([extract_features(s, 360.0) for s in clean])
    pipe = Pipeline([("model", DummyClassifier(strategy="most_frequent"))]).fit(X, y)
    noise = [rng.normal(size=4000) for _ in range(2)]
    df1 = noisy_heldout_metrics(pipe, y, clean, noise, [24.0], 360.0,
                                feature_columns=list(X.columns), seed=7)
    df2 = noisy_heldout_metrics(pipe, y, clean, noise, [24.0], 360.0,
                                feature_columns=list(X.columns), seed=7)
    assert np.allclose(df1["accuracy"], df2["accuracy"])
    assert np.allclose(df1["macro_f1"], df2["macro_f1"])


def test_reserved_records_for_nstdb():
    # Spec #3: records 118/119 and their stress versions are reserved.
    records = ["100", "118", "119", "200"]
    nstdb = ["118e00", "118e06", "119e00", "118e_6"]
    assert reserved_records_for_nstdb(records, nstdb) == {"118", "119"}


def test_reserved_records_no_direct_matches():
    assert reserved_records_for_nstdb(["100", "200"], ["118e00", "bw"]) == set()
