"""Tests for scripts/feature_ablation.py ablation variant logic (Phase 6)."""

import numpy as np
import pandas as pd

from scripts.feature_ablation import FEATURE_ABLATIONS, ablation_columns


def test_ablation_variants_defined():
    assert set(FEATURE_ABLATIONS) == {
        "full",
        "drop_rr_features",
        "drop_morphology",
        "drop_spectral",
        "drop_stats",
        "top12_by_rf_importance",
    }


def test_ablation_columns_full():
    X = pd.DataFrame(np.zeros((10, 5)), columns=["a", "b", "c", "d", "e"])
    y = np.array(["N"] * 10)
    cols = ablation_columns("full", X, y)
    assert cols == ["a", "b", "c", "d", "e"]


def test_ablation_columns_drop_variants():
    X = pd.DataFrame(
        np.zeros((10, 8)),
        columns=[
            "rr_s",
            "rr_ratio",
            "rise_time_s",
            "fall_time_s",
            "zero_crossings",
            "spectral_centroid",
            "skew",
            "p95",
        ],
    )
    y = np.array(["N"] * 10)
    assert "rr_s" not in ablation_columns("drop_rr_features", X, y)
    cols = ablation_columns("drop_morphology", X, y)
    assert "rise_time_s" not in cols and "fall_time_s" not in cols
    assert "zero_crossings" not in cols
    cols = ablation_columns("drop_spectral", X, y)
    assert not ({"spectral_centroid"} & set(cols))
    cols = ablation_columns("drop_stats", X, y)
    assert "skew" not in cols and "p95" not in cols


def test_rf_selection_never_sees_validation_labels():
    """Permuting held-out labels must not change the selected columns."""
    rng = np.random.default_rng(7)
    X = pd.DataFrame(rng.normal(size=(100, 20)))
    y = np.array(["N", "S"] * 50)
    train = np.arange(80)
    validation = np.arange(80, 100)
    original = ablation_columns("top12_by_rf_importance", X, y, train)
    perturbed = y.copy()
    perturbed[validation] = perturbed[validation][::-1]
    assert not np.array_equal(perturbed[validation], y[validation])
    changed = ablation_columns("top12_by_rf_importance", X, perturbed, train)
    assert changed == original


def test_ablation_columns_top12_by_rf_importance():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(100, 20)))
    y = np.array(["N"] * 50 + ["S"] * 50)
    cols = ablation_columns("top12_by_rf_importance", X, y)
    assert len(cols) == 12
    assert set(cols) <= set(X.columns)
