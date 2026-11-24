"""Reference solution for patient-independent ECG arrhythmia classification.

Self-contained and synthetic. The fixture deliberately includes overlapping
class distributions, patient-specific offsets, measurement noise, class
imbalance and missing/corrupted values. It does not reproduce the physiological
complexity of MIT-BIH and is strictly an educational benchmark.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

CLASSES = ("N", "S", "V", "F")
FEATURE_COLUMNS = ("rr_s", "qrs_width_s", "qrs_amp", "spectral_centroid")
SPLIT_SEED = 42
N_SPLITS = 4


def make_fixture(
    n_patients: int = 18, beats_per_patient: int = 24, seed: int = 424242
) -> pd.DataFrame:
    """Deterministic synthetic beat table with patient identifiers."""
    rng = np.random.default_rng(seed)
    profiles = {
        "N": (0.86, 0.082, 1.00, 0.62),
        "S": (0.71, 0.098, 0.94, 0.59),
        "V": (0.60, 0.134, 1.42, 0.73),
        "F": (0.77, 0.116, 1.22, 0.69),
    }
    # Deliberate class imbalance through unequal quotas.
    quotas = {"N": 12, "S": 6, "V": 5, "F": 1} if beats_per_patient == 24 else None
    rows = []
    for patient in range(n_patients):
        offset = rng.normal(0, 0.10, size=4)
        for beat in range(beats_per_patient):
            if quotas is not None:
                choices = []
                for label, count in quotas.items():
                    choices.extend([label] * count)
                label = choices[beat]
            else:
                label = CLASSES[beat % len(CLASSES)]
            values = np.asarray(profiles[label]) + offset
            values = values + rng.normal(0, 0.055, size=4)
            if rng.random() < 0.03:
                values[rng.integers(0, 4)] = np.nan
            if rng.random() < 0.01:
                values[rng.integers(0, 4)] += rng.choice([-1, 1]) * 0.8
            rows.append(
                {
                    "patient_id": f"P{patient:03d}",
                    "label": label,
                    "rr_s": values[0],
                    "qrs_width_s": values[1],
                    "qrs_amp": values[2],
                    "spectral_centroid": values[3],
                }
            )
    return pd.DataFrame(rows)


def build_model(random_state: int = SPLIT_SEED) -> Pipeline:
    """Impute/scale/model. Every component is fit on training rows only."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=3000, class_weight="balanced", random_state=random_state
                ),
            ),
        ]
    )


def _validate(
    features: pd.DataFrame, labels: Sequence[str], groups: Sequence[str] | None = None
) -> pd.DataFrame:
    if features.empty or len(labels) == 0:
        raise ValueError("features and labels must be non-empty")
    if len(features) != len(labels):
        raise ValueError("features and labels must have the same length")
    if groups is not None and len(groups) != len(features):
        raise ValueError("groups must match features length")
    missing = [c for c in FEATURE_COLUMNS if c not in features.columns]
    if missing:
        raise ValueError(f"missing feature columns: {', '.join(missing)}")
    unknown = set(labels) - set(CLASSES)
    if unknown:
        raise ValueError(f"unsupported labels: {sorted(unknown)}")
    if len(set(labels)) < 2:
        raise ValueError("training data must contain at least two classes")
    return features


def expected_folds(groups: Sequence[str], labels: Sequence[str]):
    """Canonical fold definition shared by the task specification and grader."""
    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=SPLIT_SEED
    )
    X_dummy = np.zeros((len(labels), len(FEATURE_COLUMNS)))
    return list(splitter.split(X_dummy, np.asarray(labels), np.asarray(groups)))


def fit_predict(
    train_features: pd.DataFrame,
    train_labels: Sequence[str],
    test_features: pd.DataFrame,
) -> np.ndarray:
    """Fit on training rows only and return one label per test row."""
    _validate(train_features, train_labels)
    model = build_model()
    model.fit(train_features[list(FEATURE_COLUMNS)], np.asarray(train_labels))
    return model.predict(test_features[list(FEATURE_COLUMNS)])


def patient_independent_oof(
    features: pd.DataFrame,
    labels: Sequence[str],
    groups: Sequence[str],
    n_splits: int = N_SPLITS,
) -> dict:
    """Grouped OOF predictions; delegates every fold to ``fit_predict``.

    Contract: returns ``predictions`` (one label per input row), ``fold_groups``
    (one ``{"train": [...], "test": [...]}`` per fold, in fold order) and
    ``metrics`` recomputed from the assembled OOF predictions.
    """
    _validate(features, labels, groups)
    y = np.asarray(labels)
    g = np.asarray(groups)
    if n_splits != N_SPLITS:
        raise ValueError(f"n_splits must be {N_SPLITS}")
    splits = expected_folds(g, y)
    predictions = np.empty(len(y), dtype=object)
    fold_groups = []
    for train_idx, test_idx in splits:
        train_set, test_set = set(g[train_idx]), set(g[test_idx])
        if train_set & test_set:
            raise RuntimeError("patient leakage: train/test groups overlap")
        fold_groups.append({"train": sorted(train_set), "test": sorted(test_set)})
        predictions[test_idx] = fit_predict(
            features.iloc[train_idx], y[train_idx], features.iloc[test_idx]
        )

    oof = np.asarray(predictions, dtype=str)
    metrics = compute_metrics(y, oof)
    return {"predictions": oof.tolist(), "fold_groups": fold_groups, "metrics": metrics}


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "f1_macro": round(
            float(
                f1_score(
                    y_true,
                    y_pred,
                    labels=list(CLASSES),
                    average="macro",
                    zero_division=0,
                )
            ),
            4,
        ),
    }
