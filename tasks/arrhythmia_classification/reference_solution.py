"""Reference solution for patient-independent ECG arrhythmia classification.

Self-contained and synthetic: the fixture encodes beat morphology patterns for
four AAMI classes with subject-specific offsets, so a subject-level split is
meaningful without downloading PhysioNet records.
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


def make_fixture(
    n_patients: int = 12, beats_per_patient: int = 20, seed: int = 424242
) -> pd.DataFrame:
    """Deterministic synthetic beat table with patient identifiers."""
    rng = np.random.default_rng(seed)
    profiles = {
        "N": (0.85, 0.080, 1.00, 0.62),
        "S": (0.62, 0.090, 0.92, 0.55),
        "V": (0.50, 0.145, 1.65, 0.80),
        "F": (0.72, 0.120, 1.32, 0.72),
    }
    rows = []
    for patient in range(n_patients):
        offset = rng.normal(0, 0.04, size=4)
        for beat in range(beats_per_patient):
            label = CLASSES[beat % len(CLASSES)]
            means = np.asarray(profiles[label]) + offset
            values = means + rng.normal(0, 0.035, size=4)
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


def build_model(random_state: int = 42) -> Pipeline:
    """Scaler + imputer + balanced logistic regression, fit on train only."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000, class_weight="balanced", random_state=random_state
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
    missing = [column for column in FEATURE_COLUMNS if column not in features.columns]
    if missing:
        raise ValueError(f"missing feature columns: {', '.join(missing)}")
    unknown = set(labels) - set(CLASSES)
    if unknown:
        raise ValueError(f"unsupported labels: {sorted(unknown)}")
    if len(set(labels)) < 2:
        raise ValueError("training data must contain at least two classes")
    return features


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
    n_splits: int = 4,
) -> dict:
    """Grouped out-of-fold predictions; no patient occurs in train and test."""
    _validate(features, labels, groups)
    y = np.asarray(labels)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    X = features[list(FEATURE_COLUMNS)]
    predictions = np.empty(len(y), dtype=object)
    fold_groups = []
    for train_idx, test_idx in splitter.split(X, y, groups):
        train_groups = set(np.asarray(groups)[train_idx])
        test_groups = set(np.asarray(groups)[test_idx])
        if train_groups & test_groups:
            raise RuntimeError("patient leakage: train/test groups overlap")
        fold_groups.append({"train": sorted(train_groups), "test": sorted(test_groups)})
        model = build_model()
        model.fit(X.iloc[train_idx], y[train_idx])
        predictions[test_idx] = model.predict(X.iloc[test_idx])

    oof = np.asarray(predictions, dtype=str)
    metrics = {
        "accuracy": round(float(accuracy_score(y, oof)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y, oof)), 4),
        "f1_macro": round(
            float(
                f1_score(y, oof, labels=list(CLASSES), average="macro", zero_division=0)
            ),
            4,
        ),
    }
    return {"predictions": oof.tolist(), "fold_groups": fold_groups, "metrics": metrics}
