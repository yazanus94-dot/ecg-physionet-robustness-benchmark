"""Starter solution for the patient-independent arrhythmia benchmark."""

from __future__ import annotations

import numpy as np
import pandas as pd


def patient_independent_oof(
    features: pd.DataFrame, labels: np.ndarray, groups: np.ndarray, n_splits: int = 4
) -> dict:
    """Return grouped OOF predictions by calling ``fit_predict`` per fold.

    Contract: use the documented StratifiedGroupKFold(n_splits=4,
    shuffle=True, random_state=42) definition; keep patients entirely within
    one fold; return ``predictions`` (one per input row), ``fold_groups``
    (canonical train/test patient lists per fold) and ``metrics`` recomputed
    from the assembled OOF predictions. The grader independently regenerates
    the folds and calls ``fit_predict`` itself.
    """
    raise NotImplementedError("Implement grouped out-of-fold classification")


def fit_predict(
    train_features: pd.DataFrame, train_labels: np.ndarray, test_features: pd.DataFrame
) -> np.ndarray:
    """Train on the given patients only; predict one label per test row."""
    raise NotImplementedError("Implement train-only fit and predict")
