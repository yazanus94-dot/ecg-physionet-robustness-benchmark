"""Starter solution for the patient-independent arrhythmia benchmark."""

from __future__ import annotations

import numpy as np
import pandas as pd


def patient_independent_oof(
    features: pd.DataFrame, labels: np.ndarray, groups: np.ndarray, n_splits: int = 4
) -> dict:
    """Return grouped out-of-fold predictions and per-fold patient sets.

    Must validate inputs, keep each patient entirely in one fold, fit all
    preprocessing on training rows only and return:
    ``{"predictions": [...], "fold_groups": [{"train": [...], "test": [...]}], "metrics": {...}}``
    """
    raise NotImplementedError("Implement grouped out-of-fold classification")


def fit_predict(
    train_features: pd.DataFrame, train_labels: np.ndarray, test_features: pd.DataFrame
) -> np.ndarray:
    """Train on the given patients only; predict one label per test row."""
    raise NotImplementedError("Implement train-only fit and predict")
