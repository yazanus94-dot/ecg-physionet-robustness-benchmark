"""Invalid: constant predictor."""

import numpy as np
from reference_solution import (
    _validate,
    compute_metrics,
    expected_folds,
)


def fit_predict(train_features, train_labels, test_features):
    _validate(train_features, train_labels)
    return np.repeat("N", len(test_features))


def patient_independent_oof(features, labels, groups, n_splits=4):
    _validate(features, labels, groups)
    y = np.asarray(labels)
    g = np.asarray(groups)
    preds = np.empty(len(y), dtype=object)
    fold_groups = []
    for tr, va in expected_folds(g, y):
        fold_groups.append({"train": sorted(set(g[tr])), "test": sorted(set(g[va]))})
        preds[va] = fit_predict(features.iloc[tr], y[tr], features.iloc[va])
    oof = np.asarray(preds, dtype=str)
    return {
        "predictions": oof.tolist(),
        "fold_groups": fold_groups,
        "metrics": compute_metrics(y, oof),
    }
