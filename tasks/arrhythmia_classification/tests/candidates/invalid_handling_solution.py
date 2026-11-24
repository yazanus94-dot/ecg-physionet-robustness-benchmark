"""Invalid: no input validation, raises arbitrary exceptions."""

import numpy as np
from reference_solution import (
    compute_metrics,
    expected_folds,
)
from reference_solution import (
    fit_predict as _base_fit_predict,
)


def fit_predict(train_features, train_labels, test_features):
    return _base_fit_predict(train_features, train_labels, test_features)


def patient_independent_oof(features, labels, groups, n_splits=4):
    if (
        len(labels) == 0
        or len(labels) != len(features)
        or len(groups) != len(features)
        or len(set(labels)) < 2
    ):
        raise TypeError("invalid input is not handled")
    y = np.asarray(labels)
    g = np.asarray(groups)
    preds = np.empty(len(y), dtype=object)
    for tr, va in expected_folds(g, y):
        preds[va] = fit_predict(
            features.iloc[tr], [y[i] for i in tr], features.iloc[va]
        )
    return {
        "predictions": preds,
        "fold_groups": [],
        "metrics": compute_metrics(y, preds),
    }
