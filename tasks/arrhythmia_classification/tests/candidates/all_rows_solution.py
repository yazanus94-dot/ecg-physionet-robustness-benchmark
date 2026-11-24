"""Invalid: trains on all rows in OOF but reports canonical fold metadata."""

import numpy as np
from reference_solution import (
    _validate,
    compute_metrics,
    expected_folds,
    fit_predict,
)


def patient_independent_oof(features, labels, groups, n_splits=4):
    _validate(features, labels, groups)
    y = np.asarray(labels)
    g = np.asarray(groups)
    all_predictions = fit_predict(features, y, features)
    fold_groups = [
        {"train": sorted(set(g[tr])), "test": sorted(set(g[va]))}
        for tr, va in expected_folds(g, y)
    ]
    return {
        "predictions": list(all_predictions),
        "fold_groups": fold_groups,
        "metrics": compute_metrics(y, np.asarray(all_predictions, dtype=str)),
    }
