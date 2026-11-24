"""Invalid: returns OOF predictions for only the first fold."""

from reference_solution import expected_folds
from reference_solution import patient_independent_oof as _reference_oof


def fit_predict(train_features, train_labels, test_features):
    from reference_solution import fit_predict as _fit_predict

    return _fit_predict(train_features, train_labels, test_features)


def patient_independent_oof(features, labels, groups, n_splits=4):
    result = _reference_oof(features, labels, groups, n_splits)
    folds = expected_folds(groups, labels)
    incomplete = [""] * len(result["predictions"])
    for idx in folds[0][1]:
        incomplete[idx] = result["predictions"][idx]
    result["predictions"] = incomplete
    return result
