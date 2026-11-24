"""Invalid: correct fold predictions but fabricated fold metadata."""

from reference_solution import patient_independent_oof as _reference_oof


def fit_predict(train_features, train_labels, test_features):
    from reference_solution import fit_predict as _fit_predict

    return _fit_predict(train_features, train_labels, test_features)


def patient_independent_oof(features, labels, groups, n_splits=4):
    result = _reference_oof(features, labels, groups, n_splits)
    result["fold_groups"] = [
        {"train": ["FABRICATED"], "test": ["FABRICATED"]} for _ in result["fold_groups"]
    ]
    return result
