"""Regenerate expected_output.json from the reference on the synthetic fixture."""

import json
from pathlib import Path

from reference_solution import (
    FEATURE_COLUMNS,
    compute_metrics,
    expected_folds,
    make_fixture,
    patient_independent_oof,
)
from sklearn.dummy import DummyClassifier

HERE = Path(__file__).resolve().parent


def main() -> None:
    frame = make_fixture()
    X = frame[list(FEATURE_COLUMNS)]
    y = frame["label"].to_numpy()
    g = frame["patient_id"].to_numpy()
    result = patient_independent_oof(X, y, g)

    dummy_pred = __import__("numpy").empty(len(y), dtype=object)
    for train_idx, test_idx in expected_folds(g, y):
        dummy = DummyClassifier(strategy="most_frequent").fit(
            X.iloc[train_idx], y[train_idx]
        )
        dummy_pred[test_idx] = dummy.predict(X.iloc[test_idx])
    dummy_metrics = compute_metrics(
        y, __import__("numpy").asarray(dummy_pred, dtype=str)
    )

    payload = {
        "task": "arrhythmia_classification",
        "note": "Synthetic educational benchmark. Reference and dummy metrics are calibration evidence.",
        "fixture": {
            "generator": "reference_solution.make_fixture",
            "seed": 424242,
            "n_patients": len(set(g)),
            "n_beats": len(frame),
            "classes": ["N", "S", "V", "F"],
            "class_distribution": frame["label"].value_counts().to_dict(),
        },
        "metrics": result["metrics"],
        "dummy_metrics": dummy_metrics,
    }
    (HERE / "expected_output.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
