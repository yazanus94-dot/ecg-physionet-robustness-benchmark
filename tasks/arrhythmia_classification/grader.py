"""Automated grader for the patient-independent arrhythmia benchmark task.

Usage:
    python grader.py [path/to/candidate.py]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
from reference_solution import CLASSES, FEATURE_COLUMNS, make_fixture
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

HERE = Path(__file__).resolve().parent


def _load_candidate(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("candidate_solution", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load candidate from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _check(name: str, ok: bool, detail: str, results: dict) -> None:
    results[name] = {"passed": bool(ok), "detail": detail}


def grade(candidate_path: Path) -> dict:
    candidate = _load_candidate(candidate_path)
    frame = make_fixture()
    features = frame[list(FEATURE_COLUMNS)]
    labels = frame["label"].to_numpy()
    groups = frame["patient_id"].to_numpy()
    results: dict[str, dict] = {}

    try:
        result = candidate.patient_independent_oof(features, labels, groups)
    except Exception as exc:  # noqa: BLE001
        _check(
            "runs",
            False,
            f"patient_independent_oof raised {type(exc).__name__}: {exc}",
            results,
        )
        return {"checks": results, "metrics": {}, "passed": False}
    _check("runs", True, "grouped OOF returned", results)

    predictions = np.asarray(result.get("predictions", []), dtype=str)
    _check(
        "prediction_length",
        len(predictions) == len(frame),
        f"expected {len(frame)}, got {len(predictions)}",
        results,
    )
    _check(
        "valid_labels",
        bool(set(predictions) <= set(CLASSES)),
        f"unexpected labels: {sorted(set(predictions) - set(CLASSES))}",
        results,
    )

    metrics = (
        {
            "accuracy": round(float(accuracy_score(labels, predictions)), 4),
            "balanced_accuracy": round(
                float(balanced_accuracy_score(labels, predictions)), 4
            ),
            "f1_macro": round(
                float(
                    f1_score(
                        labels,
                        predictions,
                        labels=list(CLASSES),
                        average="macro",
                        zero_division=0,
                    )
                ),
                4,
            ),
        }
        if len(predictions) == len(frame)
        else {}
    )

    splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
    expected_folds = list(splitter.split(features, labels, groups))
    fold_groups = result.get("fold_groups", [])
    overlap_free = len(fold_groups) == len(expected_folds)
    if overlap_free:
        for fold in fold_groups:
            if set(fold["train"]) & set(fold["test"]):
                overlap_free = False
    _check(
        "patient_independence",
        overlap_free,
        "every fold reports disjoint train/test patient sets",
        results,
    )

    # Dummy baseline on the same grouped folds.
    dummy_pred = np.empty(len(frame), dtype=object)
    for train_idx, test_idx in expected_folds:
        dummy = DummyClassifier(strategy="most_frequent").fit(
            features.iloc[train_idx], labels[train_idx]
        )
        dummy_pred[test_idx] = dummy.predict(features.iloc[test_idx])
    baseline = balanced_accuracy_score(labels, np.asarray(dummy_pred, dtype=str))
    _check(
        "beats_dummy",
        metrics.get("balanced_accuracy", 0) >= baseline + 0.15,
        f"model {metrics.get('balanced_accuracy')} vs dummy {baseline:.4f}",
        results,
    )
    _check(
        "balanced_accuracy",
        metrics.get("balanced_accuracy", 0) >= 0.80,
        f"got {metrics.get('balanced_accuracy')}",
        results,
    )

    second = candidate.patient_independent_oof(features, labels, groups)
    _check(
        "determinism",
        second.get("predictions") == result.get("predictions"),
        "two identical calls give identical predictions",
        results,
    )

    # Adversarial: shuffled test labels must not change fit_predict output.
    train_idx, test_idx = expected_folds[0]
    pred_a = candidate.fit_predict(
        features.iloc[train_idx], labels[train_idx], features.iloc[test_idx]
    )
    shuffled_test = features.iloc[test_idx].copy()
    shuffled_test["label"] = np.random.default_rng(0).permutation(labels[test_idx])
    pred_b = candidate.fit_predict(
        features.iloc[train_idx], labels[train_idx], shuffled_test
    )
    _check(
        "test_labels_unused",
        bool(np.array_equal(np.asarray(pred_a), np.asarray(pred_b))),
        "test label column cannot influence predictions",
        results,
    )

    bad_cases = {
        "empty": (features.iloc[:0], labels[:0], groups[:0]),
        "length_mismatch": (features, labels[:-1], groups[:-1]),
        "single_class": (
            features[labels == "N"],
            labels[labels == "N"],
            groups[labels == "N"],
        ),
        "bad_group_length": (features, labels, groups[:-1]),
    }
    for name, (X_bad, y_bad, g_bad) in bad_cases.items():
        try:
            candidate.patient_independent_oof(X_bad, y_bad, g_bad)
            _check(f"invalid_{name}", False, "expected ValueError", results)
        except ValueError:
            _check(f"invalid_{name}", True, "ValueError raised", results)
        except Exception as exc:  # noqa: BLE001
            _check(
                f"invalid_{name}",
                False,
                f"wrong exception {type(exc).__name__}",
                results,
            )

    with_nan = features.copy()
    with_nan.loc[with_nan.index[:5], "qrs_amp"] = np.nan
    try:
        nan_pred = candidate.fit_predict(
            with_nan.iloc[train_idx], labels[train_idx], with_nan.iloc[test_idx]
        )
        _check(
            "nan_imputation",
            len(nan_pred) == len(test_idx),
            "NaN features imputed",
            results,
        )
    except Exception as exc:  # noqa: BLE001
        _check("nan_imputation", False, f"raised {type(exc).__name__}", results)

    expected = json.loads((HERE / "expected_output.json").read_text(encoding="utf-8"))
    for key, target in expected["metrics"].items():
        _check(
            f"expected_{key}",
            abs(metrics.get(key, -1) - target) <= 0.03,
            f"got {metrics.get(key)}, expected {target} +/- 0.03",
            results,
        )

    passed = all(item["passed"] for item in results.values())
    return {"checks": results, "metrics": metrics, "passed": passed}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "candidate", nargs="?", default=str(HERE / "starter_solution.py")
    )
    args = parser.parse_args()
    report = grade(Path(args.candidate))
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"RESULT: {'PASS' if report['passed'] else 'FAIL'}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
