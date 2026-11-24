"""Automated grader for the patient-independent arrhythmia benchmark.

Usage:
    python grader.py [path/to/candidate.py]

The grader regenerates the canonical folds itself and calls the candidate's
``fit_predict`` once per fold. It never trusts self-reported fold metadata as
proof of patient independence.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
from reference_solution import (
    CLASSES,
    FEATURE_COLUMNS,
    N_SPLITS,
    compute_metrics,
    expected_folds,
    make_fixture,
)
from sklearn.dummy import DummyClassifier

HERE = Path(__file__).resolve().parent
MIN_BALANCED_ACCURACY = 0.65
MIN_F1_MACRO = 0.55
MIN_DUMMY_MARGIN = 0.15


def _load_candidate(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("candidate_solution", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load candidate from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _check(name: str, ok: bool, detail: str, results: dict) -> None:
    results[name] = {"passed": bool(ok), "detail": detail}


def _as_strings(predictions: Any) -> np.ndarray:
    return np.asarray(predictions, dtype=str).reshape(-1)


def grade(candidate_path: Path) -> dict:
    candidate = _load_candidate(candidate_path)
    results: dict[str, dict] = {}
    if not hasattr(candidate, "patient_independent_oof") or not hasattr(
        candidate, "fit_predict"
    ):
        _check(
            "interface", False, "candidate must expose both required functions", results
        )
        return {"checks": results, "metrics": {}, "passed": False}
    _check("interface", True, "both required functions present", results)

    frame = make_fixture()
    features = frame[list(FEATURE_COLUMNS)]
    labels = frame["label"].to_numpy()
    groups = frame["patient_id"].to_numpy()
    folds = expected_folds(groups, labels)

    # Independent fold sanity checks.
    test_indices = [idx for _, test_idx in folds for idx in test_idx]
    _check(
        "every_sample_once",
        sorted(test_indices) == list(range(len(frame))),
        "each sample appears in exactly one held-out fold",
        results,
    )
    no_overlap = all(not (set(groups[tr]) & set(groups[va])) for tr, va in folds)
    _check(
        "canonical_folds_patient_disjoint",
        no_overlap,
        "canonical folds contain no patient in both train and test",
        results,
    )
    _check("n_folds", len(folds) == N_SPLITS, f"expected {N_SPLITS} folds", results)

    try:
        submitted = candidate.patient_independent_oof(features, labels, groups)
    except Exception as exc:  # noqa: BLE001
        _check(
            "runs",
            False,
            f"patient_independent_oof raised {type(exc).__name__}: {exc}",
            results,
        )
        return {"checks": results, "metrics": {}, "passed": False}
    _check("runs", True, "patient_independent_oof returned", results)

    submitted_predictions = _as_strings(submitted.get("predictions", []))
    _check(
        "prediction_length",
        len(submitted_predictions) == len(frame),
        f"expected {len(frame)}, got {len(submitted_predictions)}",
        results,
    )
    _check(
        "valid_labels",
        bool(set(submitted_predictions) <= set(CLASSES)),
        "all predictions use supported AAMI labels",
        results,
    )

    # Independently assemble OOF predictions by calling the candidate per fold.
    independent = np.empty(len(frame), dtype=object)
    independent_ok = True
    for train_idx, test_idx in folds:
        try:
            independent[test_idx] = candidate.fit_predict(
                features.iloc[train_idx], labels[train_idx], features.iloc[test_idx]
            )
        except Exception as exc:  # noqa: BLE001
            independent_ok = False
            _check(
                f"fold_predict_{len(independent[test_idx])}",
                False,
                f"fit_predict raised {type(exc).__name__}: {exc}",
                results,
            )
            break
    if independent_ok:
        independent = _as_strings(independent)
        _check(
            "predictions_from_disjoint_folds",
            bool(np.array_equal(independent, submitted_predictions)),
            "submitted OOF matches predictions produced from canonical patient-disjoint folds",
            results,
        )

        expected_groups = [
            {
                "train": sorted(set(groups[tr])),
                "test": sorted(set(groups[va])),
            }
            for tr, va in folds
        ]
        _check(
            "fold_metadata_exact",
            submitted.get("fold_groups") == expected_groups,
            "reported fold_groups equal the canonical fold assignments",
            results,
        )

        metrics = compute_metrics(labels, independent)
    else:
        metrics = {}

    # Dummy baseline on the same folds.
    dummy_pred = np.empty(len(frame), dtype=object)
    for train_idx, test_idx in folds:
        dummy = DummyClassifier(strategy="most_frequent").fit(
            features.iloc[train_idx], labels[train_idx]
        )
        dummy_pred[test_idx] = dummy.predict(features.iloc[test_idx])
    dummy_metrics = compute_metrics(labels, _as_strings(dummy_pred))

    if metrics:
        _check(
            "balanced_accuracy",
            metrics["balanced_accuracy"] >= MIN_BALANCED_ACCURACY,
            f"got {metrics['balanced_accuracy']}, need >= {MIN_BALANCED_ACCURACY}",
            results,
        )
        _check(
            "f1_macro",
            metrics["f1_macro"] >= MIN_F1_MACRO,
            f"got {metrics['f1_macro']}, need >= {MIN_F1_MACRO}",
            results,
        )
        _check(
            "beats_dummy",
            metrics["balanced_accuracy"]
            >= dummy_metrics["balanced_accuracy"] + MIN_DUMMY_MARGIN,
            f"model {metrics['balanced_accuracy']}, dummy {dummy_metrics['balanced_accuracy']}",
            results,
        )

    # Reported metrics must match independently recomputed metrics.
    reported = submitted.get("metrics", {})
    if metrics and isinstance(reported, dict):
        consistent = all(
            abs(float(reported.get(key, -1)) - value) <= 0.02
            for key, value in metrics.items()
        )
        _check(
            "metrics_not_fabricated",
            consistent,
            "reported metrics match recomputed OOF metrics",
            results,
        )
    else:
        _check("metrics_not_fabricated", False, "metrics missing", results)

    # Determinism.
    second = candidate.patient_independent_oof(features, labels, groups)
    _check(
        "determinism",
        bool(
            np.array_equal(
                _as_strings(second.get("predictions", [])), submitted_predictions
            )
        ),
        "two identical calls give identical predictions",
        results,
    )

    # Test-label leakage: a label column in test features cannot be used.
    train_idx, test_idx = folds[0]
    test_with_labels = features.iloc[test_idx].copy()
    test_with_labels["label"] = np.random.default_rng(0).permutation(labels[test_idx])
    try:
        pred_a = _as_strings(
            candidate.fit_predict(
                features.iloc[train_idx], labels[train_idx], features.iloc[test_idx]
            )
        )
        pred_b = _as_strings(
            candidate.fit_predict(
                features.iloc[train_idx], labels[train_idx], test_with_labels
            )
        )
        _check(
            "test_labels_unused",
            bool(np.array_equal(pred_a, pred_b)),
            "adding a permuted test label column does not change predictions",
            results,
        )
    except Exception as exc:  # noqa: BLE001
        _check("test_labels_unused", False, f"raised {type(exc).__name__}", results)

    # Preprocessing boundary: one test row must not affect other test rows.
    changed = features.iloc[test_idx].copy()
    perturbed = changed.copy()
    perturbed.iloc[0] = perturbed.iloc[0] + 5.0
    try:
        base = _as_strings(
            candidate.fit_predict(features.iloc[train_idx], labels[train_idx], changed)
        )
        altered = _as_strings(
            candidate.fit_predict(
                features.iloc[train_idx], labels[train_idx], perturbed
            )
        )
        others = np.arange(1, len(base))
        _check(
            "preprocessing_train_only",
            bool(np.array_equal(base[others], altered[others])),
            "changing one test row does not change other test predictions",
            results,
        )
    except Exception as exc:  # noqa: BLE001
        _check(
            "preprocessing_train_only", False, f"raised {type(exc).__name__}", results
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
            len(_as_strings(nan_pred)) == len(test_idx),
            "NaN features imputed without failure",
            results,
        )
    except Exception as exc:  # noqa: BLE001
        _check("nan_imputation", False, f"raised {type(exc).__name__}", results)

    expected = json.loads((HERE / "expected_output.json").read_text(encoding="utf-8"))
    _check(
        "expected_output_present",
        bool(expected.get("metrics")),
        "reference metrics recorded as calibration evidence",
        results,
    )

    passed = all(item["passed"] for item in results.values())
    return {
        "checks": results,
        "metrics": metrics,
        "dummy_metrics": dummy_metrics,
        "passed": passed,
    }


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
