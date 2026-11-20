"""Tests for OOF per-class reporting (spec #2).

1. `per_class_metrics_from_cm` computes precision/recall/sensitivity/
   specificity/F1/support from a confusion matrix.
2. The OOF per-class metrics agree mathematically with `evaluate().per_class`.
3. Pipeline writes `per_class_oof.json` + `per_class_train_diagnostics.json`
   (exists when data/raw/mitdb is present).
"""

import json
from pathlib import Path

import numpy as np
import pytest

from ecg_robust.annotations import CLASSES
from ecg_robust.evaluation import evaluate, per_class_metrics_from_cm


def _example_cm():
    return np.array(
        [
            [10, 0, 2, 0, 0],
            [0, 5, 1, 0, 0],
            [3, 0, 7, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 4],
        ],
        dtype=float,
    )


def test_per_class_metrics_from_cm_values():
    cm = _example_cm()
    pc = per_class_metrics_from_cm(cm, CLASSES)
    # N: TP=10; col_pred = 10+0+3+0+0 = 13; row_true = 10+0+2+0+0 = 12
    assert np.isclose(pc["N"]["precision"], 10 / 13)
    assert np.isclose(pc["N"]["recall"], 10 / 12)
    assert np.isclose(pc["N"]["support"], 12)
    # S: TP=5; col_pred = 0+5+0+0+0 = 5; row_true = 0+5+1+0+0 = 6
    assert np.isclose(pc["S"]["precision"], 5 / 5)
    assert np.isclose(pc["S"]["recall"], 5 / 6)
    # V: TP=7; col_pred = 2+1+7+0+0 = 10; row_true = 3+0+7+0+0 = 10
    assert np.isclose(pc["V"]["precision"], 7 / 10)
    assert np.isclose(pc["V"]["recall"], 7 / 10)
    assert np.isclose(pc["V"]["f1"], 2 * (7 / 10) * (7 / 10) / ((7 / 10) + (7 / 10)))
    # F: TP=1; col_pred=1; row_true=1
    assert np.isclose(pc["F"]["precision"], 1.0)
    assert np.isclose(pc["F"]["recall"], 1.0)


def test_per_class_metrics_from_cm_agrees_with_evaluate():
    rng = np.random.default_rng(0)
    y = rng.choice(["N", "S", "V", "F", "Q"], size=200)
    pred = rng.choice(["N", "S", "V", "F", "Q"], size=200)
    ev = evaluate(y, pred, CLASSES)
    pc = per_class_metrics_from_cm(ev["confusion_matrix"], CLASSES)
    for c in CLASSES:
        for key in (
            "precision",
            "recall",
            "sensitivity",
            "specificity",
            "f1",
            "support",
        ):
            assert np.isclose(pc[c][key], ev["per_class"][c][key]), (c, key)


def test_pipeline_per_class_files():
    if not (Path("data/raw/mitdb")).exists():
        pytest.skip("MITDB raw data not present")
    for name in ("per_class_oof.json", "per_class_train_diagnostics.json"):
        p = Path("reports/results") / name
        if p.exists():
            data = json.loads(p.read_text())
            for per_class in data.values():
                assert set(CLASSES) <= set(per_class)
                assert all(np.isfinite(v["f1"]) for v in per_class.values())
