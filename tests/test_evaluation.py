"""Tests for evaluation metrics."""

import numpy as np

from ecg_robust.evaluation import evaluate

CLASSES = ["N", "S", "V", "F", "Q"]


def test_perfect_prediction_all_ones():
    y = ["N", "S", "V", "F", "Q", "N"]
    m = evaluate(y, y, CLASSES)
    assert m["accuracy"] == 1.0
    assert m["f1_macro"] == 1.0
    assert m["balanced_accuracy"] == 1.0
    assert m["f1_weighted"] == 1.0
    assert np.allclose([m["per_class"][c]["sensitivity"] for c in CLASSES], 1.0)


def test_confusion_matrix_labels_and_shape():
    y = ["N", "N", "V", "N", "S"]
    pred = ["N", "V", "V", "N", "S"]
    m = evaluate(y, pred, CLASSES)
    cm = np.array(m["confusion_matrix"])
    assert cm.shape == (5, 5)
    assert m["confusion_matrix_labels"] == CLASSES
    assert cm.sum() == 5


def test_accuracy_and_sensitivity_formulas():
    y = ["N", "N", "N", "N", "V"]
    pred = ["N", "N", "V", "N", "N"]
    m = evaluate(y, pred, CLASSES)
    assert np.isclose(m["accuracy"], 0.6)
    # N: TP=3, FN=1 -> recall 0.75; V: TP=0, FN=1 -> recall 0
    assert np.isclose(m["per_class"]["N"]["sensitivity"], 0.75)
    assert np.isclose(m["per_class"]["V"]["sensitivity"], 0.0)
    assert np.isclose(
        m["per_class"]["N"]["precision"], 0.75
    )  # TP=3, FP=1 (idx4 pred N, true V)
    assert np.isclose(m["per_class"]["V"]["precision"], 0.0)


def test_specificity_per_class():
    y = ["N", "N", "V"]
    pred = ["N", "V", "V"]
    m = evaluate(y, pred, CLASSES)
    # for class N: FP=1 (the V mispredicted as N? no: pred has V where true N -> FP for V)
    # N: TP=1, FP=0 -> precision 1; V: TP=1, FP=1 -> precision 0.5
    assert np.isclose(m["per_class"]["N"]["precision"], 1.0)
    assert np.isclose(m["per_class"]["V"]["precision"], 0.5)
    assert np.isclose(m["per_class"]["N"]["specificity"], 1.0)


def test_zero_support_classes_get_zero_division_0():
    y = ["N", "N"]
    pred = ["N", "N"]
    m = evaluate(y, pred, CLASSES)
    assert np.isfinite(m["f1_macro"])
    assert m["per_class"]["S"]["support"] == 0


def test_macro_and_weighted_f1_relations():
    rng = np.random.default_rng(0)
    y = rng.choice(["N", "S", "V"], size=200)
    pred = rng.choice(["N", "S", "V"], size=200)
    m = evaluate(y, pred, CLASSES)
    assert 0 <= m["f1_macro"] <= 1
    assert 0 <= m["f1_weighted"] <= 1
    assert 0 <= m["specificity_macro"] <= 1
