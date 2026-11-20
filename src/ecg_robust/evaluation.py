"""Model evaluation: metrics, per-class report, confusion matrix.

Metric definitions (per the project specification):

- ``Precision = TP / (TP + FP)``
- ``Recall = TP / (TP + FN)`` (per class; macro/micro averaged as noted)
- ``Sensitivity = TP / (TP + FN)`` (per-class recall; macro-averaged reported)
- ``Specificity = TN / (TN + FP)`` (per class; macro-averaged reported)
- ``F1 = 2 * Precision * Recall / (Precision + Recall)``
- ``Accuracy = (TP + TN) / Total``
- ``Macro F1``: arithmetic mean of the five class-specific F1 scores.
- ``Weighted F1``: F1 averaged by class support.
- ``Balanced accuracy``: mean of per-class recalls (``sklearn``).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def per_class_metrics_from_cm(cm, classes) -> dict:
    """Per-class precision, recall, sensitivity, specificity, F1, support.

    Computed from a confusion matrix ``cm`` (rows = true class, columns =
    predicted class). Used to aggregate out-of-fold predictions from the
    summed confusion matrix. Mathematically agrees with
    :func:`evaluate`'s per-class table (spec #2).
    """
    cm = np.asarray(cm, dtype=float)
    total = float(cm.sum())
    out = {}
    for i, c in enumerate(classes):
        tp = float(cm[i, i])
        row_true = float(cm[i, :].sum())  # support (true class = c)
        col_pred = float(cm[:, i].sum())  # predicted class = c
        fp = col_pred - tp
        row_true - tp
        tn = total - row_true - col_pred + tp
        precision = tp / col_pred if col_pred > 0 else 0.0
        recall = tp / row_true if row_true > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        out[c] = {
            "precision": float(precision),
            "recall": float(recall),
            "sensitivity": float(recall),
            "specificity": float(specificity),
            "f1": float(f1),
            "support": int(row_true),
        }
    return out


def evaluate(y_true, y_pred, classes) -> dict:
    """Compute all requested metrics + confusion matrix + per-class report."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = list(classes)
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    # Specificity per class: TN / (TN + FP), then macro-averaged.
    specificity = []
    for i in range(len(classes)):
        fp = int(cm[:, i].sum() - cm[i, i])
        tn = int(cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i])
        spec_i = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificity.append(spec_i)

    per_class = {
        c: {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "sensitivity": float(rec[i]),
            "specificity": float(specificity[i]),
            "f1": float(f1[i]),
            "support": int(sup[i]),
        }
        for i, c in enumerate(classes)
    }
    # Cross-check: per-class metrics from the confusion matrix agree with
    # the (y_true, y_pred) computation above.
    from ecg_robust.evaluation import per_class_metrics_from_cm as _pcm

    for c in classes:
        assert np.isclose(
            _pcm(cm, classes)[c]["precision"], per_class[c]["precision"]
        ), c
        assert np.isclose(_pcm(cm, classes)[c]["recall"], per_class[c]["recall"]), c
        assert np.isclose(
            _pcm(cm, classes)[c]["specificity"], per_class[c]["specificity"]
        ), c
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(np.mean(prec)),
        "recall_macro": float(np.mean(rec)),
        "sensitivity_macro": float(np.mean(rec)),
        "specificity_macro": float(np.mean(specificity)),
        "f1_macro": float(np.mean(f1)),
        "f1_weighted": (float(np.average(f1, weights=sup)) if sup.sum() > 0 else 0.0),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": classes,
        "per_class": per_class,
        "classification_report": classification_report(
            y_true, y_pred, labels=classes, zero_division=0
        ),
    }
