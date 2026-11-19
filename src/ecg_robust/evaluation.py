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
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(np.mean(prec)),
        "recall_macro": float(np.mean(rec)),
        "sensitivity_macro": float(np.mean(rec)),
        "specificity_macro": float(np.mean(specificity)),
        "f1_macro": float(np.mean(f1)),
        "f1_weighted": (
            float(np.average(f1, weights=sup)) if sup.sum() > 0 else 0.0
        ),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": classes,
        "per_class": per_class,
        "classification_report": classification_report(
            y_true, y_pred, labels=classes, zero_division=0
        ),
    }
