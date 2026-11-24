"""Class-weight ablation study (spec Phase 6: class imbalance & class weights).

Runs the patient-independent CV (same folds/sample as the pipeline) with
each primary model fitted **without** class weights, and writes
`reports/results/class_weights_study.csv` — a diagnostic comparison
(balanced vs unbalanced) for the report.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ecg_robust.annotations import CLASSES
from ecg_robust.evaluation import evaluate
from ecg_robust.pipeline import load_experiment


def _unweighted_models() -> dict[str, Any]:
    return {
        "dummy": DummyClassifier(strategy="most_frequent", random_state=42),
        "logistic_regression": LogisticRegression(
            max_iter=2000, random_state=42
        ),  # class_weight=None
        "random_forest": RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight=None
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=100, random_state=42
        ),
        "svm_rbf": SVC(kernel="rbf", C=1.0, random_state=42, class_weight=None),
    }


def _cv_mean(models, X_all, y_all, folds):
    out = {}
    for name, model in models.items():
        accs, f1s, bass = [], [], []
        for tr, va in folds:
            pipe = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", model),
                ]
            )
            pipe.fit(X_all.iloc[tr], y_all[tr])
            pred = np.asarray(pipe.predict(X_all.iloc[va]))
            m = evaluate(y_all[va], pred, CLASSES)
            accs.append(m["accuracy"])
            f1s.append(m["f1_macro"])
            bass.append(m["balanced_accuracy"])
        out[name] = (float(np.mean(accs)), float(np.mean(f1s)), float(np.mean(bass)))
    return out


def main() -> int:
    X_all, y_all, _g, folds = load_experiment()
    from ecg_robust.models import build_models

    balanced = _cv_mean(build_models(), X_all, y_all, folds)
    unbalanced = _cv_mean(_unweighted_models(), X_all, y_all, folds)
    rows = []
    for name in sorted(balanced):
        b = balanced[name]
        u = unbalanced.get(name, (float("nan"),) * 3)
        rows.append(
            {
                "model": name,
                "class_weight": "balanced",
                "accuracy_mean": b[0],
                "f1_macro_mean": b[1],
                "balanced_accuracy_mean": b[2],
            }
        )
        rows.append(
            {
                "model": name,
                "class_weight": "none",
                "accuracy_mean": u[0],
                "f1_macro_mean": u[1],
                "balanced_accuracy_mean": u[2],
            }
        )
    df = pd.DataFrame(rows)
    out = "reports/results/class_weights_study.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
