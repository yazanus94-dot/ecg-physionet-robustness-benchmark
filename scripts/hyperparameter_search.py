"""Hyperparameter search using training folds only (spec Phase 6).

For each model and each patient-independent CV training fold, a
`GridSearchCV` (inner CV = StratifiedGroupKFold on the fold's training
data) tunes hyperparameters on that fold's training data ONLY — the same
patient-independent design as the pipeline CV. The reported
`best_params` are diagnostics: primary performance is the unbiased
patient-independent CV table in `reports/results/cv_metrics.csv`.

Writes `reports/results/hyperparameter_search.json`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ecg_robust.models import build_models
from ecg_robust.pipeline import load_experiment

GRID = {
    "dummy": {},
    "logistic_regression": {"model__C": [0.1, 1.0, 10.0]},
    "random_forest": {
        "model__max_features": ["sqrt", "log2", None],
        "model__min_samples_leaf": [1, 5],
    },
    "hist_gradient_boosting": {"model__max_iter": [50, 100]},
    "svm_rbf": {"model__C": [0.1, 10.0]},
}


def main() -> int:
    X_all, y_all, groups_all, folds = load_experiment()
    models = build_models()
    results = []
    for name in sorted(models):
        grid = {k: v for k, v in GRID.get(name, {}).items()}
        for fold_idx, (tr, _va) in enumerate(folds):
            if not grid:
                results.append(
                    {
                        "model": name,
                        "fold": fold_idx,
                        "best_params": {},
                        "search_validation": "no grid (deterministic default)",
                    }
                )
                continue
            pipe = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", build_models()[name]),
                ]
            )
            inner_cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
            gs = GridSearchCV(
                pipe,
                grid,
                cv=inner_cv,
                scoring="accuracy",
                refit=True,
                error_score="raise",
            )
            # Patient-independent inner CV on the fold's training data only:
            # StratifiedGroupKFold requires the recording groups.
            gs.fit(X_all.iloc[tr], y_all[tr], groups=groups_all[tr])
            results.append(
                {
                    "model": name,
                    "fold": fold_idx,
                    "best_params": {
                        k.split("model__")[-1]: v for k, v in gs.best_params_.items()
                    },
                    "best_score_mean_inner_cv": float(gs.best_score_),
                    "search_validation": "training-fold-only (StratifiedGroupKFold on training data)",
                }
            )
    df = pd.DataFrame(results)
    out = Path("reports/results/hyperparameter_search.json")
    out.write_text(df.to_json(orient="records", indent=2), encoding="utf-8")
    print(df.to_string(index=False))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
