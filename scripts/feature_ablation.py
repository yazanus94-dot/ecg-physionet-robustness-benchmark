"""Feature-ablation experiments for ECG-ROBUST (Phase 6 -- feature selection).

Reuses the pipeline's sampled beat set and the 5-fold patient-independent
group splits (same seeds -> identical to the pipeline's CV design).
Evaluates the main models on feature ablations and writes
`reports/results/feature_ablation.csv`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ecg_robust.annotations import CLASSES
from ecg_robust.evaluation import evaluate
from ecg_robust.models import build_models
from ecg_robust.pipeline import load_experiment
from ecg_robust.visualization import plot_feature_ablation

FEATURE_ABLATIONS: dict[str, object] = {
    "full": lambda cols: cols,
    "drop_rr_features": lambda cols: [c for c in cols if c not in ("rr_s", "rr_ratio")],
    "drop_morphology": lambda cols: [
        c for c in cols if c not in ("rise_time_s", "fall_time_s", "zero_crossings")
    ],
    "drop_spectral": lambda cols: [
        c
        for c in cols
        if c
        not in (
            "spectral_centroid",
            "spectral_flatness",
            "fft_peak_frequency",
            "fft_total_energy",
        )
    ],
    "drop_stats": lambda cols: [
        c
        for c in cols
        if c not in ("skew", "kurtosis") + tuple(f"p{p}" for p in (5, 25, 50, 75, 95))
    ],
    "top12_by_rf_importance": None,  # special: rank by RF importance
}


def ablation_columns(variant: str, X_all: pd.DataFrame, y_all: np.ndarray) -> list[str]:
    """Columns to use for a given ablation variant."""
    if variant == "full":
        return list(X_all.columns)
    fn = FEATURE_ABLATIONS[variant]
    if fn is not None:
        return fn(list(X_all.columns))
    # top12_by_rf_importance: fit an RF (diagnostic) and keep the 12 most
    # important features by default importance.
    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier(n_estimators=200, random_state=42).fit(X_all, y_all)
    order = np.argsort(-rf.feature_importances_)[:12]
    return [X_all.columns[i] for i in order]


def _fit_eval_accuracy(X, y, model, tr, va) -> tuple[float, float, float]:
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )
    pipe.fit(X.iloc[tr], y[tr])
    pred = np.asarray(pipe.predict(X.iloc[va]))
    m = evaluate(y[va], pred, CLASSES)
    return (float(m["accuracy"]), float(m["f1_macro"]), float(m["balanced_accuracy"]))


def main() -> int:
    X_all, y_all, _groups, folds = load_experiment()
    models = build_models()
    rows = []
    for variant in FEATURE_ABLATIONS:
        cols = ablation_columns(variant, X_all, y_all)
        X_var = X_all[cols]
        for name, model in models.items():
            accs, f1s, bass = [], [], []
            for tr, va in folds:
                a, f1, b = _fit_eval_accuracy(X_var, y_all, model, tr, va)
                accs.append(a)
                f1s.append(f1)
                bass.append(b)
            rows.append(
                {
                    "variant": variant,
                    "n_features": len(cols),
                    "model": name,
                    "accuracy_mean": float(np.mean(accs)),
                    "accuracy_std": float(np.std(accs)),
                    "f1_macro_mean": float(np.mean(f1s)),
                    "balanced_accuracy_mean": float(np.mean(bass)),
                }
            )
    df = pd.DataFrame(rows)
    out = "reports/results/feature_ablation.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    plot_feature_ablation(df, "feature_ablation.png")
    print(f"wrote {out} and reports/figures/feature_ablation.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
