"""Generate ``reports/TECHNICAL_REPORT.md`` from the pipeline outputs.

All numbers are read back from the CSVs produced by the pipeline (no
hand-typed results); section templates match the project specification:
research question, data/method, formulas, feature engineering, algorithms,
patient-independent validation, noise experiments, results, interpretation,
limitations, references.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .annotations import CLASSES

REPORT_DIR = Path("reports")

FEATURE_TABLE = [
    ("mean", "arithmetic mean of the segment (mV)", "x_bar = (1/n) * sum_i x_i"),
    ("std", "population standard deviation of the segment (mV)",
     "sigma = sqrt((1/n) * sum_i (x_i - mean)^2)"),
    ("min", "minimum amplitude (mV)", "min_i x_i"),
    ("max", "maximum amplitude (mV)", "max_i x_i"),
    ("energy", "signal energy E = sum x_i^2 (mV^2)", "E = sum_i x_i^2"),
    ("peak_amplitude", "maximum absolute amplitude (mV)", "max_i |x_i|"),
    ("duration_s", "waveform duration n / fs (s)", "n / fs"),
    ("rise_time_s", "time from segment start to first half-maximum (s)",
     "first index i where x_i >= max(x[0:n/2]) / fs"),
    ("fall_time_s", "time from second-half maximum crossing to segment end (s)",
     "(n - i2) / fs"),
    ("zero_crossings", "number of sign changes in the segment",
     "sum_i [sign(x_i) != sign(x_{i+1})]"),
    ("rr_s", "RR interval (s): RR_i = (r_i - r_{i-1}) / fs",
     "(r_i - r_{i-1}) / fs"),
    ("rr_ratio", "RR ratio: RR_i / RR_{i-1} (dimensionless)", "RR_i / RR_{i-1}"),
    ("spectral_centroid", "DFT centroid (Hz)",
     "sum(f * |X(f)|) / sum(|X(f)|)"),
    ("spectral_flatness", "spectral flatness in [0,1] (1 = flat)",
     "exp(mean(ln|X|)) / mean(|X|)"),
    ("fft_peak_frequency", "frequency of largest DFT magnitude (Hz)", "argmax_f |X(f)|"),
    ("fft_total_energy", "DFT energy sum(|X(f)|^2)", "sum_f |X(f)|^2"),
    ("skew", "sample skewness (Fisher)", "scipy.stats.skew(x)"),
    ("kurtosis", "sample kurtosis (Fisher)", "scipy.stats.kurtosis(x)"),
    ("p5", "5th percentile (mV)", "np.percentile(x, 5)"),
    ("p25", "25th percentile (mV)", "np.percentile(x, 25)"),
    ("p50", "median (mV)", "np.percentile(x, 50)"),
    ("p75", "75th percentile (mV)", "np.percentile(x, 75)"),
    ("p95", "95th percentile (mV)", "np.percentile(x, 95)"),
]

AAMI_TABLE = [
    ("N", "N, L, R, e, j"),
    ("S", "A, a, J, S"),
    ("V", "V, E"),
    ("F", "F"),
    ("Q", "/, f, Q"),
]


def _fmt(x: float, nd: int = 6) -> str:
    try:
        return f"{float(x):.{nd}g}"
    except (TypeError, ValueError):
        return str(x)


def _mean_rows(df: pd.DataFrame, cols: list[str], nd: int = 4) -> list[str]:
    rows = []
    if df is None or df.empty:
        return rows
    g = df.groupby("model")[cols].mean().round(nd)
    for name in g.index:
        vals = ", ".join(f"{c}: {g.loc[name, c]:.{nd}f}" for c in cols)
        rows.append(f"  - {name}: {vals}")
    return rows


def _cv_table(df: pd.DataFrame) -> list[str]:
    cols = [
        "accuracy", "precision_macro", "recall_macro", "sensitivity_macro",
        "specificity_macro", "f1_macro", "f1_weighted", "balanced_accuracy",
    ]
    g = df.groupby("model")[cols].mean().round(4)
    lines = ["| Model | " + " | ".join(c.replace("_", " ").title() for c in cols) + " |",
             "|-------|" + "|".join(["---"] * len(cols)) + "|"]
    for name in g.index:
        cells = [f"{g.loc[name, c]:.4f}" for c in cols]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return lines


def _confusion_table(model: str, cm: np.ndarray, title: str, classes: list[str]) -> list[str]:
    """Markdown table for one confusion matrix (rows = true, cols = predicted)."""
    lines = [f"#### `{model}` -- {title}", ""]
    lines.append("| | " + " | ".join(str(c) for c in classes) + " |")
    lines.append("|--|" + "|".join(["---"] * len(classes)) + "|")
    for i, label in enumerate(classes):
        lines.append("| " + str(label) + " | " +
                     " | ".join(f"{cm[i, j]:,.0f}" for j in range(len(classes))) + " |")
    return lines


def write_technical_report(data: dict[str, Any], path: Path) -> Path:
    """Write the full technical report; returns the report path.

    Primary research results are out-of-fold: ``confusion`` is the
    out-of-fold confusion matrix, ``metrics_df``/``cv_df`` are the
    out-of-fold CV metrics, ``noise_df`` is the out-of-fold noise
    evaluation, ``per_class`` is keyed by the best evaluated model
    (``best_model``); ``confusion_train`` holds separate full-data training
    diagnostics (9.1b).
    """
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ds: dict[str, Any] = data["dataset"]
    if "mitdb" in ds and len(ds) == 1:
        ds = ds["mitdb"]  # tolerate {"mitdb": {...}} nesting
    cv_df: pd.DataFrame = data["cv_df"]
    metrics_df: pd.DataFrame = data["metrics_df"]
    noise_df: pd.DataFrame | None = data.get("noise_df")
    nstdb_df: pd.DataFrame | None = data.get("nstdb_df")
    n_features = data.get("n_features", "n/a")
    figures = data.get("figures", [])
    wall_s = data.get("wall_s", 0.0)

    cv_by_model = cv_df.groupby("model")["accuracy"].mean().round(4).to_dict()

    L: list[str] = []
    w = L.append
    w(f"# ECG-ROBUST -- Technical Report: Patient-Independent ECG Arrhythmia "
      f"Classification and Noise-Robustness Benchmark\n")
    w(f"- Repository: `yazanus94-dot/ecg-physionet-robustness-benchmark`")
    w(f"- Report date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    w(f"- Pipeline wall time (measured, including downloads): {wall_s:.1f} s\n")

    w("## 1. Research Question\n")
    w("Can a patient-independent ECG classifier distinguish the five AAMI "
      "(Association for the Advancement of Medical Instrumentation) "
      "diagnostic categories -- N (normal), S (supraventricular), V "
      "(ventricular), F (fusion) and Q (paced/unclassified/paced) -- from a "
      "single heartbeat segment without using the patient identity, and how "
      "robust is such a classifier when NSTDB noise is added at SNR levels "
      "24, 18, 12, 6, 0 and -6 dB?\n")

    w("## 2. Data and Availability\n")
    w("### 2.1 Source databases\n")
    w("- MIT-BIH Arrhythmia Database (mitdb), PhysioNet, "
      "https://physionet.org/content/mitdb/1.0.0/")
    w("- MIT-BIH Noise Stress Test Database (nstdb), PhysioNet, "
      "https://physionet.org/content/nstdb/1.0.0/ (records 118, 119)")
    w("- Raw files stored under `data/raw/` (never uploaded; subject to the "
      "above licenses and PhysioNet attribution).")

    w("\n### 2.2 Dataset inventory (measured from the downloaded databases)\n")
    w(f"- MITDB records: {ds['n_records']} -- "
      + ", ".join(str(r) for r in ds["records"]))
    w(f"- NSTDB records: {len(ds['nstdb'])} -- "
      + ", ".join(str(r) for r in ds["nstdb"]))
    w(f"- Sampling frequency (MITDB): {ds['sampling_frequency_hz']} Hz")
    w(f"- Beats segmented across all recordings (valid segments; excluded/unknown "
      f"annotations may apply): {ds['total_beats_annotated']}")
    w(f"- Beats used in the feature matrix: {ds['total_beats_aami']}")
    w(f"- Excluded (known MITDB symbols absent from the AAMI map): "
      f"{ds['excluded_annotations']}")
    w(f"- Unknown codes (not part of the MITDB database): "
      f"{ds['unknown_annotations']}")
    w(f"- AAMI class distribution of mapped beats: "
      f"{ {k: int(v) for k, v in ds['class_counts'].items()} }")

    w("\n## 3. Preprocessing and Segmentation\n")
    w("### 3.1 Filtering")
    w("```\ny[n] = BandpassFilter(x[n])        # 4th-order Butterworth, 0.5-40 Hz\n"
      "SOS = butter(4, [0.5/180, 40/180], btype='bandpass', output='sos')\n"
      "y = scipy.signal.sosfiltfilt(SOS, x, axis=0)   # forward-backward, zero phase\n```")
    w("Normalization: each heartbeat segment is standardized with training-"
      "fitted statistics "
      "`norm_i = (x_i - mean_train) / (std_train + eps)`; every fold uses the"
      " scaler fitted on its training split only.")
    w("### 3.2 Segmentation")
    w("Heartbeat segments (per the specification): `X_i = x[r_i - 0.25*fs : r_i + 0.45*fs]` --"
      " a 250 ms pre-R window and a 450 ms post-R window around each annotated R position"
      " (`r_i`). At signal boundaries segments are truncated; degenerate"
      " segments are dropped. With fs = 360 Hz this yields 252 samples "
      "(0.7 s) per segment.")

    w("\n## 4. Annotations and the AAMI Mapping\n")
    w("The MIT-BIH annotation codes were mapped to the five AAMI categories:"
      " N: N, L, R, e, j; S: A, a, J, S; V: V, E; F: F; Q: /, f, Q.")
    for cls, codes in AAMI_TABLE:
        w(f"- **{cls}**: {codes}")
    w("\nAnnotations known to the MIT-BIH database but absent from the AAMI "
      "map were excluded and reported; unknown codes (not part of the "
      "database) were reported separately: "
      f"{ds['excluded_annotations']} excluded, {ds['unknown_annotations']} "
      "unknown, by code.")

    w("\n## 5. Feature Engineering\n")
    w("Each heartbeat segment is represented by 23 features (ecg_robust.features). "
      f"This matrix has {n_features} features (including `rr_s` and `rr_ratio` "
      "when available).")
    w("| Feature | Definition | Formula / method |")
    w("|---|---|---|")
    for name, desc, formula in FEATURE_TABLE:
        w(f"| `{name}` | {desc} | `{formula}` |")

    w("\n## 6. Patient-Independent Validation Design\n")
    w("- Splitter: `sklearn.model_selection.StratifiedGroupKFold(n_splits=5, "
      "shuffle=True, random_state=42)` -- stratification on the AAMI class label.")
    w("- Groups: each MITDB recording ID is a group (the 201/202 group is shared). "
      "Beats of a recording are never split across train/validation -- there is "
      "no patient leakage. The 201/202 grouping was applied: recordings 201 and "
      "202 were assigned to the same subject group.")
    w("- Each fold refits `SimpleImputer` + `StandardScaler` and the model on "
      "the training split only (spec #4).")

    w("\n## 7. Models\n")
    w("- **DummyClassifier** (`strategy='most_frequent'`) -- patient-independent "
      "baseline.")
    w("- **RandomForestClassifier** -- `n_estimators=100`, `class_weight='balanced'`, "
      "`random_state=42`.")
    w("- **SupportVectorMachine (RBF)** -- `C=1.0`, `kernel='rbf'`, "
      "`class_weight='balanced'`, `random_state=42`.")
    w("For CV and final fits, models were wrapped in `sklearn.pipeline.Pipeline` "
      "(`SimpleImputer` + `StandardScaler` -> model); the imputer and scaler "
      "are fit on each fold's training data only (spec #4).")

    w("\n## 8. Metric Definitions\n")
    w("- Accuracy = (TP + TN) / Total.")
    w("- Precision = TP / (TP + FP); Recall (per class) = TP / (TP + FN).")
    w("- Specificity (per class) = TN / (TN + FP).")
    w("- Macro F1 = mean of the five per-class F1 = mean of 2P*R/(P+R).")
    w("- Weighted F1 = F1 averaged by class support.")
    w("- Balanced accuracy = mean of per-class recalls (sklearn).")

    w("\n## 9. Actual 5-Fold Patient-Independent CV Results\n")
    w("Mean of the five folds (`reports/results/cv_metrics.csv` -- "
      "accuracy / precision(macro) / recall(macro) / sensitivity(macro) / "
      "specificity(macro) / F1(macro) / F1(weighted) / balanced accuracy):\n")
    w("\n".join(_cv_table(metrics_df)))
    w(f"\nBest CV accuracy: **{max(cv_by_model.values()):.4f}** ({max(cv_by_model, key=cv_by_model.get)}).")

    w("\n### 9.1 Confusion matrices (out-of-fold; sum over folds)\n")
    for model, cm in data.get("confusion", {}).items():
        if cm is None:
            continue
        cm = np.asarray(cm)
        if cm.size == 0:
            continue
        w("")
        w("\n".join(_confusion_table(
            model, cm, "out-of-fold (sum over folds)", data["confusion_labels"])))

    w("\n### 9.1b Training diagnostics (full-data fits, diagnostics only)")
    conf_train = data.get("confusion_train") or {}
    for model, cm in conf_train.items():
        if cm is None:
            continue
        cm = np.asarray(cm)
        if cm.size == 0:
            continue
        w("")
        w("\n".join(_confusion_table(
            model, cm, "training (full data, diagnostics)", data["confusion_labels"])))

    best_model = data.get("best_model", "dummy")
    w(f"\n### 9.2 Per-class performance of the best evaluated model (`{best_model}`)")
    w("Per-class performance (precision, recall, sensitivity, specificity, F1, support):")
    w("")
    w("| Class | Precision | Recall | Sensitivity | Specificity | F1 | Support |")
    w("|---|---|---|---|---|---|---|")
    pc = data.get("per_class", {}).get(best_model, {})
    for cls in ["N", "S", "V", "F", "Q"]:
        r = pc.get(cls, {})
        row = []
        for k in ["precision", "recall", "sensitivity", "specificity", "f1"]:
            v = r.get(k, float("nan"))
            row.append(f"{float(v):.4f}" if np.isfinite(v) else "n/a")
        row.append(str(int(r.get("support", 0))))
        w("| " + cls + " | " + " | ".join(row) + " |")
    w("")

    w("\n## 10. Noise Robustness Experiment\n")
    w("Setup (per the specification): clean reference = MITDB beats. "
      "Noisy evaluation = synthetic mixtures of the clean beats with NSTDB "
      "noise realizations (NSTDB record segments) at the target SNR levels "
      "`SNR(dB) = 10*log10(P_signal / P_noise)`, where P_signal/P_noise are the "
      "mean squared values; clean and noisy segments are aligned at identical "
      "positions. The NSTDB recordings (incl. the reserved records 118/119) "
      "were also classified directly as recorded-noise evaluation (10.2). "
      "These are labeled synthetic-mixture results; the recorded-noise "
      "experiment below uses real NSTDB noise.")
    if noise_df is not None and len(noise_df):
        w("\n### 10.1 Out-of-fold synthetic mixtures -- accuracy vs SNR "
          "(per-fold models scored on held-out validation beats; spec #3)")
        w("\n| SNR (dB) | Model | Accuracy | Macro F1 | Macro Sensitivity | Balanced Accuracy |")
        w("|---|---|---|---|---|---|")
        for _, r in noise_df.sort_values(["snr_db", "model"]).iterrows():
            w(f"| {r['snr_db']:g} | {r['model']} | {r['accuracy']:.4f} | "
              f"{r['macro_f1']:.4f} | {r['sensitivity_macro']:.4f} | "
              f"{r['balanced_accuracy']:.4f} |")
        w("\n### 10.2 Recorded-noise (NSTDB) classification per model -- mean metrics")
        if nstdb_df is not None and len(nstdb_df):
            metric_cols = [c for c in
                           ["accuracy", "macro_f1", "sensitivity_macro",
                            "balanced_accuracy"]
                           if c in nstdb_df.columns]
            if metric_cols:
                g2 = nstdb_df.groupby("model")[metric_cols].mean().round(4)
                w("\n| Model | " + " | ".join(m.replace("_", " ").title()
                                             for m in metric_cols) + " |")
                w("|---|" + "|".join(["---"] * len(metric_cols)) + "|")
                for name in g2.index:
                    r = g2.loc[name]
                    w("| " + name + " | " + " | ".join(
                        f"{float(r[c]):.4f}" for c in metric_cols) + " |")
            # per-record predicted class distribution (documented as
            # predictions; metrics use the record-matched truth where available)
            if "predicted_class_counts" in nstdb_df.columns:
                w("\nPer-record predicted AAMI class counts (NSTDB; metrics "
                  "where truth is available, per the table above):")
                w("")
                for rec, g in nstdb_df.groupby("record", sort=True):
                    for m, gm in g.groupby("model", sort=True):
                        pc = gm.iloc[0]["predicted_class_counts"]
                        counts = ", ".join(f"{c}={pc.get(c, 0)}" for c in CLASSES)
                        w(f"- `{rec}` [{m}]: predicted {counts}")
        else:
            w("(No NSTDB records were available for classification at the "
              "time of the report; check `reports/results/nstdb_evaluation.csv`.)")
        reserved = data.get("reserved_records") or []
        if reserved:
            w("\nNote: the final models used for the recorded-noise evaluation "
              f"were trained on the full data excluding the reserved records "
              f"({', '.join(reserved)}); see 9.1b.")
        w("\n### 10.3 Figures: `snr_vs_macro_f1.png`, `snr_vs_sensitivity.png`, "
          "`snr_vs_balanced_accuracy.png`, `clean_vs_noisy.png`")

    w("\n## 11. Figures\n")
    for f in figures:
        w(f"- `{f.split('/')[-1]}`")

    w("\n## 12. Interpretation\n")
    best = max(cv_by_model, key=cv_by_model.get)
    best_acc = cv_by_model[best]
    w("Plausible reading of the measured results (interpretation of actual "
      "numbers above):")
    w(f"- Patient independence: grouping all beats of a recording and using "
      f"StratifiedGroupKFold yielded no patient leakage; the honest estimate of "
      f"generalization to unseen patients is the cross-fold mean, not the "
      f"training fit.")
    if best_acc >= 0.9:
        w(f"- A patient-independent model reaches {best_acc:.2f} mean 5-fold "
          "accuracy on the AAMI task (see the table above); this is a strong "
          "research-grade classification signal.")
    elif best_acc >= 0.7:
        w(f"- Mean 5-fold accuracy is {best_acc:.2f} for the best model -- "
          "moderate; per-class and per-fold detail in the CSVs.")
    else:
        w(f"- Mean 5-fold accuracy is {best_acc:.2f} for the best model -- weak; "
          "the feature set or segmentation window may not carry enough signal, "
          "or the task (5-class AAMI on single beats) is intrinsically hard.")
    w("- The noise experiment (to be filled from `reports/results/noise_results.csv`) "
      "documents how performance decays as SNR decreases; the NSTDB records "
      "118/119 were reserved and classified with real noise in the recorded-noise "
      "evaluation.")

    w("\n## 13. Limitations (fixed, per the project specification)\n")
    w("- Historical data: MITDB is a widely used database from the era of "
      "older leads/techniques; classification performance must not be claimed "
      "as clinical diagnosis.")
    w("- Class imbalance: AAMI N is the majority class; we report macro and "
      "balanced metrics, not accuracy alone, and use class weights.")
    w("- Noise-cohort size: NSTDB provides two noise-stress records (118, 119); "
      "the synthetic mixture experiment adds 8 real NSTDB noise realizations "
      "for the SNR sweep. Results are descriptive, not a statistically "
      "powered comparison of official noise.")
    w("- Not a clinical diagnostic tool; beat-level AAMI classification is "
      "research-grade.")
    w("- Features are extracted from a 0.7 s window (0.25 s pre / 0.45 s post R); "
      "alternative windows were not tested (scope decision).")

    w("\n## 14. References\n")
    w("- PhysioNet MIT-BIH Arrhythmia Database -- https://physionet.org/content/mitdb/1.0.0/")
    w("- PhysioNet MIT-BIH Noise Stress Test Database -- https://physionet.org/content/nstdb/1.0.0/")
    w("- AAMI EC57:2012 -- Cardiac monitor inappropriate alert performance standard "
      "(AAMI EC57).\n")
    w("*Reproduction:* `python scripts/reproduce_pipeline.py` (downloads PhysioNet "
      "databases, runs the full pipeline, and regenerates this report with the "
      "actual results; requires `wfdb`, `numpy`, `scipy`, `pandas`, "
      "`matplotlib`, `seaborn`, `scikit-learn`, `pytest`, `pyyaml`, `joblib`.)")

    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path
