"""The ECG-ROBUST pipeline — full research pipeline end to end.

Steps (per the project specification):
1. Download (``download_all``).
2. ECG processing + segmentation + AAMI classification + features.
3. Patient-independent CV (Dummy / RandomForest / SVM) — ``cv_df``.
4. Noise robustness (``noise_sweep`` + ``run_nstdb_evaluation``).
5. Visualization (``reports/figures/``).
6. Reports (``reports/results/*.csv`` + ``reports/TECHNICAL_REPORT.md``).

The pipeline is deterministic: seeds are fixed and set per stage.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .annotations import CLASSES, annotation_report, classify_annotations
from .dataset import (
    MITDB_DIR,
    NSTDB_DIR,
    database_records,
    load_ecg_record,
)
from .download import download_all
from .evaluation import evaluate
from .features import extract_features
from .models import build_models
from .noise import extract_nstdb_noise, run_nstdb_evaluation, run_noise_sweep
from .preprocessing import bandpass_filter
from .segmentation import extract_segments, rr_intervals, rr_ratios
from .splits import make_groups, patient_splits
from .visualization import save_all_figures

RESULTS_DIR = Path("reports/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Computational tractability: SVC (RBF) fit on the full ~112k-beat MITDB
# feature matrix is infeasible, so the CV folds, the final models and the
# noise-sweep prediction set use a seeded per-class sample of beats
# (≤ MAX_BEATS_PER_CLASS per AAMI class). The full feature matrix is kept in
# data/features/ecg_features.csv; the sample is fully reproducible (seed 42).
MAX_BEATS_PER_CLASS = 1000


def _sampled_beat_positions(
    labels: list[str], max_per_class: int, seed: int
) -> list[int]:
    """Deterministic per-class beat positions used for CV and final models."""
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    for cls in CLASSES:
        sel = [i for i, c in enumerate(labels) if c == cls]
        if len(sel) > max_per_class:
            sel = rng.choice(
                np.asarray(sel, dtype=int), size=max_per_class, replace=False
            ).tolist()
        chosen.extend(int(i) for i in sel)
    return sorted(chosen)


def _record_signal_and_fs(rec) -> tuple[np.ndarray, int]:
    if hasattr(rec, "samples"):
        sig = np.asarray(rec.samples)
    else:
        sig = rec.to_dataframe().to_numpy()
    return sig, int(rec.fs)


def _symbol_list(ann) -> list[str]:
    return [str(s) for s in np.asarray(ann.symbol)]


def main() -> int:
    t_start = time.time()
    rng_seed = 42

    print("[1/9] Downloading PhysioNet databases (MITDB + NSTDB)...")
    download_all()
    records = database_records(MITDB_DIR)
    nstdb_records = database_records(NSTDB_DIR)
    print(f"      MITDB records: {len(records)}; NSTDB records: {len(nstdb_records)}")

    print("[2/9] ECG processing, segmentation, AAMI classification, features...")
    feature_rows: list[dict[str, Any]] = []
    beat_segments: list[np.ndarray] = []
    beat_labels: list[str] = []
    beat_keys: list[tuple[str, int]] = []
    signal_by_record: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    r_positions_by_record: dict[str, list[int]] = {}
    ann_reports: dict[str, dict[str, Any]] = {}

    for rec_id in records:
        rec, ann = load_ecg_record(rec_id, MITDB_DIR)
        sig, fs = _record_signal_and_fs(rec)
        sig_ch0 = sig[:, 0] if sig.ndim > 1 else sig
        sig_f = bandpass_filter(sig_ch0, fs)
        signal_by_record[rec_id] = (sig_ch0, sig_f)
        r_pos = [int(s) for s in np.asarray(ann.sample)]
        codes = _symbol_list(ann)
        r_positions_by_record[rec_id] = r_pos
        segments = extract_segments(sig_f, fs, r_pos)
        mapped, excluded, unknown = classify_annotations(codes)
        ann_reports[rec_id] = annotation_report(mapped, excluded, unknown)
        rrs = rr_intervals(r_pos, fs)
        rr_rat = rr_ratios(rrs)
        for i, seg in enumerate(segments):
            rr_i = float(rrs[i]) if i < len(rrs) else None
            rr_r = float(rr_rat[i]) if i < len(rr_rat) else None
            cls = mapped[i] if i < len(mapped) else None
            feature_rows.append(
                {
                    "record": rec_id,
                    "beat_idx": i,
                    "class": cls,
                    "r_sample": int(r_pos[i]) if i < len(r_pos) else -1,
                    **extract_features(seg, fs, rr_i, rr_r),
                }
            )
            if cls is not None:
                beat_segments.append(np.asarray(seg, dtype=np.float32))
                beat_labels.append(cls)
                beat_keys.append((rec_id, i))

    (RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    Path("data/features").mkdir(parents=True, exist_ok=True)
    feats = pd.DataFrame(feature_rows)
    feats.to_csv(Path("data/features/ecg_features.csv"), index=False)

    print(f"      total beats segmented: {len(feature_rows)}; "
          f"AAMI-mapped beats: {len(beat_labels)}")
    total_excluded = sum(int(v) for r in ann_reports.values()
                         for v in r["excluded"].values())
    total_unknown = sum(int(v) for r in ann_reports.values()
                        for v in r["unknown"].values())
    print(f"      excluded (known MITDB, not in AAMI map): {total_excluded}; "
          f"unknown codes: {total_unknown}")
    counts = pd.Series(beat_labels).value_counts().to_dict()
    print(f"      AAMI class counts: {counts}")

    chosen = _sampled_beat_positions(beat_labels, MAX_BEATS_PER_CLASS, seed=rng_seed)
    feat_subset = feats[feats["class"].notna()].reset_index(drop=True)
    X_all = feat_subset.iloc[chosen][
        [c for c in feats.columns if c not in ("record", "beat_idx", "class", "r_sample")]
    ].reset_index(drop=True)
    fill_vals = X_all.median(numeric_only=True)
    X_all = X_all.fillna(fill_vals)
    y_all = np.array([feat_subset["class"].iloc[i] for i in chosen])
    groups_all = make_groups([feat_subset["record"].iloc[i] for i in chosen])
    beat_segments = [beat_segments[i] for i in chosen]
    beat_labels = [beat_labels[i] for i in chosen]
    n_beats = int(len(y_all))
    print(f"      CV + final-model sample: {n_beats} beats "
          f"(max {MAX_BEATS_PER_CLASS} per class)")

    print("[3/9] Patient-independent cross-validation (5-fold StratifiedGroupKFold)...")
    n_splits = 5
    folds = list(patient_splits(y_all, groups_all, n_splits=n_splits, random_state=rng_seed))
    models = build_models(random_state=rng_seed)
    cv_rows: list[dict[str, Any]] = []
    conf_sum: dict[str, np.ndarray] = {name: np.zeros((5, 5)) for name in models}
    for name, model in models.items():
        for fold_idx, (tr, va) in enumerate(folds):
            pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
            pipe.fit(X_all.iloc[tr], y_all[tr])
            pred = np.asarray(pipe.predict(X_all.iloc[va]))
            m = evaluate(y_all[va], pred, CLASSES)
            conf_sum[name][: len(CLASSES), : len(CLASSES)] += np.array(m["confusion_matrix"])
            cv_rows.append(
                {
                    "model": name,
                    "fold": fold_idx,
                    "n_train": int(len(tr)),
                    "n_val": int(len(va)),
                    "accuracy": m["accuracy"],
                    "precision_macro": m["precision_macro"],
                    "recall_macro": m["recall_macro"],
                    "sensitivity_macro": m["sensitivity_macro"],
                    "specificity_macro": m["specificity_macro"],
                    "f1_macro": m["f1_macro"],
                    "f1_weighted": m["f1_weighted"],
                    "balanced_accuracy": m["balanced_accuracy"],
                }
            )
    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(RESULTS_DIR / "cv_metrics.csv", index=False)
    print("      per-model mean CV metrics:")
    print(cv_df.groupby("model")[["accuracy", "f1_macro", "sensitivity_macro", "balanced_accuracy"]]
          .mean().round(4).to_string())

    # final models fitted on all mapped beats (used for the noise experiment)
    models_final = build_models(random_state=rng_seed)
    per_class_eval: dict[str, Any] = {}
    for name, model in models_final.items():
        pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
        pipe.fit(X_all, y_all)
        pred = np.asarray(pipe.predict(X_all))
        m = evaluate(y_all, pred, CLASSES)
        conf_sum[name][: len(CLASSES), : len(CLASSES)] = np.array(m["confusion_matrix"])
        per_class_eval[name] = m["per_class"]
    (RESULTS_DIR / "per_class_evaluation.json").write_text(
        json.dumps(per_class_eval, indent=2), encoding="utf-8")

    print("[4/9] Noise robustness experiment (MIT-BIH NSTDB)...")
    fs = 360
    noise_realizations = []
    for rec_id in nstdb_records[:8]:
        try:
            noise_realizations.append(extract_nstdb_noise(rec_id, fs))
            print(f"      NSTDB noise realization: {rec_id} "
                  f"({len(noise_realizations[-1])} samples)")
        except Exception as e:  # noqa: BLE001
            print(f"      warning: NSTDB record {rec_id} skipped: {e}")
    noise_df = run_noise_sweep(
        models_final, beat_segments, beat_labels, noise_realizations, fs=fs,
        feature_columns=list(X_all.columns),
        feature_defaults={k: float(v) for k, v in fill_vals.to_dict().items()},
    )
    noise_df.to_csv(RESULTS_DIR / "noise_results.csv", index=False)
    print(noise_df.groupby("model")[["accuracy", "macro_f1", "sensitivity_macro",
                                     "balanced_accuracy"]].mean().round(4).to_string())

    nstdb_df, nstdb_details = run_nstdb_evaluation(
        models_final, nstdb_records, fs=fs,
        feature_columns=list(X_all.columns),
        feature_defaults={k: float(v) for k, v in fill_vals.to_dict().items()},
    )
    nstdb_df.to_csv(RESULTS_DIR / "nstdb_evaluation.csv", index=False)
    (RESULTS_DIR / "nstdb_details.json").write_text(
        json.dumps(nstdb_details, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o),
                   indent=2), encoding="utf-8")

    print("[5/9] Visualization...")
    segs_by_class: dict[str, list[np.ndarray]] = {c: [] for c in CLASSES}
    for seg, cls in zip(beat_segments, beat_labels):
        segs_by_class[cls].append(seg)
    r_all = np.array([r for rec in records for r in r_positions_by_record[rec]])
    clean_seg = beat_segments[0] if beat_segments else None
    noisy_seg = None
    if clean_seg is not None and noise_realizations:
        from .noise import add_noise
        noisy_seg, _ach = add_noise(clean_seg, np.resize(noise_realizations[0], len(clean_seg)), 12.0)

    metrics_df = (cv_df.groupby("model")[
        ["accuracy", "precision_macro", "recall_macro", "sensitivity_macro",
         "specificity_macro", "f1_macro", "f1_weighted", "balanced_accuracy"]
    ].mean().reset_index())
    figs = save_all_figures(
        signal_by_record[records[0]][0] if signal_by_record else np.array([]),
        signal_by_record[records[0]][1] if signal_by_record else np.array([]),
        r_all,
        segs_by_class,
        counts,
        metrics_df,
        conf_sum,
        CLASSES,
        clean_seg=clean_seg,
        noisy_seg=noisy_seg,
        noise_df=noise_df,
    )

    print("[6/9] Reports: dataset summary + technical report...")
    from .technical_report import write_technical_report

    dataset_summary = {
        "mitdb": {
            "records": records,
            "n_records": len(records),
            "total_beats_annotated": int(len(feature_rows)),
            "total_beats_aami": n_beats,
            "class_counts": counts,
            "sampling_frequency_hz": fs,
            "nstdb": nstdb_records,
            "excluded_annotations": total_excluded,
            "unknown_annotations": total_unknown,
            "excluded_by_record_sample": {
                k: {kk: vv for kk, vv in v.items() if kk in ("excluded", "unknown")}
                for k, v in list(ann_reports.items())[:5]
            },
        }
    }
    (RESULTS_DIR / "dataset_summary.json").write_text(
        json.dumps(dataset_summary, indent=2), encoding="utf-8"
    )
    (RESULTS_DIR / "confusion_matrices.json").write_text(
        json.dumps({k: np.asarray(v).tolist() for k, v in conf_sum.items()},
                   indent=2),
        encoding="utf-8",
    )
    write_technical_report(
        {
            "dataset": dataset_summary["mitdb"],
            "cv_df": cv_df,
            "metrics_df": metrics_df,
            "noise_df": noise_df,
            "nstdb_df": nstdb_df,
            "n_features": int(len(X_all.columns)),
            "figures": figs,
            "wall_s": time.time() - t_start,
            "confusion": {k: v.tolist() for k, v in conf_sum.items()},
            "confusion_labels": CLASSES,
            "per_class": per_class_eval,
        },
        Path("reports/TECHNICAL_REPORT.md"),
    )

    print(f"[9/9] Done in {time.time() - t_start:.1f}s.")
    print(f"      Reports: {RESULTS_DIR} (cv_metrics, noise_results, nstdb_evaluation, dataset_summary)")
    print(f"      Figures: {len(figs)} saved to reports/figures/")
    print(f"      Technical report: reports/TECHNICAL_REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
