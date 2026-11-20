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
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .annotations import (
    CLASSES,
    annotate_annotations,
    annotation_report,
    classify_annotations,
)
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
from .noise import (
    SNRS_DB,
    extract_nstdb_noise,
    noisy_heldout_metrics,
    run_nstdb_evaluation,
)
from .preprocessing import bandpass_filter
from .segmentation import extract_beats, rr_intervals, rr_ratios
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


def _new_pipeline(model) -> Pipeline:
    """Scaler + median imputer fitted per fold inside the sklearn pipeline.

    Both are fit on the fold's training data only (spec #4); the model is
    fit on the same fitted pipeline, so scaling/imputation and fitting stay
    inside one pipeline.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model),
    ])


def reserved_records_for_nstdb(records, nstdb_records) -> set[str]:
    """MITDB records corresponding to NSTDB-recorded evaluations.

    Per the specification, records 118 and 119 — and their noise-stress
    versions present in the NSTDB (e.g. ``118e00``, ``119e00``) — are
    reserved: the corresponding clean patient records are excluded from
    final-model training for the recorded-noise evaluation (spec #3).
    """
    nstdb_set = {str(r) for r in nstdb_records}
    reserved = set()
    for r in records:
        r = str(r)
        if r in nstdb_set or any(n.startswith(r) for n in nstdb_set):
            reserved.add(r)
    return reserved


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
    excluded_counter: dict[str, int] = {}
    unknown_counter: dict[str, int] = {}

    fs = 360  # MIT-BIH sampling frequency (Hz)
    for rec_id in records:
        rec, ann = load_ecg_record(rec_id, MITDB_DIR)
        sig, fs = _record_signal_and_fs(rec)
        sig_ch0 = sig[:, 0] if sig.ndim > 1 else sig
        sig_f = bandpass_filter(sig_ch0, fs)
        signal_by_record[rec_id] = (sig_ch0, sig_f)
        r_pos = [int(s) for s in np.asarray(ann.sample)]
        codes = _symbol_list(ann)
        r_positions_by_record[rec_id] = r_pos
        annot_recs = annotate_annotations(codes, samples=r_pos)
        mapped, excluded, unknown = classify_annotations(codes)
        ann_reports[rec_id] = annotation_report(mapped, excluded, unknown)
        for k, v in excluded.items():
            excluded_counter[k] = excluded_counter.get(k, 0) + v
        for k, v in unknown.items():
            unknown_counter[k] = unknown_counter.get(k, 0) + v
        rrs = rr_intervals(r_pos, fs)
        rr_rat = rr_ratios(rrs)
        # Spec #1: segments are aligned to annotation records (index/sample/
        # symbol/class travel together); excluded/unknown annotations are
        # represented explicitly and never shift labels.
        beats = extract_beats(sig_f, fs, annot_recs)
        for beat in beats:
            rr_i = float(rrs[beat.index]) if beat.index < len(rrs) else None
            rr_r = float(rr_rat[beat.index]) if beat.index < len(rr_rat) else None
            feature_rows.append(
                {
                    "record": rec_id,
                    "beat_idx": beat.index,
                    "class": beat.aami_class,
                    "r_sample": beat.r_sample,
                    **extract_features(beat.segment, fs, rr_i, rr_r),
                }
            )
            if beat.aami_class is not None:
                beat_segments.append(np.asarray(beat.segment, dtype=np.float32))
                beat_labels.append(beat.aami_class)
                beat_keys.append((rec_id, beat.index))

    (RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    Path("data/features").mkdir(parents=True, exist_ok=True)
    feats = pd.DataFrame(feature_rows)
    feats.to_csv(Path("data/features/ecg_features.csv"), index=False)

    print(f"      total beats segmented: {len(feature_rows)}; "
          f"AAMI-mapped beats: {len(beat_labels)}")
    total_excluded = sum(excluded_counter.values())
    total_unknown = sum(unknown_counter.values())
    print(f"      excluded (known MITDB, not in AAMI map): {total_excluded}; "
          f"unknown codes: {total_unknown}")
    counts = pd.Series(beat_labels).value_counts().to_dict()
    print(f"      AAMI class counts: {counts}")

    chosen = _sampled_beat_positions(beat_labels, MAX_BEATS_PER_CLASS, seed=rng_seed)
    feat_subset = feats[feats["class"].notna()].reset_index(drop=True)
    X_all = feat_subset.iloc[chosen][
        [c for c in feats.columns if c not in ("record", "beat_idx", "class", "r_sample")]
    ].reset_index(drop=True)
    y_all = np.array([feat_subset["class"].iloc[i] for i in chosen])
    groups_all = make_groups([feat_subset["record"].iloc[i] for i in chosen])
    beat_segments = [beat_segments[i] for i in chosen]
    beat_labels = [beat_labels[i] for i in chosen]
    beat_keys = [beat_keys[i] for i in chosen]
    n_beats = int(len(y_all))
    print(f"      CV + final-model sample: {n_beats} beats "
          f"(max {MAX_BEATS_PER_CLASS} per class)")

    print("[3/9] Patient-independent cross-validation (5-fold StratifiedGroupKFold)...")
    noise_realizations = []
    for rec_id in nstdb_records[:8]:
        try:
            noise_realizations.append(extract_nstdb_noise(rec_id, fs))
            print(f"      NSTDB noise realization: {rec_id} "
                  f"({len(noise_realizations[-1])} samples)")
        except Exception as e:  # noqa: BLE001
            print(f"      warning: NSTDB record {rec_id} skipped: {e}")
    n_splits = 5
    folds = list(patient_splits(y_all, groups_all, n_splits=n_splits, random_state=rng_seed))
    models = build_models(random_state=rng_seed)
    cv_rows: list[dict[str, Any]] = []
    conf_oof: dict[str, np.ndarray] = {name: np.zeros((len(CLASSES), len(CLASSES))) for name in models}
    oof_noise_rows: list[dict[str, Any]] = []
    snrs = SNRS_DB
    noise_rng = np.random.default_rng(rng_seed + 1)
    for name, model in models.items():
        for fold_idx, (tr, va) in enumerate(folds):
            pipe = _new_pipeline(model)
            pipe.fit(X_all.iloc[tr], y_all[tr])
            pred = np.asarray(pipe.predict(X_all.iloc[va]))
            m = evaluate(y_all[va], pred, CLASSES)
            conf_oof[name][: len(CLASSES), : len(CLASSES)] += np.array(m["confusion_matrix"])
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
            # Out-of-fold noise evaluation (spec #3): the same out-of-fold
            # model scores noisy versions of its held-out validation beats.
            df_n = noisy_heldout_metrics(
                pipe,
                y_all[va],
                [beat_segments[v] for v in va],
                noise_realizations,
                snrs,
                fs,
                feature_columns=list(X_all.columns),
                seed=int(noise_rng.integers(0, 2**31 - 1)),
            )
            for _, r in df_n.iterrows():
                oof_noise_rows.append(
                    {
                        "snr_db": float(r["snr_db"]),
                        "model": name,
                        "n_beats": int(r["n_beats"]),
                        "accuracy": float(r["accuracy"]),
                        "macro_f1": float(r["macro_f1"]),
                        "sensitivity_macro": float(r["sensitivity_macro"]),
                        "balanced_accuracy": float(r["balanced_accuracy"]),
                    }
                )
    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(RESULTS_DIR / "cv_metrics.csv", index=False)
    cv_means = cv_df.groupby("model")["accuracy"].mean().round(4)
    best_model = cv_means.idxmax()
    print("      per-model mean CV metrics:")
    print(cv_df.groupby("model")[["accuracy", "f1_macro", "sensitivity_macro", "balanced_accuracy"]]
          .mean().round(4).to_string())

    print("[4/9] Noise robustness experiment (MIT-BIH NSTDB; out-of-fold noise evaluation)...")
    noise_df = pd.DataFrame(oof_noise_rows)
    noise_df.to_csv(RESULTS_DIR / "noise_results.csv", index=False)
    print(noise_df.groupby("model")[["accuracy", "macro_f1", "sensitivity_macro",
                                     "balanced_accuracy"]].mean().round(4).to_string())

    # Final models: train on all non-reserved beats (spec #3 — reserved clean
    # records corresponding to the NSTDB evaluation are excluded).
    reserved = reserved_records_for_nstdb(records, nstdb_records)
    final_mask = np.array([key[0] not in reserved for key in beat_keys], dtype=bool)

    # Final models (used for the recorded-noise evaluation); trained with the
    # same fitted-pipeline structure as CV (imputer + scaler + model).
    models_final: dict[str, Pipeline] = {}
    per_class_eval: dict[str, Any] = {}
    conf_train: dict[str, np.ndarray] = {}
    for name, model in build_models(random_state=rng_seed).items():
        pipe = _new_pipeline(model)
        pipe.fit(X_all[final_mask], y_all[final_mask])
        pred = np.asarray(pipe.predict(X_all[final_mask]))
        m = evaluate(y_all[final_mask], pred, CLASSES)
        conf_train[name] = np.array(m["confusion_matrix"])
        per_class_eval[name] = m["per_class"]
        models_final[name] = pipe
    (RESULTS_DIR / "per_class_evaluation.json").write_text(
        json.dumps(per_class_eval, indent=2), encoding="utf-8")

    nstdb_df, nstdb_details = run_nstdb_evaluation(
        models_final, nstdb_records, fs=fs, feature_columns=list(X_all.columns),
    )
    nstdb_df.to_csv(RESULTS_DIR / "nstdb_evaluation.csv", index=False)
    (RESULTS_DIR / "nstdb_details.json").write_text(
        json.dumps(nstdb_details, indent=2), encoding="utf-8")

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
        conf_oof,
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
        json.dumps({
            "oof": {k: np.asarray(v).tolist() for k, v in conf_oof.items()},
            "train": {k: np.asarray(v).tolist() for k, v in conf_train.items()},
        }, indent=2),
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
            "confusion": {k: v.tolist() for k, v in conf_oof.items()},
            "confusion_train": {k: v.tolist() for k, v in conf_train.items()},
            "best_model": best_model,
            "reserved_records": sorted(reserved),
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
