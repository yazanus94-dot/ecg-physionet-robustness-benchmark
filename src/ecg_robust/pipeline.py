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
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .annotations import (
    CLASSES,
    annotate_annotations,
    annotation_report,
    classify_annotations,
)
from .config import config_path_from_env, load_config
from .dataset import (
    MITDB_DIR,
    NSTDB_DIR,
    _read_local,
    database_records,
    load_ecg_record,
)
from .download import download_all
from .evaluation import evaluate, per_class_metrics_from_cm
from .features import extract_features
from .models import build_models
from .noise import (
    SNRS_DB,
    extract_nstdb_noise,
    generate_noisy_features,
    run_nstdb_evaluation,
)
from .preprocessing import bandpass_filter
from .segmentation import (
    AFTER_S,
    BEFORE_S,
    MIN_LENGTH,
    extract_beats,
    rr_features_for_valid_beats,
)
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


def load_record_safely(rec_id: str, directory: Path) -> tuple[Any | None, str | None]:
    """Load a recording, returning ``(None, reason)`` for incomplete downloads.

    ``database_records`` discovers IDs from ``*.dat`` alone, so a truncated
    record without a matching ``.hea``/``.atr`` would otherwise abort the run.
    """
    try:
        rec, ann = load_ecg_record(rec_id, directory)
        _sig, _fs = _record_signal_and_fs(rec)
        if len(np.asarray(ann.sample)) != len(_symbol_list(ann)):
            raise ValueError("annotation positions/symbols misaligned")
        return (rec, ann), None
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        return None, str(exc)


def _new_pipeline(model) -> Pipeline:
    """Scaler + median imputer fitted per fold inside the sklearn pipeline.

    Both are fit on the fold's training data only (spec #4); the model is
    fit on the same fitted pipeline, so scaling/imputation and fitting stay
    inside one pipeline.
    """
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


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


def load_experiment(
    features_csv: str = "data/features/ecg_features.csv",
    n_splits: int = 5,
    random_state: int = 42,
):
    """Reproduce the pipeline's sampled design from the saved features CSV.

    Returns ``(X_all, y_all, groups_all, folds)`` — the same arrays/folds
    ``main()`` uses (same sampling seed/constants, patient-independent
    StratifiedGroupKFold). Used by the ablation and hyperparameter-search
    scripts (spec Phase 6).
    """
    df = pd.read_csv(features_csv)
    feat_subset = df[df["class"].notna()].reset_index(drop=True)
    chosen = _sampled_beat_positions(
        list(feat_subset["class"]), MAX_BEATS_PER_CLASS, seed=random_state
    )
    X_all = feat_subset.iloc[chosen][
        [c for c in df.columns if c not in ("record", "beat_idx", "class", "r_sample")]
    ].reset_index(drop=True)
    y_all = np.array([feat_subset["class"].iloc[i] for i in chosen])
    groups_all = make_groups([feat_subset["record"].iloc[i] for i in chosen])
    folds = list(
        patient_splits(y_all, groups_all, n_splits=n_splits, random_state=random_state)
    )
    return X_all, y_all, groups_all, folds


def main() -> int:
    t_start = time.time()
    config = load_config(config_path_from_env())
    rng_seed = config["random_state"]
    n_splits = config["n_splits"]
    print(f"      configuration: n_splits={n_splits}, seed={rng_seed}")

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

    fs: int | None = None  # sampling frequency (verified per record below)
    fs_values: set[int] = set()
    skipped_records: list[str] = []
    for rec_id in records:
        loaded, reason = load_record_safely(rec_id, MITDB_DIR)
        if loaded is None:
            # Report the skip explicitly, never fabricate data for the record.
            skipped_records.append(rec_id)
            print(f"    WARNING: skipping incomplete record {rec_id}: {reason}")
            continue
        rec, ann = loaded
        sig, rec_fs = _record_signal_and_fs(rec)
        fs = rec_fs
        fs_values.add(fs)
        sig_ch0 = sig[:, 0] if sig.ndim > 1 else sig
        sig_f = bandpass_filter(sig_ch0, fs)
        signal_by_record[rec_id] = (sig_ch0, sig_f)
        r_pos = [int(s) for s in np.asarray(ann.sample)]
        codes = _symbol_list(ann)
        r_positions_by_record[rec_id] = r_pos
        annot_recs = annotate_annotations(
            codes, samples=r_pos, signal_length=len(sig_f)
        )
        mapped, excluded, unknown = classify_annotations(codes)
        ann_reports[rec_id] = annotation_report(mapped, excluded, unknown)
        for k, v in excluded.items():
            excluded_counter[k] = excluded_counter.get(k, 0) + v
        for k, v in unknown.items():
            unknown_counter[k] = unknown_counter.get(k, 0) + v
        # Spec #1: segments are aligned to annotation records (index/sample/
        # symbol/class travel together); excluded/unknown annotations are
        # represented explicitly and never shift labels.
        beats = extract_beats(sig_f, fs, annot_recs)
        # Phase 6: verify the segmentation window/sample count invariant on
        # every retained beat (segments are truncated at boundaries, so they
        # may be shorter than the full window but never longer).
        seg_max = round(BEFORE_S * fs) + round(AFTER_S * fs)
        for beat in beats:
            assert MIN_LENGTH <= len(beat.segment) <= seg_max, (
                f"record {rec_id}: segment length out of window "
                f"[{MIN_LENGTH}, {seg_max}]: {len(beat.segment)}"
            )
        valid_beats = [b for b in beats if b.aami_class is not None]
        rr_seq = rr_features_for_valid_beats(valid_beats, fs)
        for beat, (rr_i, rr_r) in zip(valid_beats, rr_seq):
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

    if fs_values:
        if len(fs_values) != 1:
            raise RuntimeError(
                "inconsistent sampling frequencies across MITDB records: "
                f"{sorted(fs_values)}"
            )
        fs = min(fs_values)
    else:
        raise RuntimeError("no MITDB records found in the MITDB directory")
    print(f"      sampling frequency (verified from actual records): {fs} Hz")

    (RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    Path("data/features").mkdir(parents=True, exist_ok=True)
    feats = pd.DataFrame(feature_rows)
    feats.to_csv(Path("data/features/ecg_features.csv"), index=False)

    print(
        f"      total beats segmented: {len(feature_rows)}; "
        f"AAMI-mapped beats: {len(beat_labels)}"
    )
    total_excluded = sum(excluded_counter.values())
    total_unknown = sum(unknown_counter.values())
    print(
        f"      excluded (known MITDB, not in AAMI map): {total_excluded}; "
        f"unknown codes: {total_unknown}"
    )
    counts: dict[str, int] = {
        str(k): int(v) for k, v in pd.Series(beat_labels).value_counts().items()
    }
    print(f"      AAMI class counts: {counts}")

    chosen = _sampled_beat_positions(beat_labels, MAX_BEATS_PER_CLASS, seed=rng_seed)
    feat_subset = feats[feats["class"].notna()].reset_index(drop=True)
    X_all = feat_subset.iloc[chosen][
        [
            c
            for c in feats.columns
            if c not in ("record", "beat_idx", "class", "r_sample")
        ]
    ].reset_index(drop=True)
    y_all = np.array([feat_subset["class"].iloc[i] for i in chosen])
    groups_all = make_groups([feat_subset["record"].iloc[i] for i in chosen])
    beat_segments = [beat_segments[i] for i in chosen]
    beat_labels = [beat_labels[i] for i in chosen]
    beat_keys = [beat_keys[i] for i in chosen]
    n_beats = len(y_all)
    print(
        f"      CV + final-model sample: {n_beats} beats "
        f"(max {MAX_BEATS_PER_CLASS} per class)"
    )

    print("[3/9] Patient-independent cross-validation (5-fold StratifiedGroupKFold)...")
    noise_realizations = []
    realization_record_ids: list[str] = []
    nstdb_fs_values: set[int] = set()
    for rec_id in nstdb_records[:8]:
        try:
            # Phase 3/6: the noise-realization fs is read from the NSTDB
            # record itself (verified inside `extract_nstdb_noise`).
            nstdb_fs = int(_read_local(rec_id, NSTDB_DIR, "rdsamp").fs)
            noise_realizations.append(extract_nstdb_noise(rec_id, nstdb_fs, NSTDB_DIR))
            realization_record_ids.append(rec_id)
            nstdb_fs_values.add(nstdb_fs)
            print(
                f"      NSTDB noise realization: {rec_id} "
                f"({len(noise_realizations[-1])} samples, fs={nstdb_fs} Hz)"
            )
        except Exception as e:  # noqa: BLE001
            print(f"      warning: NSTDB record {rec_id} skipped: {e}")
    if nstdb_fs_values:
        print(f"      NSTDB sampling frequencies (actual): {sorted(nstdb_fs_values)}")
    folds = list(
        patient_splits(y_all, groups_all, n_splits=n_splits, random_state=rng_seed)
    )
    # Phase 6: patient-independent validation — no group (recording; 201/202
    # share a group) may appear in both a training and a validation fold.
    for tr, va in folds:
        assert not set(groups_all[tr]) & set(groups_all[va]), (
            "train/validation groups overlap — patient leakage!"
        )
    models = build_models(random_state=rng_seed)
    cv_rows: list[dict[str, Any]] = []
    conf_oof: dict[str, np.ndarray] = {
        name: np.zeros((len(CLASSES), len(CLASSES))) for name in models
    }
    oof_per_class: dict[str, Any] = {}
    oof_noise_rows: list[dict[str, Any]] = []
    fitted_pipes: dict[int, dict[str, Any]] = {}
    fold_class_counts: dict[int, dict[str, Any]] = {}
    snrs = SNRS_DB
    noise_rng = np.random.default_rng(rng_seed + 1)
    print(
        f"      noise-realization rng seed: {rng_seed + 1} "
        "(one shared stream; per fold/SNR the noisy validation matrix is "
        "generated once by generate_noisy_features and shared by every model)"
    )
    for name, model in models.items():
        for fold_idx, (tr, va) in enumerate(folds):
            pipe = _new_pipeline(model)
            pipe.fit(X_all.iloc[tr], y_all[tr])
            pred = np.asarray(pipe.predict(X_all.iloc[va]))
            m = evaluate(y_all[va], pred, CLASSES)
            conf_oof[name][: len(CLASSES), : len(CLASSES)] += np.array(
                m["confusion_matrix"]
            )
            fitted_pipes.setdefault(fold_idx, {})[name] = pipe
            # Phase 6: per-fold class distributions (validation + training).
            fold_class_counts[fold_idx] = {
                "val": {c: int((y_all[va] == c).sum()) for c in CLASSES},
                "train": {c: int((y_all[tr] == c).sum()) for c in CLASSES},
            }
            cv_rows.append(
                {
                    "model": name,
                    "fold": fold_idx,
                    "n_train": len(tr),
                    "n_val": len(va),
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
    # Phase 2: per-class metrics from out-of-fold predictions (summed OOF CM).
    oof_per_class = {
        name: per_class_metrics_from_cm(cm, CLASSES) for name, cm in conf_oof.items()
    }
    # Phase 6: report any missing classes in a fold (validation population).
    missing_fold_classes = {
        str(fold_idx): [c for c in CLASSES if counts.get("val", {}).get(c, 0) == 0]
        for fold_idx, counts in fold_class_counts.items()
        if any(counts.get("val", {}).get(c, 0) == 0 for c in CLASSES)
    }
    if missing_fold_classes:
        print(
            f"      WARNING: folds with missing validation classes: "
            f"{missing_fold_classes}"
        )
    else:
        print("      per-fold validation class counts: complete in all folds")

    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(RESULTS_DIR / "cv_metrics.csv", index=False)
    (RESULTS_DIR / "fold_class_counts.json").write_text(
        json.dumps(fold_class_counts, indent=2), encoding="utf-8"
    )
    cv_means = cv_df.groupby("model")["accuracy"].mean().round(4)
    best_model = cv_means.idxmax()
    print("      per-model mean CV metrics:")
    print(
        cv_df.groupby("model")[
            ["accuracy", "f1_macro", "sensitivity_macro", "balanced_accuracy"]
        ]
        .mean()
        .round(4)
        .to_string()
    )

    print(
        "[4/9] Noise robustness experiment (MIT-BIH NSTDB; out-of-fold noise evaluation)..."
    )
    # Phase 4: fair design (spec #4) — deterministic noise assignments per
    # fold/SNR; the SAME noisy validation samples are shared across models.
    # Out-of-fold: each model scores its own fold's held-out beats (no
    # final-model training here — training diagnostics only).
    for fold_idx, (tr, va) in enumerate(folds):
        val_clean = [beat_segments[v] for v in va]
        y_val = np.asarray(y_all[va])
        noisy_by_snr, achieved_by_snr = generate_noisy_features(
            val_clean,
            noise_realizations,
            snrs,
            fs,
            noise_rng,
            feature_columns=list(X_all.columns),
        )
        for snr in snrs:
            Xn = noisy_by_snr[snr]
            achieved = achieved_by_snr[snr]
            if np.isfinite(achieved) and abs(achieved - snr) > max(0.5, 0.1 * abs(snr)):
                print(
                    f"      warning: achieved SNR {achieved:.2f} != requested {snr:g}"
                )
            for name, pipe in fitted_pipes[fold_idx].items():
                pred = np.asarray(pipe.predict(Xn))
                _prec, rec, f1, _sup = precision_recall_fscore_support(
                    y_val, pred, labels=CLASSES, zero_division=0
                )
                oof_noise_rows.append(
                    {
                        "snr_db": float(snr),
                        "fold": fold_idx,
                        "model": name,
                        "n_beats": len(pred),
                        "achieved_snr_db": float(achieved)
                        if np.isfinite(achieved)
                        else np.nan,
                        "accuracy": float(accuracy_score(y_val, pred)),
                        "macro_f1": float(np.mean(f1)),
                        "sensitivity_macro": float(np.mean(rec)),
                        "balanced_accuracy": float(
                            balanced_accuracy_score(y_val, pred)
                        ),
                    }
                )
    noise_df = pd.DataFrame(oof_noise_rows)
    noise_df.to_csv(RESULTS_DIR / "noise_results.csv", index=False)
    # Phase 3: document the noise-realization seed and per-SNR achieved SNR.
    achieved_mean_by_snr: dict[str, float | None] = {}
    for s in snrs:
        sub = noise_df[noise_df["snr_db"] == s]["achieved_snr_db"]
        achieved_mean_by_snr[str(float(s))] = (
            float(np.nanmean(sub))
            if len(sub) > 0 and bool(np.isfinite(sub).any())
            else None
        )
    noise_config = {
        "noise_rng_seed": int(rng_seed + 1),
        "design": (
            "single shared noise-realization stream; per fold/SNR the noisy "
            "validation feature matrix is generated once by "
            "generate_noisy_features and shared by every model in the fold"
        ),
        "snrs_db": [float(s) for s in snrs],
        "n_noise_realizations": len(noise_realizations),
        "noise_record_ids": list(realization_record_ids),
        "noise_realization_fs_values": [int(f) for f in sorted(nstdb_fs_values)],
        "achieved_snr_db_by_snr": achieved_mean_by_snr,
    }
    (RESULTS_DIR / "noise_config.json").write_text(
        json.dumps(noise_config, indent=2), encoding="utf-8"
    )
    print(
        noise_df.groupby("model")[
            ["accuracy", "macro_f1", "sensitivity_macro", "balanced_accuracy"]
        ]
        .mean()
        .round(4)
        .to_string()
    )

    # Final models: train on all non-reserved beats (spec #3 — reserved clean
    # records corresponding to the NSTDB evaluation are excluded).
    reserved = reserved_records_for_nstdb(records, nstdb_records)
    # Phase 6: verify the reservation logic against the actual record names.
    assert reserved <= set(records), (
        f"reserved records not present in MITDB: {sorted(reserved - set(records))}"
    )
    final_mask = np.array([key[0] not in reserved for key in beat_keys], dtype=bool)
    # Phase 6: every beat from a reserved record must be excluded from the
    # final training set.
    assert not any(
        key[0] in reserved for key, keep in zip(beat_keys, final_mask) if keep
    ), "final training must exclude beats of reserved records"
    n_reserved_total = int(sum(key[0] in reserved for key in beat_keys))
    print(
        f"      reserved records excluded from final training: "
        f"{sorted(reserved)} ({n_reserved_total} beats of the modeling design)"
    )

    # Final models (used for the recorded-noise evaluation); trained with the
    # same fitted-pipeline structure as CV (imputer + scaler + model).
    models_final: dict[str, Pipeline] = {}
    per_class_train_diagnostics: dict[str, Any] = {}
    conf_train: dict[str, np.ndarray] = {}
    for name, model in build_models(random_state=rng_seed).items():
        pipe = _new_pipeline(model)
        pipe.fit(X_all[final_mask], y_all[final_mask])
        pred = np.asarray(pipe.predict(X_all[final_mask]))
        m = evaluate(y_all[final_mask], pred, CLASSES)
        conf_train[name] = np.array(m["confusion_matrix"])
        per_class_train_diagnostics[name] = m["per_class"]
        models_final[name] = pipe
    (RESULTS_DIR / "per_class_oof.json").write_text(
        json.dumps(oof_per_class, indent=2), encoding="utf-8"
    )
    (RESULTS_DIR / "per_class_train_diagnostics.json").write_text(
        json.dumps(per_class_train_diagnostics, indent=2), encoding="utf-8"
    )

    nstdb_df, nstdb_details = run_nstdb_evaluation(
        models_final,
        nstdb_records,
        fs=fs,
        feature_columns=list(X_all.columns),
    )
    nstdb_df.to_csv(RESULTS_DIR / "nstdb_evaluation.csv", index=False)
    (RESULTS_DIR / "nstdb_details.json").write_text(
        json.dumps(nstdb_details, indent=2), encoding="utf-8"
    )

    print("[5/9] Visualization...")
    segs_by_class: dict[str, list[np.ndarray]] = {c: [] for c in CLASSES}
    for seg, cls in zip(beat_segments, beat_labels):
        segs_by_class[cls].append(seg)
    r_all = np.array([r for rec in records for r in r_positions_by_record[rec]])
    clean_seg = beat_segments[0] if beat_segments else None
    noisy_seg = None
    if clean_seg is not None and noise_realizations:
        from .noise import add_noise

        noisy_seg, _ach = add_noise(
            clean_seg, np.resize(noise_realizations[0], len(clean_seg)), 12.0
        )

    metrics_df = (
        cv_df.groupby("model")[
            [
                "accuracy",
                "precision_macro",
                "recall_macro",
                "sensitivity_macro",
                "specificity_macro",
                "f1_macro",
                "f1_weighted",
                "balanced_accuracy",
            ]
        ]
        .mean()
        .reset_index()
    )
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
    # Phase 6: include the feature-ablation figure when the ablation script
    # has been run (it is generated by scripts/feature_ablation.py).
    if Path("reports/figures/feature_ablation.png").exists():
        figs.append("reports/figures/feature_ablation.png")

    print("[6/9] Reports: dataset summary + technical report...")
    from .technical_report import write_technical_report

    dataset_summary = {
        "mitdb": {
            "records": records,
            "n_records": len(records),
            "skipped_incomplete_records": skipped_records,
            "n_records_used": len(signal_by_record),
            "n_reserved_beats": n_reserved_total,
            "max_beats_per_class": MAX_BEATS_PER_CLASS,
            "total_beats_annotated": len(feature_rows),
            "total_beats_aami": n_beats,
            "total_beats_mapped_full": int(feat_subset.shape[0]),
            "class_counts": counts,
            "sampling_frequency_hz": int(fs),
            "sampling_frequencies_hz": [int(f) for f in sorted(fs_values)],
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
        json.dumps(
            {
                "oof": {k: np.asarray(v).tolist() for k, v in conf_oof.items()},
                "train": {k: np.asarray(v).tolist() for k, v in conf_train.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_technical_report(
        {
            "dataset": dataset_summary["mitdb"],
            "cv_df": cv_df,
            "metrics_df": metrics_df,
            "noise_df": noise_df,
            "nstdb_df": nstdb_df,
            "n_features": len(X_all.columns),
            "figures": figs,
            "wall_s": time.time() - t_start,
            "confusion": {k: v.tolist() for k, v in conf_oof.items()},
            "confusion_train": {k: v.tolist() for k, v in conf_train.items()},
            "best_model": best_model,
            "reserved_records": sorted(reserved),
            "confusion_labels": CLASSES,
            "per_class_oof": oof_per_class,
            "per_class_train_diagnostics": per_class_train_diagnostics,
            "max_beats_per_class": MAX_BEATS_PER_CLASS,
            "noise_rng_seed": int(rng_seed + 1),
        },
        Path("reports/TECHNICAL_REPORT.md"),
    )

    print(f"[9/9] Done in {time.time() - t_start:.1f}s.")
    print(
        f"      Reports: {RESULTS_DIR} (cv_metrics, noise_results, nstdb_evaluation, dataset_summary)"
    )
    print(f"      Figures: {len(figs)} saved to reports/figures/")
    print("      Technical report: reports/TECHNICAL_REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
