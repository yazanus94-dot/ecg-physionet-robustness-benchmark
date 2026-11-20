"""Noise robustness experiment (MIT-BIH Noise Stress Test Database).

Experiment design (per the project spec):

- Clean reference: MITDB beats (filtered, annotated) — "identical heartbeat
  positions wherever valid" is used when comparing noisy versions.
- Noisy evaluation:
  (a) Synthetic mixtures: clean beats + NSTDB noise realizations (the
      NSTDB signals) at the target SNR levels.
  (b) Recorded-noise evaluation: segmented NSTDB recordings (including the
      reserved noise-stress records 118 and 119) classified directly.
- SNR formula: ``SNR(dB) = 10 * log10(P_signal / P_noise)``.
- Plots: SNR vs macro F1 / sensitivity / balanced accuracy; clean vs noisy
  ECG.

Nothing synthetic is presented as an official recorded-noise experiment:
the NSTDB noise realizations are labeled as such, and the synthetic mixtures
are explicitly reported as synthetic.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
)

from .annotations import AAMI_MAP, CLASSES, AnnotationRecord
from .features import extract_features
from .preprocessing import bandpass_filter

SNRS_DB = [24.0, 18.0, 12.0, 6.0, 0.0, -6.0]


def _align_features(X: pd.DataFrame, feature_columns, feature_defaults=None):
    """Reindex ``X`` to the fitted ``feature_columns``.

    Missing columns become NaN — the fitted ``SimpleImputer`` in the
    pipeline fills them at ``predict`` time using training-fitted
    statistics (``feature_defaults``, when given, are applied eagerly too).
    """
    if feature_columns is None:
        return X
    X = X.reindex(columns=feature_columns)
    if feature_defaults:
        X = X.fillna(feature_defaults)
    return X


def estimate_snr_db(signal, noise) -> float:
    """``SNR(dB) = 10*log10(P_signal/P_noise)`` for two equal-length arrays."""
    ps = float(np.mean(np.asarray(signal, dtype=float) ** 2))
    pn = float(np.mean(np.asarray(noise, dtype=float) ** 2))
    if pn <= 0:
        return float("inf")
    return float(10.0 * math.log10(ps / pn))


def scale_noise(noise, snr_db: float, signal):
    """Scale ``noise`` so that SNR(signal, noise) == snr_db."""
    signal = np.asarray(signal, dtype=float)
    noise = np.asarray(noise, dtype=float)
    ps = float(np.mean(signal**2))
    pn = float(np.mean(noise**2))
    if pn <= 0 or ps <= 0:
        raise ValueError("signal and noise must have nonzero power")
    scale = math.sqrt(ps / pn) * 10.0 ** (-snr_db / 20.0)
    return noise * scale


def add_noise(signal, noise, snr_db: float) -> tuple[np.ndarray, float]:
    """Return ``(noisy_signal, achieved_snr_db)``: ``signal + scaled noise``."""
    signal = np.asarray(signal, dtype=float)
    noise_scaled = scale_noise(noise, snr_db, signal)
    noisy = signal + noise_scaled
    return noisy, estimate_snr_db(signal, noise_scaled)


def extract_nstdb_noise(
    record_id: str, fs: float | None = None, directory=None
) -> np.ndarray:
    """Load an NSTDB recording as the noise realization (bandpassed).

    The sampling frequency is read from the record itself; when ``fs`` is
    given it must match the record's own fs (spec #6: verification against
    the actual record rather than an assumption).
    """
    from .dataset import NSTDB_DIR, _read_local

    rec = _read_local(record_id, directory or NSTDB_DIR, "rdsamp")
    sig = np.asarray(
        rec.samples if hasattr(rec, "samples") else rec.to_dataframe().to_numpy()
    )
    fs_rec = int(rec.fs)
    if sig.ndim > 1:
        sig = sig[:, 0]  # first channel
    if fs is None:
        fs = fs_rec
    elif int(fs) != fs_rec:
        raise ValueError(
            f"record {record_id} has fs {fs_rec} Hz, not the requested {fs} Hz"
        )
    return bandpass_filter(sig, fs_rec)


def run_noise_sweep(
    models: dict[str, Any],
    clean_segments: list[np.ndarray],
    labels: list[str],
    noise_realizations: list[np.ndarray],
    fs: float = 360.0,
    snrs: list[float] | None = None,
    seed: int = 42,
    feature_columns: list[str] | None = None,
    feature_defaults: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Sweep SNR levels: add NSTDB noise to clean beats, classify, measure.

    For each SNR level, every clean beat segment is mixed with a cycled
    noise realization, features are re-extracted, and the fitted models
    (pipelines) predict on the noisy features. Returns a DataFrame with
    columns: ``snr_db``, ``model``, ``accuracy``, ``macro_f1``,
    ``sensitivity_macro``, ``balanced_accuracy``.
    """
    snrs = SNRS_DB if snrs is None else list(snrs)
    rng = np.random.default_rng(seed)
    rows = []
    for snr in snrs:
        feats = []
        for i, seg in enumerate(clean_segments):
            if not noise_realizations:
                noisy = np.asarray(seg, dtype=float)
            else:
                noise = noise_realizations[i % len(noise_realizations)]
                noise_seg = np.resize(noise, len(seg))
                # small deterministic phase jitter for variety
                shift = int(
                    rng.integers(
                        -max(1, len(noise_seg) // 8), max(2, len(noise_seg) // 8)
                    )
                )
                noise_seg = np.roll(noise_seg, shift)
                noisy, _achieved = add_noise(seg, noise_seg, snr)
            feats.append(extract_features(noisy, fs))
        X = _align_features(pd.DataFrame(feats), feature_columns, feature_defaults)
        for name, model in models.items():
            pred = np.asarray(model.predict(X))
            _prec, rec, f1, _sup = precision_recall_fscore_support(
                np.asarray(labels), pred, labels=CLASSES, zero_division=0
            )
            rows.append(
                {
                    "snr_db": snr,
                    "model": name,
                    "accuracy": float(accuracy_score(labels, pred)),
                    "macro_f1": float(np.mean(f1)),
                    "sensitivity_macro": float(np.mean(rec)),
                    "balanced_accuracy": float(balanced_accuracy_score(labels, pred)),
                }
            )
    return pd.DataFrame(rows)


def generate_noisy_features(
    clean_segments: list[np.ndarray],
    noise_realizations: list[np.ndarray],
    snrs: list[float],
    fs: float,
    rng: np.random.Generator,
    feature_columns: list[str] | None = None,
) -> tuple[dict, dict]:
    """Generate the noisy feature matrices for each SNR **once** (fair design).

    For each SNR level, every clean beat segment is mixed with a cycled NSTDB
    noise realization (deterministic phase jitter from ``rng`` — a single rng
    state per call, shared across all models), and features are re-extracted.

    Returns ``(noisy_by_snr, achieved_by_snr)``: ``noisy_by_snr[snr]`` is the
    (aligned) feature DataFrame used by *every* model in the same fold — the
    same noisy validation samples are shared across classifiers (spec #4) —
    and ``achieved_by_snr[snr]`` is the mean achieved SNR of the noisy set
    (requested vs achieved SNR consistency is verifiable from the recorded
    ``achieved_snr_db`` values; ``add_noise`` scales the noise so that
    ``SNR(signal, noise) == snr_db``).
    """
    noisy_by_snr: dict[float, pd.DataFrame] = {}
    achieved_by_snr: dict[float, float] = {}
    for snr in snrs:
        feats = []
        achieveds = []
        for i, seg in enumerate(clean_segments):
            if not noise_realizations:
                noisy = np.asarray(seg, dtype=float)
                achieveds.append(float("inf"))
            else:
                noise = noise_realizations[i % len(noise_realizations)]
                noise_seg = np.resize(noise, len(seg))
                # small deterministic phase jitter (one rng call — shared across models)
                shift = int(
                    rng.integers(
                        -max(1, len(noise_seg) // 8), max(2, len(noise_seg) // 8)
                    )
                )
                noise_seg = np.roll(noise_seg, shift)
                noisy, achieved = add_noise(seg, noise_seg, snr)
                achieveds.append(float(achieved))
            feats.append(extract_features(noisy, fs))
        Xn = _align_features(pd.DataFrame(feats), feature_columns, None)
        noisy_by_snr[float(snr)] = Xn
        achieved_by_snr[float(snr)] = (
            float(np.mean(achieveds)) if achieveds else float("nan")
        )
    return noisy_by_snr, achieved_by_snr


def _mitdb_reference_compatible(
    m_rec, m_ann, sig: np.ndarray, fs_rec: int
) -> dict[str, Any] | None:
    """Verify that a matched clean MITDB record is a valid ground truth for
    the NSTDB recording (spec #2).

    Returns a small compatibility dict when the reference is compatible —
    same sampling frequency, at least one AAMI-mapped reference annotation,
    and every reference R position inside the NSTDB signal bounds —
    otherwise ``None``: when the annotation/signal sample-count correspondence
    cannot be established, the reference must not be used as ground truth.
    """
    if int(m_rec.fs) != int(fs_rec):
        return None
    syms = [str(s) for s in np.asarray(m_ann.symbol)]
    pos = np.asarray(m_ann.sample, dtype=int)
    if len(pos) != len(syms) or len(pos) == 0:
        return None
    if int(pos.min()) < 0 or int(pos.max()) >= len(sig):
        return None
    n_mapped = int(sum(s in AAMI_MAP for s in syms))
    if n_mapped == 0:
        return None
    return {
        "fs_match": True,
        "n_reference_annotations": len(pos),
        "n_mapped_reference_annotations": n_mapped,
        "nstdb_n_samples": len(sig),
        "max_reference_r_sample": int(pos.max()),
    }


def run_nstdb_evaluation(
    models: dict[str, Any],
    nstdb_records: list[str],
    fs: float = 360.0,
    directory=None,
    segment_args: tuple[float, float] | None = None,
    feature_columns: list[str] | None = None,
    feature_defaults: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Classify segmented NSTDB recordings (recorded-noise evaluation).

    Spec #3: annotation-aware — the labels travel with the aligned beats
    via :func:`ecg_robust.annotations.annotate_annotations` +
    :func:`ecg_robust.segmentation.extract_beats` (original annotation
    index/sample/symbol preserved; no truncation or assumed alignment after
    filtering). RR interval/ratio features use the clean-pipeline indexing
    convention (``rr_features_for_valid_beats``).

    Ground truth is established by the first available source, recorded per
    record as ``truth_source`` (spec #2):

    1. ``"nstdb_annotations"`` — the NSTDB record's own annotations;
    2. ``"mitdb_reference"`` — the matched clean MITDB record's annotations,
       when :func:`_mitdb_reference_compatible` accepts the beat/signal
       correspondence and the reference beats segment the NSTDB signal;
    3. ``"r_peak_fallback"`` — detected R peaks; **no ground truth**:
       predictions are reported without inventing metrics.

    The ``fs`` argument is verified against each record's actual fs.

    Returns ``(df, details)``: one row per record × model with ``n_beats``
    (predicted), ``predicted_class_counts``, ``n_valid_truth``/``n_valid``,
    ``n_annotations``, ``n_excluded_annotations``,
    ``n_unknown_annotations``, ``n_unaligned_annotations`` (annotations not
    aligned to a valid beat), ``truth_source``, and the metrics
    ``accuracy``/``macro_f1``/``sensitivity_macro``/``balanced_accuracy`` —
    metrics only where valid reference truth exists. ``details`` is a list of
    per-record dicts with ``record``, ``n_beats``, ``n_samples``, ``fs``,
    ``segment_lengths_head``, the alignment counters and ``truth_source``
    (plus ``mitdb_compat`` when the MITDB reference was used).
    """
    from .annotations import annotate_annotations
    from .dataset import MITDB_DIR, NSTDB_DIR, _read_local, database_records
    from .segmentation import extract_beats, rr_features_for_valid_beats

    before, after = segment_args if segment_args is not None else (0.25, 0.45)
    rows = []
    details: list[dict[str, Any]] = []
    for rec_id in nstdb_records:
        rec = _read_local(rec_id, directory or NSTDB_DIR, "rdsamp")
        if hasattr(rec, "samples"):
            sig = np.asarray(rec.samples)
        else:
            sig = rec.to_dataframe().to_numpy()
        fs_rec = int(rec.fs)
        if fs is not None and int(fs) != fs_rec:
            raise ValueError(
                f"record {rec_id} has fs {fs_rec} Hz, not the requested {fs} Hz"
            )
        sig = sig[:, 0] if sig.ndim > 1 else sig
        sig = bandpass_filter(sig, fs_rec)

        # --- (a) annotation-aware alignment (spec #3) ---
        truth_source = "r_peak_fallback"
        mitdb_compat: dict[str, Any] | None = None
        beats = None
        annot_recs = None
        r_pos_full: list[int] = []
        try:
            ann = _read_local(rec_id, directory or NSTDB_DIR, "rdann", extension="atr")
            codes = [str(s) for s in np.asarray(ann.symbol)]
            r_pos_full = [int(s) for s in np.asarray(ann.sample)]
            annot_recs = annotate_annotations(
                codes, samples=r_pos_full, signal_length=len(sig)
            )
            beats = extract_beats(sig, fs_rec, annot_recs, before, after)
            if beats:
                truth_source = "nstdb_annotations"
            else:
                # NSTDB annotations produced no usable beats: no valid truth.
                beats = None
        except Exception:  # noqa: BLE001 -- annotation reading is best-effort
            beats = None

        # --- (b) fallback: MITDB-matched record (annotated clean reference) ---
        # Ground truth = the matched clean MITDB record's annotations; the
        # segments and features come from the NSTDB recording. The
        # correspondence is accepted only when the two are compatible
        # (spec #2): same fs, at least one AAMI-mapped reference annotation,
        # and every reference R position inside the NSTDB signal bounds.
        if beats is None and rec_id in database_records(MITDB_DIR):
            try:
                m_rec, m_ann = (
                    _read_local(rec_id, MITDB_DIR, "rdsamp"),
                    _read_local(rec_id, MITDB_DIR, "rdann", extension="atr"),
                )
                mitdb_compat = _mitdb_reference_compatible(m_rec, m_ann, sig, fs_rec)
                if mitdb_compat is not None:
                    m_codes = [str(s) for s in np.asarray(m_ann.symbol)]
                    m_r = [int(s) for s in np.asarray(m_ann.sample)]
                    m_recs = annotate_annotations(m_codes, samples=m_r)
                    candidate = extract_beats(sig, fs_rec, m_recs, before, after)
                    if candidate:
                        beats = candidate
                        annot_recs = m_recs
                        r_pos_full = m_r
                        truth_source = "mitdb_reference"
            except Exception:  # noqa: BLE001 -- MITDB fallback is best-effort
                beats = None
                mitdb_compat = None

        # --- (c) R-peak fallback: no labels — predictions only ---
        if beats is None:
            truth_source = "r_peak_fallback"
            from scipy.signal import find_peaks

            peaks, _ = find_peaks(-sig, distance=int(fs_rec * 0.6))
            r_pos_full = peaks.tolist()
            annot_recs = [
                AnnotationRecord(
                    index=i,
                    symbol="R",
                    sample=int(pp),
                    aami_class=None,
                    status="unknown",
                )
                for i, pp in enumerate(r_pos_full)
            ]
            beats = extract_beats(sig, fs_rec, annot_recs, before, after)

        if not beats:
            continue

        # Features aligned to the beats (spec #3): RR interval/ratio features
        # use the same indexing convention as the clean pipeline
        # (`rr_features_for_valid_beats`).
        valid_beats = [b for b in beats if b.aami_class is not None]
        rr_seq = rr_features_for_valid_beats(valid_beats, fs_rec)
        feats = []
        for beat, (rr_s, rr_r) in zip(valid_beats, rr_seq):
            feats.append(extract_features(beat.segment, fs_rec, rr_s, rr_r))
        X = _align_features(pd.DataFrame(feats), feature_columns, feature_defaults)

        y_true = [b.aami_class for b in valid_beats]  # aligned with valid beats

        # Phase 2: record annotation-alignment diagnostics for the record.
        # ``n_unaligned_annotations`` = annotations that do not correspond to
        # a retained valid beat (excluded/unknown annotations plus mapped
        # annotations whose segment was dropped); in the R-peak fallback
        # path, the detected peaks not retained as beats.
        n_annotations = len(annot_recs) if annot_recs else 0
        n_excluded = (
            int(sum(1 for r in annot_recs if r.status == "excluded"))
            if annot_recs
            else 0
        )
        n_unknown = (
            int(sum(1 for r in annot_recs if r.status == "unknown"))
            if annot_recs
            else 0
        )
        n_valid = len(valid_beats)
        n_unaligned = int(n_annotations - n_valid)

        for name, model in models.items():
            if X.shape[0] == 0:
                # No valid beats to score: sklearn raises on 0-sample input;
                # report an empty prediction set — no metrics are invented.
                pred = np.empty(0, dtype=object)
            else:
                pred = np.asarray(model.predict(X))
            row = {
                "record": rec_id,
                "model": name,
                "n_beats": len(pred),
                "predicted_class_counts": {c: int(np.sum(pred == c)) for c in CLASSES},
                "n_valid_truth": n_valid,
                "n_valid": n_valid,
                "n_annotations": n_annotations,
                "n_excluded_annotations": int(n_excluded),
                "n_unknown_annotations": int(n_unknown),
                "n_unaligned_annotations": int(n_unaligned),
                "truth_source": truth_source,
            }
            # Metrics only where valid reference truth is available
            # (spec #3: no truncation, labels genuinely associated with the
            # evaluated beats).
            pairs = [(v, pr) for v, pr in zip(y_true, pred) if v is not None]
            if pairs:
                yt = np.array([pr[0] for pr in pairs])
                pr = np.array([pr[1] for pr in pairs])
                _prec, rec, f1, _sup = precision_recall_fscore_support(
                    yt, pr, labels=CLASSES, zero_division=0
                )
                row.update(
                    {
                        "accuracy": float(accuracy_score(yt, pr)),
                        "macro_f1": float(np.mean(f1)),
                        "sensitivity_macro": float(np.mean(rec)),
                        "balanced_accuracy": float(balanced_accuracy_score(yt, pr)),
                    }
                )
            rows.append(row)
        detail_entry: dict[str, Any] = {
            "record": rec_id,
            "n_beats": n_valid,
            "n_samples": len(sig),
            "fs": int(fs_rec),
            "segment_lengths_head": [len(b.segment) for b in beats[:8]],
            "n_annotations": n_annotations,
            "n_valid": n_valid,
            "n_excluded_annotations": int(n_excluded),
            "n_unknown_annotations": int(n_unknown),
            "n_unaligned_annotations": int(n_unaligned),
            "truth_source": truth_source,
        }
        if mitdb_compat is not None:
            detail_entry["mitdb_compat"] = mitdb_compat
        details.append(detail_entry)
    return pd.DataFrame(rows), details
