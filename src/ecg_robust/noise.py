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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support

from .annotations import AAMI_MAP, CLASSES
from .dataset import load_nstdb_signal
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
    ps = float(np.mean(signal ** 2))
    pn = float(np.mean(noise ** 2))
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


def extract_nstdb_noise(record_id: str, fs: float, directory=None) -> np.ndarray:
    """Load an NSTDB recording as the noise realization (bandpassed)."""
    from .dataset import NSTDB_DIR, load_nstdb_signal

    sig = load_nstdb_signal(record_id, directory or NSTDB_DIR)
    sig = np.asarray(sig, dtype=float)
    if sig.ndim > 1:
        sig = sig[:, 0]  # first channel
    return bandpass_filter(sig, fs)


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
                shift = int(rng.integers(-max(1, len(noise_seg) // 8),
                                          max(2, len(noise_seg) // 8)))
                noise_seg = np.roll(noise_seg, shift)
                noisy, _achieved = add_noise(seg, noise_seg, snr)
            feats.append(extract_features(noisy, fs))
        X = _align_features(pd.DataFrame(feats), feature_columns, feature_defaults)
        for name, model in models.items():
            pred = np.asarray(model.predict(X))
            prec, rec, f1, _sup = precision_recall_fscore_support(
                np.asarray(labels), pred, labels=CLASSES, zero_division=0
            )
            rows.append(
                {
                    "snr_db": snr,
                    "model": name,
                    "accuracy": float(accuracy_score(labels, pred)),
                    "macro_f1": float(np.mean(f1)),
                    "sensitivity_macro": float(np.mean(rec)),
                    "balanced_accuracy": float(
                        balanced_accuracy_score(labels, pred)
                    ),
                }
            )
    return pd.DataFrame(rows)


def noisy_heldout_metrics(
    pipe: Any,
    y_val: np.ndarray,
    clean_segments: list[np.ndarray],
    noise_realizations: list[np.ndarray],
    snrs: list[float],
    fs: float,
    feature_columns: list[str] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Evaluate a fitted pipeline on noisy versions of held-out beats.

    For each SNR level, each held-out beat segment is mixed with a cycled
    NSTDB noise realization (deterministic seeded jitter), features are
    re-extracted, and ``pipe.predict`` is applied — i.e., out-of-fold models
    evaluate their corresponding held-out validation beats only (spec #3).
    Returns a DataFrame with ``snr_db``, ``n_beats``, ``accuracy``,
    ``macro_f1``, ``sensitivity_macro`` and ``balanced_accuracy``.
    """
    rng = np.random.default_rng(seed)
    rows = []
    y_val = np.asarray(y_val)
    for snr in snrs:
        feats = []
        for i, seg in enumerate(clean_segments):
            if not noise_realizations:
                noisy = np.asarray(seg, dtype=float)
            else:
                noise = noise_realizations[i % len(noise_realizations)]
                noise_seg = np.resize(noise, len(seg))
                shift = int(rng.integers(-max(1, len(noise_seg) // 8),
                                          max(2, len(noise_seg) // 8)))
                noise_seg = np.roll(noise_seg, shift)
                noisy, _achieved = add_noise(seg, noise_seg, snr)
            feats.append(extract_features(noisy, fs))
        Xn = _align_features(pd.DataFrame(feats), feature_columns, None)
        pred = np.asarray(pipe.predict(Xn))
        prec, rec, f1, _sup = precision_recall_fscore_support(
            y_val, pred, labels=CLASSES, zero_division=0
        )
        rows.append(
            {
                "snr_db": float(snr),
                "n_beats": int(len(pred)),
                "accuracy": float(accuracy_score(y_val, pred)),
                "macro_f1": float(np.mean(f1)),
                "sensitivity_macro": float(np.mean(rec)),
                "balanced_accuracy": float(balanced_accuracy_score(y_val, pred)),
            }
        )
    return pd.DataFrame(rows)


def run_nstdb_evaluation(
    models: dict[str, Any],
    nstdb_records: list[str],
    fs: float = 360.0,
    directory=None,
    segment_args: tuple[float, float] | None = None,
    feature_columns: list[str] | None = None,
    feature_defaults: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Classify segmented NSTDB recordings (recorded-noise evaluation).

    Each NSTDB record is loaded, its annotations used to segment the
    signal (with an R-peak fallback when annotations are unavailable),
    features are extracted, and the fitted models predict. Returns a
    DataFrame with ``record``, ``model``, ``accuracy``, ``macro_f1``,
    ``sensitivity_macro``, ``balanced_accuracy``, plus the segmented
    signals and annotation codes saved for plotting
    (``signals``, ``codes`` columns hold lists of arrays/strings).
    """
    from .segmentation import extract_segments, rr_intervals, rr_ratios

    from .dataset import _read_local
    from .features import extract_features
    from .annotations import classify_annotations

    before, after = segment_args or (0.25, 0.45)
    rows = []
    details: list[dict[str, Any]] = []
    for rec_id in nstdb_records:
        from .dataset import NSTDB_DIR
        rec = _read_local(rec_id, directory or NSTDB_DIR, "rdsamp")
        if hasattr(rec, "samples"):
            sig = np.asarray(rec.samples)
        else:
            sig = rec.to_dataframe().to_numpy()
        fs_rec = int(rec.fs)
        sig = sig[:, 0] if sig.ndim > 1 else sig
        sig = bandpass_filter(sig, fs_rec)
        try:
            ann = _read_local(rec_id, directory or NSTDB_DIR, "rdann", extension="atr")
            r_pos = [int(s) for s in np.asarray(ann.sample)]
            codes = [str(s) for s in np.asarray(ann.symbol)]
        except Exception:
            from scipy.signal import find_peaks

            peaks, _ = find_peaks(-sig, distance=int(fs_rec * 0.6))
            r_pos = peaks.tolist()
            codes = []
        segments = extract_segments(sig, fs_rec, r_pos, before, after)
        rrs = rr_intervals(r_pos, fs_rec)
        rr_rat = rr_ratios(rrs)
        feats, labels_y = [], []
        for seg in segments:
            feats.append(extract_features(seg, fs_rec))
        if not feats:
            continue
        X = _align_features(pd.DataFrame(feats), feature_columns, feature_defaults)
        # labels for metrics: prefer the matched MITDB record; else use the
        # NSTDB's own annotations (NSTDB records carry beat annotations;
        # non-AAMI symbols such as onsets map to None and are excluded).
        from .dataset import MITDB_DIR, database_records

        y_true = None
        if rec_id in database_records(MITDB_DIR):
            m_rec, m_ann = _read_local(rec_id, MITDB_DIR, "rdsamp"), _read_local(
                rec_id, MITDB_DIR, "rdann", extension="atr"
            )
            if hasattr(m_rec, "samples"):
                m_sig = np.asarray(m_rec.samples)
            else:
                m_sig = m_rec.to_dataframe().to_numpy()
            m_r = [int(s) for s in np.asarray(m_ann.sample)]
            m_segs = extract_segments(m_sig[:, 0] if m_sig.ndim > 1 else m_sig,
                                      int(m_rec.fs), m_r, before, after)
            m_mapped, _excl, _unk = classify_annotations(
                [str(s) for s in np.asarray(m_ann.symbol)]
            )
            y_true = [c for c in m_mapped][: len(X)]
        if y_true is None and codes:
            # NSTDB annotations as truth (may contain None for non-AAMI
            # symbols such as onset markers — filtered in the metric step).
            aami_codes = [AAMI_MAP.get(str(s)) for s in codes]
            if any(v is not None for v in aami_codes):
                y_true = list(aami_codes[: len(X)])
        for name, model in models.items():
            pred = np.asarray(model.predict(X))
            row = {
                "record": rec_id,
                "model": name,
                "n_beats": int(len(pred)),
                "predicted_class_counts": {
                    c: int(np.sum(pred == c)) for c in CLASSES
                },
            }
            if y_true is not None:
                pairs = [(v, p) for v, p in zip(y_true, pred) if v is not None]
                if pairs:
                    yt = np.array([p[0] for p in pairs])
                    pr = np.array([p[1] for p in pairs])
                    prec, rec, f1, _sup = precision_recall_fscore_support(
                        yt, pr, labels=CLASSES, zero_division=0
                    )
                    row.update(
                        {
                            "accuracy": float(accuracy_score(yt, pr)),
                            "macro_f1": float(np.mean(f1)),
                            "sensitivity_macro": float(np.mean(rec)),
                            "balanced_accuracy": float(
                                balanced_accuracy_score(yt, pr)
                            ),
                        }
                    )
            rows.append(row)
        details.append({
            "record": rec_id,
            "n_segments": int(len(segments)),
            "n_samples": int(len(sig)),
            "fs": int(fs_rec),
            "segment_lengths_head": [int(len(s)) for s in segments[:8]],
        })
    return pd.DataFrame(rows), details
