"""Reference solution for the ECG Arrhythmia Robustness Benchmark task.

Uses only the project's validated processing functions:

- `ecg_robust.annotations.annotate_annotations` (AAMI mapping + alignment)
- `ecg_robust.segmentation.extract_beats` (aligned heartbeat segments)
- `ecg_robust.segmentation.rr_intervals` / `rr_ratios` / `rr_features_for_beat`
  (RR interval / ratio features with the clean-pipeline indexing convention)
- `ecg_robust.features.extract_features` (23 documented features)
- `ecg_robust.models.build_models` (the validated classifier set)
- `ecg_robust.evaluation.evaluate` (per-class + overall metrics)

Public entry points:

- `make_fixture() -> dict` — deterministic synthetic fixture (fixed seed).
- `process_record(x, fs, r_positions, symbols) -> pd.DataFrame` — segment
  + feature extraction for a filtered ECG record.
- `solve_fixture(fixture, model_name="random_forest") -> dict` — fit the
  model on the fixture beats and return `predictions`, `truth`, `metrics`,
  and the aligned `beats` table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ecg_robust.annotations import CLASSES, annotate_annotations
from ecg_robust.evaluation import evaluate
from ecg_robust.features import extract_features
from ecg_robust.models import build_models
from ecg_robust.segmentation import extract_beats, rr_features_for_valid_beats


def make_fixture() -> dict:
    """Deterministic synthetic fixture: a 12-s signal with 8 R peaks."""
    fs = 360.0
    n = int(fs * 12)
    rng = np.random.default_rng(20261119)
    t = np.arange(n) / fs
    x = 0.2 * np.sin(2 * np.pi * 60 * t) + 0.03 * rng.normal(size=n)
    r_positions = [round((i + 0.2) * 0.75 * fs) for i in range(8)]
    symbols = ["N", "S", "V", "F", "Q", "N", "S", "V"]
    return {"x": x, "fs": fs, "r_positions": r_positions, "symbols": symbols}


def process_record(x, fs, r_positions, symbols) -> pd.DataFrame:
    """Segment + feature extraction for a filtered ECG record.

    Annotation-aware (spec #1/#3): labels travel with the aligned beats via
    ``annotate_annotations`` + ``extract_beats``; RR features use the
    clean-pipeline convention (``rr_features_for_valid_beats``).
    """
    annot_recs = annotate_annotations(symbols, samples=r_positions)
    beats = extract_beats(x, fs, annot_recs)
    valid_beats = [b for b in beats if b.aami_class is not None]
    rr_seq = rr_features_for_valid_beats(valid_beats, fs)
    rows = []
    for beat, (rr_s, rr_r) in zip(valid_beats, rr_seq):
        rows.append(
            {
                "index": beat.index,
                "r_sample": beat.r_sample,
                "symbol": beat.symbol,
                "aami_class": beat.aami_class,
                "status": beat.status,
                **extract_features(beat.segment, fs, rr_s, rr_r),
            }
        )
    return pd.DataFrame(rows)


def solve_fixture(fixture: dict, model_name: str = "random_forest") -> dict:
    """Fit the model on the fixture beats and return outputs for the grader."""
    df = process_record(
        fixture["x"], fixture["fs"], fixture["r_positions"], fixture["symbols"]
    )
    feat_cols = [
        c
        for c in df.columns
        if c not in ("index", "r_sample", "symbol", "aami_class", "status")
    ]
    mask = df["aami_class"].notna()
    X = df.loc[mask, feat_cols].copy()
    # Fill missing RR values (the first beat of a record has no preceding RR
    # — a documented missing value) with the feature-matrix median, as the
    # pipeline does via its median imputer (spec #1).
    X = X.fillna(X.median(numeric_only=True))
    y = df.loc[mask, "aami_class"].astype(str)
    model = build_models(random_state=42)[model_name]
    model.fit(X, y)
    pred = np.asarray(model.predict(X))
    ev = evaluate(y.values, pred, CLASSES)
    return {
        "model": model_name,
        "predictions": [str(p) for p in pred],
        "truth": [str(t) for t in y],
        "metrics": ev,
        "beats": df.to_dict("records"),
    }
