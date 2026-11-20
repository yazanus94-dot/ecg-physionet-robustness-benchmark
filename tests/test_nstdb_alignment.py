"""Phase 2 tests — NSTDB recorded-noise evaluation label alignment (spec #3).

The evaluated segments and their truth labels must be aligned: each valid
segment carries the original annotation index/sample/symbol/AAMI class.
"""

import numpy as np
import pytest

from ecg_robust.annotations import annotate_annotations
from ecg_robust.dataset import NSTDB_DIR, database_records
from ecg_robust.segmentation import (
    extract_beats,
    rr_features_for_valid_beats,
)


def _beats(r_positions, symbols, sig, fs):
    annot_recs = annotate_annotations(symbols, samples=r_positions)
    return extract_beats(sig, fs, annot_recs), annot_recs


def _valid(beats):
    return [b for b in beats if b.aami_class is not None]


def test_dropped_segment_in_middle():
    # One beat's segment is dropped (degenerate at the signal boundary).
    fs = 360.0
    sig = np.zeros(3600)
    beats, _ = _beats([900, 1800, 9000], ["N", "N", "N"], sig, fs)
    valid = _valid(beats)
    assert len(valid) == 2
    assert [b.r_sample for b in valid] == [900, 1800]


def test_excluded_annotation_between_valid_beats():
    fs = 360.0
    sig = np.zeros(20000)
    beats, _ = _beats([1000, 2000, 3000, 4000], ["N", "T", "N", "N"], sig, fs)
    valid = _valid(beats)
    assert [b.symbol for b in valid] == ["N", "N", "N"]
    rr = rr_features_for_valid_beats(valid, fs)
    assert rr[1][0] == pytest.approx((3000 - 1000) / 360.0)  # spans the excluded 'T'


def test_unknown_annotation_between_valid_beats():
    fs = 360.0
    sig = np.zeros(20000)
    beats, _ = _beats([1000, 2000, 3000, 4000], ["N", "c", "N", "N"], sig, fs)
    valid = _valid(beats)
    rr = rr_features_for_valid_beats(valid, fs)
    assert rr[1][0] == pytest.approx((3000 - 1000) / 360.0)


def test_annotation_near_beginning():
    fs = 360.0
    sig = np.zeros(20000)
    beats, _ = _beats([0, 1000, 2000], ["N", "N", "N"], sig, fs)
    rr = rr_features_for_valid_beats(_valid(beats), fs)
    assert rr[0] == (None, None)
    assert rr[1][0] == pytest.approx((1000 - 0) / 360.0)


def test_annotation_near_end():
    fs = 360.0
    sig = np.zeros(20000)
    beats, _ = _beats([19500, 19700, 19900], ["N", "N", "N"], sig, fs)
    rr = rr_features_for_valid_beats(_valid(beats), fs)
    assert rr[1][0] == pytest.approx((19700 - 19500) / 360.0)
    assert rr[2][0] == pytest.approx((19900 - 19700) / 360.0)


def test_record_with_no_annotations():
    beats, annot_recs = _beats([], [], np.zeros(2000), 360.0)
    assert beats == []
    assert annot_recs == []


def test_annotation_beyond_signal_dropped():
    fs = 360.0
    sig = np.zeros(3600)
    beats, _ = _beats([1000, 9500], ["N", "N"], sig, fs)
    valid = _valid(beats)
    assert [b.r_sample for b in valid] == [1000]


def test_ground_truth_not_invented_when_unavailable():
    # When there is no valid reference truth, `run_nstdb_evaluation` reports
    # predictions and descriptive statistics, not fabricated accuracy/F1.
    # (covered by the real-data smoke test below; the invariant asserted here:
    # the truth list contains only mapped AAMI classes or None.)
    beats, _ = _beats([1000, 2000], ["N", "T"], np.zeros(20000), 360.0)
    y_true = [b.aami_class for b in beats]
    assert all(v is None or v in {"N", "S", "V", "F", "Q"} for v in y_true)


def test_run_nstdb_evaluation_reports_n_valid_truth():
    # End-to-end smoke (real data): n_valid_truth <= n_beats, predictions
    # exist for every beat, metrics finite where truth exists, and the
    # Phase-2 alignment diagnostics are recorded per record.
    if not database_records(NSTDB_DIR):
        pytest.skip("NSTDB not downloaded")
    from sklearn.dummy import DummyClassifier

    from ecg_robust.noise import run_nstdb_evaluation

    try:
        import pandas as pd

        from ecg_robust.pipeline import MAX_BEATS_PER_CLASS, _sampled_beat_positions

        df = pd.read_csv("data/features/ecg_features.csv")
        feat_subset = df[df["class"].notna()].reset_index(drop=True)
        chosen = _sampled_beat_positions(
            list(feat_subset["class"]), MAX_BEATS_PER_CLASS, 42
        )
        X = feat_subset.iloc[chosen][
            [
                c
                for c in df.columns
                if c not in ("record", "beat_idx", "class", "r_sample")
            ]
        ].reset_index(drop=True)
        y = feat_subset["class"].iloc[chosen].astype(str).to_numpy()
        model = DummyClassifier(strategy="most_frequent").fit(X.head(400), y[:400])
    except Exception:  # noqa: BLE001 -- synthetic fallback when artifacts are missing
        X = np.random.default_rng(0).normal(size=(50, 4))
        y = np.array(["N"] * 50)
        model = DummyClassifier(strategy="most_frequent").fit(X, y)

    df_out, details = run_nstdb_evaluation(
        {"dummy": model}, database_records(NSTDB_DIR)[:2]
    )
    assert len(df_out) == 2
    for _, row in df_out.iterrows():
        assert row["n_beats"] > 0
        assert row["n_valid_truth"] <= row["n_beats"]
        assert int(row["n_beats"]) == sum(row["predicted_class_counts"].values())
        # Phase 2: alignment diagnostics recorded per record
        assert row["n_excluded_annotations"] >= 0
        assert row["n_unknown_annotations"] >= 0
        assert row["n_unaligned_annotations"] >= 0
        if row["n_valid_truth"] > 0 and "accuracy" in row:
            assert 0.0 <= row["accuracy"] <= 1.0
        else:
            assert "accuracy" not in row
    assert len(details) == 2
    for d in details:
        assert "record" in d and "n_beats" in d
        assert d["n_excluded_annotations"] >= 0
        assert d["n_unknown_annotations"] >= 0


def test_mitdb_reference_compatibility_unit():
    # Phase 2: `_mitdb_reference_compatible` accepts a clean MITDB reference
    # only when the beat/signal correspondence is compatible (spec #2): same
    # fs, at least one AAMI-mapped reference annotation, and all reference R
    # positions inside the NSTDB signal bounds. Annotation and signal
    # sample-count mismatches -> None.
    import numpy as np

    from ecg_robust.noise import _mitdb_reference_compatible

    class _Rec:
        def __init__(self, fs, samples):
            self.fs = fs
            self.sample = np.asarray(samples)

    class _Ann:
        def __init__(self, symbols, samples):
            self.symbol = np.asarray(symbols)
            self.sample = np.asarray(samples)

    sig = np.zeros(10000)
    ok = _mitdb_reference_compatible(
        _Rec(360, [100, 200, 300]),
        _Ann(["N", "A", "T"], [100, 200, 300]),
        sig,
        360,
    )
    assert ok is not None
    assert ok["fs_match"] is True
    assert ok["n_reference_annotations"] == 3
    assert ok["n_mapped_reference_annotations"] == 2  # N->N, A->S; T excluded
    assert ok["max_reference_r_sample"] == 300
    # Sampling-frequency mismatch -> not compatible.
    assert (
        _mitdb_reference_compatible(
            _Rec(250, [100, 200, 300]),
            _Ann(["N", "A", "T"], [100, 200, 300]),
            sig,
            360,
        )
        is None
    )
    # Annotation/signal sample-count mismatch (reference R position outside
    # the signal bounds) -> not compatible.
    assert (
        _mitdb_reference_compatible(
            _Rec(360, [100, 200, 99999]),
            _Ann(["N", "A", "T"], [100, 200, 99999]),
            sig,
            360,
        )
        is None
    )
    # No AAMI-mapped reference annotations -> not compatible.
    assert (
        _mitdb_reference_compatible(
            _Rec(360, [100, 200]), _Ann(["T", "P"], [100, 200]), sig, 360
        )
        is None
    )


def test_record_without_annotations_reports_no_truth():
    # End-to-end (real data, Phase 2): every NSTDB record whose evaluation
    # has no valid reference truth gets `truth_source == "r_peak_fallback"`;
    # predictions are reported and no metrics are invented when there is no
    # valid truth. Where truth exists, `truth_source` is the annotation
    # source and metrics are finite.
    if not database_records(NSTDB_DIR):
        pytest.skip("NSTDB not downloaded")
    import pandas as pd
    from sklearn.dummy import DummyClassifier

    from ecg_robust.noise import run_nstdb_evaluation

    try:
        from ecg_robust.pipeline import MAX_BEATS_PER_CLASS, _sampled_beat_positions

        df = pd.read_csv("data/features/ecg_features.csv")
        feat_subset = df[df["class"].notna()].reset_index(drop=True)
        chosen = _sampled_beat_positions(
            list(feat_subset["class"]), MAX_BEATS_PER_CLASS, 42
        )
        X = feat_subset.iloc[chosen][
            [
                c
                for c in df.columns
                if c not in ("record", "beat_idx", "class", "r_sample")
            ]
        ].reset_index(drop=True)
        y = feat_subset["class"].iloc[chosen].astype(str).to_numpy()
        model = DummyClassifier(strategy="most_frequent").fit(X.head(400), y[:400])
    except Exception:  # noqa: BLE001 -- synthetic fallback when artifacts missing
        model = DummyClassifier(strategy="most_frequent").fit(
            np.zeros((10, 4)), ["N"] * 10
        )

    out, details = run_nstdb_evaluation({"dummy": model}, database_records(NSTDB_DIR))
    assert len(out) == len(details)
    no_truth_records = set()
    for _, row in out.iterrows():
        assert row["truth_source"] in (
            "nstdb_annotations",
            "mitdb_reference",
            "r_peak_fallback",
        )
        if row["n_valid_truth"] == 0:
            no_truth_records.add(row["record"])
            assert row["truth_source"] == "r_peak_fallback"
        else:
            assert row["truth_source"] in ("nstdb_annotations", "mitdb_reference")
            if "accuracy" in row and pd.notna(row["accuracy"]):
                assert 0.0 <= row["accuracy"] <= 1.0
    # At least one NSTDB record has no valid truth (e.g. bw, em, ma:
    # annotation-less recordings): metrics are absent or NaN — never invented.
    assert no_truth_records, "expected at least one NSTDB record without valid truth"
    if "accuracy" in out.columns:
        sub = out[out["record"].isin(no_truth_records)]
        assert sub["accuracy"].isna().all()
    # Details: every record reports its alignment diagnostics.
    for d in details:
        assert d["n_valid"] >= 0
        assert d["n_excluded_annotations"] >= 0
        assert d["n_unknown_annotations"] >= 0
        assert d["n_unaligned_annotations"] >= 0
