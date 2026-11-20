# RUBRIC — ECG Arrhythmia Robustness Benchmark task (expert task)

Grading criteria for the expert task in [`TASK.md`](TASK.md). Score each
item 1 point if passed; partial credit at the grader's discretion.

## A. Signal processing (spec #1 + #3)

1. AIMI? no — 4th-order Butterworth, 0.5–40 Hz bandpass, zero-phase
   (`sosfiltfilt` via `ecg_robust.preprocessing.bandpass_filter`).
2. Segmentation `X_i = x[r_i − 0.25·fs : r_i + 0.45·fs]` — correct
   semantics (250 ms pre-R, 450 ms post-R); segments aligned to annotated
   R positions (`extract_beats`).
3. Annotation/segment alignment preserved: retained beats carry the
   original annotation index/sample/symbol/AAMI class
   (`annotate_annotations` + `extract_beats`); excluded/unknown symbols
   represented without shifting subsequent labels.
4. RR features follow the documented convention:
   `RR_i = (R_i − R_(i−1))/fs` and `RR_ratio_i = RR_i / RR_(i−1)` —
   implemented as `rr_s = rrs[i−1]` (i ≥ 1), `rr_ratio = rr_rat[i−2]`
   (i ≥ 2), documented in `rr_features_for_beat`'s docstring.

## B. Feature engineering

5. All documented features present per beat (see `ecg_robust.features`):
   mean, std, min, max, energy `E = Σx²`, peak_amplitude, duration_s,
   rise_time_s, fall_time_s, zero_crossings, spectral_centroid,
   spectral_flatness, fft_peak_frequency, fft_total_energy, skew,
   kurtosis, p5/p25/p50/p75/p95, rr_s, rr_ratio.
6. Feature values finite for the fixture (no NaN/inf).

## C. Classification

7. AAMI mapping correct: N: N, L, R, e, j; S: A, a, J, S; V: V, E;
   F: F; Q: /, f, Q (`AAMI_MAP`).
8. Patient-independent validation: 5-fold `StratifiedGroupKFold` grouped
   by recording ID (`patient_splits`; 201/202 grouped) — no patient
   leakage.
9. Per-class metrics reported (precision, recall, sensitivity,
   specificity, F1, support) — from OOF confusion (`per_class_metrics_from_cm`).
10. Models: DummyClassifier / LogisticRegression / RandomForest /
    HistGradientBoosting / RBF-SVM — built via `build_models`
    (primary: per the benchmark's model set).

## D. Robustness

11. Noise robustness at SNR {24, 18, 12, 6, 0, −6} dB with
    `SNR(dB) = 10·log10(P_signal/P_noise)` (spec #4); same noisy samples
    shared across models within a fold/SNR (`generate_noisy_features`).

## E. Reproducibility

12. Fixed seeds (42) for sampling/splits/models/noise; deterministic
    outputs for fixed input (`test_solve_fixture_deterministic`).
13. Grader passes: `python grader.py` exits 0.

## F. Documentation

14. `TASK.md`, `RUBRIC.md`, `expected_output.json` present and coherent;
    README/TECHNICAL_REPORT updated for the delivered work.

Scoring: 1 point per item (max 14); pass = score ≥ 12/14 with items 12–13
hard requirements.

## G. Scientific integrity (spec — do not fabricate)

15. No fabricated results: the grader validates against
    `expected_output.json` generated from the fixed synthetic fixture with
    the current `ecg_robust` implementation; the task explicitly marks
    outputs as synthetic/reference and distinguishes official PhysioNet
    data from real/synthetic noise in the main project.
