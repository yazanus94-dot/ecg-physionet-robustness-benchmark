# ECG Arrhythmia Robustness Benchmark — Expert Task

A portfolio/production-ready task for expert engineers and researchers:
build a complete, reproducible, biomedical-signal-processing + ML benchmark
pipeline and a reference solution + grader.

This task is independently understandable and reproducible without access
to the ECG-ROBUST repository's internal state — everything it uses is
specified here, using the project's validated processing functions
(`ecg_robust.*`).

## Problem Statement

Build a patient-independent ECG arrhythmia classification benchmark on the
official PhysioNet **MIT-BIH Arrhythmia Database** (mitdb) with
**noise-robustness** against the **MIT-BIH Noise Stress Test Database**
(nstdb). The five AAMI categories are: N (N, L, R, e, j), S (A, a, J, S),
V (V, E), F (F), Q (/ , f, Q).

## Assumptions (all explicit — do not rely on undocumented state)

- **Inputs (reference solution):** the pipeline's segment feature matrix
  is available at `data/features/ecg_features.csv` (produced by
  `python -m ecg_robust.pipeline`). The reference solution works directly
  on raw ECG segments + R-peak annotations instead (see
  `reference_solution.process_record`).
- **Models:** the validated `ecg_robust.models.build_models()` (Dummy,
  Logistic, RandomForest, HistGradientBoosting, RBF-SVM),
  `ecg_robust.splits.patient_splits()` (5-fold StratifiedGroupKFold,
  patient-independent).
- **Validation design:** 5-fold StratifiedGroupKFold by recording ID —
  no patient leakage (201/202 grouped). Per-fold scaling/preprocessing.
- **Noise:** NSTDB noise (records 118e00…119e_6, bw, em, ma), synthetic
  mixtures at SNR {24, 18, 12, 6, 0, −6 dB}; recorded-noise evaluation on
  the NSTDB records. Same noisy samples across classifiers within a
  fold/SNR.
- **Reproducibility:** seeds are fixed: splits/models/noise = 42;
  sampling = 42; no randomness without an explicit seed.

## Inputs

- `reference_solution.make_fixture()` — a deterministic synthetic fixture
  (fixed seed) used by the grader: a 12-s synthetic signal with 8 annotated
  R peaks, labels in the AAMI map.
- The grader validates the reference solution on this fixture against
  `expected_output.json` (regenerate with `generate_expected_output.py`).

## Outputs

The reference solution must produce (`grader.py` checks these):

1. `beats` — the aligned beat table: per beat, the original annotation
   `index`, `r_sample`, `symbol`, `aami_class`, `status` + the 23–24
   documented feature columns (including `rr_s`, `rr_ratio` with the clean
   pipeline's indexing convention: `rr_s = rrs[i-1]` when `i >= 1`,
   `rr_ratio = rr_rat[i-2]` when `i >= 2`).
2. `predictions` — one AAMI class per beat.
3. `truth` — the AAMI truth (for mapped annotations).
4. `metrics` — per-class and overall metrics (`accuracy`, `f1_macro`,
   `f1_weighted`, `balanced_accuracy`, plus per-class entries).
5. Determinism: two runs of the reference solution on the same fixture
   produce identical `predictions` and `truth` (fixed seeds).

## Evaluation Criteria (see `RUBRIC.md`)

Coverage of signal processing correctness, feature engineering,
classification validity (patient-independence, OOF reporting), noise
handling, reproducibility, and documentation. No fabrication — the
reference solution is validated against `expected_output.json`
generated from the fixed fixture with the current `ecg_robust`
implementation.
