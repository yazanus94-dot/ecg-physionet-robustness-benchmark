# Task B — Patient-Independent ECG Arrhythmia Classification

## Problem

Build a four-class (N, S, V, F) beat classifier validated with **patient-disjoint** folds. The
**synthetic educational** fixture does not reproduce the full physiological complexity of
MIT-BIH; no real dataset or hidden label is used.

## Contract

- `fit_predict(train_features, train_labels, test_features) -> np.ndarray`
  trains only on the supplied rows and returns one label per test row.
- `patient_independent_oof(features, labels, groups, n_splits=4) -> dict`
  must return `predictions` (one per row), `fold_groups` (canonical
  `{"train": [...], "test": [...]}` per fold) and `metrics` recomputed from the OOF predictions.
  It must **delegate each fold to `fit_predict`**.

## Independent verification (not self-reported metadata)

The grader regenerates folds with `StratifiedGroupKFold(n_splits=4, shuffle=True,
random_state=42)`, verifies each sample appears in exactly one test fold, calls the candidate's
`fit_predict` once per fold, assembles its own OOF predictions, and requires them to equal the
submitted predictions exactly. `fold_groups` must match the canonical assignment. Reported
metrics must match recomputed metrics. Additional checks: test-label independence, a
preprocessing-boundary probe (changing one test row must not alter other test predictions),
NaN imputation, determinism and invalid-input handling.

## Fixture (harder than a trivial demo)

432 beats, 18 patients, 24 beats/patient. Distributions overlap across classes; patients have
individual offsets; measurement noise, ~3% missing values, occasional corrupted values and a
4:2:2:0.4 class imbalance are present. Seeded (`424242`). Reference and dummy metrics in
`expected_output.json` are calibration evidence from actual runs.

## Acceptance thresholds (calibrated to the reference/dummy runs)

- balanced accuracy >= **0.65**
- macro F1 >= **0.55**
- balanced accuracy >= dummy + **0.15**

## Reproduction

```bash
python generate_expected_output.py
python grader.py reference_solution.py   # PASS
python grader.py starter_solution.py     # FAIL
python -m pytest -q tests                # includes adversarial candidates
```

## Files

`TASK.md`, `RUBRIC.md`, `starter_solution.py`, `reference_solution.py`, `grader.py`,
`expected_output.json`, `generate_expected_output.py`, `tests/`. `tests/candidates/` contains
deliberately invalid solutions (all-rows training, fabricated metadata, constant predictor,
label leakage, incomplete OOF, bad input handling, fabricated metrics) and a valid alternative.

Synthetic educational benchmark only — no clinical or challenge claim.
