# Task B — Patient-Independent ECG Arrhythmia Classification

## Problem

Build a reproducible beat-level classifier for four AAMI classes (N, S, V, F) using only
**subject-disjoint** folds. No recording or patient may appear in both training and test, and all
preprocessing must be fitted on training rows only. This is a **synthetic** benchmark: the
fixture encodes morphology patterns with subject-specific offsets; no PhysioNet records or hidden
labels are used.

## Input / output schema

`patient_independent_oof(features, labels, groups, n_splits=4) -> dict` with:
- `features`: DataFrame with `rr_s`, `qrs_width_s`, `qrs_amp`, `spectral_centroid`;
- `labels`: four-class string array;
- `groups`: patient identifiers;
- returns `predictions` (one label per row), `fold_groups` (`train`/`test` patient lists per fold),
  `metrics` (`accuracy`, `balanced_accuracy`, `f1_macro`).

`fit_predict(train_features, train_labels, test_features) -> np.ndarray` trains on the supplied
rows only and returns one label per test row. Invalid inputs (empty, length mismatch, unsupported
labels, single-class training, bad group length) must raise `ValueError`.

## Grading

`python grader.py path/to/candidate.py` verifies:
1. output schema and allowed labels;
2. every fold has disjoint train/test patient sets (checked against `StratifiedGroupKFold`);
3. balanced accuracy >= 0.80 and >= dummy baseline + 0.15;
4. determinism of repeated OOF runs;
5. test label columns cannot influence predictions;
6. invalid inputs raise `ValueError`; NaN features are imputed without failure;
7. all metrics in `expected_output.json` reproduced within +/-0.03.

Exit 0 only when every check passes.

## Reference approach (technical explanation)

`StratifiedGroupKFold(shuffle=True, random_state=42)` splits by patient while preserving class
proportions. Each fold fits a fresh `Pipeline(SimpleImputer(median), StandardScaler,
LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42))` on training rows
only. OOF predictions are aggregated before metric computation, so every row is scored exactly
once by a model that never saw its patient. Class weighting addresses imbalance; the dummy
comparison uses the same folds.

## Reproduction

```bash
python generate_expected_output.py
python grader.py reference_solution.py # PASS
python grader.py starter_solution.py   # FAIL
python -m pytest -q tests
```

## Files

`TASK.md`, `RUBRIC.md`, `starter_solution.py`, `reference_solution.py`, `grader.py`,
`expected_output.json`, `generate_expected_output.py`, `tests/`.

Synthetic benchmark evidence only — never clinical validation or challenge participation.
