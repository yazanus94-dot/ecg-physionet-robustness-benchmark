# ECG-ROBUST — Benchmark of patient-independent ECG arrhythmia classification

**ecg-physionet-robustness-benchmark** — "ECG-ROBUST: Patient-Independent ECG
Arrhythmia Classification and Noise-Robustness Benchmark."

A reproducible research pipeline built on the official PhysioNet databases
[MIT-BIH Arrhythmia Database (mitdb)](https://physionet.org/content/mitdb/1.0.0/)
and [MIT-BIH Noise Stress Test Database (nstdb)](https://physionet.org/content/nstdb/1.0.0/):
AAMI-categorized (N/S/V/F/Q) heartbeat classification evaluated
patient-independently with 5-fold `StratifiedGroupKFold` (no patient
leakage; recordings 201 and 202 grouped) for three classifiers
(DummyClassifier, RandomForest, RBF-SVM), plus a noise-robustness sweep on
NSTDB noise realizations at SNR 24/18/12/6/0/−6 dB.

## Results (actual, measured)

All numbers below are ACTUAL, measured on the data in this repository
(`python -m ecg_robust.pipeline`, seeds 42). Sources:
[`reports/results/`](reports/results/), [`reports/TECHNICAL_REPORT.md`](reports/TECHNICAL_REPORT.md).

Method points (spec-corrections, see the technical report):
- **Out-of-fold**: every primary metric is computed from out-of-fold
  predictions — each model is fitted on its training folds only (spec #3).
- **No preprocessing leakage**: per-fold median imputation + scaling are
  fitted inside the sklearn pipeline on training folds (spec #4).
- **Annotation/segment alignment** is explicit (spec #1).
- **Reserved records**: final models for the recorded-noise evaluation exclude
  the reserved records 118/119 (spec #3); CV folds use all recording groups
  (patient-independent).

### Dataset summary (measured)

- MIT-BIH Arrhythmia Database (48 records, fs = 360 Hz); NSTDB (15 records:
  118e00, 118e06, 118e12, 118e18, 118e24, 118e_6, 119e00, 119e06, 119e12,
  119e18, 119e24, 119e_6, bw, em, ma).
- Beats segmented via reference annotations (`x[r_i - 0.25*fs : r_i + 0.45*fs]`):
  109,494 segments; AAMI-mapped beats used in the feature matrix: 4,803
  (computational cap of 1,000 beats per AAMI class; full class distribution of
  mapped beats: N=90,631, Q=8,043, V=7,236, S=2,781, F=803).
- Annotations excluded (known MITDB symbols not in the AAMI map): 2,704;
  unknown codes: 449.

### 5-fold patient-independent CV (actual, out-of-fold)

Mean over the five folds (`reports/results/cv_metrics.csv`):

| Model | Accuracy | Precision (macro) | Recall (macro) | Sensitivity (macro) | Specificity (macro) | F1 (macro) | F1 (weighted) | Balanced Accuracy |
|---|---|---|---|---|---|---|---|---|
| dummy | 0.1149 | 0.0230 | 0.1600 | 0.1600 | 0.8000 | 0.0399 | 0.0302 | 0.1600 |
| hist_gradient_boosting | 0.6253 | 0.5924 | 0.5632 | 0.5632 | 0.9077 | 0.5303 | 0.6096 | 0.5880 |
| logistic_regression | 0.5680 | 0.5338 | 0.5301 | 0.5301 | 0.8940 | 0.4935 | 0.5711 | 0.5558 |
| random_forest | 0.5490 | 0.5784 | 0.5259 | 0.5259 | 0.8873 | 0.4834 | 0.5247 | 0.5514 |
| svm_rbf | 0.5770 | 0.5485 | 0.5315 | 0.5315 | 0.8942 | 0.4979 | 0.5601 | 0.5561 |

- Best CV accuracy: **0.6253** (hist_gradient_boosting).
- CV bullet points: DummyClassifier (most_frequent): accuracy **0.1149**,
  macro F1 0.0399; HistGradientBoostingClassifier: accuracy **0.6253**,
  macro F1 0.5303; LogisticRegression: accuracy **0.5680**, macro F1
  0.4935; RandomForestClassifier: accuracy **0.5490**, macro F1 0.4834;
  RBF-SVM: accuracy **0.5770**, macro F1 0.4979.

### Noise robustness, synthetic mixtures (actual, out-of-fold)

NSTDB noise realizations added to the MITDB beats at the target SNR levels
`{24, 18, 12, 6, 0, -6 dB}` (`SNR = 10*log10(P_signal/P_noise)`); mean over
levels of the **out-of-fold** models scored on held-out validation beats
(`reports/results/noise_results.csv`):

| Model | Accuracy | Macro F1 | Macro Sensitivity | Balanced Accuracy |
|---|---|---|---|---|
| dummy | 0.2151 | 0.0670 | 0.1600 | 0.1600 |
| hist_gradient_boosting | 0.4117 | 0.3007 | 0.3759 | 0.3925 |
| logistic_regression | 0.4972 | 0.3892 | 0.4094 | 0.4251 |
| random_forest | 0.4955 | 0.3799 | 0.4345 | 0.4527 |
| svm_rbf | 0.5437 | 0.4320 | 0.4674 | 0.4845 |

Every model in a fold/SNR is scored on the **identical** noisy samples
(fairness regression tests); the noise-realization rng seed and the
achieved SNR per level are recorded in `reports/results/noise_config.json`.

### Recorded-noise evaluation (actual; NSTDB annotations as truth)

All 15 NSTDB records classified with the final models (trained excluding the
reserved records 118/119), evaluated against the NSTDB record annotations
(`reports/results/nstdb_evaluation.csv`; mean per model):

| Model | Accuracy | Macro F1 | Macro Sensitivity | Balanced Accuracy |
|---|---|---|---|---|
| dummy | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| hist_gradient_boosting | 0.6149 | 0.2517 | 0.2950 | 0.6449 |
| logistic_regression | 0.5023 | 0.2447 | 0.2462 | 0.5552 |
| random_forest | 0.5012 | 0.2074 | 0.2527 | 0.5682 |
| svm_rbf | 0.5089 | 0.2043 | 0.2516 | 0.5669 |

(These are real recorded-noise numbers, not synthetic mixtures. The dummy
predicts the most frequent class of the final training data. Records without
valid ground truth — `bw`, `em`, `ma` (no usable annotations; R-peak
fallback) — report predictions and descriptive statistics only: `n_beats`,
`predicted_class_counts`, `truth_source`; their metric columns are absent and
are never invented. Per-record beat/truth counts are in the CSV; final models
were trained excluding the reserved records 118/119.)

### Expert reference task

`tasks/ecg_arrhythmia_robustness/` — a portfolio-ready expert task with
`TASK.md`, a validated `reference_solution.py`, an automated `grader.py`,
`expected_output.json` and `RUBRIC.md` — independently understandable, with
synthetic fixtures (no proprietary raw data). `python grader.py` passes.

## Project structure

```
.
├── data/                     # downloaded raw PhysioNet data (data/raw/) + features (data/features/)
├── src/ecg_robust/           # the pipeline package (dataset, download, annotations,
│                             #   preprocessing, segmentation, features, splits, models,
│                             #   noise, evaluation, visualization, pipeline, technical_report)
├── scripts/                  # reproduce_pipeline.py
├── configs/default.yaml      # pipeline configuration (fs, snrs, models, ...)
├── tests/                    # pytest suite (annotations, preprocessing, segmentation,
│                             #   features, splits, models, evaluation, noise, dataset, reproducibility)
├── reports/
│   ├── TECHNICAL_REPORT.md   # full methods + ACTUAL results
│   ├── figures/              # ecg_original, ecg_filtered, ecg_segmentation,
│   │                         #   class_distribution, ecg_classes_comparison,
│   │                         #   confusion_matrix_*.png, model_performance, clean_vs_noisy,
│   │                         #   snr_vs_*.png, noise_performance.png
│   └── results/              # cv_metrics.csv, noise_results.csv, noise_config.json,
│                             #   nstdb_evaluation.csv, nstdb_details.json,
│                             #   per_class_oof.json, per_class_train_diagnostics.json,
│                             #   dataset_summary.json, fold_class_counts.json,
│                             #   confusion_matrices.json, feature_ablation.csv,
│                             #   class_weights_study.csv, hyperparameter_search.json
├── requirements.txt / pyproject.toml / environment.yml
├── .github/workflows/test.yml # CI: pytest on push/PR
└── LICENSE, CITATION.cff, .gitignore
```

## Running the pipeline

```bash
pip install -r requirements.txt          # or: pip install -e .
python -m ecg_robust.pipeline            # downloads mitdb + nstdb into data/raw/, runs CV, noise, reports
python -m pytest -q                      # unit tests
```

Dev tools (lint/typecheck):

```bash
pip install -r dev-requirements.txt
python -m ruff check src tests
python -m ruff format --check src tests
python -m mypy src
```

Or in one command:

```bash
python scripts/reproduce_pipeline.py
```

## Method summary (see [reports/TECHNICAL_REPORT.md](reports/TECHNICAL_REPORT.md) for details and formulas)

- **Data:** MIT-BIH Arrhythmia Database (48 records, fs = 360 Hz) + MIT-BIH
  Noise Stress Test Database (15 records: 118e00…119e_6, bw, em, ma). Raw
  data never committed.
- **AAMI mapping:** N={N,L,R,e,j}, S={A,a,J,S}, V={V,E}, F={F}, Q={/,f,Q};
  excluded/unknown annotations reported (`annotate_annotations`).
- **Preprocessing:** 4th-order Butterworth 0.5–40 Hz (`sosfiltfilt`,
  zero-phase), heartbeat segment `x[r_i − 0.25·fs : r_i + 0.45·fs]`
  (252 samples at fs=360), training-fitted normalization; per-fold
  median imputation inside the sklearn pipeline.
- **Features:** mean, std, min/max, energy `E=Σx²`, peak amplitude,
  morphology (rise/fall times, zero crossings), RR_i `=(r_i−r_{i−1})/fs`
  (spec #1 indexing), RR ratios, duration, spectral features, and
  statistical features (skew, kurtosis, percentiles).
- **Validation:** 5-fold `StratifiedGroupKFold` on recording IDs (201/202
  grouped); per-fold scaling/imputation; no patient leakage (verified at run
  time). All primary metrics are **out-of-fold** (spec #2).
- **NSTDB recorded-noise evaluation:** predictions for every NSTDB record;
  metrics only where valid ground truth exists (`n_valid_truth`); records
  without usable annotations report predictions only — metrics are never
  invented (`nstdb_evaluation.csv`, `nstdb_details.json`).
- **Reproducibility:** seeds are fixed (42); the noise-realization rng seed
  and the per-SNR achieved SNR are recorded in
  `reports/results/noise_config.json`; the modeling sample is the
  per-class-capped design (≤ 1,000 beats/AAMI class, seed 42; the full
  mapped-beat dataset is in `data/features/ecg_features.csv`).
- **Metrics:** accuracy, precision, recall, sensitivity, specificity,
  macro F1, weighted F1, balanced accuracy, per-class performance and
  confusion matrices (OOF primary, training diagnostics separately labeled).
- **Noise:** NSTDB-based synthetic mixtures at SNR
  `10·log10(P_signal/P_noise)` for {24, 18, 12, 6, 0, −6 dB}; clean vs noisy
  at identical positions; same noisy samples shared across classifiers
  within a fold (spec #4); plots of SNR vs macro F1 / sensitivity /
  balanced accuracy; real recorded-noise evaluation of NSTDB records
  (118e00…119e_6, bw, em, ma) — labeled as such, not presented as
  official-noise claims; reserved records 118/119 excluded from final
  training (spec #3).
- **Classifiers (spec Phase 6 — stronger experimental comparison):**
  DummyClassifier (most_frequent), LogisticRegression (C-balanced),
  RandomForest (n_estimators=100, class_weight="balanced"),
  HistGradientBoosting (native imbalance handling), RBF-SVM
  (C=1.0, class_weight="balanced"). Primary CV results + model comparison
  in `reports/results/cv_metrics.csv`, `reports/results/class_weights_study.csv`,
  `reports/results/feature_ablation.csv`, `reports/results/hyperparameter_search.json`.

## License & attribution

[MIT License](LICENSE) for the code. MIT-BIH databases: PhysioNet
(https://physionet.org); raw files under `data/raw/` subject to PhysioNet's
licensing and are never committed. See [CITATION.cff](CITATION.cff).

## Limitations

Benchmarking/research tool, not a clinical diagnostic system: beat-level AAMI
classification is research-grade and the outputs must not be used for
clinical decisions. Data is coarse/older leads (MITDB); class imbalance is
handled with macro/balanced metrics and class weights (see
`class_weights_study.csv`); the noise cohort is small (the recorded-noise
evaluation covers the NSTDB records); the per-class modeling sample is capped
at 1,000 beats/class (computational constraint for the SVM comparison); where
ground truth cannot be established the recorded-noise table reports
predictions and descriptive statistics only. Spec-correctness notes:
annotation/segment alignment is explicit (spec #1); primary metrics are
out-of-fold (spec #2); NSTDB recorded-noise labels are alignment-aware
(spec #3); noise is fair across classifiers (spec #4); annotation
validation is enforced (spec #5).
