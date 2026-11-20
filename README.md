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
  112,647 segments; AAMI-mapped beats used in the feature matrix: 4,803
  (computational cap of 1,000 beats per AAMI class; full class distribution of
  mapped beats: N=90,631, Q=8,043, V=7,236, S=2,781, F=803).
- Annotations excluded (known MITDB symbols not in the AAMI map): 2,704;
  unknown codes: 449.

### 5-fold patient-independent CV (actual, out-of-fold)

Mean over the five folds (`reports/results/cv_metrics.csv`):

| Model | Accuracy | Precision (macro) | Recall (macro) | Sensitivity (macro) | Specificity (macro) | F1 (macro) | F1 (weighted) | Balanced Accuracy |
|---|---|---|---|---|---|---|---|---|
| dummy | 0.1149 | 0.0230 | 0.1600 | 0.1600 | 0.8000 | 0.0399 | 0.0302 | 0.1600 |
| random_forest | 0.4404 | 0.4653 | 0.4118 | 0.4118 | 0.8597 | 0.3522 | 0.3958 | 0.4339 |
| svm_rbf | 0.4596 | 0.4651 | 0.4177 | 0.4177 | 0.8641 | 0.3837 | 0.4395 | 0.4381 |

- Best CV accuracy: **0.4596** (svm_rbf).
- CV bullet points: DummyClassifier (most_frequent): accuracy **0.1149**,
  macro F1 0.0399; RandomForestClassifier: accuracy **0.4404**, macro F1
  0.3522; RBF-SVM: accuracy **0.4596**, macro F1 0.3837.

### Noise robustness, synthetic mixtures (actual, out-of-fold)

NSTDB noise realizations added to the MITDB beats at the target SNR levels
`{24, 18, 12, 6, 0, -6 dB}` (`SNR = 10*log10(P_signal/P_noise)`); mean over
levels of the **out-of-fold** models scored on held-out validation beats
(`reports/results/noise_results.csv`):

| Model | Accuracy | Macro F1 | Macro Sensitivity | Balanced Accuracy |
|---|---|---|---|---|
| dummy | 0.1149 | 0.0399 | 0.1600 | 0.1600 |
| random_forest | 0.3312 | 0.2312 | 0.3092 | 0.3260 |
| svm_rbf | 0.3519 | 0.2812 | 0.3325 | 0.3499 |

### Recorded-noise evaluation (actual; NSTDB-annotated beats)

All 15 NSTDB records classified with the final models (trained excluding the
reserved records 118/119), evaluated against the NSTDB record annotations
(`reports/results/nstdb_evaluation.csv`; mean per model):

| Model | Accuracy | Macro F1 | Macro Sensitivity | Balanced Accuracy |
|---|---|---|---|---|
| dummy | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| random_forest | 0.4820 | 0.1903 | 0.2336 | 0.5340 |
| svm_rbf | 0.4221 | 0.1741 | 0.2212 | 0.5085 |

(These are real recorded-noise numbers, not synthetic mixtures. The dummy
predicts the most frequent class of the final training data — Q — and the
NSTDB stress records contain no paced (Q) beats, so its accuracy is 0.0;
the true/predicted beat counts per record are in the CSV.)

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
│   └── results/              # cv_metrics.csv, noise_results.csv,
│                             #   nstdb_evaluation.csv, nstdb_details.json,
│                             #   per_class_evaluation.json, dataset_summary.json
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

Or in one command:

```bash
python scripts/reproduce_pipeline.py
```

## Method summary (see [reports/TECHNICAL_REPORT.md](reports/TECHNICAL_REPORT.md) for details and formulas)

- **Data:** MIT-BIH Arrhythmia Database (48 records, fs = 360 Hz) + MIT-BIH
  Noise Stress Test Database (records 118, 119). Raw data never committed.
- **AAMI mapping:** N={N,L,R,e,j}, S={A,a,J,S}, V={V,E}, F={F}, Q={/,f,Q};
  excluded/unknown annotations reported.
- **Preprocessing:** 4th-order Butterworth 0.5–40 Hz (`sosfiltfilt`,
  zero-phase), heartbeat segment `x[r_i − 0.25·fs : r_i + 0.45·fs]`
  (252 samples at fs=360), training-fitted normalization.
- **Features:** mean, std, min/max, energy `E=Σx²`, peak amplitude,
  morphology (rise/fall times, zero crossings), RR_i `=(r_i−r_{i−1})/fs`,
  RR ratios, duration, spectral features, and statistical features
  (skew, kurtosis, percentiles).
- **Validation:** 5-fold `StratifiedGroupKFold` on recording IDs (201/202
  grouped); per-fold scaling/preprocessing; no patient leakage.
- **Metrics:** accuracy, precision, recall, sensitivity, specificity,
  macro F1, weighted F1, balanced accuracy, per-class performance and
  confusion matrices.
- **Noise:** NSTDB-based synthetic mixtures at SNR
  `10·log10(P_signal/P_noise)` for {24, 18, 12, 6, 0, −6 dB}; clean vs noisy
  at identical positions; plots of SNR vs macro F1 / sensitivity /
  balanced accuracy; real recorded-noise evaluation of NSTDB records
  (118, 119) — labeled as such, not presented as official-noise claims.
- **Classifiers:** DummyClassifier (most_frequent), RandomForest
  (n_estimators=100, class_weight="balanced"), RBF-SVM (C=1.0,
  class_weight="balanced").

## License & attribution

[MIT License](LICENSE) for the code. MIT-BIH databases: PhysioNet
(https://physionet.org); raw files under `data/raw/` subject to PhysioNet's
licensing and are never committed. See [CITATION.cff](CITATION.cff).

## Limitations

Historical data; class imbalance (macro/balanced metrics used); noise-cohort
size (2 NSTDB records; 8 realizations); research-grade, not clinical.
