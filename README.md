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

> **NOTE — results pending:** this branch contains the fully reproducible
> pipeline and unit tests; the pipeline has been run and the actual CV and
> noise results are recorded below. See [`reports/TECHNICAL_REPORT.md`](reports/TECHNICAL_REPORT.md)
> and [`reports/results/`](reports/results/).

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
