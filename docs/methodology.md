# Methodology — ECG-ROBUST

## Dataset
- MIT-BIH Arrhythmia Database (48 records, fs = 360 Hz).
- MIT-BIH Noise Stress Test Database (15 records).
- Beats segmented via reference annotations: `x[r_i - 0.25*fs : r_i + 0.45*fs]`.
- AAMI categorization: N / S / V / F / Q.

## Validation
- 5-fold StratifiedGroupKFold, patient-independent.
- Records 201 and 202 grouped.
- All primary metrics from out-of-fold predictions.

## Preprocessing (no leakage)
- Per-fold median imputation + scaling fitted inside the sklearn pipeline on training folds only.

## Models
DummyClassifier, LogisticRegression, RandomForest, RBF-SVM, HistGradientBoosting.

## Noise robustness
- NSTDB realizations at SNR 24/18/12/6/0/-6 dB.
- Results in `reports/results/noise_results.csv`.

## Reproducibility
- Seeds: 42. Run `python -m ecg_robust.pipeline`.
- Config override via `ECG_ROBUST_CONFIG` (validated).
