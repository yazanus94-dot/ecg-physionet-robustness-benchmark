# ECG-ROBUST -- Technical Report: Patient-Independent ECG Arrhythmia Classification and Noise-Robustness Benchmark

- Repository: `yazanus94-dot/ecg-physionet-robustness-benchmark`
- Report date: 2026-11-20 16:15 UTC
- Pipeline wall time (measured, including downloads): 411.2 s

## 1. Research Question

Can a patient-independent ECG classifier distinguish the five AAMI (Association for the Advancement of Medical Instrumentation) diagnostic categories -- N (normal), S (supraventricular), V (ventricular), F (fusion) and Q (paced/unclassified/paced) -- from a single heartbeat segment without using the patient identity, and how robust is such a classifier when NSTDB noise is added at SNR levels 24, 18, 12, 6, 0 and -6 dB?

## 2. Data and Availability

### 2.1 Source databases

- MIT-BIH Arrhythmia Database (mitdb), PhysioNet, https://physionet.org/content/mitdb/1.0.0/
- MIT-BIH Noise Stress Test Database (nstdb), PhysioNet, https://physionet.org/content/nstdb/1.0.0/ (records 118, 119)
- Raw files stored under `data/raw/` (never uploaded; subject to the above licenses and PhysioNet attribution).

### 2.2 Dataset inventory (measured from the downloaded databases)

- MITDB records: 48 -- 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 111, 112, 113, 114, 115, 116, 117, 118, 119, 121, 122, 123, 124, 200, 201, 202, 203, 205, 207, 208, 209, 210, 212, 213, 214, 215, 217, 219, 220, 221, 222, 223, 228, 230, 231, 232, 233, 234
- NSTDB records: 15 -- 118e00, 118e06, 118e12, 118e18, 118e24, 118e_6, 119e00, 119e06, 119e12, 119e18, 119e24, 119e_6, bw, em, ma
- Sampling frequency (MITDB): 360 Hz
- Beats segmented across all recordings (valid segments; excluded/unknown annotations may apply): 109494
- Beats used in the modeling sample (class-capped): 4803 (full mapped-beat dataset: 109494 beats)
- Excluded (known MITDB symbols absent from the AAMI map): 2704
- Unknown codes (not part of the MITDB database): 449
- AAMI class distribution of the full mapped-beat dataset: {'N': 90631, 'Q': 8043, 'V': 7236, 'S': 2781, 'F': 803}

## 2.3 Results files and column definitions (Phase 5)

- `reports/results/cv_metrics.csv`: per-fold CV metrics (out-of-fold); columns: model, fold, n_train, n_val, accuracy, precision_macro, recall_macro, sensitivity_macro, specificity_macro, f1_macro, f1_weighted, balanced_accuracy.
- `reports/results/noise_results.csv`: out-of-fold synthetic-mixture noise metrics; columns: snr_db, fold, model, n_beats, achieved_snr_db (requested vs achieved SNR, verified by `add_noise`), accuracy, macro_f1, sensitivity_macro, balanced_accuracy. All models of a fold/SNR are scored on identical noisy samples.
- `reports/results/noise_config.json`: noise-experiment configuration — `noise_rng_seed` (noise-realization rng seed), the SNR levels, the number and identities of the NSTDB noise realizations used, their sampling frequencies (verified from the records), the mean achieved SNR per SNR level, and the design note (single shared stream; per fold/SNR the noisy validation matrix is generated once by `generate_noisy_features` and shared by every model).
- `reports/results/nstdb_evaluation.csv`: recorded-noise evaluation; columns: record, model, n_beats (predicted beats), n_valid_truth / n_valid (beats scored with valid reference truth), n_annotations, n_excluded_annotations, n_unknown_annotations, n_unaligned_annotations (annotations not aligned to a valid beat), truth_source (`nstdb_annotations` / `mitdb_reference` / `r_peak_fallback`), predicted_class_counts, and the metrics accuracy, macro_f1, sensitivity_macro, balanced_accuracy — computed over the beats with valid truth; metrics are absent when there is no valid truth (metrics are never invented); missing values are NaN.
- `reports/results/per_class_oof.json`: per-class OOF metrics (precision, recall, sensitivity, specificity, f1, support) per model.
- `reports/results/per_class_train_diagnostics.json`: per-class training-fit metrics (diagnostics only, not test performance).
- `reports/results/confusion_matrices.json`: OOF (`oof`) and training (`train`) confusion matrices, class order: N, S, V, F, Q.
- `reports/results/fold_class_counts.json`: per-fold class distributions — validation (`val`) and training (`train`) supports per class per fold.
- `reports/results/feature_ablation.csv`: feature-ablation results (variant, n_features, model, accuracy_mean, accuracy_std, f1_macro_mean, balanced_accuracy_mean).
- `reports/results/class_weights_study.csv`: class_weight on/off study (accuracy_mean, f1_macro_mean, balanced_accuracy_mean).
- `reports/results/hyperparameter_search.json`: per-model, per-fold best params from training-fold-only GridSearchCV.
- `data/features/ecg_features.csv`: the full extracted feature dataset for all valid segments (record, beat_idx, class, r_sample + 23 features); `rr_s`/`rr_ratio` are NaN where unavailable.
- `reports/results/nstdb_details.json`: per-NSTDB-record alignment diagnostics (n_beats, n_samples, fs, segment_lengths_head, n_annotations, n_valid, n_excluded_annotations, n_unknown_annotations, n_unaligned_annotations, truth_source, mitdb_compat when the MITDB reference was used).

**Full extracted dataset vs the class-capped modeling sample.** `data/features/ecg_features.csv` holds the full mapped-beat dataset of 109494 beats (segments of the 48 MIT-BIH recordings). The modeling sample is the class-capped design of 4803 beats — the seeded per-class sample of at most 1000 beats per AAMI class (`MAX_BEATS_PER_CLASS`, seed 42). Why the cap exists: the RBF-SVM classifier fit on the full ~112k-beat MITDB feature matrix is infeasible for the patient-independent comparison, so the five classifiers are fitted and compared on the capped, fully reproducible sample; the full matrix is retained for other uses (e.g., feature ablation, model studies). The patient-independent validation design, the out-of-fold evaluation and the reserved-record logic all operate on this modeling sample.

## 3. Preprocessing and Segmentation

### 3.1 Filtering
```
y[n] = BandpassFilter(x[n])        # 4th-order Butterworth, 0.5-40 Hz
SOS = butter(4, [0.5/180, 40/180], btype='bandpass', output='sos')
y = scipy.signal.sosfiltfilt(SOS, x, axis=0)   # forward-backward, zero phase
```
Normalization: each heartbeat segment is standardized with training-fitted statistics `norm_i = (x_i - mean_train) / (std_train + eps)`; every fold uses the scaler fitted on its training split only.
### 3.2 Segmentation
Heartbeat segments (per the specification): `X_i = x[r_i - 0.25*fs : r_i + 0.45*fs]` -- a 250 ms pre-R window and a 450 ms post-R window around each annotated R position (`r_i`). At signal boundaries segments are truncated; degenerate segments are dropped. With fs = 360 Hz this yields 252 samples (0.7 s) per segment.

## 4. Annotations and the AAMI Mapping

The MIT-BIH annotation codes were mapped to the five AAMI categories: N: N, L, R, e, j; S: A, a, J, S; V: V, E; F: F; Q: /, f, Q.
- **N**: N, L, R, e, j
- **S**: A, a, J, S
- **V**: V, E
- **F**: F
- **Q**: /, f, Q

Annotations known to the MIT-BIH database but absent from the AAMI map were excluded and reported; unknown codes (not part of the database) were reported separately: 2704 excluded, 449 unknown, by code.

## 5. Feature Engineering

Each heartbeat segment is represented by 23 features (ecg_robust.features). This matrix has 23 features (including `rr_s` and `rr_ratio` when available).
| Feature | Definition | Formula / method |
|---|---|---|
| `mean` | arithmetic mean of the segment (mV) | `x_bar = (1/n) * sum_i x_i` |
| `std` | population standard deviation of the segment (mV) | `sigma = sqrt((1/n) * sum_i (x_i - mean)^2)` |
| `min` | minimum amplitude (mV) | `min_i x_i` |
| `max` | maximum amplitude (mV) | `max_i x_i` |
| `energy` | signal energy E = sum x_i^2 (mV^2) | `E = sum_i x_i^2` |
| `peak_amplitude` | maximum absolute amplitude (mV) | `max_i |x_i|` |
| `duration_s` | waveform duration n / fs (s) | `n / fs` |
| `rise_time_s` | time from segment start to first half-maximum (s) | `first index i where x_i >= max(x[0:n/2]) / fs` |
| `fall_time_s` | time from second-half maximum crossing to segment end (s) | `(n - i2) / fs` |
| `zero_crossings` | number of sign changes in the segment | `sum_i [sign(x_i) != sign(x_{i+1})]` |
| `rr_s` | RR interval (s): RR_i = (r_i - r_{i-1}) / fs | `(r_i - r_{i-1}) / fs` |
| `rr_ratio` | RR ratio: RR_i / RR_{i-1} (dimensionless) | `RR_i / RR_{i-1}` |
| `spectral_centroid` | DFT centroid (Hz) | `sum(f * |X(f)|) / sum(|X(f)|)` |
| `spectral_flatness` | spectral flatness in [0,1] (1 = flat) | `exp(mean(ln|X|)) / mean(|X|)` |
| `fft_peak_frequency` | frequency of largest DFT magnitude (Hz) | `argmax_f |X(f)|` |
| `fft_total_energy` | DFT energy sum(|X(f)|^2) | `sum_f |X(f)|^2` |
| `skew` | sample skewness (Fisher) | `scipy.stats.skew(x)` |
| `kurtosis` | sample kurtosis (Fisher) | `scipy.stats.kurtosis(x)` |
| `p5` | 5th percentile (mV) | `np.percentile(x, 5)` |
| `p25` | 25th percentile (mV) | `np.percentile(x, 25)` |
| `p50` | median (mV) | `np.percentile(x, 50)` |
| `p75` | 75th percentile (mV) | `np.percentile(x, 75)` |
| `p95` | 95th percentile (mV) | `np.percentile(x, 95)` |

## 6. Patient-Independent Validation Design

- Splitter: `sklearn.model_selection.StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` -- stratification on the AAMI class label.
- Groups: each MITDB recording ID is a group (the 201/202 group is shared). Beats of a recording are never split across train/validation -- there is no patient leakage. The 201/202 grouping was applied: recordings 201 and 202 were assigned to the same subject group.
- Each fold refits `SimpleImputer` + `StandardScaler` and the model on the training split only (spec #4).
- Group disjointness between training and validation is verified at run time: the pipeline raises if any recording group appears in both a training and a validation fold (patient-leakage check, spec Phase 6).

## 7. Models (Phase 6 -- stronger experimental comparison)

Primary models:
- **DummyClassifier** (`strategy='most_frequent'`) -- patient-independent baseline.
- **LogisticRegression** (`max_iter=2000`, `class_weight='balanced'`, `random_state=42`).
- **RandomForestClassifier** -- `n_estimators=100`, `class_weight='balanced'`, `random_state=42`.
- **HistGradientBoostingClassifier** (`max_iter=100`, `random_state=42`; handles class imbalance natively).
- **SupportVectorMachine (RBF)** -- `C=1.0`, `kernel='rbf'`, `class_weight='balanced'`, `random_state=42`.

Class imbalance and weights: AAMI class supports in the sampled design: N=90631, S=2781, V=7236, F=803, Q=8043. `class_weight='balanced'` (w_c = n / (C * n_c)) is used for logistic regression, random forest and SVM; HistGB handles imbalance natively. A balanced-vs-unbalanced study is in `reports/results/class_weights_study.csv`.
For CV and final fits, models were wrapped in `sklearn.pipeline.Pipeline` (`SimpleImputer` + `StandardScaler` -> model); the imputer and scaler are fit on each fold's training data only (spec #4).

## 8. Metric Definitions

- Accuracy = (TP + TN) / Total.
- Precision = TP / (TP + FP); Recall (per class) = TP / (TP + FN).
- Specificity (per class) = TN / (TN + FP).
- Macro F1 = mean of the five per-class F1 = mean of 2P*R/(P+R).
- Weighted F1 = F1 averaged by class support.
- Balanced accuracy = mean of per-class recalls (sklearn).

### 9.0 Class distribution per CV fold (out-of-fold validation sets)

| Fold | N | S | V | F | Q |
|---|---|---|---|---|---|
| 0 | 192 | 506 | 204 | 14 | 250 |
| 1 | 206 | 145 | 197 | 376 | 214 |
| 2 | 199 | 74 | 205 | 20 | 282 |
| 3 | 198 | 153 | 203 | 7 | 254 |
| 4 | 205 | 122 | 191 | 386 | 0 |

Folds with missing validation classes: ['4'].


## 9. Actual 5-Fold Patient-Independent CV Results

Mean of the five folds (`reports/results/cv_metrics.csv` -- accuracy / precision(macro) / recall(macro) / sensitivity(macro) / specificity(macro) / F1(macro) / F1(weighted) / balanced accuracy):

| Model | Accuracy | Precision Macro | Recall Macro | Sensitivity Macro | Specificity Macro | F1 Macro | F1 Weighted | Balanced Accuracy |
|-------|---|---|---|---|---|---|---|---|
| dummy | 0.1149 | 0.0230 | 0.1600 | 0.1600 | 0.8000 | 0.0399 | 0.0302 | 0.1600 |
| hist_gradient_boosting | 0.6253 | 0.5924 | 0.5632 | 0.5632 | 0.9077 | 0.5303 | 0.6096 | 0.5880 |
| logistic_regression | 0.5680 | 0.5338 | 0.5301 | 0.5301 | 0.8940 | 0.4935 | 0.5711 | 0.5558 |
| random_forest | 0.5490 | 0.5784 | 0.5259 | 0.5259 | 0.8873 | 0.4834 | 0.5247 | 0.5514 |
| svm_rbf | 0.5770 | 0.5485 | 0.5315 | 0.5315 | 0.8942 | 0.4979 | 0.5601 | 0.5561 |

Best CV accuracy: **0.6253** (hist_gradient_boosting).

### 9.1 Confusion matrices (out-of-fold; sum over folds)


#### `dummy` -- out-of-fold (sum over folds)

| | N | S | V | F | Q |
|--|---|---|---|---|---|
| N | 192 | 603 | 0 | 0 | 205 |
| S | 506 | 372 | 0 | 0 | 122 |
| V | 204 | 605 | 0 | 0 | 191 |
| F | 14 | 403 | 0 | 0 | 386 |
| Q | 250 | 750 | 0 | 0 | 0 |

#### `logistic_regression` -- out-of-fold (sum over folds)

| | N | S | V | F | Q |
|--|---|---|---|---|---|
| N | 632 | 175 | 52 | 70 | 71 |
| S | 444 | 378 | 93 | 33 | 52 |
| V | 17 | 49 | 806 | 70 | 58 |
| F | 192 | 166 | 176 | 262 | 7 |
| Q | 266 | 95 | 29 | 17 | 593 |

#### `random_forest` -- out-of-fold (sum over folds)

| | N | S | V | F | Q |
|--|---|---|---|---|---|
| N | 774 | 74 | 28 | 81 | 43 |
| S | 507 | 364 | 106 | 20 | 3 |
| V | 51 | 29 | 862 | 46 | 12 |
| F | 141 | 109 | 389 | 164 | 0 |
| Q | 464 | 6 | 118 | 0 | 412 |

#### `hist_gradient_boosting` -- out-of-fold (sum over folds)

| | N | S | V | F | Q |
|--|---|---|---|---|---|
| N | 779 | 49 | 29 | 90 | 53 |
| S | 484 | 359 | 136 | 20 | 1 |
| V | 57 | 33 | 852 | 49 | 9 |
| F | 253 | 141 | 234 | 175 | 0 |
| Q | 217 | 5 | 57 | 2 | 719 |

#### `svm_rbf` -- out-of-fold (sum over folds)

| | N | S | V | F | Q |
|--|---|---|---|---|---|
| N | 708 | 126 | 47 | 65 | 54 |
| S | 459 | 376 | 124 | 16 | 25 |
| V | 53 | 39 | 818 | 73 | 17 |
| F | 335 | 75 | 190 | 200 | 3 |
| Q | 224 | 16 | 143 | 11 | 606 |

### 9.1b Training diagnostics (full-data fits, diagnostics only)

#### `dummy` -- training (full data, diagnostics)

| | N | S | V | F | Q |
|--|---|---|---|---|---|
| N | 0 | 0 | 0 | 0 | 977 |
| S | 0 | 0 | 0 | 0 | 969 |
| V | 0 | 0 | 0 | 0 | 942 |
| F | 0 | 0 | 0 | 0 | 803 |
| Q | 0 | 0 | 0 | 0 | 1,000 |

#### `logistic_regression` -- training (full data, diagnostics)

| | N | S | V | F | Q |
|--|---|---|---|---|---|
| N | 727 | 114 | 19 | 68 | 49 |
| S | 56 | 847 | 27 | 33 | 6 |
| V | 9 | 26 | 825 | 46 | 36 |
| F | 47 | 15 | 38 | 700 | 3 |
| Q | 81 | 15 | 7 | 3 | 894 |

#### `random_forest` -- training (full data, diagnostics)

| | N | S | V | F | Q |
|--|---|---|---|---|---|
| N | 977 | 0 | 0 | 0 | 0 |
| S | 0 | 969 | 0 | 0 | 0 |
| V | 0 | 0 | 942 | 0 | 0 |
| F | 0 | 0 | 0 | 803 | 0 |
| Q | 0 | 0 | 0 | 0 | 1,000 |

#### `hist_gradient_boosting` -- training (full data, diagnostics)

| | N | S | V | F | Q |
|--|---|---|---|---|---|
| N | 977 | 0 | 0 | 0 | 0 |
| S | 0 | 969 | 0 | 0 | 0 |
| V | 0 | 0 | 942 | 0 | 0 |
| F | 0 | 0 | 0 | 803 | 0 |
| Q | 0 | 0 | 0 | 0 | 1,000 |

#### `svm_rbf` -- training (full data, diagnostics)

| | N | S | V | F | Q |
|--|---|---|---|---|---|
| N | 901 | 31 | 8 | 35 | 2 |
| S | 52 | 903 | 5 | 9 | 0 |
| V | 3 | 16 | 895 | 24 | 4 |
| F | 38 | 5 | 26 | 734 | 0 |
| Q | 37 | 3 | 0 | 0 | 960 |

### 9.2 Per-class performance of the best evaluated model (`hist_gradient_boosting`) -- out-of-fold
Per-class performance from out-of-fold predictions (spec #2); computed via `per_class_metrics_from_cm` on the summed OOF confusion matrix, agreeing with the OOF confusion table in 9.1. Saved to `reports/results/per_class_oof.json`.

| Class | Precision | Recall | Sensitivity | Specificity | F1 | Support |
|---|---|---|---|---|---|---|
| N | 0.4352 | 0.7790 | 0.7790 | 0.7342 | 0.5584 | 1000 |
| S | 0.6116 | 0.3590 | 0.3590 | 0.9400 | 0.4524 | 1000 |
| V | 0.6514 | 0.8520 | 0.8520 | 0.8801 | 0.7383 | 1000 |
| F | 0.5208 | 0.2179 | 0.2179 | 0.9597 | 0.3073 | 803 |
| Q | 0.9194 | 0.7190 | 0.7190 | 0.9834 | 0.8070 | 1000 |


### 9.2b Training diagnostics per class (full-data fits)
Per-class performance from full-data training fits (`reports/results/per_class_train_diagnostics.json`). **These are training diagnostics only — they must not be reported as test performance.**

| Model | Class | Precision | Recall | Sensitivity | Specificity | F1 | Support |
|---|---|---|---|---|---|---|---|
| dummy | N | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 977 |
| dummy | S | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 969 |
| dummy | V | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 942 |
| dummy | F | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 803 |
| dummy | Q | 0.2132 | 1.0000 | 1.0000 | 0.0000 | 0.3514 | 1000 |
| hist_gradient_boosting | N | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 977 |
| hist_gradient_boosting | S | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 969 |
| hist_gradient_boosting | V | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 942 |
| hist_gradient_boosting | F | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 803 |
| hist_gradient_boosting | Q | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1000 |
| logistic_regression | N | 0.7902 | 0.7441 | 0.7441 | 0.9480 | 0.7665 | 977 |
| logistic_regression | S | 0.8328 | 0.8741 | 0.8741 | 0.9543 | 0.8530 | 969 |
| logistic_regression | V | 0.9007 | 0.8758 | 0.8758 | 0.9757 | 0.8881 | 942 |
| logistic_regression | F | 0.8235 | 0.8717 | 0.8717 | 0.9614 | 0.8469 | 803 |
| logistic_regression | Q | 0.9049 | 0.8940 | 0.8940 | 0.9745 | 0.8994 | 1000 |
| random_forest | N | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 977 |
| random_forest | S | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 969 |
| random_forest | V | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 942 |
| random_forest | F | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 803 |
| random_forest | Q | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1000 |
| svm_rbf | N | 0.8739 | 0.9222 | 0.9222 | 0.9650 | 0.8974 | 977 |
| svm_rbf | S | 0.9426 | 0.9319 | 0.9319 | 0.9852 | 0.9372 | 969 |
| svm_rbf | V | 0.9582 | 0.9501 | 0.9501 | 0.9896 | 0.9542 | 942 |
| svm_rbf | F | 0.9152 | 0.9141 | 0.9141 | 0.9825 | 0.9146 | 803 |
| svm_rbf | Q | 0.9938 | 0.9600 | 0.9600 | 0.9984 | 0.9766 | 1000 |

## 10. Noise Robustness Experiment

Setup (per the specification): clean reference = MITDB beats. Noisy evaluation = synthetic mixtures of the clean beats with NSTDB noise realizations (NSTDB record segments) at the target SNR levels `SNR(dB) = 10*log10(P_signal / P_noise)`, where P_signal/P_noise are the mean squared values; clean and noisy segments are aligned at identical positions. The NSTDB recordings (incl. the reserved records 118/119) were also classified directly as recorded-noise evaluation (10.2). These are labeled synthetic-mixture results; the recorded-noise experiment below uses real NSTDB noise.

### 10.1 Out-of-fold synthetic mixtures -- accuracy vs SNR (per-fold models scored on held-out validation beats; spec #3)
Fairness (spec #3): the noisy validation feature matrices are generated once per fold/SNR by `generate_noisy_features`, and the same noisy samples are scored by every model in the fold — `n_beats` is identical across models per fold/SNR (see `reports/results/noise_results.csv` and the fairness regression tests). The noise-realization rng seed and the per-SNR achieved SNR are recorded in `reports/results/noise_config.json`.

| SNR (dB) | Model | Accuracy | Macro F1 | Macro Sensitivity | Balanced Accuracy |
|---|---|---|---|---|---|
| -6 | dummy | 0.2144 | 0.0706 | 0.2000 | 0.2000 |
| -6 | dummy | 0.1880 | 0.0633 | 0.2000 | 0.2000 |
| -6 | dummy | 0.3615 | 0.1062 | 0.2000 | 0.2000 |
| -6 | dummy | 0.3117 | 0.0950 | 0.2000 | 0.2000 |
| -6 | dummy | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| -6 | hist_gradient_boosting | 0.1115 | 0.0904 | 0.2312 | 0.2312 |
| -6 | hist_gradient_boosting | 0.1872 | 0.1182 | 0.1985 | 0.1985 |
| -6 | hist_gradient_boosting | 0.2346 | 0.1310 | 0.2211 | 0.2211 |
| -6 | hist_gradient_boosting | 0.2160 | 0.1020 | 0.1772 | 0.1772 |
| -6 | hist_gradient_boosting | 0.2954 | 0.1794 | 0.2358 | 0.2947 |
| -6 | logistic_regression | 0.2084 | 0.1514 | 0.3400 | 0.3400 |
| -6 | logistic_regression | 0.3515 | 0.1931 | 0.2411 | 0.2411 |
| -6 | logistic_regression | 0.3423 | 0.2185 | 0.3152 | 0.3152 |
| -6 | logistic_regression | 0.0638 | 0.0637 | 0.0516 | 0.0516 |
| -6 | logistic_regression | 0.3794 | 0.2192 | 0.2118 | 0.2647 |
| -6 | random_forest | 0.1852 | 0.1039 | 0.2195 | 0.2195 |
| -6 | random_forest | 0.1986 | 0.1275 | 0.2148 | 0.2148 |
| -6 | random_forest | 0.2769 | 0.1616 | 0.2499 | 0.2499 |
| -6 | random_forest | 0.2626 | 0.1330 | 0.2387 | 0.2387 |
| -6 | random_forest | 0.3108 | 0.2108 | 0.2599 | 0.3249 |
| -6 | svm_rbf | 0.2144 | 0.1311 | 0.2415 | 0.2415 |
| -6 | svm_rbf | 0.1766 | 0.1052 | 0.2015 | 0.2015 |
| -6 | svm_rbf | 0.3449 | 0.1815 | 0.2633 | 0.2633 |
| -6 | svm_rbf | 0.2920 | 0.1606 | 0.2630 | 0.2630 |
| -6 | svm_rbf | 0.3274 | 0.2067 | 0.3044 | 0.3805 |
| 0 | dummy | 0.2144 | 0.0706 | 0.2000 | 0.2000 |
| 0 | dummy | 0.1880 | 0.0633 | 0.2000 | 0.2000 |
| 0 | dummy | 0.3615 | 0.1062 | 0.2000 | 0.2000 |
| 0 | dummy | 0.3117 | 0.0950 | 0.2000 | 0.2000 |
| 0 | dummy | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 0 | hist_gradient_boosting | 0.4657 | 0.3074 | 0.3871 | 0.3871 |
| 0 | hist_gradient_boosting | 0.2794 | 0.2303 | 0.2846 | 0.2846 |
| 0 | hist_gradient_boosting | 0.2962 | 0.2026 | 0.2715 | 0.2715 |
| 0 | hist_gradient_boosting | 0.2798 | 0.1598 | 0.2286 | 0.2286 |
| 0 | hist_gradient_boosting | 0.4248 | 0.2981 | 0.3498 | 0.4373 |
| 0 | logistic_regression | 0.4099 | 0.3215 | 0.3918 | 0.3918 |
| 0 | logistic_regression | 0.4763 | 0.3438 | 0.3927 | 0.3927 |
| 0 | logistic_regression | 0.4077 | 0.2726 | 0.3444 | 0.3444 |
| 0 | logistic_regression | 0.3227 | 0.2124 | 0.2286 | 0.2286 |
| 0 | logistic_regression | 0.3352 | 0.2649 | 0.2254 | 0.2817 |
| 0 | random_forest | 0.5043 | 0.3220 | 0.4005 | 0.4005 |
| 0 | random_forest | 0.3120 | 0.2690 | 0.3175 | 0.3175 |
| 0 | random_forest | 0.4077 | 0.2888 | 0.3758 | 0.3758 |
| 0 | random_forest | 0.3301 | 0.1891 | 0.2675 | 0.2675 |
| 0 | random_forest | 0.4723 | 0.3475 | 0.3765 | 0.4706 |
| 0 | svm_rbf | 0.5617 | 0.4353 | 0.4575 | 0.4575 |
| 0 | svm_rbf | 0.3216 | 0.2742 | 0.3332 | 0.3332 |
| 0 | svm_rbf | 0.5436 | 0.4077 | 0.4272 | 0.4272 |
| 0 | svm_rbf | 0.3387 | 0.2095 | 0.2738 | 0.2738 |
| 0 | svm_rbf | 0.3684 | 0.2650 | 0.3230 | 0.4037 |
| 6 | dummy | 0.2144 | 0.0706 | 0.2000 | 0.2000 |
| 6 | dummy | 0.1880 | 0.0633 | 0.2000 | 0.2000 |
| 6 | dummy | 0.3615 | 0.1062 | 0.2000 | 0.2000 |
| 6 | dummy | 0.3117 | 0.0950 | 0.2000 | 0.2000 |
| 6 | dummy | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 6 | hist_gradient_boosting | 0.6398 | 0.4242 | 0.4981 | 0.4981 |
| 6 | hist_gradient_boosting | 0.4165 | 0.3520 | 0.4015 | 0.4015 |
| 6 | hist_gradient_boosting | 0.3795 | 0.3131 | 0.3707 | 0.3707 |
| 6 | hist_gradient_boosting | 0.3325 | 0.2039 | 0.2988 | 0.2988 |
| 6 | hist_gradient_boosting | 0.4259 | 0.2932 | 0.3685 | 0.4606 |
| 6 | logistic_regression | 0.6895 | 0.5048 | 0.5174 | 0.5174 |
| 6 | logistic_regression | 0.6125 | 0.5154 | 0.5345 | 0.5345 |
| 6 | logistic_regression | 0.5346 | 0.4440 | 0.4617 | 0.4617 |
| 6 | logistic_regression | 0.4982 | 0.3246 | 0.3588 | 0.3588 |
| 6 | logistic_regression | 0.3462 | 0.2992 | 0.2885 | 0.3606 |
| 6 | random_forest | 0.7196 | 0.5337 | 0.5625 | 0.5625 |
| 6 | random_forest | 0.5062 | 0.4383 | 0.4703 | 0.4703 |
| 6 | random_forest | 0.4833 | 0.4083 | 0.4593 | 0.4593 |
| 6 | random_forest | 0.4258 | 0.2922 | 0.3690 | 0.3690 |
| 6 | random_forest | 0.4978 | 0.3500 | 0.4122 | 0.5153 |
| 6 | svm_rbf | 0.7170 | 0.5507 | 0.5565 | 0.5565 |
| 6 | svm_rbf | 0.6845 | 0.6432 | 0.6379 | 0.6379 |
| 6 | svm_rbf | 0.5538 | 0.4483 | 0.4603 | 0.4603 |
| 6 | svm_rbf | 0.4883 | 0.3657 | 0.3754 | 0.3754 |
| 6 | svm_rbf | 0.4004 | 0.2836 | 0.3590 | 0.4487 |
| 12 | dummy | 0.2144 | 0.0706 | 0.2000 | 0.2000 |
| 12 | dummy | 0.1880 | 0.0633 | 0.2000 | 0.2000 |
| 12 | dummy | 0.3615 | 0.1062 | 0.2000 | 0.2000 |
| 12 | dummy | 0.3117 | 0.0950 | 0.2000 | 0.2000 |
| 12 | dummy | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 12 | hist_gradient_boosting | 0.6647 | 0.4503 | 0.5192 | 0.5192 |
| 12 | hist_gradient_boosting | 0.5351 | 0.4572 | 0.5087 | 0.5087 |
| 12 | hist_gradient_boosting | 0.4282 | 0.3943 | 0.4622 | 0.4622 |
| 12 | hist_gradient_boosting | 0.3472 | 0.2153 | 0.3387 | 0.3387 |
| 12 | hist_gradient_boosting | 0.3949 | 0.2697 | 0.3493 | 0.4366 |
| 12 | logistic_regression | 0.7350 | 0.5657 | 0.5570 | 0.5570 |
| 12 | logistic_regression | 0.6968 | 0.6388 | 0.6321 | 0.6321 |
| 12 | logistic_regression | 0.5667 | 0.4999 | 0.4927 | 0.4927 |
| 12 | logistic_regression | 0.5399 | 0.3586 | 0.3901 | 0.3901 |
| 12 | logistic_regression | 0.4126 | 0.3411 | 0.3584 | 0.4480 |
| 12 | random_forest | 0.7659 | 0.5903 | 0.5968 | 0.5968 |
| 12 | random_forest | 0.6037 | 0.5227 | 0.5628 | 0.5628 |
| 12 | random_forest | 0.5577 | 0.5042 | 0.5144 | 0.5144 |
| 12 | random_forest | 0.5018 | 0.3504 | 0.4204 | 0.4204 |
| 12 | random_forest | 0.4646 | 0.3263 | 0.4003 | 0.5004 |
| 12 | svm_rbf | 0.7650 | 0.6018 | 0.6111 | 0.6111 |
| 12 | svm_rbf | 0.7926 | 0.7561 | 0.7422 | 0.7422 |
| 12 | svm_rbf | 0.6256 | 0.5608 | 0.5623 | 0.5623 |
| 12 | svm_rbf | 0.6663 | 0.4994 | 0.5296 | 0.5296 |
| 12 | svm_rbf | 0.3894 | 0.2713 | 0.3509 | 0.4386 |
| 18 | dummy | 0.2144 | 0.0706 | 0.2000 | 0.2000 |
| 18 | dummy | 0.1880 | 0.0633 | 0.2000 | 0.2000 |
| 18 | dummy | 0.3615 | 0.1062 | 0.2000 | 0.2000 |
| 18 | dummy | 0.3117 | 0.0950 | 0.2000 | 0.2000 |
| 18 | dummy | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 18 | hist_gradient_boosting | 0.6973 | 0.4921 | 0.5801 | 0.5801 |
| 18 | hist_gradient_boosting | 0.6054 | 0.5063 | 0.5671 | 0.5671 |
| 18 | hist_gradient_boosting | 0.4436 | 0.4030 | 0.4452 | 0.4452 |
| 18 | hist_gradient_boosting | 0.3472 | 0.2113 | 0.3112 | 0.3112 |
| 18 | hist_gradient_boosting | 0.3794 | 0.2543 | 0.3403 | 0.4254 |
| 18 | logistic_regression | 0.7419 | 0.5814 | 0.5671 | 0.5671 |
| 18 | logistic_regression | 0.7100 | 0.6509 | 0.6441 | 0.6441 |
| 18 | logistic_regression | 0.5231 | 0.4575 | 0.4408 | 0.4408 |
| 18 | logistic_regression | 0.5558 | 0.3816 | 0.4025 | 0.4025 |
| 18 | logistic_regression | 0.4469 | 0.3649 | 0.3942 | 0.4927 |
| 18 | random_forest | 0.8105 | 0.6786 | 0.6996 | 0.6996 |
| 18 | random_forest | 0.6547 | 0.5574 | 0.6057 | 0.6057 |
| 18 | random_forest | 0.5795 | 0.5277 | 0.5136 | 0.5136 |
| 18 | random_forest | 0.5141 | 0.3538 | 0.4343 | 0.4343 |
| 18 | random_forest | 0.4159 | 0.2881 | 0.3613 | 0.4516 |
| 18 | svm_rbf | 0.7753 | 0.6167 | 0.6239 | 0.6239 |
| 18 | svm_rbf | 0.8251 | 0.7935 | 0.7793 | 0.7793 |
| 18 | svm_rbf | 0.6192 | 0.5675 | 0.5496 | 0.5496 |
| 18 | svm_rbf | 0.7104 | 0.5193 | 0.5322 | 0.5322 |
| 18 | svm_rbf | 0.3927 | 0.2795 | 0.3596 | 0.4495 |
| 24 | dummy | 0.2144 | 0.0706 | 0.2000 | 0.2000 |
| 24 | dummy | 0.1880 | 0.0633 | 0.2000 | 0.2000 |
| 24 | dummy | 0.3615 | 0.1062 | 0.2000 | 0.2000 |
| 24 | dummy | 0.3117 | 0.0950 | 0.2000 | 0.2000 |
| 24 | dummy | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 24 | hist_gradient_boosting | 0.7093 | 0.5107 | 0.6055 | 0.6055 |
| 24 | hist_gradient_boosting | 0.6142 | 0.5182 | 0.5783 | 0.5783 |
| 24 | hist_gradient_boosting | 0.4500 | 0.4138 | 0.4536 | 0.4536 |
| 24 | hist_gradient_boosting | 0.3521 | 0.2472 | 0.3428 | 0.3428 |
| 24 | hist_gradient_boosting | 0.3971 | 0.2726 | 0.3524 | 0.4404 |
| 24 | logistic_regression | 0.7590 | 0.6069 | 0.5992 | 0.5992 |
| 24 | logistic_regression | 0.7127 | 0.6560 | 0.6482 | 0.6482 |
| 24 | logistic_regression | 0.5244 | 0.4606 | 0.4460 | 0.4460 |
| 24 | logistic_regression | 0.5632 | 0.4004 | 0.4098 | 0.4098 |
| 24 | logistic_regression | 0.4502 | 0.3635 | 0.3973 | 0.4966 |
| 24 | random_forest | 0.8328 | 0.7004 | 0.7216 | 0.7216 |
| 24 | random_forest | 0.6582 | 0.5593 | 0.6080 | 0.6080 |
| 24 | random_forest | 0.6436 | 0.5719 | 0.5528 | 0.5528 |
| 24 | random_forest | 0.5374 | 0.3805 | 0.4793 | 0.4793 |
| 24 | random_forest | 0.4325 | 0.3085 | 0.3713 | 0.4642 |
| 24 | svm_rbf | 0.7959 | 0.6492 | 0.6578 | 0.6578 |
| 24 | svm_rbf | 0.8348 | 0.8023 | 0.7899 | 0.7899 |
| 24 | svm_rbf | 0.6628 | 0.5630 | 0.5516 | 0.5516 |
| 24 | svm_rbf | 0.7374 | 0.5445 | 0.5550 | 0.5550 |
| 24 | svm_rbf | 0.3838 | 0.2665 | 0.3503 | 0.4379 |

### 10.2 Recorded-noise (NSTDB) classification per model -- mean metrics

Metric population for the recorded-noise table (10.2): accuracy, macro_f1, sensitivity_macro and balanced_accuracy are computed over the beats with valid reference truth (`n_valid_truth`; the metrics columns are absent when `n_valid_truth == 0` — predictions are still reported). Metrics are never invented: where ground truth cannot be established (`truth_source == "r_peak_fallback"`) the table reports predictions and descriptive statistics only.

| Model | Accuracy | Macro F1 | Sensitivity Macro | Balanced Accuracy |
|---|---|---|---|---|
| dummy | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| hist_gradient_boosting | 0.6149 | 0.2517 | 0.2950 | 0.6449 |
| logistic_regression | 0.5023 | 0.2447 | 0.2462 | 0.5552 |
| random_forest | 0.5012 | 0.2074 | 0.2527 | 0.5682 |
| svm_rbf | 0.5089 | 0.2043 | 0.2516 | 0.5669 |

Per-record predicted AAMI class counts (NSTDB; metrics where truth is available, per the table above):

- `118e00` [dummy]: predicted N=0, S=0, V=0, F=0, Q=2278
- `118e00` [hist_gradient_boosting]: predicted N=769, S=5, V=207, F=23, Q=1274
- `118e00` [logistic_regression]: predicted N=327, S=99, V=210, F=458, Q=1184
- `118e00` [random_forest]: predicted N=328, S=7, V=725, F=31, Q=1187
- `118e00` [svm_rbf]: predicted N=296, S=7, V=1028, F=71, Q=876
- `118e06` [dummy]: predicted N=0, S=0, V=0, F=0, Q=2278
- `118e06` [hist_gradient_boosting]: predicted N=732, S=5, V=213, F=26, Q=1302
- `118e06` [logistic_regression]: predicted N=389, S=71, V=188, F=243, Q=1387
- `118e06` [random_forest]: predicted N=351, S=7, V=594, F=46, Q=1280
- `118e06` [svm_rbf]: predicted N=360, S=6, V=835, F=102, Q=975
- `118e12` [dummy]: predicted N=0, S=0, V=0, F=0, Q=2278
- `118e12` [hist_gradient_boosting]: predicted N=747, S=6, V=176, F=14, Q=1335
- `118e12` [logistic_regression]: predicted N=375, S=73, V=122, F=86, Q=1622
- `118e12` [random_forest]: predicted N=418, S=7, V=441, F=41, Q=1371
- `118e12` [svm_rbf]: predicted N=503, S=6, V=424, F=144, Q=1201
- `118e18` [dummy]: predicted N=0, S=0, V=0, F=0, Q=2278
- `118e18` [hist_gradient_boosting]: predicted N=846, S=6, V=136, F=8, Q=1282
- `118e18` [logistic_regression]: predicted N=308, S=76, V=57, F=43, Q=1794
- `118e18` [random_forest]: predicted N=506, S=8, V=355, F=31, Q=1378
- `118e18` [svm_rbf]: predicted N=636, S=5, V=191, F=114, Q=1332
- `118e24` [dummy]: predicted N=0, S=0, V=0, F=0, Q=2278
- `118e24` [hist_gradient_boosting]: predicted N=898, S=6, V=126, F=6, Q=1242
- `118e24` [logistic_regression]: predicted N=240, S=81, V=42, F=35, Q=1880
- `118e24` [random_forest]: predicted N=583, S=7, V=323, F=25, Q=1340
- `118e24` [svm_rbf]: predicted N=613, S=5, V=142, F=134, Q=1384
- `118e_6` [dummy]: predicted N=0, S=0, V=0, F=0, Q=2278
- `118e_6` [hist_gradient_boosting]: predicted N=805, S=5, V=189, F=13, Q=1266
- `118e_6` [logistic_regression]: predicted N=221, S=245, V=199, F=508, Q=1105
- `118e_6` [random_forest]: predicted N=323, S=6, V=805, F=21, Q=1123
- `118e_6` [svm_rbf]: predicted N=291, S=5, V=1071, F=54, Q=857
- `119e00` [dummy]: predicted N=0, S=0, V=0, F=0, Q=1987
- `119e00` [hist_gradient_boosting]: predicted N=1066, S=0, V=603, F=5, Q=313
- `119e00` [logistic_regression]: predicted N=1102, S=24, V=419, F=374, Q=68
- `119e00` [random_forest]: predicted N=874, S=0, V=839, F=2, Q=272
- `119e00` [svm_rbf]: predicted N=868, S=1, V=1089, F=12, Q=17
- `119e06` [dummy]: predicted N=0, S=0, V=0, F=0, Q=1987
- `119e06` [hist_gradient_boosting]: predicted N=1165, S=0, V=596, F=8, Q=218
- `119e06` [logistic_regression]: predicted N=1260, S=2, V=452, F=191, Q=82
- `119e06` [random_forest]: predicted N=1017, S=0, V=664, F=6, Q=300
- `119e06` [svm_rbf]: predicted N=1032, S=1, V=847, F=68, Q=39
- `119e12` [dummy]: predicted N=0, S=0, V=0, F=0, Q=1987
- `119e12` [hist_gradient_boosting]: predicted N=1391, S=0, V=495, F=3, Q=98
- `119e12` [logistic_regression]: predicted N=1423, S=1, V=449, F=41, Q=73
- `119e12` [random_forest]: predicted N=1276, S=0, V=505, F=7, Q=199
- `119e12` [svm_rbf]: predicted N=1330, S=0, V=531, F=88, Q=38
- `119e18` [dummy]: predicted N=0, S=0, V=0, F=0, Q=1987
- `119e18` [hist_gradient_boosting]: predicted N=1525, S=0, V=447, F=1, Q=14
- `119e18` [logistic_regression]: predicted N=1518, S=1, V=444, F=4, Q=20
- `119e18` [random_forest]: predicted N=1501, S=1, V=452, F=2, Q=31
- `119e18` [svm_rbf]: predicted N=1517, S=0, V=446, F=19, Q=5
- `119e24` [dummy]: predicted N=0, S=0, V=0, F=0, Q=1987
- `119e24` [hist_gradient_boosting]: predicted N=1542, S=0, V=444, F=0, Q=1
- `119e24` [logistic_regression]: predicted N=1541, S=1, V=444, F=1, Q=0
- `119e24` [random_forest]: predicted N=1539, S=0, V=445, F=3, Q=0
- `119e24` [svm_rbf]: predicted N=1537, S=0, V=446, F=4, Q=0
- `119e_6` [dummy]: predicted N=0, S=0, V=0, F=0, Q=1987
- `119e_6` [hist_gradient_boosting]: predicted N=1080, S=0, V=570, F=3, Q=334
- `119e_6` [logistic_regression]: predicted N=987, S=105, V=422, F=437, Q=36
- `119e_6` [random_forest]: predicted N=860, S=0, V=900, F=0, Q=227
- `119e_6` [svm_rbf]: predicted N=853, S=0, V=1131, F=1, Q=2
- `bw` [dummy]: predicted N=0, S=0, V=0, F=0, Q=0
- `bw` [hist_gradient_boosting]: predicted N=0, S=0, V=0, F=0, Q=0
- `bw` [logistic_regression]: predicted N=0, S=0, V=0, F=0, Q=0
- `bw` [random_forest]: predicted N=0, S=0, V=0, F=0, Q=0
- `bw` [svm_rbf]: predicted N=0, S=0, V=0, F=0, Q=0
- `em` [dummy]: predicted N=0, S=0, V=0, F=0, Q=0
- `em` [hist_gradient_boosting]: predicted N=0, S=0, V=0, F=0, Q=0
- `em` [logistic_regression]: predicted N=0, S=0, V=0, F=0, Q=0
- `em` [random_forest]: predicted N=0, S=0, V=0, F=0, Q=0
- `em` [svm_rbf]: predicted N=0, S=0, V=0, F=0, Q=0
- `ma` [dummy]: predicted N=0, S=0, V=0, F=0, Q=0
- `ma` [hist_gradient_boosting]: predicted N=0, S=0, V=0, F=0, Q=0
- `ma` [logistic_regression]: predicted N=0, S=0, V=0, F=0, Q=0
- `ma` [random_forest]: predicted N=0, S=0, V=0, F=0, Q=0
- `ma` [svm_rbf]: predicted N=0, S=0, V=0, F=0, Q=0

Note: the final models used for the recorded-noise evaluation were trained on the full data excluding the reserved records (118, 119); see 9.1b.

### 10.3 Figures: `snr_vs_macro_f1.png`, `snr_vs_sensitivity.png`, `snr_vs_balanced_accuracy.png`, `clean_vs_noisy.png`

## 14. Feature Selection (Phase 6)

Feature-ablation experiments (patient-independent CV; same folds as the pipeline; `reports/results/feature_ablation.csv` + figure `reports/figures/feature_ablation.png`):

| Variant | n_features | Model | Accuracy mean | Accuracy std | F1 macro mean | Balanced accuracy mean |
|---|---|---|---|---|---|---|
| drop_morphology | 20 | hist_gradient_boosting | 0.6144 | 0.1831 | 0.5220 | 0.5823 |
| drop_morphology | 20 | logistic_regression | 0.5597 | 0.1144 | 0.4862 | 0.5497 |
| drop_morphology | 20 | random_forest | 0.5207 | 0.1114 | 0.4529 | 0.5253 |
| drop_morphology | 20 | svm_rbf | 0.5185 | 0.0979 | 0.4474 | 0.5032 |
| drop_morphology | 20 | dummy | 0.1149 | 0.0656 | 0.0399 | 0.1600 |
| drop_rr_features | 21 | logistic_regression | 0.4141 | 0.1055 | 0.3551 | 0.4151 |
| drop_rr_features | 21 | svm_rbf | 0.3986 | 0.0926 | 0.3279 | 0.3938 |
| drop_rr_features | 21 | hist_gradient_boosting | 0.3927 | 0.1114 | 0.3029 | 0.3809 |
| drop_rr_features | 21 | random_forest | 0.3489 | 0.0888 | 0.2712 | 0.3677 |
| drop_rr_features | 21 | dummy | 0.1149 | 0.0656 | 0.0399 | 0.1600 |
| drop_spectral | 19 | hist_gradient_boosting | 0.6162 | 0.1763 | 0.5191 | 0.5775 |
| drop_spectral | 19 | random_forest | 0.5481 | 0.1276 | 0.4795 | 0.5515 |
| drop_spectral | 19 | svm_rbf | 0.5394 | 0.0929 | 0.4629 | 0.5269 |
| drop_spectral | 19 | logistic_regression | 0.5081 | 0.0781 | 0.4458 | 0.5232 |
| drop_spectral | 19 | dummy | 0.1149 | 0.0656 | 0.0399 | 0.1600 |
| drop_stats | 16 | hist_gradient_boosting | 0.6540 | 0.1550 | 0.5584 | 0.6066 |
| drop_stats | 16 | svm_rbf | 0.6177 | 0.1378 | 0.5277 | 0.5751 |
| drop_stats | 16 | random_forest | 0.6170 | 0.1076 | 0.5355 | 0.5988 |
| drop_stats | 16 | logistic_regression | 0.5905 | 0.0994 | 0.5186 | 0.5706 |
| drop_stats | 16 | dummy | 0.1149 | 0.0656 | 0.0399 | 0.1600 |
| full | 23 | hist_gradient_boosting | 0.6253 | 0.1794 | 0.5303 | 0.5880 |
| full | 23 | svm_rbf | 0.5770 | 0.1042 | 0.4979 | 0.5561 |
| full | 23 | logistic_regression | 0.5680 | 0.0934 | 0.4935 | 0.5558 |
| full | 23 | random_forest | 0.5490 | 0.1077 | 0.4834 | 0.5514 |
| full | 23 | dummy | 0.1149 | 0.0656 | 0.0399 | 0.1600 |
| top12_by_rf_importance | 12 | hist_gradient_boosting | 0.6037 | 0.1567 | 0.5159 | 0.5858 |
| top12_by_rf_importance | 12 | random_forest | 0.5536 | 0.1026 | 0.4776 | 0.5524 |
| top12_by_rf_importance | 12 | svm_rbf | 0.5347 | 0.0901 | 0.4579 | 0.5229 |
| top12_by_rf_importance | 12 | logistic_regression | 0.4760 | 0.0794 | 0.4300 | 0.5063 |
| top12_by_rf_importance | 12 | dummy | 0.1149 | 0.0656 | 0.0399 | 0.1600 |

Hyperparameter search (training folds only; `reports/results/hyperparameter_search.json`):

| Model | Fold | Best params | Best inner-CV accuracy |
|---|---|---|---|

## 11. Figures

- `reports\figures\ecg_original.png`
- `reports\figures\ecg_filtered.png`
- `reports\figures\ecg_segmentation.png`
- `reports\figures\class_distribution.png`
- `reports\figures\ecg_classes_comparison.png`
- `reports\figures\confusion_matrix_dummy.png`
- `reports\figures\confusion_matrix_logistic_regression.png`
- `reports\figures\confusion_matrix_random_forest.png`
- `reports\figures\confusion_matrix_hist_gradient_boosting.png`
- `reports\figures\confusion_matrix_svm_rbf.png`
- `reports\figures\model_performance.png`
- `reports\figures\clean_vs_noisy.png`
- `reports\figures\snr_vs_macro_f1.png`
- `reports\figures\snr_vs_sensitivity.png`
- `reports\figures\snr_vs_balanced_accuracy.png`
- `feature_ablation.png`

## 12. Interpretation

Plausible reading of the measured results (interpretation of actual numbers above):
- Patient independence: grouping all beats of a recording and using StratifiedGroupKFold yielded no patient leakage; the honest estimate of generalization to unseen patients is the cross-fold mean, not the training fit.
- Mean 5-fold accuracy is 0.63 for the best model -- weak; the feature set or segmentation window may not carry enough signal, or the task (5-class AAMI on single beats) is intrinsically hard.
- The noise experiment (to be filled from `reports/results/noise_results.csv`) documents how performance decays as SNR decreases; the NSTDB records 118/119 were reserved and classified with real noise in the recorded-noise evaluation.

## 13. Limitations (fixed, per the project specification)

- Historical data: MITDB is a widely used database from the era of older leads/techniques; classification performance must not be claimed as clinical diagnosis.
- Class imbalance: AAMI N is the majority class; we report macro and balanced metrics, not accuracy alone, and use class weights.
- Noise-cohort size: NSTDB provides two noise-stress records (118, 119); the synthetic mixture experiment adds 8 real NSTDB noise realizations for the SNR sweep. Results are descriptive, not a statistically powered comparison of official noise.
- Not a clinical diagnostic system: this is a research/benchmarking tool. Beat-level AAMI classification is research-grade; the outputs must not be used for clinical decisions.
- Recorded-noise evaluation (10.2): metrics are reported only where valid reference truth exists (`n_valid_truth > 0`); where ground truth cannot be established the table reports predictions and descriptive statistics without fabricating accuracy or F1.
- Features are extracted from a 0.7 s window (0.25 s pre / 0.45 s post R); alternative windows were not tested (scope decision).

## 14. References

- PhysioNet MIT-BIH Arrhythmia Database -- https://physionet.org/content/mitdb/1.0.0/
- PhysioNet MIT-BIH Noise Stress Test Database -- https://physionet.org/content/nstdb/1.0.0/
- AAMI EC57:2012 -- Cardiac monitor inappropriate alert performance standard (AAMI EC57).

*Reproduction:* `python scripts/reproduce_pipeline.py` (downloads PhysioNet databases, runs the full pipeline, and regenerates this report with the actual results; requires `wfdb`, `numpy`, `scipy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `pytest`, `pyyaml`, `joblib`.)
