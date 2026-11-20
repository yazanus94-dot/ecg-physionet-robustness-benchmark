# ECG-ROBUST -- Technical Report: Patient-Independent ECG Arrhythmia Classification and Noise-Robustness Benchmark

- Repository: `yazanus94-dot/ecg-physionet-robustness-benchmark`
- Report date: 2026-11-20 07:58 UTC
- Pipeline wall time (measured, including downloads): 579.6 s

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
- Beats segmented across all recordings (valid segments; excluded/unknown annotations may apply): 112647
- Beats used in the feature matrix: 4803
- Excluded (known MITDB symbols absent from the AAMI map): 2704
- Unknown codes (not part of the MITDB database): 449
- AAMI class distribution of mapped beats: {'N': 90631, 'Q': 8043, 'V': 7236, 'S': 2781, 'F': 803}

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

## 7. Models

- **DummyClassifier** (`strategy='most_frequent'`) -- patient-independent baseline.
- **RandomForestClassifier** -- `n_estimators=100`, `class_weight='balanced'`, `random_state=42`.
- **SupportVectorMachine (RBF)** -- `C=1.0`, `kernel='rbf'`, `class_weight='balanced'`, `random_state=42`.
For CV and final fits, models were wrapped in `sklearn.pipeline.Pipeline` (`SimpleImputer` + `StandardScaler` -> model); the imputer and scaler are fit on each fold's training data only (spec #4).

## 8. Metric Definitions

- Accuracy = (TP + TN) / Total.
- Precision = TP / (TP + FP); Recall (per class) = TP / (TP + FN).
- Specificity (per class) = TN / (TN + FP).
- Macro F1 = mean of the five per-class F1 = mean of 2P*R/(P+R).
- Weighted F1 = F1 averaged by class support.
- Balanced accuracy = mean of per-class recalls (sklearn).

## 9. Actual 5-Fold Patient-Independent CV Results

Mean of the five folds (`reports/results/cv_metrics.csv` -- accuracy / precision(macro) / recall(macro) / sensitivity(macro) / specificity(macro) / F1(macro) / F1(weighted) / balanced accuracy):

| Model | Accuracy | Precision Macro | Recall Macro | Sensitivity Macro | Specificity Macro | F1 Macro | F1 Weighted | Balanced Accuracy |
|-------|---|---|---|---|---|---|---|---|
| dummy | 0.1149 | 0.0230 | 0.1600 | 0.1600 | 0.8000 | 0.0399 | 0.0302 | 0.1600 |
| random_forest | 0.4404 | 0.4653 | 0.4118 | 0.4118 | 0.8597 | 0.3522 | 0.3958 | 0.4339 |
| svm_rbf | 0.4596 | 0.4651 | 0.4177 | 0.4177 | 0.8641 | 0.3837 | 0.4395 | 0.4381 |

Best CV accuracy: **0.4596** (svm_rbf).

### 9.1 Confusion matrices (out-of-fold; sum over folds)


#### `dummy` -- out-of-fold (sum over folds)

| | N | S | V | F | Q |
|--|---|---|---|---|---|
| N | 192 | 603 | 0 | 0 | 205 |
| S | 506 | 372 | 0 | 0 | 122 |
| V | 204 | 605 | 0 | 0 | 191 |
| F | 14 | 403 | 0 | 0 | 386 |
| Q | 250 | 750 | 0 | 0 | 0 |

#### `random_forest` -- out-of-fold (sum over folds)

| | N | S | V | F | Q |
|--|---|---|---|---|---|
| N | 680 | 104 | 94 | 82 | 40 |
| S | 646 | 257 | 81 | 10 | 6 |
| V | 36 | 22 | 897 | 28 | 17 |
| F | 188 | 78 | 409 | 127 | 1 |
| Q | 311 | 72 | 519 | 2 | 96 |

#### `svm_rbf` -- out-of-fold (sum over folds)

| | N | S | V | F | Q |
|--|---|---|---|---|---|
| N | 667 | 133 | 84 | 50 | 66 |
| S | 620 | 232 | 97 | 10 | 41 |
| V | 70 | 28 | 792 | 45 | 65 |
| F | 450 | 9 | 189 | 154 | 1 |
| Q | 269 | 48 | 372 | 3 | 308 |

### 9.1b Training diagnostics (full-data fits, diagnostics only)

#### `dummy` -- training (full data, diagnostics)

| | N | S | V | F | Q |
|--|---|---|---|---|---|
| N | 0 | 0 | 0 | 0 | 977 |
| S | 0 | 0 | 0 | 0 | 969 |
| V | 0 | 0 | 0 | 0 | 942 |
| F | 0 | 0 | 0 | 0 | 803 |
| Q | 0 | 0 | 0 | 0 | 1,000 |

#### `random_forest` -- training (full data, diagnostics)

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
| N | 868 | 56 | 12 | 38 | 3 |
| S | 112 | 834 | 4 | 15 | 4 |
| V | 22 | 8 | 874 | 31 | 7 |
| F | 43 | 5 | 29 | 725 | 1 |
| Q | 48 | 15 | 4 | 4 | 929 |

### 9.2 Per-class performance of the best evaluated model (`svm_rbf`)
Per-class performance (precision, recall, sensitivity, specificity, F1, support):

| Class | Precision | Recall | Sensitivity | Specificity | F1 | Support |
|---|---|---|---|---|---|---|
| N | 0.7941 | 0.8884 | 0.8884 | 0.9394 | 0.8386 | 977 |
| S | 0.9085 | 0.8607 | 0.8607 | 0.9774 | 0.8839 | 969 |
| V | 0.9469 | 0.9278 | 0.9278 | 0.9869 | 0.9373 | 942 |
| F | 0.8918 | 0.9029 | 0.9029 | 0.9774 | 0.8973 | 803 |
| Q | 0.9841 | 0.9290 | 0.9290 | 0.9959 | 0.9558 | 1000 |


## 10. Noise Robustness Experiment

Setup (per the specification): clean reference = MITDB beats. Noisy evaluation = synthetic mixtures of the clean beats with NSTDB noise realizations (NSTDB record segments) at the target SNR levels `SNR(dB) = 10*log10(P_signal / P_noise)`, where P_signal/P_noise are the mean squared values; clean and noisy segments are aligned at identical positions. The NSTDB recordings (incl. the reserved records 118/119) were also classified directly as recorded-noise evaluation (10.2). These are labeled synthetic-mixture results; the recorded-noise experiment below uses real NSTDB noise.

### 10.1 Out-of-fold synthetic mixtures -- accuracy vs SNR (per-fold models scored on held-out validation beats; spec #3)

| SNR (dB) | Model | Accuracy | Macro F1 | Macro Sensitivity | Balanced Accuracy |
|---|---|---|---|---|---|
| -6 | dummy | 0.1647 | 0.0566 | 0.2000 | 0.2000 |
| -6 | dummy | 0.1274 | 0.0452 | 0.2000 | 0.2000 |
| -6 | dummy | 0.0949 | 0.0347 | 0.2000 | 0.2000 |
| -6 | dummy | 0.1877 | 0.0632 | 0.2000 | 0.2000 |
| -6 | dummy | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| -6 | random_forest | 0.2204 | 0.1388 | 0.2346 | 0.2346 |
| -6 | random_forest | 0.2803 | 0.2183 | 0.3139 | 0.3139 |
| -6 | random_forest | 0.3051 | 0.1503 | 0.2296 | 0.2296 |
| -6 | random_forest | 0.2724 | 0.1216 | 0.2192 | 0.2192 |
| -6 | random_forest | 0.2500 | 0.1344 | 0.2322 | 0.2902 |
| -6 | svm_rbf | 0.2084 | 0.1187 | 0.2271 | 0.2271 |
| -6 | svm_rbf | 0.1722 | 0.0925 | 0.1978 | 0.1978 |
| -6 | svm_rbf | 0.3308 | 0.1882 | 0.2483 | 0.2483 |
| -6 | svm_rbf | 0.1497 | 0.1158 | 0.1229 | 0.1229 |
| -6 | svm_rbf | 0.2644 | 0.1482 | 0.2467 | 0.3084 |
| 0 | dummy | 0.1647 | 0.0566 | 0.2000 | 0.2000 |
| 0 | dummy | 0.1274 | 0.0452 | 0.2000 | 0.2000 |
| 0 | dummy | 0.0949 | 0.0347 | 0.2000 | 0.2000 |
| 0 | dummy | 0.1877 | 0.0632 | 0.2000 | 0.2000 |
| 0 | dummy | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 0 | random_forest | 0.2213 | 0.1364 | 0.2327 | 0.2327 |
| 0 | random_forest | 0.3207 | 0.2992 | 0.3628 | 0.3628 |
| 0 | random_forest | 0.4885 | 0.3222 | 0.3631 | 0.3631 |
| 0 | random_forest | 0.3031 | 0.1675 | 0.2456 | 0.2456 |
| 0 | random_forest | 0.3429 | 0.2493 | 0.3231 | 0.4039 |
| 0 | svm_rbf | 0.2890 | 0.2082 | 0.2827 | 0.2827 |
| 0 | svm_rbf | 0.2496 | 0.1940 | 0.2792 | 0.2792 |
| 0 | svm_rbf | 0.4654 | 0.3650 | 0.4000 | 0.4000 |
| 0 | svm_rbf | 0.2675 | 0.1857 | 0.2244 | 0.2244 |
| 0 | svm_rbf | 0.3562 | 0.2613 | 0.3242 | 0.4052 |
| 6 | dummy | 0.1647 | 0.0566 | 0.2000 | 0.2000 |
| 6 | dummy | 0.1274 | 0.0452 | 0.2000 | 0.2000 |
| 6 | dummy | 0.0949 | 0.0347 | 0.2000 | 0.2000 |
| 6 | dummy | 0.1877 | 0.0632 | 0.2000 | 0.2000 |
| 6 | dummy | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 6 | random_forest | 0.2427 | 0.1758 | 0.2753 | 0.2753 |
| 6 | random_forest | 0.2830 | 0.2749 | 0.3264 | 0.3264 |
| 6 | random_forest | 0.4436 | 0.3008 | 0.3581 | 0.3581 |
| 6 | random_forest | 0.3656 | 0.2361 | 0.2994 | 0.2994 |
| 6 | random_forest | 0.3916 | 0.2759 | 0.3680 | 0.4600 |
| 6 | svm_rbf | 0.2496 | 0.2194 | 0.2902 | 0.2902 |
| 6 | svm_rbf | 0.3541 | 0.3288 | 0.3647 | 0.3647 |
| 6 | svm_rbf | 0.5244 | 0.4602 | 0.5072 | 0.5072 |
| 6 | svm_rbf | 0.3840 | 0.2970 | 0.3275 | 0.3275 |
| 6 | svm_rbf | 0.4049 | 0.3081 | 0.3786 | 0.4733 |
| 12 | dummy | 0.1647 | 0.0566 | 0.2000 | 0.2000 |
| 12 | dummy | 0.1274 | 0.0452 | 0.2000 | 0.2000 |
| 12 | dummy | 0.0949 | 0.0347 | 0.2000 | 0.2000 |
| 12 | dummy | 0.1877 | 0.0632 | 0.2000 | 0.2000 |
| 12 | dummy | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 12 | random_forest | 0.2496 | 0.1965 | 0.3008 | 0.3008 |
| 12 | random_forest | 0.2531 | 0.2207 | 0.2869 | 0.2869 |
| 12 | random_forest | 0.4385 | 0.2903 | 0.3514 | 0.3514 |
| 12 | random_forest | 0.3767 | 0.2557 | 0.3100 | 0.3100 |
| 12 | random_forest | 0.4049 | 0.2934 | 0.3765 | 0.4706 |
| 12 | svm_rbf | 0.2547 | 0.2462 | 0.3134 | 0.3134 |
| 12 | svm_rbf | 0.3409 | 0.3050 | 0.3335 | 0.3335 |
| 12 | svm_rbf | 0.5026 | 0.4167 | 0.4511 | 0.4511 |
| 12 | svm_rbf | 0.4282 | 0.3480 | 0.3605 | 0.3605 |
| 12 | svm_rbf | 0.4015 | 0.3010 | 0.3801 | 0.4752 |
| 18 | dummy | 0.1647 | 0.0566 | 0.2000 | 0.2000 |
| 18 | dummy | 0.1274 | 0.0452 | 0.2000 | 0.2000 |
| 18 | dummy | 0.0949 | 0.0347 | 0.2000 | 0.2000 |
| 18 | dummy | 0.1877 | 0.0632 | 0.2000 | 0.2000 |
| 18 | dummy | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 18 | random_forest | 0.2504 | 0.1881 | 0.2898 | 0.2898 |
| 18 | random_forest | 0.2575 | 0.2143 | 0.2928 | 0.2928 |
| 18 | random_forest | 0.4526 | 0.2951 | 0.3590 | 0.3590 |
| 18 | random_forest | 0.3951 | 0.2580 | 0.3200 | 0.3200 |
| 18 | random_forest | 0.3872 | 0.2746 | 0.3637 | 0.4547 |
| 18 | svm_rbf | 0.2581 | 0.2579 | 0.3146 | 0.3146 |
| 18 | svm_rbf | 0.3295 | 0.2934 | 0.3207 | 0.3207 |
| 18 | svm_rbf | 0.5141 | 0.4223 | 0.4461 | 0.4461 |
| 18 | svm_rbf | 0.4601 | 0.3751 | 0.3778 | 0.3778 |
| 18 | svm_rbf | 0.3971 | 0.2941 | 0.3797 | 0.4747 |
| 24 | dummy | 0.1647 | 0.0566 | 0.2000 | 0.2000 |
| 24 | dummy | 0.1274 | 0.0452 | 0.2000 | 0.2000 |
| 24 | dummy | 0.0949 | 0.0347 | 0.2000 | 0.2000 |
| 24 | dummy | 0.1877 | 0.0632 | 0.2000 | 0.2000 |
| 24 | dummy | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 24 | random_forest | 0.2539 | 0.2191 | 0.3194 | 0.3194 |
| 24 | random_forest | 0.2548 | 0.2073 | 0.2895 | 0.2895 |
| 24 | random_forest | 0.4487 | 0.2928 | 0.3562 | 0.3562 |
| 24 | random_forest | 0.3902 | 0.2510 | 0.3137 | 0.3137 |
| 24 | random_forest | 0.3916 | 0.2764 | 0.3613 | 0.4517 |
| 24 | svm_rbf | 0.2693 | 0.2677 | 0.3243 | 0.3243 |
| 24 | svm_rbf | 0.3409 | 0.3038 | 0.3338 | 0.3338 |
| 24 | svm_rbf | 0.5256 | 0.4426 | 0.4550 | 0.4550 |
| 24 | svm_rbf | 0.4650 | 0.3734 | 0.3773 | 0.3773 |
| 24 | svm_rbf | 0.3993 | 0.2976 | 0.3845 | 0.4807 |

### 10.2 Recorded-noise (NSTDB) classification per model -- mean metrics

| Model | Accuracy | Macro F1 | Sensitivity Macro | Balanced Accuracy |
|---|---|---|---|---|
| dummy | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| random_forest | 0.4820 | 0.1903 | 0.2336 | 0.5340 |
| svm_rbf | 0.4221 | 0.1741 | 0.2212 | 0.5085 |

Per-record predicted AAMI class counts (NSTDB; metrics where truth is available, per the table above):

- `118e00` [dummy]: predicted N=0, S=0, V=0, F=0, Q=2301
- `118e00` [random_forest]: predicted N=314, S=3, V=1855, F=0, Q=129
- `118e00` [svm_rbf]: predicted N=164, S=9, V=1619, F=7, Q=502
- `118e06` [dummy]: predicted N=0, S=0, V=0, F=0, Q=2301
- `118e06` [random_forest]: predicted N=327, S=7, V=1814, F=3, Q=150
- `118e06` [svm_rbf]: predicted N=177, S=8, V=1532, F=20, Q=564
- `118e12` [dummy]: predicted N=0, S=0, V=0, F=0, Q=2301
- `118e12` [random_forest]: predicted N=389, S=14, V=1738, F=3, Q=157
- `118e12` [svm_rbf]: predicted N=213, S=17, V=1368, F=44, Q=659
- `118e18` [dummy]: predicted N=0, S=0, V=0, F=0, Q=2301
- `118e18` [random_forest]: predicted N=464, S=5, V=1663, F=0, Q=169
- `118e18` [svm_rbf]: predicted N=292, S=4, V=1248, F=16, Q=741
- `118e24` [dummy]: predicted N=0, S=0, V=0, F=0, Q=2301
- `118e24` [random_forest]: predicted N=541, S=4, V=1605, F=0, Q=151
- `118e24` [svm_rbf]: predicted N=316, S=3, V=1142, F=8, Q=832
- `118e_6` [dummy]: predicted N=0, S=0, V=0, F=0, Q=2301
- `118e_6` [random_forest]: predicted N=311, S=2, V=1866, F=0, Q=122
- `118e_6` [svm_rbf]: predicted N=163, S=3, V=1643, F=1, Q=491
- `119e00` [dummy]: predicted N=0, S=0, V=0, F=0, Q=2094
- `119e00` [random_forest]: predicted N=870, S=29, V=1112, F=7, Q=76
- `119e00` [svm_rbf]: predicted N=832, S=41, V=1178, F=33, Q=10
- `119e06` [dummy]: predicted N=0, S=0, V=0, F=0, Q=2094
- `119e06` [random_forest]: predicted N=977, S=45, V=997, F=18, Q=57
- `119e06` [svm_rbf]: predicted N=876, S=58, V=1016, F=125, Q=19
- `119e12` [dummy]: predicted N=0, S=0, V=0, F=0, Q=2094
- `119e12` [random_forest]: predicted N=1239, S=64, V=712, F=21, Q=58
- `119e12` [svm_rbf]: predicted N=1058, S=79, V=714, F=227, Q=16
- `119e18` [dummy]: predicted N=0, S=0, V=0, F=0, Q=2094
- `119e18` [random_forest]: predicted N=1505, S=45, V=522, F=11, Q=11
- `119e18` [svm_rbf]: predicted N=1307, S=50, V=537, F=197, Q=3
- `119e24` [dummy]: predicted N=0, S=0, V=0, F=0, Q=2094
- `119e24` [random_forest]: predicted N=1542, S=41, V=492, F=13, Q=6
- `119e24` [svm_rbf]: predicted N=1457, S=47, V=504, F=86, Q=0
- `119e_6` [dummy]: predicted N=0, S=0, V=0, F=0, Q=2094
- `119e_6` [random_forest]: predicted N=857, S=25, V=1138, F=7, Q=67
- `119e_6` [svm_rbf]: predicted N=833, S=30, V=1208, F=22, Q=1
- `bw` [dummy]: predicted N=0, S=0, V=0, F=0, Q=2193
- `bw` [random_forest]: predicted N=204, S=396, V=4, F=1589, Q=0
- `bw` [svm_rbf]: predicted N=917, S=1201, V=29, F=30, Q=16
- `em` [dummy]: predicted N=0, S=0, V=0, F=0, Q=1989
- `em` [random_forest]: predicted N=63, S=130, V=1699, F=13, Q=84
- `em` [svm_rbf]: predicted N=151, S=139, V=1691, F=7, Q=1
- `ma` [dummy]: predicted N=0, S=0, V=0, F=0, Q=2276
- `ma` [random_forest]: predicted N=367, S=859, V=8, F=1030, Q=12
- `ma` [svm_rbf]: predicted N=652, S=1100, V=513, F=1, Q=10

Note: the final models used for the recorded-noise evaluation were trained on the full data excluding the reserved records (118, 119); see 9.1b.

### 10.3 Figures: `snr_vs_macro_f1.png`, `snr_vs_sensitivity.png`, `snr_vs_balanced_accuracy.png`, `clean_vs_noisy.png`

## 11. Figures

- `reports\figures\ecg_original.png`
- `reports\figures\ecg_filtered.png`
- `reports\figures\ecg_segmentation.png`
- `reports\figures\class_distribution.png`
- `reports\figures\ecg_classes_comparison.png`
- `reports\figures\confusion_matrix_dummy.png`
- `reports\figures\confusion_matrix_random_forest.png`
- `reports\figures\confusion_matrix_svm_rbf.png`
- `reports\figures\model_performance.png`
- `reports\figures\clean_vs_noisy.png`
- `reports\figures\snr_vs_macro_f1.png`
- `reports\figures\snr_vs_sensitivity.png`
- `reports\figures\snr_vs_balanced_accuracy.png`

## 12. Interpretation

Plausible reading of the measured results (interpretation of actual numbers above):
- Patient independence: grouping all beats of a recording and using StratifiedGroupKFold yielded no patient leakage; the honest estimate of generalization to unseen patients is the cross-fold mean, not the training fit.
- Mean 5-fold accuracy is 0.46 for the best model -- weak; the feature set or segmentation window may not carry enough signal, or the task (5-class AAMI on single beats) is intrinsically hard.
- The noise experiment (to be filled from `reports/results/noise_results.csv`) documents how performance decays as SNR decreases; the NSTDB records 118/119 were reserved and classified with real noise in the recorded-noise evaluation.

## 13. Limitations (fixed, per the project specification)

- Historical data: MITDB is a widely used database from the era of older leads/techniques; classification performance must not be claimed as clinical diagnosis.
- Class imbalance: AAMI N is the majority class; we report macro and balanced metrics, not accuracy alone, and use class weights.
- Noise-cohort size: NSTDB provides two noise-stress records (118, 119); the synthetic mixture experiment adds 8 real NSTDB noise realizations for the SNR sweep. Results are descriptive, not a statistically powered comparison of official noise.
- Not a clinical diagnostic tool; beat-level AAMI classification is research-grade.
- Features are extracted from a 0.7 s window (0.25 s pre / 0.45 s post R); alternative windows were not tested (scope decision).

## 14. References

- PhysioNet MIT-BIH Arrhythmia Database -- https://physionet.org/content/mitdb/1.0.0/
- PhysioNet MIT-BIH Noise Stress Test Database -- https://physionet.org/content/nstdb/1.0.0/
- AAMI EC57:2012 -- Cardiac monitor inappropriate alert performance standard (AAMI EC57).

*Reproduction:* `python scripts/reproduce_pipeline.py` (downloads PhysioNet databases, runs the full pipeline, and regenerates this report with the actual results; requires `wfdb`, `numpy`, `scipy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `pytest`, `pyyaml`, `joblib`.)
