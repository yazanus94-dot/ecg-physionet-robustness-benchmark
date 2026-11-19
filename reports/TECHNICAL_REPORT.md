# ECG-ROBUST -- Technical Report: Patient-Independent ECG Arrhythmia Classification and Noise-Robustness Benchmark

- Repository: `yazanus94-dot/ecg-physionet-robustness-benchmark`
- Report date: 2026-11-19 18:45 UTC
- Pipeline wall time (measured, including downloads): 363.5 s

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
- Beats segmented across all recordings (AAMI-mappable + excluded): 112647
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
- Each fold refits `StandardScaler` and the model on the training split only.

## 7. Models

- **DummyClassifier** (`strategy='most_frequent'`) -- patient-independent baseline.
- **RandomForestClassifier** -- `n_estimators=100`, `class_weight='balanced'`, `random_state=42`.
- **SupportVectorMachine (RBF)** -- `C=1.0`, `kernel='rbf'`, `class_weight='balanced'`, `random_state=42`.
For CV, models were wrapped in `sklearn.pipeline.Pipeline` (`StandardScaler` -> model).

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
| random_forest | 0.3067 | 0.3104 | 0.2844 | 0.2844 | 0.8267 | 0.2420 | 0.2659 | 0.2981 |
| svm_rbf | 0.3031 | 0.3091 | 0.2740 | 0.2740 | 0.8249 | 0.2562 | 0.2975 | 0.2881 |

Best CV accuracy: **0.3067** (random_forest).

### 9.1 Confusion matrices (sum over folds)

#### `dummy` -- predicted class x true class

| | N | S | V | F | Q |
|--|---|---|---|---|---|
| N | 1,000 | 0 | 0 | 0 | 0 |
| S | 1,000 | 0 | 0 | 0 | 0 |
| V | 1,000 | 0 | 0 | 0 | 0 |
| F | 803 | 0 | 0 | 0 | 0 |
| Q | 1,000 | 0 | 0 | 0 | 0 |
#### `random_forest` -- predicted class x true class

| | N | S | V | F | Q |
|--|---|---|---|---|---|
| N | 1,000 | 0 | 0 | 0 | 0 |
| S | 0 | 1,000 | 0 | 0 | 0 |
| V | 0 | 0 | 1,000 | 0 | 0 |
| F | 0 | 0 | 0 | 803 | 0 |
| Q | 0 | 0 | 0 | 0 | 1,000 |
#### `svm_rbf` -- predicted class x true class

| | N | S | V | F | Q |
|--|---|---|---|---|---|
| N | 676 | 112 | 138 | 59 | 15 |
| S | 121 | 797 | 55 | 21 | 6 |
| V | 236 | 53 | 545 | 133 | 33 |
| F | 39 | 9 | 59 | 694 | 2 |
| Q | 38 | 24 | 19 | 0 | 919 |

### 9.2 Per-class performance (dummy classifier)
Per-class performance (precision, recall, sensitivity, specificity, F1, support):

| Class | Precision | Recall | Sensitivity | Specificity | F1 | Support |
|---|---|---|---|---|---|---|
| N | 0.2082 | 1.0000 | 1.0000 | 0.0000 | 0.3446 | 1000 |
| S | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 1000 |
| V | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 1000 |
| F | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 803 |
| Q | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 1000 |


## 10. Noise Robustness Experiment

Setup (per the specification): clean reference = MITDB beats. Noisy evaluation = synthetic mixtures of the clean beats with NSTDB noise realizations (NSTDB record segments) at the target SNR levels `SNR(dB) = 10*log10(P_signal / P_noise)`, where P_signal/P_noise are the mean squared values; clean and noisy segments are aligned at identical positions. The NSTDB recordings (incl. the reserved records 118/119) were also classified directly as recorded-noise evaluation (10.2). These are labeled synthetic-mixture results; the recorded-noise experiment below uses real NSTDB noise.

### 10.1 Synthetic mixtures -- accuracy vs SNR

| SNR (dB) | Model | Accuracy | Macro F1 | Macro Sensitivity | Balanced Accuracy |
|---|---|---|---|---|---|
| -6 | dummy | 0.2082 | 0.0689 | 0.2000 | 0.2000 |
| -6 | random_forest | 0.2251 | 0.1216 | 0.2162 | 0.2162 |
| -6 | svm_rbf | 0.2082 | 0.0689 | 0.2000 | 0.2000 |
| 0 | dummy | 0.2082 | 0.0689 | 0.2000 | 0.2000 |
| 0 | random_forest | 0.2213 | 0.1153 | 0.2126 | 0.2126 |
| 0 | svm_rbf | 0.2082 | 0.0689 | 0.2000 | 0.2000 |
| 6 | dummy | 0.2082 | 0.0689 | 0.2000 | 0.2000 |
| 6 | random_forest | 0.2205 | 0.1045 | 0.2118 | 0.2118 |
| 6 | svm_rbf | 0.2082 | 0.0689 | 0.2000 | 0.2000 |
| 12 | dummy | 0.2082 | 0.0689 | 0.2000 | 0.2000 |
| 12 | random_forest | 0.2192 | 0.0957 | 0.2106 | 0.2106 |
| 12 | svm_rbf | 0.2082 | 0.0689 | 0.2000 | 0.2000 |
| 18 | dummy | 0.2082 | 0.0689 | 0.2000 | 0.2000 |
| 18 | random_forest | 0.2147 | 0.0885 | 0.2062 | 0.2062 |
| 18 | svm_rbf | 0.2082 | 0.0689 | 0.2000 | 0.2000 |
| 24 | dummy | 0.2082 | 0.0689 | 0.2000 | 0.2000 |
| 24 | random_forest | 0.2172 | 0.0907 | 0.2086 | 0.2086 |
| 24 | svm_rbf | 0.2082 | 0.0689 | 0.2000 | 0.2000 |

### 10.2 Recorded-noise (NSTDB) classification per model -- mean metrics

| Model | Accuracy | Macro F1 | Sensitivity Macro | Balanced Accuracy |
|---|---|---|---|---|
| dummy | 0.8637 | 0.1849 | 0.2000 | 0.4167 |
| random_forest | 0.8287 | 0.2588 | 0.2719 | 0.5997 |
| svm_rbf | 0.1152 | 0.0379 | 0.2000 | 0.4167 |

Per-record predicted AAMI class counts (NSTDB; metrics where truth is available, per the table above):

- `118e00` [dummy]: predicted N=2301, S=0, V=0, F=0, Q=0
- `118e00` [random_forest]: predicted N=1510, S=0, V=706, F=2, Q=83
- `118e00` [svm_rbf]: predicted N=0, S=0, V=2301, F=0, Q=0
- `118e06` [dummy]: predicted N=2301, S=0, V=0, F=0, Q=0
- `118e06` [random_forest]: predicted N=1878, S=0, V=348, F=1, Q=74
- `118e06` [svm_rbf]: predicted N=0, S=0, V=2301, F=0, Q=0
- `118e12` [dummy]: predicted N=2301, S=0, V=0, F=0, Q=0
- `118e12` [random_forest]: predicted N=2125, S=0, V=148, F=1, Q=27
- `118e12` [svm_rbf]: predicted N=0, S=0, V=2301, F=0, Q=0
- `118e18` [dummy]: predicted N=2301, S=0, V=0, F=0, Q=0
- `118e18` [random_forest]: predicted N=2236, S=0, V=59, F=0, Q=6
- `118e18` [svm_rbf]: predicted N=0, S=0, V=2301, F=0, Q=0
- `118e24` [dummy]: predicted N=2301, S=0, V=0, F=0, Q=0
- `118e24` [random_forest]: predicted N=2241, S=0, V=56, F=0, Q=4
- `118e24` [svm_rbf]: predicted N=0, S=0, V=2301, F=0, Q=0
- `118e_6` [dummy]: predicted N=2301, S=0, V=0, F=0, Q=0
- `118e_6` [random_forest]: predicted N=1386, S=0, V=867, F=1, Q=47
- `118e_6` [svm_rbf]: predicted N=0, S=0, V=2301, F=0, Q=0
- `119e00` [dummy]: predicted N=2094, S=0, V=0, F=0, Q=0
- `119e00` [random_forest]: predicted N=1184, S=0, V=857, F=0, Q=53
- `119e00` [svm_rbf]: predicted N=0, S=0, V=2094, F=0, Q=0
- `119e06` [dummy]: predicted N=2094, S=0, V=0, F=0, Q=0
- `119e06` [random_forest]: predicted N=1525, S=0, V=554, F=2, Q=13
- `119e06` [svm_rbf]: predicted N=0, S=0, V=2094, F=0, Q=0
- `119e12` [dummy]: predicted N=2094, S=0, V=0, F=0, Q=0
- `119e12` [random_forest]: predicted N=1701, S=0, V=393, F=0, Q=0
- `119e12` [svm_rbf]: predicted N=0, S=0, V=2094, F=0, Q=0
- `119e18` [dummy]: predicted N=2094, S=0, V=0, F=0, Q=0
- `119e18` [random_forest]: predicted N=1709, S=0, V=385, F=0, Q=0
- `119e18` [svm_rbf]: predicted N=0, S=0, V=2094, F=0, Q=0
- `119e24` [dummy]: predicted N=2094, S=0, V=0, F=0, Q=0
- `119e24` [random_forest]: predicted N=1674, S=0, V=420, F=0, Q=0
- `119e24` [svm_rbf]: predicted N=0, S=0, V=2094, F=0, Q=0
- `119e_6` [dummy]: predicted N=2094, S=0, V=0, F=0, Q=0
- `119e_6` [random_forest]: predicted N=1022, S=0, V=1028, F=0, Q=44
- `119e_6` [svm_rbf]: predicted N=0, S=0, V=2094, F=0, Q=0
- `bw` [dummy]: predicted N=2193, S=0, V=0, F=0, Q=0
- `bw` [random_forest]: predicted N=1630, S=0, V=563, F=0, Q=0
- `bw` [svm_rbf]: predicted N=0, S=0, V=2193, F=0, Q=0
- `em` [dummy]: predicted N=1989, S=0, V=0, F=0, Q=0
- `em` [random_forest]: predicted N=1651, S=0, V=328, F=3, Q=7
- `em` [svm_rbf]: predicted N=0, S=0, V=1989, F=0, Q=0
- `ma` [dummy]: predicted N=2276, S=0, V=0, F=0, Q=0
- `ma` [random_forest]: predicted N=2122, S=0, V=154, F=0, Q=0
- `ma` [svm_rbf]: predicted N=0, S=0, V=2276, F=0, Q=0

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
- Mean 5-fold accuracy is 0.31 for the best model -- weak; the feature set or segmentation window may not carry enough signal, or the task (5-class AAMI on single beats) is intrinsically hard.
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
