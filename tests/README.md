# Test suite for ECG-ROBUST (`tests/`)

| File | Covers |
|---|---|
| `test_annotations.py` | AAMI mapping, alignment (`annotate_annotations`), annotation statuses, no-shift guarantees, validation (length mismatch, negative samples). |
| `test_preprocessing.py` | 4th-order Butterworth bandpass (sosfiltfilt, DC removal), training-fitted normalization. |
| `test_segmentation.py` | `extract_segments`/`extract_beats` windows, boundary clamping, RR interval/ratio conventions. |
| `test_rr_features.py` | RR interval and RR ratio indexing convention (spec #1) with known R-peak sequences. |
| `test_features.py` | Feature extraction values and finiteness (spec: documented features; non-finite rejection). |
| `test_splits.py` | Patient-independent group separation (recording IDs; 201/202 grouped). |
| `test_models.py` | `build_models` (5 primary models), training + prediction, seeded reproducibility. |
| `test_evaluation.py` | Metrics definitions, per-class metrics, agreement with confusion matrices. |
| `test_oof_per_class.py` | OOF per-class reporting (spec #2): `per_class_metrics_from_cm` + agreement with `evaluate`. |
| `test_noise.py` | SNR formula/scaling, `run_noise_sweep` (legacy), fair noise helpers. |
| `test_oof_noise.py` | Fair noise experiments (spec #4): shared/deterministic noisy samples, achieved-SNR consistency. |
| `test_nstdb_alignment.py` | NSTDB recorded-noise label alignment (spec #3) — beats carry original annotation identity + RR convention. |
| `test_imputation.py` | Fold-specific imputation inside the sklearn pipeline (spec #4). |
| `test_report.py` | `write_technical_report` — OOF sections, training diagnostics, per-class of best model. |
| `test_feature_ablation.py` | Ablation variant definitions (spec Phase 6). |
| `test_dataset.py` | MITDB access — skipped unless `data/raw/mitdb` present. |
| `test_reproducibility.py` | Seeded reproducibility of models. |

Run all: `python -m pytest -q` (the expert task tests live in `tasks/ecg_arrhythmia_robustness/tests/`).
