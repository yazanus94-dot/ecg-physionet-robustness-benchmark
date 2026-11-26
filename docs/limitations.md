# Limitations — ECG-ROBUST

- Reported metrics are fold means/std, not confidence intervals.
- Noise experiment uses contaminated-ECG realizations, not pure noise.
- Model selection on evaluation folds (winner's curse) — see SCIENTIFIC_DECISIONS.md.
- Feature ablation results are stale (leaky prior implementation).
- Synthetic-mixture stress tests only; not clinical validation.
- Requires licensed MITDB/NSTDB data for full reproduction.
