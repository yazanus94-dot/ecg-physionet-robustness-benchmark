# Architecture — ECG-ROBUST

## Pipeline overview

```
MIT-BIH Arrhythmia Database (mitdb)  +  MIT-BIH Noise Stress Test (nstdb)
            |                                          |
            v                                          v
   annotations.py  (symbol -> AAMI N/S/V/F/Q)   noise.py (noise realizations)
            |                                          |
            v                                          v
   segmentation.py  (0.25s before / 0.45s after R-peak, fs=360 Hz)
            |
            v
   features.py  (RR intervals, morphological & statistical features)
            |
            v
   splits.py  (StratifiedGroupKFold, patient-independent, records 201/202 grouped)
            |
            v
   preprocessing.py  (per-fold median imputation + scaling, no leakage)
            |
            v
   models.py  (Dummy, LogisticRegression, RandomForest, RBF-SVM, HistGradientBoosting)
            |
            v
   evaluation.py  (out-of-fold metrics, confusion matrices, per-class)
            |
            v
   technical_report.py  ->  reports/TECHNICAL_REPORT.md, reports/results/*.csv
   visualization.py     ->  reports/figures/*.png
```

## Data flow and leakage prevention

- All preprocessing is fitted inside the sklearn pipeline on training folds only.
- Patient/grouping ensures no recording appears in both train and test.
- Out-of-fold predictions are used for every primary metric.
- Feature selection (where applied) is refit per fold.
