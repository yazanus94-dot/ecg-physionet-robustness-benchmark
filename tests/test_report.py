"""Spec #2: technical report generation from out-of-fold result sources."""

import pandas as pd

from ecg_robust.technical_report import write_technical_report


def _mock_data():
    cv_rows = []
    for m in ["dummy", "random_forest", "svm_rbf"]:
        for f in range(5):
            cv_rows.append({
                "model": m, "fold": f, "n_train": 3600, "n_val": 803,
                "accuracy": 0.30 + 0.01 * f,
                "precision_macro": 0.31, "recall_macro": 0.29,
                "sensitivity_macro": 0.29, "specificity_macro": 0.83,
                "f1_macro": 0.25, "f1_weighted": 0.30, "balanced_accuracy": 0.30,
            })
    cv_df = pd.DataFrame(cv_rows)
    metrics_df = cv_df.groupby("model")[
        ["accuracy", "precision_macro", "recall_macro", "sensitivity_macro",
         "specificity_macro", "f1_macro", "f1_weighted", "balanced_accuracy"]
    ].mean().reset_index()
    noise_df = pd.DataFrame([
        {"snr_db": 24.0, "model": "random_forest", "n_beats": 803,
         "accuracy": 0.41, "macro_f1": 0.31, "sensitivity_macro": 0.36,
         "balanced_accuracy": 0.41},
    ])
    nstdb_df = pd.DataFrame([
        {"record": "118e00", "model": "random_forest", "n_beats": 2301,
         "predicted_class_counts": {"N": 1510, "S": 0, "V": 706, "F": 2, "Q": 83},
         "accuracy": 0.79, "macro_f1": 0.18, "sensitivity_macro": 0.20,
         "balanced_accuracy": 0.33},
    ])
    return {
        "dataset": {
            "records": ["100", "118"], "n_records": 48,
            "total_beats_annotated": 100, "total_beats_aami": 50,
            "class_counts": {"N": 30, "S": 10, "V": 8, "F": 2},
            "sampling_frequency_hz": 360, "nstdb": ["118e00"],
            "excluded_annotations": 3, "unknown_annotations": 1,
        },
        "cv_df": cv_df, "metrics_df": metrics_df,
        "noise_df": noise_df, "nstdb_df": nstdb_df,
        "n_features": 23, "figures": ["ecg_original.png"],
        "wall_s": 1.0,
        "confusion": {"random_forest": [[1, 0, 0, 0, 0], [0, 5, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]},
        "confusion_train": {"random_forest": [[2, 0, 0, 0, 0], [0, 8, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]},
        "best_model": "random_forest",
        "reserved_records": ["118", "119"],
        "confusion_labels": ["N", "S", "V", "F", "Q"],
        "per_class": {
            "random_forest": {
                "N": {"precision": 0.9, "recall": 0.8, "sensitivity": 0.8,
                      "specificity": 0.95, "f1": 0.85, "support": 30},
                "S": {"precision": 0.0, "recall": 0.0, "sensitivity": 0.0,
                      "specificity": 1.0, "f1": 0.0, "support": 10},
                "V": {"precision": 0.0, "recall": 0.0, "sensitivity": 0.0,
                      "specificity": 1.0, "f1": 0.0, "support": 8},
                "F": {"precision": 0.0, "recall": 0.0, "sensitivity": 0.0,
                      "specificity": 1.0, "f1": 0.0, "support": 2},
                "Q": {"precision": 0.0, "recall": 0.0, "sensitivity": 0.0,
                      "specificity": 1.0, "f1": 0.0, "support": 0},
            }
        },
    }


def test_report_contains_oof_sections(tmp_path):
    path = write_technical_report(_mock_data(), tmp_path / "TECHNICAL_REPORT.md")
    text = path.read_text(encoding="utf-8")
    assert "out-of-fold" in text
    assert "9.1 Confusion matrices (out-of-fold" in text
    assert "### 9.1b Training diagnostics" in text
    assert "Best CV accuracy" in text


def test_report_per_class_of_best_model(tmp_path):
    path = write_technical_report(_mock_data(), tmp_path / "TECHNICAL_REPORT.md")
    text = path.read_text(encoding="utf-8")
    assert "Per-class performance of the best evaluated model" in text
    assert "| N | 0.9000 | 0.8000 | 0.8000 | 0.9500 | 0.8500 | 30 |" in text


def test_report_reserved_records_note(tmp_path):
    path = write_technical_report(_mock_data(), tmp_path / "TECHNICAL_REPORT.md")
    text = path.read_text(encoding="utf-8")
    assert "118, 119" in text
    assert "9.1b" in text
