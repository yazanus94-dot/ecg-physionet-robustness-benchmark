"""Scientific figures — saved to ``reports/figures/``.

Figures produced:
1. ``ecg_original.png``        — original (raw, unfiltered) ECG waveform.
2. ``ecg_filtered.png``        — filtered ECG waveform (4th-order Butterworth).
3. ``ecg_segmentation.png``    — beat segmentation on the filtered signal
                                  (reference annotation positions marked).
4. ``class_distribution.png``  — heartbeat class distribution (bar chart).
5. ``ecg_classes_comparison.png`` — ECG waveform comparisons between classes
                                  (mean heartbeat per class, overlay).
6. ``confusion_matrix_<model>.png`` — model confusion matrices (one per model).
7. ``model_performance.png``   — model performance comparison (bar chart of
                                  accuracy, macro F1, macro sensitivity,
                                  balanced accuracy per model).
8. ``clean_vs_noisy.png``      — clean versus noisy ECG segments.
9. ``snr_vs_macro_f1.png``     — SNR versus macro F1 (per model).
10. ``noise_performance.png``  — sensitivity and balanced accuracy under noise.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .annotations import CLASSES

FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _time_axis(fs: float, n: int) -> np.ndarray:
    return np.arange(n) / fs


def plot_signal(x: np.ndarray, fs: float, title: str, out: str) -> None:
    """Plot the first ~5 s of a signal."""
    n = min(len(x), int(5.0 * fs))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(_time_axis(fs, n), x[:n], lw=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ECG (mV)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / out, dpi=140)
    plt.close(fig)


def plot_segmentation(
    x: np.ndarray, r_positions: np.ndarray, fs: float, out: str = "ecg_segmentation.png"
) -> None:
    """Plot the filtered signal with reference annotation positions marked."""
    n = min(len(x), int(6.0 * fs))
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(_time_axis(fs, n), x[:n], lw=0.8, color="#1f77b4")
    sel = r_positions[(r_positions >= 0) & (r_positions < n)]
    ax.vlines(
        sel / fs,
        ax.get_ylim()[0] - 1,
        ax.get_ylim()[1] + 1,
        colors="red",
        alpha=0.35,
        linewidths=0.8,
        label="R annotations (ref)",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ECG (mV)")
    ax.set_title("Beat segmentation around reference annotations")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / out, dpi=140)
    plt.close(fig)


def plot_class_distribution(
    class_counts: dict[str, int], out: str = "class_distribution.png"
) -> None:
    """Bar chart of the heartbeat class distribution."""
    classes = CLASSES
    counts = [class_counts.get(c, 0) for c in classes]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(classes, counts, color="#4c78a8")
    ax.set_xlabel("AAMI class")
    ax.set_ylabel("Beats")
    ax.set_title("Heartbeat class distribution (MIT-BIH Arrhythmia Database)")
    for b in bars:
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.5,
            f"{b.get_height():,}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / out, dpi=140)
    plt.close(fig)


def plot_class_comparison(
    segments: dict[str, list[np.ndarray]],
    fs: float,
    out: str = "ecg_classes_comparison.png",
) -> None:
    """Overlay mean heartbeat per AAMI class (per-class waveform comparison)."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 6))
    axes = axes.flatten()
    ax_iter = iter(axes)
    ax = next(ax_iter)
    for c in CLASSES:
        segs = [s for s in segments.get(c, [])]
        if len(segs) == 0:
            continue
        n = min(len(s) for s in segs)
        mean = np.mean([s[:n] for s in segs], axis=0)
        ax.plot(np.arange(n) / fs, mean, label=c, alpha=0.8)
    # combined plot in the last axis
    ax.set_title("Mean heartbeat per class (overlay)")
    ax.set_xlabel("Time since segment start (s)")
    ax.set_ylabel("ECG (mV)")
    ax.legend(fontsize=8)
    for c in CLASSES:
        ax = next(ax_iter)
        segs = segments.get(c, [])
        if len(segs) == 0:
            ax.set_visible(False)
            continue
        n = min(len(s) for s in segs)
        mean = np.mean([s[:n] for s in segs], axis=0)
        ax.plot(np.arange(n) / fs, mean, color="#4c78a8")
        ax.set_title(f"Class {c}: n = {len(segs)}")
        ax.set_xlabel("Time since segment start (s)")
        ax.set_ylabel("ECG (mV)")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / out, dpi=140)
    plt.close(fig)


def plot_confusion_matrix(
    cm: np.ndarray, classes: list[str], model_name: str, out: str | None = None
) -> None:
    """Heatmap of the confusion matrix for one model."""
    if out is None:
        out = f"confusion_matrix_{model_name}.png"
    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im)
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(
                j,
                i,
                f"{cm[i, j]:,}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=8,
            )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix — {model_name}")
    fig.tight_layout()
    fig.savefig(FIG_DIR / out, dpi=140)
    plt.close(fig)


def plot_model_performance(
    metrics_df: pd.DataFrame, out: str = "model_performance.png"
) -> None:
    """Grouped bar chart: accuracy, macro F1, macro sensitivity, balanced accuracy per model."""
    metrics_df = metrics_df.sort_values("model")
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    x = np.arange(len(metrics_df["model"]))
    width = 0.2
    for i, col in enumerate(
        ["accuracy", "macro_f1", "sensitivity_macro", "balanced_accuracy"]
    ):
        if col not in metrics_df.columns:
            continue
        ax.bar(
            x + i * width, metrics_df[col], width, label=col.replace("_", " ").title()
        )
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(metrics_df["model"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model performance comparison (5-fold stratified group CV)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / out, dpi=140)
    plt.close(fig)


def plot_clean_vs_noisy(
    clean: np.ndarray, noisy: np.ndarray, fs: float, out: str = "clean_vs_noisy.png"
) -> None:
    """Plot clean vs noisy ECG segments side by side."""
    n = min(len(clean), len(noisy), int(5.0 * fs))
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(_time_axis(fs, n), clean[:n], color="#4c78a8", lw=0.8)
    axes[0].set_title("Clean ECG segment")
    axes[0].set_ylabel("ECG (mV)")
    axes[1].plot(_time_axis(fs, n), noisy[:n], color="#e45756", lw=0.8)
    axes[1].set_title("Noisy ECG segment (synthetic mixture)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("ECG (mV)")
    for a in axes:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / out, dpi=140)
    plt.close(fig)


def plot_snr_vs_metric(
    noise_df: pd.DataFrame, metric: str, out: str, label: str
) -> None:
    """Line plot of ``metric`` vs SNR per model."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, g in noise_df.groupby("model", sort=True):
        g = g.sort_values("snr_db")
        ax.plot(g["snr_db"], g[metric], marker="o", label=name)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel(label)
    ax.set_title(f"{label} vs SNR (noise robustness, synthetic mixtures)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xticks(sorted(noise_df["snr_db"].unique()))
    fig.tight_layout()
    fig.savefig(FIG_DIR / out, dpi=140)
    plt.close(fig)


def plot_feature_ablation(df: pd.DataFrame, out: str = "feature_ablation.png") -> None:
    """Bar chart: mean 5-fold CV accuracy by feature variant and model."""
    pivot = df.pivot_table(
        index="variant", columns="model", values="accuracy_mean", aggfunc="first"
    )
    fig, ax = plt.subplots(figsize=(9.5, 5))
    pivot.plot(kind="bar", ax=ax, colormap="viridis")
    ax.set_ylabel("Mean 5-fold CV accuracy")
    ax.set_title("Feature ablation (same patient-independent folds)")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / out, dpi=140)
    plt.close(fig)


def save_all_figures(
    ecg_original: np.ndarray,
    ecg_filtered: np.ndarray,
    r_positions: np.ndarray,
    segments_by_class: dict[str, list[np.ndarray]],
    class_counts: dict[str, int],
    metrics_df: pd.DataFrame,
    confusion: dict[str, np.ndarray],
    classes: list[str],
    clean_seg: np.ndarray | None = None,
    noisy_seg: np.ndarray | None = None,
    noise_df: pd.DataFrame | None = None,
) -> list[str]:
    """Generate all scientific figures; returns the list of figure paths."""
    made: list[str] = []
    fs = 360.0
    plot_signal(
        ecg_original, fs, "Original ECG waveform (first 5 s)", "ecg_original.png"
    )
    made.append("ecg_original.png")
    plot_signal(
        ecg_filtered,
        fs,
        "Filtered ECG waveform (first 5 s, 4th-order Butterworth 0.5–40 Hz)",
        "ecg_filtered.png",
    )
    made.append("ecg_filtered.png")
    plot_segmentation(ecg_filtered, r_positions, fs)
    made.append("ecg_segmentation.png")
    plot_class_distribution(class_counts)
    made.append("class_distribution.png")
    plot_class_comparison(segments_by_class, fs)
    made.append("ecg_classes_comparison.png")
    for model, cm in confusion.items():
        plot_confusion_matrix(cm, classes, model)
        made.append(f"confusion_matrix_{model}.png")
    if metrics_df is not None and len(metrics_df):
        plot_model_performance(metrics_df)
        made.append("model_performance.png")
    if clean_seg is not None and noisy_seg is not None:
        plot_clean_vs_noisy(clean_seg, noisy_seg, fs)
        made.append("clean_vs_noisy.png")
    if noise_df is not None and len(noise_df):
        plot_snr_vs_metric(noise_df, "macro_f1", "snr_vs_macro_f1.png", "Macro F1")
        made.append("snr_vs_macro_f1.png")
        plot_snr_vs_metric(
            noise_df, "sensitivity_macro", "snr_vs_sensitivity.png", "Macro sensitivity"
        )
        made.append("snr_vs_sensitivity.png")
        plot_snr_vs_metric(
            noise_df,
            "balanced_accuracy",
            "snr_vs_balanced_accuracy.png",
            "Balanced accuracy",
        )
        made.append("snr_vs_balanced_accuracy.png")
    return [str(FIG_DIR / m) for m in made]
