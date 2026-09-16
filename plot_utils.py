"""
Plotting and logging utilities for S3-Det training and evaluation.
Generates comprehensive high-resolution curves for all losses (Total, QFL, DFL, NWD/SIoU),
accuracy/validation metrics (Precision, Recall, F1, AP50), learning rate schedules, and summary grids.
"""
import os
import json
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_history(history, log_dir):
    """Save training history to JSON and CSV formats."""
    os.makedirs(log_dir, exist_ok=True)
    json_path = os.path.join(log_dir, "history.json")
    with open(json_path, "w") as f:
        json.dump(history, f, indent=2)

    csv_path = os.path.join(log_dir, "history.csv")
    if history and "epoch" in history:
        keys = list(history.keys())
        n_rows = len(history["epoch"])
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(keys)
            for i in range(n_rows):
                writer.writerow([history[k][i] if i < len(history[k]) else "" for k in keys])


def plot_training_curves(history, plot_dir):
    """
    Generate and save all loss curves, accuracy curves, validation curves,
    and a combined summary grid.
    """
    os.makedirs(plot_dir, exist_ok=True)
    epochs = history.get("epoch", [])
    if not epochs:
        return

    # Style configuration
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.autolayout": True
    })

    # 1. Total Loss Curve
    if "loss" in history:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, history["loss"], label="Train Total Loss", color="#1f77b4", lw=2, marker="o", ms=3)
        if "val_loss" in history and any(v is not None for v in history["val_loss"]):
            ax.plot(epochs, history["val_loss"], label="Val Total Loss", color="#d62728", lw=2, ls="--", marker="s", ms=3)
        ax.set_title("S³-Det Total Loss Curve", fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Total Loss")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()
        fig.savefig(os.path.join(plot_dir, "total_loss.png"), dpi=200)
        plt.close(fig)

    # 2. QFL (Classification) Loss Curve
    if "loss_qfl" in history:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, history["loss_qfl"], label="Train QFL (Class) Loss", color="#2ca02c", lw=2, marker="o", ms=3)
        if "val_loss_qfl" in history and any(v is not None for v in history["val_loss_qfl"]):
            ax.plot(epochs, history["val_loss_qfl"], label="Val QFL Loss", color="#98df8a", lw=2, ls="--", marker="s", ms=3)
        ax.set_title("Quality Focal Loss (QFL β=2.0) Curve", fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("QFL Loss")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()
        fig.savefig(os.path.join(plot_dir, "qfl_loss.png"), dpi=200)
        plt.close(fig)

    # 3. DFL (Distribution Focal) Loss Curve
    if "loss_dfl" in history:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, history["loss_dfl"], label="Train DFL Loss", color="#ff7f0e", lw=2, marker="o", ms=3)
        if "val_loss_dfl" in history and any(v is not None for v in history["val_loss_dfl"]):
            ax.plot(epochs, history["val_loss_dfl"], label="Val DFL Loss", color="#ffbb78", lw=2, ls="--", marker="s", ms=3)
        ax.set_title("Distribution Focal Loss (DFL) Curve", fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("DFL Loss")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()
        fig.savefig(os.path.join(plot_dir, "dfl_loss.png"), dpi=200)
        plt.close(fig)

    # 4. Bounding Box Regression Loss Curve (NWD or SIoU)
    bbox_key = "loss_bbox" if "loss_bbox" in history else ("loss_nwd" if "loss_nwd" in history else "loss_siou")
    if bbox_key in history:
        loss_type_name = history.get("loss_type", ["BBox Regression"])[-1].upper() if "loss_type" in history else "BBox"
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, history[bbox_key], label=f"Train {loss_type_name} Loss", color="#9467bd", lw=2, marker="o", ms=3)
        val_bbox_key = f"val_{bbox_key}"
        if val_bbox_key in history and any(v is not None for v in history[val_bbox_key]):
            ax.plot(epochs, history[val_bbox_key], label=f"Val {loss_type_name} Loss", color="#c5b0d5", lw=2, ls="--", marker="s", ms=3)
        ax.set_title(f"{loss_type_name} Bounding Box Regression Loss Curve", fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(f"{loss_type_name} Loss")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()
        fig.savefig(os.path.join(plot_dir, "bbox_reg_loss.png"), dpi=200)
        plt.close(fig)

    # 5. Validation / Accuracy Metrics Curve (Precision, Recall, F1, AP50)
    has_val_metrics = any(k in history for k in ["val_precision", "val_recall", "val_f1", "val_ap50"])
    if has_val_metrics:
        fig, ax = plt.subplots(figsize=(8, 5))
        if "val_precision" in history:
            ax.plot(epochs, history["val_precision"], label="Precision", color="#1f77b4", lw=2, marker="^", ms=3)
        if "val_recall" in history:
            ax.plot(epochs, history["val_recall"], label="Recall", color="#2ca02c", lw=2, marker="v", ms=3)
        if "val_f1" in history:
            ax.plot(epochs, history["val_f1"], label="F1-Score", color="#d62728", lw=2.5, marker="o", ms=4)
        if "val_ap50" in history:
            ax.plot(epochs, history["val_ap50"], label="AP@0.5", color="#9467bd", lw=2, ls="-.", marker="d", ms=3)
        ax.set_title("Validation Accuracy & Performance Metrics", fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score (0.0 - 1.0)")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="lower right")
        fig.savefig(os.path.join(plot_dir, "validation_metrics.png"), dpi=200)
        plt.close(fig)

    # 6. Learning Rate Schedule Curve
    if "lr" in history:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, history["lr"], label="Learning Rate", color="#8c564b", lw=2)
        ax.set_title("Cosine Annealing Learning Rate Schedule", fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("LR")
        ax.set_yscale("log")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()
        fig.savefig(os.path.join(plot_dir, "lr_schedule.png"), dpi=200)
        plt.close(fig)

    # 7. Combined 2x3 Master Dashboard Overview
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Top-Left: All Loss Components
    ax = axes[0, 0]
    if "loss" in history:
        ax.plot(epochs, history["loss"], label="Total Loss", color="#1f77b4", lw=2)
    if "loss_qfl" in history:
        ax.plot(epochs, history["loss_qfl"], label="QFL", color="#2ca02c", lw=1.8)
    if "loss_dfl" in history:
        ax.plot(epochs, history["loss_dfl"], label="DFL", color="#ff7f0e", lw=1.8)
    if bbox_key in history:
        name = history.get("loss_type", ["BBox"])[-1].upper() if "loss_type" in history else "BBox"
        ax.plot(epochs, history[bbox_key], label=name, color="#9467bd", lw=1.8)
    ax.set_title("Loss Breakdown (Train)", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    # Top-Middle: Total Loss (Train vs Val)
    ax = axes[0, 1]
    if "loss" in history:
        ax.plot(epochs, history["loss"], label="Train Total", color="#1f77b4", lw=2)
    if "val_loss" in history and any(v is not None for v in history["val_loss"]):
        ax.plot(epochs, history["val_loss"], label="Val Total", color="#d62728", lw=2, ls="--")
    ax.set_title("Total Loss (Train vs Val)", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    # Top-Right: BBox Regression Loss
    ax = axes[0, 2]
    if bbox_key in history:
        name = history.get("loss_type", ["BBox"])[-1].upper() if "loss_type" in history else "BBox"
        ax.plot(epochs, history[bbox_key], label=f"Train {name}", color="#9467bd", lw=2)
        if f"val_{bbox_key}" in history and any(v is not None for v in history[f"val_{bbox_key}"]):
            ax.plot(epochs, history[f"val_{bbox_key}"], label=f"Val {name}", color="#c5b0d5", lw=2, ls="--")
        ax.set_title(f"{name} Localization Loss", fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()

    # Bottom-Left: Validation Metrics (Precision & Recall)
    ax = axes[1, 0]
    if "val_precision" in history:
        ax.plot(epochs, history["val_precision"], label="Precision", color="#1f77b4", lw=2)
    if "val_recall" in history:
        ax.plot(epochs, history["val_recall"], label="Recall", color="#2ca02c", lw=2)
    ax.set_title("Validation Precision & Recall", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    # Bottom-Middle: Validation F1 & AP50
    ax = axes[1, 1]
    if "val_f1" in history:
        ax.plot(epochs, history["val_f1"], label="F1-Score", color="#d62728", lw=2.5)
    if "val_ap50" in history:
        ax.plot(epochs, history["val_ap50"], label="AP@0.5", color="#9467bd", lw=2, ls="-.")
    ax.set_title("Validation F1 & AP@0.5", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    # Bottom-Right: Learning Rate Schedule
    ax = axes[1, 2]
    if "lr" in history:
        ax.plot(epochs, history["lr"], label="Learning Rate", color="#8c564b", lw=2)
        ax.set_yscale("log")
    ax.set_title("LR Schedule", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("LR (log scale)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "training_summary_grid.png"), dpi=200)
    plt.close(fig)


# ═══════════════════════════════════════ NEW PLOTS ADDED BELOW ═══════════════════

def plot_f1_vs_threshold(thresh_values, thresh_precision, thresh_recall, thresh_f1, out_path):
    """F1, Precision, Recall vs confidence threshold — helps pick operating point."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresh_values, thresh_f1,        color="#2ca02c", lw=2.5, label="F1",        marker="o", ms=3)
    ax.plot(thresh_values, thresh_precision,  color="#1f77b4", lw=2,   label="Precision",  ls="--")
    ax.plot(thresh_values, thresh_recall,     color="#d62728", lw=2,   label="Recall",     ls=":")
    best_idx = int(max(range(len(thresh_f1)), key=lambda i: thresh_f1[i]))
    best_thr = thresh_values[best_idx]
    best_f1  = thresh_f1[best_idx]
    ax.axvline(best_thr, color="gray", ls="--", alpha=0.6, label=f"Best F1={best_f1:.3f} @ thr={best_thr:.2f}")
    ax.set_title("F1 / Precision / Recall vs Confidence Threshold", fontweight="bold", fontsize=13)
    ax.set_xlabel("Confidence Threshold")
    ax.set_ylabel("Score")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.grid(True, ls="--", alpha=0.5); ax.legend(loc="best", fontsize=10)
    fig.tight_layout(); fig.savefig(out_path, dpi=200); plt.close(fig)


def plot_recall_vs_size(size_bins, size_recall, out_path):
    """Recall vs object size bins — the key tiny-UAV diagnostic."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd"][:len(size_bins)]
    bars = ax.bar(size_bins, size_recall, color=colors, edgecolor="white", linewidth=0.8, width=0.6)
    for bar, r in zip(bars, size_recall):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{r:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_title("Recall vs Object Size (sqrt(wh) bins)", fontweight="bold", fontsize=13)
    ax.set_xlabel("Object Size [px]"); ax.set_ylabel("Recall")
    ax.set_ylim(0, 1.15); ax.grid(True, axis="y", ls="--", alpha=0.5)
    fig.tight_layout(); fig.savefig(out_path, dpi=200); plt.close(fig)


def plot_ap_vs_size(size_bins, size_ap50, out_path):
    """AP50 vs object size bins."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd"][:len(size_bins)]
    bars = ax.bar(size_bins, size_ap50, color=colors, edgecolor="white", linewidth=0.8, width=0.6)
    for bar, v in zip(bars, size_ap50):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_title("AP@50 vs Object Size (sqrt(wh) bins)", fontweight="bold", fontsize=13)
    ax.set_xlabel("Object Size [px]"); ax.set_ylabel("AP@50")
    ax.set_ylim(0, 1.1); ax.grid(True, axis="y", ls="--", alpha=0.5)
    fig.tight_layout(); fig.savefig(out_path, dpi=200); plt.close(fig)


def plot_nwd_iou_diagnostic(displacements, ious, nwds, gt_w, gt_h, out_path):
    """NWD vs IoU as a function of center displacement — research figure."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(displacements, ious,  color="#d62728", lw=2.5, label="IoU")
    ax.plot(displacements, nwds,  color="#1f77b4", lw=2.5, label="NWD", ls="--")
    ax.axhline(0.5, color="gray", ls=":", alpha=0.6, label="Threshold = 0.5")
    ax.set_title(f"IoU vs NWD vs Center Displacement  (GT box {gt_w:.0f}x{gt_h:.0f} px)",
                 fontweight="bold", fontsize=12)
    ax.set_xlabel("Center Displacement [px]")
    ax.set_ylabel("Similarity Score")
    ax.set_ylim(0, 1.05); ax.set_xlim(0, displacements[-1])
    ax.grid(True, ls="--", alpha=0.5); ax.legend(loc="upper right", fontsize=11)
    fig.tight_layout(); fig.savefig(out_path, dpi=200); plt.close(fig)


def plot_val_metrics_epoch(history, plot_dir):
    """Per-epoch validation metrics: AP50, AP50:95, FPPI, center error, per-size recall."""
    os.makedirs(plot_dir, exist_ok=True)
    epochs = history.get("epoch", [])
    if not epochs:
        return

    def _safe(key):
        vals = history.get(key, [])
        return [v if v is not None else float("nan") for v in vals]

    # AP50 vs AP50:95
    if "val_ap50" in history or "val_ap50_95" in history:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, _safe("val_ap50"),    label="AP@50",    color="#1f77b4", lw=2, marker="o", ms=3)
        ax.plot(epochs, _safe("val_ap50_95"), label="AP@50:95", color="#ff7f0e", lw=2, ls="--", marker="s", ms=3)
        ax.set_title("Validation AP Curves", fontweight="bold"); ax.set_xlabel("Epoch"); ax.set_ylabel("AP")
        ax.set_ylim(0, 1.05); ax.grid(True, ls="--", alpha=0.5); ax.legend()
        fig.tight_layout(); fig.savefig(os.path.join(plot_dir, "val_ap_curves.png"), dpi=200); plt.close(fig)

    # Tiny-object recall (all size bins)
    bin_keys = [k for k in history if k.startswith("val_recall_bin")]
    if bin_keys:
        fig, ax = plt.subplots(figsize=(9, 5))
        colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd"]
        for i, bk in enumerate(sorted(bin_keys)):
            label = bk.replace("val_recall_", "Recall ")
            ax.plot(epochs, _safe(bk), label=label, color=colors[i % len(colors)], lw=2, marker="o", ms=2)
        ax.set_title("Recall per Object Size vs Epoch", fontweight="bold"); ax.set_xlabel("Epoch"); ax.set_ylabel("Recall")
        ax.set_ylim(0, 1.05); ax.grid(True, ls="--", alpha=0.5); ax.legend(fontsize=9)
        fig.tight_layout(); fig.savefig(os.path.join(plot_dir, "val_size_recall.png"), dpi=200); plt.close(fig)

    # FPPI
    if "val_fppi" in history:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, _safe("val_fppi"), color="#d62728", lw=2, marker="o", ms=3)
        ax.set_title("False Positives Per Image (FPPI) vs Epoch", fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel("FPPI")
        ax.grid(True, ls="--", alpha=0.5)
        fig.tight_layout(); fig.savefig(os.path.join(plot_dir, "val_fppi.png"), dpi=200); plt.close(fig)

    # Center error
    if "val_center_err" in history:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, _safe("val_center_err"),      label="Abs center err [px]", color="#1f77b4", lw=2)
        ax.plot(epochs, _safe("val_center_err_norm"), label="Norm center err",      color="#ff7f0e", lw=2, ls="--")
        ax.set_title("Center Localization Error vs Epoch", fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Error"); ax.legend()
        ax.grid(True, ls="--", alpha=0.5)
        fig.tight_layout(); fig.savefig(os.path.join(plot_dir, "val_center_error.png"), dpi=200); plt.close(fig)

    # IoU vs NWD on TPs
    if "val_tp_iou" in history or "val_tp_nwd" in history:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, _safe("val_tp_iou"), label="Mean TP IoU",  color="#2ca02c", lw=2, marker="o", ms=3)
        ax.plot(epochs, _safe("val_tp_nwd"), label="Mean TP NWD",  color="#9467bd", lw=2, ls="--", marker="s", ms=3)
        ax.set_title("TP IoU and NWD Quality vs Epoch", fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Score"); ax.set_ylim(0, 1.05); ax.legend()
        ax.grid(True, ls="--", alpha=0.5)
        fig.tight_layout(); fig.savefig(os.path.join(plot_dir, "val_iou_nwd.png"), dpi=200); plt.close(fig)
