"""
Master Runner & Benchmark Comparison for Drone Image Classifiers
================================================================
Sequentially trains and benchmarks all 4 vision models on the drone dataset:
  1. ResNet-18            (https://huggingface.co/microsoft/resnet-18)
  2. MobileNet-v3-Large   (https://huggingface.co/qualcomm/MobileNet-v3-Large)
  3. EfficientNet-B0      (https://huggingface.co/google/efficientnet-b0)
  4. GhostNet-100         (https://huggingface.co/timm/ghostnet_100.in1k)

Features:
  - Trains each model with AMP, early stopping, and automatic logging.
  - Generates individual metric logs, confusion matrices, ROC/PR curves.
  - Automatically compiles a Master Benchmark Dashboard comparing:
      * Accuracy, Precision, Recall, F1
      * ROC-AUC
      * Training & Validation Loss curves over epochs
      * Validation F1 progression
      * Model Parameter Count
  - Saves benchmark_summary.md and benchmark_summary.csv for easy reporting.

Usage:
  python run_all_classifiers.py
  python run_all_classifiers.py --data-root dataset_full --epochs 50 --batch-size 32
"""

import os
import sys
import json
import csv
import time
import argparse
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODELS_TO_BENCHMARK = [
    ("resnet18", "ResNet-18", "microsoft/resnet-18"),
    ("mobilenetv3_large", "MobileNet-v3-Large", "qualcomm/MobileNet-v3-Large"),
    ("efficientnet_b0", "EfficientNet-B0", "google/efficientnet-b0"),
    ("ghostnet_100", "GhostNet-100", "timm/ghostnet_100.in1k"),
]


def run_benchmark(data_root="dataset_full", epochs=50, batch_size=32, lr=3e-4, img_size=224, patience=10, device="cuda", out_dir="runs_classifiers"):
    os.makedirs(out_dir, exist_ok=True)
    all_summaries = []

    print("\n" + "█" * 80)
    print("  STARTING BENCHMARK FOR 4 SOTA CLASSIFIERS ON DRONE DATASET")
    print(f"  Dataset: {data_root} | Epochs: {epochs} | Batch: {batch_size} | Device: {device}")
    print("█" * 80 + "\n")

    for model_key, display_name, hf_source in MODELS_TO_BENCHMARK:
        print("\n" + "=" * 80)
        print(f"  [RUNNING {display_name}] ({hf_source})")
        print("=" * 80)

        cmd = [
            sys.executable, "train_classifier.py",
            "--model", model_key,
            "--data-root", data_root,
            "--epochs", str(epochs),
            "--batch-size", str(batch_size),
            "--lr", str(lr),
            "--img-size", str(img_size),
            "--patience", str(patience),
            "--device", device,
            "--out-dir", out_dir,
        ]

        t_start = time.time()
        try:
            subprocess.run(cmd, check=True)
            elapsed_min = (time.time() - t_start) / 60.0
            print(f"  --> {display_name} training finished in {elapsed_min:.1f} minutes.")
        except subprocess.CalledProcessError as e:
            print(f"  [ERROR] Training failed for {display_name}: {e}")
            continue

        # Load metrics summary
        summary_path = os.path.join(out_dir, model_key, "metrics_summary.json")
        if os.path.isfile(summary_path):
            with open(summary_path, "r") as f:
                s = json.load(f)
                s["train_time_min"] = round(elapsed_min, 1)
                all_summaries.append(s)

    # Generate Comparative Dashboard
    generate_comparison_dashboard(out_dir, all_summaries)
    print_and_save_summary_table(out_dir, all_summaries)


def generate_comparison_dashboard(out_dir: str, summaries: list):
    if not summaries:
        print("[Warning] No completed model summaries found to plot comparison.")
        return

    print("\nGenerating Master Benchmark Comparison Dashboard...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    model_names = [s["display_name"] for s in summaries]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"][:len(summaries)]

    # 1. Bar Chart: Classification Metrics (Accuracy, Precision, Recall, F1)
    ax1 = axes[0, 0]
    metrics = ["best_accuracy", "best_precision", "best_recall", "best_f1"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]

    x = np.arange(len(metric_labels))
    width = 0.8 / len(summaries)

    for idx, s in enumerate(summaries):
        values = [s.get(m, 0.0) * 100 for m in metrics]
        offset = (idx - len(summaries) / 2 + 0.5) * width
        bars = ax1.bar(x + offset, values, width, label=s["display_name"], color=colors[idx], alpha=0.9)
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.8, f"{h:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax1.set_title("Performance Metrics Comparison", fontweight="bold", fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metric_labels, fontweight="bold")
    ax1.set_ylabel("Score (%)", fontweight="bold")
    ax1.set_ylim(0, 115)
    ax1.grid(True, axis="y", ls="--", alpha=0.5)
    ax1.legend(loc="lower right")

    # 2. Bar Chart: ROC-AUC & Parameters (M)
    ax2 = axes[0, 1]
    x_models = np.arange(len(model_names))
    aucs = [s.get("best_auc", 0.0) for s in summaries]
    params = [s.get("total_params_M", 0.0) for s in summaries]

    ax2_twin = ax2.twinx()
    b1 = ax2.bar(x_models - 0.2, aucs, 0.38, label="ROC-AUC", color="#2ca02c", alpha=0.85)
    b2 = ax2_twin.bar(x_models + 0.2, params, 0.38, label="Parameters (M)", color="#9467bd", alpha=0.85)

    for bar in b1:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar in b2:
        h = bar.get_height()
        ax2_twin.text(bar.get_x() + bar.get_width() / 2, h + 0.2, f"{h:.1f}M", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax2.set_title("ROC-AUC Score vs Model Complexity", fontweight="bold", fontsize=13)
    ax2.set_xticks(x_models)
    ax2.set_xticklabels(model_names, fontweight="bold")
    ax2.set_ylabel("ROC-AUC Score (Green)", color="#2ca02c", fontweight="bold")
    ax2.set_ylim(0, 1.15)
    ax2_twin.set_ylabel("Total Parameters (M) (Purple)", color="#9467bd", fontweight="bold")
    ax2_twin.set_ylim(0, max(params) * 1.35 if params else 30)
    ax2.grid(True, axis="y", ls="--", alpha=0.5)

    # 3. Learning Curves: Validation F1 vs Epoch
    ax3 = axes[1, 0]
    for idx, (m_key, d_name, _) in enumerate(MODELS_TO_BENCHMARK):
        hist_path = os.path.join(out_dir, m_key, "history.json")
        if os.path.isfile(hist_path):
            with open(hist_path) as f:
                h = json.load(f)
                epochs = h.get("epoch", [])
                val_f1 = h.get("val_f1", [])
                ax3.plot(epochs, val_f1, label=d_name, color=colors[idx % len(colors)], lw=2, marker="o", ms=3)

    ax3.set_title("Validation F1 Score vs Epoch", fontweight="bold", fontsize=13)
    ax3.set_xlabel("Epoch", fontweight="bold")
    ax3.set_ylabel("Validation F1", fontweight="bold")
    ax3.set_ylim(0.0, 1.05)
    ax3.grid(True, ls="--", alpha=0.5)
    ax3.legend(loc="lower right")

    # 4. Learning Curves: Validation Loss vs Epoch
    ax4 = axes[1, 1]
    for idx, (m_key, d_name, _) in enumerate(MODELS_TO_BENCHMARK):
        hist_path = os.path.join(out_dir, m_key, "history.json")
        if os.path.isfile(hist_path):
            with open(hist_path) as f:
                h = json.load(f)
                epochs = h.get("epoch", [])
                val_loss = h.get("val_loss", [])
                ax4.plot(epochs, val_loss, label=d_name, color=colors[idx % len(colors)], lw=2, marker="s", ms=3)

    ax4.set_title("Validation CrossEntropy Loss vs Epoch", fontweight="bold", fontsize=13)
    ax4.set_xlabel("Epoch", fontweight="bold")
    ax4.set_ylabel("Val Loss", fontweight="bold")
    ax4.grid(True, ls="--", alpha=0.5)
    ax4.legend(loc="upper right")

    fig.suptitle("Drone vs No-Drone Classification Benchmark: 4 SOTA Architectures", fontsize=16, fontweight="bold", y=0.99)
    fig.tight_layout()
    plot_path = os.path.join(out_dir, "benchmark_comparison.png")
    fig.savefig(plot_path, dpi=250)
    plt.close(fig)
    print(f"  ==> Benchmark comparison plot saved to: {plot_path}")


def print_and_save_summary_table(out_dir: str, summaries: list):
    if not summaries:
        return

    csv_path = os.path.join(out_dir, "benchmark_summary.csv")
    md_path = os.path.join(out_dir, "benchmark_summary.md")

    headers = ["Model", "HF Source", "Params (M)", "Accuracy (%)", "Precision", "Recall", "F1-Score", "ROC-AUC", "Train Time (m)"]
    rows = []

    for s in summaries:
        rows.append([
            s["display_name"],
            s.get("hf_source", ""),
            f"{s.get('total_params_M', 0.0):.2f}",
            f"{s.get('best_accuracy', 0.0) * 100:.2f}%",
            f"{s.get('best_precision', 0.0):.4f}",
            f"{s.get('best_recall', 0.0):.4f}",
            f"{s.get('best_f1', 0.0):.4f}",
            f"{s.get('best_auc', 0.0):.4f}",
            f"{s.get('train_time_min', 0.0)}",
        ])

    # Save CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    # Save Markdown
    md_lines = [
        "# Drone vs No-Drone Classification Benchmark Results\n",
        f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for r in rows:
        md_lines.append("| " + " | ".join(r) + " |")

    with open(md_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")

    # Print to console
    print("\n" + "=" * 90)
    print("  FINAL CLASSIFIER BENCHMARK SUMMARY")
    print("=" * 90)
    print(f"{'Model':<20} | {'Params':<8} | {'Accuracy':<10} | {'Precision':<9} | {'Recall':<8} | {'F1-Score':<8} | {'ROC-AUC':<8}")
    print("-" * 90)
    for s in summaries:
        print(f"{s['display_name']:<20} | {s.get('total_params_M', 0):>6.2f}M | {s.get('best_accuracy', 0)*100:>8.2f}% | {s.get('best_precision', 0):>9.4f} | {s.get('best_recall', 0):>8.4f} | {s.get('best_f1', 0):>8.4f} | {s.get('best_auc', 0):>8.4f}")
    print("=" * 90)
    print(f"Full markdown report saved to: {md_path}")
    print(f"Full CSV summary saved to:    {csv_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master runner for 4 drone classifiers benchmark")
    parser.add_argument("--data-root", type=str, default="dataset_full", help="Path to dataset root")
    parser.add_argument("--epochs", type=int, default=50, help="Epochs per model")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (32 recommended on GPU)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--img-size", type=int, default=224, help="Image size (224 standard)")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--out-dir", type=str, default="runs_classifiers", help="Output directory")
    args = parser.parse_args()

    run_benchmark(
        data_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        img_size=args.img_size,
        patience=args.patience,
        device=args.device,
        out_dir=args.out_dir,
    )
