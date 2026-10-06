"""
Evaluate Best Classifier Checkpoints on CARS Dataset (drone vs no_drone)
========================================================================
Evaluates the 4 trained vision models on:
  - CARS/class1_B : no_drone -> Label 0 (Negative Background)
  - CARS/class2_D : drone    -> Label 1 (Positive Drone)

Calculates standard binary classification metrics:
  - Overall Accuracy (%)
  - Drone Recall / Accuracy (%) : TP / total_drone
  - No-Drone Accuracy (%)       : TN / total_no_drone (Specificity)
  - False Positive Rate (FPR)   : FP / total_no_drone
  - Precision, F1-Score, and ROC-AUC
  - Confusion Matrix (TN, FP, FN, TP)

Generates:
  - runs_classifiers/cars_eval/cars_comparison_plot.png
  - runs_classifiers/cars_eval/cars_benchmark_summary.md
  - runs_classifiers/cars_eval/cars_benchmark_summary.csv

Usage:
  python eval_cars_dataset.py
  python eval_cars_dataset.py --cars-dir CARS --device cuda --batch-size 64
"""

import os
import sys
import glob
import json
import csv
import argparse
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 4 Models
MODELS_TO_EVAL = [
    ("resnet18", "ResNet-18", "resnet18"),
    ("mobilenetv3_large", "MobileNet-v3-Large", "mobilenetv3_large_100"),
    ("efficientnet_b0", "EfficientNet-B0", "efficientnet_b0"),
    ("ghostnet_100", "GhostNet-100", "ghostnet_100"),
]

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


class CARSDataset(Dataset):
    """Loads no_drone (class1_B -> 0) and drone (class2_D -> 1) images."""
    def __init__(self, samples: List[Tuple[str, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            with open(path, "rb") as f:
                img = Image.open(f).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (0, 0, 0))

        if self.transform is not None:
            img = self.transform(img)

        return img, label, path


def find_cars_folder(user_path: str = None) -> str:
    """Finds the CARS folder automatically."""
    candidates = [
        user_path,
        "CARS",
        "cars",
        "../CARS",
        "dataset/CARS",
        "dataset_full/CARS",
    ]
    for c in candidates:
        if c and os.path.isdir(c):
            b_exists = any(os.path.isdir(os.path.join(c, x)) for x in ["class1_B", "class1_b", "class1", "no_drone"])
            d_exists = any(os.path.isdir(os.path.join(c, x)) for x in ["class2_D", "class2_d", "class2", "drone"])
            if b_exists or d_exists:
                return os.path.abspath(c)
    if user_path and os.path.isdir(user_path):
        return os.path.abspath(user_path)
    return "CARS"


def scan_cars_dataset(cars_dir: str) -> List[Tuple[str, int]]:
    """Scans class1_B (no_drone=0) and class2_D (drone=1)."""
    if not os.path.isdir(cars_dir):
        raise FileNotFoundError(f"CARS folder not found at '{cars_dir}'. Please specify --cars-dir.")

    samples = []
    # class1_B = no_drone (0), class2_D = drone (1)
    no_drone_folders = ["class1_B", "class1_b", "class1", "no_drone"]
    drone_folders = ["class2_D", "class2_d", "class2", "drone", "drones"]

    def _collect(subfolders, label):
        found = False
        for sf in subfolders:
            folder_path = os.path.join(cars_dir, sf)
            if os.path.isdir(folder_path):
                found = True
                for root, _, files in os.walk(folder_path):
                    for f in files:
                        if f.lower().endswith(IMG_EXTS):
                            samples.append((os.path.join(root, f), label))
                break
        return found

    found_no_drone = _collect(no_drone_folders, 0)
    found_drone = _collect(drone_folders, 1)

    if not found_no_drone and not found_drone:
        subdirs = sorted([d for d in os.listdir(cars_dir) if os.path.isdir(os.path.join(cars_dir, d))])
        print(f"[CARS] Subdirectories found: {subdirs}")
        for d in subdirs:
            lbl = 0 if "1" in d or "b" in d.lower() else 1
            for root, _, files in os.walk(os.path.join(cars_dir, d)):
                for f in files:
                    if f.lower().endswith(IMG_EXTS):
                        samples.append((os.path.join(root, f), lbl))

    return samples


def find_checkpoint(model_key: str, checkpoints_root: str = "runs_classifiers") -> str:
    """Finds best checkpoint for a given model."""
    search_paths = [
        os.path.join(checkpoints_root, model_key, "checkpoint_best.pt"),
        os.path.join(checkpoints_root, model_key, "best.pt"),
        os.path.join(checkpoints_root, f"{model_key}_best.pt"),
        os.path.join(checkpoints_root, f"checkpoints_{model_key}", "checkpoint_best.pt"),
        os.path.join(checkpoints_root, f"checkpoints_{model_key}", "s3det_best.pt"),
        os.path.join(checkpoints_root, model_key, "checkpoint_latest.pt"),
    ]
    for p in search_paths:
        if os.path.isfile(p):
            return p

    matches = glob.glob(os.path.join(checkpoints_root, f"*{model_key}*", "*.pt"))
    for m in matches:
        if "best" in m.lower():
            return m
    if matches:
        return matches[0]

    return None


def extract_state_dict(ckpt):
    """Safely extracts weights regardless of container."""
    if isinstance(ckpt, dict):
        for key in ["model_state", "model", "state_dict"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    return ckpt


def load_model(timm_name: str, ckpt_path: str, device: torch.device):
    """Creates model and loads checkpoint weights."""
    import timm
    model = timm.create_model(timm_name, pretrained=False, num_classes=2)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = extract_state_dict(ckpt)

    cleaned_dict = {}
    for k, v in state_dict.items():
        clean_k = k.replace("module.", "")
        cleaned_dict[clean_k] = v

    try:
        model.load_state_dict(cleaned_dict, strict=True)
    except RuntimeError:
        missing, unexpected = model.load_state_dict(cleaned_dict, strict=False)
        print(f"    [Notice] Non-strict load for {timm_name}: missing={len(missing)}, unexpected={len(unexpected)}")

    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def evaluate_on_cars(model, loader, device):
    all_targets = []
    all_preds = []
    all_probs = []

    for images, targets, _ in loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=-1)
        preds = torch.argmax(probs, dim=-1)

        all_targets.extend(targets.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
        all_probs.extend(probs[:, 1].cpu().numpy().tolist())

    targets_np = np.array(all_targets)
    preds_np = np.array(all_preds)
    probs_np = np.array(all_probs)
    n = max(len(targets_np), 1)

    # Class 0: no_drone, Class 1: drone
    tp = int(np.sum((preds_np == 1) & (targets_np == 1)))  # Drone detected as Drone
    fn = int(np.sum((preds_np == 0) & (targets_np == 1)))  # Drone missed (predicted as no_drone)
    tn = int(np.sum((preds_np == 0) & (targets_np == 0)))  # no_drone correctly classified
    fp = int(np.sum((preds_np == 1) & (targets_np == 0)))  # no_drone falsely predicted as Drone

    total_no_drone = tn + fp
    total_drone = tp + fn

    overall_acc = float((tp + tn) / n)
    no_drone_acc = float(tn / max(total_no_drone, 1))      # Specificity
    drone_acc = float(tp / max(total_drone, 1))            # Recall / Sensitivity
    fpr = float(fp / max(total_no_drone, 1))               # False Positive Rate
    precision = float(tp / max(tp + fp, 1e-9))
    f1 = float(2 * precision * drone_acc / max(precision + drone_acc, 1e-9))

    auc = 0.5
    try:
        from sklearn.metrics import roc_auc_score
        if len(np.unique(targets_np)) > 1:
            auc = float(roc_auc_score(targets_np, probs_np))
    except Exception:
        pass

    return {
        "overall_accuracy": overall_acc,
        "no_drone_accuracy": no_drone_acc,
        "drone_accuracy": drone_acc,
        "fpr": fpr,
        "precision": precision,
        "recall": drone_acc,
        "f1": f1,
        "auc": auc,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "total_no_drone": total_no_drone,
        "total_drone": total_drone,
        "total_images": n,
    }


def plot_cars_results(results: List[Dict], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    if not results:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    model_names = [r["display_name"] for r in results]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"][:len(results)]

    # 1. Overall, No-Drone, and Drone Accuracy Comparison
    ax1 = axes[0, 0]
    metrics = ["overall_accuracy", "no_drone_accuracy", "drone_accuracy"]
    labels = ["Overall Accuracy", "no_drone Accuracy", "drone Recall"]
    x = np.arange(len(labels))
    width = 0.8 / len(results)

    for idx, r in enumerate(results):
        vals = [r[m] * 100 for m in metrics]
        offset = (idx - len(results) / 2 + 0.5) * width
        bars = ax1.bar(x + offset, vals, width, label=r["display_name"], color=colors[idx], alpha=0.9)
        for b in bars:
            h = b.get_height()
            ax1.text(b.get_x() + b.get_width() / 2, h + 1.0, f"{h:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax1.set_title("Accuracy Breakdown (drone vs no_drone)", fontweight="bold", fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontweight="bold")
    ax1.set_ylabel("Accuracy (%)", fontweight="bold")
    ax1.set_ylim(0, 115)
    ax1.grid(True, axis="y", ls="--", alpha=0.5)
    ax1.legend(loc="lower right")

    # 2. False Positive Rate (FPR)
    ax2 = axes[0, 1]
    x_models = np.arange(len(model_names))
    fpr_rates = [r["fpr"] * 100 for r in results]
    bars_fa = ax2.bar(x_models, fpr_rates, color=["#d62728" if fa > 10 else "#2ca02c" for fa in fpr_rates], width=0.5)
    for b in bars_fa:
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width() / 2, h + 0.5, f"{h:.2f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax2.set_title("False Positive Rate (no_drone misclassified as drone)", fontweight="bold", fontsize=13)
    ax2.set_xticks(x_models)
    ax2.set_xticklabels(model_names, fontweight="bold")
    ax2.set_ylabel("False Positive Rate (%) [Lower is Better]", fontweight="bold")
    ax2.set_ylim(0, max(fpr_rates + [15]) * 1.3)
    ax2.grid(True, axis="y", ls="--", alpha=0.5)

    # 3 & 4: Confusion Matrices
    for panel_idx, pair in enumerate([results[:2], results[2:4]]):
        ax = axes[1, panel_idx]
        ax.axis("off")
        title = "Confusion Matrices: " + " & ".join(r["display_name"] for r in pair)
        ax.set_title(title, fontweight="bold", fontsize=13, pad=10)

        lines = []
        for r in pair:
            cm_text = (
                f"── {r['display_name']} ──\n"
                f"                  Predicted no_drone       Predicted drone\n"
                f"  Actual no_drone   {r['tn']:>8d} ({r['no_drone_accuracy']*100:5.1f}%)      {r['fp']:>7d} ({r['fpr']*100:5.1f}% FP)\n"
                f"  Actual drone      {r['fn']:>8d} ({r['fn']/max(r['total_drone'],1)*100:5.1f}% FN)    {r['tp']:>7d} ({r['drone_accuracy']*100:5.1f}% TP)\n"
                f"  Accuracy: {r['overall_accuracy']*100:.2f}%  |  F1: {r['f1']:.4f}  |  ROC-AUC: {r['auc']:.4f}\n"
            )
            lines.append(cm_text)

        ax.text(0.05, 0.5, "\n\n".join(lines), fontfamily="monospace", fontsize=10, va="center",
                bbox=dict(boxstyle="round,pad=1", facecolor="#f8f9fa", edgecolor="#ced4da"))

    fig.suptitle("CARS Test Dataset Benchmark: drone (class2_D) vs no_drone (class1_B)", fontsize=16, fontweight="bold", y=0.99)
    fig.tight_layout()
    plot_path = os.path.join(out_dir, "cars_comparison_plot.png")
    fig.savefig(plot_path, dpi=250)
    plt.close(fig)
    print(f"\n  ==> Comparison plot saved to: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate best classifier checkpoints on CARS (drone vs no_drone)")
    parser.add_argument("--cars-dir", type=str, default="CARS", help="Path to CARS dataset directory")
    parser.add_argument("--checkpoints-dir", type=str, default="runs_classifiers", help="Path to checkpoints folder")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--img-size", type=int, default=224, help="Image resolution")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--out-dir", type=str, default="runs_classifiers/cars_eval", help="Output directory")
    args = parser.parse_args()

    cars_folder = find_cars_folder(args.cars_dir)
    print("\n" + "█" * 80)
    print("  EVALUATING 4 CLASSIFIER CHECKPOINTS ON CARS DATASET")
    print(f"  CARS Directory: {cars_folder}")
    print(f"  Checkpoints Directory: {args.checkpoints_dir}")
    print("█" * 80 + "\n")

    try:
        samples = scan_cars_dataset(cars_folder)
    except FileNotFoundError as e:
        print(f"[Error] {e}")
        sys.exit(1)

    n_no_drone = sum(1 for _, lbl in samples if lbl == 0)
    n_drone = sum(1 for _, lbl in samples if lbl == 1)
    print(f"  Found {len(samples)} total test images in CARS:")
    print(f"    - no_drone (class1_B): {n_no_drone} images")
    print(f"    - drone    (class2_D): {n_drone} images\n")

    if len(samples) == 0:
        print("[Error] No valid images found in CARS. Please check folder contents.")
        sys.exit(1)

    device = torch.device(args.device if (torch.cuda.is_available() and "cuda" in args.device) else "cpu")
    print(f"  Inference Device: {device} | Batch Size: {args.batch_size}")

    eval_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = CARSDataset(samples, transform=eval_transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    results = []
    for model_key, display_name, timm_name in MODELS_TO_EVAL:
        print("-" * 80)
        print(f"  Testing {display_name} ({timm_name})...")
        ckpt_path = find_checkpoint(model_key, args.checkpoints_dir)
        if not ckpt_path:
            print(f"    [Warning] Checkpoint for '{model_key}' not found in {args.checkpoints_dir}. Skipping.")
            continue

        print(f"    Loaded Checkpoint: {ckpt_path}")
        try:
            model = load_model(timm_name, ckpt_path, device)
            res = evaluate_on_cars(model, loader, device)
            res["model_key"] = model_key
            res["display_name"] = display_name
            res["checkpoint"] = ckpt_path
            results.append(res)

            print(f"    --> Overall Accuracy:      {res['overall_accuracy']*100:.2f}%")
            print(f"    --> Drone Recall (TP):     {res['drone_accuracy']*100:.2f}% ({res['tp']}/{res['total_drone']})")
            print(f"    --> no_drone Accuracy:     {res['no_drone_accuracy']*100:.2f}% ({res['tn']}/{res['total_no_drone']})")
            print(f"    --> False Positive Rate:   {res['fpr']*100:.2f}% ({res['fp']} false alarms)")
            print(f"    --> F1-Score:              {res['f1']:.4f} | ROC-AUC: {res['auc']:.4f}")
        except Exception as e:
            print(f"    [Error evaluating {display_name}]: {e}")

    if not results:
        print("\n[Error] No models were successfully evaluated.")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    plot_cars_results(results, args.out_dir)

    csv_path = os.path.join(args.out_dir, "cars_benchmark_summary.csv")
    md_path = os.path.join(args.out_dir, "cars_benchmark_summary.md")

    headers = [
        "Model", "Overall Acc (%)", "no_drone Acc (%)", "drone Recall (%)",
        "False Positives (FP)", "Missed Drones (FN)", "Precision", "F1-Score", "ROC-AUC"
    ]
    rows = []
    for r in results:
        rows.append([
            r["display_name"],
            f"{r['overall_accuracy']*100:.2f}%",
            f"{r['no_drone_accuracy']*100:.2f}%",
            f"{r['drone_accuracy']*100:.2f}%",
            f"{r['fp']} / {r['total_no_drone']} ({r['fpr']*100:.2f}%)",
            f"{r['fn']} / {r['total_drone']}",
            f"{r['precision']:.4f}",
            f"{r['f1']:.4f}",
            f"{r['auc']:.4f}",
        ])

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    md_content = [
        "# CARS Test Dataset Evaluation: drone vs no_drone\n",
        f"- **Dataset Path**: `{cars_folder}`\n",
        f"- **Total no_drone (`class1_B`)**: {n_no_drone}\n",
        f"- **Total drone (`class2_D`)**: {n_drone}\n\n",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        md_content.append("| " + " | ".join(row) + " |")

    with open(md_path, "w") as f:
        f.write("\n".join(md_content) + "\n")

    print("\n" + "=" * 90)
    print("  FINAL CARS BENCHMARK SUMMARY (drone vs no_drone)")
    print("=" * 90)
    print(f"{'Model':<20} | {'Overall Acc':<11} | {'no_drone Acc':<12} | {'Drone Recall':<12} | {'FPR':<7} | {'F1':<6} | {'AUC':<6}")
    print("-" * 90)
    for r in results:
        print(f"{r['display_name']:<20} | {r['overall_accuracy']*100:>10.2f}% | {r['no_drone_accuracy']*100:>11.2f}% | {r['drone_accuracy']*100:>11.2f}% | {r['fpr']*100:>6.2f}% | {r['f1']:>6.4f} | {r['auc']:>6.4f}")
    print("=" * 90)
    print(f"\nSaved Markdown Report to: {md_path}")
    print(f"Saved CSV Report to:      {csv_path}\n")


if __name__ == "__main__":
    main()
