"""
Evaluate Best Classifier Checkpoints on the CARS (Birds vs Drones) Dataset
==========================================================================
Evaluates the 4 trained vision models on:
  - CARS/class1_B : Birds   -> Label 0 (Negative / No Drone)
  - CARS/class2_D : Drones  -> Label 1 (Positive / Drone)

Calculates:
  - Overall Accuracy (%)
  - Bird Accuracy / Specificity (% of birds correctly recognized as NOT drones)
  - Drone Accuracy / Recall (% of actual drones detected)
  - Bird False Alarm Rate (% of birds falsely flagged as drones)
  - Precision, F1-Score, and ROC-AUC
  - Confusion Matrix (TN, FP, FN, TP)

Generates:
  - runs_classifiers/cars_eval/cars_comparison_plot.png
  - runs_classifiers/cars_eval/cars_benchmark_summary.md
  - runs_classifiers/cars_eval/cars_benchmark_summary.csv

Usage:
  python eval_cars_dataset.py
  python eval_cars_dataset.py --cars-dir /path/to/CARS --device cuda
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

# Model Registry matching train_classifier.py
MODELS_TO_EVAL = [
    ("resnet18", "ResNet-18", "resnet18"),
    ("mobilenetv3_large", "MobileNet-v3-Large", "mobilenetv3_large_100"),
    ("efficientnet_b0", "EfficientNet-B0", "efficientnet_b0"),
    ("ghostnet_100", "GhostNet-100", "ghostnet_100"),
]

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


class CARSDataset(Dataset):
    """Loads bird images (class1_B -> 0) and drone images (class2_D -> 1)."""
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
    """Finds the CARS folder automatically if possible."""
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
            # Check for class1_B and class2_D
            b_exists = any(os.path.isdir(os.path.join(c, x)) for x in ["class1_B", "class1_b", "class1", "birds"])
            d_exists = any(os.path.isdir(os.path.join(c, x)) for x in ["class2_D", "class2_d", "class2", "drones"])
            if b_exists or d_exists:
                return os.path.abspath(c)
    if user_path and os.path.isdir(user_path):
        return os.path.abspath(user_path)
    return "CARS"


def scan_cars_dataset(cars_dir: str) -> List[Tuple[str, int]]:
    """Scans class1_B (birds=0) and class2_D (drones=1)."""
    if not os.path.isdir(cars_dir):
        raise FileNotFoundError(f"CARS folder not found at '{cars_dir}'. Please specify --cars-dir.")

    samples = []
    # Birds -> class 0
    bird_folders = ["class1_B", "class1_b", "class1", "birds", "bird"]
    drone_folders = ["class2_D", "class2_d", "class2", "drones", "drone"]

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

    found_birds = _collect(bird_folders, 0)
    found_drones = _collect(drone_folders, 1)

    if not found_birds and not found_drones:
        # Fallback: scan all subdirectories
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

    # Fuzzy match
    matches = glob.glob(os.path.join(checkpoints_root, f"*{model_key}*", "*.pt"))
    for m in matches:
        if "best" in m.lower():
            return m
    if matches:
        return matches[0]

    return None


def extract_state_dict(ckpt):
    """Safely extracts model weights regardless of how checkpoint was saved."""
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

    # Clean state dict keys if needed
    cleaned_dict = {}
    for k, v in state_dict.items():
        clean_k = k.replace("module.", "")
        cleaned_dict[clean_k] = v

    try:
        model.load_state_dict(cleaned_dict, strict=True)
    except RuntimeError:
        # Load non-strictly if head or wrappers mismatch
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

    # Class 0: Birds, Class 1: Drones
    tp = int(np.sum((preds_np == 1) & (targets_np == 1)))  # Drone detected as Drone
    fn = int(np.sum((preds_np == 0) & (targets_np == 1)))  # Drone missed (predicted as Bird)
    tn = int(np.sum((preds_np == 0) & (targets_np == 0)))  # Bird correctly rejected as Bird
    fp = int(np.sum((preds_np == 1) & (targets_np == 0)))  # Bird falsely flagged as Drone

    total_birds = tn + fp
    total_drones = tp + fn

    overall_acc = float((tp + tn) / n)
    bird_acc = float(tn / max(total_birds, 1))           # Specificity
    drone_acc = float(tp / max(total_drones, 1))         # Recall / Sensitivity
    bird_false_alarm = float(fp / max(total_birds, 1))   # False Positive Rate on Birds
    precision = float(tp / max(tp + fp, 1e-9))
    f1 = float(2 * precision * drone_acc / max(precision + drone_acc, 1e-9))

    # ROC AUC
    auc = 0.5
    try:
        from sklearn.metrics import roc_auc_score
        if len(np.unique(targets_np)) > 1:
            auc = float(roc_auc_score(targets_np, probs_np))
    except Exception:
        pass

    return {
        "overall_accuracy": overall_acc,
        "bird_accuracy": bird_acc,
        "drone_accuracy": drone_acc,
        "bird_false_alarm": bird_false_alarm,
        "precision": precision,
        "recall": drone_acc,
        "f1": f1,
        "auc": auc,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "total_birds": total_birds,
        "total_drones": total_drones,
        "total_images": n,
    }


def plot_cars_results(results: List[Dict], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    if not results:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    model_names = [r["display_name"] for r in results]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"][:len(results)]

    # 1. Overall, Bird, and Drone Accuracy Comparison
    ax1 = axes[0, 0]
    metrics = ["overall_accuracy", "bird_accuracy", "drone_accuracy"]
    labels = ["Overall Accuracy", "Bird Accuracy (Specificity)", "Drone Accuracy (Recall)"]
    x = np.arange(len(labels))
    width = 0.8 / len(results)

    for idx, r in enumerate(results):
        vals = [r[m] * 100 for m in metrics]
        offset = (idx - len(results) / 2 + 0.5) * width
        bars = ax1.bar(x + offset, vals, width, label=r["display_name"], color=colors[idx], alpha=0.9)
        for b in bars:
            h = b.get_height()
            ax1.text(b.get_x() + b.get_width() / 2, h + 1.0, f"{h:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax1.set_title("Accuracy Breakdown on CARS Dataset", fontweight="bold", fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontweight="bold")
    ax1.set_ylabel("Accuracy (%)", fontweight="bold")
    ax1.set_ylim(0, 115)
    ax1.grid(True, axis="y", ls="--", alpha=0.5)
    ax1.legend(loc="lower right")

    # 2. Bird False Alarm Rate (Lower is Better!)
    ax2 = axes[0, 1]
    x_models = np.arange(len(model_names))
    fa_rates = [r["bird_false_alarm"] * 100 for r in results]
    bars_fa = ax2.bar(x_models, fa_rates, color=["#d62728" if fa > 10 else "#2ca02c" for fa in fa_rates], width=0.5)
    for b in bars_fa:
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width() / 2, h + 0.5, f"{h:.2f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax2.set_title("Bird False Alarm Rate (% Birds Misclassified as Drones)", fontweight="bold", fontsize=13)
    ax2.set_xticks(x_models)
    ax2.set_xticklabels(model_names, fontweight="bold")
    ax2.set_ylabel("False Alarm Rate (%) [Lower is Better]", fontweight="bold")
    ax2.set_ylim(0, max(fa_rates + [15]) * 1.3)
    ax2.grid(True, axis="y", ls="--", alpha=0.5)

    # 3 & 4: Confusion Matrices
    # Subplot 3: First 2 models
    # Subplot 4: Next 2 models
    for panel_idx, pair in enumerate([results[:2], results[2:4]]):
        ax = axes[1, panel_idx]
        ax.axis("off")
        title = "Confusion Matrices: " + " & ".join(r["display_name"] for r in pair)
        ax.set_title(title, fontweight="bold", fontsize=13, pad=10)

        # Draw mini confusion matrices as text table
        lines = []
        for r in pair:
            cm_text = (
                f"── {r['display_name']} ──\n"
                f"                Predicted Bird       Predicted Drone\n"
                f"  Actual Bird   {r['tn']:>7d} ({r['bird_accuracy']*100:5.1f}%)    {r['fp']:>7d} ({r['bird_false_alarm']*100:5.1f}% FP)\n"
                f"  Actual Drone  {r['fn']:>7d} ({r['fn']/max(r['total_drones'],1)*100:5.1f}% FN)  {r['tp']:>7d} ({r['drone_accuracy']*100:5.1f}% TP)\n"
                f"  Accuracy: {r['overall_accuracy']*100:.2f}%  |  F1: {r['f1']:.4f}  |  AUC: {r['auc']:.4f}\n"
            )
            lines.append(cm_text)

        ax.text(0.05, 0.5, "\n\n".join(lines), fontfamily="monospace", fontsize=10, va="center",
                bbox=dict(boxstyle="round,pad=1", facecolor="#f8f9fa", edgecolor="#ced4da"))

    fig.suptitle("CARS Test Dataset: Birds (class1_B) vs Drones (class2_D)", fontsize=16, fontweight="bold", y=0.99)
    fig.tight_layout()
    plot_path = os.path.join(out_dir, "cars_comparison_plot.png")
    fig.savefig(plot_path, dpi=250)
    plt.close(fig)
    print(f"\n  ==> Comparison plot saved to: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate best classifier checkpoints on CARS (Birds vs Drones)")
    parser.add_argument("--cars-dir", type=str, default="CARS", help="Path to CARS dataset directory")
    parser.add_argument("--checkpoints-dir", type=str, default="runs_classifiers", help="Path to checkpoints folder")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
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

    # 1. Scan CARS samples
    try:
        samples = scan_cars_dataset(cars_folder)
    except FileNotFoundError as e:
        print(f"[Error] {e}")
        sys.exit(1)

    n_birds = sum(1 for _, lbl in samples if lbl == 0)
    n_drones = sum(1 for _, lbl in samples if lbl == 1)
    print(f"  Found {len(samples)} total test images in CARS:")
    print(f"    - Birds  (class1_B): {n_birds} images")
    print(f"    - Drones (class2_D): {n_drones} images\n")

    if len(samples) == 0:
        print("[Error] No valid images found in CARS. Please check folder contents.")
        sys.exit(1)

    device = torch.device(args.device if (torch.cuda.is_available() and "cuda" in args.device) else "cpu")
    print(f"  Inference Device: {device}")

    eval_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = CARSDataset(samples, transform=eval_transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 2. Evaluate Each Model
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
            print(f"    --> Bird Accuracy (Spec):  {res['bird_accuracy']*100:.2f}% ({res['tn']}/{res['total_birds']})")
            print(f"    --> Drone Accuracy (Rec):  {res['drone_accuracy']*100:.2f}% ({res['tp']}/{res['total_drones']})")
            print(f"    --> Bird False Alarm:      {res['bird_false_alarm']*100:.2f}% ({res['fp']} birds misclassified)")
            print(f"    --> F1-Score:              {res['f1']:.4f} | ROC-AUC: {res['auc']:.4f}")
        except Exception as e:
            print(f"    [Error evaluating {display_name}]: {e}")

    if not results:
        print("\n[Error] No models were successfully evaluated.")
        sys.exit(1)

    # 3. Save Summary & Generate Comparison
    os.makedirs(args.out_dir, exist_ok=True)
    plot_cars_results(results, args.out_dir)

    # Save CSV & Markdown
    csv_path = os.path.join(args.out_dir, "cars_benchmark_summary.csv")
    md_path = os.path.join(args.out_dir, "cars_benchmark_summary.md")

    headers = [
        "Model", "Overall Acc (%)", "Bird Acc (%)", "Drone Acc (%)",
        "Bird False Alarms", "Drone Misses", "Precision", "Recall", "F1-Score", "ROC-AUC"
    ]
    rows = []
    for r in results:
        rows.append([
            r["display_name"],
            f"{r['overall_accuracy']*100:.2f}%",
            f"{r['bird_accuracy']*100:.2f}%",
            f"{r['drone_accuracy']*100:.2f}%",
            f"{r['fp']} / {r['total_birds']} ({r['bird_false_alarm']*100:.2f}%)",
            f"{r['fn']} / {r['total_drones']}",
            f"{r['precision']:.4f}",
            f"{r['recall']:.4f}",
            f"{r['f1']:.4f}",
            f"{r['auc']:.4f}",
        ])

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    md_content = [
        "# CARS Test Dataset Evaluation: Birds vs Drones\n",
        f"- **Dataset Path**: `{cars_folder}`\n",
        f"- **Total Birds (`class1_B`)**: {n_birds}\n",
        f"- **Total Drones (`class2_D`)**: {n_drones}\n\n",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        md_content.append("| " + " | ".join(row) + " |")

    with open(md_path, "w") as f:
        f.write("\n".join(md_content) + "\n")

    # Print clean table to terminal
    print("\n" + "=" * 90)
    print("  FINAL CARS BENCHMARK SUMMARY (BIRDS vs DRONES)")
    print("=" * 90)
    print(f"{'Model':<20} | {'Overall Acc':<11} | {'Bird Acc':<10} | {'Drone Acc':<10} | {'Bird FA %':<9} | {'F1':<6} | {'AUC':<6}")
    print("-" * 90)
    for r in results:
        print(f"{r['display_name']:<20} | {r['overall_accuracy']*100:>10.2f}% | {r['bird_accuracy']*100:>9.2f}% | {r['drone_accuracy']*100:>9.2f}% | {r['bird_false_alarm']*100:>8.2f}% | {r['f1']:>6.4f} | {r['auc']:>6.4f}")
    print("=" * 90)
    print(f"\nSaved Markdown Report to: {md_path}")
    print(f"Saved CSV Report to:      {csv_path}\n")


if __name__ == "__main__":
    main()
