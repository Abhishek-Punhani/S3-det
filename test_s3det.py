"""
Test and Evaluate Best S3-Det Checkpoint
========================================
Supports evaluating the best S3-Det checkpoint (s3det_best.pt) on:
  1. CARS Dataset (class1_B no_drone vs class2_D drone):
     - Evaluates detector's image-level classification performance (Drone vs No Drone).
     - Computes Accuracy, Drone Recall, No-Drone Accuracy, False Positive Rate (FPR), FPPI, F1.
     - Performs a confidence threshold sweep (0.10 -> 0.50) to find the optimal operating point.
  2. Full Detection Dataset (dataset_full / test split):
     - Computes standard object detection metrics: AP@50, AP@50:95, Precision, Recall, F1.
     - Computes per-size recall bins (bin0 <=8px, bin1 8-16px, bin2 16-32px, etc.).
     - Computes FPPI (False Positives Per Image) and localization center error.

Usage:
  # Test on CARS (drone vs no_drone):
  python test_s3det.py --cars-dir CARS --device cuda

  # Test on full detection test dataset:
  python test_s3det.py --data-root dataset_full --split test --device cuda

  # Auto-evaluates on whatever is found:
  python test_s3det.py
"""

import os
import sys
import glob
import math
import json
import csv
import argparse
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import S3DetConfig
from model import S3Det
from dataset import YOLODroneDataset, collate_fn
from eval_engine import evaluate_epoch

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# ── Checkpoint & Model Loading ───────────────────────────────────────────────
def find_s3det_checkpoint(user_path: str = None) -> str:
    """Finds best S3-Det checkpoint automatically."""
    candidates = [
        user_path,
        "checkpoints/s3det_best.pt",
        "checkpoints_s3net/s3det_best.pt",
        "s3det_best.pt",
        "checkpoints/s3det_latest.pt",
        "s3det_latest.pt",
    ]
    for c in candidates:
        if c and os.path.isfile(c):
            return os.path.abspath(c)

    matches = glob.glob("checkpoints*/**/s3det_best.pt", recursive=True)
    if matches:
        return os.path.abspath(matches[0])
    matches_any = glob.glob("checkpoints*/**/*.pt", recursive=True)
    if matches_any:
        return os.path.abspath(matches_any[0])
    return None


def load_s3det_model(ckpt_path: str, device: torch.device, config_path: str = "data_config.yaml"):
    """Loads S3-Det model using checkpoint weights and saved config."""
    if os.path.isfile(config_path):
        cfg = S3DetConfig.from_yaml(config_path)
    else:
        cfg = S3DetConfig()

    print(f"[Loading] Checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Restore saved config if available
    if isinstance(ckpt, dict) and "cfg" in ckpt and isinstance(ckpt["cfg"], dict):
        saved_cfg = ckpt["cfg"]
        for k, v in saved_cfg.items():
            if hasattr(cfg, k):
                try:
                    setattr(cfg, k, v)
                except Exception:
                    pass

    model = S3Det(cfg)

    # Extract state dict (prefer EMA weights if available for cleaner evaluations)
    state_dict = None
    if isinstance(ckpt, dict):
        if "ema" in ckpt and ckpt["ema"] is not None and isinstance(ckpt["ema"], dict):
            print("  --> Using EMA shadow weights from checkpoint.")
            state_dict = ckpt["ema"]
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            state_dict = ckpt["model"]
        elif "model_state" in ckpt and isinstance(ckpt["model_state"], dict):
            state_dict = ckpt["model_state"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    cleaned_dict = {}
    for k, v in state_dict.items():
        cleaned_dict[k.replace("module.", "")] = v

    model.load_state_dict(cleaned_dict, strict=True)
    model = model.to(device)
    model.eval()
    return model, cfg


# ── CARS Dataset Evaluation (Image-Level Detection) ───────────────────────────
class CARSImageDataset(Dataset):
    """Loads images from CARS for inference."""
    def __init__(self, samples: List[Tuple[str, int]], img_size: int = 640):
        self.samples = samples
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            with open(path, "rb") as f:
                img = Image.open(f).convert("RGB")
        except Exception:
            img = Image.new("RGB", (self.img_size, self.img_size), (0, 0, 0))

        tensor = self.transform(img)
        return tensor, label, path


def scan_cars_folder(cars_dir: str) -> List[Tuple[str, int]]:
    """Scans class1_B (no_drone=0) and class2_D (drone=1)."""
    samples = []
    no_drone_folders = ["class1_B", "class1_b", "class1", "no_drone"]
    drone_folders = ["class2_D", "class2_d", "class2", "drone", "drones"]

    def _collect(subfolders, label):
        for sf in subfolders:
            p = os.path.join(cars_dir, sf)
            if os.path.isdir(p):
                for root, _, files in os.walk(p):
                    for f in files:
                        if f.lower().endswith(IMG_EXTS):
                            samples.append((os.path.join(root, f), label))
                return True
        return False

    _collect(no_drone_folders, 0)
    _collect(drone_folders, 1)
    return samples


@torch.no_grad()
def evaluate_s3det_on_cars(model, loader, device, thresholds=[0.10, 0.20, 0.25, 0.30, 0.40, 0.50]):
    """
    Evaluates detector on CARS dataset across multiple score thresholds.
    An image is predicted as 'drone' if at least 1 detection has score >= thresh.
    """
    print(f"\n[CARS] Running detector inference on {len(loader.dataset)} images...")

    raw_results = []
    # Collect all max scores and detections
    for images, targets, paths in loader:
        images = images.to(device)
        preds = model(images)  # List of dicts with 'boxes', 'scores', 'labels'

        for pred, target in zip(preds, targets):
            scores = pred["scores"].cpu().numpy()
            max_score = float(np.max(scores)) if len(scores) > 0 else 0.0
            n_boxes = len(scores)
            raw_results.append({
                "target": int(target),
                "max_score": max_score,
                "all_scores": scores,
                "n_boxes": n_boxes,
            })

    # Evaluate across thresholds
    sweep_results = []
    targets_all = np.array([r["target"] for r in raw_results])
    max_scores_all = np.array([r["max_score"] for r in raw_results])
    total_imgs = len(targets_all)
    total_drone = int(np.sum(targets_all == 1))
    total_no_drone = int(np.sum(targets_all == 0))

    for thr in thresholds:
        preds_all = (max_scores_all >= thr).astype(int)

        tp = int(np.sum((preds_all == 1) & (targets_all == 1)))
        fn = int(np.sum((preds_all == 0) & (targets_all == 1)))
        tn = int(np.sum((preds_all == 0) & (targets_all == 0)))
        fp = int(np.sum((preds_all == 1) & (targets_all == 0)))

        acc = (tp + tn) / max(total_imgs, 1)
        drone_rec = tp / max(total_drone, 1)
        no_drone_acc = tn / max(total_no_drone, 1)
        fpr = fp / max(total_no_drone, 1)
        precision = tp / max(tp + fp, 1e-9)
        f1 = 2 * precision * drone_rec / max(precision + drone_rec, 1e-9)

        # Count total false positive boxes on no_drone images
        fp_boxes = sum(np.sum(r["all_scores"] >= thr) for r in raw_results if r["target"] == 0)
        fppi = fp_boxes / max(total_no_drone, 1)

        sweep_results.append({
            "threshold": thr,
            "overall_accuracy": acc,
            "drone_recall": drone_rec,
            "no_drone_accuracy": no_drone_acc,
            "fpr": fpr,
            "precision": precision,
            "f1": f1,
            "fppi": fppi,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "total_drone": total_drone,
            "total_no_drone": total_no_drone,
        })

    return sweep_results


def plot_cars_sweep(sweep_results: List[Dict], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    thrs = [r["threshold"] for r in sweep_results]
    accs = [r["overall_accuracy"] * 100 for r in sweep_results]
    recs = [r["drone_recall"] * 100 for r in sweep_results]
    no_accs = [r["no_drone_accuracy"] * 100 for r in sweep_results]
    f1s = [r["f1"] for r in sweep_results]
    fprs = [r["fpr"] * 100 for r in sweep_results]

    best_idx = int(np.argmax([r["overall_accuracy"] for r in sweep_results]))
    best = sweep_results[best_idx]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Accuracy & F1 vs Threshold
    ax1.plot(thrs, accs, marker="o", color="#1f77b4", lw=2, label="Overall Accuracy (%)")
    ax1.plot(thrs, recs, marker="s", color="#2ca02c", lw=2, label="Drone Recall (%)")
    ax1.plot(thrs, no_accs, marker="^", color="#ff7f0e", lw=2, label="no_drone Accuracy (%)")
    ax1.axvline(best["threshold"], color="gray", ls="--", alpha=0.7, label=f"Best Acc ({best['overall_accuracy']*100:.1f}%) @ thr={best['threshold']}")
    ax1.set_title("S³-Det on CARS: Accuracy vs Confidence Threshold", fontweight="bold")
    ax1.set_xlabel("Confidence Threshold")
    ax1.set_ylabel("Score (%)")
    ax1.set_ylim(0, 105)
    ax1.grid(True, ls="--", alpha=0.5)
    ax1.legend()

    # Panel 2: F1 and False Positive Rate
    ax2.plot(thrs, f1s, marker="o", color="#d62728", lw=2, label="F1-Score")
    ax2_twin = ax2.twinx()
    ax2_twin.plot(thrs, fprs, marker="x", color="#9467bd", lw=2, ls="--", label="False Positive Rate (%)")
    ax2.set_title("S³-Det on CARS: F1 Score & False Positive Rate", fontweight="bold")
    ax2.set_xlabel("Confidence Threshold")
    ax2.set_ylabel("F1-Score", color="#d62728", fontweight="bold")
    ax2_twin.set_ylabel("False Positive Rate (%)", color="#9467bd", fontweight="bold")
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, ls="--", alpha=0.5)

    fig.tight_layout()
    plot_path = os.path.join(out_dir, "s3det_cars_eval.png")
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    print(f"  ==> Saved plot to: {plot_path}")


# ── Full Detection Evaluation on dataset_full ────────────────────────────────
def evaluate_detection_split(model, cfg, data_root: str, split: str, device: torch.device, out_dir: str):
    print(f"\n[Detection Eval] Evaluating S3-Det on {data_root}/{split} split...")
    dataset = YOLODroneDataset(data_root, split, img_size=cfg.input_size)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=collate_fn)

    results = evaluate_epoch(
        model=model,
        loader=loader,
        device=device,
        score_thresh=cfg.score_thresh,
        iou_thresh=0.50,
        nwd_constant=cfg.nwd_constant,
        size_bins=list(cfg.tiny_size_bins),
        ap_iou_thresholds=list(cfg.ap_iou_thresholds),
    )

    print("\n" + "=" * 80)
    print(f"  S³-DETECTION RESULTS ON '{split.upper()}' SPLIT ({len(dataset)} images)")
    print("=" * 80)
    print(f"  AP@50       : {results.get('val_ap50', 0):.4f}")
    print(f"  AP@50:95    : {results.get('val_ap50_95', 0):.4f}")
    print(f"  Precision   : {results.get('val_precision', 0):.4f}")
    print(f"  Recall      : {results.get('val_recall', 0):.4f}")
    print(f"  F1-Score    : {results.get('val_f1', 0):.4f}")
    print(f"  FPPI        : {results.get('val_fppi', 0):.3f} false positives / image")
    print(f"  Center Err  : {results.get('val_center_err', 0):.2f} px")
    print("-" * 80)
    print("  Per-Size Recall Bins:")
    for k in sorted(results.keys()):
        if k.startswith("val_recall_bin"):
            bin_name = k.replace("val_recall_", "")
            print(f"    {bin_name:<15}: {results[k]:.4f}")
    print("=" * 80)

    # Save to JSON
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f"s3det_{split}_metrics.json")
    clean_results = {k: float(v) if isinstance(v, (int, float, np.floating, np.integer)) else str(v) for k, v in results.items()}
    with open(json_path, "w") as f:
        json.dump(clean_results, f, indent=2)
    print(f"  Saved detection metrics to: {json_path}")
    return results


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Test and evaluate best S3-Det checkpoint")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to s3det_best.pt checkpoint")
    parser.add_argument("--config", type=str, default="data_config.yaml", help="Path to config YAML")
    parser.add_argument("--cars-dir", type=str, default="CARS", help="Path to CARS dataset folder")
    parser.add_argument("--data-root", type=str, default="dataset_full", help="Path to full detection dataset")
    parser.add_argument("--split", type=str, default="test", help="Split to evaluate (test or val)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--out-dir", type=str, default="runs_s3det_eval", help="Output directory")
    args = parser.parse_args()

    device = torch.device(args.device if (torch.cuda.is_available() and "cuda" in args.device) else "cpu")
    ckpt_path = find_s3det_checkpoint(args.ckpt)

    if not ckpt_path:
        print("[Error] Could not find S3-Det checkpoint (e.g. checkpoints/s3det_best.pt).")
        print("Please specify path with --ckpt <path_to_checkpoint.pt>")
        sys.exit(1)

    model, cfg = load_s3det_model(ckpt_path, device, args.config)
    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Check if CARS dataset is available to evaluate
    cars_path = os.path.abspath(args.cars_dir) if os.path.isdir(args.cars_dir) else None
    if cars_path:
        samples = scan_cars_folder(cars_path)
        if len(samples) > 0:
            print("\n" + "█" * 80)
            print(f"  EVALUATING S³-DET ON CARS DATASET ({len(samples)} images)")
            print(f"  Path: {cars_path}")
            print("█" * 80)

            cars_ds = CARSImageDataset(samples, img_size=cfg.input_size)
            cars_loader = DataLoader(cars_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

            sweep = evaluate_s3det_on_cars(model, cars_loader, device)
            plot_cars_sweep(sweep, args.out_dir)

            # Print Table
            print("\n" + "=" * 92)
            print(f"{'Thresh':<8} | {'Overall Acc':<11} | {'no_drone Acc':<12} | {'Drone Recall':<12} | {'FPR':<7} | {'F1':<6} | {'FPPI':<6}")
            print("=" * 92)
            for r in sweep:
                star = " ★ [BEST ACC]" if r["overall_accuracy"] == max(x["overall_accuracy"] for x in sweep) else ""
                print(f"{r['threshold']:<8.2f} | {r['overall_accuracy']*100:>10.2f}% | {r['no_drone_accuracy']*100:>11.2f}% | {r['drone_recall']*100:>11.2f}% | {r['fpr']*100:>6.2f}% | {r['f1']:>6.4f} | {r['fppi']:>6.3f}{star}")
            print("=" * 92)

            # Save CSV
            csv_path = os.path.join(args.out_dir, "s3det_cars_summary.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(list(sweep[0].keys()))
                for r in sweep:
                    writer.writerow(list(r.values()))
            print(f"  Saved CARS summary CSV to: {csv_path}")

    # 2. Check if detection dataset (dataset_full) is available
    det_split_dir = os.path.join(args.data_root, args.split)
    if os.path.isdir(det_split_dir):
        evaluate_detection_split(model, cfg, args.data_root, args.split, device, args.out_dir)

    print(f"\nAll evaluations complete! Outputs saved in '{args.out_dir}'.")


if __name__ == "__main__":
    main()
