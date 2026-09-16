"""
test.py — Full evaluation of an S3-Det checkpoint on the test split.

Produces:
  Metrics (printed + saved as JSON):
    Precision, Recall, F1
    AP@50, AP@50:95
    AP/P/R per object size bin  (<=8, 8-16, 16-32, 32-64, >64 px)
    FPPI
    Center error (mean / median / p95, normalized)
    W/H relative error
    TP IoU mean, TP NWD mean

  Plots (saved to --out-dir):
    precision_recall_curve.png
    f1_vs_threshold.png
    recall_vs_size.png
    ap50_vs_size.png
    nwd_iou_diagnostic.png
    test_pred_*.png  (visual overlays, Green=GT, Red=Pred)

Usage:
    python test.py --checkpoint checkpoints/s3det_best.pt
    python test.py --checkpoint checkpoints/s3det_best.pt --config data_config.yaml
"""
import argparse
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

import torch
from torch.utils.data import DataLoader

from config import S3DetConfig
from model import S3Det
from dataset import YOLODroneDataset, collate_fn
from eval_engine import evaluate, nwd_iou_diagnostic
from plot_utils import (plot_f1_vs_threshold, plot_recall_vs_size,
                        plot_ap_vs_size, plot_nwd_iou_diagnostic)


def plot_pr_curve(recalls, precisions, ap50, out_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recalls, precisions, color="#1f77b4", lw=2.5,
            label=f"S3-Det (AP@0.50 = {ap50:.3f})")
    ax.fill_between(recalls, precisions, alpha=0.1, color="#1f77b4")
    ax.set_title("Precision-Recall Curve on Test Set", fontweight="bold", fontsize=13)
    ax.set_xlabel("Recall", fontsize=12); ax.set_ylabel("Precision", fontsize=12)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.grid(True, ls="--", alpha=0.6); ax.legend(loc="lower left", fontsize=11)
    fig.tight_layout(); fig.savefig(out_path, dpi=200); plt.close(fig)


def visualize_test_predictions(visual_samples, out_dir, num_visuals=12):
    os.makedirs(out_dir, exist_ok=True)
    for idx, sample in enumerate(visual_samples[:num_visuals]):
        img_path = sample["path"]
        try:
            img = Image.open(img_path).convert("RGB")
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(img); ax.axis("off")

            for gb in sample["gt_boxes"]:
                gx1, gy1, gx2, gy2 = gb
                rect = patches.Rectangle((gx1, gy1), gx2-gx1, gy2-gy1,
                                         linewidth=2, edgecolor="#00ff00", facecolor="none")
                ax.add_patch(rect)
                ax.text(gx1, max(0, gy1-3), "GT:drone", color="white", fontsize=8,
                        bbox=dict(facecolor="#00aa00", alpha=0.8, edgecolor="none", pad=1))

            for pb, score in zip(sample["pred_boxes"], sample["pred_scores"]):
                px1, py1, px2, py2 = pb
                rect = patches.Rectangle((px1, py1), px2-px1, py2-py1,
                                         linewidth=2, edgecolor="#ff0055",
                                         facecolor="none", linestyle="--")
                ax.add_patch(rect)
                ax.text(px1, min(img.height-10, py2+12), f"Pred:{score:.2f}",
                        color="white", fontsize=8,
                        bbox=dict(facecolor="#ff0055", alpha=0.8, edgecolor="none", pad=1))

            ax.set_title(f"{os.path.basename(img_path)} (Green=GT  Red=Pred)",
                         fontsize=9, fontweight="bold")
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"test_pred_{idx+1:02d}.png"), dpi=160)
            plt.close(fig)
        except Exception as e:
            print(f"  [WARN] Could not render visual for {img_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate S3-Det checkpoint on test split")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt")
    parser.add_argument("--config",     default="data_config.yaml")
    parser.add_argument("--data-root",  default=None)
    parser.add_argument("--split",      default=None, help="Override test split name")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--score-thresh", type=float, default=None)
    parser.add_argument("--iou-thresh",   type=float, default=0.5)
    parser.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir",      default=None)
    parser.add_argument("--num-visuals",  type=int, default=12)
    args = parser.parse_args()

    # ── Config ───────────────────────────────────────────────────────────────
    cfg = S3DetConfig.from_yaml(args.config) if os.path.exists(args.config) else S3DetConfig()
    if args.data_root:   cfg.data_root   = args.data_root
    if args.score_thresh: cfg.score_thresh = args.score_thresh
    if args.split:       cfg.test_split  = args.split

    out_dir = args.out_dir or os.path.join(cfg.plot_dir, "test")
    os.makedirs(out_dir, exist_ok=True)

    # ── Load model ───────────────────────────────────────────────────────────
    print(f"Loading checkpoint: {args.checkpoint}")
    try:
        ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location=args.device)

    if "cfg" in ckpt and isinstance(ckpt["cfg"], dict):
        for k, v in ckpt["cfg"].items():
            if hasattr(cfg, k):
                try: setattr(cfg, k, v)
                except Exception: pass

    model = S3Det(cfg).to(args.device)

    # Prefer EMA weights if checkpoint has them (they give better final metrics)
    if ckpt.get("ema") is not None:
        model.load_state_dict(ckpt["ema"])
        weights_tag = "EMA weights"
    else:
        model.load_state_dict(ckpt["model"])
        weights_tag = "live weights"

    total_p, _ = model.count_params()
    eff = model.model_summary(input_size=cfg.input_size)
    print(f"  Using: {weights_tag}  (epoch {ckpt.get('epoch', '?')})")

    # Resolved NWD constant: use checkpoint's saved value if present
    nwd_C = ckpt.get("nwd_C", None)
    if nwd_C is None:
        # Recompute from config
        nwd_C = cfg.nwd_constant * (cfg.input_size / 640.0) if getattr(cfg, "nwd_normalize_to_input", True) else cfg.nwd_constant
    print(f"  NWD constant (C): {nwd_C:.3f}  (input={cfg.input_size}px)")

    # ── Dataset ──────────────────────────────────────────────────────────────
    print(f"Loading test split: {cfg.data_root}/{cfg.test_split}")
    test_set = YOLODroneDataset(cfg.data_root, cfg.test_split,
                                img_size=cfg.input_size, augment=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             shuffle=False, collate_fn=collate_fn)
    print(f"Test samples: {len(test_set)}")

    # ── Full evaluation ───────────────────────────────────────────────────────
    print("\nRunning comprehensive evaluation...")
    result, visual_samples = evaluate(
        model, test_loader, torch.device(args.device),
        score_thresh=cfg.score_thresh,
        nwd_constant=nwd_C,
        size_bins=list(cfg.tiny_size_bins),
        num_visuals=args.num_visuals,
        iou_thresh=args.iou_thresh,
        ap_iou_thresholds=list(cfg.ap_iou_thresholds),
    )

    result.print_summary()

    # ── Save metrics JSON ─────────────────────────────────────────────────────
    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    print(f"\nMetrics JSON -> {metrics_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    # 1. PR curve
    pr_path = os.path.join(out_dir, "precision_recall_curve.png")
    plot_pr_curve(result.recalls, result.precisions, result.ap50, pr_path)
    print(f"PR curve      -> {pr_path}")

    # 2. F1 vs threshold
    if result.thresh_values:
        f1t_path = os.path.join(out_dir, "f1_vs_threshold.png")
        plot_f1_vs_threshold(result.thresh_values, result.thresh_precision,
                             result.thresh_recall, result.thresh_f1, f1t_path)
        print(f"F1 vs thresh  -> {f1t_path}")

    # 3. Recall vs size
    if result.size_histogram_bins:
        rvs_path = os.path.join(out_dir, "recall_vs_size.png")
        plot_recall_vs_size(result.size_histogram_bins,
                            result.size_histogram_recall, rvs_path)
        print(f"Recall/size   -> {rvs_path}")

    # 4. AP50 vs size
    if result.size_bins:
        avs_path = os.path.join(out_dir, "ap50_vs_size.png")
        plot_ap_vs_size([b["label"] for b in result.size_bins],
                        [b["ap50"]  for b in result.size_bins], avs_path)
        print(f"AP50/size     -> {avs_path}")

    # 5. NWD vs IoU diagnostic
    disp, ious, nwds = nwd_iou_diagnostic(gt_w=8.0, gt_h=8.0,
                                          nwd_constant=cfg.nwd_constant)
    diag_path = os.path.join(out_dir, "nwd_iou_diagnostic.png")
    plot_nwd_iou_diagnostic(disp, ious, nwds, gt_w=8.0, gt_h=8.0, out_path=diag_path)
    print(f"NWD diagnostic -> {diag_path}")

    # 6. Visual overlays
    if visual_samples:
        pred_dir = os.path.join(out_dir, "predictions")
        visualize_test_predictions(visual_samples, pred_dir, args.num_visuals)
        print(f"Pred overlays -> {pred_dir}/ ({len(visual_samples)} images)")

    print(f"\nAll outputs saved to: {out_dir}/")


if __name__ == "__main__":
    main()
