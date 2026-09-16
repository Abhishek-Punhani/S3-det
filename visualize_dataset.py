"""
Dataset inspection and exploratory data analysis (EDA) for S3-Det.
Visualizes:
1. Sample images with ground-truth bounding box overlays (drone positives + no_drone negatives).
2. Class balance & image counts (train vs test, drone vs no_drone).
3. Bounding box size distributions (width, height, area, aspect ratio).
4. Spatial center coordinates 2D heatmap.

Usage:
    python visualize_dataset.py --config data_config.yaml
    python visualize_dataset.py --data-root /path/to/dataset --out-dir plots/dataset
"""
import argparse
import os
import glob
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from config import S3DetConfig
from dataset import _find_images_and_labels, IMG_EXTS


def analyze_split(split_dir):
    """Gathers statistics and sample paths for a split (train or test)."""
    drone_dir = os.path.join(split_dir, "drone")
    no_drone_dir = os.path.join(split_dir, "no_drone")

    drone_pairs = _find_images_and_labels(drone_dir) if os.path.isdir(drone_dir) else []
    no_drone_pairs = _find_images_and_labels(no_drone_dir) if os.path.isdir(no_drone_dir) else []

    boxes_list = []  # list of (w, h, area, aspect_ratio, xc, yc)
    boxes_per_image = []

    for _, lbl_path in drone_pairs:
        img_boxes = 0
        if lbl_path and os.path.isfile(lbl_path):
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        xc, yc, w, h = [float(x) for x in parts[1:5]]
                        area = w * h
                        ar = w / max(h, 1e-6)
                        boxes_list.append((w, h, area, ar, xc, yc))
                        img_boxes += 1
        boxes_per_image.append(img_boxes)

    # For no_drone, all have 0 boxes
    boxes_per_image.extend([0] * len(no_drone_pairs))

    stats = {
        "drone_count": len(drone_pairs),
        "no_drone_count": len(no_drone_pairs),
        "total_images": len(drone_pairs) + len(no_drone_pairs),
        "boxes": boxes_list,
        "boxes_per_image": boxes_per_image,
        "drone_pairs": drone_pairs,
        "no_drone_pairs": no_drone_pairs,
    }
    return stats


def plot_sample_grid(drone_pairs, no_drone_pairs, out_path, num_samples=8):
    """Draws sample images with ground truth bounding boxes."""
    selected_pos = random.sample(drone_pairs, min(len(drone_pairs), num_samples // 2)) if drone_pairs else []
    selected_neg = random.sample(no_drone_pairs, min(len(no_drone_pairs), num_samples - len(selected_pos))) if no_drone_pairs else []
    samples = [(p, "drone (positive)") for p in selected_pos] + [(p, "no_drone (negative)") for p in selected_neg]

    if not samples:
        print("No samples found to visualize.")
        return

    n_cols = 4
    n_rows = (len(samples) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = np.array(axes).reshape(-1)

    for ax in axes:
        ax.axis("off")

    for i, ((img_path, lbl_path), label_str) in enumerate(samples):
        ax = axes[i]
        try:
            img = Image.open(img_path).convert("RGB")
            w_orig, h_orig = img.size
            ax.imshow(img)
            ax.set_title(f"{os.path.basename(img_path)}\n[{label_str}]", fontsize=9, fontweight="bold")

            if lbl_path and os.path.isfile(lbl_path):
                with open(lbl_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            xc, yc, bw, bh = [float(x) for x in parts[1:5]]
                            x1 = (xc - bw / 2) * w_orig
                            y1 = (yc - bh / 2) * h_orig
                            box_w = bw * w_orig
                            box_h = bh * h_orig
                            rect = patches.Rectangle(
                                (x1, y1), box_w, box_h,
                                linewidth=2, edgecolor="#00ff00", facecolor="none"
                            )
                            ax.add_patch(rect)
                            ax.text(x1, max(0, y1 - 4), "drone", color="white", fontsize=8,
                                    bbox=dict(facecolor="#00ff00", alpha=0.8, edgecolor="none", pad=1))
        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  Saved sample visualization -> {out_path}")


def plot_dataset_statistics(train_stats, test_stats, out_path):
    """Generates analytical charts of the dataset distributions."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # 1. Image Counts Bar Chart
    ax = axes[0, 0]
    splits = ["Train", "Test"]
    drone_counts = [train_stats["drone_count"], test_stats["drone_count"]]
    no_drone_counts = [train_stats["no_drone_count"], test_stats["no_drone_count"]]
    x = np.arange(len(splits))
    width = 0.35

    ax.bar(x - width/2, drone_counts, width, label="Drone (Positives)", color="#1f77b4")
    ax.bar(x + width/2, no_drone_counts, width, label="No-Drone (Negatives)", color="#aec7e8")
    ax.set_ylabel("Number of Images")
    ax.set_title("Dataset Image Count by Split & Class", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")

    # Collect all boxes for geometry analysis
    all_boxes = train_stats["boxes"] + test_stats["boxes"]
    if all_boxes:
        widths = [b[0] for b in all_boxes]
        heights = [b[1] for b in all_boxes]
        areas = [b[2] for b in all_boxes]
        ars = [b[3] for b in all_boxes]
        xcs = [b[4] for b in all_boxes]
        ycs = [b[5] for b in all_boxes]

        # 2. Box Width vs Height Scatter / Density
        ax = axes[0, 1]
        ax.scatter(widths, heights, alpha=0.4, color="#ff7f0e", s=15, edgecolors="none")
        ax.set_xlabel("Normalized Box Width")
        ax.set_ylabel("Normalized Box Height")
        ax.set_title("Bounding Box Dimensions (W vs H)", fontweight="bold")
        ax.set_xlim(0, max(0.5, max(widths) * 1.1 if widths else 1.0))
        ax.set_ylim(0, max(0.5, max(heights) * 1.1 if heights else 1.0))
        ax.grid(True, linestyle="--", alpha=0.5)

        # 3. Object Area Distribution Histogram
        ax = axes[0, 2]
        ax.hist(areas, bins=30, color="#2ca02c", edgecolor="black", alpha=0.7)
        ax.set_xlabel("Normalized Area (w * h)")
        ax.set_ylabel("Target Box Frequency")
        ax.set_title("Object Scale Distribution (Tiny / Small / Medium)", fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)

        # 4. Aspect Ratio Distribution
        ax = axes[1, 0]
        # Filter extreme outliers for clear visualization
        clipped_ars = [min(max(a, 0.2), 5.0) for a in ars]
        ax.hist(clipped_ars, bins=30, color="#9467bd", edgecolor="black", alpha=0.7)
        ax.set_xlabel("Aspect Ratio (Width / Height)")
        ax.set_ylabel("Count")
        ax.set_title("Bounding Box Aspect Ratio Distribution", fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)

        # 5. Spatial Center Heatmap (Where targets appear in frame)
        ax = axes[1, 1]
        h2d, xedges, yedges = np.histogram2d(xcs, ycs, bins=25, range=[[0, 1], [0, 1]])
        im = ax.imshow(h2d.T, origin="upper", extent=[0, 1, 1, 0], cmap="plasma", aspect="auto")
        ax.set_xlabel("Normalized X Center")
        ax.set_ylabel("Normalized Y Center")
        ax.set_title("Object Spatial Location 2D Heatmap", fontweight="bold")
        fig.colorbar(im, ax=ax, label="Target Density")

        # 6. Targets per Image Histogram
        ax = axes[1, 2]
        bpi = train_stats["boxes_per_image"]
        max_boxes = max(bpi) if bpi else 1
        ax.hist(bpi, bins=range(0, max_boxes + 2), color="#d62728", edgecolor="black", alpha=0.7, align="left")
        ax.set_xlabel("Number of Drone Targets per Image")
        ax.set_ylabel("Image Frequency")
        ax.set_title("Targets per Image Distribution", fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)
    else:
        for i in range(1, 6):
            r, c = i // 3, i % 3
            axes[r, c].text(0.5, 0.5, "No bounding box annotations found", ha="center", va="center")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  Saved dataset overview stats -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize and analyze UAV drone dataset")
    parser.add_argument("--config", default="data_config.yaml", help="Path to config YAML")
    parser.add_argument("--data-root", default=None, help="Dataset root directory")
    parser.add_argument("--out-dir", default=None, help="Output directory for plots")
    parser.add_argument("--num-samples", type=int, default=8, help="Number of sample images to draw")
    args = parser.parse_args()

    if os.path.exists(args.config):
        cfg = S3DetConfig.from_yaml(args.config)
    else:
        cfg = S3DetConfig()

    data_root = args.data_root or cfg.data_root
    out_dir = args.out_dir or os.path.join(cfg.plot_dir, "dataset")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Analyzing dataset at: {data_root}")
    train_dir = os.path.join(data_root, cfg.train_split)
    test_dir = os.path.join(data_root, cfg.test_split)

    if not os.path.isdir(train_dir):
        print(f"Warning: Train directory not found at {train_dir}")
        train_stats = {"drone_count": 0, "no_drone_count": 0, "total_images": 0, "boxes": [], "boxes_per_image": [], "drone_pairs": [], "no_drone_pairs": []}
    else:
        train_stats = analyze_split(train_dir)

    if not os.path.isdir(test_dir):
        print(f"Warning: Test directory not found at {test_dir}")
        test_stats = {"drone_count": 0, "no_drone_count": 0, "total_images": 0, "boxes": [], "boxes_per_image": [], "drone_pairs": [], "no_drone_pairs": []}
    else:
        test_stats = analyze_split(test_dir)

    print(f"Train Split: {train_stats['drone_count']} drone images, {train_stats['no_drone_count']} no_drone images, {len(train_stats['boxes'])} total boxes.")
    print(f"Test Split:  {test_stats['drone_count']} drone images, {test_stats['no_drone_count']} no_drone images, {len(test_stats['boxes'])} total boxes.")

    # 1. Samples Grid
    sample_out = os.path.join(out_dir, "dataset_samples.png")
    all_drone_pairs = train_stats["drone_pairs"] + test_stats["drone_pairs"]
    all_no_drone_pairs = train_stats["no_drone_pairs"] + test_stats["no_drone_pairs"]
    plot_sample_grid(all_drone_pairs, all_no_drone_pairs, sample_out, num_samples=args.num_samples)

    # 2. Statistics and Distributions
    stats_out = os.path.join(out_dir, "dataset_overview.png")
    plot_dataset_statistics(train_stats, test_stats, stats_out)

    print(f"Dataset visualization completed successfully. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
