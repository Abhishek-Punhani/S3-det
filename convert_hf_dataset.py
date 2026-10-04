"""
Master Dataset Converter - S3-Det
==========================================================================
Combines THREE datasets into the S3-Det directory layout:

  <output_dir>/
    train/
      drone/      <- images + YOLO .txt  (class 0 = drone/UAV)
      no_drone/   <- images ONLY, NO .txt (model learns to ignore these)
    test/
      drone/
      no_drone/

SOURCE 1: HuggingFace 'pathikg/drone-detection-dataset'
  - 51,446 train + 2,625 test images, all containing drones
  - Streamed from HF hub (no 59 GB full download needed)

SOURCE 2: Kaggle 'avldevelopment/air-to-air-object-detection-dataset' (720p only)
  - UAV images  -> drone/ folder + YOLO .txt  (class 0)
  - Bird/Helicopter/Airliner/Balloon -> no_drone/ folder, NO .txt
  - BBox format: [x1,y1,x2,y2] absolute pixels, 1280x720

SOURCE 3: Kaggle 'panshull/full-dataset'
  - Only no_drone/ subfolders are used (drone/ subfolders are skipped)
  - Pure background negatives, NO .txt files written
  - Structure: train/batch_00X/no_drone/

Usage:
  python convert_hf_dataset.py --output-dir dataset_full
  python convert_hf_dataset.py --output-dir dataset_full --skip-hf
  python convert_hf_dataset.py --output-dir dataset_full --skip-kaggle
  python convert_hf_dataset.py --output-dir dataset_subset --max-hf-train 500

Prerequisites:
  pip install datasets pillow kaggle
  Place kaggle.json in ~/.kaggle/ before running
==========================================================================
"""

import argparse
import glob
import json
import os
import random
import shutil
import subprocess
import zipfile
from collections import defaultdict
from io import BytesIO

from PIL import Image

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

# Kaggle Air-to-Air category ID for UAV/drone
KAGGLE_UAV_ID = 1


# ============================================================
# PART 1 - HuggingFace Dataset (all drones, COCO -> YOLO)
# ============================================================

def convert_hf_split(dataset_split, split_name, output_dir, max_samples=None):
    drone_dir    = os.path.join(output_dir, split_name, "drone")
    no_drone_dir = os.path.join(output_dir, split_name, "no_drone")
    os.makedirs(drone_dir,    exist_ok=True)
    os.makedirs(no_drone_dir, exist_ok=True)

    pos_count = neg_count = total_boxes = count = 0

    for sample in dataset_split:
        if max_samples and count >= max_samples:
            break

        img_id  = sample.get("image_id", count)
        img_raw = sample["image"]
        width   = sample.get("width")
        height  = sample.get("height")

        if isinstance(img_raw, Image.Image):
            img = img_raw.convert("RGB")
        elif isinstance(img_raw, dict) and "bytes" in img_raw:
            img = Image.open(BytesIO(img_raw["bytes"])).convert("RGB")
        else:
            img = Image.open(img_raw).convert("RGB")

        if width is None or height is None:
            width, height = img.size

        # COCO [xmin, ymin, w, h] -> YOLO normalised [xc, yc, w, h]
        bboxes     = sample.get("objects", {}).get("bbox", [])
        yolo_lines = []
        for box in bboxes:
            if len(box) < 4:
                continue
            x_min, y_min, bw, bh = box[:4]
            if bw <= 0 or bh <= 0:
                continue
            xc = min(max((x_min + bw / 2.0) / width,  0.0), 1.0)
            yc = min(max((y_min + bh / 2.0) / height, 0.0), 1.0)
            nw = min(max(bw / width,  0.0), 1.0)
            nh = min(max(bh / height, 0.0), 1.0)
            yolo_lines.append(f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n")

        stem = f"hf_{split_name}_{img_id:07d}"

        if yolo_lines:
            img.save(os.path.join(drone_dir, f"{stem}.jpg"), quality=95)
            with open(os.path.join(drone_dir, f"{stem}.txt"), "w") as f:
                f.writelines(yolo_lines)
            pos_count  += 1
            total_boxes += len(yolo_lines)
        else:
            img.save(os.path.join(no_drone_dir, f"{stem}.jpg"), quality=95)
            neg_count += 1

        count += 1
        if count % 5000 == 0:
            print(f"  [HF/{split_name}] {count} processed "
                  f"(+{pos_count} drone  -{neg_count} bg  {total_boxes} boxes)...")

    print(f"  [HF/{split_name} DONE] +{pos_count} drone  -{neg_count} bg  {total_boxes} boxes")


# ============================================================
# PART 2 - Kaggle Air-to-Air (720p, AirSim format, class-aware)
# ============================================================

def download_kaggle(kaggle_dir):
    """Download zip, extract ONLY 720p files (no masks), delete zip."""
    base_720 = os.path.join(kaggle_dir, "720p")
    if os.path.isdir(base_720):
        print(f"  Kaggle 720p dir already exists - skipping download.")
        return

    os.makedirs(kaggle_dir, exist_ok=True)
    zip_name = "air-to-air-object-detection-dataset.zip"
    zip_path = os.path.join(kaggle_dir, zip_name)

    if not os.path.isfile(zip_path):
        print("  Downloading Kaggle dataset (~168 GB zip)...")
        subprocess.run(
            ["kaggle", "datasets", "download",
             "-d", "avldevelopment/air-to-air-object-detection-dataset",
             "-p", kaggle_dir],
            check=True,
        )
    else:
        print(f"  Zip already present - skipping download.")

    print("  Scanning zip for 720p entries...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [
            m for m in zf.namelist()
            if m.startswith("720p/") and "/masks/" not in m
        ]
        total = len(members)
        print(f"  Extracting {total} files from 720p/ (skipping other resolutions and masks)...")
        for i, member in enumerate(members):
            zf.extract(member, kaggle_dir)
            if (i + 1) % 10000 == 0:
                print(f"  Extracted {i + 1}/{total}...")

    print("  Deleting zip to free disk space...")
    os.remove(zip_path)
    print("  Kaggle download done!")


def _find_kaggle_base(kaggle_dir):
    """Dynamically find the folder containing train/ and val/ subfolders.
    Handles both kaggle_air/720p/train/ and kaggle_air/720p/720p/train/ nesting."""
    hits = glob.glob(os.path.join(kaggle_dir, "**", "train_annotations.json"), recursive=True)
    if not hits:
        return None
    train_folder = os.path.dirname(hits[0])   # .../train
    base = os.path.dirname(train_folder)       # .../720p  (or .../720p/720p)
    return base


def parse_airsim_json(json_path):
    """
    AirSim annotation format:
    [
      {
        "filename": "Desert_0600_Blizzard_720p_image_119.png",
        "boundingBoxes": {
          "UAV":  [[x1,y1,x2,y2], ...],
          "Bird": [[x1,y1,x2,y2], ...],
        }
      }, ...
    ]
    """
    with open(json_path) as f:
        return json.load(f)


def process_kaggle_split(images_dir, json_path, out_split, output_dir, prefix):
    """
    Sorts each image:
      - Has "UAV" in boundingBoxes -> drone/ + YOLO .txt (only UAV boxes, class 0)
      - No "UAV" (only Bird/Helicopter/etc.) -> no_drone/, NO .txt
      - Image with BOTH UAV + Bird -> drone/, only UAV boxes written
    BBox format: [x1,y1,x2,y2] absolute pixels. Image size: 1280x720 (720p always).
    """
    if not os.path.isfile(json_path):
        print(f"  WARNING: {json_path} not found - skipping.")
        return 0, 0

    IMG_W, IMG_H = 1280, 720

    drone_dir    = os.path.join(output_dir, out_split, "drone")
    no_drone_dir = os.path.join(output_dir, out_split, "no_drone")
    os.makedirs(drone_dir,    exist_ok=True)
    os.makedirs(no_drone_dir, exist_ok=True)

    records = parse_airsim_json(json_path)
    pos = neg = total_boxes = 0

    for idx, rec in enumerate(records):
        filename        = rec.get("filename", "")
        src             = os.path.join(images_dir, filename)
        if not os.path.isfile(src):
            continue

        bboxes_by_class = rec.get("boundingBoxes", {})
        uav_boxes       = bboxes_by_class.get("UAV", [])
        ext             = os.path.splitext(filename)[1] or ".png"
        stem            = f"kgl_{prefix}_{idx:07d}"

        if uav_boxes:
            yolo_lines = []
            for box in uav_boxes:
                x1, y1, x2, y2 = box
                if x2 <= x1 or y2 <= y1:
                    continue
                xc = min(max(((x1 + x2) / 2.0) / IMG_W, 0.0), 1.0)
                yc = min(max(((y1 + y2) / 2.0) / IMG_H, 0.0), 1.0)
                nw = min(max((x2 - x1) / IMG_W,          0.0), 1.0)
                nh = min(max((y2 - y1) / IMG_H,          0.0), 1.0)
                yolo_lines.append(f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n")

            shutil.copy(src, os.path.join(drone_dir, f"{stem}{ext}"))
            with open(os.path.join(drone_dir, f"{stem}.txt"), "w") as f:
                f.writelines(yolo_lines)
            pos         += 1
            total_boxes += len(yolo_lines)
        else:
            # Bird / Helicopter / Airliner / Balloon - pure background
            shutil.copy(src, os.path.join(no_drone_dir, f"{stem}{ext}"))
            neg += 1

        if (idx + 1) % 2000 == 0:
            print(f"  [Kaggle/{prefix}] {idx+1}/{len(records)} "
                  f"(+{pos} drone  -{neg} bg)...")

    print(f"  [Kaggle/{prefix} -> {out_split} DONE]  "
          f"+{pos} drone  -{neg} bg  {total_boxes} UAV boxes")
    return pos, neg


def process_kaggle(kaggle_dir, output_dir):
    base = _find_kaggle_base(kaggle_dir)
    if base is None:
        print(f"  WARNING: Could not find train_annotations.json under {kaggle_dir} - skipping.")
        return

    print(f"  Kaggle base detected at: {base}")

    splits = [
        (os.path.join(base, "train", "images"),
         os.path.join(base, "train", "train_annotations.json"),
         "train", "train"),
        (os.path.join(base, "val", "images"),
         os.path.join(base, "val",   "val_annotations.json"),
         "test",  "val"),
    ]
    tp = tn = 0
    for images_dir, json_path, out_split, prefix in splits:
        print(f"\n  Processing Kaggle/{prefix} -> {out_split}/...")
        p, n = process_kaggle_split(images_dir, json_path, out_split, output_dir, prefix)
        tp += p
        tn += n
    print(f"\n  [Kaggle DONE]  +{tp} drone images  -{tn} bg images")


# ============================================================
# PART 3 - Panshull background negatives (images only, no labels)
# ============================================================

def add_panshull_negatives(panshull_dir, output_dir, train_ratio=0.90):
    """
    Dataset structure:
      train/batch_001/drone/   <- we SKIP this
      train/batch_001/no_drone/ <- we ONLY take from here
      ...5 batches total...
      test/ (same structure)

    All images go to output_dir as pure background negatives (NO .txt files).
    90% -> train/no_drone/, 10% -> test/no_drone/
    """
    all_imgs = (
        glob.glob(os.path.join(panshull_dir, "**/no-drone/**/*.jpg"),  recursive=True) +
        glob.glob(os.path.join(panshull_dir, "**/no-drone/**/*.jpeg"), recursive=True) +
        glob.glob(os.path.join(panshull_dir, "**/no-drone/**/*.png"),  recursive=True)
    )

    if not all_imgs:
        print(f"  WARNING: No no-drone images found under {panshull_dir}")
        print(f"  Expected: {panshull_dir}/train/batch_00X/no-drone/")
        return

    print(f"  Found {len(all_imgs)} no_drone images across all batches.")

    random.seed(42)
    random.shuffle(all_imgs)
    train_count = int(len(all_imgs) * train_ratio)

    train_neg = os.path.join(output_dir, "train", "no_drone")
    test_neg  = os.path.join(output_dir, "test",  "no_drone")
    os.makedirs(train_neg, exist_ok=True)
    os.makedirs(test_neg,  exist_ok=True)

    for i, src in enumerate(all_imgs):
        dest_dir = train_neg if i < train_count else test_neg
        ext      = os.path.splitext(src)[1] or ".jpg"
        shutil.copy(src, os.path.join(dest_dir, f"panshull_neg_{i:07d}{ext}"))
        if (i + 1) % 5000 == 0:
            print(f"  [Panshull] {i+1}/{len(all_imgs)} copied...")

    print(f"  [Panshull DONE]  {train_count} -> train/no_drone  |  "
          f"{len(all_imgs) - train_count} -> test/no_drone")


# ============================================================
# SUMMARY + MAIN
# ============================================================

def print_summary(output_dir):
    print("\n" + "="*58)
    print("  FINAL DATASET SUMMARY")
    print("="*58)
    grand_imgs = grand_labels = 0
    for split in ["train", "test"]:
        for folder in ["drone", "no_drone"]:
            d = os.path.join(output_dir, split, folder)
            if not os.path.isdir(d):
                continue
            imgs   = len([f for f in os.listdir(d)
                          if f.lower().endswith((".jpg", ".jpeg", ".png"))])
            labels = len([f for f in os.listdir(d) if f.endswith(".txt")])
            print(f"  {split}/{folder:<10}  {imgs:>8} images   {labels:>8} label files")
            grand_imgs   += imgs
            grand_labels += labels
    print(f"  {'TOTAL':<20}  {grand_imgs:>8} images   {grand_labels:>8} label files")
    print("="*58)


def main():
    parser = argparse.ArgumentParser(
        description="Combine HuggingFace + Kaggle + Panshull datasets into S3-Det format"
    )
    parser.add_argument("--output-dir",    default="dataset_full",
                        help="Root output directory")
    parser.add_argument("--kaggle-dir",    default="kaggle_air",
                        help="Where to download/find the Kaggle air-to-air dataset")
    parser.add_argument("--panshull-dir",  default="panshull_negatives",
                        help="Path to downloaded panshull/full-dataset")
    parser.add_argument("--max-hf-train",  type=int, default=None,
                        help="Limit HF train images (for smoke-test subsets)")
    parser.add_argument("--max-hf-test",   type=int, default=None,
                        help="Limit HF test images (for smoke-test subsets)")
    parser.add_argument("--skip-hf",       action="store_true",
                        help="Skip the HuggingFace dataset")
    parser.add_argument("--skip-kaggle",   action="store_true",
                        help="Skip the Kaggle air-to-air dataset")
    parser.add_argument("--skip-panshull", action="store_true",
                        help="Skip the Panshull negatives dataset")
    parser.add_argument("--no-wipe",       action="store_true",
                        help="Do NOT delete output-dir before running "
                             "(safe to resume a killed run)")
    args = parser.parse_args()

    # Wipe existing output (skip if --no-wipe)
    if os.path.exists(args.output_dir) and not args.no_wipe:
        print(f"Deleting existing directory: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    elif args.no_wipe and os.path.exists(args.output_dir):
        print(f"  --no-wipe set: keeping existing data in {args.output_dir}")


    # ── Step 1: HuggingFace ────────────────────────────────────────
    if not args.skip_hf:
        if load_dataset is None:
            raise ImportError("Run: pip install datasets")
        print("\n" + "="*58)
        print("  STEP 1/3 - HuggingFace: pathikg/drone-detection-dataset")
        print("="*58)
        hf = load_dataset("pathikg/drone-detection-dataset", streaming=True)
        if "train" in hf:
            print("\n  Processing HF train split...")
            convert_hf_split(hf["train"], "train", args.output_dir, args.max_hf_train)
        if "test" in hf:
            print("\n  Processing HF test split...")
            convert_hf_split(hf["test"],  "test",  args.output_dir, args.max_hf_test)

    # ── Step 2: Kaggle Air-to-Air (720p) ──────────────────────────
    if not args.skip_kaggle:
        print("\n" + "="*58)
        print("  STEP 2/3 - Kaggle: air-to-air-object-detection (720p)")
        print("="*58)
        download_kaggle(args.kaggle_dir)
        process_kaggle(args.kaggle_dir, args.output_dir)

    # ── Step 3: Panshull negatives ─────────────────────────────────
    if not args.skip_panshull and os.path.isdir(args.panshull_dir):
        print("\n" + "="*58)
        print("  STEP 3/3 - Panshull: background negative images")
        print("="*58)
        add_panshull_negatives(args.panshull_dir, args.output_dir)
    else:
        print(f"\n  INFO: Panshull dir '{args.panshull_dir}' not found - skipping.")
        print(f"  To add it: kaggle datasets download -d panshull/full-dataset "
              f"--unzip -p {args.panshull_dir}")

    print_summary(args.output_dir)
    print(f"\nDone! Update data_config.yaml:")
    print(f"  data_root: \"{os.path.abspath(args.output_dir)}\"")
    print(f"\nThen train:")
    print(f"  python train.py --data-root {args.output_dir} "
          f"--epochs 100 --batch-size 32 --device cuda")


if __name__ == "__main__":
    main()
