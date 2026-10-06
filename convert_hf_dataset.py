"""
Master Dataset Converter - S3-Det
==========================================================================
Combines FIVE dataset sources into the S3-Det directory layout:

  <output_dir>/
    train/
      drone/      <- images + YOLO .txt  (class 0 = drone/UAV)
      no_drone/   <- images ONLY, NO .txt (model learns to ignore these)
    test/
      drone/
      no_drone/

SOURCE 1: HuggingFace 'pathikg/drone-detection-dataset'
  - ~54k drone images (train + test splits)

SOURCE 2: Kaggle 'avldevelopment/air-to-air-object-detection-dataset' (720p)
  - UAV -> drone/, Bird/Helicopter -> no_drone/

SOURCE 3: Kaggle 'panshull/full-dataset'
  - Pure background negatives -> no_drone/

SOURCE 4: HuggingFace 'chriamue/bird-species-dataset'   [--hf-birds]
  - Real-world bird photos -> no_drone/ (hard negatives for bird/drone confusion)
  - 90% train, 10% test split applied automatically

SOURCE 5: 8 Roboflow Datasets                           [--roboflow-key KEY]
  - myworkspace-0p4nk/drone-bird-detection-backup
  - drovsbir/birdvsdrone1
  - dataset-xe876/bvd3500
  - sky-sd2zq/bird_only-pt0bm
  - milind-ashok-dadore/small-aerial-object-detection-fwbd7
  - abdulrahman-eidhah/drone-detection-new
  - ghada-34ysy/air-vehicles
  - yolov12-drone-detection/drone-jrg57
  Classes with drone/uav -> drone/, everything else -> no_drone/
  Train/test split: 90%/10% applied per dataset

NOTE: CARS folder is NEVER touched — it is strictly for evaluation only!

Usage:
  # Original 3 sources only:
  python convert_hf_dataset.py --output-dir dataset_full

  # Add HF bird species as hard negatives:
  python convert_hf_dataset.py --output-dir dataset_full --no-wipe --hf-birds

  # Add all 8 Roboflow datasets:
  python convert_hf_dataset.py --output-dir dataset_full --no-wipe --roboflow-key YOUR_KEY

  # Full build (all 5 sources at once):
  python convert_hf_dataset.py --output-dir dataset_full --hf-birds --roboflow-key YOUR_KEY

Prerequisites:
  pip install datasets pillow kaggle roboflow pyyaml
  Place kaggle.json in ~/.kaggle/ before running Kaggle steps
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
# PART 4 - HuggingFace 'chriamue/bird-species-dataset'
#           Real bird photos -> no_drone/  (hard negatives)
# ============================================================

DRONE_KEYWORDS = {"drone", "uav", "quadcopter", "multirotor", "uas"}

ROBOFLOW_PROJECTS = [
    {"workspace": "myworkspace-0p4nk", "project": "drone-bird-detection-backup", "version": 1, "desc": "Drone & Bird Detection"},
    {"workspace": "drovsbir",           "project": "birdvsdrone1",                "version": 1, "desc": "Bird vs Drone 1"},
    {"workspace": "dataset-xe876",      "project": "bvd3500",                     "version": 1, "desc": "BVD 3500"},
    {"workspace": "sky-sd2zq",          "project": "bird_only-pt0bm",             "version": 1, "desc": "Bird Only (pure negatives)"},
    {"workspace": "milind-ashok-dadore","project": "small-aerial-object-detection-fwbd7", "version": 1, "desc": "Small Aerial Object Detection"},
    {"workspace": "abdulrahman-eidhah", "project": "drone-detection-new",         "version": 1, "desc": "Drone Detection New"},
    {"workspace": "ghada-34ysy",        "project": "air-vehicles",                "version": 1, "desc": "Air Vehicles"},
    {"workspace": "yolov12-drone-detection", "project": "drone-jrg57",            "version": 1, "desc": "YOLOv12 Drone Detection"},
]


def add_hf_bird_species(output_dir, max_samples=None, train_ratio=0.90):
    """
    Streams chriamue/bird-species-dataset from HuggingFace.
    All images are real birds -> no_drone/ (hard negative examples).
    Applies a 90/10 train/test split.
    """
    if load_dataset is None:
        raise ImportError("Run: pip install datasets")

    print("\n  Streaming HF bird-species-dataset...")
    try:
        # pass trust_remote_code=True because HF deprecated automatic script execution
        ds = load_dataset("chriamue/bird-species-dataset", split="train", streaming=True, trust_remote_code=True)
    except Exception as e:
        print(f"  [Warning] Failed to load chriamue/bird-species-dataset: {e}")
        return

    train_no_drone = os.path.join(output_dir, "train", "no_drone")
    test_no_drone  = os.path.join(output_dir, "test",  "no_drone")
    os.makedirs(train_no_drone, exist_ok=True)
    os.makedirs(test_no_drone,  exist_ok=True)

    saved_train = saved_test = count = 0
    rng = random.Random(42)

    for sample in ds:
        if max_samples and count >= max_samples:
            break

        img_raw = sample.get("image")
        if img_raw is None:
            count += 1
            continue

        try:
            if isinstance(img_raw, Image.Image):
                img = img_raw.convert("RGB")
            elif isinstance(img_raw, dict) and "bytes" in img_raw:
                img = Image.open(BytesIO(img_raw["bytes"])).convert("RGB")
            else:
                img = Image.open(img_raw).convert("RGB")

            stem = f"hf_birdspec_{count:07d}.jpg"
            if rng.random() < train_ratio:
                img.save(os.path.join(train_no_drone, stem), quality=92)
                saved_train += 1
            else:
                img.save(os.path.join(test_no_drone, stem), quality=92)
                saved_test += 1
        except Exception:
            pass

        count += 1
        if count % 1000 == 0:
            print(f"    [HF birds] {count} processed -> train:{saved_train}  test:{saved_test}")

    print(f"  [HF bird-species DONE]  train/no_drone +{saved_train}  test/no_drone +{saved_test}")


# ============================================================
# PART 5 - Roboflow Datasets (8 datasets)
#           Drone classes -> drone/ + YOLO .txt
#           All other classes -> no_drone/ (no .txt)
#           90% train / 10% test split applied per image
# ============================================================

def _parse_rf_yaml(yaml_path):
    """Parses Roboflow data.yaml to get {class_id: class_name} map."""
    try:
        import yaml
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        names = data.get("names", [])
        if isinstance(names, list):
            return {i: n.lower().strip() for i, n in enumerate(names)}
        elif isinstance(names, dict):
            return {int(k): str(v).lower().strip() for k, v in names.items()}
    except Exception:
        pass
    return {}


def _is_drone_class(class_name):
    return any(kw in class_name for kw in DRONE_KEYWORDS)


def _ingest_roboflow_folder(rf_folder, output_dir, prefix, train_ratio=0.90):
    """
    Ingests one unzipped Roboflow YOLO dataset folder.

    Scans for train/, valid/, test/ subfolders. For each image:
      - If any annotation line maps to a drone class -> drone/ + YOLO .txt (class 0 only)
      - Otherwise -> no_drone/ (no .txt)
    Applies a fresh 90/10 train/test split regardless of Roboflow's original split.
    """
    IMG_EXTS_SET = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    drone_train    = os.path.join(output_dir, "train", "drone")
    no_drone_train = os.path.join(output_dir, "train", "no_drone")
    drone_test     = os.path.join(output_dir, "test",  "drone")
    no_drone_test  = os.path.join(output_dir, "test",  "no_drone")
    for d in [drone_train, no_drone_train, drone_test, no_drone_test]:
        os.makedirs(d, exist_ok=True)

    # Parse class map from data.yaml
    yaml_files = glob.glob(os.path.join(rf_folder, "**", "data.yaml"), recursive=True)
    class_map = _parse_rf_yaml(yaml_files[0]) if yaml_files else {}
    if class_map:
        drone_ids = {cid for cid, name in class_map.items() if _is_drone_class(name)}
        print(f"    Class map: {class_map}  |  drone_ids: {drone_ids}")
    else:
        # No class map -> assume single-class drone dataset if name suggests drones
        folder_lower = os.path.basename(rf_folder).lower()
        drone_ids = {0} if any(kw in folder_lower for kw in DRONE_KEYWORDS) else set()
        print(f"    No class map found, inferring drone_ids={drone_ids} from folder name")

    # Collect all image files (flatten all Roboflow splits into one pool)
    all_img_paths = []
    for root, _, files in os.walk(rf_folder):
        for fname in files:
            if os.path.splitext(fname)[1].lower() in IMG_EXTS_SET:
                all_img_paths.append(os.path.join(root, fname))

    rng = random.Random(42)
    pos_train = pos_test = neg_train = neg_test = 0

    for idx, img_path in enumerate(all_img_paths):
        ext = os.path.splitext(img_path)[1]
        base = os.path.splitext(os.path.basename(img_path))[0]

        # Find matching label file
        label_path = os.path.splitext(img_path)[0] + ".txt"
        if not os.path.isfile(label_path):
            # Try sibling labels/ directory
            parent = os.path.dirname(img_path)
            labels_dir = os.path.join(os.path.dirname(parent), "labels")
            alt = os.path.join(labels_dir, base + ".txt")
            if os.path.isfile(alt):
                label_path = alt

        # Parse annotation lines
        has_drone = False
        drone_yolo_lines = []

        if os.path.isfile(label_path):
            with open(label_path, "r") as lf:
                for line in lf:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cid = int(parts[0])
                        if cid in drone_ids:
                            has_drone = True
                            # Remap to class 0 (our universal drone class)
                            drone_yolo_lines.append(f"0 {' '.join(parts[1:5])}\n")
        elif drone_ids == {0} and not class_map:
            # Drone-only dataset with no label file = background-free drone image
            # Treat unlabelled as negative to be safe
            has_drone = False

        # 90/10 split
        use_train = rng.random() < train_ratio
        new_stem = f"rf_{prefix}_{idx:07d}"

        if has_drone:
            dest_dir = drone_train if use_train else drone_test
            shutil.copy(img_path, os.path.join(dest_dir, f"{new_stem}{ext}"))
            with open(os.path.join(dest_dir, f"{new_stem}.txt"), "w") as f:
                f.writelines(drone_yolo_lines)
            if use_train: pos_train += 1
            else:         pos_test  += 1
        else:
            dest_dir = no_drone_train if use_train else no_drone_test
            shutil.copy(img_path, os.path.join(dest_dir, f"{new_stem}{ext}"))
            if use_train: neg_train += 1
            else:         neg_test  += 1

    print(f"    [Done] drone: train+{pos_train} test+{pos_test}  |  "
          f"no_drone: train+{neg_train} test+{neg_test}")
    return pos_train + pos_test, neg_train + neg_test


def add_roboflow_datasets(api_key, output_dir, tmp_dir="tmp_roboflow"):
    """Downloads all 8 Roboflow datasets via the Roboflow SDK and merges them."""
    try:
        from roboflow import Roboflow
    except ImportError:
        raise ImportError("Run: pip install roboflow pyyaml")

    os.makedirs(tmp_dir, exist_ok=True)
    rf = Roboflow(api_key=api_key)

    total_pos = total_neg = 0
    for item in ROBOFLOW_PROJECTS:
        ws, proj, ver, desc = item["workspace"], item["project"], item["version"], item["desc"]
        print(f"\n  --> {desc}  ({ws}/{proj})")
        try:
            dest = os.path.join(tmp_dir, f"{ws}_{proj}")
            project_obj = rf.workspace(ws).project(proj)
            
            # Try to get the available versions dynamically to avoid "Version not found" errors
            available_versions = []
            try:
                available_versions = project_obj.versions()
            except Exception:
                pass
            
            target_version = ver
            if available_versions:
                # Pick the latest version available in the project
                target_version = available_versions[-1].version
                
            print(f"      Downloading version {target_version}...")
            project_obj.version(target_version).download("yolov8", location=dest)
            
            prefix = f"{ws[:6]}_{proj[:8]}".replace("-", "").replace("_", "")[:12]
            pos, neg = _ingest_roboflow_folder(dest, output_dir, prefix=prefix)
            total_pos += pos
            total_neg += neg
            shutil.rmtree(dest, ignore_errors=True)
        except Exception as e:
            print(f"    [Warning] Failed {ws}/{proj}: {e}")

    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"\n  [Roboflow DONE]  +{total_pos} drone images  +{total_neg} no_drone images")


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
    parser.add_argument("--hf-birds",      action="store_true",
                        help="Add HuggingFace bird-species-dataset as hard negatives (Source 4)")
    parser.add_argument("--roboflow-key",  type=str, default=None,
                        help="Roboflow API key to download all 8 drone/bird datasets (Source 5)")
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
        hf = load_dataset("pathikg/drone-detection-dataset", streaming=True, trust_remote_code=True)
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
        print("  STEP 3 - Panshull: background negative images")
        print("="*58)
        add_panshull_negatives(args.panshull_dir, args.output_dir)
    else:
        print(f"\n  INFO: Panshull dir '{args.panshull_dir}' not found - skipping.")
        print(f"  To add it: kaggle datasets download -d panshull/full-dataset "
              f"--unzip -p {args.panshull_dir}")

    # ── Step 4: HuggingFace Bird Species (hard negatives) ──────────
    if args.hf_birds:
        print("\n" + "="*58)
        print("  STEP 4 - HuggingFace: chriamue/bird-species-dataset")
        print("           (real bird photos -> no_drone/, 90/10 split)")
        print("="*58)
        add_hf_bird_species(args.output_dir)

    # ── Step 5: Roboflow Datasets ──────────────────────────────────
    if args.roboflow_key:
        print("\n" + "="*58)
        print("  STEP 5 - Roboflow: 8 drone + bird datasets")
        print("           (drone classes -> drone/, others -> no_drone/, 90/10 split)")
        print("="*58)
        add_roboflow_datasets(args.roboflow_key, args.output_dir)

    print_summary(args.output_dir)
    print(f"\nDone! Update data_config.yaml:")
    print(f"  data_root: \"{os.path.abspath(args.output_dir)}\"")
    print(f"\nThen train:")
    print(f"  python train.py --data-root {args.output_dir} "
          f"--epochs 100 --batch-size 32 --device cuda")


if __name__ == "__main__":
    main()
