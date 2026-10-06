"""
Dataset Expansion Tool for S3-Det and Drone Classifiers
=======================================================
Adds extra real-world Bird and Drone datasets to `dataset_full/train/`:
  1. HuggingFace: 'chriamue/bird-species-dataset' -> Pure Birds (no_drone)
  2. Roboflow Datasets (Bird vs Drone, Small Aerial Objects, Air Vehicles):
     - myworkspace-0p4nk/drone-bird-detection-backup
     - drovsbir/birdvsdrone1
     - dataset-xe876/bvd3500
     - sky-sd2zq/bird_only-pt0bm
     - milind-ashok-dadore/small-aerial-object-detection-fwbd7
     - abdulrahman-eidhah/drone-detection-new
     - ghada-34ysy/air-vehicles
     - yolov12-drone-detection/drone-jrg57

NOTE: CARS dataset is STRICTLY kept separate for testing and is NEVER modified!

Usage:
  # 1. Download & add HuggingFace bird species dataset:
  python add_extra_datasets.py --hf-birds

  # 2. Download & add Roboflow datasets using your free Roboflow API key:
  python add_extra_datasets.py --roboflow-key YOUR_API_KEY

  # 3. If you manually downloaded/unzipped Roboflow folders into a directory:
  python add_extra_datasets.py --ingest-dir /path/to/downloaded_folders

  # 4. Or do both at once:
  python add_extra_datasets.py --hf-birds --roboflow-key YOUR_API_KEY
"""

import os
import sys
import glob
import shutil
import argparse
from io import BytesIO
from typing import Dict, List, Tuple
from PIL import Image

ROBOFLOW_PROJECTS = [
    {
        "workspace": "myworkspace-0p4nk",
        "project": "drone-bird-detection-backup",
        "version": 1,
        "desc": "Drone & Bird Detection",
    },
    {
        "workspace": "drovsbir",
        "project": "birdvsdrone1",
        "version": 1,
        "desc": "Bird vs Drone 1",
    },
    {
        "workspace": "dataset-xe876",
        "project": "bvd3500",
        "version": 1,
        "desc": "BVD 3500 (Bird vs Drone 3500)",
    },
    {
        "workspace": "sky-sd2zq",
        "project": "bird_only-pt0bm",
        "version": 1,
        "desc": "Bird Only (Pure Negatives)",
    },
    {
        "workspace": "milind-ashok-dadore",
        "project": "small-aerial-object-detection-fwbd7",
        "version": 1,
        "desc": "Small Aerial Object Detection",
    },
    {
        "workspace": "abdulrahman-eidhah",
        "project": "drone-detection-new",
        "version": 1,
        "desc": "Drone Detection New",
    },
    {
        "workspace": "ghada-34ysy",
        "project": "air-vehicles",
        "version": 1,
        "desc": "Air Vehicles (Helicopters/Planes vs Drones)",
    },
    {
        "workspace": "yolov12-drone-detection",
        "project": "drone-jrg57",
        "version": 1,
        "desc": "YOLOv12 Drone Detection",
    },
]

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Keywords for classification
BIRD_KEYWORDS = ["bird", "birds", "avian", "flyer", "pigeon", "seagull", "crow", "eagle"]
DRONE_KEYWORDS = ["drone", "drones", "uav", "quadcopter", "multirotor"]
OTHER_VEHICLE_KEYWORDS = ["plane", "airplane", "helicopter", "aircraft", "balloon", "airliner"]


# ── Step 1: HuggingFace Bird Species Dataset ──────────────────────────────────
def add_hf_bird_species(output_dir: str = "dataset_full", max_samples: int = None):
    """
    Downloads chriamue/bird-species-dataset from HuggingFace.
    All images are birds -> saved directly to <output_dir>/train/no_drone/.
    """
    print("\n" + "=" * 78)
    print("  [HuggingFace] Downloading 'chriamue/bird-species-dataset' (Birds -> no_drone)")
    print("=" * 78)

    try:
        from datasets import load_dataset
    except ImportError:
        print("[Error] HuggingFace datasets library not installed. Run: pip install datasets")
        return 0

    no_drone_dir = os.path.join(output_dir, "train", "no_drone")
    os.makedirs(no_drone_dir, exist_ok=True)

    try:
        ds = load_dataset("chriamue/bird-species-dataset", split="train", streaming=True)
    except Exception as e:
        print(f"[Error loading HF dataset]: {e}")
        return 0

    count = 0
    saved = 0
    prefix = "hf_birdspec"

    print("  Streaming images and adding to train/no_drone/...")
    for sample in ds:
        if max_samples and count >= max_samples:
            break

        count += 1
        img_raw = sample.get("image")
        if img_raw is None:
            continue

        try:
            if isinstance(img_raw, Image.Image):
                img = img_raw.convert("RGB")
            elif isinstance(img_raw, dict) and "bytes" in img_raw:
                img = Image.open(BytesIO(img_raw["bytes"])).convert("RGB")
            else:
                img = Image.open(img_raw).convert("RGB")

            filename = f"{prefix}_{count:07d}.jpg"
            img.save(os.path.join(no_drone_dir, filename), quality=92)
            saved += 1

            if saved % 500 == 0:
                print(f"    Saved {saved} real bird negative images...")
        except Exception as e:
            continue

    print(f"  --> Successfully added {saved} real bird images to '{no_drone_dir}'!\n")
    return saved


# ── Step 2: Roboflow Dataset Ingestion ────────────────────────────────────────
def parse_roboflow_yaml(yaml_path: str) -> Dict[int, str]:
    """Parses data.yaml from Roboflow YOLO download to determine class mapping."""
    import yaml
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    names = data.get("names", [])
    if isinstance(names, list):
        return {idx: name.lower().strip() for idx, name in enumerate(names)}
    elif isinstance(names, dict):
        return {int(idx): str(name).lower().strip() for idx, name in names.items()}
    return {}


def ingest_roboflow_folder(rf_folder: str, output_dir: str = "dataset_full") -> Tuple[int, int]:
    """
    Scans an unzipped Roboflow folder, parses data.yaml, and copies images/annotations:
      - Images with drone boxes -> train/drone/ (+ YOLO .txt box annotations)
      - Images with bird/plane/no-drone boxes (or no boxes) -> train/no_drone/
    """
    drone_dir = os.path.join(output_dir, "train", "drone")
    no_drone_dir = os.path.join(output_dir, "train", "no_drone")
    os.makedirs(drone_dir, exist_ok=True)
    os.makedirs(no_drone_dir, exist_ok=True)

    yaml_files = glob.glob(os.path.join(rf_folder, "**", "data.yaml"), recursive=True)
    class_map = {}
    if yaml_files:
        try:
            class_map = parse_roboflow_yaml(yaml_files[0])
            print(f"    Found class map in {os.path.basename(yaml_files[0])}: {class_map}")
        except Exception:
            pass

    # Find image directories (train/valid/test within the downloaded folder)
    img_files = []
    for root, _, files in os.walk(rf_folder):
        for f in files:
            if f.lower().endswith(IMG_EXTS) and not f.startswith("._"):
                img_files.append(os.path.join(root, f))

    pos_added = 0
    neg_added = 0
    folder_prefix = os.path.basename(os.path.normpath(rf_folder))[:10].replace(" ", "_")

    for idx, img_path in enumerate(img_files):
        ext = os.path.splitext(img_path)[1]
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.splitext(img_path)[0] + ".txt"

        # Also check ../labels/base_name.txt
        if not os.path.isfile(label_path):
            parent = os.path.dirname(img_path)
            labels_dir = os.path.join(os.path.dirname(parent), "labels")
            alt_label = os.path.join(labels_dir, base_name + ".txt")
            if os.path.isfile(alt_label):
                label_path = alt_label

        has_drone = False
        drone_lines = []

        if os.path.isfile(label_path):
            with open(label_path, "r") as lf:
                for line in lf:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        cls_name = class_map.get(cls_id, "")

                        is_drone = False
                        # If class name has drone keywords, or if project only contains drones
                        if cls_name:
                            if any(k in cls_name for k in DRONE_KEYWORDS):
                                is_drone = True
                        else:
                            # If no class map, check if folder name hints drone only
                            if "drone" in rf_folder.lower() and "bird" not in rf_folder.lower():
                                is_drone = True

                        if is_drone:
                            has_drone = True
                            # Format box for class 0 (drone)
                            drone_lines.append(f"0 {' '.join(parts[1:5])}\n")

        new_stem = f"rf_{folder_prefix}_{idx:06d}"
        if has_drone:
            # Positive: drone
            dest_img = os.path.join(drone_dir, f"{new_stem}{ext}")
            dest_txt = os.path.join(drone_dir, f"{new_stem}.txt")
            shutil.copy(img_path, dest_img)
            with open(dest_txt, "w") as f:
                f.writelines(drone_lines)
            pos_added += 1
        else:
            # Negative: no_drone (bird, aircraft, or background)
            dest_img = os.path.join(no_drone_dir, f"{new_stem}{ext}")
            shutil.copy(img_path, dest_img)
            neg_added += 1

    return pos_added, neg_added


def download_roboflow_projects(api_key: str, tmp_dir: str = "tmp_roboflow", output_dir: str = "dataset_full"):
    """Downloads all 8 Roboflow projects via official Roboflow SDK and merges them."""
    try:
        from roboflow import Roboflow
    except ImportError:
        print("[Error] Roboflow package not installed. Run: pip install roboflow pyyaml")
        return

    os.makedirs(tmp_dir, exist_ok=True)
    rf = Roboflow(api_key=api_key)

    total_pos = 0
    total_neg = 0

    print("\n" + "=" * 78)
    print(f"  [Roboflow] Downloading & Processing {len(ROBOFLOW_PROJECTS)} Datasets")
    print("=" * 78)

    for item in ROBOFLOW_PROJECTS:
        ws = item["workspace"]
        proj = item["project"]
        desc = item["desc"]
        v_num = item["version"]

        print(f"\n--> Fetching: {desc} ({ws}/{proj})...")
        try:
            project_obj = rf.workspace(ws).project(proj)
            version_obj = project_obj.version(v_num)
            dest_path = os.path.join(tmp_dir, f"{ws}_{proj}")

            print(f"    Downloading in YOLO format...")
            version_obj.download("yolov8", location=dest_path)

            pos, neg = ingest_roboflow_folder(dest_path, output_dir=output_dir)
            total_pos += pos
            total_neg += neg
            print(f"    [Done] Added {pos} drone images and {neg} no_drone images.")

            # Clean up raw folder to save disk space
            shutil.rmtree(dest_path, ignore_errors=True)
        except Exception as e:
            print(f"    [Warning] Failed to download {ws}/{proj}: {e}")

    # Remove tmp folder
    shutil.rmtree(tmp_dir, ignore_errors=True)

    print("\n" + "=" * 78)
    print(f"  Roboflow Ingestion Complete! Added {total_pos} drones and {total_neg} no_drones.")
    print("=" * 78 + "\n")


def print_dataset_summary(data_root: str = "dataset_full"):
    """Prints current dataset count."""
    train_drone = len([f for f in glob.glob(os.path.join(data_root, "train", "drone", "*")) if f.lower().endswith(IMG_EXTS)])
    train_no_drone = len([f for f in glob.glob(os.path.join(data_root, "train", "no_drone", "*")) if f.lower().endswith(IMG_EXTS)])

    test_drone = len([f for f in glob.glob(os.path.join(data_root, "test", "drone", "*")) if f.lower().endswith(IMG_EXTS)])
    test_no_drone = len([f for f in glob.glob(os.path.join(data_root, "test", "no_drone", "*")) if f.lower().endswith(IMG_EXTS)])

    print("\n" + "█" * 78)
    print("  CURRENT DATASET INVENTORY")
    print("█" * 78)
    print(f"  Train split:")
    print(f"    - drone    : {train_drone:>7d} images (class 1)")
    print(f"    - no_drone : {train_no_drone:>7d} images (class 0)")
    print(f"    - TOTAL    : {train_drone + train_no_drone:>7d} images")
    if test_drone or test_no_drone:
        print(f"  Test split (Held-out):")
        print(f"    - drone    : {test_drone:>7d} images")
        print(f"    - no_drone : {test_no_drone:>7d} images")
    print("█" * 78 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Expand dataset with real Bird and Drone datasets")
    parser.add_argument("--output-dir", type=str, default="dataset_full", help="Target dataset root folder")
    parser.add_argument("--hf-birds", action="store_true", help="Download chriamue/bird-species-dataset from HF")
    parser.add_argument("--roboflow-key", type=str, default=None, help="Roboflow API key to download all 8 datasets")
    parser.add_argument("--ingest-dir", type=str, default=None, help="Path to manually unzipped Roboflow folder(s)")
    parser.add_argument("--max-hf", type=int, default=None, help="Max HF bird samples to download")
    args = parser.parse_args()

    # Safety check: CARS directory protection
    if "cars" in args.output_dir.lower():
        print("[FATAL] Refusing to modify CARS directory! CARS is reserved strictly for evaluation.")
        sys.exit(1)

    has_action = False

    # 1. HuggingFace Birds
    if args.hf_birds:
        has_action = True
        add_hf_bird_species(output_dir=args.output_dir, max_samples=args.max_hf)

    # 2. Roboflow via API key
    if args.roboflow_key:
        has_action = True
        download_roboflow_projects(api_key=args.roboflow_key, output_dir=args.output_dir)

    # 3. Manual Ingest Directory
    if args.ingest_dir and os.path.isdir(args.ingest_dir):
        has_action = True
        print(f"\nIngesting manually downloaded Roboflow directory: {args.ingest_dir}")
        subdirs = [os.path.join(args.ingest_dir, d) for d in os.listdir(args.ingest_dir) if os.path.isdir(os.path.join(args.ingest_dir, d))]
        if not subdirs:
            subdirs = [args.ingest_dir]

        total_p = 0
        total_n = 0
        for sd in subdirs:
            print(f"  Processing: {os.path.basename(sd)}...")
            p, n = ingest_roboflow_folder(sd, output_dir=args.output_dir)
            total_p += p
            total_n += n
            print(f"    --> Added {p} drones, {n} no_drones.")
        print(f"Total Added: {total_p} drones, {total_n} no_drones.\n")

    if not has_action:
        print("[Notice] No action specified. Usage examples:")
        print("  python add_extra_datasets.py --hf-birds")
        print("  python add_extra_datasets.py --roboflow-key <YOUR_KEY>")
        print("  python add_extra_datasets.py --ingest-dir <PATH_TO_ROBOFLOW_UNZIPPED>")

    print_dataset_summary(args.output_dir)


if __name__ == "__main__":
    main()
