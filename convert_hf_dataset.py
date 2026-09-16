"""
Download and convert the Hugging Face 'pathikg/drone-detection-dataset'
into the exact S3-Det directory layout:

    <output_dir>/
      train/
        drone/      (images + YOLO .txt annotations: '0 xc yc w h')
        no_drone/   (background negatives, no drone present)
      test/
        drone/
        no_drone/

Prerequisites:
    pip install datasets pillow tqdm

Usage:
    # Convert entire dataset (51k train, 2.6k test)
    python convert_hf_dataset.py --output-dir dataset_hf

    # Convert a smaller subset for quick experimentation (e.g. 2000 train, 500 test)
    python convert_hf_dataset.py --output-dir dataset_subset --max-train 2000 --max-test 500
"""
import argparse
import os
from io import BytesIO
from PIL import Image

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None


def convert_split(dataset_split, split_name, output_dir, max_samples=None):
    drone_dir = os.path.join(output_dir, split_name, "drone")
    no_drone_dir = os.path.join(output_dir, split_name, "no_drone")
    os.makedirs(drone_dir, exist_ok=True)
    os.makedirs(no_drone_dir, exist_ok=True)

    pos_count = 0
    neg_count = 0
    total_boxes = 0

    count = 0
    for sample in dataset_split:
        if max_samples and count >= max_samples:
            break

        img_id = sample.get("image_id", count)
        img_raw = sample["image"]
        width = sample.get("width")
        height = sample.get("height")

        # Load PIL image
        if isinstance(img_raw, Image.Image):
            img = img_raw.convert("RGB")
        elif isinstance(img_raw, dict) and "bytes" in img_raw:
            img = Image.open(BytesIO(img_raw["bytes"])).convert("RGB")
        else:
            img = Image.open(img_raw).convert("RGB")

        if width is None or height is None:
            width, height = img.size

        objects = sample.get("objects", {})
        bboxes = objects.get("bbox", [])

        # Process Bounding Boxes (COCO [x_min, y_min, w, h] -> YOLO normalized [xc, yc, w, h])
        yolo_lines = []
        for box in bboxes:
            if len(box) >= 4:
                x_min, y_min, bw, bh = box[:4]
                if bw <= 0 or bh <= 0:
                    continue
                xc = (x_min + bw / 2.0) / width
                yc = (y_min + bh / 2.0) / height
                nw = bw / width
                nh = bh / height
                # Clamp to 0-1 range
                xc = min(max(xc, 0.0), 1.0)
                yc = min(max(yc, 0.0), 1.0)
                nw = min(max(nw, 0.0), 1.0)
                nh = min(max(nh, 0.0), 1.0)
                yolo_lines.append(f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n")

        stem = f"drone_{split_name}_{img_id:06d}"

        if len(yolo_lines) > 0:
            # Positive sample with drone target
            img_path = os.path.join(drone_dir, f"{stem}.jpg")
            lbl_path = os.path.join(drone_dir, f"{stem}.txt")
            img.save(img_path, quality=95)
            with open(lbl_path, "w") as f:
                f.writelines(yolo_lines)
            pos_count += 1
            total_boxes += len(yolo_lines)
        else:
            # Negative background sample (no drone)
            img_path = os.path.join(no_drone_dir, f"{stem}.jpg")
            img.save(img_path, quality=95)
            neg_count += 1

        count += 1
        if count % 1000 == 0:
            print(f"[{split_name}] Processed {count} images ({pos_count} positive, {neg_count} negative, {total_boxes} boxes)...")

    print(f"[{split_name} Complete] Total: {count} images | Positive: {pos_count} | Negative: {neg_count} | Total Boxes: {total_boxes}")


def main():
    parser = argparse.ArgumentParser(description="Convert HF drone-detection-dataset to S3-Det format")
    parser.add_argument("--output-dir", default="dataset_hf", help="Directory where converted dataset will be saved")
    parser.add_argument("--max-train", type=int, default=None, help="Max train samples to process (default: all)")
    parser.add_argument("--max-test", type=int, default=None, help="Max test samples to process (default: all)")
    args = parser.parse_args()

    if load_dataset is None:
        raise ImportError(
            "The 'datasets' library is required to run this converter.\n"
            "Please run: pip install datasets"
        )

    print("Connecting to Hugging Face: pathikg/drone-detection-dataset (streaming mode)...")
    hf_dataset = load_dataset("pathikg/drone-detection-dataset", streaming=True)

    if "train" in hf_dataset:
        print("\nProcessing train split...")
        convert_split(hf_dataset["train"], "train", args.output_dir, args.max_train)

    if "test" in hf_dataset:
        print("\nProcessing test split...")
        convert_split(hf_dataset["test"], "test", args.output_dir, args.max_test)


    print("\nDataset conversion finished!")
    print(f"To train S3-Det on this dataset, update data_config.yaml:")
    print(f"  data_root: \"{os.path.abspath(args.output_dir)}\"")


if __name__ == "__main__":
    main()
