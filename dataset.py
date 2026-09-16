"""Dataset loader for a layout like:

    <root>/
      train/
        drone/      images (+ YOLO .txt labels, class id 0 = drone)
        no_drone/   images (treated as background negatives -- no objects)
      test/
        drone/
        no_drone/

Inside drone/ (and no_drone/, if it has labels at all) either of these is
accepted automatically:
    drone/images/*.jpg + drone/labels/*.txt
    drone/*.jpg + drone/*.txt                (same folder)

Label files are standard YOLO txt: one line per box,
    class_id x_center y_center width height     (all normalized 0-1)

If your layout differs (different subfolder names, VOC/COCO labels, etc.)
adjust `_find_images_and_labels` / `_load_labels` below -- everything else
is layout-agnostic.
"""
import os
import glob
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _find_images_and_labels(class_dir):
    img_subdir = os.path.join(class_dir, "images")
    lbl_subdir = os.path.join(class_dir, "labels")
    if os.path.isdir(img_subdir):
        images = sorted(p for p in glob.glob(os.path.join(img_subdir, "*"))
                         if p.lower().endswith(IMG_EXTS))
        label_dir = lbl_subdir if os.path.isdir(lbl_subdir) else img_subdir
    else:
        images = sorted(p for p in glob.glob(os.path.join(class_dir, "*"))
                         if p.lower().endswith(IMG_EXTS))
        label_dir = class_dir

    pairs = []
    for img_path in images:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(label_dir, stem + ".txt")
        pairs.append((img_path, label_path if os.path.isfile(label_path) else None))
    return pairs


def letterbox(image, target_size):
    w, h = image.size
    scale = min(target_size / w, target_size / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    image = image.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (target_size, target_size), (114, 114, 114))
    pad_x, pad_y = (target_size - nw) // 2, (target_size - nh) // 2
    canvas.paste(image, (pad_x, pad_y))
    return canvas, scale, pad_x, pad_y


def _augment(image, boxes, target_size):
    """Full augmentation pipeline for training (paper Section 3.3):
    1. Random scale [0.5, 1.5] → letterbox
    2. Color jitter (brightness/contrast/saturation ±30%)
    3. Random perspective warp (p=0.3)
    4. Horizontal flip (p=0.5)

    Args:
        image: PIL.Image in original resolution
        boxes: list of [x1,y1,x2,y2] in original pixel coords (may be empty)
        target_size: int, output canvas size

    Returns:
        image: PIL.Image at target_size × target_size
        boxes: np.ndarray (N,4) at target_size coords, or empty (0,4)
    """
    import numpy as np

    w, h = image.size
    boxes_arr = np.array(boxes, dtype=np.float32).reshape(-1, 4) if boxes else np.zeros((0, 4), dtype=np.float32)

    # 1. Random scale  [0.5, 1.5]
    scale_factor = random.uniform(0.5, 1.5)
    new_w = max(1, int(round(w * scale_factor)))
    new_h = max(1, int(round(h * scale_factor)))
    image = image.resize((new_w, new_h), Image.BILINEAR)
    if len(boxes_arr):
        boxes_arr[:, [0, 2]] *= scale_factor
        boxes_arr[:, [1, 3]] *= scale_factor

    # 2. Letterbox to target_size
    image, lb_scale, pad_x, pad_y = letterbox(image, target_size)
    if len(boxes_arr):
        boxes_arr[:, [0, 2]] = boxes_arr[:, [0, 2]] * lb_scale + pad_x
        boxes_arr[:, [1, 3]] = boxes_arr[:, [1, 3]] * lb_scale + pad_y

    # 3. Color jitter — brightness, contrast, saturation ±30%
    from PIL import ImageEnhance
    for Enhancer in [ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color]:
        factor = random.uniform(0.7, 1.3)
        image = Enhancer(image).enhance(factor)

    # 4. Random perspective (p=0.3) — only if there are boxes to transform
    if random.random() < 0.3:
        img_np = np.array(image, dtype=np.uint8)
        img_h, img_w = img_np.shape[:2]
        distort = target_size * 0.05  # max 5% of image size per corner perturbation
        src_pts = np.float32([[0, 0], [img_w, 0], [img_w, img_h], [0, img_h]])
        dst_pts = src_pts + np.random.uniform(-distort, distort, src_pts.shape).astype(np.float32)
        dst_pts[:, 0] = dst_pts[:, 0].clip(0, img_w)
        dst_pts[:, 1] = dst_pts[:, 1].clip(0, img_h)
        try:
            import cv2
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            img_np = cv2.warpPerspective(img_np, M, (img_w, img_h), borderValue=(114, 114, 114))
            image = Image.fromarray(img_np)
            if len(boxes_arr):
                # Transform all 4 corners of each box through the homography
                corners = np.stack([
                    boxes_arr[:, [0, 1]], boxes_arr[:, [2, 1]],
                    boxes_arr[:, [2, 3]], boxes_arr[:, [0, 3]]
                ], axis=1).reshape(-1, 2)  # (N*4, 2)
                ones = np.ones((corners.shape[0], 1), dtype=np.float32)
                pts_h = np.hstack([corners, ones])  # (N*4, 3)
                transformed = (M @ pts_h.T).T  # (N*4, 3)
                transformed = transformed[:, :2] / (transformed[:, 2:3] + 1e-7)
                transformed = transformed.reshape(-1, 4, 2)
                x_coords = transformed[:, :, 0]
                y_coords = transformed[:, :, 1]
                boxes_arr[:, 0] = x_coords.min(axis=1)
                boxes_arr[:, 1] = y_coords.min(axis=1)
                boxes_arr[:, 2] = x_coords.max(axis=1)
                boxes_arr[:, 3] = y_coords.max(axis=1)
        except ImportError:
            pass  # cv2 not installed — perspective augmentation skipped
        # NOTE: cv2 computation errors (e.g. degenerate matrix) are NOT caught here.
        # If perspective transform fails for a real reason, the exception will surface.

    # 5. Horizontal flip (p=0.5)
    if random.random() < 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        if len(boxes_arr):
            x1 = boxes_arr[:, 0].copy()
            x2 = boxes_arr[:, 2].copy()
            boxes_arr[:, 0] = target_size - x2
            boxes_arr[:, 2] = target_size - x1

    return image, boxes_arr



class YOLODroneDataset(Dataset):
    CLASS_FOLDERS = ("drone", "no_drone")

    def __init__(self, root, split, img_size=640, augment=False):
        self.img_size = img_size
        self.augment = augment
        self.samples = []  # (img_path, label_path_or_None, is_negative_folder)

        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        for cls in self.CLASS_FOLDERS:
            cls_dir = os.path.join(split_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            is_negative = (cls == "no_drone")
            for img_path, label_path in _find_images_and_labels(cls_dir):
                self.samples.append((img_path, label_path, is_negative))

        if not self.samples:
            raise RuntimeError(
                f"No images found under {split_dir}. Expected drone/ and/or "
                f"no_drone/ subfolders (optionally with an images/ subfolder)."
            )

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _load_labels(label_path, is_negative, orig_w, orig_h):
        boxes, labels = [], []
        if is_negative or label_path is None:
            return boxes, labels
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                _cls_id, xc, yc, bw, bh = parts[:5]
                xc, yc, bw, bh = float(xc), float(yc), float(bw), float(bh)
                x1 = (xc - bw / 2) * orig_w
                y1 = (yc - bh / 2) * orig_h
                x2 = (xc + bw / 2) * orig_w
                y2 = (yc + bh / 2) * orig_h
                boxes.append([x1, y1, x2, y2])
                labels.append(0)  # single class: drone
        return boxes, labels

    def __getitem__(self, idx):
        img_path, label_path, is_negative = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size

        boxes, labels = self._load_labels(label_path, is_negative, orig_w, orig_h)
        # boxes are currently in original-image pixel coords (x1,y1,x2,y2)

        if self.augment:
            image, boxes = _augment(image, boxes, self.img_size)
        else:
            image, scale, pad_x, pad_y = letterbox(image, self.img_size)
            if boxes:
                boxes = np.array(boxes, dtype=np.float32)
                boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale + pad_x
                boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale + pad_y
            else:
                boxes = np.zeros((0, 4), dtype=np.float32)

        boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4) if len(boxes) else np.zeros((0, 4), dtype=np.float32)
        # Clip to image bounds and remove degenerate boxes
        if len(boxes):
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, self.img_size)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, self.img_size)
            keep = (boxes[:, 2] - boxes[:, 0] > 1) & (boxes[:, 3] - boxes[:, 1] > 1)
            boxes = boxes[keep]
            labels = [labels[i] for i in range(len(labels)) if (i < len(keep) and keep[i])] if labels else []

        img_tensor = torch.from_numpy(
            np.array(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        )
        target = {
            "boxes": torch.from_numpy(boxes),
            "labels": torch.tensor(labels, dtype=torch.long),
            "path": img_path,
        }
        return img_tensor, target


def collate_fn(batch):
    images = torch.stack([b[0] for b in batch], dim=0)
    targets = [b[1] for b in batch]
    return images, targets
