"""
Standalone Classifier Training & Evaluation Suite for Drone Detection
====================================================================
Trains and evaluates 4 standard vision models (Drone vs No Drone):
  1. ResNet-18              (microsoft/resnet-18)
  2. MobileNet-v3-Large     (qualcomm/MobileNet-v3-Large)
  3. EfficientNet-B0        (google/efficientnet-b0)
  4. GhostNet-100           (timm/ghostnet_100.in1k)

Dataset layout expected:
  <data_root>/
    train/
      drone/     (images with drones)
      no_drone/  (images without drones)
    test/        (optional, otherwise auto 85/15 train/val split)
      drone/
      no_drone/

Outputs saved to:
  runs_classifiers/<model_name>/
    - checkpoint_best.pt
    - checkpoint_latest.pt
    - history.csv
    - history.json
    - metrics_summary.json
    - loss_accuracy_curves.png
    - confusion_matrix.png
    - roc_pr_curves.png
"""

import os
import sys
import time
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

# Model Registry
MODEL_REGISTRY = {
    "resnet18": {
        "timm_name": "resnet18",
        "display_name": "ResNet-18",
        "hf_source": "microsoft/resnet-18",
    },
    "mobilenetv3_large": {
        "timm_name": "mobilenetv3_large_100",
        "display_name": "MobileNet-v3-Large",
        "hf_source": "qualcomm/MobileNet-v3-Large",
    },
    "efficientnet_b0": {
        "timm_name": "efficientnet_b0",
        "display_name": "EfficientNet-B0",
        "hf_source": "google/efficientnet-b0",
    },
    "ghostnet_100": {
        "timm_name": "ghostnet_100",
        "display_name": "GhostNet-100",
        "hf_source": "timm/ghostnet_100.in1k",
    },
}

ALIAS_MAP = {
    "resnet18": "resnet18",
    "resnet-18": "resnet18",
    "microsoft/resnet-18": "resnet18",
    "mobilenetv3": "mobilenetv3_large",
    "mobilenet_v3": "mobilenetv3_large",
    "mobilenetv3_large": "mobilenetv3_large",
    "mobilenetv3_large_100": "mobilenetv3_large",
    "qualcomm/mobilenet-v3-large": "mobilenetv3_large",
    "efficientnet": "efficientnet_b0",
    "efficientnet_b0": "efficientnet_b0",
    "google/efficientnet-b0": "efficientnet_b0",
    "ghostnet": "ghostnet_100",
    "ghostnet_100": "ghostnet_100",
    "timm/ghostnet_100.in1k": "ghostnet_100",
}

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# ── Dataset Definition ────────────────────────────────────────────────────────
class DroneClassificationDataset(Dataset):
    """
    Loads images from drone/ (label 1) and no_drone/ (label 0).
    Ignores YOLO .txt annotation files automatically.
    """
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
        except Exception as e:
            # Fallback for rare corrupt images: return black image
            img = Image.new("RGB", (224, 224), (0, 0, 0))

        if self.transform is not None:
            img = self.transform(img)

        return img, label


def scan_split_folder(split_dir: str) -> List[Tuple[str, int]]:
    """Scan split_dir for drone/ and no_drone/ subfolders."""
    samples = []
    class_map = {"no_drone": 0, "drone": 1}

    for class_name, label in class_map.items():
        folder = os.path.join(split_dir, class_name)
        if not os.path.isdir(folder):
            # Also check images/ subfolder if nested
            alt_folder = os.path.join(folder, "images")
            if os.path.isdir(alt_folder):
                folder = alt_folder
            else:
                continue

        try:
            files = os.listdir(folder)
        except OSError:
            continue

        for f in files:
            if f.lower().endswith(IMG_EXTS):
                samples.append((os.path.join(folder, f), label))

    return samples


def get_dataset_splits(data_root: str, val_ratio: float = 0.15, seed: int = 42):
    """
    Returns (train_samples, val_samples).
    If test/ exists, uses train/ for training and test/ for validation.
    Otherwise splits train/ into train and val.
    """
    train_dir = os.path.join(data_root, "train")
    test_dir = os.path.join(data_root, "test")

    if not os.path.isdir(train_dir):
        # Maybe data_root is already train dir
        if os.path.isdir(os.path.join(data_root, "drone")):
            train_dir = data_root
        else:
            raise FileNotFoundError(f"Could not find train/ folder in {data_root}")

    train_samples = scan_split_folder(train_dir)
    if not train_samples:
        raise RuntimeError(f"No valid images found in {train_dir} under drone/ or no_drone/!")

    if os.path.isdir(test_dir):
        val_samples = scan_split_folder(test_dir)
        if len(val_samples) > 0:
            print(f"[Dataset] Using {train_dir} ({len(train_samples)} imgs) and separate test/ ({len(val_samples)} imgs)")
            return train_samples, val_samples

    # Auto split train_samples
    print(f"[Dataset] No test/ folder found. Splitting {len(train_samples)} images into {(1-val_ratio)*100:.0f}% train / {val_ratio*100:.0f}% val (seed={seed})")
    rng = np.random.RandomState(seed)
    indices = np.arange(len(train_samples))
    rng.shuffle(indices)

    val_count = int(len(train_samples) * val_ratio)
    val_idx = indices[:val_count]
    train_idx = indices[val_count:]

    train_split = [train_samples[i] for i in train_idx]
    val_split = [train_samples[i] for i in val_idx]

    return train_split, val_split


# ── Model Factory ─────────────────────────────────────────────────────────────
def build_classifier(model_key: str, num_classes: int = 2, pretrained: bool = True):
    """Creates classifier using timm with 2 output logits."""
    try:
        import timm
    except ImportError:
        raise ImportError("timm is required. Run: pip install timm")

    timm_name = MODEL_REGISTRY[model_key]["timm_name"]
    print(f"[Model] Creating '{timm_name}' (pretrained={pretrained}, num_classes={num_classes})...")
    model = timm.create_model(timm_name, pretrained=pretrained, num_classes=num_classes)
    return model


def count_parameters(model: nn.Module) -> Tuple[float, float]:
    """Returns (total_params_M, trainable_params_M)."""
    total = sum(p.numel() for p in model.parameters()) / 1e6
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    return total, trainable


# ── Metrics & Evaluation ──────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp=True):
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_preds = []
    all_probs = []

    device_type = "cuda" if "cuda" in str(device) else "cpu"

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        with torch.amp.autocast(device_type, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, targets)

        total_loss += loss.item() * len(targets)
        probs = torch.softmax(outputs, dim=-1)
        preds = torch.argmax(probs, dim=-1)

        all_targets.extend(targets.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
        all_probs.extend(probs[:, 1].cpu().numpy().tolist())

    n = max(len(all_targets), 1)
    mean_loss = total_loss / n

    targets_np = np.array(all_targets)
    preds_np = np.array(all_preds)
    probs_np = np.array(all_probs)

    # Compute classification metrics
    tp = np.sum((preds_np == 1) & (targets_np == 1))
    fp = np.sum((preds_np == 1) & (targets_np == 0))
    fn = np.sum((preds_np == 0) & (targets_np == 1))
    tn = np.sum((preds_np == 0) & (targets_np == 0))

    accuracy = float((tp + tn) / n)
    precision = float(tp / (tp + fp + 1e-9))
    recall = float(tp / (tp + fn + 1e-9))
    f1 = float(2 * precision * recall / (precision + recall + 1e-9))

    # ROC AUC
    auc = compute_auc(targets_np, probs_np)

    metrics = {
        "loss": mean_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp),
            "fn": int(fn), "tp": int(tp),
        },
        "all_targets": targets_np,
        "all_probs": probs_np,
        "all_preds": preds_np,
    }
    return metrics


def compute_auc(y_true, y_score):
    """Calculates ROC AUC without requiring sklearn if not installed."""
    try:
        from sklearn.metrics import roc_auc_score
        if len(np.unique(y_true)) > 1:
            return float(roc_auc_score(y_true, y_score))
        return 0.5
    except ImportError:
        pass

    # Pure NumPy fallback for ROC AUC (trapezoid rule)
    if len(np.unique(y_true)) <= 1:
        return 0.5
    desc_score_indices = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[desc_score_indices]
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    tpr = np.cumsum(y_true_sorted == 1) / n_pos
    fpr = np.cumsum(y_true_sorted == 0) / n_neg
    return float(np.trapz(tpr, fpr))


# ── Plotting Utilities ────────────────────────────────────────────────────────
def save_plots(history: Dict, eval_metrics: Dict, out_dir: str, display_name: str):
    os.makedirs(out_dir, exist_ok=True)
    epochs = history.get("epoch", [])
    if not epochs:
        return

    # 1. Training & Validation Curves (Loss, Accuracy, F1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", color="#1f77b4", lw=2)
    axes[0].plot(epochs, history["val_loss"], label="Val Loss", color="#d62728", lw=2, ls="--")
    axes[0].set_title(f"{display_name} - CrossEntropy Loss", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, ls="--", alpha=0.5)
    axes[0].legend()

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], label="Train Acc", color="#2ca02c", lw=2)
    axes[1].plot(epochs, history["val_acc"], label="Val Acc", color="#9467bd", lw=2, ls="--")
    axes[1].set_title(f"{display_name} - Accuracy", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].grid(True, ls="--", alpha=0.5)
    axes[1].legend()

    # F1 & AUC
    axes[2].plot(epochs, history["val_f1"], label="Val F1", color="#ff7f0e", lw=2)
    axes[2].plot(epochs, history["val_auc"], label="Val ROC-AUC", color="#8c564b", lw=2, ls=":")
    axes[2].set_title(f"{display_name} - F1 & ROC-AUC", fontweight="bold")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Score")
    axes[2].set_ylim(0.0, 1.02)
    axes[2].grid(True, ls="--", alpha=0.5)
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "loss_accuracy_curves.png"), dpi=200)
    plt.close(fig)

    # 2. Confusion Matrix Plot
    cm = eval_metrics["confusion_matrix"]
    matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    matrix_norm = matrix.astype("float") / (matrix.sum(axis=1)[:, np.newaxis] + 1e-9)

    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(matrix_norm, cmap=plt.cm.Blues, vmin=0, vmax=1)
    fig.colorbar(cax)

    classes = ["No Drone (0)", "Drone (1)"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted Label", fontweight="bold")
    ax.set_ylabel("True Label", fontweight="bold")
    ax.set_title(f"Confusion Matrix: {display_name}", fontweight="bold", pad=15)

    for i in range(2):
        for j in range(2):
            count = matrix[i, j]
            pct = matrix_norm[i, j] * 100
            text_color = "white" if matrix_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{count}\n({pct:.1f}%)", ha="center", va="center", color=text_color, fontweight="bold")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=200)
    plt.close(fig)

    # 3. ROC & PR Curves
    try:
        from sklearn.metrics import roc_curve, precision_recall_curve
        y_true = eval_metrics["all_targets"]
        y_probs = eval_metrics["all_probs"]

        fpr, tpr, _ = roc_curve(y_true, y_probs)
        precision, recall, _ = precision_recall_curve(y_true, y_probs)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"AUC = {eval_metrics['auc']:.4f}")
        ax1.plot([0, 1], [0, 1], color="gray", ls="--")
        ax1.set_title(f"{display_name} - ROC Curve", fontweight="bold")
        ax1.set_xlabel("False Positive Rate")
        ax1.set_ylabel("True Positive Rate")
        ax1.grid(True, ls="--", alpha=0.5)
        ax1.legend(loc="lower right")

        ax2.plot(recall, precision, color="#2ca02c", lw=2, label=f"Best F1 = {eval_metrics['f1']:.4f}")
        ax2.set_title(f"{display_name} - Precision-Recall Curve", fontweight="bold")
        ax2.set_xlabel("Recall")
        ax2.set_ylabel("Precision")
        ax2.grid(True, ls="--", alpha=0.5)
        ax2.legend(loc="lower left")

        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "roc_pr_curves.png"), dpi=200)
        plt.close(fig)
    except Exception as e:
        print(f"[Warning] Could not generate ROC/PR curve plots: {e}")


# ── Training Routine ──────────────────────────────────────────────────────────
def train_model(
    model_name: str,
    data_root: str = "dataset_full",
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 3e-4,
    img_size: int = 224,
    num_workers: int = 4,
    patience: int = 10,
    device_str: str = "cuda",
    out_root: str = "runs_classifiers",
):
    model_key = ALIAS_MAP.get(model_name.lower().strip(), model_name.lower().strip())
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(MODEL_REGISTRY.keys())}")

    info = MODEL_REGISTRY[model_key]
    display_name = info["display_name"]
    out_dir = os.path.join(out_root, model_key)
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device(device_str if (torch.cuda.is_available() and "cuda" in device_str) else "cpu")
    use_amp = (device.type == "cuda")

    print("\n" + "═" * 78)
    print(f"  Training Classifier: {display_name} ({info['hf_source']})")
    print(f"  Device: {device} | AMP: {use_amp} | Batch: {batch_size} | Epochs: {epochs} | Img: {img_size}px")
    print(f"  Output directory: {out_dir}")
    print("═" * 78)

    # 1. Datasets & Transforms
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_samples, val_samples = get_dataset_splits(data_root)
    train_dataset = DroneClassificationDataset(train_samples, transform=train_transform)
    val_dataset = DroneClassificationDataset(val_samples, transform=val_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device.type == "cuda"), drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda")
    )

    n_train_pos = sum(1 for _, lbl in train_samples if lbl == 1)
    n_train_neg = len(train_samples) - n_train_pos
    print(f"  Train split: {len(train_samples)} images ({n_train_pos} drone, {n_train_neg} no_drone)")
    print(f"  Val split  : {len(val_samples)} images")

    # 2. Build Model
    model = build_classifier(model_key, num_classes=2, pretrained=True).to(device)
    total_m, train_m = count_parameters(model)
    print(f"  Model Parameters: Total = {total_m:.2f}M | Trainable = {train_m:.2f}M")

    # Loss function with class weighting if imbalanced
    weight_tensor = None
    if n_train_pos > 0 and n_train_neg > 0:
        w_neg = 1.0
        w_pos = n_train_neg / max(n_train_pos, 1)
        weight_tensor = torch.tensor([w_neg, min(w_pos, 5.0)], dtype=torch.float32, device=device)
        print(f"  Loss Class Weights: no_drone=1.0, drone={weight_tensor[1].item():.2f}")

    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = {
        "epoch": [], "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [], "val_precision": [],
        "val_recall": [], "val_f1": [], "val_auc": [], "lr": []
    }

    best_f1 = -1.0
    best_metrics = {}
    epochs_no_improve = 0

    # 3. Epoch Loop
    print("\n" + "-" * 78)
    print(f"{'Epoch':^7} | {'Train Loss':^10} | {'Train Acc':^9} | {'Val Loss':^9} | {'Val Acc':^8} | {'Val F1':^7} | {'Val AUC':^7} | {'Time':^6}")
    print("-" * 78)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss_acc = 0.0
        correct = 0
        total = 0

        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast(device.type, enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss_acc += loss.item() * len(targets)
            preds = torch.argmax(outputs, dim=-1)
            correct += (preds == targets).sum().item()
            total += len(targets)

        scheduler.step()

        train_loss = train_loss_acc / max(total, 1)
        train_acc = correct / max(total, 1)

        # Validation
        val_res = evaluate(model, val_loader, criterion, device, use_amp=use_amp)
        elapsed = time.time() - t0
        curr_lr = optimizer.param_groups[0]["lr"]

        history["epoch"].append(epoch)
        history["train_loss"].append(round(train_loss, 4))
        history["train_acc"].append(round(train_acc, 4))
        history["val_loss"].append(round(val_res["loss"], 4))
        history["val_acc"].append(round(val_res["accuracy"], 4))
        history["val_precision"].append(round(val_res["precision"], 4))
        history["val_recall"].append(round(val_res["recall"], 4))
        history["val_f1"].append(round(val_res["f1"], 4))
        history["val_auc"].append(round(val_res["auc"], 4))
        history["lr"].append(curr_lr)

        star = ""
        if val_res["f1"] > best_f1:
            best_f1 = val_res["f1"]
            best_metrics = val_res.copy()
            epochs_no_improve = 0
            star = " ★ [BEST]"
            # Save best checkpoint
            torch.save({
                "model_name": model_key,
                "epoch": epoch,
                "model_state": model.state_dict(),
                "metrics": {k: v for k, v in val_res.items() if not isinstance(v, np.ndarray)},
            }, os.path.join(out_dir, "checkpoint_best.pt"))
        else:
            epochs_no_improve += 1

        print(f"[{epoch:03d}/{epochs:03d}] | {train_loss:10.4f} | {train_acc*100:8.2f}% | {val_res['loss']:9.4f} | {val_res['accuracy']*100:7.2f}% | {val_res['f1']:7.4f} | {val_res['auc']:7.4f} | {elapsed:5.1f}s{star}")

        # Save latest checkpoint & history after every epoch
        torch.save({
            "model_name": model_key,
            "epoch": epoch,
            "model_state": model.state_dict(),
        }, os.path.join(out_dir, "checkpoint_latest.pt"))

        # Save history json & csv
        with open(os.path.join(out_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

        with open(os.path.join(out_dir, "history.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(list(history.keys()))
            for row_i in range(len(history["epoch"])):
                writer.writerow([history[k][row_i] for k in history.keys()])

        # Update per-epoch curves
        save_plots(history, val_res, out_dir, display_name)

        # Early stopping check
        if epochs_no_improve >= patience:
            print(f"\n[Early Stopping] No improvement in F1 for {patience} epochs. Stopping early at epoch {epoch}.")
            break

    # Save final best summary
    summary = {
        "model_key": model_key,
        "display_name": display_name,
        "hf_source": info["hf_source"],
        "total_params_M": round(total_m, 2),
        "best_epoch": int(best_metrics.get("epoch", epoch)),
        "best_accuracy": round(best_metrics.get("accuracy", 0.0), 4),
        "best_precision": round(best_metrics.get("precision", 0.0), 4),
        "best_recall": round(best_metrics.get("recall", 0.0), 4),
        "best_f1": round(best_metrics.get("f1", 0.0), 4),
        "best_auc": round(best_metrics.get("auc", 0.0), 4),
        "confusion_matrix": best_metrics.get("confusion_matrix", {}),
    }
    with open(os.path.join(out_dir, "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 78)
    print(f"  Finished {display_name}! Best F1: {best_f1:.4f} | Best Acc: {summary['best_accuracy']*100:.2f}% | Best AUC: {summary['best_auc']:.4f}")
    print(f"  Artifacts saved in: {out_dir}")
    print("=" * 78 + "\n")
    return summary


# ── Command Line Interface ────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train image classifier on Drone vs No Drone")
    parser.add_argument("--model", type=str, default="resnet18", help=f"Model: {list(MODEL_REGISTRY.keys())} or 'all'")
    parser.add_argument("--data-root", type=str, default="dataset_full", help="Folder containing train/ and optional test/")
    parser.add_argument("--epochs", type=int, default=50, help="Max epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--img-size", type=int, default=224, help="Input image size")
    parser.add_argument("--workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--out-dir", type=str, default="runs_classifiers", help="Output directory")
    args = parser.parse_args()

    train_model(
        model_name=args.model,
        data_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        img_size=args.img_size,
        num_workers=args.workers,
        patience=args.patience,
        device_str=args.device,
        out_root=args.out_dir,
    )


if __name__ == "__main__":
    main()
