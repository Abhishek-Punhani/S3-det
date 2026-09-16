"""
Train S3-Det on UAV drone / no_drone dataset.
Supports both paper-exact SIoU and NWD bounding box losses,
per-epoch checkpointing, validation metrics tracking (Precision, Recall, F1, AP50),
and automated plotting of all loss and accuracy curves at each epoch.

Usage:
    python train.py --config data_config.yaml
    python train.py --data-root /path/to/dataset --loss nwd --epochs 100 --batch-size 16
"""
import argparse
import math
import os
import time
import json
from dataclasses import asdict
import torch
from torch.utils.data import DataLoader

from config import S3DetConfig
from model import S3Det
from dataset import YOLODroneDataset, collate_fn
from eval_engine import evaluate_epoch
from plot_utils import plot_training_curves, plot_val_metrics_epoch, save_history


def lr_scale_at(step, warmup_iters, total_iters, min_lr, base_lr):
    """Linear warmup followed by cosine annealing decay."""
    if step < warmup_iters:
        return max(step, 1) / max(1, warmup_iters)
    progress = (step - warmup_iters) / max(1, total_iters - warmup_iters)
    cosine = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    return (min_lr + (base_lr - min_lr) * cosine) / base_lr


def effective_nwd_constant(cfg):
    """
    Return the NWD normalization constant scaled to the current input resolution.
    
    cfg.nwd_constant is defined at 640×640. If nwd_normalize_to_input=True,
    scale it proportionally so that the same physical-pixel NWD threshold
    corresponds to the same loss value regardless of input size.
    
    Example:
        640px: C = 12.8   →  boxes within ~12.8px std are NWD≈e^(-1)≈0.37
        416px: C = 12.8 * (416/640) = 8.32   →  same relative coverage
    """
    if getattr(cfg, "nwd_normalize_to_input", True):
        return cfg.nwd_constant * (cfg.input_size / 640.0)
    return cfg.nwd_constant


class ModelEMA:
    """
    Exponential Moving Average of model weights.

    Shadow weights = decay * shadow + (1 - decay) * live_weights
    Evaluated on EMA weights for more stable final metrics.

    Usage:
        ema = ModelEMA(model, decay=0.9998, warmup_iters=100)
        # inside training loop, after optimizer.step():
        ema.update(model, step)
        # for validation:
        with ema.average_parameters():
            run_eval(model)
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.9998,
                 warmup_iters: int = 100):
        import copy
        self.shadow = copy.deepcopy(model).eval()
        # Freeze shadow — only updated manually
        for p in self.shadow.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.warmup_iters = warmup_iters
        self.num_updates = 0

    @torch.no_grad()
    def update(self, model: torch.nn.Module, step: int):
        """Update shadow weights. Uses lower effective decay during warmup."""
        if step < self.warmup_iters:
            # Linear warmup: effective decay = step / warmup_iters * target_decay
            d = self.decay * (step / max(1, self.warmup_iters))
        else:
            d = self.decay
        self.num_updates += 1
        for s_param, m_param in zip(self.shadow.parameters(), model.parameters()):
            s_param.data.mul_(d).add_(m_param.data.float() * (1.0 - d))
        # Also copy buffers (BN running stats etc.)
        for s_buf, m_buf in zip(self.shadow.buffers(), model.buffers()):
            s_buf.data.copy_(m_buf.data)

    from contextlib import contextmanager

    @contextmanager
    def average_parameters(self, model: torch.nn.Module):
        """
        Context manager: temporarily swap model weights with EMA shadow.
        Restores original weights on exit.
        """
        import copy
        orig_state = copy.deepcopy(model.state_dict())
        model.load_state_dict(self.shadow.state_dict())
        try:
            yield model
        finally:
            model.load_state_dict(orig_state)



@torch.no_grad()
def _val_losses(model, loader, device):
    """Compute average validation losses only (no detection matching)."""
    model.eval()
    running = {}
    for images, targets in loader:
        images = images.to(device)
        losses = model(images, targets)
        for k, v in losses.items():
            # Skip non-scalar metadata and diagnostic counters
            if k in ("num_pos", "num_gts", "fallback_count", "fallback_rate", "loss_type"):
                continue
            scalar = float(v.detach()) if hasattr(v, "detach") else float(v)
            running[k] = running.get(k, 0.0) + scalar
    n = max(len(loader), 1)
    return {f"val_{k}": v / n for k, v in running.items()}



def main():
    parser = argparse.ArgumentParser(description="Train S3-Det for UAV Drone Detection")
    parser.add_argument("--config", default="data_config.yaml", help="Path to config YAML")
    parser.add_argument("--data-root", default=None, help="Folder containing train/ and test/")
    parser.add_argument("--loss", default=None, choices=["nwd", "siou"], help="Loss type: nwd or siou")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=None, help="Checkpoints directory")
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume from")
    args = parser.parse_args()

    # Load configuration from YAML or defaults
    if os.path.exists(args.config):
        print(f"Loading configuration from: {args.config}")
        cfg = S3DetConfig.from_yaml(args.config)
    else:
        cfg = S3DetConfig()

    # CLI overrides
    if args.data_root:
        cfg.data_root = args.data_root
    if args.loss:
        cfg.loss_type = args.loss
    if args.epochs:
        cfg.epochs = args.epochs
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.lr:
        cfg.lr = args.lr
    if args.img_size:
        cfg.input_size = args.img_size
    if args.workers is not None:
        cfg.workers = args.workers
    if args.device:
        cfg.device = args.device
    if args.out:
        cfg.checkpoint_dir = args.out

    device_str = cfg.device if (torch.cuda.is_available() and "cuda" in cfg.device) else "cpu"
    device = torch.device(device_str)

    # Create directories
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.plot_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)

    # Initialize Dataset and DataLoader
    print(f"Loading training dataset from: {cfg.data_root}/{cfg.train_split}")
    full_train_set = YOLODroneDataset(cfg.data_root, cfg.train_split, img_size=cfg.input_size, augment=True)

    # ── Train / Val split ─────────────────────────────────────────────────
    # Carve val_fraction from the TRAINING data.
    # TEST split is kept strictly untouched during all training.
    val_loader = None
    val_fraction = getattr(cfg, "val_fraction", 0.15)
    val_split_seed = getattr(cfg, "val_split_seed", 42)
    if val_fraction > 0 and len(full_train_set) > 1:
        n_val   = max(1, int(len(full_train_set) * val_fraction))
        n_train = len(full_train_set) - n_val
        gen = torch.Generator().manual_seed(val_split_seed)
        train_set, val_set = torch.utils.data.random_split(
            full_train_set, [n_train, n_val], generator=gen
        )
        # Val set should not augment — wrap with augment=False by re-creating
        # (random_split gives a Subset; we disable augment via the underlying dataset)
        full_train_set.augment = True   # training portion: augment ON
        val_set_no_aug = YOLODroneDataset(cfg.data_root, cfg.train_split,
                                          img_size=cfg.input_size, augment=False)
        _, val_set_no_aug_split = torch.utils.data.random_split(
            val_set_no_aug, [n_train, n_val], generator=torch.Generator().manual_seed(val_split_seed)
        )
        print(f"Train/Val split (seed={val_split_seed}): "
              f"{n_train} train | {n_val} val  ({val_fraction:.0%} of training data)")
        print(f"[NOTE] Test split ({cfg.test_split}) is UNTOUCHED — held for final evaluation only.")
        val_loader = DataLoader(val_set_no_aug_split, batch_size=cfg.batch_size, shuffle=False,
                                num_workers=cfg.workers, collate_fn=collate_fn)
    else:
        train_set = full_train_set
        print("[WARN] val_fraction=0: training without validation. "
              "Test split will not be evaluated during training.")

    train_loader = DataLoader(
        train_set, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.workers, collate_fn=collate_fn,
        drop_last=(len(train_set) > cfg.batch_size)
    )
    print(f"Train samples: {len(train_set)} | Batches per epoch: {len(train_loader)}")

    # Initialize Model
    model = S3Det(cfg).to(device)
    eff = model.model_summary(input_size=cfg.input_size)
    total_p, _ = model.count_params()
    print(f"  Loss Function: QFL (β={cfg.qfl_beta}) + DFL + {cfg.loss_type.upper()}")
    print(f"  EMA={cfg.ema}  AMP={cfg.amp and torch.cuda.is_available()}  NWD-norm={cfg.nwd_normalize_to_input}")
    print(f"  Device: {device_str}")


    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    start_epoch = 0
    # ---- extended history dict — all per-epoch scalars ----
    history = {
        "epoch": [],
        # train losses
        "loss": [], "loss_qfl": [], "loss_dfl": [], "loss_bbox": [], "loss_type": [], "lr": [],
        # assignment diagnostics (fallback rate is KEY for tiny-object research)
        "fallback_rate": [],    # fraction of GT boxes needing nearest-point fallback
        "num_pos": [],          # avg positive points per batch
        # val losses
        "val_loss": [], "val_loss_qfl": [], "val_loss_dfl": [], "val_loss_bbox": [],
        # core detection metrics
        "val_precision": [], "val_recall": [], "val_f1": [],
        "val_ap50": [], "val_ap50_95": [],
        # deployment
        "val_fppi": [],
        # localization quality
        "val_center_err": [], "val_center_err_norm": [],
        "val_tp_iou": [], "val_tp_nwd": [],
        # per-size bin recall (bins: <=8, 8-16, 16-32, 32-64, >64)
        "val_recall_bin0": [], "val_recall_bin1": [], "val_recall_bin2": [],
        "val_recall_bin3": [], "val_recall_bin4": [],
        # per-size bin AP50
        "val_ap50_bin0": [], "val_ap50_bin1": [], "val_ap50_bin2": [],
        "val_ap50_bin3": [], "val_ap50_bin4": [],
    }


    if args.resume and os.path.isfile(args.resume):
        print(f"Resuming checkpoint: {args.resume}")
        try:
            ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0)
        history_path = os.path.join(cfg.log_dir, "history.json")
        if os.path.exists(history_path):
            with open(history_path, "r") as f:
                history = json.load(f)
        print(f"Resumed from epoch {start_epoch}")

    # ── EMA setup ────────────────────────────────────────────────────────────
    ema = None
    if cfg.ema:
        ema = ModelEMA(model, decay=cfg.ema_decay, warmup_iters=cfg.ema_warmup_iters)
        print(f"EMA enabled  (decay={cfg.ema_decay}, warmup={cfg.ema_warmup_iters} steps)")

    # ── AMP setup (CUDA-only; no-op on CPU) ──────────────────────────────────
    use_amp = cfg.amp and torch.cuda.is_available()
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    if use_amp:
        print(f"AMP enabled (torch.amp)")

    # ── Resolved NWD constant (normalized to input resolution) ───────────────
    nwd_C = effective_nwd_constant(cfg)
    if cfg.nwd_normalize_to_input and cfg.loss_type == "nwd":
        print(f"NWD constant: {cfg.nwd_constant} (defined at 640px) "
              f"→ {nwd_C:.3f} (scaled for {cfg.input_size}px input)")
    # Propagate to model's loss function so training and eval use the same C
    model.loss_fn.nwd_constant = nwd_C


    total_iters = cfg.epochs * max(len(train_loader), 1)
    step = start_epoch * max(len(train_loader), 1)
    best_f1 = -1.0

    print("\n" + "="*80)
    print(f"Starting Training for {cfg.epochs - start_epoch} epochs (Epoch {start_epoch+1} -> {cfg.epochs})")
    print("="*80)

    for epoch in range(start_epoch, cfg.epochs):
        epoch_idx = epoch + 1
        model.train()
        t0 = time.time()
        running = {}

        for images, targets in train_loader:
            images = images.to(device)

            # Cosine LR warmup and decay
            scale = lr_scale_at(step, cfg.warmup_iters, total_iters, cfg.min_lr, cfg.lr)
            for g in optimizer.param_groups:
                g["lr"] = cfg.lr * scale

            # ── AMP forward + backward ────────────────────────────────────
            optimizer.zero_grad()
            device_type = "cuda" if torch.cuda.is_available() else "cpu"
            with torch.amp.autocast(device_type, enabled=use_amp):
                losses = model(images, targets)

            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           max_norm=cfg.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            # ── EMA update ────────────────────────────────────────────────
            if ema is not None:
                ema.update(model, step)

            for k, v in losses.items():
                if k in ("loss_type",):
                    continue
                if k in ("fallback_count", "num_gts"):
                    # accumulate raw counts for rate calculation
                    running[k] = running.get(k, 0) + int(v) if isinstance(v, (int, float)) else running.get(k, 0) + int(v.item())
                elif k == "fallback_rate":
                    # accumulate for averaging (raw rate per batch)
                    running["fallback_rate_sum"] = running.get("fallback_rate_sum", 0.0) + float(v)
                elif k == "num_pos":
                    running["num_pos_sum"] = running.get("num_pos_sum", 0) + int(v)
                else:
                    running[k] = running.get(k, 0.0) + float(v.detach() if hasattr(v, "detach") else v)
            step += 1

        n_train = max(len(train_loader), 1)
        epoch_loss  = running.get("loss",      0.0) / n_train
        epoch_qfl   = running.get("loss_qfl",  0.0) / n_train
        epoch_dfl   = running.get("loss_dfl",  0.0) / n_train
        epoch_bbox  = running.get("loss_bbox", 0.0) / n_train
        current_lr  = optimizer.param_groups[0]["lr"]
        # fallback_rate: fraction of GT boxes that needed nearest-point assignment
        epoch_fb_rate  = running.get("fallback_rate_sum", 0.0) / n_train
        epoch_num_pos  = running.get("num_pos_sum", 0)    # total positives this epoch

        history["epoch"].append(epoch_idx)
        history["loss"].append(round(epoch_loss, 4))
        history["loss_qfl"].append(round(epoch_qfl, 4))
        history["loss_dfl"].append(round(epoch_dfl, 4))
        history["loss_bbox"].append(round(epoch_bbox, 4))
        history["loss_type"].append(cfg.loss_type)
        history["lr"].append(current_lr)
        history["fallback_rate"].append(round(epoch_fb_rate, 4))
        history["num_pos"].append(epoch_num_pos)

        # ── Validation at epoch end ─────────────────────────────────────────
        val_str = ""
        all_val_keys = [k for k in history if k.startswith("val_")]
        if val_loader is not None and cfg.eval_every_epoch:
            # Run validation on EMA weights if available (more stable estimates)
            eval_model = ema.shadow if (ema is not None) else model
            # 1. Val losses (always on live model — EMA shadow has no loss_fn)
            val_losses = _val_losses(model, val_loader, device)
            # 2. Full detection metrics on EMA shadow or live model
            val_det = evaluate_epoch(
                eval_model, val_loader, device,
                score_thresh=cfg.score_thresh,
                nwd_constant=nwd_C,
                size_bins=list(cfg.tiny_size_bins),
                ap_iou_thresholds=list(cfg.ap_iou_thresholds),
            )
            val_res = {**val_losses, **val_det}

            for k in all_val_keys:
                v = val_res.get(k, 0.0)
                history[k].append(round(float(v), 4) if v is not None else None)

            ema_tag = " [EMA]" if ema is not None else ""
            val_str = (
                f" |{ema_tag} Loss={val_res.get('val_loss', 0):.4f} "
                f"P={val_res.get('val_precision', 0):.3f} "
                f"R={val_res.get('val_recall', 0):.3f} "
                f"F1={val_res.get('val_f1', 0):.3f} "
                f"AP50={val_res.get('val_ap50', 0):.3f} "
                f"AP50:95={val_res.get('val_ap50_95', 0):.3f} "
                f"FPPI={val_res.get('val_fppi', 0):.2f} "
                f"TinyR={val_res.get('val_recall_bin0', 0):.3f}"
            )
            # Best checkpoint by F1
            if val_res.get("val_f1", 0.0) > best_f1:
                best_f1 = val_res["val_f1"]
                best_path = os.path.join(cfg.checkpoint_dir, "s3det_best.pt")
                torch.save({
                    "model":     model.state_dict(),
                    "ema":       ema.shadow.state_dict() if ema else None,
                    "optimizer": optimizer.state_dict(),
                    "epoch":     epoch_idx,
                    "cfg":       asdict(cfg),
                    "metrics":   val_res,
                    "nwd_C":     nwd_C,
                }, best_path)
                val_str += " ★[NEW BEST]"
        else:
            for k in all_val_keys:
                history[k].append(None)

        epoch_time = time.time() - t0
        print(f"Epoch [{epoch_idx:03d}/{cfg.epochs:03d}] "
              f"Loss={epoch_loss:.4f} (QFL={epoch_qfl:.4f}, DFL={epoch_dfl:.4f}, {cfg.loss_type.upper()}={epoch_bbox:.4f}) "
              f"LR={current_lr:.2e} FB={epoch_fb_rate:.1%} Pos={epoch_num_pos} "
              f"Time={epoch_time:.1f}s{val_str}")

        # ------------------------------------------------------------------
        # 1. Save checkpoint AT EVERY EPOCH
        # ------------------------------------------------------------------
        if cfg.save_every_epoch:
            ckpt_epoch_path = os.path.join(cfg.checkpoint_dir, f"s3det_epoch_{epoch_idx:03d}.pt")
            latest_path = os.path.join(cfg.checkpoint_dir, "s3det_latest.pt")
            save_payload = {
                "model":     model.state_dict(),
                "ema":       ema.shadow.state_dict() if ema else None,
                "optimizer": optimizer.state_dict(),
                "epoch":     epoch_idx,
                "cfg":       asdict(cfg),
                "nwd_C":     nwd_C,
            }
            torch.save(save_payload, ckpt_epoch_path)
            torch.save(save_payload, latest_path)

        # ------------------------------------------------------------------
        # 2. Save history logs & generate all plots AT EVERY EPOCH
        # ------------------------------------------------------------------
        save_history(history, cfg.log_dir)
        curves_dir = os.path.join(cfg.plot_dir, "curves")
        plot_training_curves(history, curves_dir)
        plot_val_metrics_epoch(history, curves_dir)   # AP50:95, FPPI, size bins, center err

    print("\n" + "="*80)
    print(f"Training Complete! Checkpoints stored in '{cfg.checkpoint_dir}', all curves plotted in '{cfg.plot_dir}/curves'.")
    print("="*80)


if __name__ == "__main__":
    main()
