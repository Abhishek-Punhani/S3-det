"""
Smoke test to verify S3-Det wiring, loss functions, decoding, and plotting.
Tests:
1. Paper-exact IRFA Neck forward pass.
2. Training forward + backward with NWD loss.
3. Training forward + backward with SIoU loss.
4. Model inference & detection decoding.
5. History logging and curve plotting routines.

Usage:
    python smoke_test.py
"""
import os
import torch
import shutil

from config import S3DetConfig
from model import S3Det
from plot_utils import plot_training_curves, save_history


def main():
    torch.manual_seed(42)
    print("=" * 60)
    print("Running S³-Det End-to-End Smoke Test")
    print("=" * 60)

    # 1. Test Model Architecture & Parameter Count
    cfg = S3DetConfig()
    cfg.input_size = 320
    model = S3Det(cfg)
    total_params, _ = model.count_params()
    print(f"[1/5] Model initialized: {total_params / 1e6:.2f}M parameters.")

    images = torch.rand(2, 3, cfg.input_size, cfg.input_size)
    targets = [
        {"boxes": torch.tensor([[40.0, 40.0, 90.0, 90.0], [150.0, 120.0, 180.0, 160.0]]),
         "labels": torch.tensor([0, 0])},
        {"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,), dtype=torch.long)},  # Negative (no_drone)
    ]

    # 2. Test NWD Loss Forward & Backward
    print("[2/5] Testing NWD Loss forward and backward pass...")
    cfg.loss_type = "nwd"
    model.train()
    losses_nwd = model(images, targets)
    assert torch.isfinite(losses_nwd["loss"]), "NWD loss is not finite!"
    assert "loss_nwd" in losses_nwd, "loss_nwd missing from losses dict"
    losses_nwd["loss"].backward()
    print(f"      NWD Total Loss: {losses_nwd['loss'].item():.4f} (QFL={losses_nwd['loss_qfl'].item():.4f}, DFL={losses_nwd['loss_dfl'].item():.4f}, NWD={losses_nwd['loss_nwd'].item():.4f}) -> OK")

    # 3. Test SIoU Loss Forward & Backward
    print("[3/5] Testing SIoU Loss (Paper Exact) forward and backward pass...")
    cfg.loss_type = "siou"
    model.zero_grad()
    losses_siou = model(images, targets)
    assert torch.isfinite(losses_siou["loss"]), "SIoU loss is not finite!"
    assert "loss_siou" in losses_siou, "loss_siou missing from losses dict"
    losses_siou["loss"].backward()
    print(f"      SIoU Total Loss: {losses_siou['loss'].item():.4f} (QFL={losses_siou['loss_qfl'].item():.4f}, DFL={losses_siou['loss_dfl'].item():.4f}, SIoU={losses_siou['loss_siou'].item():.4f}) -> OK")

    # 4. Test Evaluation & Decoding
    print("[4/5] Testing Inference Decoding...")
    model.eval()
    with torch.no_grad():
        preds = model(images)
    assert len(preds) == 2, f"Expected 2 predictions, got {len(preds)}"
    print(f"      Inference output: image 0: {preds[0]['boxes'].shape[0]} dets, image 1: {preds[1]['boxes'].shape[0]} dets -> OK")

    # 5. Test Plotting & Logging Utilities
    print("[5/5] Testing Plotting and Logging Utilities...")
    dummy_history = {
        "epoch": [1, 2, 3],
        "loss": [5.2, 3.8, 2.9],
        "loss_qfl": [1.1, 0.7, 0.5],
        "loss_dfl": [2.5, 2.0, 1.6],
        "loss_bbox": [1.6, 1.1, 0.8],
        "loss_type": ["nwd", "nwd", "nwd"],
        "lr": [0.0001, 0.0005, 0.001],
        "val_loss": [5.5, 4.0, 3.1],
        "val_loss_qfl": [1.2, 0.8, 0.6],
        "val_loss_dfl": [2.6, 2.1, 1.7],
        "val_loss_bbox": [1.7, 1.1, 0.8],
        "val_precision": [0.45, 0.65, 0.78],
        "val_recall": [0.38, 0.59, 0.72],
        "val_f1": [0.41, 0.62, 0.75],
        "val_ap50": [0.35, 0.58, 0.71],
    }
    test_plot_dir = "plots/smoke_test"
    test_log_dir = "logs/smoke_test"
    save_history(dummy_history, test_log_dir)
    plot_training_curves(dummy_history, test_plot_dir)

    expected_plots = [
        "total_loss.png",
        "qfl_loss.png",
        "dfl_loss.png",
        "bbox_reg_loss.png",
        "validation_metrics.png",
        "lr_schedule.png",
        "training_summary_grid.png"
    ]
    for p in expected_plots:
        plot_file = os.path.join(test_plot_dir, p)
        assert os.path.exists(plot_file), f"Expected plot {p} was not generated!"
    print(f"      All {len(expected_plots)} plots successfully generated and verified in {test_plot_dir} -> OK")

    # Clean up smoke test outputs
    if os.path.exists(test_plot_dir):
        shutil.rmtree(test_plot_dir)
    if os.path.exists(test_log_dir):
        shutil.rmtree(test_log_dir)

    print("\n" + "=" * 60)
    print("ALL SMOKE TESTS PASSED SUCCESSFULLY! The pipeline is ready.")
    print("=" * 60)


if __name__ == "__main__":
    main()
