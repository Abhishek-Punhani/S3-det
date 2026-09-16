# S³-Det: Spectro-Spatial Synergistic Network for UAV Drone Detection

A complete, verified reproduction of the **S³-Det** architecture from:
> *"S³-Det: Efficient UAV object detection using spectro-spatial synergistic learning and implicit recursive refinement"*, Nature Scientific Reports, 2024 (s41598-026-53405-7).

Includes support for **both** the paper's original **SIoU** loss (Version A: Paper Baseline) and **NWD** (Normalized Wasserstein Distance, Version B: Enhanced Tiny Object Detection), configurable via a single YAML file or CLI flag.

---

## 📁 Repository Structure

| File | Description |
|---|---|
| `data_config.yaml` | **Central YAML configuration** for dataset paths, hyperparameters, and loss choices |
| `config.py` | Dataclass configuration loader supporting YAML load/save |
| `modules.py` | Building blocks: **SFG** (Spectral Frequency Gating), **GSA** (Granular Split Attention), **SSB**, and **IRU** |
| `backbone.py` | **S³Net** backbone (Stem + 3 SSB stages at strides 4, 8, 16) |
| `neck.py` | **IRFA** neck (lateral convs + paper-exact recursive fusion via shared IRU, Eqs. 13–14) |
| `head.py` | **LCR-Head** (3×3 DWConv classification branch, 5×5 DWConv regression branch) |
| `losses.py` | Quality Focal Loss (**QFL** β=2.0), Distribution Focal Loss (**DFL**), **SIoU** (Eqs. 19–21), and **NWD** |
| `dataset.py` | Dataset loader for `drone/` (positives with YOLO txt boxes) and `no_drone/` (negative backgrounds) |
| `plot_utils.py` | Publication-ready curve generation for all losses and validation metrics |
| `visualize_dataset.py` | Dataset inspection: bounding box EDA, aspect ratio distributions, spatial heatmap, sample overlays |
| `train.py` | Training loop with **per-epoch checkpoints**, validation tracking, and live curve plotting |
| `test.py` | Test evaluation with PR curve, AP@0.5, and bounding box prediction overlays |
| `smoke_test.py` | Fast end-to-end sanity test on synthetic data |
| `make_sample_dataset.py` | Generates a miniature mock dataset to test pipelines instantly |

---

## 🚀 Quick Start Guide

### 1. Run the Smoke Test
Verify that the entire model, forward/backward passes, NWD & SIoU losses, decoding, and plotting work:
```bash
python smoke_test.py
```

### 2. Configure Your Dataset
Open `data_config.yaml` and set your dataset root path:
```yaml
data_root: "/path/to/your/dataset"
train_split: "train"
test_split: "test"
loss_type: "nwd"     # Choose "nwd" or "siou"
epochs: 100
batch_size: 16
```

Your dataset folder structure:
```
<data_root>/
  train/
    drone/      # Images + YOLO .txt annotations (class 0 = drone)
    no_drone/   # Background negative images (no drones)
  test/
    drone/
    no_drone/
```

### 3. Visualize & Inspect Your Dataset
Generate sample images with ground truth boxes and dataset distribution statistics:
```bash
python visualize_dataset.py --config data_config.yaml
```
Output plots will be saved to `plots/dataset/`:
- `dataset_samples.png`: Grid of positive and negative samples with drone boxes labeled.
- `dataset_overview.png`: Object scale distributions, bounding box aspect ratios, and spatial heatmaps.

### 4. Train the Model
Train S³-Det with automatic **per-epoch checkpointing** and dynamic curve plotting:
```bash
# Using data_config.yaml:
python train.py --config data_config.yaml

# Or with CLI flags:
python train.py --data-root /path/to/dataset --loss nwd --epochs 100 --batch-size 16
```
Outputs saved at **every epoch**:
- Checkpoints: `checkpoints/s3det_epoch_001.pt`, `checkpoints/s3det_epoch_002.pt`, ..., `s3det_best.pt`, `s3det_latest.pt`
- Logs: `logs/history.json`, `logs/history.csv`
- Curves:
  - `plots/curves/total_loss.png`
  - `plots/curves/qfl_loss.png`
  - `plots/curves/dfl_loss.png`
  - `plots/curves/bbox_reg_loss.png` (NWD / SIoU)
  - `plots/curves/validation_metrics.png` (Precision, Recall, F1, AP50)
  - `plots/curves/lr_schedule.png`
  - `plots/curves/training_summary_grid.png` (Full multi-panel summary dashboard)

### 5. Evaluate Checkpoints
Evaluate any saved epoch checkpoint on your test split:
```bash
python test.py --checkpoint checkpoints/s3det_best.pt --config data_config.yaml
```
This generates:
- Precision, Recall, F1-Score, and AP@0.50 printed to terminal.
- `plots/test/precision_recall_curve.png`: Test set PR curve.
- `plots/test/predictions/`: Sample images with ground-truth (green) and predicted (red) bounding boxes.

---

## 🔬 Paper Architecture & Mathematical Highlights

1. **IRFA Neck (Eqs. 13–14)**:
   A single `IRU` (Implicit Refinement Unit) is instantiated once and reused 4 times across lateral fusion stages:
   $$\begin{aligned}
   G_{td} &= \psi(P_i; \theta_{shared}), \quad P_i^{td} = P_i + \text{Upsample}(P_{i+1}) \odot G_{td} \\
   G_{bu} &= \psi(N_i; \theta_{shared}), \quad N_i^{out} = N_i + \text{Downsample}(N_{i-1}) \odot G_{bu}
   \end{aligned}$$
2. **LCR-Head**:
   Decoupled branches:
   - Classification: $3\times 3$ Depthwise Convolutions for translation invariance.
   - Regression: $5\times 5$ Depthwise Convolutions for expanded spatial receptive field context around small objects.
3. **Loss Functions**:
   $$\mathcal{L}_{total} = \lambda_{qfl} \mathcal{L}_{QFL} + \lambda_{dfl} \mathcal{L}_{DFL} + \lambda_{bbox} \mathcal{L}_{bbox}$$
   - **QFL**: Quality Focal Loss with $\beta = 2.0$.
   - **DFL**: Distribution Focal Loss over continuous box coordinate intervals.
   - **SIoU** (Eqs. 19–21): Angle, distance, and shape cost penalties.
   - **NWD**: 2D Gaussian Wasserstein distance metric for micro targets where IoU is overly sensitive.
