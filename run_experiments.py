import os
import json
import subprocess
import matplotlib.pyplot as plt

# The 4 requested backbones
BACKBONES = [
    "resnet18",
    "mobilenetv3_large_100",
    "efficientnet_b0",
    "ghostnet_100"
]

EPOCHS = 100
BATCH_SIZE = 16
DATA_ROOT = "dataset_full"
DEVICE = "cuda"

def run_training():
    for backbone in BACKBONES:
        print(f"\n{'='*60}")
        print(f"Starting training for backbone: {backbone}")
        print(f"{'='*60}\n")
        
        # Command matches train.py arguments
        cmd = [
            "python", "train.py",
            "--data-root", DATA_ROOT,
            "--backbone", backbone,
            "--epochs", str(EPOCHS),
            "--batch-size", str(BATCH_SIZE),
            "--device", DEVICE
        ]
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error training {backbone}: {e}")
            print("Skipping to next backbone...")

def plot_comparisons():
    print("\nGenerating comparative plots...")
    
    metrics_to_plot = {
        "val_ap50": "Validation AP@50",
        "val_recall_bin0": "Tiny Drone Recall (Bin 0)",
        "loss": "Training Total Loss",
        "val_loss": "Validation Total Loss"
    }
    
    histories = {}
    
    # Load all histories
    for backbone in BACKBONES:
        history_file = f"checkpoints_{backbone}/history.json"
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                histories[backbone] = json.load(f)
        else:
            print(f"Warning: {history_file} not found. Did training complete?")
            
    if not histories:
        print("No history files found. Exiting plotting.")
        return

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    
    for i, (metric_key, metric_title) in enumerate(metrics_to_plot.items()):
        ax = axes[i]
        
        for j, (backbone, history) in enumerate(histories.items()):
            if metric_key in history:
                epochs = history.get("epoch", [])
                
                # Some lists might have None values (like val metrics before first eval)
                # Replace None with NaN for plotting
                safe_vals = [v if v is not None else float("nan") for v in history[metric_key]]
                
                ax.plot(epochs, safe_vals, label=backbone, color=colors[j % len(colors)], lw=2)
                
        ax.set_title(metric_title, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric_title)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()
        
    fig.suptitle(f"S3-Det Backbone Comparison ({EPOCHS} Epochs)", fontsize=16, fontweight="bold")
    fig.tight_layout()
    
    plot_path = "experiments_comparison.png"
    fig.savefig(plot_path, dpi=300)
    print(f"Comparison plot saved to {plot_path}")

if __name__ == "__main__":
    # Check if timm is installed
    try:
        import timm
    except ImportError:
        print("timm not found. Installing...")
        subprocess.run(["pip", "install", "timm"], check=True)
        
    # Run the automated pipeline
    run_training()
    plot_comparisons()
