"""
Central configuration for the S3-Det reimplementation.

Where a value is explicitly given in the paper (Scientific Reports,
s41598-026-53405-7), it's used directly. Where the paper's text doesn't
expose the exact number (e.g. per-stage channel widths, block counts), a
reasonable, clearly-labelled engineering default is used instead -- see
README.md "Assumptions" section for the full list.
"""
from dataclasses import dataclass, field, asdict
from typing import Tuple, Dict, List, Any
import yaml
import os


@dataclass
class S3DetConfig:
    # --- dataset & paths ---
    data_root: str = "dataset"
    train_split: str = "train"
    test_split: str = "test"
    val_fraction: float = 0.15        # fraction of train data carved out for validation at runtime
    val_split_seed: int  = 42          # RNG seed for the train/val random_split (reproducibility)
    class_names: List[str] = field(default_factory=lambda: ["drone"])
    checkpoint_dir: str = "checkpoints"
    plot_dir: str = "plots"
    log_dir: str = "logs"

    # --- per-size evaluation bins (sqrt(w*h) in pixels) ---
    # Each tuple is (label, max_sqrt_area). ">64" is implicit last bucket.
    tiny_size_bins: List[int] = field(default_factory=lambda: [8, 16, 32, 64])

    # --- AP IoU thresholds for AP50:95 ---
    ap_iou_thresholds: List[float] = field(
        default_factory=lambda: [round(t * 0.05, 2) for t in range(10, 20)]  # 0.50..0.95
    )

    # --- input / task ---
    input_size: int = 640
    num_classes: int = 1  # single class: "drone". no_drone folder = background negatives

    # --- backbone (S3Net) ---
    stem_channels: int = 32
    stage_channels: Tuple[int, int, int] = (64, 128, 256)   # C2, C3, C4 widths [ASSUMED]
    stage_blocks: Tuple[int, int, int] = (2, 2, 2)          # SSB blocks per stage [ASSUMED]
    ssb_expansion: int = 4       # lambda (1x1 expansion ratio inside SSB) [ASSUMED]
    sfg_reduction: int = 8       # r (channel-attention reduction ratio in SFG) [ASSUMED]

    # --- neck (IRFA) ---
    neck_channels: int = 96      # [ASSUMED]
    strides: Tuple[int, int, int] = (4, 8, 16)   # P2, P3, P4 [ASSUMED - see README]
    regress_ranges: Tuple[Tuple[float, float], ...] = ((0, 64), (64, 128), (128, 1e8))
    center_sampling_radius: float = 2.5

    # --- head (LCR-Head) ---
    reg_max: int = 16            # DFL distribution bins
    stacked_convs: int = 2       # matches the paper's 2x DWConv per branch

    # --- loss ---
    loss_type: str = "nwd"       # "nwd" or "siou" (paper baseline is siou, enhanced is nwd)
    qfl_beta: float = 2.0        # paper-specified
    nwd_constant: float = 12.8   # NWD normalization constant, tuned for small objects
    loss_weights: Dict[str, float] = field(
        default_factory=lambda: {"qfl": 1.0, "dfl": 0.25, "nwd": 2.0, "siou": 2.0}
    )

    # --- training (paper-specified) ---
    lr: float = 1e-3
    weight_decay: float = 0.05
    epochs: int = 100
    batch_size: int = 16
    warmup_iters: int = 500
    min_lr: float = 5e-5
    save_every_epoch: bool = True
    eval_every_epoch: bool = True
    workers: int = 4
    device: str = "cuda"
    grad_clip_norm: float = 35.0    # max gradient norm for clipping

    # --- EMA (Exponential Moving Average of weights) ---
    # Stabilizes training; validation is run on EMA weights for better final metrics.
    ema: bool = True
    ema_decay: float = 0.9998       # typical for object detection (0.999x range)
    ema_warmup_iters: int = 100     # steps before EMA kicks in (let model warm up first)

    # --- AMP (Automatic Mixed Precision) ---
    # Enabled automatically only when CUDA is available. No-op on CPU.
    amp: bool = True

    # --- NWD constant normalization ---
    # If True, nwd_constant is treated as defined at input_size=640 and scaled
    # proportionally: effective_C = nwd_constant * (input_size / 640).
    # This keeps the NWD loss semantics consistent when you change resolution.
    nwd_normalize_to_input: bool = True

    # --- Loss auto-balancing ---
    # After loss_balance_warmup_iters steps, rescale individual loss weights so
    # they contribute roughly equally in the gradient norm sense.
    # Set to 0 to disable (use fixed loss_weights throughout).
    loss_balance_iters: int = 0     # disabled by default; set e.g. 200 to enable

    # --- inference ---
    score_thresh: float = 0.3
    nms_iou: float = 0.6
    max_dets: int = 100

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "S3DetConfig":
        """Load configuration from a YAML file."""
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f) or {}

        # Convert lists to tuples only for tuple-typed fields
        for tuple_field in ["stage_channels", "stage_blocks", "strides"]:
            if tuple_field in data and isinstance(data[tuple_field], list):
                data[tuple_field] = tuple(data[tuple_field])
        if "regress_ranges" in data and isinstance(data["regress_ranges"], list):
            data["regress_ranges"] = tuple(tuple(r) for r in data["regress_ranges"])

        # Filter out keys not in dataclass
        valid_keys = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    def to_yaml(self, yaml_path: str):
        """Save configuration to a YAML file."""
        d = asdict(self)
        # Convert tuples to lists for cleaner yaml output
        for k, v in d.items():
            if isinstance(v, tuple):
                d[k] = list(v)
        os.makedirs(os.path.dirname(os.path.abspath(yaml_path)), exist_ok=True)
        with open(yaml_path, "w") as f:
            yaml.safe_dump(d, f, sort_keys=False, default_flow_style=False)
