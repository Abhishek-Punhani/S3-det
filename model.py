import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone import S3Net
from neck import IRFA
from head import LCRHead
from losses import S3DetLoss, generate_points
from nms import nms


class S3Det(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = S3Net(cfg)
        self.neck = IRFA(cfg.stage_channels, cfg.neck_channels, cfg)
        self.head = LCRHead(cfg.neck_channels, cfg.num_classes, cfg.reg_max,
                             cfg.stacked_convs, num_levels=len(cfg.strides))
        self.loss_fn = S3DetLoss(cfg)

    def forward(self, images, targets=None):
        c2, c3, c4 = self.backbone(images)
        feats = self.neck(c2, c3, c4)
        cls_outs, reg_outs = self.head(feats)
        if targets is not None:
            return self.loss_fn(cls_outs, reg_outs, targets)
        return self._decode(cls_outs, reg_outs)

    @torch.no_grad()
    def _decode(self, cls_outs, reg_outs):
        cfg = self.cfg
        device = cls_outs[0].device
        feat_sizes = [c.shape[-2:] for c in cls_outs]
        points_per_level = generate_points(feat_sizes, cfg.strides, device)
        batch_size = cls_outs[0].shape[0]
        reg_max = cfg.reg_max
        bins = torch.arange(reg_max + 1, device=device, dtype=torch.float32)

        results = []
        for b in range(batch_size):
            all_scores, all_boxes = [], []
            for cls_o, reg_o, pts, s in zip(cls_outs, reg_outs, points_per_level, cfg.strides):
                scores = cls_o[b].permute(1, 2, 0).reshape(-1, cfg.num_classes).sigmoid()
                reg = reg_o[b].permute(1, 2, 0).reshape(-1, 4, reg_max + 1)
                prob = F.softmax(reg, dim=-1)
                ltrb = (prob * bins).sum(-1) * s
                boxes = torch.stack([
                    pts[:, 0] - ltrb[:, 0], pts[:, 1] - ltrb[:, 1],
                    pts[:, 0] + ltrb[:, 2], pts[:, 1] + ltrb[:, 3],
                ], dim=-1)
                all_scores.append(scores)
                all_boxes.append(boxes)
            scores = torch.cat(all_scores, dim=0)
            boxes = torch.cat(all_boxes, dim=0)

            max_scores, labels = scores.max(dim=-1)
            keep = max_scores > cfg.score_thresh
            boxes, max_scores, labels = boxes[keep], max_scores[keep], labels[keep]

            if boxes.numel() > 0:
                idx = nms(boxes, max_scores, cfg.nms_iou)[: cfg.max_dets]
                boxes, max_scores, labels = boxes[idx], max_scores[idx], labels[idx]

            results.append({"boxes": boxes.cpu(), "scores": max_scores.cpu(), "labels": labels.cpu()})
        return results

    def count_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def count_flops(self, input_size=640):
        """
        Estimate GFLOPs for a single image at (input_size × input_size).

        Tries torchinfo first (most accurate); falls back to a manual
        analytical estimate by summing 2*C_in*C_out*k*k*H*W over all Conv2d layers.
        Returns (gflops_float, method_str).
        """
        try:
            from torchinfo import summary
            dummy = torch.zeros(1, 3, input_size, input_size,
                                device=next(self.parameters()).device)
            s = summary(self, input_data=dummy, verbose=0)
            gflops = s.total_mult_adds / 1e9
            return round(gflops, 3), "torchinfo"
        except ImportError:
            pass

        # Analytical fallback — register forward hooks to catch actual feature map sizes
        flops = [0]
        hooks = []

        def _conv_hook(m, inp, out):
            x = inp[0]
            _, C_in, H_in, W_in = x.shape
            _, C_out, H_out, W_out = out.shape
            k_h, k_w = m.kernel_size
            groups = m.groups
            # MACs = C_out * (C_in/groups) * k_h * k_w * H_out * W_out
            macs = C_out * (C_in // groups) * k_h * k_w * H_out * W_out
            flops[0] += 2 * macs  # FLOPs = 2 * MACs

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                hooks.append(m.register_forward_hook(_conv_hook))

        was_training = self.training
        self.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, input_size, input_size,
                                device=next(self.parameters()).device)
            self(dummy)
        if was_training:
            self.train()
        for h in hooks:
            h.remove()

        gflops = round(flops[0] / 1e9, 3)
        return gflops, "analytical"

    def model_summary(self, input_size=640):
        """Print the model efficiency table row (params, flops, model size)."""
        total, trainable = self.count_params()
        gflops, method = self.count_flops(input_size)
        fp32_mb = total * 4 / 1024**2
        int8_mb  = total * 1 / 1024**2
        print(f"\n{'─'*56}")
        print(f"  S³-Det Model Summary  (input: {input_size}×{input_size})")
        print(f"{'─'*56}")
        print(f"  Parameters  : {total/1e6:.3f}M  (trainable: {trainable/1e6:.3f}M)")
        print(f"  GFLOPs      : {gflops:.3f}  [{method}]")
        print(f"  Model size  : FP32 = {fp32_mb:.1f} MB  |  INT8 ≈ {int8_mb:.1f} MB")
        print(f"{'─'*56}\n")
        return {"params_M": total/1e6, "gflops": gflops, "fp32_mb": fp32_mb, "int8_mb": int8_mb}
