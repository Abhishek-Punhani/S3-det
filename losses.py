"""Loss functions for S3-Det.

Supports both paper-exact SIoU (Eqs. 19-21) and NWD (Normalized Wasserstein Distance):
    L_total = w_qfl * L_QFL + w_dfl * L_DFL + w_bbox * L_bbox (SIoU or NWD)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- helpers --

def generate_points(feat_sizes, strides, device):
    """Per-level grid-cell centers, in original-image pixel coordinates."""
    points = []
    for (h, w), s in zip(feat_sizes, strides):
        ys, xs = torch.meshgrid(
            torch.arange(h, device=device, dtype=torch.float32),
            torch.arange(w, device=device, dtype=torch.float32),
            indexing="ij",
        )
        xs = (xs + 0.5) * s
        ys = (ys + 0.5) * s
        points.append(torch.stack([xs, ys], dim=-1).reshape(-1, 2))
    return points


def bbox_iou(box1, box2, eps=1e-7):
    """Elementwise IoU between two (N,4) xyxy tensors -> (N,)."""
    x1 = torch.max(box1[:, 0], box2[:, 0])
    y1 = torch.max(box1[:, 1], box2[:, 1])
    x2 = torch.min(box1[:, 2], box2[:, 2])
    y2 = torch.min(box1[:, 3], box2[:, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area1 = (box1[:, 2] - box1[:, 0]).clamp(min=0) * (box1[:, 3] - box1[:, 1]).clamp(min=0)
    area2 = (box2[:, 2] - box2[:, 0]).clamp(min=0) * (box2[:, 3] - box2[:, 1]).clamp(min=0)
    return inter / (area1 + area2 - inter + eps)


def nwd_loss(pred_boxes, target_boxes, constant=12.8, eps=1e-7):
    """1 - NWD, where boxes are modelled as 2D Gaussians N((cx,cy),
    diag((w/2)^2,(h/2)^2)) and NWD = exp(-sqrt(W2)/constant)."""
    pcx = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2
    pcy = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2
    pw = (pred_boxes[:, 2] - pred_boxes[:, 0]).clamp(min=eps)
    ph = (pred_boxes[:, 3] - pred_boxes[:, 1]).clamp(min=eps)

    tcx = (target_boxes[:, 0] + target_boxes[:, 2]) / 2
    tcy = (target_boxes[:, 1] + target_boxes[:, 3]) / 2
    tw = (target_boxes[:, 2] - target_boxes[:, 0]).clamp(min=eps)
    th = (target_boxes[:, 3] - target_boxes[:, 1]).clamp(min=eps)

    center_term = (pcx - tcx) ** 2 + (pcy - tcy) ** 2
    size_term = ((pw - tw) / 2) ** 2 + ((ph - th) / 2) ** 2
    w2 = (center_term + size_term).clamp(min=0)
    nwd = torch.exp(-torch.sqrt(w2 + eps) / constant)
    return 1 - nwd


def siou_loss(pred_boxes, target_boxes, eps=1e-7):
    """SIoU loss according to paper (Eqs. 19-21):
    L_SIoU = 1 - IoU + (Delta + Omega) / 2
    where Delta is distance cost taking angle Lambda into account,
    and Omega is shape cost.
    """
    px1, py1, px2, py2 = pred_boxes.unbind(-1)
    tx1, ty1, tx2, ty2 = target_boxes.unbind(-1)

    pcx = (px1 + px2) / 2
    pcy = (py1 + py2) / 2
    pw = (px2 - px1).clamp(min=eps)
    ph = (py2 - py1).clamp(min=eps)

    tcx = (tx1 + tx2) / 2
    tcy = (ty1 + ty2) / 2
    tw = (tx2 - tx1).clamp(min=eps)
    th = (ty2 - ty1).clamp(min=eps)

    iou = bbox_iou(pred_boxes, target_boxes, eps=eps)

    cw = (torch.max(px2, tx2) - torch.min(px1, tx1)).clamp(min=eps)
    ch = (torch.max(py2, ty2) - torch.min(py1, ty1)).clamp(min=eps)

    dx = (pcx - tcx).abs()
    dy = (pcy - tcy).abs()
    rho = torch.sqrt(dx ** 2 + dy ** 2 + eps)

    # Angle cost Lambda (Eq. 19)
    sin_alpha = (torch.min(dx, dy) / rho).clamp(-1.0 + eps, 1.0 - eps)
    alpha = torch.arcsin(sin_alpha)
    sin_angle = torch.sin(alpha - math.pi / 4)
    angle_cost = 1.0 - 2.0 * (sin_angle ** 2)

    # Distance cost Delta (Eq. 20)
    gamma = 2.0 - angle_cost
    rho_x = (dx / cw) ** 2
    rho_y = (dy / ch) ** 2
    dist_cost = (1.0 - torch.exp(-gamma * rho_x)) + (1.0 - torch.exp(-gamma * rho_y))

    # Shape cost Omega (Eq. 21)
    omega_w = (pw - tw).abs() / torch.max(pw, tw)
    omega_h = (ph - th).abs() / torch.max(ph, th)
    shape_cost = (1.0 - torch.exp(-omega_w)) ** 4 + (1.0 - torch.exp(-omega_h)) ** 4

    siou = iou - (dist_cost + shape_cost) / 2.0
    return 1.0 - siou


# ------------------------------------------------------------------ losses --

class QualityFocalLoss(nn.Module):
    """QFL(sigma, y) = -|y - sigma|^beta * [(1-y)log(1-sigma) + y log(sigma)]"""
    def __init__(self, beta=2.0):
        super().__init__()
        self.beta = beta

    def forward(self, logits, targets):
        pred_sigmoid = logits.sigmoid()
        scale_factor = (pred_sigmoid - targets).abs().pow(self.beta)
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        return (loss * scale_factor).sum()


class DistributionFocalLoss(nn.Module):
    """Cross-entropy against the two bins straddling a continuous target,
    weighted by linear interpolation distance (Li et al., GFL)."""
    def forward(self, pred_dist, target):
        num_bins = pred_dist.shape[-1]
        target = target.clamp(0, num_bins - 1 - 1e-3)
        left = target.long()
        right = left + 1
        weight_left = right.float() - target
        weight_right = target - left.float()
        loss = (F.cross_entropy(pred_dist, left, reduction="none") * weight_left
                + F.cross_entropy(pred_dist, right, reduction="none") * weight_right)
        return loss.sum()


# --------------------------------------------------------------- assembly --

class S3DetLoss(nn.Module):
    """FCOS-style point assignment (center-sampling + per-level scale
    range) followed by QFL + DFL + NWD. This is a simplified stand-in for
    whatever exact assigner the paper uses internally (not fully specified
    in the text) -- see README for how to swap in ATSS/SimOTA later."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.qfl = QualityFocalLoss(beta=cfg.qfl_beta)
        self.dfl = DistributionFocalLoss()
        self.reg_max = cfg.reg_max
        self.strides = cfg.strides
        self.regress_ranges = cfg.regress_ranges
        self.center_sampling_radius = cfg.center_sampling_radius
        self.num_classes = cfg.num_classes
        self.loss_weights = cfg.loss_weights
        self.loss_type = getattr(cfg, "loss_type", "nwd").lower()
        self.nwd_constant = getattr(cfg, "nwd_constant", 12.8)

    def forward(self, cls_outs, reg_outs, targets):
        device = cls_outs[0].device
        feat_sizes = [c.shape[-2:] for c in cls_outs]
        points_per_level = generate_points(feat_sizes, self.strides, device)
        num_points_per_level = [p.shape[0] for p in points_per_level]
        all_points = torch.cat(points_per_level, dim=0)

        range_per_point = torch.cat([
            torch.tensor(self.regress_ranges[i], device=device, dtype=torch.float32)
                .unsqueeze(0).repeat(n, 1)
            for i, n in enumerate(num_points_per_level)
        ], dim=0)
        strides_per_point = torch.cat([
            torch.full((n,), s, device=device, dtype=torch.float32)
            for n, s in zip(num_points_per_level, self.strides)
        ], dim=0)

        batch_size = cls_outs[0].shape[0]
        cls_flat = torch.cat([
            c.permute(0, 2, 3, 1).reshape(batch_size, -1, self.num_classes) for c in cls_outs
        ], dim=1)
        reg_flat = torch.cat([
            r.permute(0, 2, 3, 1).reshape(batch_size, -1, 4, self.reg_max + 1) for r in reg_outs
        ], dim=1)

        total_qfl  = cls_flat.new_zeros(())
        total_dfl  = cls_flat.new_zeros(())
        total_bbox = cls_flat.new_zeros(())
        total_pos  = 0
        total_gts  = 0          # across batch — for fallback rate
        total_fallback = 0      # GT objects rescued by nearest-point fallback

        for b in range(batch_size):
            gt_boxes  = targets[b]["boxes"].to(device)
            gt_labels = targets[b]["labels"].to(device)
            qfl_target = torch.zeros((all_points.shape[0], self.num_classes), device=device)

            if gt_boxes.numel() == 0:
                total_qfl = total_qfl + self.qfl(cls_flat[b], qfl_target)
                continue

            pos_mask, matched_gt_idx, n_fallback = self._assign(
                all_points, range_per_point, strides_per_point, gt_boxes)

            total_gts     += gt_boxes.shape[0]
            total_fallback += n_fallback

            if pos_mask.sum() == 0:
                total_qfl = total_qfl + self.qfl(cls_flat[b], qfl_target)
                continue

            pos_points    = all_points[pos_mask]
            pos_strides   = strides_per_point[pos_mask]
            pos_gt_boxes  = gt_boxes[matched_gt_idx[pos_mask]]
            pos_gt_labels = gt_labels[matched_gt_idx[pos_mask]]
            pos_reg_dist  = reg_flat[b][pos_mask]

            pred_ltrb = self._decode_dfl(pos_reg_dist) * pos_strides.unsqueeze(-1)
            pred_boxes = torch.stack([
                pos_points[:, 0] - pred_ltrb[:, 0],
                pos_points[:, 1] - pred_ltrb[:, 1],
                pos_points[:, 0] + pred_ltrb[:, 2],
                pos_points[:, 1] + pred_ltrb[:, 3],
            ], dim=-1)

            ious = bbox_iou(pred_boxes.detach(), pos_gt_boxes).clamp(min=0)
            qfl_target[pos_mask, pos_gt_labels] = ious
            total_qfl = total_qfl + self.qfl(cls_flat[b], qfl_target)

            target_ltrb = torch.stack([
                (pos_points[:, 0] - pos_gt_boxes[:, 0]) / pos_strides,
                (pos_points[:, 1] - pos_gt_boxes[:, 1]) / pos_strides,
                (pos_gt_boxes[:, 2] - pos_points[:, 0]) / pos_strides,
                (pos_gt_boxes[:, 3] - pos_points[:, 1]) / pos_strides,
            ], dim=-1).clamp(min=0, max=self.reg_max - 0.01)

            dfl_sum = (self.dfl(pos_reg_dist[:, 0], target_ltrb[:, 0])
                       + self.dfl(pos_reg_dist[:, 1], target_ltrb[:, 1])
                       + self.dfl(pos_reg_dist[:, 2], target_ltrb[:, 2])
                       + self.dfl(pos_reg_dist[:, 3], target_ltrb[:, 3]))
            total_dfl = total_dfl + dfl_sum

            loss_type = getattr(self.cfg, "loss_type", self.loss_type).lower()
            if loss_type == "siou":
                bbox_cost = siou_loss(pred_boxes, pos_gt_boxes).sum()
            else:
                bbox_cost = nwd_loss(pred_boxes, pos_gt_boxes, constant=self.nwd_constant).sum()
            total_bbox = total_bbox + bbox_cost

            total_pos += int(pos_mask.sum().item())

        total_pos = max(total_pos, 1)
        loss_type = getattr(self.cfg, "loss_type", self.loss_type).lower()
        w_qfl   = self.loss_weights.get("qfl",   1.0)
        w_dfl   = self.loss_weights.get("dfl",   0.25)
        w_bbox  = self.loss_weights.get(loss_type, self.loss_weights.get("bbox", 2.0))

        loss_qfl  = w_qfl  * total_qfl  / total_pos
        loss_dfl  = w_dfl  * total_dfl  / total_pos
        loss_bbox = w_bbox * total_bbox / total_pos
        loss      = loss_qfl + loss_dfl + loss_bbox

        fallback_rate = total_fallback / max(total_gts, 1)

        return {
            "loss":             loss,
            "loss_qfl":         loss_qfl,
            "loss_dfl":         loss_dfl,
            "loss_bbox":        loss_bbox,
            "loss_nwd":         loss_bbox if loss_type == "nwd"  else cls_flat.new_zeros(()),
            "loss_siou":        loss_bbox if loss_type == "siou" else cls_flat.new_zeros(()),
            "loss_type":        loss_type,
            "num_pos":          total_pos,
            "num_gts":          total_gts,
            "fallback_count":   total_fallback,   # GT boxes rescued by nearest-point fallback
            "fallback_rate":    fallback_rate,     # fraction of GTs needing fallback
        }

    def _decode_dfl(self, reg_dist):
        prob = F.softmax(reg_dist, dim=-1)
        bins = torch.arange(self.reg_max + 1, device=reg_dist.device, dtype=torch.float32)
        return (prob * bins).sum(-1)

    def _assign(self, points, ranges, strides, gt_boxes):
        """FCOS-style assignment with nearest-point fallback for tiny objects.

        Returns: (pos_mask, matched_gt_idx, n_fallback)
            pos_mask        : (num_points,) bool — positive grid points
            matched_gt_idx  : (num_points,) int  — GT index per positive point
            n_fallback      : int — number of GT boxes rescued by nearest-point fallback
        """
        num_points = points.shape[0]
        num_gt     = gt_boxes.shape[0]

        xs, ys = points[:, 0].unsqueeze(1), points[:, 1].unsqueeze(1)
        gx1, gy1, gx2, gy2 = [gt_boxes[:, i].unsqueeze(0) for i in range(4)]

        l, t, r, b = xs - gx1, ys - gy1, gx2 - xs, gy2 - ys
        ltrb = torch.stack([l, t, r, b], dim=-1)
        inside_box = ltrb.min(-1).values > 0

        cx, cy = (gx1 + gx2) / 2, (gy1 + gy2) / 2
        radius  = strides.unsqueeze(1) * self.center_sampling_radius
        inside_center = ((xs - cx).abs() < radius) & ((ys - cy).abs() < radius)

        max_reg = ltrb.max(-1).values
        inside_range = (max_reg >= ranges[:, 0].unsqueeze(1)) & (max_reg <= ranges[:, 1].unsqueeze(1))

        valid = inside_box & inside_center & inside_range  # (num_points, num_gt)

        # ---- Nearest-point fallback for GT boxes with no positives ----------
        gt_covered = valid.any(dim=0)        # (num_gt,) — which GTs have ≥1 positive
        n_fallback = int((~gt_covered).sum().item())
        if not gt_covered.all():
            uncovered = (~gt_covered).nonzero(as_tuple=False).squeeze(1)
            gt_cx = cx.squeeze(0)[uncovered]   # (U,)
            gt_cy = cy.squeeze(0)[uncovered]   # (U,)
            dist2 = (points[:, 0].unsqueeze(1) - gt_cx.unsqueeze(0)) ** 2 + \
                    (points[:, 1].unsqueeze(1) - gt_cy.unsqueeze(0)) ** 2  # (P, U)
            nearest_point_idx = dist2.argmin(dim=0)  # (U,)
            for u_idx, gt_idx in zip(nearest_point_idx, uncovered):
                valid[u_idx, gt_idx] = True

        # ---- Resolve ambiguity: multiple GTs valid for same point -----------
        areas = ((gx2 - gx1) * (gy2 - gy1)).expand(num_points, num_gt).clone()
        areas[~valid] = 1e8
        min_area, matched_gt_idx = areas.min(dim=1)
        pos_mask = min_area < 1e8
        return pos_mask, matched_gt_idx, n_fallback

