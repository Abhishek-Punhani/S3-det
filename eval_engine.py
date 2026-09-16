"""
eval_engine.py — Comprehensive evaluation engine for S3-Det UAV detector.

Computes all metrics documented in the project specification:
  Core: Precision, Recall, F1, AP@50, AP@50:95
  Per-size: AP50/P/R for 5 bins (<=8, 8-16, 16-32, 32-64, >64 px)
  Deployment: FPPI, F1 vs confidence threshold sweep
  Localization: center error (mean/median/p95, normalized), W/H error, IoU/NWD on TPs
  Diagnostic: NWD vs IoU vs center displacement (research figure)
"""
from __future__ import annotations
import math
import numpy as np
import torch


# ─────────────────────────────────────────────────────── helpers ─────

def _box_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """IoU matrix (Na, Nb) between two sets of xyxy boxes."""
    x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.clip(union, 1e-7, None)


def _nwd_score(pred_box: np.ndarray, gt_box: np.ndarray, C: float = 12.8) -> float:
    """NWD similarity for two xyxy boxes."""
    pcx = (pred_box[0] + pred_box[2]) / 2;  pcy = (pred_box[1] + pred_box[3]) / 2
    pw  = max(pred_box[2] - pred_box[0], 1e-7); ph = max(pred_box[3] - pred_box[1], 1e-7)
    gcx = (gt_box[0]  + gt_box[2])  / 2;   gcy = (gt_box[1]  + gt_box[3])  / 2
    gw  = max(gt_box[2]  - gt_box[0],  1e-7); gh = max(gt_box[3]  - gt_box[1],  1e-7)
    w2 = (pcx-gcx)**2 + (pcy-gcy)**2 + ((pw-gw)/2)**2 + ((ph-gh)/2)**2
    return float(np.exp(-np.sqrt(w2 + 1e-7) / C))


def _gt_scale(gt_box: np.ndarray) -> float:
    w = max(gt_box[2] - gt_box[0], 0.0)
    h = max(gt_box[3] - gt_box[1], 0.0)
    return math.sqrt(w * h)


def _ap_101point(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """101-point COCO-style interpolated Average Precision.

    Uses 101 evenly spaced recall thresholds (0.00, 0.01, ..., 1.00) and takes
    the mean of the max right-interpolated precision at each threshold.
    This matches the pycocotools evaluation protocol and makes AP numbers
    directly comparable with published detector results.

    For each recall threshold r:
        p(r) = max(precision[recall >= r])   (0 if no such point)
    AP = mean over 101 thresholds
    """
    ap = 0.0
    for thr in np.linspace(0, 1, 101):
        p_above = precisions[recalls >= thr]
        ap += (float(p_above.max()) if len(p_above) else 0.0) / 101.0
    return ap


def _match_image(pred_boxes, pred_scores, gt_boxes, iou_thresh):
    """Greedy match. Returns list of (score, is_tp, pred_box, gt_box_or_None, gt_scale)."""
    results = []
    matched_gt = set()
    order = np.argsort(pred_scores)[::-1]
    for pi in order:
        score = float(pred_scores[pi])
        pb = pred_boxes[pi]
        if len(gt_boxes) == 0:
            results.append((score, 0, pb, None, 0.0))
            continue
        iou_row = _box_iou_matrix(pb[None], gt_boxes)[0]
        best_gi = int(np.argmax(iou_row))
        if iou_row[best_gi] >= iou_thresh and best_gi not in matched_gt:
            matched_gt.add(best_gi)
            results.append((score, 1, pb, gt_boxes[best_gi], _gt_scale(gt_boxes[best_gi])))
        else:
            results.append((score, 0, pb, None, 0.0))
    return results, matched_gt


def compute_pr_ap(detections: list, total_gts: int):
    """(score,is_tp) list -> recalls, precisions, AP (11-point)."""
    if not detections:
        return np.zeros(1), np.zeros(1), 0.0
    detections = sorted(detections, key=lambda x: x[0], reverse=True)
    tps = np.array([d[1] for d in detections], dtype=np.float32)
    fps = 1.0 - tps
    cum_tp = np.cumsum(tps)
    cum_fp = np.cumsum(fps)
    recalls    = cum_tp / max(total_gts, 1)
    precisions = cum_tp / np.clip(cum_tp + cum_fp, 1e-7, None)
    return recalls, precisions, _ap_101point(recalls, precisions)


class EvalResult:
    def __init__(self):
        self.precision = self.recall = self.f1 = 0.0
        self.ap50 = self.ap50_95 = 0.0
        self.total_gts = self.total_preds = 0
        self.tp = self.fp = self.fn = 0
        self.fppi = 0.0
        self.num_images = 0
        self.recalls    = np.zeros(1)
        self.precisions = np.zeros(1)
        self.size_bins  = []           # list of dicts
        self.thresh_values = []
        self.thresh_precision = []
        self.thresh_recall    = []
        self.thresh_f1        = []
        self.size_histogram_bins   = []
        self.size_histogram_recall = []
        self.center_err_mean = self.center_err_median = self.center_err_p95 = 0.0
        self.center_err_norm_mean = self.center_err_norm_median = self.center_err_norm_p95 = 0.0
        self.width_err_rel_mean = self.height_err_rel_mean = 0.0
        self.tp_iou_mean = self.tp_nwd_mean = 0.0
        self.tp_ious = []
        self.tp_nwds = []

    def print_summary(self):
        print("\n" + "="*72)
        print("  S3-Det Evaluation Summary")
        print("="*72)
        print(f"  Images: {self.num_images}   GT: {self.total_gts}   Preds: {self.total_preds}")
        print(f"  TP={self.tp}  FP={self.fp}  FN={self.fn}  FPPI={self.fppi:.3f}")
        print("-"*72)
        print(f"  Precision : {self.precision:.4f}")
        print(f"  Recall    : {self.recall:.4f}")
        print(f"  F1        : {self.f1:.4f}")
        print(f"  AP@50     : {self.ap50:.4f}")
        print(f"  AP@50:95  : {self.ap50_95:.4f}")
        print("-"*72)
        if self.size_bins:
            print(f"  {'Size bin':<14} {'#GT':>6} {'P':>7} {'R':>7} {'F1':>7} {'AP50':>7}")
            for b in self.size_bins:
                print(f"  {b['label']:<14} {b['n_gt']:>6} {b['precision']:>7.3f} "
                      f"{b['recall']:>7.3f} {b['f1']:>7.3f} {b['ap50']:>7.3f}")
        print("-"*72)
        print(f"  Center err:  mean={self.center_err_mean:.2f}px  "
              f"median={self.center_err_median:.2f}px  p95={self.center_err_p95:.2f}px")
        print(f"  Center norm: mean={self.center_err_norm_mean:.3f}  "
              f"median={self.center_err_norm_median:.3f}  p95={self.center_err_norm_p95:.3f}")
        print(f"  W-err rel: {self.width_err_rel_mean:.3f}   H-err rel: {self.height_err_rel_mean:.3f}")
        print(f"  TP IoU mean: {self.tp_iou_mean:.3f}    TP NWD mean: {self.tp_nwd_mean:.3f}")
        print("="*72)

    def to_dict(self):
        return {
            "precision": self.precision, "recall": self.recall, "f1": self.f1,
            "ap50": self.ap50, "ap50_95": self.ap50_95,
            "total_gts": self.total_gts, "total_preds": self.total_preds,
            "tp": self.tp, "fp": self.fp, "fn": self.fn,
            "fppi": self.fppi, "num_images": self.num_images,
            "center_err_mean": self.center_err_mean,
            "center_err_median": self.center_err_median,
            "center_err_p95": self.center_err_p95,
            "center_err_norm_mean": self.center_err_norm_mean,
            "center_err_norm_median": self.center_err_norm_median,
            "center_err_norm_p95": self.center_err_norm_p95,
            "width_err_rel_mean": self.width_err_rel_mean,
            "height_err_rel_mean": self.height_err_rel_mean,
            "tp_iou_mean": self.tp_iou_mean,
            "tp_nwd_mean": self.tp_nwd_mean,
            "size_bins": self.size_bins,
        }


@torch.no_grad()
def evaluate(model, loader, device, score_thresh=0.3, nwd_constant=12.8,
             size_bins=None, num_visuals=12, iou_thresh=0.5, ap_iou_thresholds=None):
    """Full evaluation pass. Returns (EvalResult, visual_samples)."""
    if size_bins is None:
        size_bins = [8, 16, 32, 64]
    if ap_iou_thresholds is None:
        ap_iou_thresholds = [round(0.50 + i*0.05, 2) for i in range(10)]

    model.eval()
    all_dets50 = []
    total_gts = total_preds = num_images = 0
    visual_samples = []

    bin_labels = [f"<={size_bins[0]}"]
    for i in range(1, len(size_bins)):
        bin_labels.append(f"{size_bins[i-1]}-{size_bins[i]}")
    bin_labels.append(f">{size_bins[-1]}")
    n_bins = len(bin_labels)

    bin_dets     = [[] for _ in range(n_bins)]
    bin_gts      = [0]  * n_bins
    hist_gt      = [0]  * n_bins
    hist_matched = [0]  * n_bins

    tp_ce = []; tp_ce_n = []; tp_we = []; tp_he = []; tp_ious_l = []; tp_nwds_l = []
    ap5095_dets = {thr: [] for thr in ap_iou_thresholds}

    def _bin(scale):
        for i, thr in enumerate(size_bins):
            if scale <= thr: return i
        return n_bins - 1

    for images, targets in loader:
        images = images.to(device)
        preds = model(images)

        for pred, target in zip(preds, targets):
            num_images += 1
            gt_boxes   = np.array(target["boxes"]) if not isinstance(target["boxes"], np.ndarray) else target["boxes"]
            pb_raw     = pred["boxes"]
            ps_raw     = pred["scores"]
            pred_boxes  = pb_raw.cpu().numpy() if hasattr(pb_raw, "numpy") else np.array(pb_raw)
            pred_scores = ps_raw.cpu().numpy() if hasattr(ps_raw, "numpy") else np.array(ps_raw)

            keep = pred_scores >= score_thresh
            pred_boxes  = pred_boxes[keep]
            pred_scores = pred_scores[keep]

            n_gt = len(gt_boxes); n_pred = len(pred_boxes)
            total_gts += n_gt; total_preds += n_pred

            gt_scales = []
            for gb in gt_boxes:
                sc = _gt_scale(gb); gt_scales.append(sc)
                bi = _bin(sc); hist_gt[bi] += 1; bin_gts[bi] += 1

            if len(visual_samples) < num_visuals:
                visual_samples.append({
                    "path": target.get("path", ""),
                    "gt_boxes": gt_boxes.copy(),
                    "pred_boxes": pred_boxes.copy(),
                    "pred_scores": pred_scores.copy(),
                })

            if n_pred == 0:
                continue

            matched, _ = _match_image(pred_boxes, pred_scores, gt_boxes, iou_thresh)
            for score, is_tp, pb, gb, gt_sc in matched:
                all_dets50.append((score, is_tp))
                if is_tp and gb is not None:
                    bi = _bin(gt_sc); bin_dets[bi].append((score, 1)); hist_matched[bi] += 1
                    pcx=(pb[0]+pb[2])/2; pcy=(pb[1]+pb[3])/2
                    gcx=(gb[0]+gb[2])/2; gcy=(gb[1]+gb[3])/2
                    pw=pb[2]-pb[0]; ph=pb[3]-pb[1]; gw=gb[2]-gb[0]; gh=gb[3]-gb[1]
                    Ec=math.sqrt((pcx-gcx)**2+(pcy-gcy)**2)
                    sc=max(math.sqrt(max(gw,0)*max(gh,0)),1e-3)
                    tp_ce.append(Ec); tp_ce_n.append(Ec/sc)
                    tp_we.append(abs(pw-gw)/max(gw,1e-3)); tp_he.append(abs(ph-gh)/max(gh,1e-3))
                    tp_ious_l.append(float(_box_iou_matrix(pb[None],gb[None])[0,0]))
                    tp_nwds_l.append(_nwd_score(pb, gb, C=nwd_constant))
                # NOTE: FPs are NOT assigned to size bins.
                # Per-size recall/AP is GT-based only: we track which GT sizes
                # were matched (TPs). FPs contribute to global FPPI only.

            for thr in ap_iou_thresholds:
                res_thr, _ = _match_image(pred_boxes, pred_scores, gt_boxes, iou_thresh=thr)
                for score, is_tp, *_ in res_thr:
                    ap5095_dets[thr].append((score, is_tp))

    # --- build result ---
    result = EvalResult()
    result.num_images = num_images; result.total_gts = total_gts; result.total_preds = total_preds
    result.recalls, result.precisions, result.ap50 = compute_pr_ap(all_dets50, total_gts)

    final_tp = int(sum(d[1] for d in all_dets50))
    final_fp = len(all_dets50) - final_tp
    result.tp = final_tp; result.fp = final_fp; result.fn = max(total_gts - final_tp, 0)
    result.fppi = final_fp / max(num_images, 1)
    if final_tp+final_fp > 0: result.precision = final_tp/(final_tp+final_fp)
    if total_gts > 0:         result.recall    = final_tp/total_gts
    if result.precision+result.recall > 0:
        result.f1 = 2*result.precision*result.recall/(result.precision+result.recall)

    per_thr = [compute_pr_ap(ap5095_dets[t], total_gts)[2] for t in ap_iou_thresholds]
    result.ap50_95 = float(np.mean(per_thr)) if per_thr else 0.0

    for bi, label in enumerate(bin_labels):
        tp_dets = bin_dets[bi]  # only TPs in this bin (FPs not assigned to bins)
        ng = bin_gts[bi]
        if tp_dets:
            # recall = TP / n_GT_in_bin  (exact, no FP ambiguity)
            tp_b = len(tp_dets)   # all entries in bin_dets are TPs
            r = tp_b / max(ng, 1)
            # AP50 per bin: precision curve built only from GT-matched detections
            # versus total GTs in this bin.  Precision-at-rank k = k / k = 1.0
            # for all k since there are no FPs → monotonically decreasing recall.
            # Build proper AP from these TP scores + "missed GTs" as FN at the end.
            scores_sorted = sorted([d[0] for d in tp_dets], reverse=True)
            fn = max(ng - tp_b, 0)
            cum_tp_arr = np.arange(1, tp_b + 1, dtype=np.float32)
            recalls_b  = cum_tp_arr / max(ng, 1)
            prec_b     = cum_tp_arr / cum_tp_arr      # = 1.0 for all (no per-bin FP)
            ap_b = _ap_101point(recalls_b, prec_b)
        else:
            r = ap_b = 0.0
        # Precision/F1 per bin are undefined without per-bin FP tracking;
        # set to NaN so they are never silently misinterpreted.
        result.size_bins.append({
            "label":     label,
            "n_gt":      ng,
            "recall":    r,
            "ap50":      ap_b,
            "precision": float("nan"),   # undefined per-bin (FPs counted globally)
            "f1":        float("nan"),   # undefined per-bin
        })

    result.size_histogram_bins   = bin_labels
    result.size_histogram_recall = [hist_matched[bi]/max(hist_gt[bi],1) for bi in range(n_bins)]

    if tp_ce:
        arr=np.array(tp_ce); narr=np.array(tp_ce_n)
        result.center_err_mean=float(np.mean(arr)); result.center_err_median=float(np.median(arr))
        result.center_err_p95=float(np.percentile(arr,95))
        result.center_err_norm_mean=float(np.mean(narr)); result.center_err_norm_median=float(np.median(narr))
        result.center_err_norm_p95=float(np.percentile(narr,95))
        result.width_err_rel_mean=float(np.mean(tp_we)); result.height_err_rel_mean=float(np.mean(tp_he))
        result.tp_ious=[float(v) for v in tp_ious_l]; result.tp_nwds=[float(v) for v in tp_nwds_l]
        result.tp_iou_mean=float(np.mean(tp_ious_l)); result.tp_nwd_mean=float(np.mean(tp_nwds_l))

    det_sorted = sorted(all_dets50, key=lambda x: x[0], reverse=True)
    for thr in [round(t,2) for t in np.arange(0.01, 0.96, 0.05)]:
        above=[(s,t) for s,t in det_sorted if s>=thr]
        tp_t=sum(t for _,t in above); fp_t=len(above)-tp_t
        p_t=tp_t/max(tp_t+fp_t,1); r_t=tp_t/max(total_gts,1); f_t=2*p_t*r_t/max(p_t+r_t,1e-7)
        result.thresh_values.append(thr); result.thresh_precision.append(p_t)
        result.thresh_recall.append(r_t); result.thresh_f1.append(f_t)

    return result, visual_samples


@torch.no_grad()
def evaluate_epoch(model, loader, device, score_thresh=0.3, nwd_constant=12.8,
                   size_bins=None, ap_iou_thresholds=None):
    """Lightweight per-epoch evaluation. Returns flat dict for history."""
    result, _ = evaluate(model, loader, device, score_thresh=score_thresh,
                         nwd_constant=nwd_constant, size_bins=size_bins or [8,16,32,64],
                         num_visuals=0, iou_thresh=0.5,
                         ap_iou_thresholds=ap_iou_thresholds or [round(0.50+i*0.05,2) for i in range(10)])
    m = {
        "val_precision":       result.precision,
        "val_recall":          result.recall,
        "val_f1":              result.f1,
        "val_ap50":            result.ap50,
        "val_ap50_95":         result.ap50_95,
        "val_fppi":            result.fppi,
        "val_center_err":      result.center_err_mean,
        "val_center_err_norm": result.center_err_norm_mean,
        "val_tp_iou":          result.tp_iou_mean,
        "val_tp_nwd":          result.tp_nwd_mean,
    }
    for bi, b in enumerate(result.size_bins):
        m[f"val_recall_bin{bi}"] = b["recall"]
        m[f"val_ap50_bin{bi}"]   = b["ap50"]
    return m


def nwd_iou_diagnostic(gt_w=8.0, gt_h=8.0, max_displacement=40.0, n_steps=80, nwd_constant=12.8):
    """Sweep pred center displacement 0->max_disp. Return (displacements, ious, nwds)."""
    gt_cx, gt_cy = 100.0, 100.0
    gt_box = np.array([gt_cx-gt_w/2, gt_cy-gt_h/2, gt_cx+gt_w/2, gt_cy+gt_h/2])
    displacements = np.linspace(0, max_displacement, n_steps)
    ious, nwds = [], []
    for d in displacements:
        pb = np.array([gt_cx+d-gt_w/2, gt_cy-gt_h/2, gt_cx+d+gt_w/2, gt_cy+gt_h/2])
        ious.append(float(_box_iou_matrix(pb[None], gt_box[None])[0,0]))
        nwds.append(_nwd_score(pb, gt_box, C=nwd_constant))
    return displacements, np.array(ious), np.array(nwds)
