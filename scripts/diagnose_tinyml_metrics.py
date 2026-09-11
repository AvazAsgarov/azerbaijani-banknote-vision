"""Diagnostic analysis script to evaluate genuine IoU-matched mAP50 on the test split.

Calculates strict Precision, Recall, False Positives, and F1-score across all
Azerbaijani banknote classes for TinyML edge models.
"""

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.core.config import ProjectPaths
from src.tinyml.dataset import ANCHORS_S16, ANCHORS_S32, BanknoteDataset, DENOMINATIONS
from src.tinyml.model import YOLOFastestV2
from src.tinyml.train import compute_box_iou, decode_predictions, non_max_suppression


def evaluate_genuine_map50(
    model_path: Path,
    data_dir: Optional[Path] = None,
    conf_thresh: float = 0.15,
    iou_thresh: float = 0.50,
) -> Dict[str, Any]:
    """Evaluates strict bounding box detection metrics on the test dataset.

    Args:
        model_path: Path to serialized PyTorch model checkpoint.
        data_dir: Path to processed dataset directory. Defaults to ProjectPaths.
        conf_thresh: Confidence score threshold for candidate detections.
        iou_thresh: IoU overlap threshold for true positive assignment.

    Returns:
        Dictionary mapping denomination names to their classification and detection scores.
    """
    paths = ProjectPaths()
    effective_data_dir = data_dir or paths.processed_data_dir

    if not model_path.exists():
        print(f"Notice: Model checkpoint not found at {model_path}. Skipping evaluation.")
        return {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating genuine metrics for: {model_path} on device: {device}", flush=True)

    model = YOLOFastestV2(num_classes=len(DENOMINATIONS)).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    test_dataset = BanknoteDataset(effective_data_dir, split="test", img_size=160, augment=False, cache_in_ram=True)
    dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    total_gt = {c: 0 for c in range(len(DENOMINATIONS))}
    true_positives = {c: 0 for c in range(len(DENOMINATIONS))}
    false_positives = {c: 0 for c in range(len(DENOMINATIONS))}
    total_predictions = {c: 0 for c in range(len(DENOMINATIONS))}

    with torch.no_grad():
        for imgs, target16, target32 in dataloader:
            imgs = imgs.to(device)
            p16, p32 = model(imgs)
            batch_preds = decode_predictions(p16, p32, conf_thresh=conf_thresh)
            batch_size = imgs.shape[0]

            for b in range(batch_size):
                # 1. Extract genuine GT boxes
                gt_boxes = []
                # Check S16
                t16_mask = (target16[b, ..., 4] == 1.0)
                if t16_mask.any():
                    ta, ty, tx = torch.where(t16_mask)
                    for i in range(len(ta)):
                        a = ta[i].item()
                        y = ty[i].item()
                        x = tx[i].item()
                        cls_id = int(torch.argmax(target16[b, a, y, x, 5:]).item())
                        xc = (x + target16[b, a, y, x, 0].item()) / 10.0
                        yc = (y + target16[b, a, y, x, 1].item()) / 10.0
                        w = (ANCHORS_S16[a][0] * math.exp(target16[b, a, y, x, 2].item())) / 160.0
                        h = (ANCHORS_S16[a][1] * math.exp(target16[b, a, y, x, 3].item())) / 160.0
                        gt_boxes.append((cls_id, [xc, yc, w, h]))
                        total_gt[cls_id] += 1

                # Check S32
                t32_mask = (target32[b, ..., 4] == 1.0)
                if t32_mask.any():
                    ta, ty, tx = torch.where(t32_mask)
                    for i in range(len(ta)):
                        a = ta[i].item()
                        y = ty[i].item()
                        x = tx[i].item()
                        cls_id = int(torch.argmax(target32[b, a, y, x, 5:]).item())
                        xc = (x + target32[b, a, y, x, 0].item()) / 5.0
                        yc = (y + target32[b, a, y, x, 1].item()) / 5.0
                        w = (ANCHORS_S32[a][0] * math.exp(target32[b, a, y, x, 2].item())) / 160.0
                        h = (ANCHORS_S32[a][1] * math.exp(target32[b, a, y, x, 3].item())) / 160.0
                        gt_boxes.append((cls_id, [xc, yc, w, h]))
                        total_gt[cls_id] += 1

                # 2. Extract NMS predictions
                preds = non_max_suppression(batch_preds[b], iou_thresh=0.45)
                matched_gt = set()

                for p in preds:
                    p_cls = int(p[5])
                    p_box = p[:4]
                    total_predictions[p_cls] += 1

                    best_match_idx = -1
                    best_match_iou = 0.0

                    for g_idx, (g_cls, g_box) in enumerate(gt_boxes):
                        if g_idx not in matched_gt and g_cls == p_cls:
                            iou = compute_box_iou(p_box, g_box)
                            if iou > best_match_iou:
                                best_match_iou = iou
                                best_match_idx = g_idx

                    if best_match_idx != -1 and best_match_iou >= iou_thresh:
                        true_positives[p_cls] += 1
                        matched_gt.add(best_match_idx)
                    else:
                        false_positives[p_cls] += 1

    print("\n--- Genuine Strict IoU-Matched Test Results (IoU >= 0.50) ---", flush=True)
    report = {}
    for c, denom in enumerate(DENOMINATIONS):
        tp = true_positives[c]
        fp = false_positives[c]
        tot = total_gt[c]
        pred_cnt = total_predictions[c]
        prec = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / float(tot) if tot > 0 else 0.0
        f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        report[denom] = {
            "GT_count": tot,
            "Predictions": pred_cnt,
            "True_Positives": tp,
            "False_Positives": fp,
            "Precision": prec,
            "Recall": rec,
            "F1_Score": f1
        }
        print(f"{denom:10s} | GT: {tot:3d} | Preds: {pred_cnt:3d} | TP: {tp:3d} | FP: {fp:3d} | Prec: {prec*100:5.1f}% | Rec: {rec*100:5.1f}% | F1: {f1*100:5.1f}%", flush=True)

    mean_f1 = sum(r["F1_Score"] for r in report.values()) / float(len(report))
    mean_prec = sum(r["Precision"] for r in report.values()) / float(len(report))
    mean_rec = sum(r["Recall"] for r in report.values()) / float(len(report))
    print(f"\nOverall Strict Mean Precision: {mean_prec*100:.2f}% | Mean Recall: {mean_rec*100:.2f}% | Mean F1: {mean_f1*100:.2f}%", flush=True)
    return report

if __name__ == "__main__":
    paths = ProjectPaths()
    parser = argparse.ArgumentParser(description="Evaluate genuine test mAP50 for TinyML models")
    parser.add_argument(
        "--model",
        type=Path,
        default=paths.artifacts_dir / "tinyml" / "yolo_fastestv2_best.pt",
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=paths.processed_data_dir,
        help="Path to processed data directory",
    )
    args = parser.parse_args()
    evaluate_genuine_map50(args.model, data_dir=args.data_dir)
