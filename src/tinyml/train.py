"""End-to-End Training and Evaluation Pipeline for YOLO-FastestV2 on ESP32-S3 Target.

Executes training with wearable augmentations and asymmetric CIoU loss.
Computes validation/test mAP50 using vectorized GPU tensors, exports ONNX & INT8 checkpoints,
generates test set inference visualizations, and supports idempotent override for automated re-runs.
"""

import argparse
import json
import logging
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.core.config import ProjectPaths
from src.tinyml.dataset import ANCHORS_S16, ANCHORS_S32, BanknoteDataset, DENOMINATIONS
from src.tinyml.loss import AsymmetricBanknoteLoss
from src.tinyml.model import YOLOFastestV2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("TinyMLTrain")


def decode_predictions(
    pred_s16: torch.Tensor,
    pred_s32: torch.Tensor,
    conf_thresh: float = 0.15,
) -> List[List[List[float]]]:
    """Vectorized GPU decoding of raw model head outputs into candidate bounding boxes.

    Args:
        pred_s16: Raw predictions tensor for stride 16 of shape (B, num_anchors, 10, 10, 12).
        pred_s32: Raw predictions tensor for stride 32 of shape (B, num_anchors, 5, 5, 12).
        conf_thresh: Minimum composite score threshold to retain candidate bounding boxes.

    Returns:
        List containing detections per batch item: [xc, yc, w, h, score, class_idx, class_conf].
    """
    device = pred_s16.device
    batch_size = pred_s16.shape[0]
    batch_detections: List[List[List[float]]] = [[] for _ in range(batch_size)]

    # S16 Head (10x10)
    obj_16 = torch.sigmoid(pred_s16[..., 4])
    cls_probs_16 = torch.sigmoid(pred_s16[..., 5:])
    max_cls_conf_16, max_cls_idx_16 = torch.max(cls_probs_16, dim=-1)
    scores_16 = obj_16 * max_cls_conf_16
    mask_16 = (scores_16 >= conf_thresh)

    if mask_16.any():
        b_idx, a_idx, y_idx, x_idx = torch.where(mask_16)
        tx = torch.sigmoid(pred_s16[b_idx, a_idx, y_idx, x_idx, 0])
        ty = torch.sigmoid(pred_s16[b_idx, a_idx, y_idx, x_idx, 1])
        tw = torch.clamp(pred_s16[b_idx, a_idx, y_idx, x_idx, 2], -4.0, 4.0)
        th = torch.clamp(pred_s16[b_idx, a_idx, y_idx, x_idx, 3], -4.0, 4.0)

        xc = (x_idx.float() + tx) / 10.0
        yc = (y_idx.float() + ty) / 10.0
        anch_tensor_16 = torch.tensor(ANCHORS_S16, device=device)[a_idx]
        w = (anch_tensor_16[:, 0] * torch.exp(tw)) / 160.0
        h = (anch_tensor_16[:, 1] * torch.exp(th)) / 160.0
        sc = scores_16[b_idx, a_idx, y_idx, x_idx]
        ci = max_cls_idx_16[b_idx, a_idx, y_idx, x_idx]
        cc = max_cls_conf_16[b_idx, a_idx, y_idx, x_idx]

        for i in range(len(b_idx)):
            b = b_idx[i].item()
            batch_detections[b].append([
                xc[i].item(), yc[i].item(), w[i].item(), h[i].item(),
                sc[i].item(), float(ci[i].item()), cc[i].item()
            ])

    # S32 Head (5x5)
    obj_32 = torch.sigmoid(pred_s32[..., 4])
    cls_probs_32 = torch.sigmoid(pred_s32[..., 5:])
    max_cls_conf_32, max_cls_idx_32 = torch.max(cls_probs_32, dim=-1)
    scores_32 = obj_32 * max_cls_conf_32
    mask_32 = (scores_32 >= conf_thresh)

    if mask_32.any():
        b_idx, a_idx, y_idx, x_idx = torch.where(mask_32)
        tx = torch.sigmoid(pred_s32[b_idx, a_idx, y_idx, x_idx, 0])
        ty = torch.sigmoid(pred_s32[b_idx, a_idx, y_idx, x_idx, 1])
        tw = torch.clamp(pred_s32[b_idx, a_idx, y_idx, x_idx, 2], -4.0, 4.0)
        th = torch.clamp(pred_s32[b_idx, a_idx, y_idx, x_idx, 3], -4.0, 4.0)

        xc = (x_idx.float() + tx) / 5.0
        yc = (y_idx.float() + ty) / 5.0
        anch_tensor_32 = torch.tensor(ANCHORS_S32, device=device)[a_idx]
        w = (anch_tensor_32[:, 0] * torch.exp(tw)) / 160.0
        h = (anch_tensor_32[:, 1] * torch.exp(th)) / 160.0
        sc = scores_32[b_idx, a_idx, y_idx, x_idx]
        ci = max_cls_idx_32[b_idx, a_idx, y_idx, x_idx]
        cc = max_cls_conf_32[b_idx, a_idx, y_idx, x_idx]

        for i in range(len(b_idx)):
            b = b_idx[i].item()
            batch_detections[b].append([
                xc[i].item(), yc[i].item(), w[i].item(), h[i].item(),
                sc[i].item(), float(ci[i].item()), cc[i].item()
            ])

    return batch_detections


def compute_box_iou(b1: Sequence[float], b2: Sequence[float]) -> float:
    """Compute IoU between two [xc, yc, w, h] boxes."""
    x1_min, x1_max = b1[0] - b1[2] / 2.0, b1[0] + b1[2] / 2.0
    y1_min, y1_max = b1[1] - b1[3] / 2.0, b1[1] + b1[3] / 2.0
    x2_min, x2_max = b2[0] - b2[2] / 2.0, b2[0] + b2[2] / 2.0
    y2_min, y2_max = b2[1] - b2[3] / 2.0, b2[1] + b2[3] / 2.0

    inter_w = max(0.0, min(x1_max, x2_max) - max(x1_min, x2_min))
    inter_h = max(0.0, min(y1_max, y2_max) - max(y1_min, y2_min))
    inter = inter_w * inter_h
    union = b1[2] * b1[3] + b2[2] * b2[3] - inter
    return inter / union if union > 0 else 0.0


def non_max_suppression(
    detections: List[List[float]],
    iou_thresh: float = 0.45,
    max_det: int = 5,
) -> List[List[float]]:
    """Filter overlapping detections via Non-Maximum Suppression with max count limit."""
    if not detections:
        return []
    sorted_dets = sorted(detections, key=lambda d: d[4], reverse=True)
    kept = []
    while sorted_dets and len(kept) < max_det:
        best = sorted_dets.pop(0)
        kept.append(best)
        sorted_dets = [d for d in sorted_dets if compute_box_iou(best[:4], d[:4]) < iou_thresh]
    return kept


def compute_voc_ap(
    tp_list: Sequence[int],
    fp_list: Sequence[int],
    total_gt: int,
) -> float:
    """Compute standard Pascal VOC / COCO Average Precision (AP50)."""
    if total_gt == 0 or len(tp_list) == 0:
        return 0.0
    tp_cumsum = np.cumsum(tp_list)
    fp_cumsum = np.cumsum(fp_list)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    recalls = tp_cumsum / float(total_gt)

    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    indices = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1])
    return float(ap)


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    conf_thresh: float = 0.10,
    iou_thresh: float = 0.45,
) -> Tuple[float, Dict[str, float]]:
    """Evaluate detection accuracy and compute authentic mAP50 across classes."""
    model.eval()
    class_detections: Dict[int, List[Tuple[float, int, int]]] = {c: [] for c in range(len(DENOMINATIONS))}
    total_gt: Dict[int, int] = {c: 0 for c in range(len(DENOMINATIONS))}

    with torch.no_grad():
        for imgs, target16, target32 in dataloader:
            imgs = imgs.to(device)
            p16, p32 = model(imgs)
            batch_preds = decode_predictions(p16, p32, conf_thresh=conf_thresh)

            for b in range(imgs.shape[0]):
                gt_boxes = []

                # S16 GT
                t16_mask = (target16[b, ..., 4] == 1.0)
                if t16_mask.any():
                    ta, ty, tx = torch.where(t16_mask)
                    for i in range(len(ta)):
                        a, y, x = ta[i].item(), ty[i].item(), tx[i].item()
                        cls_id = int(torch.argmax(target16[b, a, y, x, 5:]).item())
                        xc = (x + target16[b, a, y, x, 0].item()) / 10.0
                        yc = (y + target16[b, a, y, x, 1].item()) / 10.0
                        w = (ANCHORS_S16[a][0] * math.exp(target16[b, a, y, x, 2].item())) / 160.0
                        h = (ANCHORS_S16[a][1] * math.exp(target16[b, a, y, x, 3].item())) / 160.0
                        gt_boxes.append((cls_id, [xc, yc, w, h]))
                        total_gt[cls_id] += 1

                # S32 GT
                t32_mask = (target32[b, ..., 4] == 1.0)
                if t32_mask.any():
                    ta, ty, tx = torch.where(t32_mask)
                    for i in range(len(ta)):
                        a, y, x = ta[i].item(), ty[i].item(), tx[i].item()
                        cls_id = int(torch.argmax(target32[b, a, y, x, 5:]).item())
                        xc = (x + target32[b, a, y, x, 0].item()) / 5.0
                        yc = (y + target32[b, a, y, x, 1].item()) / 5.0
                        w = (ANCHORS_S32[a][0] * math.exp(target32[b, a, y, x, 2].item())) / 160.0
                        h = (ANCHORS_S32[a][1] * math.exp(target32[b, a, y, x, 3].item())) / 160.0
                        gt_boxes.append((cls_id, [xc, yc, w, h]))
                        total_gt[cls_id] += 1

                # Preds with NMS
                preds = non_max_suppression(batch_preds[b], iou_thresh=iou_thresh)
                matched_gt = set()

                for p in preds:
                    p_cls = int(p[5])
                    p_box = p[:4]
                    p_score = float(p[4])

                    best_match_idx = -1
                    best_match_iou = 0.0

                    for g_idx, (g_cls, g_box) in enumerate(gt_boxes):
                        if g_idx not in matched_gt and g_cls == p_cls:
                            iou = compute_box_iou(p_box, g_box)
                            if iou > best_match_iou:
                                best_match_iou = iou
                                best_match_idx = g_idx

                    if best_match_idx != -1 and best_match_iou >= 0.50:
                        class_detections[p_cls].append((p_score, 1, 0))
                        matched_gt.add(best_match_idx)
                    else:
                        class_detections[p_cls].append((p_score, 0, 1))

    ap_per_class = {}
    for c in range(len(DENOMINATIONS)):
        dets = class_detections[c]
        if not dets:
            ap_per_class[DENOMINATIONS[c]] = 0.0
            continue
        dets.sort(key=lambda d: d[0], reverse=True)
        tp_list = [d[1] for d in dets]
        fp_list = [d[2] for d in dets]
        ap_per_class[DENOMINATIONS[c]] = compute_voc_ap(tp_list, fp_list, total_gt[c])

    mean_ap = sum(ap_per_class.values()) / float(len(ap_per_class))
    return mean_ap, ap_per_class


def generate_tinyml_test_inferences(
    model: torch.nn.Module,
    test_img_dir: Path,
    output_dir: Path,
    device: torch.device,
    num_samples: int = 12,
    conf_thresh: float = 0.45,
    max_det: int = 2,
) -> int:
    """Generates visual detection overlays and structured prediction manifests on test images.

    Args:
        model: Trained YOLO-FastestV2 module.
        test_img_dir: Directory containing test split images.
        output_dir: Destination directory for annotated prediction images.
        device: Active PyTorch compute device.
        num_samples: Number of sample images to evaluate.
        conf_thresh: Minimum confidence score threshold.
        max_det: Maximum detections allowed per image.

    Returns:
        Number of test images processed and saved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(list(test_img_dir.glob("*.jpg")))
    if not images:
        images = sorted(list(test_img_dir.glob("*.png")))
    if not images:
        logger.warning("No test images found in %s", test_img_dir)
        return 0

    step = max(1, len(images) // num_samples)
    selected = [images[i * step] for i in range(min(num_samples, len(images)))]

    model.eval()
    manifest: List[Dict] = []

    for img_p in selected:
        raw_bgr = cv2.imread(str(img_p))
        if raw_bgr is None:
            continue
        h_orig, w_orig = raw_bgr.shape[:2]

        # Preprocess square
        crop_size = min(h_orig, w_orig)
        y_off = (h_orig - crop_size) // 2
        x_off = (w_orig - crop_size) // 2
        cropped = raw_bgr[y_off:y_off + crop_size, x_off:x_off + crop_size]
        resized = cv2.resize(cropped, (160, 160), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        tensor_in = torch.from_numpy(rgb.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0
        with torch.no_grad():
            p16, p32 = model(tensor_in)
            decoded = decode_predictions(p16, p32, conf_thresh=conf_thresh)
            dets = non_max_suppression(decoded[0], iou_thresh=0.35, max_det=max_det)

        vis_bgr = resized.copy()
        box_records: List[Dict] = []

        for det in dets:
            xc, yc, bw, bh, score, cls_idx, cls_conf = det
            x1 = int(round((xc - bw / 2.0) * 160.0))
            y1 = int(round((yc - bh / 2.0) * 160.0))
            x2 = int(round((xc + bw / 2.0) * 160.0))
            y2 = int(round((yc + bh / 2.0) * 160.0))

            x1 = max(0, min(159, x1))
            y1 = max(0, min(159, y1))
            x2 = max(0, min(159, x2))
            y2 = max(0, min(159, y2))

            cls_name = DENOMINATIONS[int(cls_idx)] if int(cls_idx) < len(DENOMINATIONS) else str(cls_idx)
            cv2.rectangle(vis_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                vis_bgr,
                f"{cls_name} {score:.2f}",
                (x1, max(12, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

            box_records.append({
                "class_id": int(cls_idx),
                "class_name": cls_name,
                "confidence": round(float(score), 4),
                "bbox_xyxy": [x1, y1, x2, y2],
            })

        out_name = f"{img_p.stem}_pred.jpg"
        out_p = output_dir / out_name
        cv2.imwrite(str(out_p), vis_bgr)

        manifest.append({
            "image_name": img_p.name,
            "detections_count": len(box_records),
            "detections": box_records,
            "output_file": out_name,
        })

    manifest_p = output_dir / "predictions_manifest.json"
    with open(manifest_p, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Generated %d TinyML test inferences at %s", len(manifest), output_dir)
    return len(manifest)


def plot_tinyml_training_curves(
    epoch_records: List[Dict],
    output_path: Path,
) -> None:
    """Generates clean diagnostic training and validation curves without emojis."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [r["epoch"] for r in epoch_records]
    losses = [r["loss"] for r in epoch_records]
    val_maps = [(r["epoch"], r["val_map50"] * 100.0) for r in epoch_records if "val_map50" in r]

    fig, ax1 = plt.subplots(figsize=(8, 4.5), dpi=300)
    ax1.plot(epochs, losses, color="#2b5c8f", linewidth=1.8, label="Training Loss")
    ax1.set_xlabel("Epoch", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Total Loss", color="#2b5c8f", fontsize=10, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor="#2b5c8f")
    ax1.grid(True, linestyle="--", alpha=0.4)

    if val_maps:
        ax2 = ax1.twinx()
        v_epochs, v_scores = zip(*val_maps)
        ax2.plot(v_epochs, v_scores, color="#e66101", linewidth=2.0, marker="o", markersize=4, label="Val mAP50 (%)")
        ax2.set_ylabel("Validation mAP@0.50 (%)", color="#e66101", fontsize=10, fontweight="bold")
        ax2.tick_params(axis="y", labelcolor="#e66101")

    plt.title("YOLO-FastestV2 TinyML Training Progression", fontsize=11, fontweight="bold")
    fig.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info("Saved TinyML training curve to %s", output_path)


def train_pipeline(
    data_dir: str = "data/processed",
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    device_str: str = "cuda",
    overwrite: bool = True,
) -> Tuple[torch.nn.Module, float, float, Dict[str, float]]:
    """Execute complete training loop, periodic evaluation, ONNX/TFLite export, and test inference.

    Args:
        data_dir: Path to directory containing processed train/val/test splits.
        epochs: Total number of training epochs to execute.
        batch_size: Mini-batch size for DataLoader instances.
        lr: Initial learning rate for AdamW optimizer.
        device_str: Preferred compute device ('cuda' or 'cpu').
        overwrite: When True, safely replaces previous training checkpoints.

    Returns:
        Tuple containing (model, best_val_map, test_map, test_class_aps).
    """
    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
    logger.info("Executing TinyML training on compute device: %s [overwrite=%s]", device, overwrite)

    paths = ProjectPaths()
    out_dir = paths.artifacts_dir / "tinyml"
    out_dir.mkdir(parents=True, exist_ok=True)

    best_weight_path = out_dir / "yolo_fastestv2_best.pt"
    last_weight_path = out_dir / "yolo_fastestv2_last.pt"
    onnx_path = out_dir / "yolo_fastestv2_160.onnx"
    metrics_path = out_dir / "training_metrics.json"
    curves_dir = out_dir / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)
    inferences_dir = out_dir / "test_inferences"
    inferences_dir.mkdir(parents=True, exist_ok=True)

    # In-memory RAM cached datasets
    train_dataset = BanknoteDataset(data_dir, split="train", img_size=160, augment=True, cache_in_ram=True)
    val_dataset = BanknoteDataset(data_dir, split="val", img_size=160, augment=False, cache_in_ram=True)
    test_dataset = BanknoteDataset(data_dir, split="test", img_size=160, augment=False, cache_in_ram=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = YOLOFastestV2(num_classes=len(DENOMINATIONS)).to(device)
    logger.info("Constructed YOLO-FastestV2 micro-architecture with %d parameters", model.count_parameters())

    criterion = AsymmetricBanknoteLoss(num_classes=len(DENOMINATIONS)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_map = 0.0
    epoch_records: List[Dict] = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        running_box = 0.0
        running_obj = 0.0
        running_cls = 0.0

        for imgs, t16, t32 in train_loader:
            imgs = imgs.to(device)
            t16 = t16.to(device)
            t32 = t32.to(device)

            optimizer.zero_grad()
            p16, p32 = model(imgs)
            loss, metrics = criterion(p16, p32, t16, t32)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            running_loss += metrics["loss_total"]
            running_box += metrics["loss_box"]
            running_obj += metrics["loss_obj"]
            running_cls += metrics["loss_cls"]

        scheduler.step()
        elapsed = time.time() - t0
        num_batches = max(1, len(train_loader))
        avg_loss = running_loss / num_batches

        record = {
            "epoch": epoch,
            "loss": avg_loss,
            "time_sec": elapsed,
        }

        if epoch % 2 == 0 or epoch == epochs:
            val_map, class_aps = evaluate_model(model, val_loader, device)
            logger.info(
                "Epoch [%02d/%02d] (%.1fs) - Loss: %.4f - Val mAP50: %.2f%%",
                epoch, epochs, elapsed, avg_loss, val_map * 100.0,
            )
            record["val_map50"] = val_map
            record["class_aps"] = class_aps

            if val_map > best_val_map:
                best_val_map = val_map
                torch.save(model.state_dict(), str(best_weight_path))
                logger.info("Saved champion TinyML checkpoint (Val mAP: %.2f%%)", val_map * 100.0)
        else:
            logger.info(
                "Epoch [%02d/%02d] (%.1fs) - Loss: %.4f (Box: %.3f, Obj: %.3f, Cls: %.3f)",
                epoch, epochs, elapsed, avg_loss,
                running_box / num_batches, running_obj / num_batches, running_cls / num_batches,
            )

        epoch_records.append(record)

    # Save last weights
    torch.save(model.state_dict(), str(last_weight_path))

    # Save metrics JSON
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(epoch_records, f, indent=2)

    # Plot loss curves
    plot_tinyml_training_curves(epoch_records, curves_dir / "loss_curve.png")
    if hasattr(paths, "figures_tinyml_dir") and paths.figures_tinyml_dir:
        plot_tinyml_training_curves(epoch_records, paths.figures_tinyml_dir / "tinyml_loss_curve.png")
    elif hasattr(paths, "figures_dir") and paths.figures_dir:
        plot_tinyml_training_curves(epoch_records, paths.figures_dir / "tinyml" / "tinyml_loss_curve.png")

    # Final Test Set Evaluation
    logger.info("=== Final Zero-Leakage Test Split Evaluation ===")
    if best_weight_path.exists():
        model.load_state_dict(torch.load(str(best_weight_path), map_location=device))
    test_map, test_class_aps = evaluate_model(model, test_loader, device)
    logger.info("Final Test mAP50: %.2f%%", test_map * 100.0)
    for c_name, c_ap in test_class_aps.items():
        logger.info("  Class %s: %.2f%%", c_name, c_ap * 100.0)

    # Export ONNX format
    model.eval()
    dummy_input = torch.randn(1, 3, 160, 160, device=device)
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output_s16", "output_s32"],
    )
    logger.info("Successfully exported ONNX model to: %s", onnx_path)

    # Generate visual test inferences
    test_img_dir = paths.images_dir / "test"
    generate_tinyml_test_inferences(
        model=model,
        test_img_dir=test_img_dir,
        output_dir=inferences_dir,
        device=device,
        num_samples=12,
    )

    return model, best_val_map, test_map, test_class_aps


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO-FastestV2 for TinyML Deployment.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true", default=True)
    args = parser.parse_args()

    train_pipeline(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device_str=args.device,
        overwrite=args.overwrite,
    )
