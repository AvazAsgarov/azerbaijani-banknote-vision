"""Multi-Task Asymmetric Cost-Sensitive Loss Function with Complete IoU (CIoU).

Tailored for Assistive Banknote Detection on ESP32-S3 microcontroller deployment.
Integrates Complete-IoU (CIoU) coordinate regression, balanced objectness with
downweighted background penalty, and asymmetric cost-sensitive classification.
"""

import math
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.tinyml.dataset import ANCHORS_S16, ANCHORS_S32


def bbox_ciou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Computes Complete IoU (CIoU) between two batches of boxes in (x1, y1, x2, y2) format.

    Args:
        box1: Tensor of shape (N, 4) containing [x1, y1, x2, y2].
        box2: Tensor of shape (N, 4) containing [x1, y1, x2, y2].
        eps: Small epsilon constant for numerical stability.

    Returns:
        Tensor of shape (N,) containing CIoU values clamped in [-1.0, 1.0].
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # Intersection rectangle
    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)
    inter_area = (inter_x2 - inter_x1).clamp(min=0.0) * (inter_y2 - inter_y1).clamp(min=0.0)

    # Box areas and union
    w1, h1 = (b1_x2 - b1_x1).clamp(min=eps), (b1_y2 - b1_y1).clamp(min=eps)
    w2, h2 = (b2_x2 - b2_x1).clamp(min=eps), (b2_y2 - b2_y1).clamp(min=eps)
    union_area = w1 * h1 + w2 * h2 - inter_area + eps
    iou = inter_area / union_area

    # Smallest enclosing box
    c_x1 = torch.min(b1_x1, b2_x1)
    c_y1 = torch.min(b1_y1, b2_y1)
    c_x2 = torch.max(b1_x2, b2_x2)
    c_y2 = torch.max(b1_y2, b2_y2)
    c_w = (c_x2 - c_x1).clamp(min=eps)
    c_h = (c_y2 - c_y1).clamp(min=eps)
    c2 = c_w ** 2 + c_h ** 2 + eps

    # Center distance
    b1_cx = (b1_x1 + b1_x2) / 2.0
    b1_cy = (b1_y1 + b1_y2) / 2.0
    b2_cx = (b2_x1 + b2_x2) / 2.0
    b2_cy = (b2_y1 + b2_y2) / 2.0
    rho2 = (b1_cx - b2_cx) ** 2 + (b1_cy - b2_cy) ** 2

    # Aspect ratio consistency
    v = (4.0 / (math.pi ** 2)) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
    with torch.no_grad():
        alpha = v / ((1.0 - iou) + v + eps)

    ciou = iou - (rho2 / c2 + alpha * v)
    return ciou.clamp(min=-1.0, max=1.0)


class AsymmetricBanknoteLoss(nn.Module):
    """Computes loss across dual detection heads with CIoU and asymmetric classification penalty.

    Attributes:
        num_classes: Number of detection target classes.
        lambda_box: Weight multiplier for bounding box CIoU loss.
        lambda_obj: Weight multiplier for objectness loss.
        lambda_noobj: Downweighting multiplier for background objectness cells.
        lambda_cls: Weight multiplier for asymmetric classification loss.
        beta_c: Escalating penalty buffer for high-denomination false positives.
    """

    def __init__(
        self,
        num_classes: int = 7,
        lambda_box: float = 7.5,
        lambda_obj: float = 1.0,
        lambda_noobj: float = 0.25,
        lambda_cls: float = 0.75,
    ) -> None:
        """Initialize the asymmetric cost-sensitive loss.

        Args:
            num_classes: Number of detection target classes.
            lambda_box: Weight multiplier for bounding box regression loss.
            lambda_obj: Weight multiplier for objectness loss.
            lambda_noobj: Downweighting factor for background objectness cells.
            lambda_cls: Weight multiplier for asymmetric classification loss.
        """
        super().__init__()
        self.num_classes = num_classes
        self.lambda_box = lambda_box
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.lambda_cls = lambda_cls

        # False-positive penalty multiplier beta_c:
        # Higher values penalize false positives on high-value denominations
        self.register_buffer(
            "beta_c",
            torch.tensor([1.0, 1.0, 2.0, 2.0, 4.0, 6.0, 6.0], dtype=torch.float32)
        )
        self.register_buffer("anchors_16", torch.tensor(ANCHORS_S16, dtype=torch.float32))
        self.register_buffer("anchors_32", torch.tensor(ANCHORS_S32, dtype=torch.float32))

    def _compute_head_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        grid_size: Optional[int] = None,
        anchors: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute loss for a single detection head using CIoU.

        Args:
            pred: Predicted tensor of shape (B, num_anchors, H, W, 5 + num_classes).
            target: Ground truth tensor of shape (B, num_anchors, H, W, 5 + num_classes).
            grid_size: Dimension of the spatial grid (10 or 5).
            anchors: Tensor of anchor box dimensions for this head (3, 2).

        Returns:
            Tuple of (total_head_loss, loss_box, loss_obj, loss_cls).
        """
        device = pred.device
        if grid_size is None:
            grid_size = pred.shape[2]
        if anchors is None:
            if grid_size == 10:
                anchors = self.anchors_16
            elif grid_size == 5:
                anchors = self.anchors_32
            else:
                num_a = pred.shape[1]
                anchors = torch.ones((num_a, 2), device=device, dtype=torch.float32) * 50.0
        obj_mask = (target[..., 4] == 1.0)
        noobj_mask = (target[..., 4] == 0.0)

        # 1. CIoU Bounding Box Regression Loss
        loss_box = torch.tensor(0.0, device=device)
        if obj_mask.sum() > 0:
            b_idx, a_idx, y_idx, x_idx = torch.where(obj_mask)

            # Predicted coordinates
            tx = torch.sigmoid(pred[b_idx, a_idx, y_idx, x_idx, 0])
            ty = torch.sigmoid(pred[b_idx, a_idx, y_idx, x_idx, 1])
            tw = torch.clamp(pred[b_idx, a_idx, y_idx, x_idx, 2], -4.0, 4.0)
            th = torch.clamp(pred[b_idx, a_idx, y_idx, x_idx, 3], -4.0, 4.0)

            p_xc = (x_idx.float() + tx) / float(grid_size)
            p_yc = (y_idx.float() + ty) / float(grid_size)
            anch = anchors[a_idx]
            p_w = (anch[:, 0] * torch.exp(tw)) / 160.0
            p_h = (anch[:, 1] * torch.exp(th)) / 160.0

            p_x1 = p_xc - p_w / 2.0
            p_y1 = p_yc - p_h / 2.0
            p_x2 = p_xc + p_w / 2.0
            p_y2 = p_yc + p_h / 2.0
            pred_boxes = torch.stack([p_x1, p_y1, p_x2, p_y2], dim=-1)

            # Target coordinates
            t_tx = target[b_idx, a_idx, y_idx, x_idx, 0]
            t_ty = target[b_idx, a_idx, y_idx, x_idx, 1]
            t_tw = target[b_idx, a_idx, y_idx, x_idx, 2]
            t_th = target[b_idx, a_idx, y_idx, x_idx, 3]

            t_xc = (x_idx.float() + t_tx) / float(grid_size)
            t_yc = (y_idx.float() + t_ty) / float(grid_size)
            t_w = (anch[:, 0] * torch.exp(t_tw)) / 160.0
            t_h = (anch[:, 1] * torch.exp(t_th)) / 160.0

            t_x1 = t_xc - t_w / 2.0
            t_y1 = t_yc - t_h / 2.0
            t_x2 = t_xc + t_w / 2.0
            t_y2 = t_yc + t_h / 2.0
            target_boxes = torch.stack([t_x1, t_y1, t_x2, t_y2], dim=-1)

            ciou = bbox_ciou(pred_boxes, target_boxes)
            loss_box = (1.0 - ciou).mean()

        # 2. Balanced Objectness Loss with Forceful Background Suppression
        pred_obj = pred[..., 4]
        target_obj = target[..., 4]

        bce_obj = F.binary_cross_entropy_with_logits(pred_obj, target_obj, reduction="none")

        loss_obj_pos = torch.tensor(0.0, device=device)
        if obj_mask.sum() > 0:
            loss_obj_pos = bce_obj[obj_mask].sum() / max(1.0, float(obj_mask.sum()))

        loss_obj_neg = torch.tensor(0.0, device=device)
        if noobj_mask.sum() > 0:
            # Normalize negative objectness loss by batch size rather than by the total count
            # of negative cells (~374 per image), preventing gradient dilution so background
            # objectness is actively driven to zero.
            loss_obj_neg = bce_obj[noobj_mask].sum() / (float(pred.shape[0]) * 10.0)

        loss_obj = loss_obj_pos + self.lambda_noobj * loss_obj_neg

        # 3. Asymmetric Classification Loss
        loss_cls = torch.tensor(0.0, device=device)
        if obj_mask.sum() > 0:
            pred_cls_logits = pred[..., 5:][obj_mask]
            target_cls = target[..., 5:][obj_mask]

            pred_probs = torch.sigmoid(pred_cls_logits)
            p_t = torch.clamp(pred_probs, 1e-6, 1.0 - 1e-6)

            # Focal loss modulation: (1 - p)^gamma
            loss_pos = - target_cls * ((1.0 - p_t) ** 1.5) * torch.log(p_t)
            beta = self.beta_c.unsqueeze(0)
            loss_neg = - beta * (1.0 - target_cls) * (p_t ** 1.5) * torch.log(1.0 - p_t)

            loss_cls = (loss_pos + loss_neg).sum(dim=-1).mean()

        total = (
            self.lambda_box * loss_box +
            self.lambda_obj * loss_obj +
            self.lambda_cls * loss_cls
        )
        return total, loss_box, loss_obj, loss_cls

    def forward(
        self,
        pred_s16: torch.Tensor,
        pred_s32: torch.Tensor,
        target_s16: torch.Tensor,
        target_s32: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Execute loss computation across both feature scales.

        Args:
            pred_s16: Predictions from stride 16 head (B, 3, 10, 10, 12).
            pred_s32: Predictions from stride 32 head (B, 3, 5, 5, 12).
            target_s16: Targets for stride 16 head (B, 3, 10, 10, 12).
            target_s32: Targets for stride 32 head (B, 3, 5, 5, 12).

        Returns:
            Tuple of (total_loss, metrics_dictionary).
        """
        loss16, box16, obj16, cls16 = self._compute_head_loss(
            pred_s16, target_s16, grid_size=10, anchors=self.anchors_16
        )
        loss32, box32, obj32, cls32 = self._compute_head_loss(
            pred_s32, target_s32, grid_size=5, anchors=self.anchors_32
        )

        total_loss = loss16 + loss32
        metrics = {
            "loss_total": total_loss.item(),
            "loss_box": (box16 + box32).item(),
            "loss_obj": (obj16 + obj32).item(),
            "loss_cls": (cls16 + cls32).item()
        }
        return total_loss, metrics
