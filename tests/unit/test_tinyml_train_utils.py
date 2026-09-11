"""
Unit tests for TinyML training utilities, bounding box decoders, and metrics.
Verifies box IoU calculations, NMS filtering, and Pascal VOC AP integration.
"""

import pytest
import numpy as np
import torch

from src.tinyml.train import (
    compute_box_iou,
    non_max_suppression,
    compute_voc_ap,
    decode_predictions
)


@pytest.mark.unit
class TestBoundingBoxUtilities:
    """Test suite verifying box IoU and non-maximum suppression algorithms."""

    def test_compute_box_iou_identical_boxes(self):
        """Verify IoU between identical bounding boxes equals 1.0.

        Args:
            None.

        Returns:
            None.
        """
        box = [0.5, 0.5, 0.4, 0.4]
        iou = compute_box_iou(box, box)
        assert pytest.approx(iou, 1e-4) == 1.0

    def test_compute_box_iou_disjoint_boxes(self):
        """Verify IoU between non-overlapping bounding boxes equals 0.0.

        Args:
            None.

        Returns:
            None.
        """
        b1 = [0.2, 0.2, 0.1, 0.1]
        b2 = [0.8, 0.8, 0.1, 0.1]
        iou = compute_box_iou(b1, b2)
        assert iou == 0.0

    def test_non_max_suppression_eliminates_duplicates(self):
        """Verify NMS suppresses overlapping candidate boxes with lower confidence scores.

        Args:
            None.

        Returns:
            None.
        """
        dets = [
            [0.50, 0.50, 0.40, 0.40, 0.90, 0, 0.95],
            [0.51, 0.51, 0.39, 0.39, 0.70, 0, 0.80],
            [0.80, 0.80, 0.10, 0.10, 0.85, 1, 0.90]
        ]
        kept = non_max_suppression(dets, iou_thresh=0.45)

        assert len(kept) == 2
        assert kept[0][4] == 0.90
        assert kept[1][4] == 0.85


@pytest.mark.unit
class TestEvaluationMetrics:
    """Test suite verifying Pascal VOC / COCO AP50 calculation logic."""

    def test_compute_voc_ap_perfect_detection(self):
        """Verify Average Precision equals 1.0 under flawless true positive predictions.

        Args:
            None.

        Returns:
            None.
        """
        tp = [1, 1, 1, 1]
        fp = [0, 0, 0, 0]
        ap = compute_voc_ap(tp, fp, total_gt=4)
        assert pytest.approx(ap, 1e-4) == 1.0

    def test_compute_voc_ap_zero_true_positives(self):
        """Verify Average Precision equals 0.0 when zero predictions match ground truth targets.

        Args:
            None.

        Returns:
            None.
        """
        tp = [0, 0, 0]
        fp = [1, 1, 1]
        ap = compute_voc_ap(tp, fp, total_gt=5)
        assert ap == 0.0

    def test_compute_voc_ap_empty_inputs(self):
        """Verify Average Precision handles empty detection lists gracefully without dividing by zero.

        Args:
            None.

        Returns:
            None.
        """
        assert compute_voc_ap([], [], total_gt=5) == 0.0
        assert compute_voc_ap([1], [0], total_gt=0) == 0.0
