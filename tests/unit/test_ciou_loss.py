"""Unit tests for CIoU regression loss."""
import pytest
from src.tinyml.ciou_loss import ciou_loss, _box_iou

class TestBoxIoU:
    def test_identical_boxes(self):
        assert _box_iou((0.5,0.5,0.4,0.2), (0.5,0.5,0.4,0.2)) == pytest.approx(1.0, abs=1e-5)
    def test_symmetric(self):
        a, b = (0.5,0.5,0.4,0.2), (0.6,0.5,0.3,0.15)
        assert _box_iou(a,b) == pytest.approx(_box_iou(b,a), abs=1e-5)

class TestCIoULoss:
    def test_zero_for_perfect(self):
        box = (0.5,0.5,0.3,0.2)
        assert ciou_loss(box, box) == pytest.approx(0.0, abs=1e-5)
    def test_positive_for_imperfect(self):
        assert ciou_loss((0.5,0.5,0.4,0.2), (0.6,0.6,0.3,0.15)) > 0.0
