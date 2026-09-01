"""Unit tests for k-means anchor optimizer."""
import pytest
from src.tinyml.anchor_optimizer import kmeans_anchors, AnchorSet

BOXES = [(0.3,0.15),(0.32,0.14),(0.31,0.16),(0.5,0.25),(0.52,0.24),(0.51,0.26),
         (0.7,0.35),(0.72,0.34),(0.71,0.36),(0.2,0.10),(0.22,0.09),(0.21,0.11),
         (0.6,0.30),(0.62,0.29),(0.61,0.31),(0.4,0.20),(0.42,0.19),(0.41,0.21)]

class TestKMeansAnchors:
    def test_returns_k_anchors(self):
        assert len(kmeans_anchors(BOXES, k=6).anchors) == 6
    def test_avg_iou_valid(self):
        assert 0.0 < kmeans_anchors(BOXES, k=6).avg_iou <= 1.0
    def test_anchors_sorted_by_area(self):
        result = kmeans_anchors(BOXES, k=6)
        areas = [w*h for w,h in result.anchors]
        assert areas == sorted(areas)
    def test_reproducible(self):
        assert kmeans_anchors(BOXES, k=3, seed=42).anchors == kmeans_anchors(BOXES, k=3, seed=42).anchors
    def test_raises_on_insufficient_boxes(self):
        with pytest.raises(ValueError):
            kmeans_anchors([(0.5, 0.5)], k=6)
