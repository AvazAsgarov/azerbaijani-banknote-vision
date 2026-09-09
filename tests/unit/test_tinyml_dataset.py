"""
Unit tests for TinyML dataset loading, anchor matching, and image augmentations.
Verifies augmentation dimensions, target matrix encodings, and numerical safety.
"""

import pytest
import numpy as np
import cv2

from src.tinyml.dataset import (
    compute_anchor_iou,
    apply_motion_blur,
    apply_cutout,
    crop_and_resize_with_boxes,
    BanknoteDataset,
    ANCHORS_S16,
    ANCHORS_S32,
    DENOMINATIONS
)


@pytest.mark.unit
class TestDatasetAugmentationsAndGeometry:
    """Test suite verifying optical augmentations and anchor calculations."""

    def test_compute_anchor_iou_exact_match(self):
        """Verify IoU returns 1.0 when bounding box dimensions match anchor dimensions exactly.

        Args:
            None.

        Returns:
            None.
        """
        iou = compute_anchor_iou((100.0, 50.0), np.array([100.0, 50.0]))
        assert pytest.approx(iou, 1e-4) == 1.0

    def test_compute_anchor_iou_disjoint_or_zero(self):
        """Verify IoU returns 0.0 when dimensions collapse to zero.

        Args:
            None.

        Returns:
            None.
        """
        iou = compute_anchor_iou((0.0, 0.0), np.array([100.0, 50.0]))
        assert iou == 0.0

    def test_apply_motion_blur_preserves_dimensions(self):
        """Verify motion blur filter retains original image resolution and channel depth.

        Args:
            None.

        Returns:
            None.
        """
        dummy_img = np.full((160, 160, 3), 128, dtype=np.uint8)
        blurred = apply_motion_blur(dummy_img, kernel_size=5, angle=30)
        assert blurred.shape == (160, 160, 3)
        assert blurred.dtype == np.uint8

    def test_crop_and_resize_with_boxes_preserves_aspect_ratio(self):
        """Verify center cropping transforms portrait rectangles into square targets.

        Args:
            None.

        Returns:
            None.
        """
        portrait_img = np.full((1600, 1200, 3), 128, dtype=np.uint8)
        boxes = [(0, 0.5, 0.5, 0.4, 0.4)]
        cropped, adj_boxes = crop_and_resize_with_boxes(portrait_img, boxes, target_size=160)

        assert cropped.shape == (160, 160, 3)
        assert len(adj_boxes) == 1
        cls_id, xc, yc, bw, bh = adj_boxes[0]
        assert cls_id == 0
        assert 0.0 <= xc <= 1.0
        assert 0.0 <= yc <= 1.0

    def test_apply_cutout_creates_occlusion_patch(self):
        """Verify synthetic cutout modifies pixel values within expected patch boundaries.

        Args:
            None.

        Returns:
            None.
        """
        dummy_img = np.zeros((160, 160, 3), dtype=np.uint8)
        occluded = apply_cutout(dummy_img, num_patches=1, min_size=0.10, max_size=0.25)

        assert occluded.shape == (160, 160, 3)
        assert (occluded > 0).any()


@pytest.mark.unit
class TestTargetEncoding:
    """Test suite verifying target matrix generation for dual detection heads."""

    def test_encode_targets_shapes_and_anchor_assignment(self):
        """Verify target encoding produces proper matrix shapes with valid object indicators.

        Args:
            None.

        Returns:
            None.
        """
        dataset = BanknoteDataset.__new__(BanknoteDataset)
        dataset.img_size = 160
        dataset.num_classes = len(DENOMINATIONS)

        boxes = [(5, 0.5, 0.5, 0.6, 0.3)]
        t16, t32 = dataset.encode_targets(boxes)

        assert t16.shape == (3, 10, 10, 12)
        assert t32.shape == (3, 5, 5, 12)

        has_s16_obj = (t16[..., 4] == 1.0).any()
        has_s32_obj = (t32[..., 4] == 1.0).any()
        assert has_s16_obj or has_s32_obj

    def test_dataset_cached_len_and_getitem(self):
        """Verify cached dataset returns valid tensor and target tuples.

        Args:
            None.

        Returns:
            None.
        """
        dataset = BanknoteDataset.__new__(BanknoteDataset)
        dataset.img_size = 160
        dataset.num_classes = len(DENOMINATIONS)
        dataset.augment = False
        dataset.samples = [("dummy.jpg", "dummy.txt")]
        dataset.cached_images = [np.full((160, 160, 3), 128, dtype=np.uint8)]
        dataset.cached_boxes = [[(0, 0.5, 0.5, 0.4, 0.4)]]

        assert len(dataset) == 1
        img_tensor, t16, t32 = dataset[0]

        assert img_tensor.shape == (3, 160, 160)
        assert t16.shape == (3, 10, 10, 12)
        assert t32.shape == (3, 5, 5, 12)
