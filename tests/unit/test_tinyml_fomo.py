"""Unit test suite for TinyML FOMO (MobileNetV1 0.35x) architecture and decoding logic.

Verifies structural tensor dimensions, parameter counts, loss calculations,
and centroid heatmap prediction decoding for edge deployment.
"""

import pytest
import numpy as np
import torch

from src.tinyml.fomo_model import FOMOMobileNetV1, ConvBNReLU, DepthwiseSeparableConv
from src.tinyml.fomo_dataset import FOMOFocalLoss
from scripts.mobile_bridge import decode_fomo_predictions


@pytest.mark.unit
class TestFOMOArchitecture:
    """Test suite verifying FOMO MobileNetV1 neural network topology."""

    def test_conv_bn_relu_forward(self):
        """Verify standard ConvBNReLU block preserves or downsamples spatial dimensions."""
        block = ConvBNReLU(in_planes=3, out_planes=16, kernel_size=3, stride=2, padding=1)
        x = torch.randn(2, 3, 96, 96)
        out = block(x)
        assert out.shape == (2, 16, 48, 48)

    def test_depthwise_separable_conv_forward(self):
        """Verify depthwise separable convolution forward pass."""
        dw_block = DepthwiseSeparableConv(in_planes=16, out_planes=32, stride=2)
        x = torch.randn(2, 16, 48, 48)
        out = dw_block(x)
        assert out.shape == (2, 32, 24, 24)

    def test_fomo_model_forward_pass_96x96(self):
        """Verify FOMO output grid dimensions (12x12) for 96x96 input."""
        model = FOMOMobileNetV1(num_classes=7, width_mult=0.35, input_size=96)
        model.eval()
        x = torch.randn(1, 3, 96, 96)
        with torch.no_grad():
            logits = model(x)
        # Output shape: (B, num_classes + 1, 12, 12)
        assert logits.shape == (1, 8, 12, 12)

    def test_fomo_model_parameter_count(self):
        """Verify parameter volume is strictly within micro edge budget (<100K params)."""
        model = FOMOMobileNetV1(num_classes=7, width_mult=0.35, input_size=96)
        total_params = model.count_parameters()
        assert 50_000 < total_params < 100_000, f"Expected <100k params, got {total_params}"


@pytest.mark.unit
class TestFOMOFocalLoss:
    """Test suite verifying FOMO Focal Loss for class imbalance."""

    def test_focal_loss_computation_and_backward(self):
        """Verify loss computes finite scalar and enables gradient backpropagation."""
        loss_fn = FOMOFocalLoss(num_classes=7, gamma=2.0, alpha_bg=0.25)
        logits = torch.randn(2, 8, 12, 12, requires_grad=True)
        # Background is 0, class labels 1..7
        targets = torch.randint(0, 8, (2, 12, 12), dtype=torch.long)

        loss = loss_fn(logits, targets)
        assert torch.isfinite(loss)
        assert loss.item() > 0.0

        loss.backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()


@pytest.mark.unit
class TestFOMODecoder:
    """Test suite verifying bridge centroid heatmap decoding."""

    def test_decode_fomo_empty_when_all_background(self):
        """Verify no false positive detections produced when channel 0 (background) dominates."""
        heatmap = np.zeros((8, 12, 12), dtype=np.float32)
        heatmap[0, :, :] = 10.0  # Dominant background logit
        dets = decode_fomo_predictions(heatmap, conf_thresh=0.50)
        assert len(dets) == 0

    def test_decode_fomo_detects_confident_centroid(self):
        """Verify centroid is detected at expected location and class ID."""
        heatmap = np.zeros((8, 12, 12), dtype=np.float32)
        heatmap[0, :, :] = 0.0
        # Class 1 (001_azn) peaked at grid cell (6, 6)
        heatmap[1, 6, 6] = 20.0

        dets = decode_fomo_predictions(heatmap, conf_thresh=0.50)
        assert len(dets) == 1
        d = dets[0]
        assert d["denomination_id"] == 0
        assert d["class_code"] == "001_azn"
        assert d["confidence"] > 0.95
        # Grid cell (6, 6) has center (6.5 / 12, 6.5 / 12) ~ (0.542, 0.542)
        x1, y1, x2, y2 = d["bbox"]
        xc = (x1 + x2) / 2.0
        yc = (y1 + y2) / 2.0
        assert abs(xc - 0.542) < 0.05
        assert abs(yc - 0.542) < 0.05
