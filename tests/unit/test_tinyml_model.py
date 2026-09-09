"""
Unit tests for YOLO-FastestV2 neural network architecture and depthwise separable blocks.
Verifies structural dimensions, parameter allocations, and forward pass integrity.
"""

import pytest
import torch

from src.tinyml.model import DepthwiseSeparableBlock, YOLOFastestV2


@pytest.mark.unit
class TestDepthwiseSeparableBlock:
    """Test suite verifying modular depthwise separable convolution blocks."""

    def test_block_shape_preservation_stride_one(self):
        """Verify output tensor dimensions match input dimensions when stride equals one.

        Args:
            None.

        Returns:
            None.
        """
        block = DepthwiseSeparableBlock(in_channels=32, out_channels=32, stride=1, expand_ratio=2)
        dummy_input = torch.randn(2, 32, 20, 20)
        output = block(dummy_input)

        assert output.shape == torch.Size([2, 32, 20, 20])

    def test_block_spatial_downsampling_stride_two(self):
        """Verify spatial dimensions halve when stride equals two.

        Args:
            None.

        Returns:
            None.
        """
        block = DepthwiseSeparableBlock(in_channels=16, out_channels=48, stride=2, expand_ratio=3)
        dummy_input = torch.randn(4, 16, 40, 40)
        output = block(dummy_input)

        assert output.shape == torch.Size([4, 48, 20, 20])

    def test_block_expansion_ratio_one_fallback(self):
        """Verify block functions reliably when expansion ratio equals one.

        Args:
            None.

        Returns:
            None.
        """
        block = DepthwiseSeparableBlock(in_channels=24, out_channels=24, stride=1, expand_ratio=1)
        dummy_input = torch.randn(1, 24, 10, 10)
        output = block(dummy_input)

        assert output.shape == torch.Size([1, 24, 10, 10])
        assert torch.isfinite(output).all()


@pytest.mark.unit
class TestYOLOFastestV2Architecture:
    """Test suite verifying YOLO-FastestV2 full network topology and head outputs."""

    def test_model_instantiation_parameter_budget(self):
        """Verify model instantiates with parameter counts strictly below edge hardware limits.

        Args:
            None.

        Returns:
            None.
        """
        model = YOLOFastestV2(num_classes=7)
        param_count = model.count_parameters()

        assert param_count > 100000
        assert param_count < 280000

    def test_forward_pass_output_tensor_shapes(self):
        """Verify dual detection heads produce exact feature grid dimensions.

        Args:
            None.

        Returns:
            None.
        """
        model = YOLOFastestV2(num_classes=7)
        model.eval()
        dummy_input = torch.randn(2, 3, 160, 160)

        with torch.no_grad():
            out_s16, out_s32 = model(dummy_input)

        assert out_s16.shape == torch.Size([2, 3, 10, 10, 12])
        assert out_s32.shape == torch.Size([2, 3, 5, 5, 12])

    def test_forward_pass_finite_values_under_extreme_inputs(self):
        """Verify forward pass outputs remain finite without generating NaN or Inf artifacts.

        Args:
            None.

        Returns:
            None.
        """
        model = YOLOFastestV2(num_classes=7)
        model.eval()
        zeros_input = torch.zeros(1, 3, 160, 160)
        ones_input = torch.ones(1, 3, 160, 160)

        with torch.no_grad():
            zeros_s16, zeros_s32 = model(zeros_input)
            ones_s16, ones_s32 = model(ones_input)

        assert torch.isfinite(zeros_s16).all()
        assert torch.isfinite(zeros_s32).all()
        assert torch.isfinite(ones_s16).all()
        assert torch.isfinite(ones_s32).all()

    def test_anchor_buffer_registration(self):
        """Verify anchor dimension buffers register correctly in model state.

        Args:
            None.

        Returns:
            None.
        """
        model = YOLOFastestV2(num_classes=7)

        assert hasattr(model, "anchors_s16")
        assert hasattr(model, "anchors_s32")
        assert model.anchors_s16.shape == torch.Size([3, 2])
        assert model.anchors_s32.shape == torch.Size([3, 2])
