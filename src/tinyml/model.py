"""
YOLO-FastestV2 Micro-Detector Architecture tailored for ESP32-S3 Sense.
Implements depthwise-separable convolutional backbone with dual-scale detection heads.
Optimized for single-cycle internal SRAM allocation (<= 285 KB activation arena).
"""

from typing import Tuple
import torch
import torch.nn as nn

class ConvBNReLU(nn.Module):
    """Convolution followed by Batch Normalization and ReLU6 activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 1,
    ) -> None:
        """Initialize ConvBNReLU layer.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Size of the convolution kernel.
            stride: Convolution stride.
            padding: Zero-padding added to both sides.
            groups: Number of blocked connections from input to output channels.
        """
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute forward pass.

        Args:
            x: Input feature map tensor.

        Returns:
            Activated feature map tensor.
        """
        return self.act(self.bn(self.conv(x)))

class DepthwiseSeparableBlock(nn.Module):
    """Depthwise separable block with inverted bottleneck channel expansion."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expand_ratio: int = 2,
    ) -> None:
        """Initialize inverted residual depthwise separable block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            stride: Stride of the depthwise convolution.
            expand_ratio: Expansion multiplier for the hidden bottleneck.
        """
        super().__init__()
        self.stride = stride
        hidden_dim = int(in_channels * expand_ratio)
        self.use_residual = (self.stride == 1 and in_channels == out_channels)

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0))
        
        layers.append(ConvBNReLU(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim))
        layers.append(nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute forward pass with optional residual connection.

        Args:
            x: Input feature map tensor.

        Returns:
            Processed feature map tensor.
        """
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)

class YOLOFastestV2(nn.Module):
    """Tailored YOLO-FastestV2 for Azerbaijani Banknote Identification.

    Outputs:
        scale16: Feature grid 10x10 with 3 anchors (large banknotes)
        scale32: Feature grid 5x5 with 3 anchors (distant banknotes)
    """

    def __init__(self, num_classes: int = 7, num_anchors: int = 3) -> None:
        """Initialize YOLO-FastestV2 model.

        Args:
            num_classes: Number of banknote target classes.
            num_anchors: Number of anchor boxes per feature scale.
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.out_channels_per_anchor = 4 + 1 + num_classes  # (tx, ty, tw, th, obj, classes)
        total_head_channels = num_anchors * self.out_channels_per_anchor

        # Backbone Definition
        # Stage 1: 160x160x3 -> 80x80x12
        self.stage1 = ConvBNReLU(3, 12, kernel_size=3, stride=2, padding=1)

        # Stage 2: 80x80x12 -> 40x40x24
        self.stage2 = DepthwiseSeparableBlock(12, 24, stride=2, expand_ratio=2)

        # Stage 3: 40x40x24 -> 20x20x48
        self.stage3 = nn.Sequential(
            DepthwiseSeparableBlock(24, 48, stride=2, expand_ratio=2),
            DepthwiseSeparableBlock(48, 48, stride=1, expand_ratio=2)
        )

        # Stage 4: 20x20x48 -> 10x10x96 (Stride 16 output)
        self.stage4 = nn.Sequential(
            DepthwiseSeparableBlock(48, 96, stride=2, expand_ratio=2),
            DepthwiseSeparableBlock(96, 96, stride=1, expand_ratio=2),
            DepthwiseSeparableBlock(96, 96, stride=1, expand_ratio=2)
        )

        # Stage 5: 10x10x96 -> 5x5x96 (Stride 32 output)
        self.stage5 = nn.Sequential(
            DepthwiseSeparableBlock(96, 96, stride=2, expand_ratio=2),
            DepthwiseSeparableBlock(96, 96, stride=1, expand_ratio=2)
        )

        # Detection Heads (1x1 Conv)
        self.head_s16 = nn.Conv2d(96, total_head_channels, kernel_size=1, stride=1, padding=0)
        self.head_s32 = nn.Conv2d(96, total_head_channels, kernel_size=1, stride=1, padding=0)

        # Defined Anchor Grid Dimensions clustered for Azerbaijani Banknotes
        # S16 Anchors (10x10): [118, 58], [56, 112], [84, 82]
        # S32 Anchors (5x5):   [62, 32],  [30, 60],   [40, 42]
        self.register_buffer(
            "anchors_s16",
            torch.tensor([[118.0, 58.0], [56.0, 112.0], [84.0, 82.0]], dtype=torch.float32)
        )
        self.register_buffer(
            "anchors_s32",
            torch.tensor([[62.0, 32.0], [30.0, 60.0], [40.0, 42.0]], dtype=torch.float32)
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize convolution and batch norm parameters."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass executing feature extraction and dual-scale head projection.

        Args:
            x: Input batch tensor of shape (B, 3, 160, 160).

        Returns:
            Tuple containing:
                out_s16: Tensor of shape (B, num_anchors, 10, 10, out_channels_per_anchor).
                out_s32: Tensor of shape (B, num_anchors, 5, 5, out_channels_per_anchor).
        """
        b = x.shape[0]
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)  # Stride 16: 10x10x96
        x5 = self.stage5(x4)  # Stride 32: 5x5x96

        raw_s16 = self.head_s16(x4)
        raw_s32 = self.head_s32(x5)

        # Reshape to (B, num_anchors, H, W, attributes)
        out_s16 = raw_s16.view(b, self.num_anchors, self.out_channels_per_anchor, 10, 10)
        out_s16 = out_s16.permute(0, 1, 3, 4, 2).contiguous()

        out_s32 = raw_s32.view(b, self.num_anchors, self.out_channels_per_anchor, 5, 5)
        out_s32 = out_s32.permute(0, 1, 3, 4, 2).contiguous()

        return out_s16, out_s32

    def count_parameters(self) -> int:
        """Calculate total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
