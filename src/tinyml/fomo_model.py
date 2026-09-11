"""FOMO (Faster Objects, More Objects) MobileNetV1 Architecture for TinyML ESP32-S3.

Micro-scale edge centroid detector that predicts object centroids on a downsampled grid (12x12 for 96x96 input, or 20x20 for 160x160).
Eliminates bounding box regression overhead, achieving ultra-high framerates (>40 FPS on ESP32-S3).
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple


class ConvBNReLU(nn.Sequential):
    """Standard Convolution + BatchNorm + ReLU sequence."""

    def __init__(self, in_planes: int, out_planes: int, kernel_size: int = 3, stride: int = 1, padding: int = 1) -> None:
        super().__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
        )


class DepthwiseSeparableConv(nn.Sequential):
    """Standard MobileNetV1 Depthwise Separable Convolution block."""

    def __init__(self, in_planes: int, out_planes: int, stride: int = 1) -> None:
        super().__init__(
            # Depthwise
            nn.Conv2d(in_planes, in_planes, 3, stride, 1, groups=in_planes, bias=False),
            nn.BatchNorm2d(in_planes),
            nn.ReLU(inplace=True),
            # Pointwise
            nn.Conv2d(in_planes, out_planes, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
        )


class FOMOMobileNetV1(nn.Module):
    """FOMO Centroid Detector with MobileNetV1 Backbone.

    Predicts grid heatmaps with (num_classes + 1) channels, where channel 0 is background.
    """

    def __init__(self, num_classes: int = 7, width_mult: float = 0.35, input_size: int = 96) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.width_mult = width_mult
        self.input_size = input_size

        def ch(c: int) -> int:
            return max(8, int(c * width_mult))

        # Stride 8 downsampling backbone (96x96 -> 12x12 or 160x160 -> 20x20)
        self.features = nn.Sequential(
            ConvBNReLU(3, ch(32), stride=2),             # /2
            DepthwiseSeparableConv(ch(32), ch(64), 1),
            DepthwiseSeparableConv(ch(64), ch(128), 2),   # /4
            DepthwiseSeparableConv(ch(128), ch(128), 1),
            DepthwiseSeparableConv(ch(128), ch(256), 2),  # /8
            DepthwiseSeparableConv(ch(256), ch(256), 1),
            DepthwiseSeparableConv(ch(256), ch(512), 1),
            DepthwiseSeparableConv(ch(512), ch(512), 1),
        )

        # Centroid classifier head: (num_classes + 1 for background)
        self.head = nn.Conv2d(ch(512), num_classes + 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass generating raw class logits per grid cell of shape (B, num_classes + 1, H_out, W_out)."""
        feat = self.features(x)
        return self.head(feat)

    def count_parameters(self) -> int:
        """Calculates total trainable parameter volume."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
