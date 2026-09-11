"""TinyML Subsystem for Embedded Edge Banknote Detection.

Houses micro-detector architectures (YOLO-FastestV2), streaming dataset loaders,
wearable augmentations, asymmetric cost-sensitive loss criteria, quantization
and TFLite / C++ firmware export routines, and multi-tier assistive safety guardrails.
"""

from .dataset import (
    ANCHORS_S16,
    ANCHORS_S32,
    DENOMINATIONS,
    BanknoteDataset,
    apply_cutout,
    apply_motion_blur,
    compute_anchor_iou,
    crop_and_resize_with_boxes,
)
from .loss import AsymmetricBanknoteLoss
from .model import ConvBNReLU, DepthwiseSeparableBlock, YOLOFastestV2
from .safety_guard import DYNAMIC_THRESHOLDS, AssistiveSafetyGuard
from .tflite_builder import (
    build_and_export_full_pipeline,
    convert_to_tflite_int8,
)

__all__ = [
    "ANCHORS_S16",
    "ANCHORS_S32",
    "DENOMINATIONS",
    "DYNAMIC_THRESHOLDS",
    "AssistiveSafetyGuard",
    "AsymmetricBanknoteLoss",
    "BanknoteDataset",
    "ConvBNReLU",
    "DepthwiseSeparableBlock",
    "YOLOFastestV2",
    "apply_cutout",
    "apply_motion_blur",
    "build_and_export_full_pipeline",
    "compute_anchor_iou",
    "convert_to_tflite_int8",
    "crop_and_resize_with_boxes",
]
