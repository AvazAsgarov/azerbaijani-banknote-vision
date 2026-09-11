"""
TensorFlow / TFLite Builder and Full INT8 Quantizer for YOLO-FastestV2.
Converts PyTorch checkpoints into verified TFLite FlatBuffer binaries (magic TFL3).
Generates C++ firmware header arrays for ESP32-S3 MicroInterpreter deployment.
"""

import os
import sys
from pathlib import Path
from typing import Callable, Generator, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import torch

from src.core.config import ProjectPaths
from src.tinyml.model import YOLOFastestV2


def build_keras_conv_bn_relu(x: tf.Tensor, out_filters: int, kernel_size: int = 3, stride: int = 1, prefix: str = "") -> tf.Tensor:
    """Construct Convolution + BatchNorm + ReLU6 sequential block.

    Args:
        x: Input tensor.
        out_filters: Target channel dimension.
        kernel_size: Spatial receptive field diameter.
        stride: Convolutional spatial stride.
        prefix: Unique layer identification prefix.

    Returns:
        Tensor transformed by padded conv, batchnorm, and relu6.
    """
    if kernel_size > 1:
        x = layers.ZeroPadding2D(padding=1, name=f"{prefix}_pad")(x)
        conv_padding = "valid"
    else:
        conv_padding = "valid"

    x = layers.Conv2D(
        filters=out_filters,
        kernel_size=kernel_size,
        strides=stride,
        padding=conv_padding,
        use_bias=False,
        name=f"{prefix}_conv"
    )(x)
    x = layers.BatchNormalization(
        epsilon=1e-5,
        momentum=0.9,
        name=f"{prefix}_bn"
    )(x)
    x = layers.ReLU(max_value=6.0, name=f"{prefix}_relu6")(x)
    return x


def build_keras_dw_separable_block(
    x: tf.Tensor,
    in_channels: int,
    out_channels: int,
    stride: int = 1,
    expand_ratio: int = 2,
    prefix: str = ""
) -> tf.Tensor:
    """Construct MobileNetV2-style depthwise separable inverted bottleneck block.

    Args:
        x: Input tensor.
        in_channels: Incoming channel dimension.
        out_channels: Outgoing channel dimension.
        stride: Spatial downsampling stride.
        expand_ratio: Expansion multiplier for intermediate depthwise layer.
        prefix: Identification namespace prefix.

    Returns:
        Feature tensor with optional residual addition.
    """
    shortcut = x
    hidden_dim = int(in_channels * expand_ratio)

    curr = x
    if expand_ratio != 1:
        curr = build_keras_conv_bn_relu(
            curr,
            out_filters=hidden_dim,
            kernel_size=1,
            stride=1,
            prefix=f"{prefix}_expand"
        )

    curr = layers.ZeroPadding2D(padding=1, name=f"{prefix}_dw_pad")(curr)
    curr = layers.DepthwiseConv2D(
        kernel_size=3,
        strides=stride,
        padding="valid",
        use_bias=False,
        name=f"{prefix}_dw_conv"
    )(curr)
    curr = layers.BatchNormalization(
        epsilon=1e-5,
        momentum=0.9,
        name=f"{prefix}_dw_bn"
    )(curr)
    curr = layers.ReLU(max_value=6.0, name=f"{prefix}_dw_relu6")(curr)

    curr = layers.Conv2D(
        filters=out_channels,
        kernel_size=1,
        strides=1,
        padding="valid",
        use_bias=False,
        name=f"{prefix}_proj_conv"
    )(curr)
    curr = layers.BatchNormalization(
        epsilon=1e-5,
        momentum=0.9,
        name=f"{prefix}_proj_bn"
    )(curr)

    if stride == 1 and in_channels == out_channels:
        return layers.Add(name=f"{prefix}_add")([shortcut, curr])
    return curr


def build_keras_yolo_fastestv2(input_shape: Tuple[int, int, int] = (160, 160, 3), num_classes: int = 7, num_anchors: int = 3) -> tf.keras.Model:
    """Build functional Keras model corresponding to YOLO-FastestV2 micro-architecture.

    Args:
        input_shape: Spatial and channel dimensions of camera feed.
        num_classes: Number of distinct denomination categories.
        num_anchors: Number of anchor boxes per detection scale.

    Returns:
        Compiled Keras Model producing dual detection scale tensors.
    """
    out_channels_per_anchor = 4 + 1 + num_classes
    total_head_channels = num_anchors * out_channels_per_anchor

    inputs = layers.Input(shape=input_shape, name="input_image")

    # Stage 1: 160x160x3 -> 80x80x12
    x = build_keras_conv_bn_relu(inputs, out_filters=12, kernel_size=3, stride=2, prefix="stage1")

    # Stage 2: 80x80x12 -> 40x40x24
    x = build_keras_dw_separable_block(x, in_channels=12, out_channels=24, stride=2, expand_ratio=2, prefix="stage2")

    # Stage 3: 40x40x24 -> 20x20x48
    x = build_keras_dw_separable_block(x, in_channels=24, out_channels=48, stride=2, expand_ratio=2, prefix="stage3_b0")
    x = build_keras_dw_separable_block(x, in_channels=48, out_channels=48, stride=1, expand_ratio=2, prefix="stage3_b1")

    # Stage 4: 20x20x48 -> 10x10x96
    x = build_keras_dw_separable_block(x, in_channels=48, out_channels=96, stride=2, expand_ratio=2, prefix="stage4_b0")
    x = build_keras_dw_separable_block(x, in_channels=96, out_channels=96, stride=1, expand_ratio=2, prefix="stage4_b1")
    x4 = build_keras_dw_separable_block(x, in_channels=96, out_channels=96, stride=1, expand_ratio=2, prefix="stage4_b2")

    # Stage 5: 10x10x96 -> 5x5x96
    x = build_keras_dw_separable_block(x4, in_channels=96, out_channels=96, stride=2, expand_ratio=2, prefix="stage5_b0")
    x5 = build_keras_dw_separable_block(x, in_channels=96, out_channels=96, stride=1, expand_ratio=2, prefix="stage5_b1")

    # Output Heads (1x1 Conv with bias)
    head_s16 = layers.Conv2D(
        filters=total_head_channels,
        kernel_size=1,
        strides=1,
        padding="valid",
        use_bias=True,
        name="head_s16"
    )(x4)

    head_s32 = layers.Conv2D(
        filters=total_head_channels,
        kernel_size=1,
        strides=1,
        padding="valid",
        use_bias=True,
        name="head_s32"
    )(x5)

    model = models.Model(inputs=inputs, outputs=[head_s16, head_s32], name="yolo_fastestv2_keras")
    return model


def transfer_pt_conv_bn(pt_conv_bn_relu, keras_model, prefix: str) -> None:
    """Transfer parameters from PyTorch ConvBNReLU to Keras counterpart.

    Args:
        pt_conv_bn_relu: PyTorch ConvBNReLU module.
        keras_model: Destination Keras model.
        prefix: Target layer namespace identifier.

    Returns:
        None.
    """
    conv_w = pt_conv_bn_relu.conv.weight.detach().cpu().numpy()
    conv_w = np.transpose(conv_w, (2, 3, 1, 0))
    keras_model.get_layer(f"{prefix}_conv").set_weights([conv_w])

    bn_gamma = pt_conv_bn_relu.bn.weight.detach().cpu().numpy()
    bn_beta = pt_conv_bn_relu.bn.bias.detach().cpu().numpy()
    bn_mean = pt_conv_bn_relu.bn.running_mean.detach().cpu().numpy()
    bn_var = pt_conv_bn_relu.bn.running_var.detach().cpu().numpy()
    keras_model.get_layer(f"{prefix}_bn").set_weights([bn_gamma, bn_beta, bn_mean, bn_var])


def transfer_pt_dw_block(pt_block, keras_model, prefix: str) -> None:
    """Transfer parameters from PyTorch DepthwiseSeparableBlock to Keras layers.

    Args:
        pt_block: PyTorch DepthwiseSeparableBlock module.
        keras_model: Target Keras functional model.
        prefix: Unique layer prefix.

    Returns:
        None.
    """
    sub_modules = list(pt_block.conv.children())
    idx = 0

    if len(sub_modules) == 4:
        transfer_pt_conv_bn(sub_modules[0], keras_model, f"{prefix}_expand")
        idx = 1

    dw_cbr = sub_modules[idx]
    dw_w = dw_cbr.conv.weight.detach().cpu().numpy()
    dw_w = np.transpose(dw_w, (2, 3, 0, 1))
    keras_model.get_layer(f"{prefix}_dw_conv").set_weights([dw_w])

    dw_gamma = dw_cbr.bn.weight.detach().cpu().numpy()
    dw_beta = dw_cbr.bn.bias.detach().cpu().numpy()
    dw_mean = dw_cbr.bn.running_mean.detach().cpu().numpy()
    dw_var = dw_cbr.bn.running_var.detach().cpu().numpy()
    keras_model.get_layer(f"{prefix}_dw_bn").set_weights([dw_gamma, dw_beta, dw_mean, dw_var])

    proj_conv = sub_modules[idx + 1]
    proj_w = proj_conv.weight.detach().cpu().numpy()
    proj_w = np.transpose(proj_w, (2, 3, 1, 0))
    keras_model.get_layer(f"{prefix}_proj_conv").set_weights([proj_w])

    proj_bn = sub_modules[idx + 2]
    proj_gamma = proj_bn.weight.detach().cpu().numpy()
    proj_beta = proj_bn.bias.detach().cpu().numpy()
    proj_mean = proj_bn.running_mean.detach().cpu().numpy()
    proj_var = proj_bn.running_var.detach().cpu().numpy()
    keras_model.get_layer(f"{prefix}_proj_bn").set_weights([proj_gamma, proj_beta, proj_mean, proj_var])


def transfer_all_weights(pt_model: YOLOFastestV2, keras_model: tf.keras.Model) -> None:
    """Transfer complete parameter dictionary from PyTorch detector to Keras model.

    Args:
        pt_model: Trained PyTorch YOLOFastestV2 instance.
        keras_model: Uninitialized Keras functional architecture.

    Returns:
        None.
    """
    transfer_pt_conv_bn(pt_model.stage1, keras_model, "stage1")
    transfer_pt_dw_block(pt_model.stage2, keras_model, "stage2")

    transfer_pt_dw_block(pt_model.stage3[0], keras_model, "stage3_b0")
    transfer_pt_dw_block(pt_model.stage3[1], keras_model, "stage3_b1")

    transfer_pt_dw_block(pt_model.stage4[0], keras_model, "stage4_b0")
    transfer_pt_dw_block(pt_model.stage4[1], keras_model, "stage4_b1")
    transfer_pt_dw_block(pt_model.stage4[2], keras_model, "stage4_b2")

    transfer_pt_dw_block(pt_model.stage5[0], keras_model, "stage5_b0")
    transfer_pt_dw_block(pt_model.stage5[1], keras_model, "stage5_b1")

    head16_w = pt_model.head_s16.weight.detach().cpu().numpy()
    head16_w = np.transpose(head16_w, (2, 3, 1, 0))
    head16_b = pt_model.head_s16.bias.detach().cpu().numpy()
    keras_model.get_layer("head_s16").set_weights([head16_w, head16_b])

    head32_w = pt_model.head_s32.weight.detach().cpu().numpy()
    head32_w = np.transpose(head32_w, (2, 3, 1, 0))
    head32_b = pt_model.head_s32.bias.detach().cpu().numpy()
    keras_model.get_layer("head_s32").set_weights([head32_w, head32_b])


def verify_numerical_parity(pt_model: YOLOFastestV2, keras_model: tf.keras.Model, tolerance: float = 1e-4) -> float:
    """Verify maximum numerical discrepancy between PyTorch and Keras forward passes.

    Args:
        pt_model: PyTorch model in evaluation mode.
        keras_model: Functional Keras model with synchronized weights.
        tolerance: Maximum tolerable absolute floating point difference.

    Returns:
        Maximum absolute deviation observed across both heads.
    """
    pt_model.eval()
    np.random.seed(42)
    sample_np = np.random.uniform(0.0, 1.0, size=(1, 3, 160, 160)).astype(np.float32)

    with torch.no_grad():
        x_pt = torch.from_numpy(sample_np)
        b = x_pt.shape[0]
        x1 = pt_model.stage1(x_pt)
        x2 = pt_model.stage2(x1)
        x3 = pt_model.stage3(x2)
        x4 = pt_model.stage4(x3)
        x5 = pt_model.stage5(x4)
        raw_s16 = pt_model.head_s16(x4)
        raw_s32 = pt_model.head_s32(x5)

    sample_tf = np.transpose(sample_np, (0, 2, 3, 1))
    tf_s16, tf_s32 = keras_model(sample_tf)

    pt_s16_nhwc = np.transpose(raw_s16.cpu().numpy(), (0, 2, 3, 1))
    pt_s32_nhwc = np.transpose(raw_s32.cpu().numpy(), (0, 2, 3, 1))

    diff_16 = float(np.max(np.abs(pt_s16_nhwc - tf_s16.numpy())))
    diff_32 = float(np.max(np.abs(pt_s32_nhwc - tf_s32.numpy())))
    max_diff = max(diff_16, diff_32)

    if max_diff > tolerance:
        raise ValueError(f"Numerical disparity exceeded tolerance: {max_diff:.6e} > {tolerance:.6e}")
    return max_diff


def export_cpp_header(model_bytes: bytes, output_header_path: str, array_name: str = "g_yolo_fastestv2_model_data") -> None:
    """Format model byte array into C++ header source with memory alignment directives.

    Args:
        model_bytes: Quantized FlatBuffer binary data.
        output_header_path: Destination path for C/C++ header.
        array_name: C identifier for constant byte array.

    Returns:
        None.
    """
    length = len(model_bytes)
    lines = [
        "// Auto-generated TinyML Model Byte Array for ESP32-S3 Sense",
        "// Target Architecture: YOLO-FastestV2 INT8 (160x160 RGB)",
        "// Format: Genuine TFLite FlatBuffer (Schema Version 3)",
        "#ifndef YOLO_FASTESTV2_MODEL_DATA_H",
        "#define YOLO_FASTESTV2_MODEL_DATA_H",
        "",
        "#include <stdint.h>",
        "",
        f"// Model binary size: {length} bytes ({length / 1024.0:.2f} KB)",
        f"alignas(16) const unsigned char {array_name}[] = {{",
    ]

    row = []
    for b in model_bytes:
        row.append(f"0x{b:02x}")
        if len(row) == 16:
            lines.append("    " + ", ".join(row) + ",")
            row = []
    if row:
        lines.append("    " + ", ".join(row))

    lines.extend([
        "};",
        "",
        f"const unsigned int {array_name}_len = {length};",
        "",
        "#endif // YOLO_FASTESTV2_MODEL_DATA_H",
        ""
    ])

    Path(output_header_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_header_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def convert_to_tflite_int8(
    keras_model: tf.keras.Model,
    data_generator: Callable[[], Generator[List[np.ndarray], None, None]],
    output_path: str
) -> bytes:
    """Convert Keras detector graph into fully quantized INT8 TFLite FlatBuffer.

    Args:
        keras_model: Trained Keras functional network.
        data_generator: Representative sample calibration generator.
        output_path: Target filesystem path for .tflite output.

    Returns:
        Binary bytes of the generated TFLite FlatBuffer.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = data_generator
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model_bytes = converter.convert()

    magic_check = tflite_model_bytes[4:8]
    if magic_check != b"TFL3":
        raise ValueError(f"Invalid FlatBuffer identifier: {magic_check!r} (expected b'TFL3')")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(tflite_model_bytes)

    return tflite_model_bytes


def build_and_export_full_pipeline(
    checkpoint_path: Optional[str] = None,
    output_tflite_path: Optional[str] = None,
    output_header_path: Optional[str] = None,
    firmware_header_path: Optional[str] = None,
) -> Tuple[str, str]:
    """Execute complete PyTorch-to-TFLite quantization and firmware header export.

    Args:
        checkpoint_path: Path to PyTorch model weights (optional).
        output_tflite_path: Target destination for quantized model (optional).
        output_header_path: Target destination for artifacts C++ header (optional).
        firmware_header_path: Firmware directory copy destination (optional).

    Returns:
        Tuple containing generated tflite path and header path.
    """
    paths = ProjectPaths()
    resolved_ckpt = str(paths.artifacts_dir / "tinyml" / "yolo_fastestv2_best.pt") if checkpoint_path is None else str(checkpoint_path)
    resolved_tflite = str(paths.artifacts_dir / "tinyml" / "yolo_fastestv2_int8.tflite") if output_tflite_path is None else str(output_tflite_path)
    resolved_header = str(paths.artifacts_dir / "tinyml" / "yolo_fastestv2_int8.h") if output_header_path is None else str(output_header_path)
    resolved_fw = str(paths.root_dir / "firmware" / "esp32s3_smart_glasses" / "main" / "yolo_fastestv2_int8.h") if firmware_header_path is None else str(firmware_header_path)

    pt_model = YOLOFastestV2(num_classes=7, num_anchors=3)
    if os.path.exists(resolved_ckpt):
        ckpt = torch.load(resolved_ckpt, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        pt_model.load_state_dict(state_dict)

    pt_model.eval()

    keras_model = build_keras_yolo_fastestv2(input_shape=(160, 160, 3), num_classes=7, num_anchors=3)
    transfer_all_weights(pt_model, keras_model)

    max_err = verify_numerical_parity(pt_model, keras_model, tolerance=1e-4)
    print(f"Verified numerical parity. Maximum absolute error: {max_err:.6e}")

    def rep_data_gen():
        rng = np.random.RandomState(42)
        for _ in range(100):
            sample = rng.uniform(0.0, 1.0, size=(1, 160, 160, 3)).astype(np.float32)
            yield [sample]

    tflite_bytes = convert_to_tflite_int8(keras_model, rep_data_gen, resolved_tflite)
    print(f"Generated quantized TFLite model at {resolved_tflite} ({len(tflite_bytes) / 1024.0:.2f} KB)")

    export_cpp_header(tflite_bytes, resolved_header)
    print(f"Generated C++ firmware header at {resolved_header}")

    if resolved_fw:
        export_cpp_header(tflite_bytes, resolved_fw)
        print(f"Synchronized C++ firmware header to {resolved_fw}")

    return resolved_tflite, resolved_header


if __name__ == "__main__":
    build_and_export_full_pipeline()
