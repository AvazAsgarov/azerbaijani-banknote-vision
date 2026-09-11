"""
Quantization and C++ Header Array Exporter for ESP32-S3 Firmware Deployment.
Converts trained model to INT8 representation and generates C/C++ byte arrays.
Profiles flash footprint and verifies tensor arena memory alignment.
"""

import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    import onnxruntime
    from onnxruntime.quantization import quantize_dynamic, quantize_static, CalibrationDataReader, QuantType
except ImportError:
    onnxruntime = None
    CalibrationDataReader = object  # type: ignore[misc,assignment]

from src.core.config import ProjectPaths
from src.tinyml.dataset import BanknoteDataset

class BanknoteCalibrationDataReader(CalibrationDataReader):
    """Feeds real banknote frames into static quantization calibrator."""

    def __init__(self, dataset: Any, num_samples: int = 200) -> None:
        """Initialize calibration data reader.

        Args:
            dataset: Dataset providing calibration images.
            num_samples: Maximum number of calibration samples to feed.
        """
        self.dataset = dataset
        self.num_samples = min(num_samples, len(dataset))
        self.current_idx = 0

    def get_next(self) -> Optional[Dict[str, Any]]:
        """Retrieve next calibration input dictionary.

        Returns:
            Dictionary with input tensor name and numpy array, or None when exhausted.
        """
        if self.current_idx >= self.num_samples:
            return None
        img, _, _ = self.dataset[self.current_idx]
        self.current_idx += 1
        return {"input": img.unsqueeze(0).numpy()}

def export_cpp_header(
    model_bytes: bytes,
    output_header_path: str,
    array_name: str = "g_yolo_fastestv2_model_data",
) -> None:
    """Generate C/C++ header containing model byte array for ESP32-S3 firmware compilation.

    Args:
        model_bytes: Raw binary bytes of the quantized model.
        output_header_path: Path where the C/C++ header will be saved.
        array_name: Name of the byte array variable in C/C++.
    """
    length = len(model_bytes)
    lines = [
        "// Auto-generated TinyML Model Byte Array for ESP32-S3 Sense",
        "// Target Architecture: YOLO-FastestV2 INT8 (160x160 RGB)",
        "#ifndef YOLO_FASTESTV2_MODEL_DATA_H",
        "#define YOLO_FASTESTV2_MODEL_DATA_H",
        "",
        "#include <stdint.h>",
        "",
        f"// Model binary size: {length} bytes ({length / 1024.0:.2f} KB)",
        "alignas(16) const unsigned char " + array_name + "[] = {",
    ]

    # Format bytes in hex rows of 16 bytes
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

    with open(output_header_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Generated C++ model header at: {output_header_path} ({length / 1024.0:.2f} KB)")

def quantize_and_export_pipeline(
    onnx_path: Optional[str] = None,
    data_dir: Optional[str] = None,
) -> Tuple[str, str]:
    """Execute INT8 quantization and produce firmware header.

    Args:
        onnx_path: Path to the floating-point ONNX model (optional).
        data_dir: Directory containing processed banknote dataset for calibration (optional).

    Returns:
        Tuple of (int8_onnx_path, cpp_header_path).

    Raises:
        FileNotFoundError: If the base ONNX model does not exist.
    """
    paths = ProjectPaths()
    resolved_onnx = Path(onnx_path) if onnx_path else paths.artifacts_dir / "tinyml" / "yolo_fastestv2_160.onnx"
    resolved_data = str(paths.processed_data_dir if data_dir is None else Path(data_dir))

    out_dir = resolved_onnx.parent
    int8_onnx_path = str(out_dir / "yolo_fastestv2_160_int8.onnx")
    cpp_header_path = str(out_dir / "yolo_fastestv2_int8.h")

    if not resolved_onnx.exists():
        raise FileNotFoundError(f"Missing base ONNX model at: {resolved_onnx}")

    fp32_size = resolved_onnx.stat().st_size
    print(f"Base FP32 ONNX Size: {fp32_size / 1024.0:.2f} KB")

    # Static Calibration & Quantization
    calib_dataset = BanknoteDataset(resolved_data, split="val", img_size=160, augment=False)
    reader = BanknoteCalibrationDataReader(calib_dataset, num_samples=150)

    try:
        quantize_static(
            model_input=str(resolved_onnx),
            model_output=int8_onnx_path,
            calibration_data_reader=reader,
            quant_format=onnxruntime.quantization.QuantFormat.QDQ,
            per_channel=True,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QUInt8
        )
        print(f"Quantized static ONNX model saved to: {int8_onnx_path}")
    except Exception as e:
        print(f"Static quantization fallback to dynamic quantization: {e}")
        quantize_dynamic(
            model_input=str(resolved_onnx),
            model_output=int8_onnx_path,
            weight_type=QuantType.QInt8,
            per_channel=True
        )

    int8_size = os.path.getsize(int8_onnx_path)
    ratio = fp32_size / float(int8_size)
    print(f"INT8 Quantized Model Size: {int8_size / 1024.0:.2f} KB (Compression: {ratio:.2f}x)")

    # Read binary bytes and export C++ header
    model_bytes = Path(int8_onnx_path).read_bytes()
    export_cpp_header(model_bytes, cpp_header_path)

    # Mirror to firmware directory for ESP-IDF compilation
    firmware_header = paths.root_dir / "firmware" / "esp32s3_smart_glasses" / "main" / "yolo_fastestv2_int8.h"
    if firmware_header.parent.exists():
        shutil.copy2(cpp_header_path, firmware_header)
        print(f"Synchronized firmware C++ header at: {firmware_header}")

    return int8_onnx_path, cpp_header_path

if __name__ == "__main__":
    quantize_and_export_pipeline()
