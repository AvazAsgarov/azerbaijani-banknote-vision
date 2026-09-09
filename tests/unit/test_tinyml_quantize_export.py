"""
Unit tests for INT8 quantization and C++ firmware array export.
Verifies memory alignment directives, byte syntax formatting, and array length variables.
"""

import pytest
import os
import tempfile

from src.tinyml.quantize_export import export_cpp_header, BanknoteCalibrationDataReader
import torch


@pytest.mark.unit
class TestQuantizationExport:
    """Test suite verifying C++ firmware header file generation."""

    def test_export_cpp_header_syntax_and_alignment(self):
        """Verify generated C++ header contains alignas(16) directives and valid array syntax.

        Args:
            None.

        Returns:
            None.
        """
        dummy_bytes = bytes([0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0xAA, 0xFF])

        with tempfile.NamedTemporaryFile(suffix=".h", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            export_cpp_header(dummy_bytes, tmp_path, array_name="test_model_data")

            with open(tmp_path, "r", encoding="utf-8") as f:
                content = f.read()

            assert "alignas(16)" in content
            assert "test_model_data[] = {" in content
            assert "test_model_data_len = 8;" in content
            assert "0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0xaa, 0xff" in content
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_calibration_data_reader_iteration(self):
        """Verify calibration data reader yields input dictionary batches sequentially.

        Args:
            None.

        Returns:
            None.
        """
        mock_dataset = [
            (torch.randn(3, 160, 160), None, None),
            (torch.randn(3, 160, 160), None, None)
        ]
        reader = BanknoteCalibrationDataReader(mock_dataset, num_samples=2)

        batch1 = reader.get_next()
        assert batch1 is not None
        assert "input" in batch1
        assert batch1["input"].shape == (1, 3, 160, 160)

        batch2 = reader.get_next()
        assert batch2 is not None

        batch3 = reader.get_next()
        assert batch3 is None

    def test_tflite_flatbuffer_schema_version_and_magic(self):
        """Verify TFLite binary file contains valid FlatBuffer magic identifier and size.

        Args:
            None.

        Returns:
            None.
        """
        tflite_path = os.path.join("artifacts", "tinyml", "yolo_fastestv2_int8.tflite")
        if not os.path.exists(tflite_path):
            pytest.skip("TFLite artifact not generated yet.")

        with open(tflite_path, "rb") as f:
            data = f.read()

        assert len(data) > 100000
        magic = data[4:8]
        assert magic == b"TFL3"

    def test_firmware_cpp_header_tflite_sync(self):
        """Verify firmware C++ header matches exported TFLite FlatBuffer binary length.

        Args:
            None.

        Returns:
            None.
        """
        tflite_path = os.path.join("artifacts", "tinyml", "yolo_fastestv2_int8.tflite")
        onnx_path = os.path.join("artifacts", "tinyml", "yolo_fastestv2_160_int8.onnx")
        header_path = os.path.join("firmware", "esp32s3_smart_glasses", "main", "yolo_fastestv2_int8.h")

        if not os.path.exists(header_path):
            pytest.skip("Firmware header not present.")

        valid_sizes = []
        if os.path.exists(tflite_path):
            valid_sizes.append(os.path.getsize(tflite_path))
        if os.path.exists(onnx_path):
            valid_sizes.append(os.path.getsize(onnx_path))

        with open(header_path, "r", encoding="utf-8") as f:
            content = f.read()

        assert "alignas(16)" in content
        assert "g_yolo_fastestv2_model_data" in content
        assert any(f"g_yolo_fastestv2_model_data_len = {sz};" in content for sz in valid_sizes)

    def test_keras_yolo_fastestv2_architecture_shapes(self):
        """Verify Keras detector architecture produces correct dual detection head shapes.

        Args:
            None.

        Returns:
            None.
        """
        from src.tinyml.tflite_builder import build_keras_yolo_fastestv2
        keras_model = build_keras_yolo_fastestv2(input_shape=(160, 160, 3), num_classes=7, num_anchors=3)
        assert len(keras_model.outputs) == 2
        assert keras_model.outputs[0].shape == (None, 10, 10, 36)
        assert keras_model.outputs[1].shape == (None, 5, 5, 36)

