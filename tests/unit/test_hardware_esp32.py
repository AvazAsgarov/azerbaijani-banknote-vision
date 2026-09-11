"""Unit Test Suite for ESP32-S3 Camera Bridge and Frame Parser."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from src.hardware.esp32_camera import (
    ESP32CameraBridge,
    find_esp32s3_port,
    JPEG_SOI,
    JPEG_EOI,
)


class TestESP32Camera(unittest.TestCase):
    """Verifies serial frame parsing, center-cropping, and telemetry calculations."""

    def test_find_esp32s3_port_matching(self) -> None:
        """Tests that ESP32-S3 VID:PID matches produce target port string."""
        mock_port = MagicMock()
        mock_port.device = "COM5"
        mock_port.vid = 0x303A
        mock_port.pid = 0x1001
        mock_port.description = "USB Serial Device (COM5)"

        with patch("serial.tools.list_ports.comports", return_value=[mock_port]):
            found = find_esp32s3_port()
            self.assertEqual(found, "COM5")

    def test_find_esp32s3_port_none(self) -> None:
        """Tests that absence of matching USB devices returns None or fallback."""
        with patch("serial.tools.list_ports.comports", return_value=[]):
            found = find_esp32s3_port()
            self.assertIsNone(found)

    def test_camera_bridge_initialization(self) -> None:
        """Tests default constructor parameters and telemetry fields."""
        bridge = ESP32CameraBridge(port="COM99", baudrate=115200)
        self.assertEqual(bridge.port_name, "COM99")
        self.assertEqual(bridge.baudrate, 115200)
        self.assertFalse(bridge.is_connected())

        telem = bridge.get_hardware_telemetry()
        self.assertEqual(telem["port"], "COM99")
        self.assertFalse(telem["connected"])
        self.assertEqual(telem["frames_captured"], 0)
        self.assertEqual(telem["thermal_state"], "NOMINAL")

    def test_jpeg_frame_extraction(self) -> None:
        """Tests extraction of complete JPEG frames from raw byte buffer."""
        bridge = ESP32CameraBridge(port="COM99")
        dummy_jpeg = JPEG_SOI + b"\x00\x10JFIF_DATA_PAYLOAD" + JPEG_EOI

        # Inject bytes into bridge buffer
        bridge._buffer.extend(b"\xaa\xbb\xcc\xdd" + dummy_jpeg + b"\x12\x34")
        extracted = bridge.read_raw_frame(max_read_time=0.1)

        self.assertIsNotNone(extracted)
        self.assertEqual(extracted, dummy_jpeg)
        self.assertEqual(bridge._frame_count, 1)
        # Remainder should stay in buffer
        self.assertEqual(bridge._buffer, bytearray(b"\x12\x34"))

    def test_preprocess_for_tinyml_dimensions(self) -> None:
        """Tests center-crop and resize logic from 640x480 to 160x160x3."""
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        tensor, (x_off, y_off, crop_sz) = ESP32CameraBridge.preprocess_for_tinyml(dummy_frame, target_size=160)

        self.assertEqual(tensor.shape, (1, 3, 160, 160))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertEqual(crop_sz, 480)
        self.assertEqual(x_off, 80)
        self.assertEqual(y_off, 0)
        self.assertTrue(0.0 <= tensor.min() <= tensor.max() <= 1.0)

    def test_thermal_telemetry_nominal_and_elevated(self) -> None:
        """Tests thermal elevation model outputs under varying simulated framerates."""
        bridge = ESP32CameraBridge(port="COM99")
        bridge._last_fps = 5.0
        telem_low = bridge.get_hardware_telemetry()
        self.assertEqual(telem_low["thermal_state"], "NOMINAL")
        self.assertTrue(telem_low["estimated_chip_temp_c"] < 55.0)

        bridge._last_fps = 35.0
        telem_high = bridge.get_hardware_telemetry()
        self.assertEqual(telem_high["thermal_state"], "ELEVATED")
        self.assertTrue(telem_high["estimated_chip_temp_c"] >= 55.0)


if __name__ == "__main__":
    unittest.main()
