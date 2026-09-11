"""Hardware interface modules for ESP32-S3 and edge devices."""

from src.hardware.esp32_camera import ESP32CameraBridge, find_esp32s3_port

__all__ = ["ESP32CameraBridge", "find_esp32s3_port"]
