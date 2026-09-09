# ESP32-S3 Smart Glasses Firmware (firmware/esp32s3_smart_glasses/)

## Purpose
Real-time embedded C++ application running on Seeed Studio XIAO ESP32S3 Sense.
Captures frames from OV2640 sensor, runs INT8 YOLO-FastestV2 inference in SRAM,
and transmits confirmed banknote detections to a companion mobile application for voice synthesis.

## Hardware Specifications
- Microcontroller: ESP32-S3 Dual-Core Xtensa LX7 @ 240 MHz.
- SRAM: 512 KB Internal SRAM (285 KB allocated for Tensor Arena).
- Sensor: OV2640 Camera Module (160x160 RGB capture).

## Key Files
- `CMakeLists.txt`: ESP-IDF build system definition.
- `main/main.cpp`: FreeRTOS task coordinator managing capture, inference, and BLE output.
- `main/camera_pipeline.cpp`: Frame capture, cropping, and RGB normalization.
- `main/model_runner.cpp`: TensorFlow Lite for Microcontrollers (TFLM) execution engine.
- `main/safety_guard.cpp`: Multi-tier temporal verification machine (T1-T4).
- `main/yolo_fastestv2_int8.h`: Quantized model byte array aligned for zero-copy SRAM execution.
