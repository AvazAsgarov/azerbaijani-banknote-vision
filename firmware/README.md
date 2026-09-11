# Embedded Edge Firmware (`firmware/`)

This directory contains the embedded C++ and Arduino firmware implementations for the **AZN-Vision** wearable smart glasses system, targeted specifically for the ultra-low-power **Seeed Studio XIAO ESP32-S3 Sense** platform.

---

## Target Hardware Specifications

* **Microcontroller:** Espressif ESP32-S3 (Dual-core Xtensa® LX7 up to 240 MHz, RISC-V coprocessor).
* **Memory Architecture:**
  - **512 KB Internal SRAM:** Houses real-time execution pipelines and the 285 KB TFLM tensor arena.
  - **8 MB Octal-SPI (OPI) PSRAM:** Buffers incoming camera DMA frames.
  - **8 MB Quad-SPI Flash:** Stores firmware binaries, model weights, and calibration tables.
* **Camera Sensor:** Omnivision OV2640 (2.0 Megapixel) connected via DVP parallel interface.
* **Connectivity:** 2.4 GHz Wi-Fi (802.11 b/g/n) & Bluetooth Low Energy (BLE 5.0).

---

## Firmware Subprojects

### 1. [`esp32s3_smart_glasses/`](esp32s3_smart_glasses/)
* **Responsible:** Kazim Mammadli
* **Description:** Standalone on-device edge intelligence pipeline executing YOLO-FastestV2 INT8 using TensorFlow Lite for Microcontrollers (TFLM).
* **Key Components:**
  - `main/app_config.h`: Pin mapping, 285 KB SRAM arena budget, and dynamic confidence thresholds.
  - `main/camera_pipeline.cpp`: OV2640 direct DMA capture and 160x160 RGB downsampling.
  - `main/model_runner.cpp`: TFLM INT8 interpreter runtime executing quantized operators.
  - `main/safety_guard.cpp`: 4-tier assistive guardrail (Geometry -> Dynamic Gate -> 3/5 Temporal Voting -> User Feedback).
  - `main/yolo_fastestv2_int8.h`: Compilable C++ byte array generated from the genuine TFLite FlatBuffer.

### 2. [`esp32_wifi_camera/`](esp32_wifi_camera/)
* **Responsible:** Nicat Alaskarli
* **Description:** Low-latency Wi-Fi video streaming firmware streaming live frames to the assistive companion mobile application and Python host bridge.
* **Key Components:**
  - `esp32_wifi_camera.ino`: High-speed camera capture loop and multi-AP Wi-Fi initialization.
  - `app_httpd.cpp`: Multipart MJPEG HTTP streaming server on port 81 and snapshot handler on port 80.
  - `board_config.h`: GPIO pin definitions for XIAO ESP32-S3 Sense.
  - `ci.yml`: Arduino-cli multi-target build profile.
