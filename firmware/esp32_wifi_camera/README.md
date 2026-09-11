# ESP32-S3 Wi-Fi Streaming Camera Firmware

**Responsible Engineer:** Nicat Alaskarli  
**Target Hardware:** Seeed Studio XIAO ESP32-S3 Sense (OV2640 / OV3660 Sensor)

---

## Overview

This firmware turns the ultra-compact Seeed Studio XIAO ESP32-S3 Sense into a high-speed, low-latency Wi-Fi video streaming camera. It provides the live numismatic video feed for the **Assistive Mobile Companion Bridge** (`scripts/mobile_bridge.py`) and the **React Native Mobile Application** (`mobile/`).

---

## Key Features

1. **Dual Protocol HTTP/MJPEG Server:**
   - Single-frame snapshot endpoint (`/capture` on port 80).
   - Continuous multipart MJPEG video stream (`/stream` on port 81).
2. **mDNS Service Discovery:**
   - Broadcasts as `http://aznvision.local:81/stream`, eliminating the need for hardcoded IP addresses on local networks.
3. **Brownout Override Protection:**
   - Disables hardware brownout detection reset loops during high-transient Wi-Fi transmission bursts over USB power.
4. **Resilient Wi-Fi Connection:**
   - Utilizes `WiFiMulti` with configurable primary and hotspot access points.
5. **Memory-Optimized Framebuffers:**
   - Allocates camera framebuffers directly in high-speed 8 MB OPI Octal-SPI PSRAM.

---

## Build & Flashing Instructions

### Arduino IDE 2.x:
1. **Board Manager:** Install `esp32` by Espressif Systems (v2.0.11+).
2. **Board Selection:** `XIAO_ESP32S3`.
3. **Board Settings:**
   - **PSRAM:** `OPI PSRAM` (Enabled).
   - **Flash Mode:** `QIO 80MHz`.
   - **Partition Scheme:** `Huge APP (3MB No OTA/1MB SPIFFS)` or Custom `partitions.csv`.
4. **Compile & Upload:** Connect via USB-C, select the COM port, and click **Upload**.

### Command Line (`arduino-cli`):
```bash
arduino-cli compile --fqbn esp32:esp32:esp32s3:PSRAM=opi,FlashMode=qio firmware/esp32_wifi_camera/
```
