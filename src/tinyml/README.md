# TinyML & Edge Pipeline (src/tinyml/)

## Purpose
Ultra-compact object detection system optimized for the Seeed Studio XIAO ESP32-S3 microcontroller
and assistive smart glasses deployment.

## Key Files
- `model.py`: PyTorch implementation of YOLO-FastestV2 (196K parameters, 160x160 input, dual-scale heads).
- `loss.py`: CIoU coordinate regression with asymmetric cost penalties (beta_c) for high denominations.
- `dataset.py`: Pre-cached RAM dataset loader for ultra-fast 160x160 training iterations.
- `train.py`: Training engine with learning rate warm-up, evaluation mAP, and test inference overlays.
- `quantize_export.py`: Calibration-based INT8 static quantization and C++ header generation.
- `safety_guard.py`: Multi-tier temporal state machine verifying detections across consecutive frames.
- `benchmark.py`: MCU latency and SRAM arena memory profiler.
