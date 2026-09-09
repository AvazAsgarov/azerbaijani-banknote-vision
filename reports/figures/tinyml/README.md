# TinyML ESP32-S3 Edge Deployment Figures (`reports/figures/tinyml/`)

Visual benchmarks and training telemetry for the ultra-lightweight YOLO-FastestV2 detector on Seeed Studio XIAO ESP32-S3 Sense.

## Catalog of Figures
1. `tinyml_loss_curve.png`: Training convergence progression showing Total Loss, CIoU Box Loss, Objectness Loss, and Classification Loss across 100 epochs.
2. `tinyml_esp32s3_pareto.png`: Flash footprint (KB) vs Peak SRAM utilization (KB) vs Inference Latency (ms) across FP32, INT8 Static, and TFLite Micro runtime.
3. `tinyml_safety_guardrail_flow.png`: Firmware execution state diagram illustrating thermal throttling, battery preservation, and denomination confidence gating.
