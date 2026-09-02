# Experiment 5: Model Compression & Edge Optimization

## Overview
Ablation of model compression, quantization, and structured pruning strategies on the champion YOLO11m detector.

## Compression Summary
| Arm | Precision | Size (MB) | Test mAP@0.50 | GPU Latency |
| :--- | :---: | :---: | :---: | :---: |
| `arm1_fp32` | FP32 | 77.02 MB | 0.9555 | 30.33 ms |
| `arm2_fp16` | FP16 | 38.63 MB | 0.9557 | 29.79 ms |
| `arm3_int8` | INT8 | 19.37 MB | 0.9430 | 20.26 ms |
| `arm4_pruned_25` | FP32-Pruned | 38.67 MB | 0.9314 | 45.2 ms |

## Directory Structure
- `arm1_fp32/`: Full precision baseline.
- `arm2_fp16/`: Half precision evaluation and weights.
- `arm3_int8/`: Post-training quantization.
- `arm4_pruned_25/`: 25% structured pruning.
- `figures/compression_pareto_frontier.png`: 300 DPI publication figure.
- `exp5_compression_comparison.csv` & `exp5_compression_comparison.json`: Metrics tables.
- `exp5_diagnostic_report.md`: Formal diagnostic report.
