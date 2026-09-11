# Configuration Files (`configs/`)

## Overview
Declarative YAML specification files defining dataset splits, model hyperparameters, ablation targets, quantization schemes, and hardware constraints.

---

## Configuration Catalog

| Configuration File | Scope / Pillar | Responsible | Key Specifications |
|:---|:---|:---|:---|
| **`data.yaml`** | YOLO Dataset Specification | Gulnar Babazade | 7 denomination classes (`001_azn` → `200_azn`), `train`/`val`/`test` relative paths |
| **`exp1_architecture_battle.yaml`** | Experiment 1 | Gulnar Babazade | Multi-architecture comparison: YOLOv8m vs YOLO11m vs RT-DETR-L |
| **`exp2_augmentation.yaml`** | Experiment 2 | Kazim Mammadli / Gulnar | 4 augmentation policies: Raw, Geometric, Photometric, Full Composite |
| **`exp3_color_space.yaml`** | Experiment 3 | Hasan Mammadov | Color space shortcut learning: RGB vs HSV Decoupling vs Grayscale |
| **`exp4_resolution_scaling.yaml`** | Experiment 4 | Gulnar Babazade | Multiscale dynamics: 320×320 vs 640×640 vs 1280×1280 |
| **`exp5_compression.yaml`** | Experiment 5 | Hasan Mammadov | Post-training compression: FP32, FP16, INT8 PTQ, 25% L1 Pruning |
| **`exp6_xai_eigencam.yaml`** | Experiment 6 | Nicat Alaskarli | Explainable AI: C2PSA / SPPF EigenCAM attention layers, motif threshold 0.50 |
| **`tinyml_edge.yaml`** | TinyML Edge Deployment | Kazim Mammadli | ESP32-S3 hardware constraints: 240 MHz, 512 KB SRAM, INT8 FlatBuffer |

