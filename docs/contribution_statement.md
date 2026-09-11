# Team Contribution Statement
### Deep Learning Final Project · AI Academy · Cohort I · 2026
**Project:** Azerbaijani Banknote Vision (AZN-Vision)  
**Repository:** https://github.com/AvazAsgarov/azerbaijani-banknote-vision  
**Release:** `v1.0.0` · 85 Commits · 5 Contributors

---

## 1. Team Contribution Summary Matrix

| Member Name | GitHub Username | Commits | Code (%) | Technical Pillars & Experiments | Slides | Overall Allocation |
| :--- | :--- | :---: | :---: | :--- | :---: | :---: |
| **Avaz Asgarov** | [AvazAsgarov](https://github.com/AvazAsgarov) | 17 | 20% | Lead Architect, Modular Core Architecture, Zero-Leakage Stratification Splitter, EDA & Integrity, React Native Mobile Companion App, Master Verification Harness | Slides 1–8 | **20%** |
| **Gulnar Babazade** | [gulnarbabazade](https://github.com/gulnarbabazade) | 17 | 20% | Deep Learning Architecture Battle (Exp 1), Multiscale Resolution Scaling (Exp 4), Annotation Standards & Ingestion Quality Gates, Dataset Augmentation Policy | Slides: Exp 1 & Exp 4 | **20%** |
| **Hasan Mammadov** | [Hasawr](https://github.com/Hasawr) | 17 | 20% | Photometrics & Chromatic Analysis, Color Space Shortcut Learning (Exp 3), Asymmetric Cost-Sensitive Loss Penalty ($\beta_c$), Model Compression & Quantization (Exp 5) | Slides: Exp 3 & Exp 5 | **20%** |
| **Kazim Mammadli** | [KazimMammadli](https://github.com/KazimMammadli) | 17 | 20% | Data Augmentation Impact Ablation (Exp 2), TinyML YOLO-FastestV2 Edge Pipeline, CIoU Loss & Anchor Optimization, ESP32-S3 Firmware & 4-Tier Safety Guard | Slides: Exp 2 & TinyML | **20%** |
| **Nicat Alaskarli** | [NicatAlaskarli](https://github.com/NicatAlaskarli) | 17 | 20% | Foundation Model Representations (DINOv2 ViT-L/14 Probing), Explainable AI & EigenCAM Heatmaps (Exp 6), Low-Latency Hardware Camera Streaming Bridge | Slides: Exp 6 & Hardware | **20%** |
| **Total** | — | **85** | **100%** | **6 Empirical Studies + Foundation Model + TinyML Edge + Mobile** | **All Slides** | **100% (Balanced)** |

---

## 2. Detailed Individual Contributions & Deliverables

### Avaz Asgarov — Lead Architect & Systems Engineer (20%)
* **Core Deliverables:**
  - Designed the modular repository architecture and clean layer separation across `src/`, `scripts/`, `configs/`, and `tests/`.
  - Implemented the zero-leakage environment-isolated stratification engine (`src/data/splitter.py`, `validator.py`, `scripts/run_zero_leakage_split.py`).
  - Executed dataset integrity audits, pairwise Hamming distance deduplication, and spatial bounding box geometry profiling (`src/eda/integrity.py`, `spatial.py`, `deep_dives.py`).
  - Built the cross-platform React Native Mobile Companion Application (`mobile/`), including the audio guidance synthesizer, transaction wallet accumulator, and Bluetooth LE protocol.
  - Engineered `scripts/mobile_bridge.py` with `CameraManager` singleton, MJPEG stream handlers, and dynamic model switching.
  - Authored the automated master verification harness (`run_all.py`), pre-flight validation gatekeeper (`scripts/preflight_gate.py`), and containerized Docker deployment (`Dockerfile`, `docker-compose.yml`).
  - **Slide Deck Delivery:** Slides 1–8 (Executive Overview, Real-World Problem Context, Methodology, Core Architectural Design, System Dataflow).

### Gulnar Babazade — Dataset & Benchmark Specialist (20%)
* **Core Deliverables:**
  - Designed and executed **Experiment 1 (Deep Learning Architecture Battle)**: systematically benchmarking YOLOv8m, YOLO11m, and RT-DETR-L on zero-leakage test data (`scripts/run_experiment_1.py`).
  - Designed and executed **Experiment 4 (Multiscale Resolution Scaling Dynamics)**: evaluating trade-offs between 320x320 mobile edge, 640x640 standard baseline, and 1280x1280 high-fidelity numismatic crops (`scripts/run_experiment_4.py`).
  - Standardized the dataset annotation protocol, edge case rules, and truncation tolerances (`docs/annotation_guidelines.md`).
  - Engineered the automated Roboflow YOLO label ingestion, bounding box coordinate normalization, and validation suite (`scripts/ingest_fixed_labels.py`, `src/data/label_validator.py`, `src/data/augment_policy.py`).
  - Created publication-ready figures for experimental evaluations and split distributions (`reports/figures/experiments_unified/`).
  - **Slide Deck Delivery:** Slides on Experiment 1 (Architecture Battle) and Experiment 4 (Multiscale Resolution Dynamics).

### Hasan Mammadov — Photometrics & Compression Specialist (20%)
* **Core Deliverables:**
  - Conducted multi-dimensional photometric profiling, color channel variance estimation, and illumination robustness analysis (`src/eda/photometric.py`, `chromatic.py`, `style.py`, `src/eda/channel_stats.py`).
  - Designed and executed **Experiment 3 (Color Space & Shortcut Learning)**: cross-evaluating RGB, HSV chromaticity decoupling, and monochrome grayscale models to prove model reliance on authentic geometric guilloche patterns rather than color shortcuts (`scripts/run_experiment_3.py`, `src/eda/color_space_convertor.py`).
  - Formulated the asymmetric cost-sensitive loss penalty vector ($\beta_c$) heavily penalizing high-denomination false positives (100 AZN and 200 AZN) to safeguard visually impaired users (`src/core/asymmetric_loss_config.py`).
  - Implemented and evaluated **Experiment 5 (Model Compression & Edge Optimization)**: FP32 baseline vs FP16 half-precision vs INT8 Post-Training Quantization (PTQ) vs 25% L1 unstructured filter pruning (`scripts/run_experiment_5.py`, `src/eda/compression_utils.py`).
  - **Slide Deck Delivery:** Slides on Experiment 3 (Color Space & Shortcut Learning) and Experiment 5 (Model Compression & Edge Optimization).

### Kazim Mammadli — TinyML & Augmentation Specialist (20%)
* **Core Deliverables:**
  - Designed and executed **Experiment 2 (Data Augmentation Impact Ablation)**: quantifying generalization deltas across Raw Baseline, Geometric, Photometric, and Full Composite augmentations (`scripts/run_experiment_2.py`).
  - Implemented the TinyML edge neural network pipeline based on YOLO-FastestV2 (`src/tinyml/`): lightweight backbone, CIoU bounding box loss (`src/tinyml/ciou_loss.py`), asymmetric focal classification, and k-means anchor optimization (`src/tinyml/anchor_optimizer.py`).
  - Quantized the model to INT8 FlatBuffer (`artifacts/tinyml/yolo_fastestv2_int8.tflite`, 289.8 KB) and generated the C++ byte array header for microcontroller compilation (`artifacts/tinyml/yolo_fastestv2_int8.h`).
  - Built SRAM budget verification models (`src/tinyml/sram_budget.py`) confirming fit within the 512 KB SRAM of Seeed Studio XIAO ESP32-S3 Sense.
  - Developed and verified embedded firmware (`firmware/esp32s3_smart_glasses/`) with camera pipeline, model runner, and 4-tier temporal safety guardrail.
  - **Slide Deck Delivery:** Slides on Experiment 2 (Data Augmentation Ablation) and TinyML Microcontroller Deployment.

### Nicat Alaskarli — Foundation Models, XAI & Hardware Specialist (20%)
* **Core Deliverables:**
  - Built the self-supervised foundation model feature extraction pipeline using DINOv2 ViT-L/14, persisting dense representation matrices with an on-disk LRU embedding cache (`src/embeddings/embedding_cache.py`, `src/data/embeddings.py`, `scripts/dino.py`).
  - Implemented linear probe classification achieving 100% test accuracy on zero-leakage partitions (`src/embeddings/linear_probe.py`).
  - Designed and executed **Experiment 6 (Explainable AI & Attention Diagnostics)**: extracting C2PSA / SPPF EigenCAM heatmaps and evaluating Numismatic Motif Alignment across 175 test samples (`scripts/run_experiment_6.py`, `src/xai/eigencam_config.py`).
  - Integrated the ESP32 Wi-Fi streaming camera firmware (`firmware/esp32_wifi_camera/`) and engineered the low-latency MJPEG streaming hardware bridge (`src/hardware/mjpeg_streamer.py`, `scripts/mobile_bridge.py`).
  - **Slide Deck Delivery:** Slides on Experiment 6 (Explainable AI & EigenCAM Attention) and Hardware Camera Integration.

---

## 3. Git Workflow & Governance

The repository development strictly adhered to enterprise software engineering standards:
* **85 Total Commits** cleanly authored and balanced (exactly 17 commits per member).
* **Feature Branches & Pull Requests** used for each major milestone with mandatory code review sign-offs.
* **Annotated Release Tag** `v1.0.0` published with release deliverables.
* **100% Passing Automated Tests** across Python (191 tests) and Mobile Jest (42 tests).

---

## 4. Team Verification & Sign-Off

All five team members have reviewed and approved this contribution statement.

* **Avaz Asgarov** ([@AvazAsgarov](https://github.com/AvazAsgarov)) — *Lead Architect & Systems Engineer* — ✅ Approved
* **Gulnar Babazade** ([@gulnarbabazade](https://github.com/gulnarbabazade)) — *Dataset & Benchmark Specialist* — ✅ Approved
* **Hasan Mammadov** ([@Hasawr](https://github.com/Hasawr)) — *Photometrics & Compression Specialist* — ✅ Approved
* **Kazim Mammadli** ([@KazimMammadli](https://github.com/KazimMammadli)) — *TinyML & Augmentation Specialist* — ✅ Approved
* **Nicat Alaskarli** ([@NicatAlaskarli](https://github.com/NicatAlaskarli)) — *XAI & Foundation Models Specialist* — ✅ Approved
