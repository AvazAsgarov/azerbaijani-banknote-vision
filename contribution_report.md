# Team Contribution Report
### Deep Learning Final Project · AI Academy · Cohort I 2026
**Project**: Azerbaijani Banknote Vision (AZN-Vision)  
**Repository**: https://github.com/AvazAsgarov/azerbaijani-banknote-vision  
**Release Tag**: 1.0.0

---

## 1. Team Contribution Summary Matrix

| Member Name | GitHub Username | Code (%) | Report Sections | Experiments & Technical Pillars | Presentation Slides | Overall Allocation |
| :--- | :--- | :---: | :--- | :--- | :---: | :---: |
| **Avaz Asgarov** | AvazAsgarov | 20% | Abstract, Architecture, System Design, Discussion | Lead Architect, Core Framework, Stratification Splitter, EDA & Integrity, Mobile Companion App, Master Harness | Slides 1–8 | **20%** |
| **Gulnar Babazade** | gulnarbabazade | 20% | Methodology, Benchmark Evaluation, Scaling Analysis | Deep Learning Architecture Battle (Exp 1), Multiscale Resolution Scaling (Exp 4), Annotation Guidelines, Ingestion & Quality Gates | Slides: Exp 1 & Exp 4 | **20%** |
| **Hasan Mammadov** | Hasawr | 20% | Photometric Profiling, Robustness, Compression | Photometrics & Chromatic Analysis, Color Space Shortcut Learning (Exp 3), Asymmetric Risk Penalty (\\beta_c), Model Compression (Exp 5) | Slides: Exp 3 & Exp 5 | **20%** |
| **Kazim Mammadli** | KazimMammadli | 20% | Augmentation Ablations, Edge Microcontroller Integration | Data Augmentation Ablation (Exp 2), TinyML YOLO-FastestV2 Edge Pipeline, ESP32-S3 Firmware Runner, Edge Benchmarks & Pareto Analysis | Slides: Exp 2 & TinyML | **20%** |
| **Nicat Alaskarli** | NicatAlaskarli | 20% | Foundation Representations, Explainability, Hardware Feeds | Foundation Model Embeddings (DINOv2 ViT-L/14 Probing), Explainable AI & EigenCAM Heatmaps (Exp 6), Hardware Camera Streaming Bridge | Slides: Exp 6 & Hardware | **20%** |
| **Total** | — | **100%** | **Full IEEE Conference Paper** | **6 Empirical Studies + Foundation Model + TinyML Edge + Mobile** | **All 21 Slides** | **100% (Balanced)** |

---

## 2. Individual Member Contributions & Deliverables

### Avaz Asgarov (AvazAsgarov · vaz.asgarov45@gmail.com)
* **Role:** Lead Architect & Systems Engineer (20%)
* **Core Deliverables:**
  - Designed the modular repository architecture and clean layer separation across src/, scripts/, configs/, and 	ests/.
  - Implemented the zero-leakage environment-isolated stratification engine (src/data/splitter.py, alidator.py, scripts/run_zero_leakage_split.py).
  - Executed dataset integrity audits, pairwise Hamming distance deduplication, and spatial bounding box geometry profiling (src/eda/integrity.py, spatial.py, deep_dives.py).
  - Built the cross-platform React Native Mobile Companion Application (mobile/), including the audio guidance synthesizer, transaction wallet accumulator, and Bluetooth LE protocol.
  - Authored the automated master verification harness (
un_all.py), pre-flight validation gatekeeper (scripts/preflight_gate.py), and containerized Docker deployment (Dockerfile, docker-compose.yml).
  - Participated in physical banknote image collection across authentic retail and lighting scenarios.
  - **Slide Deck Delivery:** Slides 1–8 (Executive Overview, Real-World Problem Context, Methodology, Core Architectural Design, System Dataflow).

### Gulnar Babazade (gulnarbabazade · gulnarbabazadee122@gmail.com)
* **Role:** Deep Learning Benchmark & Multiscale Scaling Specialist (20%)
* **Core Deliverables:**
  - Designed and executed **Experiment 1 (Deep Learning Architecture Battle)**: systematically benchmarking YOLOv8m, YOLO11m, and RT-DETR-L on zero-leakage test data (scripts/run_experiment_1.py, mAP@0.50, parameter efficiency, inference latency).
  - Designed and executed **Experiment 4 (Multiscale Resolution Scaling Dynamics)**: evaluating trade-offs between 320x320 mobile edge, 640x640 standard baseline, and 1280x1280 high-fidelity numismatic crops (scripts/run_experiment_4.py).
  - Standardized the dataset annotation protocol, edge case rules, and truncation tolerances (docs/annotation_guidelines.md).
  - Engineered the automated Roboflow YOLO label ingestion, bounding box coordinate normalization, and validation suite (scripts/ingest_fixed_labels.py, src/data/deduplicate.py).
  - Participated in physical banknote image acquisition and quality control verification across 2,522 partitioned images.
  - **Slide Deck Delivery:** Slides on Experiment 1 (Architecture Battle) and Experiment 4 (Multiscale Resolution Dynamics).

### Hasan Mammadov (Hasawr · hasanmammadov85@gmail.com)
* **Role:** Photometrics, Robustness & Model Compression Specialist (20%)
* **Core Deliverables:**
  - Conducted multi-dimensional photometric profiling, color channel variance estimation, and illumination robustness analysis (src/eda/photometric.py, chromatic.py, style.py).
  - Designed and executed **Experiment 3 (Color Space & Shortcut Learning)**: cross-evaluating RGB, HSV chromaticity decoupling, and monochrome grayscale models to prove model reliance on authentic geometric guilloche patterns rather than color shortcuts (scripts/run_experiment_3.py).
  - Formulated the asymmetric cost-sensitive loss penalty vector (\\beta_c) heavily penalizing high-denomination false positives (100 AZN and 200 AZN) to safeguard visually impaired users.
  - Implemented and evaluated **Experiment 5 (Model Compression & Edge Optimization)**: FP32 baseline vs FP16 half-precision vs INT8 Post-Training Quantization (PTQ) vs 25% L1 unstructured filter pruning (scripts/run_experiment_5.py, scripts/run_experiment_5_local.py).
  - Participated in physical banknote image collection across varying lighting conditions and illumination angles.
  - **Slide Deck Delivery:** Slides on Experiment 3 (Color Space & Shortcut Learning) and Experiment 5 (Model Compression & Edge Optimization).

### Kazim Mammadli (KazimMammadli · kazim.memmedli.005@gmail.com)
* **Role:** Edge TinyML & Data Augmentation Specialist (20%)
* **Core Deliverables:**
  - Designed and executed **Experiment 2 (Data Augmentation Impact Ablation)**: quantifying generalization deltas across Raw Baseline, Geometric, Photometric, and Full Composite augmentations (scripts/run_experiment_2.py).
  - Implemented the TinyML edge neural network pipeline based on YOLO-FastestV2 (src/tinyml/): lightweight backbone, CIoU bounding box loss, asymmetric focal classification, and anchor box optimization.
  - Quantized the model to INT8 FlatBuffer (rtifacts/tinyml/yolo_fastestv2_int8.tflite, 289.8 KB) and generated the C++ byte array header for microcontroller compilation (yolo_fastestv2_int8.h).
  - Developed and verified the embedded firmware for Seeed Studio XIAO ESP32-S3 (irmware/esp32s3_smart_glasses/): camera pipeline, model runner, and 4-tier temporal safety guardrail.
  - Built diagnostic benchmarking tools and publication-quality Pareto Frontier curves (scripts/diagnose_tinyml_metrics.py, scripts/generate_tinyml_plots.py).
  - Participated in physical banknote image collection and real-world embedded hardware capture tests.
  - **Slide Deck Delivery:** Slides on Experiment 2 (Data Augmentation Ablation) and TinyML Microcontroller Deployment.

### Nicat Alaskarli (NicatAlaskarli · 
icat.alaskarli@gmail.com)
* **Role:** Foundation Models, Explainable AI (XAI) & Hardware Streaming Specialist (20%)
* **Core Deliverables:**
  - Designed and executed **Experiment 6 (Explainable AI & Attention Diagnostics)**: extracting C2PSA / SPPF EigenCAM heatmaps and evaluating Numismatic Motif Alignment across 175 test samples (scripts/run_experiment_6.py, scripts/run_experiment_6_local.py).
  - Built the self-supervised foundation model feature extraction pipeline using DINOv2 ViT-L/14, persisting dense representation matrices and performing linear probe classification across zero-leakage splits (src/data/embeddings.py, scripts/dino.py).
  - Integrated the ESP32 Wi-Fi streaming camera firmware (irmware/esp32_wifi_camera/) and engineered the low-latency MJPEG streaming hardware bridge (src/hardware/esp32_camera.py, scripts/mobile_bridge.py).
  - Validated explainability alignment with physical banknote numismatic security features (guilloche, SPARK optical inks, coat of arms).
  - Participated in physical banknote image collection and camera sensor calibration.
  - **Slide Deck Delivery:** Slides on Experiment 6 (Explainable AI & EigenCAM Attention) and Hardware Camera Integration.

---

## 3. Team Sign-Off & Verification

All five team members have reviewed, approved, and verified the contributions documented in this report.
The codebase, experimental findings, firmware, mobile companion, and slide presentations represent a completely balanced, collaborative research endeavor.

* **Avaz Asgarov** (AvazAsgarov) — *Approved*
* **Gulnar Babazade** (gulnarbabazade) — *Approved*
* **Hasan Mammadov** (Hasawr) — *Approved*
* **Kazim Mammadli** (KazimMammadli) — *Approved*
* **Nicat Alaskarli** (NicatAlaskarli) — *Approved*
