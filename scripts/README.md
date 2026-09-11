# Production Execution Scripts (`scripts/`)

## Overview
This directory contains the production-grade CLI tools, dataset curation pipelines, deep learning experiment runners, TinyML training engines, and hardware bridge interfaces for the Azerbaijani Banknote Computer Vision research initiative.

---

## Script Architecture & Catalog

### 1. Deep Learning Experiment Suite (Experiments 1–6)
* **`run_experiment.py`**: Unified CLI orchestrator for executing Experiments 1 through 6 locally or dispatching to the remote GPU cluster.
* **`run_experiment_1.py`**: **Experiment 1 (Architecture Battle)** — Systematic benchmarking of YOLOv8m, YOLO11m, and RT-DETR-L on zero-leakage test data (latency, parameter efficiency, and mAP@0.50).
* **`run_experiment_2.py`**: **Experiment 2 (Data Augmentation Ablation)** — Evaluates Raw Baseline vs Geometric vs Photometric vs Full Composite augmentations to quantify generalization deltas.
* **`run_experiment_3.py`**: **Experiment 3 (Color Space & Shortcut Learning)** — Cross-evaluates RGB vs HSV Chromaticity Decoupling vs Monochrome Grayscale models to verify numismatic structural feature reliance.
* **`run_experiment_4.py`**: **Experiment 4 (Multiscale Resolution Dynamics)** — Evaluates trade-offs between 320×320 mobile edge, 640×640 standard baseline, and 1280×1280 high-fidelity numismatic crops.
* **`run_experiment_5.py` / `run_experiment_5_local.py`**: **Experiment 5 (Model Compression & Quantization)** — FP32 baseline vs FP16 mixed precision vs INT8 Post-Training Quantization (PTQ) vs 25% L1 structured filter pruning.
* **`run_experiment_6.py` / `run_experiment_6_local.py`**: **Experiment 6 (Explainable AI & Attention Diagnostics)** — C2PSA and SPPF EigenCAM attention heatmap extraction and Numismatic Motif Alignment evaluation across 175 test samples.
* **`deploy_master_pipeline.py`**: Orchestrates and deploys the sequential master experiment pipeline to the remote GPU cluster daemon.
* **`sync_daemon.py` / `sync_exp_artifacts.py`**: Background synchronization daemon continuously monitoring the GPU cluster, pulling completed experiment packages, and enforcing directory invariants.

### 2. Dataset Curation, Ingestion & Quality Gates
* **`build_dataset.py`**: Canonical dataset compilation pipeline from raw captures into verified YOLO directory layout.
* **`run_zero_leakage_split.py`**: Executes strict environment-isolated clustering split across training (70%), validation (15%), and test (15%) partitions.
* **`verify_split_distribution.py`**: Validates class distribution balance and mathematically verifies 0% background leakage across splits.
* **`run_dataset_audit.py`**: Audits Inter-Annotator Agreement (mIoU, Cohen's Kappa), exact binary duplicates, and near-duplicate perceptual hash (pHash) clusters.
* **`ingest_fixed_labels.py`**: Ingestion and coordinate normalization utility for Roboflow verified YOLO bounding boxes.
* **`preflight_gate.py`**: Enterprise pre-flight gatekeeper verifying data invariants, file hygiene, and unit test suites prior to GPU compute allocation.
* **`deploy_dataset.py`**: Packages and deploys the curated dataset to remote GPU training infrastructure over authenticated socket streaming.

### 3. Exploratory Data Analysis & Foundation Models
* **`run_advanced_eda.py`**: Multi-dimensional exploratory data analysis pipeline (photometrics, geometry, chromaticity, aspect ratios).
* **`dino.py`**: DINOv2 ViT-L/14 self-supervised foundation feature extractor, embedding cache builder, and linear probe evaluator.
* **`generate_diagnostic_plots.py`**: Produces publication-grade feature distribution and pairwise cosine affinity figures.

### 4. Edge TinyML, Mobile Bridge & Inference
* **`train_fomo.py`**: Lightweight centroid-based object detector (FOMO / MobileNetV1) training and ONNX export pipeline for microcontrollers.
* **`diagnose_tinyml_metrics.py`**: Strict IoU-matched mAP@0.50, precision, recall, and F1 diagnostic analyzer for edge neural networks.
* **`generate_tinyml_plots.py`**: Publication-quality Pareto Frontier and Multi-Tier Safety Guardrail architecture figures for ESP32-S3 deployment.
* **`generate_test_inferences.py`**: Generates qualitative detection overlays with bounding box coordinates across test set images.
* **`mobile_bridge.py`**: High-performance HTTP/WebSocket inference bridge connecting camera feeds and ESP32 smart glasses to GPU vision backends.

### 5. Remote Infrastructure Bridge
* **`remote_exec.py`**: Remote execution utility against Jupyter kernel over WebSockets with real-time stdout/stderr streaming.
* **`remote_gpu_bridge.py`**: GPU environment bridge, dependency synchronizer, and nvidia-smi telemetry monitor.

---

## Script Ownership & Team Attribution

| Component / Track | Lead Contributor | Key Scripts |
| :--- | :--- | :--- |
| **System Architecture & Orchestration** | **Avaz Asgarov** | `run_zero_leakage_split.py`, `preflight_gate.py`, `mobile_bridge.py` |
| **Datasets & Architecture Benchmarking** | **Gulnar Babazade** | `build_dataset.py`, `ingest_fixed_labels.py`, `run_experiment_1.py`, `run_experiment_4.py` |
| **Photometrics, Colorspace & Compression** | **Hasan Mammadov** | `run_advanced_eda.py`, `run_experiment_3.py`, `run_experiment_5.py` |
| **TinyML Edge & Microcontroller Inference**| **Kazim Mammadli** | `train_fomo.py`, `diagnose_tinyml_metrics.py`, `generate_tinyml_plots.py`, `run_experiment_2.py` |
| **Foundation Models, XAI & Telemetry** | **Nicat Alaskarli** | `dino.py`, `run_experiment_6.py`, `generate_diagnostic_plots.py`, `remote_gpu_bridge.py` |
