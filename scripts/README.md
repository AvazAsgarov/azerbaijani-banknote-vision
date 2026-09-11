# Production Execution Scripts (scripts/)

## Overview
This directory contains the production-grade CLI tools, data pipeline runners, experiment orchestration utilities, training scripts, and hardware bridge interfaces for the Azerbaijani Banknote Computer Vision research initiative.

---

## Script Architecture

### 1. Standalone Experiment Runners (Reproducible Python Pipelines)
- **run_experiment_1.py**: Experiment 1 (Architecture Battle: YOLOv8m, YOLO11m, RT-DETR-L). Evaluates architectural inductive biases, parameter efficiency, and mAP@0.50.
- **run_experiment_2.py**: Experiment 2 (Data Augmentation Ablation: Raw Baseline, Geometric, Photometric, Full Composite). Evaluates spatial vs color augmentations and generalization gaps.
- **run_experiment_3.py**: Experiment 3 (Color Space & Shortcut Learning: Full RGB, HSV Chromaticity Decoupling, Monochrome Grayscale Shortcut Test). Quantifies color shortcut reliance.
- **run_experiment_4.py**: Experiment 4 (Multiscale Resolution Dynamics: 320x320 Mobile Edge, 640x640 Standard Baseline, 1280x1280 High-Fidelity Numismatic).
- **run_experiment_5.py**: Experiment 5 (Model Compression & Edge Optimization: FP32 Baseline, FP16 Half Precision, INT8 PTQ, 25% Structured Filter Pruning).
- **run_experiment_6.py**: Experiment 6 (Explainable AI & Attention Diagnostics: C2PSA / SPPF EigenCAM heatmaps and numismatic alignment metrics).
- **run_experiment.py**: Unified CLI orchestrator for running Experiments 1 through 6 locally or dispatching to remote GPU cluster.
- **deploy_master_pipeline.py**: Orchestrates and deploys the master sequential experiment pipeline to the remote GPU cluster daemon.

- **sync_daemon.py**: Production-grade background sync daemon continuously monitoring the GPU cluster, running remote test evaluations, pulling completed experiments, eliminating stale batches, and enforcing zero empty directories.

### 2. Dataset Curation & Quality Gates
- **build_dataset.py**: Canonical dataset compilation pipeline from raw captures into YOLO layout.
- **deploy_dataset.py**: Deploys dataset package to remote GPU training cluster over authenticated socket streaming.
- **run_zero_leakage_split.py**: Executes strict environment-isolated stratified split across train, validation, and test sets.
- **verify_split_distribution.py**: Validates class distribution balance and guarantees 0% background leakage across splits.
- **run_dataset_audit.py**: Audits Inter-Annotator Agreement (mIoU, Cohen's Kappa), exact binary duplicates, and near-duplicate pHash clusters.
- **ingest_fixed_labels.py**: Ingestion and normalization utility for verified Roboflow YOLO bounding box coordinates.
- **preflight_gate.py**: Enterprise pre-flight gatekeeper verifying data invariants and test suites prior to GPU compute allocation.

### 3. Exploratory Data Analysis & Foundation Models
- **run_advanced_eda.py**: Comprehensive multi-dimensional exploratory data analysis pipeline (photometrics, geometry, chromaticity).
- **dino.py**: DINOv2 ViT-L/14 self-supervised foundation feature extractor and linear probe evaluator.
- **generate_diagnostic_plots.py**: Produces publication-grade feature distribution and pairwise cosine affinity figures.

### 4. Edge Deployment & Inference
- **generate_test_inferences.py**: Generates qualitative detection overlays with bounding box coordinates across test set images.
- **diagnose_tinyml_metrics.py**: Strict IoU-matched mAP@0.50, precision, recall, and F1 diagnostic analyzer for edge models.
- **generate_tinyml_plots.py**: Publication-quality Pareto Frontier and Multi-Tier Safety Guardrail architecture figures for ESP32-S3 deployment.
- **mobile_bridge.py**: High-performance HTTP inference server powering real-time mobile app and smart glasses vision feeds.