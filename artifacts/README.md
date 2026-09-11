# Research Artifacts & Evaluation Manifests (`artifacts/`)

**Target Domain:** Azerbaijani Banknote Vision & Edge Intelligence Pipeline  
**Dataset Scale:** 2,522 Zero-Leakage Images across 7 Denominations  
**Status:** Validated, Synchronized, and Audited for Pre-Release  
**Cryptographic Integrity:** Governed by `artifacts/provenance_manifest.json`  

---

## 1. Overview & Architectural Role

The `artifacts/` directory houses the empirical outputs, precomputed neural representations, evaluation metrics, diagnostic curves, explainability heatmaps, and edge-deployable quantized models produced across the entire research lifecycle.

All stored artifacts strictly adhere to zero-data-leakage mandates, reproducibility standards, and git storage governance policies.

```
artifacts/
├── embeddings/                             <- DINOv2 ViT-L/14 dense representation matrices
│   ├── embeddings_dinov2_vitl14.npy        <- Raw float32 feature matrix (2522x1024)
│   ├── embeddings_dinov2_vitl14.h5         <- Compressed HDF5 dataset container
│   ├── embeddings_metadata.parquet         <- Sample-level metadata mapping
│   └── README.md                           <- Technical specifications & probe metrics
├── experiments/                            <- Master empirical evaluation suite
│   ├── exp1_architecture_battle/           <- YOLOv8m vs YOLO11m vs RT-DETR-L vs DINOv2
│   ├── exp2_data_augmentation/             <- Raw vs Geometric vs Photometric vs Composite
│   ├── exp3_color_space_shortcut/          <- RGB vs HSV vs Grayscale shortcut analysis
│   ├── exp4_resolution_scaling/            <- 320x320 vs 640x640 vs 1280x1280 scaling
│   ├── exp5_model_compression/             <- FP32 vs FP16 vs INT8 PTQ vs 25% L1 Pruning
│   └── exp6_explainability_xai/            <- C2PSA EigenCAM heatmaps (175 test samples)
├── reports/                                <- Canonical JSON verification manifests
│   ├── dataset_audit_report.json           <- Inter-annotator agreement & deduplication
│   ├── split_summary.json                  <- Zero-leakage partition metrics (1760/379/383)
│   ├── dino_linear_probe_report.json       <- Linear probe baseline accuracy & per-class F1
│   ├── eda_metrics.json                    <- Color centroids, spatial geometry & PCA
│   ├── preflight_verification_report.json  <- Gatekeeper compute clearance
│   ├── coverage.json                       <- Pipeline unit test coverage (>95%)
│   └── README.md                           <- Detailed report registry & schema guide
├── tinyml/                                 <- Microcontroller deployment deliverables
│   ├── yolo_fastestv2_int8.tflite          <- INT8 quantized flatbuffer (289.8 KB, b'TFL3')
│   ├── yolo_fastestv2_int8.h               <- C++ byte array header for ESP32-S3 firmware
│   ├── training_metrics.json               <- TinyML loss curves & mAP convergence
│   ├── edge_benchmark_results.json         <- Memory arena, latency & throughput profiles
│   └── README.md                           <- Embedded hardware deployment specifications
├── provenance_manifest.json                <- Cryptographic SHA-256 hash manifest of all stages
└── README.md                               <- Master directory catalog (this document)
```

---

## 2. Directory Specifications & Artifact Registry

### 2.1. Foundation Embeddings (`artifacts/embeddings/`)
* **Model:** Meta DINOv2 ViT-L/14 (`dinov2_vitl14`, 304.3M parameters, patch size 14x14).
* **Dimensionality:** 1,024 float32 dimensions per sample across all 2,522 dataset images.
* **Storage Formats:**
  - `embeddings_dinov2_vitl14.npy` (9.85 MB): Contiguous binary matrix for linear probing and matrix operations.
  - `embeddings_dinov2_vitl14.h5` (6.57 MB): HDF5 archive with dataset attributes and provenance tags.
  - `embeddings_metadata.parquet` (0.02 MB): Metadata aligning indices with `filename`, `class_name`, `split`, and `meta_cluster`.
* **Empirical Finding:** Zero-shot linear probe achieved 42.56% test accuracy and 0.0% F1 on 20 AZN, validating the necessity of specialized fine-tuned detectors.

### 2.2. Empirical Experiments Suite (`artifacts/experiments/`)

| Experiment | Focus Area | Candidate Models / Arms | Primary Benchmark Metric | Key Finding / Verdict |
| :--- | :--- | :--- | :---: | :--- |
| **Exp 1** | Architecture Battle | YOLOv8m, YOLO11m, RT-DETR-L, DINOv2 | Test $\text{mAP}_{50}$, Latency | **YOLOv8m wins accuracy (95.96% mAP50)**; YOLO11m wins speed (7.18 ms, 139.2 FPS). Transformers rejected. |
| **Exp 2** | Data Augmentation | Raw, Geometric, Photometric, Full Composite | Test $\text{mAP}_{50}$, Robustness | **Full Composite achieves 95.82% mAP50** (+3.2% over Raw baseline, proving illumination invariance). |
| **Exp 3** | Color Space Shortcut | Full RGB, HSV Space, Grayscale Shortcut | Cross-eval $\Delta\text{mAP}$, Degradation | **RGB-to-Gray causes catastrophic drop (-34.6% mAP)**, confirming color is a primary recognition shortcut. |
| **Exp 4** | Resolution Scaling | 320x320, 640x640, 1280x1280 | Small-object $\text{AP}$, FLOPs | **640x640 is the optimal Pareto point** (95.9% mAP, 18.2 ms). 1280x adds 4x compute for +0.4% mAP. |
| **Exp 5** | Model Compression | FP32, FP16, INT8 PTQ, 25% L1 Structured Pruning | Size (MB), Latency, Drop | **FP16 retains 100% accuracy at 50% size (24.1 MB)**. INT8 PTQ compresses to 12.3 MB with 2.1% drop. |
| **Exp 6** | Explainability (XAI) | C2PSA EigenCAM Attention Heatmaps | IoU to Bounding Box, Alignment | **91.4% bounding box attention alignment**, confirming feature focus on denominational emblems. |

> [!NOTE]
> `artifacts/experiments/exp6_explainability_xai/heatmaps/` houses 175 publication-quality EigenCAM visual heatmaps covering the zero-leakage test partition across all 7 denominations.

### 2.3. TinyML Edge Deliverables (`artifacts/tinyml/`)
* **Target Hardware:** ESP32-S3 (Xtensa dual-core 32-bit LX7, 512 KB SRAM, 8 MB external PSRAM).
* **Architecture:** YOLO-FastestV2 ultra-lightweight anchor-based detector.
* **Artifacts:**
  - `yolo_fastestv2_int8.tflite` (289.8 KB): Fully integer-quantized FlatBuffer with verified `b'TFL3'` magic bytes.
  - `yolo_fastestv2_int8.h` (1.81 MB): Pure C++ byte array header ready for compilation in ESP-IDF / Arduino IDE.
  - `edge_benchmark_results.json`: Benchmarked at ~185 ms latency per inference on ESP32-S3 with <350 KB tensor arena allocation.

### 2.4. Verification Reports (`artifacts/reports/`)
* Structured machine-readable audits capturing inter-annotator agreement (IAA mIoU 77.24%, Cohen's Kappa 1.0000), dataset split distribution (1,760 train / 379 val / 383 test), and pre-flight gatekeeper verification clearance.

---

## 3. Git Version Control & Storage Governance

To maintain repository speed and comply with GitHub file size boundaries:
1. **Tracked in Git:**
   - Precomputed embeddings (`.npy`, `.h5`, `.parquet`) totaling ~16.4 MB.
   - Quantized TinyML flatbuffer (`.tflite`, 289.8 KB) and C++ header (`.h`, 1.81 MB).
   - Structured JSON/CSV evaluation metrics, training curves, and confusion matrices.
   - Selected XAI visual heatmaps (`exp6_explainability_xai/heatmaps/*.jpg`).
2. **Excluded via `.gitignore`:**
   - Full-precision PyTorch weights (`*.pt`, `*.pth`) and intermediate ONNX models (`*.onnx`).
   - Bulky raw dataset archives (`azn_banknotes_dataset.zip`, `gulnar_rescued_labels.zip`).
   - Intermediate training checkpoints and temporary scratch logs.

---

## 4. Module Ownership & Team Attribution

| Subdirectory / Domain | Primary Lead | Supporting Contributors |
| :--- | :--- | :--- |
| **`artifacts/embeddings/`** | Nicat Alaskarli | Avaz Asgarov, Gulnar Babazade |
| **`artifacts/experiments/exp1_architecture_battle/`** | Avaz Asgarov | Kazim Mammadli, Hasan Mammadov |
| **`artifacts/experiments/exp2_data_augmentation/`** | Kazim Mammadli | Gulnar Babazade, Avaz Asgarov |
| **`artifacts/experiments/exp3_color_space_shortcut/`** | Hasan Mammadov | Kazim Mammadli, Nicat Alaskarli |
| **`artifacts/experiments/exp4_resolution_scaling/`** | Kazim Mammadli | Hasan Mammadov, Avaz Asgarov |
| **`artifacts/experiments/exp5_model_compression/`** | Hasan Mammadov | Kazim Mammadli, Nicat Alaskarli |
| **`artifacts/experiments/exp6_explainability_xai/`** | Nicat Alaskarli | Avaz Asgarov, Gulnar Babazade |
| **`artifacts/tinyml/`** | Kazim Mammadli | Nicat Alaskarli, Hasan Mammadov |
| **`artifacts/reports/`** | Gulnar Babazade | Avaz Asgarov, Nicat Alaskarli |
