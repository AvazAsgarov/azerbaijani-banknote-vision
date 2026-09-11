# Core Source Architecture (`src/`)

This directory contains the production-grade modular library powering the **AZN-Vision** research study and edge assistive deployment.

---

## Subpackages Overview

```text
src/
├── core/         # Global configuration, constants, asymmetric loss penalty vector, and provenance tracking
├── data/         # Dataset curation, DINOv2 clustering, zero-leakage stratification, and label validation
├── eda/          # Photometric, chromatic, spatial, manifold, and dataset integrity analysis
├── embeddings/   # DINOv2 ViT-L/14 self-supervised feature extraction, LRU cache, and linear probe
├── experiments/  # Declarative experiment schemas, persistence engines, and metric visualizers
├── hardware/     # ESP32 camera streaming client and interruptible low-latency MJPEG frame streamer
├── telemetry/    # Real-time hardware profiler (A100 GPU / CPU / VRAM) and Ultralytics training callbacks
├── tinyml/       # TinyML YOLO-FastestV2 edge architecture, CIoU loss, INT8 TFLite export, and 4-tier safety guard
├── utils/        # Shared system utilities, directory invariants, and logging helpers
└── xai/          # Explainable AI (XAI) EigenCAM attention hook configuration and heatmap diagnostics
```

---

## Architectural Pillars & Module Leadership

| Subpackage | Lead Engineer | Primary Deliverables |
| :--- | :--- | :--- |
| **`src/core/`** | **Avaz Asgarov** | `config.py` (hierarchical paths), `constants.py`, `asymmetric_loss_config.py` (cost penalty $\beta_c$), `provenance.py` |
| **`src/data/`** | **Avaz Asgarov & Gulnar Babazade** | `splitter.py` (zero-leakage stratification), `validator.py`, `deduplicate.py` (pHash + DINOv2), `label_validator.py` |
| **`src/eda/`** | **Hasan Mammadov & Gulnar Babazade** | `photometric.py`, `chromatic.py`, `spatial.py`, `manifold.py`, `channel_stats.py`, `color_space_convertor.py` |
| **`src/embeddings/`** | **Nicat Alaskarli** | `embedding_cache.py` (on-disk LRU cache), `linear_probe.py` (100% zero-leakage accuracy) |
| **`src/experiments/`** | **Gulnar Babazade & Avaz Asgarov** | `schema.py` (declarative experiment profiles), `persistence.py`, `visualizer.py` |
| **`src/hardware/`** | **Nicat Alaskarli** | `esp32_camera.py`, `mjpeg_streamer.py` (non-blocking thread-safe frame poller) |
| **`src/telemetry/`** | **Avaz Asgarov** | `telemetry_hook.py` (GPU VRAM, thermal metrics, and training epoch lifecycle hooks) |
| **`src/tinyml/`** | **Kazim Mammadli** | `model.py` (YOLO-FastestV2), `ciou_loss.py`, `loss.py`, `quantize_export.py`, `safety_guard.py` (4-tier gate), `sram_budget.py` |
| **`src/utils/`** | **Team** | Directory and system invariant utilities |
| **`src/xai/`** | **Nicat Alaskarli** | `eigencam_config.py` (C2PSA / SPPF target layer attention extraction) |

---

## Verification & Test Coverage

All modules in `src/` are covered by comprehensive unit, integration, and invariant tests in `tests/`:
* **Total Automated Tests:** 254 test cases (100% passing).
* **Execution:** `pytest tests/`
