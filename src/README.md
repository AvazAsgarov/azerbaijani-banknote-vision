# Source Code (src/)

## Purpose
Root package for core modules, data pipelines, exploratory analysis, experiment runners,
hardware telemetry, and TinyML embedded deployment.

## Subpackages
- `core/`: Global configurations, system constants, denomination label mappings.
- `data/`: Dataset loaders, DINOv2 zero-leakage stratification, deduplication, schema validation.
- `eda/`: Photometric, spatial, manifold, style, and integrity exploratory analysis.
- `experiments/`: Scientific experiment definitions (Exp 1 through Exp 6).
- `telemetry/`: Real-time hardware profiler (A100 GPU/CPU/RAM) and metrics logging.
- `tinyml/`: YOLO-FastestV2 edge detection pipeline, INT8 quantization, C++ export, and safety guards.
- `utils/`: Common file IO, logging, and visualization utilities.
