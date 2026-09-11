# Generated Artifacts (artifacts/)

## Purpose
Central storage directory for generated model weights, quantized neural networks,
firmware headers, evaluation metrics, and cached telemetry files.

## Subdirectories
- `embeddings/`: Precomputed DINOv2 ViT-L/14 representation matrices (`.h5`, `.npy`) and Parquet metadata.
- `experiments/`: Comprehensive empirical evaluation manifests, metric summaries, and curves across Experiments 1 through 6:
  - `exp1_architecture_battle/`: YOLOv8m vs YOLO11m vs RT-DETR-L vs DINOv2
  - `exp2_data_augmentation/`: Raw vs Geometric vs Photometric vs Composite
  - `exp3_color_space_shortcut/`: RGB vs HSV vs Grayscale Cross-Evaluation
  - `exp4_resolution_scaling/`: 320x320 vs 640x640 vs 1280x1280
  - `exp5_model_compression/`: FP32 vs FP16 vs INT8 PTQ vs 25% L1 Pruning
  - `exp6_explainability_xai/`: C2PSA EigenCAM attention heatmaps & alignment
- `tinyml/`: Edge microcontroller deliverables (YOLO-FastestV2 INT8 TFLite flatbuffer, C++ firmware header, training metrics, and hardware benchmarks).
- `reports/`: Machine-readable quantitative audit reports (`dataset_audit_report.json`, `split_summary.json`, `preflight_verification_report.json`, `dino_linear_probe_report.json`).

> [!NOTE]
> Large binary archive bundles (e.g. `azn_banknotes_dataset.zip`, `gulnar_rescued_labels.zip`) and model weight checkpoints (`*.pt`, `*.onnx`) are strictly excluded from version control via `.gitignore`.
