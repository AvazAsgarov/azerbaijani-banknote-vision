# Experiments Suite (src/experiments/)

## Purpose
Implementations of the 6 core research questions benchmarked on the Azerbaijani banknote dataset.

## Key Files
- `e1_architecture_battle.py`: Evaluates YOLOv8m vs YOLO11m vs RT-DETR-L on compute-accuracy frontier.
- `e2_shortcut_learning.py`: Tests color-invariance and geometric shortcut learning via color perturbations.
- `e3_resolution_quantization.py`: Benchmarks 160x160 vs 320x320 vs 640x640 across FP32, FP16, and INT8.
- `e4_synthetic_generalization.py`: Quantifies sim-to-real transfer gap when training with synthetic data.
- `e5_few_shot.py`: Evaluates few-shot rapid adaptation (K=1, 5, 10, 20) on novel banknote variants.
- `e6_open_vocabulary.py`: Evaluates zero-shot open-vocabulary detection using text prompt queries.
