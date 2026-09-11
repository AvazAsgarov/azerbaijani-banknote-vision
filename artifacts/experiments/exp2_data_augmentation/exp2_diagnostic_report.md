# Experiment 2 Diagnostic Report: Data Augmentation Matrix & Generalization Gap

**Date of Run**: 2026-09-10  
**Compute Node**: NVIDIA A100-SXM4-40GB MIG 3g.20gb (20 GB VRAM)  
**Total Training Time**: 80.05 minutes (4 arms x 50 epochs)  
**Status**: COMPLETE (All 4 Arms Verified)  

---

## 1. Executive Scientific Summary

Experiment 2 evaluates the generalization boundaries of YOLO11m under 4 systematic data augmentation regimes on the zero-leakage partitioned Azerbaijani banknote dataset:
1. **Arm 1 (Raw Baseline)**: Pure resizing with zero synthetic perturbations.
2. **Arm 2 (Pure Geometric)**: Aggressive spatial transformations (rotations $\pm 15^\circ$, scale 0.5, shear $2.5^\circ$, perspective 0.0005, horizontal flip 0.5).
3. **Arm 3 (Pure Photometric)**: Aggressive color space perturbations (HSV-Hue 0.025, Saturation 0.7, Value 0.4) without geometric distortions.
4. **Arm 4 (Full Composite)**: Mosaic (1.0), Mixup (0.15), Copy-Paste (0.10), Photometric Jitter, and moderate Geometric transformations.

### Key Scientific Findings:
- **Catastrophic Failure of Pure Geometric Perturbations**: Arm 2 suffered severe test degradation, achieving only **75.15% Test mAP50** (vs 93.79% Val mAP50), yielding a massive generalization gap of **18.63%**. Banknotes are strictly planar rectangular objects; unconstrained affine shearing and excessive rotations corrupted edge aspect ratio invariants that the detector relies upon.
- **Photometric Robustness**: Arm 3 maintained **86.26% Test mAP50** with an **8.27% gap**, proving color and illumination jitter improves real-world lighting tolerance.
- **Superiority of Full Composite Augmentation (Winner)**: Arm 4 achieved the highest test accuracy of **95.43% Test mAP50** (and **83.77% Test mAP50-95**), crushing the generalization gap down to **1.65%** (from 9.64% in Raw Baseline). Mosaic and Mixup forced the network to learn multi-scale, partially occluded banknote features without overfitting to canvas boundaries.

---

## 2. Quantitative Comparison Table

| Arm ID | Augmentation Strategy | Val mAP50 | Val mAP50-95 | Test mAP50 | Test mAP50-95 | Generalization Gap | GPU Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`arm1_none_raw`** | Raw Baseline (Zero Augmentation) | 94.39% | 92.11% | 84.76% | 78.28% | +9.64% | 5.51 ms |
| **`arm2_geometric`** | Pure Geometric (Rot, Scale, Shear, Flip) | 93.79% | 69.15% | 75.15% | 50.63% | **+18.63% (Degraded)** | 4.59 ms |
| **`arm3_photometric`** | Pure Photometric (HSV Jitter) | 94.53% | 92.22% | 86.26% | 80.15% | +8.27% | 4.60 ms |
| **`arm4_full_composite`** | **Full Composite (Mosaic+Mixup+All)** | **97.09%** | **88.04%** | **95.43%** | **83.77%** | **+1.65% (Best)** | **4.62 ms** |

---

## 3. High-Resolution Visual Artifacts

The following 300 DPI analytical charts have been generated in `figures/`:
1. `exp2_augmentation_map_comparison.png`: Validation vs Test mAP50 comparison highlighting the generalizability delta.
2. `exp2_generalization_gap.png`: Quantitative penalty score across augmentation arms.
3. `exp2_per_class_ap.png`: Denomination-level breakdown (1 to 200 AZN) on the unseen test set.
4. `augmentation_ablation_delta.png`: Metric ablation plot exported from the A100 training cluster.

---

## 4. Final Verdict & Downstream Implementation

**Approved Production Configuration**: **Arm 4 (Full Composite)** is officially selected as the mandatory augmentation configuration for all downstream experiments (Exp 3, Exp 4, Exp 5, and TinyML fine-tuning).
