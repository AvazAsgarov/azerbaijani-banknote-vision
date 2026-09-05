# Experiment 6: Explainable AI & Attention Diagnostics Report

## 1. Executive Summary
- **Evaluation Target**: YOLO11m Full Composite Augmentation Champion (`best.pt`).
- **Feature Layer Hooked**: `model.23.cv3.2.2` (Deep semantic classification tensor).
- **Global Numismatic Alignment Score**: **57.98%**
- **Test Classification Accuracy**: **92.00%**
- **Sample Population**: 175 test images (25 balanced samples across all 7 Azerbaijani denominations).

## 2. Denomination Alignment Distribution
| Denomination | Mean Alignment Score | Top-1 Accuracy | Mean Confidence | Sample Count |
| :--- | :---: | :---: | :---: | :---: |
| `001_azn` | 76.85% | 96.00% | 0.846 | 25 |
| `005_azn` | 49.45% | 72.00% | 0.728 | 25 |
| `010_azn` | 28.77% | 76.00% | 0.641 | 25 |
| `020_azn` | 72.56% | 100.00% | 0.896 | 25 |
| `050_azn` | 77.01% | 100.00% | 0.840 | 25 |
| `100_azn` | 52.09% | 100.00% | 0.893 | 25 |
| `200_azn` | 49.09% | 100.00% | 0.855 | 25 |

## 3. Key Observations & Numismatic Inductive Bias
1. **Strongest Feature Localization**: The highest spatial attention concentration occurs in `050_azn` (77.0%) and `001_azn` (76.9%), where distinctive visual motifs (large denomination numerals, high-contrast emblems) tightly channel classification activations within bounding box borders.
2. **Hard-Pair Discrimination**: The visually challenging pair `020_azn` (72.6%) and `200_azn` (49.1%) exhibits focused energy despite subtle color differences, demonstrating that the network does not solely rely on peripheral background cues.
3. **Heatmap Overlays**: All 175 test overlays have been verified and saved with authentic timestamps in `heatmaps/cam_*.jpg`.
