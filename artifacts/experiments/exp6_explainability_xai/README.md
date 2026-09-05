# Experiment 6: Explainable AI & Attention Diagnostics

## Overview
Evaluates spatial attention concentration and inductive bias alignment of the champion banknote detector using EigenCAM.

## Key Metrics
- Target Layer: `model.23.cv3.2.2`
- Overall Numismatic Alignment Score: **57.98%**
- Overall Top-1 Classification Accuracy: **92.00%**
- Evaluated Samples: 175 test images across 7 denominations

## Directory Structure
- `figures/numismatic_alignment_by_denomination.png`: Publication-grade 300 DPI bar chart.
- `heatmaps/`: 175 EigenCAM attention overlays (`cam_*.jpg`).
- `exp6_xai_summary.csv` & `exp6_xai_summary.json`: Granular per-sample and per-class metrics.
- `exp6_diagnostic_report.md`: Formal diagnostic report.
