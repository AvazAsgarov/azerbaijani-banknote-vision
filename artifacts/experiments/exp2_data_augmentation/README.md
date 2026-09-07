# Experiment 2: Data Augmentation Matrix & Generalization Gap

This directory contains the authentic results of Experiment 2 executed on the NVIDIA A100 GPU cluster.

## Directory Structure

```
exp2_data_augmentation/
├── README.md                              <- Directory documentation
├── exp2_diagnostic_report.md              <- Full scientific post-mortem report
├── exp2_augmentation_comparison.csv      <- Benchmark comparison table (CSV)
├── exp2_augmentation_comparison.json     <- Full serialized hyperparameter and metric payload
├── figures/                               <- 300 DPI publication-grade comparison plots
│   ├── exp2_augmentation_map_comparison.png
│   ├── exp2_generalization_gap.png
│   ├── exp2_per_class_ap.png
│   └── augmentation_ablation_delta.png
├── arm1_none_raw/                         <- Arm 1 (Raw Baseline)
│   ├── curves/                            <- Training & validation plots
│   ├── logs/                              <- args.yaml & results.csv
│   └── weights/                           <- best.pt & last.pt
├── arm2_geometric/                        <- Arm 2 (Pure Geometric)
├── arm3_photometric/                      <- Arm 3 (Pure Photometric)
└── arm4_full_composite/                   <- Arm 4 (Full Composite - Winner)
```
