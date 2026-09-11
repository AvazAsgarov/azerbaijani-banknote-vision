# Scientific Reports & Publication Figures (`reports/`)

This directory serves as the centralized archive for publication-grade diagnostic figures, empirical evaluation plots, and scientific reporting assets for the **AZN-Vision** research study.

---

## Directory Organization

```text
reports/
└── figures/
    ├── ai_academy_theme/     # 10 High-contrast branded figures in AI Academy visual identity
    ├── eda/                  # 9 Comprehensive exploratory data analysis figures
    ├── embeddings/           # 6 DINOv2 ViT-L/14 representation and manifold diagnostic plots
    ├── experiments/          # 4 Visual multi-panel collages across experiment tracks
    ├── experiments_unified/  # 7 Multi-model comparison Pareto frontiers and mAP curves
    ├── hardware/             # Edge hardware physical pinout and deployment photographs
    ├── mobile/               # Assistive companion mobile application interface mockups
    ├── splits/               # 3 Zero-leakage environment isolation and stratification proofs
    └── tinyml/               # 3 Microcontroller SRAM allocation, Pareto, and safety flows
```

---

## Research Tracks & Figure Ownership

| Research Track | Primary Study | Lead Contributor | Key Diagnostic Deliverable |
| :--- | :--- | :--- | :--- |
| **Data Strategy & Stratification** | Zero-Leakage Group Split | Avaz Asgarov | `splits/zero_leakage_visual_proof.png` |
| **Exploratory Data Analysis** | Photometric & Geometry Profiling | Gulnar Babazade | `eda/dataset_sample_batch_grid.png` |
| **Experiment 1** | Deep Learning Architecture Battle | Gulnar Babazade | `experiments_unified/exp1_pareto_latency_map.png` |
| **Experiment 2** | Data Augmentation Ablation | Kazim Mammadli | `experiments_unified/exp2_augmentation_delta.png` |
| **Experiment 3** | Color Space & Shortcut Learning | Hasan Mammadov | `experiments_unified/exp3_color_space_delta.png` |
| **Experiment 4** | Multiscale Resolution Scaling | Gulnar Babazade | `experiments_unified/exp4_resolution_pareto.png` |
| **Experiment 5** | Model Compression & PTQ INT8 | Hasan Mammadov | `experiments_unified/exp5_compression_pareto.png` |
| **Experiment 6** | Explainable AI & EigenCAM | Nicat Alaskarli | `experiments_unified/exp6_numismatic_alignment.png` |
| **Foundation Model Probing** | DINOv2 ViT-L/14 Embeddings | Nicat Alaskarli | `embeddings/eda_manifold_projections.png` |
| **TinyML Edge Deployment** | YOLO-FastestV2 on ESP32-S3 | Kazim Mammadli | `tinyml/tinyml_esp32s3_pareto.png` |

---

## Governance & Hygiene

* **Internal Speaker Guides:** Speaker notes and oral rehearsal transcripts (`speaker_guides/`, `defense_qa_and_hooks/`) are strictly excluded from version control via `.gitignore` in accordance with open-source release standards.
* **Figure Quality:** All vector and raster figures are generated at **300 DPI** using Matplotlib / Seaborn scientific rendering pipelines.
