#  Azerbaijani Banknotes (AZN) Object Detection Dataset

**Deep Learning Final Project · AI Academy · Cohort I 2026**

A rigorously curated, zero-leakage benchmark dataset for detecting and recognizing Azerbaijani Manat banknotes in real-world environments.

---

## Dataset Summary

- **Total Partitioned Images:** 2,522
- **Total Bounding Boxes:** 2,561
- **Classes:** 7 denominations (`001_azn` to `200_azn`)
- **Annotators:** 4 team members (`Avaz`, `Gulnar`, `Kazim`, `Hasan`)
- **Unique Capture Environments:** 93 distinct physical scenes (`Mekan_XX`)
- **Deduplication Audit:** 71 frames pruned (3 exact binary duplicates, 49 burst-capture near-duplicates, 19 unannotated empty frames)
- **Split Strategy:** Zero-Leakage Stratified Group Split on capture environments and DINOv2 meta-clusters

---

## Directory Structure

```text
data/processed/
├── data.yaml                # Standard YOLO / Ultralytics configuration
├── dataset_manifest.csv     # Full per-image metadata manifest (2,593 rows including provenance)
├── dataset_summary.json     # Machine-readable statistical summary
├── images/                  # (Excluded from git tracking — 2,522 high-res JPGs)
│   ├── train/               # 1,760 images (69.8%)
│   ├── val/                 # 379 images (15.0%)
│   └── test/                # 383 images (15.2%)
└── labels/                  # Version-controlled normalized YOLO format txt files
    ├── train/               # 1,760 YOLO txt label files
    ├── val/                 # 379 YOLO txt label files
    └── test/                # 383 YOLO txt label files
```

---

## Class Mapping & Bounding Box Distribution

| Class ID | Class Code | Denomination | Bounding Box Count | Train Boxes | Val Boxes | Test Boxes |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| **0** | `001_azn` | 1 Manat (Gray) | 450 | 318 | 68 | 64 |
| **1** | `005_azn` | 5 Manat (Amber-Orange) | 315 | 216 | 49 | 50 |
| **2** | `010_azn` | 10 Manat (Cyan-Teal) | 499 | 344 | 68 | 87 |
| **3** | `020_azn` | 20 Manat (Emerald Green) | 314 | 224 | 48 | 42 |
| **4** | `050_azn` | 50 Manat (Gold-Yellow) | 323 | 242 | 41 | 40 |
| **5** | `100_azn` | 100 Manat (Mauve-Purple) | 378 | 268 | 57 | 53 |
| **6** | `200_azn` | 200 Manat (Azure-Blue) | 282 | 162 | 70 | 50 |
| **Total** | | **All 7 Denominations** | **2,561** | **1,774** | **401** | **386** |

---

## Annotator Contributions (Partitioned Set)

All 4 members contributed equally with balanced partitions across the 2,522 curated images:

- **Avaz:** 637 images (25.3%) · 656 bounding boxes
- **Gulnar:** 636 images (25.2%) · 646 bounding boxes
- **Kazim:** 633 images (25.1%) · 638 bounding boxes
- **Hasan:** 616 images (24.4%) · 621 bounding boxes

---

##  Zero-Leakage Validation (Stratified Group Split)

To ensure true real-world generalization and prevent **Scene/Environment Data Leakage**:
- All images from a specific capture location (`original_capture_folder`) were assigned **exclusively** to either Train, Val, or Test.
- **Intersection between Train, Val, and Test scenes is strictly 0%**:
  $$\text{Scenes}(\text{Train}) \cap \text{Scenes}(\text{Val}) = \emptyset$$
  $$\text{Scenes}(\text{Train}) \cap \text{Scenes}(\text{Test}) = \emptyset$$
  $$\text{Scenes}(\text{Val}) \cap \text{Scenes}(\text{Test}) = \emptyset$$
