#  Azerbaijani Banknotes (AZN) Object Detection Dataset

**Deep Learning Final Project · AI Academy · Cohort I 2026**

A rigorously curated, zero-leakage benchmark dataset for detecting and recognizing Azerbaijani Manat banknotes in real-world environments.

---

##  Dataset Summary

- **Total Images:** 2,593
- **Total Bounding Boxes:** 2,625
- **Classes:** 7 denominations (`1 AZN` to `200 AZN`)
- **Annotators:** 4 team members (`Avaz`, `Gulnar`, `Hasan`, `Kazim`)
- **Unique Capture Scenes:** 93 distinct physical environments (`Mekan_XX`)
- **Split Strategy:** Zero-Leakage Stratified Group Split on capture environments

---

##  Directory Structure

```text
dataset/
├── data.yaml                # Standard YOLO / Ultralytics configuration
├── dataset_manifest.csv     # Full per-image metadata manifest (2,593 rows)
├── dataset_summary.json     # Machine-readable statistical summary
├── images/
│   ├── train/               # 1,790 images (69.0%)
│   ├── val/                 # 392 images (15.1%)
│   └── test/                # 411 images (15.9%)
├── labels/
│   ├── train/               # 1,790 YOLO txt label files
│   ├── val/                 # 392 YOLO txt label files
│   └── test/                # 411 YOLO txt label files
└── master/                  # Unsplit consolidated master archive (2,593 images + labels)
    ├── images/
    └── labels/
```

---

##  Class Mapping & Bounding Box Distribution

| Class ID | Class Name | Denomination | Bounding Box Count |
| :---: | :--- | :--- | :---: |
| **0** | `001_azn` | 1 Manat | 462 |
| **1** | `005_azn` | 5 Manat | 324 |
| **2** | `010_azn` | 10 Manat | 499 |
| **3** | `020_azn` | 20 Manat | 335 |
| **4** | `050_azn` | 50 Manat | 324 |
| **5** | `100_azn` | 100 Manat | 397 |
| **6** | `200_azn` | 200 Manat | 284 |
| **Total** | | | **2,625** |

---

##  Annotator Contributions

All 4 members contributed equally with balanced partitions:

- **Avaz:** 655 images (25.3%)
- **Gulnar:** 654 images (25.2%)
- **Hasan:** 643 images (24.8%)
- **Kazim:** 641 images (24.7%)

---

##  Zero-Leakage Validation (Stratified Group Split)

To ensure true real-world generalization and prevent **Scene/Environment Data Leakage**:
- All images from a specific capture location (`original_capture_folder`) were assigned **exclusively** to either Train, Val, or Test.
- **Intersection between Train, Val, and Test scenes is strictly 0%**:
  $$\text{Scenes}(\text{Train}) \cap \text{Scenes}(\text{Val}) = \emptyset$$
  $$\text{Scenes}(\text{Train}) \cap \text{Scenes}(\text{Test}) = \emptyset$$
  $$\text{Scenes}(\text{Val}) \cap \text{Scenes}(\text{Test}) = \emptyset$$
