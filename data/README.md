# Dataset Directory (`data/`)

## Overview
This directory houses the data tier for the Azerbaijani Banknote Vision (AZN-Vision) initiative, implementing an end-to-end pipeline from multi-annotator field acquisition to verified zero-leakage training partitions.

---

## Subdirectories

### 1. [`raw/`](raw/)
* Contains multi-annotator raw capture folders, calibration images, and field footage collected across 93 unique retail and illumination environments.
* Heavy image and video archives (`*.zip`) are permanently excluded from Git version control via `.gitignore`.

### 2. [`processed/`](processed/)
* **`labels/`**: Canonical normalized YOLO format annotations for `train/` (1,760), `val/` (379), and `test/` (383) splits — total 2,522 labels.
* **`dataset_manifest.csv`**: Full metadata manifest (2,593 records) providing sample-level provenance, annotator attribution, and prune justifications.
* **`dataset_summary.json`**: Machine-readable statistical summary of class frequencies, split ratios, and leakage isolation guarantees.
* **`data.yaml`**: Self-contained Ultralytics YOLO dataset configuration mapping class IDs (`001_azn` → `200_azn`).

