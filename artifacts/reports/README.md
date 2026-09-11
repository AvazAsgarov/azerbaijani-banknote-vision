# Verification Reports & Telemetry Manifests (`artifacts/reports/`)

**Target Domain:** Azerbaijani Banknote Detection Pipeline  
**Status:** Validated, Fresh, and Synchronized  
**Provenance Verification:** Cryptographically Verified against Dataset State  

---

## 1. Directory Catalog & Report Manifest

This directory houses official structured evaluation reports, data quality audits, split distributions, and gatekeeper verification manifests in JSON format.

| Report File | Primary Stage | Metrics Captured |
| :--- | :--- | :--- |
| `split_summary.json` | Zero-Leakage Split Engine | Cluster disjointness confirmation, cross-split similarity bounds, split image counts (`train: 1760`, `val: 379`, `test: 383`), and class distribution ratios. |
| `dataset_audit_report.json` | Dataset Quality Audit | Inter-Annotator Agreement (mIoU: 77.24%, Cohen's Kappa: 1.0000), exact binary deduplication (3 pairs resolved), and perceptual hash (pHash/dHash) cluster analysis. |
| `dino_linear_probe_report.json` | Foundation Model Baseline | DINOv2 ViT-L/14 linear probe classification accuracy (35.88% Val, 42.56% Test), macro-F1 (42.25%), per-class precision/recall/F1, and confusion matrices. |
| `eda_metrics.json` | Advanced EDA Pipeline | Photometric channel statistics (RGB/HSV), spatial geometry bounding box distributions, foreground chromaticity centroids, and manifold projection coordinates. |
| `preflight_verification_report.json` | Pre-Flight Gatekeeper | Enterprise compute allocation gate status (`APPROVED_FOR_TRAINING`), unit test suite status, code coverage percentage (>95% mandate), and dataset bounding box totals. |

---

## 2. Dataset Split Baseline Reference

* **Total Active Image Corpus:** 2,522 curated images
* **Train Split:** 1,760 images (69.8%)
* **Validation Split:** 379 images (15.0%)
* **Zero-Leakage Test Split:** 383 images (15.2%)
* **Denomination Classes:** 7 (`001_azn`, `005_azn`, `010_azn`, `020_azn`, `050_azn`, `100_azn`, `200_azn`)
* **Cross-Split Contamination Rate:** 0.00% (Strict metadata cluster isolation)
