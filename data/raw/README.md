# Raw Data Directory (`data/raw/`)

## Overview
This directory stores multi-annotator raw captures, calibration sets, and uncurated field footage collected by the team across 93 distinct physical retail, illumination, and background environments in Azerbaijan.

---

## Ingestion & Version Control Policy

1. **Storage Governance:**
   - Large raw binary archives (`*.zip`) and uncurated burst captures are stored in team cloud storage and local staging drives.
   - They are strictly excluded from GitHub version control via `.gitignore` to prevent repository bloat.

2. **Curation Workflow:**
   - Raw multi-annotator packages are parsed, deduplicated (exact binary + pHash), and validated using `scripts/build_dataset.py`.
   - Verified normalized bounding box coordinates and split manifests are published directly into `data/processed/`.
