# Documentation Portal (`docs/`)

Welcome to the technical and scientific documentation hub for the **Azerbaijani Banknote Vision (AZN-Vision)** project.

This directory serves as the centralized repository for ground-truth annotation protocols, numismatic verification standards, team contribution matrices, and architectural documentation.

---

## Documentation Index

### 1. [Annotation Guidelines & Quality Assurance Protocol](annotation_guidelines.md)
* **Objective:** Establishes a rigorous, repeatable ground-truth labeling standard across all 7 circulating Azerbaijani Manat banknote denominations (`001 AZN` through `200 AZN`).
* **Numismatic Bounding Box Standards:** Sub-pixel coordinate tightness, bounding box boundary intersection rules, orientation-invariance, and strict occlusion thresholds ($\ge 75\%$).
* **Inter-Annotator Agreement (IAA):** Independent multi-annotator pilot calibration study ($N=150$) demonstrating:
  - **Mean Pairwise IoU:** **0.934 ± 0.028** (**93.4%**)
  - **Median IoU:** **0.948**
  - **Categorical Fleiss' Kappa:** **0.962** (almost perfect agreement under Landis & Koch criteria).

### 2. [Team Contribution Statement](contribution_statement.md)
* **Team Members:** Avaz Asgarov, Gulnar Babazade, Hasan Mammadov, Kazim Mammadli, Nicat Alaskarli.
* **Workload Allocation:** 100% balanced workload distribution (20% per member) across 6 empirical research tracks, TinyML edge deployment, mobile app development, and system engineering.
* **Verification Matrix:** Detailed individual deliverable breakdown, code ownership mappings, slide deck allocation, and GitHub governance audit.

### 3. Scientific Research Report (Upcoming Release)
* **Status:** In preparation for academic publication.
* **Target File:** `docs/research_paper_report.pdf` (or `docs/research_paper_report.md`).
* **Content:** Empirical findings from Experiments 1–6, DINOv2 self-supervised foundation model representation probing, and TinyML INT8 edge deployment.

---

## Data Governance & Version Control Policy

In accordance with professional GitHub repository hygiene:
* **Excluded Content:** Internal course prompts, instructor assignment briefs (`dl_final_project.md`, `dl_final_pdf.pdf`), and temporary exploratory notes are excluded from version control via `.gitignore`.
* **Tracked Deliverables:** Only peer-reviewed protocols, verified contribution statements, and publication-ready technical reports are tracked in Git.
