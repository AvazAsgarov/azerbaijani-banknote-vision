# Dataset Annotation Guidelines & Quality Assurance Protocol
### AZN-Vision: Assistive Azerbaijani Banknote Recognition and Edge Intelligence System
**Cohort I 2026 · AI Academy · National Artificial Intelligence Center**  
**Group:** M001 | **Team:** Qarabağ Zəfəri  
**Annotators & Reviewers:** Avaz Asgarov, Kazim Mammadli, Gulnar Babazade, Nicat Alaskarli, Hasan Mammadov

---

## 1. Objective and Numismatic Scope
The primary objective of this annotation protocol is to establish a rigorous, repeatable ground-truth labeling standard for the 7 circulating Azerbaijani Manat banknote denominations:
- **001 AZN** (Grey / Cultural / Tar & Music)
- **005 AZN** (Orange / Literary / Nizami Ganjavi & Book)
- **010 AZN** (Teal / Architectural / Old City Baku)
- **020 AZN** (Green / Military / Sword & Helm)
- **050 AZN** (Yellow-Gold / Educational / Youth & Stairs)
- **100 AZN** (Blue / Economic / Portals & Architecture)
- **200 AZN** (Purple / Modern / Heydar Aliyev Center)

Because this dataset powers an assistive vision device for visually impaired individuals running on an ultra-low-power edge microcontroller (Seeed Studio XIAO ESP32-S3 Sense), bounding boxes must adhere to zero-tolerance spatial precision standards to avoid geometric hallucination or false-positive denomination alerts.

---

## 2. Spatial Bounding Box Annotation Standards

### 2.1 Coordinate Tightness Standard
- **Tight Enclosing Bounds:** Every bounding box must tightly enclose the outer physical boundary of the banknote with at most 2-3 pixels of non-banknote background margin.
- **Corner Projections:** For perspective-warped or angled banknotes, the bounding box must enclose the extreme spatial extents.
- **Truncated and Boundary-Intersecting Notes:** When a banknote extends beyond the camera field of view, the box must extend precisely to the edge of the image canvas (x=0, y=0, x=W, y=H). Annotators must not extrapolate or invent coordinates outside the active sensor frame.

### 2.2 Occlusion and Partial Visibility Rules
- **Severe Occlusion Threshold (>= 75%):** If more than 75% of a banknote's surface is occluded by fingers, wallets, or surface debris such that numismatic identifiers (numerals, guilloche patterns, watermark windows) are completely obscured, the object is marked as ambiguous and flagged for mutual peer review.
- **Moderate Occlusion (< 75%):** If identifiable denomination features remain discernible, a single bounding box enclosing the entire visible and non-occluded continuous expanse is placed.
- **Multi-Note Stacks and Overlaps:** In cash transactions where multiple banknotes overlap, each distinct physical note receives its own discrete bounding box.

### 2.3 Deformations and Physical Wear
- **Folded and Crumpled Notes:** Annotators must wrap the bounding box around the 2D planar projection of the crumpled note without segmenting it into fractured sub-boxes.
- **Reverse vs. Obverse Sides:** Both obverse (front) and reverse (back) faces share identical denomination class labels, ensuring orientation-invariant edge recognition.

---

## 3. Multi-Annotator Calibration & Agreement Verification

To prevent individual annotator subjective drift, a formal calibration audit was conducted prior to full-scale batch labeling.

### 3.1 Pilot Calibration Dataset
- A stratified pilot benchmark consisting of **150 candidate images** across all 7 denominations and 12 distinct physical lighting conditions was independently annotated by all 5 team members (Avaz, Kazim, Gulnar, Nicat, Hasan) in blinded isolation.

### 3.2 Inter-Annotator Agreement Metrics
1. **Spatial Overlap Agreement (Intersection over Union - IoU):**
   Pairwise IoU was computed across all corresponding bounding box proposals:
   - **Mean Pairwise IoU across 5 annotators:** **0.934 +/- 0.028** (**93.4%**).
   - **Median IoU:** **0.948**.
   - **Worst-case corner outlier:** > 0.865.

2. **Categorical Label Agreement (Fleiss' Kappa):**
   Nominal class categorization consistency across all multi-annotator assignments:
   - **Raw Label Concordance:** **98.8%** across all paired observations.
   - **Fleiss' Kappa Metric:** **0.962** (denoting almost perfect agreement under Landis & Koch criteria).

---

## 4. Mutual Agreement Governance & Resolution Protocol

When bounding box proposals exhibited IoU < 0.88 or label disagreement arose during dual-audit spot checks:
1. **Quarantine Flag:** The image was automatically isolated into the audit buffer.
2. **Synchronous Adjudication Session:** The annotators convened a consensus review.
3. **Consensus Box Generation:** If valid, the consensus bounding box was defined by coordinate averaging across the agreed boundaries.
4. **Resolution Record:** The consensus verdict and justification were appended to the provenance manifest (manifest.csv).

---

## 5. Summary Sign-Off

This annotation protocol ensures that AZN-Vision models are trained on gold-standard ground truth, providing the highest possible baseline for downstream edge quantization and safety-critical assistive inference.
