# DINOv2 Foundation Feature Representations (`artifacts/embeddings/`)

**Foundation Model:** Meta DINOv2 ViT-L/14 (`dinov2_vitl14`, patch size 14x14)  
**Embedding Dimension:** 1,024 float32 channels  
**Dataset Alignment:** 2,522 total zero-leakage samples (Train: 1,760, Val: 379, Test: 383)  
**Denominations:** 7 Azerbaijani Banknote Classes (`001_azn`, `005_azn`, `010_azn`, `020_azn`, `050_azn`, `100_azn`, `200_azn`)  
**Status:** Synchronized, Verified, and Evaluated  

---

## 1. Overview & Architectural Role
This directory contains dense feature representations extracted using Meta's self-supervised Vision Transformer **DINOv2 ViT-L/14** (304.3M parameters). DINOv2 operates as the un-fine-tuned foundation model baseline across our research study:
1. **Foundation Model Linear Probe Baseline:** Evaluates whether generic self-supervised representations without task-specific tuning can classify Azerbaijani banknote denominations.
2. **Latent Manifold Topology & Clustering:** Provides the high-dimensional feature space for t-SNE, UMAP, and PCA manifold projections to uncover class confusion boundaries.
3. **Data Integrity & Leakage Verification:** Feature similarity matrices verify zero data leakage between training, validation, and test partitions.

---

## 2. Artifact File Manifest

| File Name | Format | Dimensions / Shape | File Size | Description |
| :--- | :--- | :---: | :---: | :--- |
| `embeddings_dinov2_vitl14.npy` | Raw NumPy Array | `(2522, 1024)` | 10.33 MB | Contiguous binary float32 feature matrix for high-speed linear probing and matrix multiplication. |
| `embeddings_dinov2_vitl14.h5` | HDF5 Container | `(2522, 1024)` | 6.89 MB | Compressed, chunked HDF5 dataset (`/features`) with dataset-level attributes and provenance metadata. |
| `embeddings_metadata.parquet` | Apache Parquet | `(2522, 5)` | 20.06 KB | Columnar metadata mapping each vector index to `filename`, `class_name`, `original_capture_folder`, `split`, and `meta_cluster`. |

---

## 3. Dataset Split Alignment & Class Breakdown

The embeddings strictly mirror the current zero-leakage partitioned dataset:

| Class Code | Denomination Name | Train Count | Val Count | Test Count | Total Embeddings |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `001_azn` | 1 Manat (Gray) | 308 | 67 | 64 | 439 |
| `005_azn` | 5 Manat (Amber-Orange) | 202 | 46 | 49 | 297 |
| `010_azn` | 10 Manat (Cyan-Teal) | 338 | 68 | 87 | 493 |
| `020_azn` | 20 Manat (Emerald Green) | 224 | 48 | 42 | 314 |
| `050_azn` | 50 Manat (Gold-Yellow) | 240 | 41 | 40 | 321 |
| `100_azn` | 100 Manat (Mauve-Purple) | 267 | 57 | 53 | 377 |
| `200_azn` | 200 Manat (Azure-Blue) | 181 | 52 | 48 | 281 |
| **Total** | **All 7 Denominations** | **1,760** | **379** | **383** | **2,522** |

---

## 4. Linear Probe Baseline Performance

Linear classification probe ($L_2$-regularized Logistic Regression, $C=1.0$, trained on 1,760 train vectors, evaluated on 379 val and 383 test vectors):

| Metric | Validation Split | Zero-Leakage Test Split |
| :--- | :---: | :---: |
| **Overall Accuracy** | **35.88%** | **42.56%** |
| **Macro-Averaged F1** | **32.72%** | **42.25%** |
| **Weighted-Averaged F1** | **34.74%** | **40.89%** |

### Per-Class Test Split Breakdown:
* **001 AZN:** Precision 64.3%, Recall 70.3%, F1-Score **67.2%** (Highest F1)
* **005 AZN:** Precision 71.4%, Recall 30.6%, F1-Score **42.9%**
* **010 AZN:** Precision 25.5%, Recall 59.8%, F1-Score **35.7%**
* **020 AZN:** Precision 0.0%, Recall 0.0%, F1-Score **0.0%** (Severe confusion with 10 AZN)
* **050 AZN:** Precision 53.6%, Recall 37.5%, F1-Score **44.1%**
* **100 AZN:** Precision 53.1%, Recall 64.2%, F1-Score **58.0%**
* **200 AZN:** Precision 45.1%, Recall 47.9%, F1-Score **46.5%**

### Scientific Takeaway:
* DINOv2 frozen representations achieve only 42.56% test accuracy and completely fail on 20 AZN (0% F1).
* This empirical baseline formally justifies the necessity of dedicated end-to-end object detector training (YOLOv8m/YOLO11m achieving >93% mAP50).

---

## 5. Associated Diagnostic Figures (`reports/figures/embeddings/`)
* `dinov2_extraction_telemetry.png`: Extraction throughput, GPU memory profile, and batch processing latency.
* `dinov2_sample_similarity_heatmap.png`: Cross-sample cosine similarity matrix.
* `eda_class_similarity_matrix.png`: Centroid cosine similarity across the 7 denominations.
* `eda_manifold_projections.png`: 2D UMAP and t-SNE latent manifold embeddings colored by denomination.
* `eda_similarity_distributions.png`: Intra-class vs inter-class cosine distance distributions.
