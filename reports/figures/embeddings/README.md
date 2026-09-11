# DINOv2 Latent Manifold & Representation Figures (`reports/figures/embeddings/`)

Visual diagnostic representations generated from 1,024-dimensional DINOv2 ViT-L/14 foundation model feature extractions.

## Catalog of Figures
1. `eda_manifold_projections.png`: 2D t-SNE and UMAP manifold visualizations colored by denomination class, displaying latent feature separation and cluster overlaps.
2. `eda_class_similarity_matrix.png`: Cosine similarity matrix between class centroids across all 7 denominations.
3. `eda_similarity_distributions.png`: Intra-class vs inter-class cosine similarity probability density functions.
4. `dinov2_sample_similarity_heatmap.png`: High-resolution cross-sample pairwise cosine similarity heatmap.
5. `dinov2_extraction_telemetry.png`: Feature extraction hardware telemetry profiling GPU memory, batch latency, and throughput.
