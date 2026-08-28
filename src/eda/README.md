# Exploratory Data Analysis (src/eda/)

## Purpose
Quantitative analysis and visualization engines inspecting distribution balance,
spatial coordinate coverage, photometric variance, and semantic manifold clustering.

## Key Files
- `spatial.py`: Aspect ratio distributions, bounding box centroid heatmaps, normalized scale analysis.
- `photometric.py`: Color channel distributions, illumination histograms, blur and contrast metrics.
- `manifold.py`: UMAP and t-SNE dimensionality reduction on DINOv2 feature embeddings.
- `integrity.py`: Dataset corruption detection, missing label discovery, duplicate file detection.
- `deep_dives.py`: Sub-cluster inspection for edge-case lighting, occlusion, and background variations.
- `style.py`: Standardized matplotlib/seaborn formatting configurations (300 DPI, dark/light themes).
