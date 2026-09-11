# Data Pipeline (src/data/)

## Purpose
Manages dataset ingestion, image embedding extraction, zero-leakage cluster splitting,
perceptual deduplication, and YOLO annotation schema validation.

## Key Files
- `dataset.py`: PyTorch dataset definitions with bounding box normalization and image caching.
- `embeddings.py`: Feature extraction using DINOv2 ViT-L/14 for visual semantic clustering.
- `splitter.py`: Stratified cluster-based train/val/test splitting guaranteeing zero inter-split leakage.
- `deduplicate.py`: Exact and perceptual hashing (pHash) to remove identical or near-duplicate frames.
- `validator.py`: Integrity checks validating bounding box bounds [0, 1] and split disjointness.
- `provenance.py`: Hash-based tracking to guarantee data lineage and auditability.
