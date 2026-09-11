"""Pytest Configuration and Shared Test Fixtures.

Provides mock project environments, synthetic images, corrupted files,
vectorized embeddings, and realistic YOLO annotations for testing.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import pytest
import cv2
import h5py

from src.core.config import ProjectPaths, ClusterConfig, SplitConfig


@pytest.fixture
def mock_paths(tmp_path: Path) -> ProjectPaths:
    """Constructs isolated directory hierarchy mimicking the repository layout.

    Args:
        tmp_path: Pytest temporary directory fixture.

    Returns:
        ProjectPaths initialized to isolated temporary workspace.
    """
    root = tmp_path / "workspace"
    root.mkdir(parents=True, exist_ok=True)

    configs_dir = root / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    data_yaml = configs_dir / "data.yaml"
    data_yaml.write_text("names:\n  0: 001_azn\nnc: 1\n", encoding="utf-8")

    processed = root / "data" / "processed"
    master_images = processed / "master" / "images"
    master_labels = processed / "master" / "labels"
    master_images.mkdir(parents=True, exist_ok=True)
    master_labels.mkdir(parents=True, exist_ok=True)

    pruned = processed / "pruned_archive"
    (pruned / "images").mkdir(parents=True, exist_ok=True)
    (pruned / "labels").mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        (processed / "images" / split).mkdir(parents=True, exist_ok=True)
        (processed / "labels" / split).mkdir(parents=True, exist_ok=True)

    artifacts = root / "artifacts"
    (artifacts / "embeddings").mkdir(parents=True, exist_ok=True)
    (artifacts / "reports").mkdir(parents=True, exist_ok=True)

    figures = root / "reports" / "figures"
    (figures / "eda").mkdir(parents=True, exist_ok=True)
    (figures / "embeddings").mkdir(parents=True, exist_ok=True)
    (figures / "splits").mkdir(parents=True, exist_ok=True)

    (root / "docs").mkdir(parents=True, exist_ok=True)
    (root / "notebooks").mkdir(parents=True, exist_ok=True)

    return ProjectPaths(
        root_dir=root,
        configs_dir=configs_dir,
        data_yaml_path=data_yaml,
        data_dir=root / "data",
        raw_data_dir=root / "data" / "raw",
        processed_data_dir=processed,
        images_dir=processed / "images",
        labels_dir=processed / "labels",
        master_images_dir=master_images,
        master_labels_dir=master_labels,
        pruned_archive_dir=pruned,
        manifest_path=processed / "dataset_manifest.csv",
        summary_json_path=processed / "dataset_summary.json",
        artifacts_dir=artifacts,
        embeddings_dir=artifacts / "embeddings",
        reports_dir=artifacts / "reports",
        embeddings_h5_path=artifacts / "embeddings" / "embeddings_dinov2_vitl14.h5",
        embeddings_parquet_path=artifacts / "embeddings" / "embeddings_metadata.parquet",
        embeddings_npy_path=artifacts / "embeddings" / "embeddings_dinov2_vitl14.npy",
        audit_report_path=artifacts / "reports" / "dataset_audit_report.json",
        split_summary_path=artifacts / "reports" / "split_summary.json",
        docs_dir=root / "docs",
        figures_dir=figures,
        figures_eda_dir=figures / "eda",
        figures_embeddings_dir=figures / "embeddings",
        figures_splits_dir=figures / "splits",
        notebooks_dir=root / "notebooks",
    )


@pytest.fixture
def synthetic_image_generator():
    """Factory generating synthetic OpenCV images with controllable patterns."""
    def _generator(
        width: int = 128,
        height: int = 128,
        image_type: str = "sharp",
        seed: int = 42,
    ) -> np.ndarray:
        if image_type == "blank":
            return np.full((height, width, 3), 128, dtype=np.uint8)
        if image_type == "black":
            return np.zeros((height, width, 3), dtype=np.uint8)
        if image_type == "gradient":
            x = np.linspace(0, 255, width, dtype=np.uint8)
            return np.tile(x, (height, 1, 3))

        # Sharp textured synthetic pattern
        rng = np.random.RandomState(seed)
        base = rng.randint(50, 200, (height, width, 3), dtype=np.uint8)
        cv2.rectangle(base, (20, 20), (width - 20, height - 20), (255, 0, 0), 3)
        cv2.circle(base, (width // 2, height // 2), 15, (0, 255, 0), -1)
        if image_type == "blurry":
            base = cv2.GaussianBlur(base, (25, 25), 0)
        return base

    return _generator


@pytest.fixture
def populated_mock_env(mock_paths: ProjectPaths, synthetic_image_generator) -> ProjectPaths:
    """Populates mock environment with realistic images, labels, and manifest."""
    classes = ["001_azn", "005_azn", "010_azn"]
    manifest_rows = []

    for cls_idx, cls_name in enumerate(classes):
        for img_idx in range(10):
            fname = f"{cls_name}_{img_idx:05d}.jpg"
            img_path = mock_paths.master_images_dir / fname
            label_path = mock_paths.master_labels_dir / f"{cls_name}_{img_idx:05d}.txt"

            img = synthetic_image_generator(128, 128, "sharp")
            cv2.imwrite(str(img_path), img)

            # Write standard YOLO annotation: class x_center y_center width height
            label_path.write_text(f"{cls_idx} 0.500000 0.500000 0.600000 0.400000\n", encoding="utf-8")

            manifest_rows.append({
                "filename": fname,
                "class_name": cls_name,
                "class_id": cls_idx,
                "capture_folder": f"series_{cls_name}",
                "original_capture_folder": f"series_{cls_name}",
                "original_filename": fname,
                "is_pruned": False,
                "prune_reason": "",
                "meta_cluster": 0 if img_idx < 7 else (1 if img_idx < 9 else 2),
                "split": "train" if img_idx < 7 else ("val" if img_idx < 9 else "test"),
            })

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(mock_paths.manifest_path, index=False)

    # Generate synthetic embeddings matching manifest count
    total_imgs = len(manifest_df)
    dim = 1024
    rng = np.random.RandomState(123)
    raw_vecs = rng.randn(total_imgs, dim).astype(np.float32)
    norms = np.linalg.norm(raw_vecs, axis=1, keepdims=True)
    normed_vecs = raw_vecs / norms

    with h5py.File(mock_paths.embeddings_h5_path, "w") as h5f:
        h5f.create_dataset("embeddings", data=normed_vecs)
        h5f.create_dataset("filenames", data=[f.encode("utf-8") for f in manifest_df["filename"]])

    np.save(mock_paths.embeddings_npy_path, normed_vecs)
    manifest_df.to_parquet(mock_paths.embeddings_parquet_path)

    return mock_paths
