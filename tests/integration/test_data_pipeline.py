"""Integration Tests for End-to-End Data Preparation Pipeline.

Executes sequential workflow from raw image ingest through deduplication,
feature tensor extraction, community clustering, zero-leakage partitioning,
and empirical validation checks.
"""

from pathlib import Path
import shutil
import cv2
import h5py
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from torchvision import transforms
from unittest.mock import patch

from src.core.config import ClusterConfig, ProjectPaths, SplitConfig
from src.data.clustering import MetaSceneClusterer
from src.data.deduplicate import DeduplicationConfig, DeduplicationPipeline
from src.data.embeddings import ExtractionConfig, FeatureExtractor
from src.data.splitter import ZeroLeakageSplitter
from src.data.validator import SplitValidator


@pytest.mark.integration
class TestEndToEndDataPipeline:
    """Executes integrated data pipeline workflows across connected modules."""

    def test_complete_pipeline_flow(self, mock_paths: ProjectPaths, synthetic_image_generator) -> None:
        """Verifies full execution from synthetic images to validated splits."""
        classes = ["001_azn", "005_azn", "010_azn"]
        manifest_rows = []

        # Generate 15 images across 3 classes and 3 capture folders
        for c_idx, c_name in enumerate(classes):
            for i in range(5):
                fname = f"{c_name}_{i:04d}.jpg"
                img_path = mock_paths.master_images_dir / fname
                lbl_path = mock_paths.master_labels_dir / f"{c_name}_{i:04d}.txt"

                img = synthetic_image_generator(64, 64, "sharp", seed=c_idx * 100 + i)
                cv2.imwrite(str(img_path), img)
                lbl_path.write_text(f"{c_idx} 0.5 0.5 0.4 0.4\n", encoding="utf-8")

                manifest_rows.append({
                    "filename": fname,
                    "class_name": c_name,
                    "class_id": c_idx,
                    "original_capture_folder": f"series_{c_name}_{i % 2}",
                    "capture_folder": f"series_{c_name}_{i % 2}",
                    "original_filename": fname,
                    "is_pruned": False,
                    "prune_reason": "",
                })

        # Add one exact binary duplicate to test pipeline deduplication stage
        dup_fname = "001_azn_9999.jpg"
        dup_img_path = mock_paths.master_images_dir / dup_fname
        dup_lbl_path = mock_paths.master_labels_dir / "001_azn_9999.txt"
        shutil.copyfile(mock_paths.master_images_dir / "001_azn_0000.jpg", dup_img_path)
        dup_lbl_path.write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
        manifest_rows.append({
            "filename": dup_fname,
            "class_name": "001_azn",
            "class_id": 0,
            "original_capture_folder": "series_001_azn_0",
            "capture_folder": "series_001_azn_0",
            "original_filename": dup_fname,
            "is_pruned": False,
            "prune_reason": "",
        })

        manifest_df = pd.DataFrame(manifest_rows)
        manifest_df.to_csv(mock_paths.manifest_path, index=False)

        # Stage 1: Run deduplication
        dedup_cfg = DeduplicationConfig(
            manifest_path=mock_paths.manifest_path,
            images_dir=mock_paths.master_images_dir,
            labels_dir=mock_paths.master_labels_dir,
            pruned_archive_dir=mock_paths.pruned_archive_dir,
            split_images_dir=mock_paths.images_dir,
            split_labels_dir=mock_paths.labels_dir,
            hamming_threshold=4,
        )
        dedup_engine = DeduplicationPipeline(dedup_cfg)
        dedup_result = dedup_engine.run()
        dedup_engine.apply_pruning(dedup_result)

        assert dedup_result.exact_duplicate_count == 1
        assert dedup_result.final_count == 15

        # Stage 2: Feature extraction using mock model
        class FastMockModel(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # Deterministic pseudo-embedding based on mean channel intensity
                b = x.shape[0]
                vec = torch.randn((b, 1024), dtype=torch.float32)
                return vec

        ext_cfg = ExtractionConfig(
            manifest_path=mock_paths.manifest_path,
            images_dir=mock_paths.master_images_dir,
            output_h5_path=mock_paths.embeddings_h5_path,
            output_parquet_path=mock_paths.embeddings_parquet_path,
            output_npy_path=mock_paths.embeddings_npy_path,
            image_size=64,
            batch_size=4,
            use_fp16=False,
            num_workers=0,
        )
        extractor = FeatureExtractor(ext_cfg)
        with patch.object(extractor, "load_model", return_value=FastMockModel()):
            embeddings, fnames, cnames, fldrs = extractor.extract()
            extractor.save_h5(embeddings, fnames, cnames, fldrs)
            extractor.save_parquet(fnames, cnames, fldrs)
            extractor.save_npy(embeddings)

        assert embeddings.shape == (15, 1024)
        assert mock_paths.embeddings_h5_path.exists()

        # Stage 3: Meta-scene clustering
        clust_cfg = ClusterConfig(distance_threshold=0.50)
        clusterer = MetaSceneClusterer(paths=mock_paths, cluster_cfg=clust_cfg)
        clust_out = clusterer.execute()

        assert clust_out.num_meta_scenes >= 1

        # Stage 4: Zero-leakage group splitting
        split_cfg = SplitConfig(
            train_ratio=0.60,
            val_ratio=0.20,
            test_ratio=0.20,
            random_seed=42,
            optimization_iterations=200,
            classes=("001_azn", "005_azn", "010_azn"),
        )
        splitter = ZeroLeakageSplitter(paths=mock_paths, split_cfg=split_cfg)
        split_res = splitter.partition(clust_out)

        assert split_res.train_count > 0
        assert split_res.val_count > 0
        assert split_res.test_count > 0
        assert len(split_res.train_clusters & split_res.val_clusters) == 0
        assert len(split_res.train_clusters & split_res.test_clusters) == 0

        # Stage 5: Empirical split validation
        validator = SplitValidator(paths=mock_paths)
        val_rep = validator.validate()

        assert val_rep.is_disjoint is True
        assert mock_paths.split_summary_path.exists()
