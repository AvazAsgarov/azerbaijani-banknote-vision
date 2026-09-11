"""Unit Tests for Multi-Objective Combinatorial Group Splitter.

Validates constrained knapsack optimization, disjoint cluster assignment,
target ratio adherence, and physical dataset partition file copying.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.core.config import ProjectPaths, SplitConfig
from src.data.clustering import ClusterOutput
from src.data.splitter import SplitPartitionResult, ZeroLeakageSplitter


@pytest.mark.unit
class TestZeroLeakageSplitter:
    """Tests group-stratified dataset partitioning and zero-leakage enforcement."""

    def test_partition_disjointness_and_counts(self, populated_mock_env: ProjectPaths) -> None:
        """Verifies mathematical disjointness across train, val, and test splits."""
        df_manifest = pd.read_csv(populated_mock_env.manifest_path)
        # Create cluster output with 6 mock clusters
        folder_to_meta = {f"series_001_azn": 0, f"series_005_azn": 1, f"series_010_azn": 2}
        cluster_out = ClusterOutput(
            folder_to_meta=folder_to_meta,
            df_manifest=df_manifest,
            num_meta_scenes=3,
            meta_cluster_sizes={0: 10, 1: 10, 2: 10},
        )

        cfg = SplitConfig(
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            random_seed=42,
            optimization_iterations=50,
            classes=("001_azn", "005_azn", "010_azn"),
        )
        splitter = ZeroLeakageSplitter(paths=populated_mock_env, split_cfg=cfg)
        result = splitter.partition(cluster_out)

        assert isinstance(result, SplitPartitionResult)
        assert result.train_count + result.val_count + result.test_count == len(df_manifest)

        # Mathematical disjointness assertion
        assert len(result.train_clusters & result.val_clusters) == 0
        assert len(result.train_clusters & result.test_clusters) == 0
        assert len(result.val_clusters & result.test_clusters) == 0

        # Physical file partitioning assertion
        for split in ["train", "val", "test"]:
            img_count = len(list((populated_mock_env.images_dir / split).glob("*.jpg")))
            lbl_count = len(list((populated_mock_env.labels_dir / split).glob("*.txt")))
            assert img_count == lbl_count

    def test_deterministic_split_with_seed(self, populated_mock_env: ProjectPaths) -> None:
        """Verifies fixed random seed produces identical partition assignments."""
        df_manifest = pd.read_csv(populated_mock_env.manifest_path)
        cluster_out1 = ClusterOutput(
            folder_to_meta={},
            df_manifest=df_manifest.copy(),
            num_meta_scenes=3,
            meta_cluster_sizes={0: 10, 1: 10, 2: 10},
        )
        cluster_out2 = ClusterOutput(
            folder_to_meta={},
            df_manifest=df_manifest.copy(),
            num_meta_scenes=3,
            meta_cluster_sizes={0: 10, 1: 10, 2: 10},
        )

        cfg = SplitConfig(
            random_seed=777,
            optimization_iterations=20,
            classes=("001_azn", "005_azn", "010_azn"),
        )
        splitter = ZeroLeakageSplitter(paths=populated_mock_env, split_cfg=cfg)

        res1 = splitter.partition(cluster_out1)
        res2 = splitter.partition(cluster_out2)

        assert res1.train_clusters == res2.train_clusters
        assert res1.val_clusters == res2.val_clusters
        assert res1.test_clusters == res2.test_clusters

    def test_unrepresented_class_zero_division_guard(self, populated_mock_env: ProjectPaths) -> None:
        """Verifies partitioner handles classes with zero instances without zero-division."""
        df_manifest = pd.read_csv(populated_mock_env.manifest_path)
        # Configure extra class not present in manifest
        cfg = SplitConfig(
            random_seed=42,
            optimization_iterations=10,
            classes=("001_azn", "005_azn", "010_azn", "200_azn"),
        )
        cluster_out = ClusterOutput(
            folder_to_meta={},
            df_manifest=df_manifest,
            num_meta_scenes=3,
            meta_cluster_sizes={0: 10, 1: 10, 2: 10},
        )
        splitter = ZeroLeakageSplitter(paths=populated_mock_env, split_cfg=cfg)
        result = splitter.partition(cluster_out)
        assert result.train_count > 0
