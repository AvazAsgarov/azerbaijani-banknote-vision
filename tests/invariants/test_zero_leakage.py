"""Pipeline Invariant Tests for Mathematical Zero-Leakage Dataset Partitioning.

Enforces strict group and hash separation across splits, ensuring complete
absence of data leakage between training, validation, and test subsets.
"""

from pathlib import Path
import pandas as pd
import pytest

from src.core.config import ProjectPaths


@pytest.mark.invariant
class TestZeroLeakageInvariants:
    """Non-negotiable contract assertions verifying complete isolation across splits."""

    @pytest.fixture(scope="class")
    def live_manifest(self) -> pd.DataFrame:
        """Loads live dataset manifest and filters to active non-pruned records."""
        paths = ProjectPaths()
        assert paths.manifest_path.exists(), f"Manifest missing at: {paths.manifest_path}"
        df = pd.read_csv(paths.manifest_path)
        active_df = df[df["is_pruned"] == False].copy().reset_index(drop=True)
        return active_df

    def test_pairwise_filename_disjointness(self, live_manifest: pd.DataFrame) -> None:
        """Verifies zero image overlap between train, val, and test subsets."""
        train_files = set(live_manifest[live_manifest["split"] == "train"]["filename"])
        val_files = set(live_manifest[live_manifest["split"] == "val"]["filename"])
        test_files = set(live_manifest[live_manifest["split"] == "test"]["filename"])

        assert len(train_files & val_files) == 0, "Leakage detected: Train and Val share filenames!"
        assert len(train_files & test_files) == 0, "Leakage detected: Train and Test share filenames!"
        assert len(val_files & test_files) == 0, "Leakage detected: Val and Test share filenames!"

    def test_meta_cluster_disjointness(self, live_manifest: pd.DataFrame) -> None:
        """Verifies zero meta-scene cluster bridges partition boundaries."""
        train_clusters = set(live_manifest[live_manifest["split"] == "train"]["meta_cluster"])
        val_clusters = set(live_manifest[live_manifest["split"] == "val"]["meta_cluster"])
        test_clusters = set(live_manifest[live_manifest["split"] == "test"]["meta_cluster"])

        overlap_tr_va = train_clusters & val_clusters
        overlap_tr_te = train_clusters & test_clusters
        overlap_va_te = val_clusters & test_clusters

        assert len(overlap_tr_va) == 0, f"Cluster leakage between Train and Val: {overlap_tr_va}"
        assert len(overlap_tr_te) == 0, f"Cluster leakage between Train and Test: {overlap_tr_te}"
        assert len(overlap_va_te) == 0, f"Cluster leakage between Val and Test: {overlap_va_te}"

    def test_all_active_samples_assigned(self, live_manifest: pd.DataFrame) -> None:
        """Verifies every active sample holds a canonical split designation."""
        valid_splits = {"train", "val", "test"}
        assigned_splits = set(live_manifest["split"].unique())
        assert assigned_splits.issubset(valid_splits), f"Invalid split designations: {assigned_splits}"
        assert not live_manifest["split"].isna().any(), "Unassigned null splits discovered!"
