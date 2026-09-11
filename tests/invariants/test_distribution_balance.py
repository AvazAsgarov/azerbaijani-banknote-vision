"""Pipeline Invariant Tests for Stratification Balance and Class Representation.

Enforces minimum representation guarantees across all 7 Azerbaijani banknote
denominations in every split (Train, Val, Test), avoiding zero-instance defects.
"""

from pathlib import Path
import pandas as pd
import pytest

from src.core.config import ProjectPaths, SplitConfig


@pytest.mark.invariant
class TestDistributionBalanceInvariants:
    """Mathematical invariants verifying partition balance and denomination presence."""

    @pytest.fixture(scope="class")
    def paths(self) -> ProjectPaths:
        """Returns project paths container."""
        return ProjectPaths()

    @pytest.fixture(scope="class")
    def active_manifest(self, paths: ProjectPaths) -> pd.DataFrame:
        """Loads non-pruned active dataset records."""
        df = pd.read_csv(paths.manifest_path)
        return df[df["is_pruned"] == False].copy().reset_index(drop=True)

    def test_all_seven_classes_present_in_every_split(self, active_manifest: pd.DataFrame) -> None:
        """Verifies every banknote denomination exists in Train, Val, and Test splits."""
        expected_classes = {
            "001_azn",
            "005_azn",
            "010_azn",
            "020_azn",
            "050_azn",
            "100_azn",
            "200_azn",
        }

        for split_name in ["train", "val", "test"]:
            split_subset = active_manifest[active_manifest["split"] == split_name]
            split_classes = set(split_subset["class_name"].unique())

            missing = expected_classes - split_classes
            assert len(missing) == 0, f"Split '{split_name}' lacks classes: {missing}"

            # Verify minimum count per class in each split
            for cls in expected_classes:
                cnt = len(split_subset[split_subset["class_name"] == cls])
                assert cnt >= 15, f"Deficient representation for {cls} in {split_name}: only {cnt} samples"

    def test_overall_split_proportions(self, active_manifest: pd.DataFrame) -> None:
        """Verifies partition sample allocations align with target ratios."""
        total_samples = len(active_manifest)
        assert total_samples == 2522, f"Active count mismatch: {total_samples} != 2522"

        train_count = len(active_manifest[active_manifest["split"] == "train"])
        val_count = len(active_manifest[active_manifest["split"] == "val"])
        test_count = len(active_manifest[active_manifest["split"] == "test"])

        assert train_count == 1760
        assert val_count == 379
        assert test_count == 383

        train_ratio = train_count / total_samples
        val_ratio = val_count / total_samples
        test_ratio = test_count / total_samples

        assert pytest.approx(train_ratio, abs=0.02) == 0.70
        assert pytest.approx(val_ratio, abs=0.02) == 0.15
        assert pytest.approx(test_ratio, abs=0.02) == 0.15
