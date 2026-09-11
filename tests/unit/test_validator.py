"""Unit Tests for Split Validation and Contamination Detection Engine.

Validates cluster disjointness assertions, cross-split cosine distance bounds,
top-k nearest neighbor contamination scores, and verification report exports.
"""

import json
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import pytest

from src.core.config import ProjectPaths
from src.data.validator import SplitValidator, ValidationReport


@pytest.mark.unit
class TestSplitValidator:
    """Tests zero-leakage mathematical verification and similarity analytics."""

    def test_validation_clean_disjoint_splits(self, populated_mock_env: ProjectPaths) -> None:
        """Verifies disjointness and similarity calculations on valid mock splits."""
        validator = SplitValidator(populated_mock_env)
        report = validator.validate()

        assert isinstance(report, ValidationReport)
        assert report.is_disjoint is True
        assert 0.0 <= report.top_k_contamination_rate <= 1.0
        assert -1.0 <= report.mean_train_test_similarity <= 1.0
        assert populated_mock_env.split_summary_path.exists()

        with open(populated_mock_env.split_summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["cluster_disjointness_confirmed"] is True
        assert "split_image_counts" in data

        # Confirm figure generation
        assert (populated_mock_env.figures_splits_dir / "split_distribution.png").exists()

    def test_validation_detects_cluster_leakage(self, populated_mock_env: ProjectPaths) -> None:
        """Verifies validator detects shared clusters bridging partition boundaries."""
        # Intentionally mutate manifest so cluster 0 is shared between train and test
        df = pd.read_csv(populated_mock_env.manifest_path)
        df.loc[df["filename"] == df.iloc[0]["filename"], "split"] = "train"
        df.loc[df["filename"] == df.iloc[0]["filename"], "meta_cluster"] = 99
        df.loc[df["filename"] == df.iloc[-1]["filename"], "split"] = "test"
        df.loc[df["filename"] == df.iloc[-1]["filename"], "meta_cluster"] = 99
        df.to_csv(populated_mock_env.manifest_path, index=False)

        validator = SplitValidator(populated_mock_env)
        report = validator.validate()

        assert report.is_disjoint is False
