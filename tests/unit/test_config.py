"""Unit Tests for Core Configuration and Dataclass Containers.

Evaluates configuration immutability, path integrity, hyperparameter ranges,
and boundary condition handling for dataset partitioning settings.
"""

from dataclasses import FrozenInstanceError
from pathlib import Path
import pytest

from src.core.config import (
    ClusterConfig,
    DatasetQualityConfig,
    MobileBridgeConfig,
    ProjectPaths,
    SplitConfig,
)


@pytest.mark.unit
class TestProjectPaths:
    """Tests path resolution and directory structure definitions."""

    def test_default_paths_exist(self) -> None:
        """Verifies default paths resolve relative to repository root."""
        paths = ProjectPaths()
        assert paths.root_dir.is_dir()
        assert paths.configs_dir == paths.root_dir / "configs"
        assert paths.processed_data_dir == paths.root_dir / "data" / "processed"
        assert paths.manifest_path.name == "dataset_manifest.csv"

    def test_frozen_immutability(self) -> None:
        """Verifies that mutating path attributes raises FrozenInstanceError."""
        paths = ProjectPaths()
        with pytest.raises(FrozenInstanceError):
            paths.root_dir = Path("/mutated/path")  # type: ignore

    def test_custom_paths_initialization(self, tmp_path: Path) -> None:
        """Verifies initialization when custom root directories are specified."""
        custom_root = tmp_path / "custom_project"
        custom_root.mkdir()
        paths = ProjectPaths(root_dir=custom_root)
        assert paths.root_dir == custom_root
        assert str(paths.configs_dir).startswith(str(custom_root))


@pytest.mark.unit
class TestClusterConfig:
    """Tests clustering hyperparameter validation and defaults."""

    def test_cluster_defaults(self) -> None:
        """Verifies canonical distance thresholds and linkage criteria."""
        cfg = ClusterConfig()
        assert cfg.distance_threshold == 0.20
        assert cfg.linkage == "average"
        assert cfg.metric == "precomputed"

    def test_cluster_immutability(self) -> None:
        """Verifies that cluster configuration attributes reject mutation."""
        cfg = ClusterConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.distance_threshold = 0.50  # type: ignore

    def test_custom_cluster_parameters(self) -> None:
        """Verifies custom threshold assignments."""
        cfg = ClusterConfig(distance_threshold=0.35, linkage="complete")
        assert cfg.distance_threshold == 0.35
        assert cfg.linkage == "complete"


@pytest.mark.unit
class TestSplitConfig:
    """Tests split ratio definitions and denomination class containers."""

    def test_split_defaults(self) -> None:
        """Verifies standard 70/15/15 target proportions and denomination tuple."""
        cfg = SplitConfig()
        assert pytest.approx(cfg.train_ratio + cfg.val_ratio + cfg.test_ratio) == 1.0
        assert cfg.train_ratio == 0.70
        assert cfg.val_ratio == 0.15
        assert cfg.test_ratio == 0.15
        assert len(cfg.classes) == 7
        assert "001_azn" in cfg.classes
        assert "200_azn" in cfg.classes

    def test_split_immutability(self) -> None:
        """Verifies that split configuration rejects in-place attribute updates."""
        cfg = SplitConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.train_ratio = 0.80  # type: ignore

    def test_custom_split_ratios(self) -> None:
        """Verifies custom ratio assignments for experimental setups."""
        cfg = SplitConfig(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=42)
        assert cfg.train_ratio == 0.8
        assert cfg.random_seed == 42
        assert cfg.optimization_iterations == 3000


@pytest.mark.unit
class TestDatasetQualityConfig:
    """Tests production quality benchmarks and denomination count specifications."""

    def test_quality_defaults(self) -> None:
        cfg = DatasetQualityConfig()
        assert cfg.min_detection_images == 2000
        assert cfg.min_classes == 7

    def test_quality_immutability(self) -> None:
        cfg = DatasetQualityConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.min_detection_images = 5000  # type: ignore


@pytest.mark.unit
class TestMobileBridgeConfig:
    """Tests mobile inference bridge configuration parameters."""

    def test_bridge_defaults(self) -> None:
        cfg = MobileBridgeConfig()
        assert cfg.default_host == "0.0.0.0"
        assert cfg.default_port == 8000
        assert cfg.default_conf_threshold == 0.50
        assert len(cfg.simulation_default_bbox) == 4

    def test_bridge_immutability(self) -> None:
        cfg = MobileBridgeConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.default_port = 9000  # type: ignore
