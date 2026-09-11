"""Unit Tests for Modular Experiment Management, Telemetry Schemas, and Visualization.

Tests data contracts, directory lifecycle management, telemetry serialization,
figure generation adhering to EDADesignSystem, and test inference integrity.
"""

from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.core.config import ProjectPaths
from src.experiments.schema import ArmConfig, RunMetrics, TestInferenceSample, ExperimentMetadata
from src.experiments.persistence import ExperimentArtifactManager
from src.experiments.visualizer import ExperimentVisualizer


@pytest.fixture
def exp_manager(mock_paths: ProjectPaths) -> ExperimentArtifactManager:
    """Provides an isolated ExperimentArtifactManager for unit testing."""
    return ExperimentArtifactManager("test_exp_dummy", paths=mock_paths)


class TestExperimentSchemas:
    """Verifies dataclass schemas and serialization behavior for experiment telemetry."""

    def test_arm_config_creation(self) -> None:
        cfg = ArmConfig(
            arm_id="arm1_baseline",
            display_name="Arm 1: Baseline",
            model_architecture="yolo11m",
            description="Initial control group",
            hparams={"imgsz": 640, "batch": 16, "lr0": 0.001},
        )
        assert cfg.arm_id == "arm1_baseline"
        assert cfg.hparams["imgsz"] == 640
        assert "lr0" in cfg.hparams

    def test_run_metrics_defaults_and_serialization(self) -> None:
        metrics = RunMetrics(
            arm_id="yolo11m",
            val_map50=0.965,
            val_map50_95=0.912,
            test_map50=0.958,
            test_map50_95=0.908,
            parameters_m=20.1,
            latency_ms=5.8,
            fps=172.4,
            model_size_mb=40.5,
        )
        data = asdict(metrics)
        assert data["arm_id"] == "yolo11m"
        assert data["val_map50_95"] == 0.912
        assert data["latency_ms"] == 5.8
        assert data["per_class_test_ap"] == {}

    def test_test_inference_sample_schema(self) -> None:
        sample = TestInferenceSample(
            image_filename="note_001.jpg",
            predictions=[{
                "class_id": 4,
                "class_name": "050_azn",
                "confidence": 0.942,
                "bbox_xyxy": [50.0, 60.0, 400.0, 300.0],
            }],
            ground_truths=[{
                "class_id": 4,
                "class_name": "050_azn",
                "bbox_xyxy": [52.0, 58.0, 398.0, 302.0],
            }],
            is_exact_match=True,
        )
        assert sample.image_filename == "note_001.jpg"
        assert len(sample.predictions) == 1
        assert sample.is_exact_match is True

    def test_experiment_metadata(self) -> None:
        meta = ExperimentMetadata(
            exp_id="exp1_architecture_battle",
            title="Architecture Battle",
            hypothesis="Modern spatial-attention CNNs will outperform frozen ViT linear probes.",
            independent_variables=["backbone_architecture", "attention_mechanism"],
            control_variables=["epochs", "image_size", "dataset_split"],
            evaluation_protocol="Zero-leakage test split mAP@0.5:0.95 and GPU inference latency",
        )
        assert meta.exp_id == "exp1_architecture_battle"
        assert len(meta.independent_variables) == 2
        assert len(meta.control_variables) == 3


class TestArtifactManager:
    """Verifies physical directory layout and file consolidation."""

    def test_initialization_creates_base_directories(self, exp_manager: ExperimentArtifactManager) -> None:
        assert exp_manager.exp_dir.exists()
        assert exp_manager.figures_dir.exists()
        assert exp_manager.reports_dir.exists()

    def test_ensure_arm_directories(self, exp_manager: ExperimentArtifactManager) -> None:
        arm_id = "test_arm_01"
        subdirs = exp_manager.ensure_arm_directories(arm_id)

        assert "weights" in subdirs
        assert "curves" in subdirs
        assert "logs" in subdirs
        assert "test_eval" in subdirs
        assert "test_inferences" in subdirs

        for key, path in subdirs.items():
            assert path.exists(), f"Subdirectory '{key}' was not created at {path}"

    def test_save_run_artifacts_migration(self, exp_manager: ExperimentArtifactManager, tmp_path: Path) -> None:
        arm_id = "arm_migration_test"
        mock_source = tmp_path / "mock_yolo_run"
        mock_source.mkdir()

        # Create mock training artifacts
        (mock_source / "results.csv").write_text("epoch,loss\n1,0.5\n2,0.3\n", encoding="utf-8")
        (mock_source / "args.yaml").write_text("epochs: 50\n", encoding="utf-8")
        (mock_source / "results.png").write_bytes(b"PNG_MOCK")
        (mock_source / "confusion_matrix.png").write_bytes(b"PNG_MOCK")

        mock_weights = mock_source / "weights"
        mock_weights.mkdir()
        (mock_weights / "best.pt").write_bytes(b"PT_MOCK_WEIGHTS")

        metrics = RunMetrics(arm_id=arm_id, val_map50=0.92, val_map50_95=0.88)
        saved = exp_manager.save_run_artifacts(
            arm_id=arm_id,
            metrics=metrics,
            source_dir=mock_source,
            weights_source=mock_weights,
        )

        arm_dir = exp_manager.get_arm_dir(arm_id)
        assert (arm_dir / "weights" / "best.pt").exists()
        assert (arm_dir / "curves" / "results.png").exists()
        assert (arm_dir / "curves" / "confusion_matrix.png").exists()
        assert (arm_dir / "logs" / "results.csv").exists()
        assert (arm_dir / "metrics_summary.json").exists()

    def test_clean_duplicate_root_figures(self, exp_manager: ExperimentArtifactManager) -> None:
        loose_fig = exp_manager.exp_dir / "loose_pareto.png"
        loose_fig.write_bytes(b"MOCK_FIGURE_DATA")

        cleaned = exp_manager.clean_duplicate_root_figures()
        assert len(cleaned) == 1
        assert not loose_fig.exists(), "Loose figure in root should have been moved/cleaned"
        assert (exp_manager.figures_dir / "loose_pareto.png").exists(), "Figure should now reside in figures/"

    def test_export_summary(self, exp_manager: ExperimentArtifactManager) -> None:
        records = [
            {"arm_id": "arm1", "map50_95": 0.85, "fps": 120.0},
            {"arm_id": "arm2", "map50_95": 0.91, "fps": 172.0},
        ]
        csv_p, json_p = exp_manager.export_summary(records, base_name="test_summary")

        assert csv_p.exists()
        assert json_p.exists()

        df = pd.read_csv(csv_p)
        assert len(df) == 2
        assert "map50_95" in df.columns


class TestExperimentVisualizer:
    """Verifies EDADesignSystem compliant figure generation."""

    def test_pareto_chart_generation(self, exp_manager: ExperimentArtifactManager) -> None:
        df = pd.DataFrame([
            {"model_id": "yolov8m", "architecture": "YOLOv8m", "test_map50_95": 0.884, "latency_gpu_ms": 7.8, "parameters_m": 25.9},
            {"model_id": "yolo11m", "architecture": "YOLOv11m", "test_map50_95": 0.912, "latency_gpu_ms": 5.8, "parameters_m": 20.1},
        ])
        fig_path = exp_manager.figures_dir / "test_pareto.png"
        res_path = ExperimentVisualizer.plot_exp1_pareto_frontier(df, fig_path)

        assert res_path.exists()
        assert res_path.stat().st_size > 0

    def test_delta_bar_chart_generation(self, exp_manager: ExperimentArtifactManager) -> None:
        df = pd.DataFrame([
            {"arm_id": "arm1_none_raw", "display_name": "Raw Baseline", "test_map50_95": 0.806},
            {"arm_id": "arm4_full_composite", "display_name": "Full Composite", "test_map50_95": 0.908},
        ])
        fig_path = exp_manager.figures_dir / "test_delta.png"
        res_path = ExperimentVisualizer.plot_exp2_augmentation_delta(df, fig_path)

        assert res_path.exists()
        assert res_path.stat().st_size > 0

    def test_color_space_comparison_chart(self, exp_manager: ExperimentArtifactManager) -> None:
        df = pd.DataFrame([
            {"arm_id": "arm1_rgb", "color_space": "rgb", "test_map50": 0.96, "test_map50_95": 0.912},
            {"arm_id": "arm2_hsv", "color_space": "hsv", "test_map50": 0.94, "test_map50_95": 0.894},
            {"arm_id": "arm3_gray", "color_space": "gray", "test_map50": 0.92, "test_map50_95": 0.863},
        ])
        fig_path = exp_manager.figures_dir / "test_color.png"
        res_path = ExperimentVisualizer.plot_exp3_color_space_comparison(df, fig_path)

        assert res_path.exists()
        assert res_path.stat().st_size > 0
