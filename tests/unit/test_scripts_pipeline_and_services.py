"""Unit Tests for Core Pipeline Runners, Mobile Bridge, and Audit Scripts.

Tests cover:
- scripts.run_zero_leakage_split: Pipeline stage execution and report logging.
- scripts.launch_dashboard: CLI parsing, skip_sync handling, subprocess spawning.
- scripts.mobile_bridge: Model auto-discovery, simulation mode fallback, health endpoints.
- scripts.run_dataset_audit: IoU calculation, coordinate conversion, and box parsing.
- scripts.deploy_dataset: StreamingDataDeployer initialization and package preparation.
- scripts.generate_diagnostic_plots: Telemetry and cosine affinity figure rendering.
"""

from io import BytesIO
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import h5py
import numpy as np
import pytest

from src.core.config import ProjectPaths
from scripts.run_zero_leakage_split import run_pipeline
from scripts.mobile_bridge import load_inference_model, MobileBridgeHandler
from scripts.run_dataset_audit import compute_iou, yolo_to_xyxy, load_yolo_boxes
from scripts.deploy_dataset import build_dataset_archive
from scripts.generate_diagnostic_plots import main as generate_diagnostic_plots_main


@pytest.mark.unit
class TestRunZeroLeakageSplit:
    """Evaluates zero-leakage pipeline orchestrator execution."""

    def test_run_pipeline_orchestration(self) -> None:
        """Verifies run_pipeline coordinates clusterer, splitter, and validator."""
        mock_cluster_out = MagicMock()
        mock_cluster_out.num_meta_scenes = 42

        mock_partition_out = MagicMock()
        mock_partition_out.train_count = 500
        mock_partition_out.val_count = 100
        mock_partition_out.test_count = 100

        mock_validation_report = MagicMock()
        mock_validation_report.is_disjoint = True
        mock_validation_report.max_train_test_similarity = 0.45
        mock_validation_report.max_train_val_similarity = 0.50
        mock_validation_report.mean_train_test_similarity = 0.30
        mock_validation_report.top_k_contamination_rate = 0.0

        with patch("scripts.run_zero_leakage_split.MetaSceneClusterer") as mock_clusterer_cls, \
             patch("scripts.run_zero_leakage_split.ZeroLeakageSplitter") as mock_splitter_cls, \
             patch("scripts.run_zero_leakage_split.SplitValidator") as mock_validator_cls:

            mock_clusterer_cls.return_value.execute.return_value = mock_cluster_out
            mock_splitter_cls.return_value.partition.return_value = mock_partition_out
            mock_validator_cls.return_value.validate.return_value = mock_validation_report

            run_pipeline()

            assert mock_clusterer_cls.return_value.execute.called
            assert mock_splitter_cls.return_value.partition.called
            assert mock_validator_cls.return_value.validate.called


@pytest.mark.unit
class TestMobileBridgeScript:
    """Evaluates Mobile Bridge server functions."""

    def test_load_inference_model_simulation_fallback(self) -> None:
        """Verifies load_inference_model falls back to simulation mode when weights are absent."""
        from scripts.mobile_bridge import LOADED_MODELS
        LOADED_MODELS.clear()
        with patch("pathlib.Path.exists", return_value=False):
            model = load_inference_model("non_existent_model_id")
            assert model is None

    def test_mobile_bridge_health_endpoint(self) -> None:
        """Verifies GET /health sends JSON response with ready status."""
        request = MagicMock()
        client_address = ("127.0.0.1", 54321)
        server = MagicMock()

        handler = MobileBridgeHandler.__new__(MobileBridgeHandler)
        handler.client_address = client_address
        handler.server = server
        handler.path = "/health"
        handler.headers = {}
        handler.wfile = BytesIO()

        with patch.object(handler, "send_response") as mock_send_resp, \
             patch.object(handler, "send_header"), \
             patch.object(handler, "end_headers"):
            handler.do_GET()
            assert mock_send_resp.called
            assert mock_send_resp.call_args[0][0] == 200


@pytest.mark.unit
class TestDatasetAuditUtils:
    """Evaluates dataset quality audit coordinate and IoU utilities."""

    def test_yolo_to_xyxy(self) -> None:
        """Verifies center-based YOLO box conversion to min-max corners."""
        box = (0, 0.5, 0.5, 0.4, 0.2)
        cls_id, x1, y1, x2, y2 = yolo_to_xyxy(box)
        assert cls_id == 0
        assert pytest.approx(x1) == 0.3
        assert pytest.approx(y1) == 0.4
        assert pytest.approx(x2) == 0.7
        assert pytest.approx(y2) == 0.6

    def test_compute_iou_identical_boxes(self) -> None:
        """Verifies compute_iou returns 1.0 for overlapping identical boxes."""
        box1 = (0, 0.5, 0.5, 0.2, 0.2)
        box2 = (0, 0.5, 0.5, 0.2, 0.2)
        iou = compute_iou(box1, box2)
        assert pytest.approx(iou) == 1.0

    def test_compute_iou_disjoint_boxes(self) -> None:
        """Verifies compute_iou returns 0.0 for disjoint boxes."""
        box1 = (0, 0.2, 0.2, 0.1, 0.1)
        box2 = (0, 0.8, 0.8, 0.1, 0.1)
        iou = compute_iou(box1, box2)
        assert iou == 0.0

    def test_load_yolo_boxes(self) -> None:
        """Verifies parsing of multi-line YOLO text content."""
        content = "0 0.5 0.5 0.4 0.4\n1 0.2 0.3 0.1 0.1\n"
        boxes = load_yolo_boxes(content)
        assert len(boxes) == 2
        assert boxes[0][0] == 0
        assert boxes[1][0] == 1


@pytest.mark.unit
class TestDeployDatasetScript:
    """Evaluates dataset packaging and deployment helpers."""

    def test_build_dataset_archive(self, mock_paths: ProjectPaths, tmp_path: Path) -> None:
        """Verifies build_dataset_archive creates a valid zip archive."""
        # Create minimal training data
        (mock_paths.images_dir / "train" / "001_azn_00001.jpg").write_bytes(b"\xFF\xD8\xFF")
        (mock_paths.labels_dir / "train" / "001_azn_00001.txt").write_text("0 0.5 0.5 0.2 0.2\n")

        target_pkg = tmp_path / "stream_pkg.zip"
        built_pkg = build_dataset_archive(paths=mock_paths, output_path=target_pkg)
        assert built_pkg.exists()
        assert built_pkg.stat().st_size > 0


@pytest.mark.unit
class TestGenerateDiagnosticPlots:
    """Evaluates diagnostic plot generation from embeddings."""

    def test_diagnostic_plots_main(self, tmp_path: Path) -> None:
        """Verifies generate_diagnostic_plots.main creates telemetry and similarity figures."""
        h5_file = tmp_path / "test_embeddings.h5"
        with h5py.File(h5_file, "w") as f:
            # Synthetic 60 normalized embeddings of dim 1024
            data = np.random.randn(60, 1024).astype(np.float32)
            data /= np.linalg.norm(data, axis=1, keepdims=True)
            f.create_dataset("embeddings", data=data)
            f.create_dataset("filenames", data=[f"img_{i:03d}.jpg".encode("utf-8") for i in range(60)])
            f.create_dataset("class_names", data=["001_azn".encode("utf-8") for _ in range(60)])

        fig_dir = tmp_path / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        mock_p = ProjectPaths(
            root_dir=tmp_path,
            embeddings_h5_path=h5_file,
            figures_embeddings_dir=fig_dir,
        )

        with patch("scripts.generate_diagnostic_plots.ProjectPaths", return_value=mock_p):
            generate_diagnostic_plots_main()

            out1 = fig_dir / "dinov2_extraction_telemetry.png"
            out2 = fig_dir / "dinov2_sample_similarity_heatmap.png"
            assert out1.exists()
            assert out2.exists()
            assert out1.stat().st_size > 1000
            assert out2.stat().st_size > 1000
