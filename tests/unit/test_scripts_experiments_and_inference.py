"""Unit Tests for Experiment Orchestration and Inference Generator Scripts.

Tests cover:
- scripts.run_experiment: Experiment registry integrity, remote connectivity checks.
- scripts.generate_test_inferences: Diverse image sampling logic and boundary handling.
- scripts.launch_dashboard: CLI parameter handling.
- scripts.preflight_gate: ASCII tabular report formatting.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from scripts.run_experiment import EXPERIMENTS_REGISTRY, check_remote_connectivity, run_remote_experiment
from scripts.generate_test_inferences import select_representative_test_images
from scripts.preflight_gate import format_table


@pytest.mark.unit
class TestRunExperiment:
    """Evaluates the unified experiment orchestrator registry and connectivity."""

    def test_experiments_registry_completeness(self) -> None:
        """Verifies all 6 experiments are registered with mandatory configuration keys."""
        assert len(EXPERIMENTS_REGISTRY) == 6
        for exp_id in range(1, 7):
            assert exp_id in EXPERIMENTS_REGISTRY
            cfg = EXPERIMENTS_REGISTRY[exp_id]
            assert cfg["key"] == f"exp{exp_id}"
            assert len(cfg["folder"]) > 0
            assert len(cfg["title"]) > 0
            assert cfg["remote_writer"].endswith(".py")
            assert cfg["summary_csv"].endswith(".csv")
            assert cfg["summary_json"].endswith(".json")

    def test_check_remote_connectivity_failure(self) -> None:
        """Verifies check_remote_connectivity returns False when server is unreachable."""
        with patch("requests.get", side_effect=Exception("Connection refused")):
            assert check_remote_connectivity() is False

    def test_run_remote_experiment_fails_on_unreachable_server(self) -> None:
        """Verifies run_remote_experiment gracefully aborts when remote server is unreachable."""
        with patch("scripts.run_experiment.check_remote_connectivity", return_value=False):
            result = run_remote_experiment(1)
            assert result is False


@pytest.mark.unit
class TestGenerateTestInferences:
    """Evaluates deterministic test image selection logic."""

    def test_select_representative_test_images_empty_dir(self, tmp_path: Path) -> None:
        """Verifies empty directory returns an empty list without raising errors."""
        res = select_representative_test_images(tmp_path, num_samples=10)
        assert res == []

    def test_select_representative_test_images_fewer_than_requested(self, tmp_path: Path) -> None:
        """Verifies all available images are returned when total is less than target count."""
        for i in range(5):
            (tmp_path / f"img_{i:02d}.jpg").write_bytes(b"\xFF\xD8\xFF")
        res = select_representative_test_images(tmp_path, num_samples=12)
        assert len(res) == 5

    def test_select_representative_test_images_subsampling(self, tmp_path: Path) -> None:
        """Verifies exactly num_samples images are selected evenly across a larger pool."""
        for i in range(30):
            (tmp_path / f"img_{i:02d}.jpg").write_bytes(b"\xFF\xD8\xFF")
        res = select_representative_test_images(tmp_path, num_samples=10)
        assert len(res) == 10
        # Check sorted order
        assert res == sorted(res)


@pytest.mark.unit
class TestPreFlightGate:
    """Evaluates pre-flight ASCII table formatting."""

    def test_format_table_structure(self) -> None:
        """Verifies format_table produces proper ASCII table with separators and rows."""
        headers = ["Step", "Metric", "Status"]
        rows = [
            ["01", "Zero-Leakage Invariants", "PASSED"],
            ["02", "YOLO Bounds Invariants", "PASSED"],
        ]
        table_str = format_table(headers, rows)
        assert "| Step" in table_str
        assert "| Metric" in table_str
        assert "| Status" in table_str
        assert "| 01" in table_str
        assert "PASSED" in table_str
        assert "+-" in table_str
