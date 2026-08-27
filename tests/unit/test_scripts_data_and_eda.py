"""Unit Tests for Dataset Curation, Label Ingestion, and EDA Scripts.

Tests cover:
- scripts.ingest_fixed_labels: YOLO coordinate boundary validation and parsing errors.
- scripts.verify_split_distribution: Class-stratified cross-tabulation table invariants.
- scripts.generate_tinyml_plots: Pareto Frontier and Safety Guardrail chart generation.
- scripts.diagnose_tinyml_metrics: Handling of missing model checkpoints.
- scripts.dino: DINOv2 linear probe validation and missing file defenses.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from src.core.config import ProjectPaths
from scripts.ingest_fixed_labels import validate_yolo_lines, ingest_labels
from scripts.verify_split_distribution import verify_distribution
from scripts.generate_tinyml_plots import generate_pareto_chart, generate_safety_guard_chart
from scripts.diagnose_tinyml_metrics import evaluate_genuine_map50
from scripts.dino import DINOv2LinearProbe


@pytest.mark.unit
class TestIngestFixedLabels:
    """Evaluates YOLO bounding box parsing and coordinate validation."""

    def test_validate_yolo_lines_valid(self) -> None:
        """Verifies valid YOLO 5-tuple lines are parsed accurately into float coordinates."""
        raw_text = "0 0.500000 0.500000 0.800000 0.400000\n6 0.250000 0.350000 0.150000 0.200000"
        boxes = validate_yolo_lines(raw_text, "valid_sample.txt")
        assert len(boxes) == 2
        assert boxes[0] == (0, 0.5, 0.5, 0.8, 0.4)
        assert boxes[1] == (6, 0.25, 0.35, 0.15, 0.2)

    def test_validate_yolo_lines_malformed_parts(self) -> None:
        """Verifies lines with incorrect element count raise ValueError."""
        raw_text = "0 0.5 0.5 0.8"  # 4 parts
        with pytest.raises(ValueError, match="Malformed YOLO line"):
            validate_yolo_lines(raw_text, "malformed.txt")

    def test_validate_yolo_lines_invalid_class_id(self) -> None:
        """Verifies class IDs outside [0, 6] range raise ValueError."""
        raw_text = "7 0.5 0.5 0.5 0.5"  # 7 is out of bounds
        with pytest.raises(ValueError, match="Invalid class ID"):
            validate_yolo_lines(raw_text, "bad_class.txt")

    def test_validate_yolo_lines_center_out_of_bounds(self) -> None:
        """Verifies center coordinates outside [0.0, 1.0] raise ValueError."""
        raw_text = "0 1.25 0.5 0.5 0.5"
        with pytest.raises(ValueError, match="Center coords out of bounds"):
            validate_yolo_lines(raw_text, "bad_center.txt")

    def test_validate_yolo_lines_dimension_out_of_bounds(self) -> None:
        """Verifies box dimensions <= 0 or > 1 raise ValueError."""
        raw_text = "0 0.5 0.5 0.0 0.5"  # zero width
        with pytest.raises(ValueError, match="Box dimensions out of bounds"):
            validate_yolo_lines(raw_text, "zero_dim.txt")

    def test_ingest_labels_missing_source_zip(self, tmp_path: Path) -> None:
        """Verifies ingest_labels exits cleanly when source zip file is absent."""
        missing_zip = tmp_path / "non_existent.zip"
        with pytest.raises(SystemExit):
            ingest_labels(missing_zip)


@pytest.mark.unit
class TestVerifySplitDistribution:
    """Evaluates split verification table calculation."""

    def test_verify_distribution_structure(self) -> None:
        """Verifies returned summary DataFrame contains expected columns and totals."""
        table = verify_distribution()
        assert isinstance(table, pd.DataFrame)
        required_cols = [
            "Denomination",
            "Total",
            "Train Count",
            "Train %",
            "Val Count",
            "Val %",
            "Test Count",
            "Test %",
        ]
        for col in required_cols:
            assert col in table.columns

        # Verify TOTAL row is present at the end
        total_row = table[table["Denomination"] == "TOTAL"]
        assert len(total_row) == 1
        grand_total = int(total_row["Total"].iloc[0])
        assert grand_total > 0


@pytest.mark.unit
class TestGenerateTinyMLPlots:
    """Evaluates generation of publication-quality TinyML figures."""

    def test_generate_pareto_chart(self, tmp_path: Path) -> None:
        """Verifies Pareto Frontier chart renders to disk at specified location."""
        out_path = tmp_path / "test_pareto.png"
        res = generate_pareto_chart(output_path=out_path)
        assert res.exists()
        assert res.stat().st_size > 1000

    def test_generate_safety_guard_chart(self, tmp_path: Path) -> None:
        """Verifies Multi-Tier Safety Guardrail flow diagram renders to disk."""
        out_path = tmp_path / "test_safety.png"
        res = generate_safety_guard_chart(output_path=out_path)
        assert res.exists()
        assert res.stat().st_size > 1000


@pytest.mark.unit
class TestDiagnoseTinyMLMetrics:
    """Evaluates TinyML diagnostic metrics evaluation helper."""

    def test_evaluate_genuine_map50_missing_checkpoint(self, tmp_path: Path) -> None:
        """Verifies evaluate_genuine_map50 returns empty dict when checkpoint does not exist."""
        non_existent_model = tmp_path / "missing_yolo.pt"
        result = evaluate_genuine_map50(non_existent_model)
        assert result == {}


@pytest.mark.unit
class TestDINOv2LinearProbe:
    """Evaluates DINOv2 feature probe loader defenses."""

    def test_missing_embeddings_file_raises_error(self, tmp_path: Path) -> None:
        """Verifies load_representations raises FileNotFoundError when embeddings do not exist."""
        probe = DINOv2LinearProbe(
            embeddings_path=tmp_path / "missing_emb.npy",
            metadata_path=tmp_path / "missing_meta.parquet",
        )
        with pytest.raises(FileNotFoundError, match="Embeddings file not located"):
            probe.load_representations()
