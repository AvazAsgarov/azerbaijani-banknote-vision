"""Pipeline Invariant Tests for Spatial Coordinates and Bounding Box Geometry.

Enforces strict geometric contracts: all bounding box coordinates must lie
within normalized [0.0, 1.0] bounds with strictly positive widths and heights.
"""

from pathlib import Path
import pandas as pd
import pytest

from src.core.config import ProjectPaths


@pytest.mark.invariant
class TestSpatialBoundsInvariants:
    """Mathematical geometric assertions verifying normalized annotation bounds."""

    @pytest.fixture(scope="class")
    def paths(self) -> ProjectPaths:
        """Returns project paths container."""
        return ProjectPaths()

    @pytest.fixture(scope="class")
    def active_labels(self, paths: ProjectPaths) -> pd.DataFrame:
        """Loads non-pruned active filenames from manifest."""
        df = pd.read_csv(paths.manifest_path)
        return df[df["is_pruned"] == False].copy().reset_index(drop=True)

    def test_bounding_box_coordinates_within_bounds(
        self, paths: ProjectPaths, active_labels: pd.DataFrame
    ) -> None:
        """Verifies every annotation in master_labels complies with [0.0, 1.0] bounds."""
        eps = 1e-4
        total_boxes = 0
        overflow_boxes = []

        for fname in active_labels["filename"]:
            lbl_name = fname.replace(".jpg", ".txt").replace(".png", ".txt")
            lbl_file = paths.master_labels_dir / lbl_name
            assert lbl_file.exists(), f"Missing label file: {lbl_file}"

            content = lbl_file.read_text(encoding="utf-8").strip()
            assert len(content) > 0, f"Empty label file discovered: {lbl_file}"

            for line_idx, line in enumerate(content.splitlines()):
                parts = line.strip().split()
                assert len(parts) == 5, f"Malformed YOLO annotation line in {lbl_name}: {line}"

                cls_id = int(parts[0])
                xc, yc, w, h = map(float, parts[1:])
                total_boxes += 1

                # Denomination class ID bounds assertion
                assert 0 <= cls_id <= 6, f"Class ID {cls_id} outside [0, 6] in {lbl_name}"

                # Dimension positivity assertion
                assert w > 0.0, f"Non-positive box width in {lbl_name}: w={w}"
                assert h > 0.0, f"Non-positive box height in {lbl_name}: h={h}"

                # Corner boundary assertion
                xmin = xc - w / 2.0
                ymin = yc - h / 2.0
                xmax = xc + w / 2.0
                ymax = yc + h / 2.0

                if (
                    xmin < -eps
                    or ymin < -eps
                    or xmax > 1.0 + eps
                    or ymax > 1.0 + eps
                ):
                    overflow_boxes.append((lbl_name, line_idx, xmin, ymin, xmax, ymax))

        assert len(overflow_boxes) == 0, f"Found {len(overflow_boxes)} overflowing boxes: {overflow_boxes[:5]}"
        assert total_boxes == 2561, f"Expected 2,561 active bounding boxes, discovered {total_boxes}"

    def test_split_directories_box_count_reconciliation(
        self, paths: ProjectPaths, active_labels: pd.DataFrame
    ) -> None:
        """Verifies split directory annotations match master counts exactly."""
        split_total_boxes = 0
        for split_name in ["train", "val", "test"]:
            lbl_dir = paths.labels_dir / split_name
            assert lbl_dir.exists(), f"Split directory missing: {lbl_dir}"
            for lbl_file in lbl_dir.glob("*.txt"):
                content = lbl_file.read_text(encoding="utf-8").strip()
                if content:
                    split_total_boxes += len(content.splitlines())

        assert split_total_boxes == 2561, f"Split directories box count {split_total_boxes} != 2561"
