"""Unit tests for the YOLO label validator."""

from pathlib import Path
import pytest
from src.data.label_validator import validate_split


def _write_label(tmp, name, lines):
    p = tmp / name
    p.write_text("\n".join(lines))
    return p


class TestLabelValidator:
    def test_valid_labels_produce_empty_report(self, tmp_path):
        _write_label(tmp_path, "img001.txt", ["0 0.5 0.5 0.3 0.2"])
        _write_label(tmp_path, "img002.txt", ["3 0.4 0.6 0.25 0.15", "6 0.7 0.3 0.2 0.1"])
        report = validate_split(tmp_path)
        assert report.ok()
        assert report.files_checked == 2
        assert report.boxes_checked == 3

    def test_out_of_range_class_id_is_flagged(self, tmp_path):
        _write_label(tmp_path, "bad.txt", ["7 0.5 0.5 0.2 0.1"])
        report = validate_split(tmp_path)
        assert not report.ok()
        assert any("Class 7" in e.message for e in report.errors)

    def test_coordinate_exceeding_one_is_flagged(self, tmp_path):
        _write_label(tmp_path, "bad.txt", ["2 0.5 1.1 0.2 0.1"])
        report = validate_split(tmp_path)
        assert not report.ok()

    def test_zero_width_box_is_flagged(self, tmp_path):
        _write_label(tmp_path, "bad.txt", ["1 0.5 0.5 0.0 0.2"])
        report = validate_split(tmp_path)
        assert not report.ok()

    def test_malformed_line_is_flagged(self, tmp_path):
        _write_label(tmp_path, "bad.txt", ["0 0.5 0.5 0.2"])
        report = validate_split(tmp_path)
        assert not report.ok()
