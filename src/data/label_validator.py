"""YOLO label validator enforcing AZN-Vision annotation quality gates."""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

NUM_CLASSES = 7


@dataclass
class ValidationError:
    file: str
    line_no: int
    message: str


@dataclass
class ValidationReport:
    errors: list = field(default_factory=list)
    files_checked: int = 0
    boxes_checked: int = 0

    def ok(self) -> bool:
        return len(self.errors) == 0


def validate_file(path: Path) -> Iterator[ValidationError]:
    for line_no, raw in enumerate(path.read_text().strip().splitlines(), start=1):
        parts = raw.split()
        if len(parts) != 5:
            yield ValidationError(str(path), line_no, f"Expected 5 fields, got {len(parts)}")
            continue
        try:
            cls_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:])
        except ValueError as exc:
            yield ValidationError(str(path), line_no, f"Parse error: {exc}")
            continue
        if not (0 <= cls_id < NUM_CLASSES):
            yield ValidationError(str(path), line_no, f"Class {cls_id} out of range [0, {NUM_CLASSES})")
        for name, val in (("cx", cx), ("cy", cy), ("w", w), ("h", h)):
            if not (0.0 <= val <= 1.0):
                yield ValidationError(str(path), line_no, f"{name}={val:.4f} outside [0, 1]")
        if w <= 0 or h <= 0:
            yield ValidationError(str(path), line_no, f"Non-positive dimension: w={w}, h={h}")


def validate_split(labels_dir: Path) -> ValidationReport:
    report = ValidationReport()
    for txt in sorted(labels_dir.rglob("*.txt")):
        report.files_checked += 1
        for err in validate_file(txt):
            report.errors.append(err)
        report.boxes_checked += sum(1 for _ in txt.read_text().strip().splitlines() if _.strip())
    return report
