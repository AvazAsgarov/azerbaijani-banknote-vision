"""Ingestion and Normalization Utility for Roboflow YOLO Bounding Box Labels.

Extracts canonical banknote annotations from a verified zip archive,
validates YOLO 5-parameter coordinates, and updates master and active
split label locations across the project.
"""

import json
import logging
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.core.config import ProjectPaths

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("LabelIngest")

PATHS = ProjectPaths()
DEFAULT_ZIP_PATH = PATHS.raw_data_dir / "fixed_labels.zip"
REGEX_PATTERN = re.compile(r"^[a-f0-9]+-(\d{3}_azn_\d{5})_jpg\.rf\.[a-zA-Z0-9_-]+\.txt$")


def validate_yolo_lines(content: str, filename: str) -> List[Tuple[int, float, float, float, float]]:
    """Validates YOLO 5-tuple format and coordinate bounds [0.0, 1.0]."""
    boxes = []
    lines = content.strip().splitlines()
    for line_idx, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) != 5:
            raise ValueError(f"Malformed YOLO line ({len(parts)} parts) in {filename}:{line_idx}: {line}")
        cls_id = int(parts[0])
        xc, yc, w, h = map(float, parts[1:])
        if not (0 <= cls_id <= 6):
            raise ValueError(f"Invalid class ID {cls_id} in {filename}:{line_idx}")
        if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0):
            raise ValueError(f"Center coords out of bounds in {filename}:{line_idx}: ({xc}, {yc})")
        if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
            raise ValueError(f"Box dimensions out of bounds in {filename}:{line_idx}: ({w}, {h})")
        boxes.append((cls_id, xc, yc, w, h))
    return boxes


def ingest_labels(zip_file_path: Optional[Path] = None) -> None:
    source_zip = zip_file_path or DEFAULT_ZIP_PATH
    if not source_zip.exists():
        logger.error("Source zip not found at: %s", source_zip)
        sys.exit(1)

    logger.info("Opening verified zip: %s", source_zip)
    with zipfile.ZipFile(source_zip, "r") as z:
        entries = [n for n in z.namelist() if n.startswith("train/labels/") and n.endswith(".txt")]
        print(f"Found {len(entries)} label files in zip.")

        stats = {
            "master": 0,
            "train": 0,
            "val": 0,
            "test": 0,
            "pruned": 0,
            "total_boxes": 0,
        }

        for entry in entries:
            bname = os.path.basename(entry)
            m = REGEX_PATTERN.match(bname)
            if not m:
                print(f"[WARNING] Skipping non-matching entry: {bname}")
                continue

            canon = m.group(1)
            raw_text = z.read(entry).decode("utf-8").strip()
            boxes = validate_yolo_lines(raw_text, bname)
            stats["total_boxes"] += len(boxes)

            # 1. Update master labels
            master_label_file = PATHS.master_labels_dir / f"{canon}.txt"
            master_label_file.parent.mkdir(parents=True, exist_ok=True)
            master_label_file.write_text(raw_text + ("\n" if raw_text else ""), encoding="utf-8")
            stats["master"] += 1

            # 2. Update split labels
            placed_in_split = False
            for split in ["train", "val", "test"]:
                img_path = PATHS.images_dir / split / f"{canon}.jpg"
                if img_path.exists():
                    target_lbl = PATHS.labels_dir / split / f"{canon}.txt"
                    target_lbl.parent.mkdir(parents=True, exist_ok=True)
                    target_lbl.write_text(raw_text + ("\n" if raw_text else ""), encoding="utf-8")
                    stats[split] += 1
                    placed_in_split = True
                    break

            if not placed_in_split:
                # Check pruned archive
                pruned_matches = list(PATHS.pruned_archive_dir.glob(f"**/{canon}.jpg"))
                if pruned_matches:
                    pruned_lbl_dir = PATHS.pruned_archive_dir / "labels"
                    pruned_lbl_dir.mkdir(parents=True, exist_ok=True)
                    pruned_lbl = pruned_lbl_dir / f"{canon}.txt"
                    pruned_lbl.write_text(raw_text + ("\n" if raw_text else ""), encoding="utf-8")
                    stats["pruned"] += 1
                else:
                    print(f"[WARNING] Image for {canon} not found in any split or pruned archive!")

    print("\nIngestion Summary:")
    print(f"  Master labels updated: {stats['master']}")
    print(f"  Train labels updated:  {stats['train']}")
    print(f"  Val labels updated:    {stats['val']}")
    print(f"  Test labels updated:   {stats['test']}")
    print(f"  Pruned labels updated: {stats['pruned']}")
    print(f"  Total bounding boxes:  {stats['total_boxes']}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest fixed Roboflow YOLO labels")
    parser.add_argument(
        "--zip-path",
        default=str(DEFAULT_ZIP_PATH),
        help=f"Path to verified labels zip archive (default: {DEFAULT_ZIP_PATH})",
    )
    args = parser.parse_args()
    ingest_labels(Path(args.zip_path))
