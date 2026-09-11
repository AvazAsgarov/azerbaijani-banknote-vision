"""Unit Tests for Exact and Perceptual Deduplication Engine.

Validates 64-bit DCT perceptual hashing, Laplacian focus variance,
exact SHA-256 binary matching, burst cluster pruning, and quarantine archiving.
"""

from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import pytest

from src.core.config import ProjectPaths
from src.data.deduplicate import (
    DeduplicationConfig,
    DeduplicationPipeline,
    DuplicateResult,
    PerceptualHasher,
)


@pytest.mark.unit
class TestPerceptualHasher:
    """Tests DCT hash transforms and focus sharpness calculations."""

    def test_compute_phash_valid_image(self, tmp_path: Path) -> None:
        """Verifies 64-bit boolean hash generation on valid synthetic image."""
        hasher = PerceptualHasher(hash_size=8, highfreq_factor=4)
        img_path = tmp_path / "test.jpg"
        img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)

        phash = hasher.compute_phash(img_path)
        assert phash is not None
        assert isinstance(phash, np.ndarray)
        assert phash.shape == (64,)
        assert phash.dtype == bool

    def test_compute_phash_nonexistent_image(self, tmp_path: Path) -> None:
        """Verifies graceful return of None when image cannot read from disk."""
        hasher = PerceptualHasher()
        missing_path = tmp_path / "absent_file.jpg"
        phash = hasher.compute_phash(missing_path)
        assert phash is None

    def test_compute_phash_identical_images(self, tmp_path: Path) -> None:
        """Verifies identical images produce zero Hamming distance."""
        hasher = PerceptualHasher()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(img, (20, 20), (80, 80), (255, 255, 255), -1)

        p1 = tmp_path / "img1.png"
        p2 = tmp_path / "img2.png"
        cv2.imwrite(str(p1), img)
        cv2.imwrite(str(p2), img)

        h1 = hasher.compute_phash(p1)
        h2 = hasher.compute_phash(p2)
        assert h1 is not None and h2 is not None
        hamming_dist = np.count_nonzero(h1 != h2)
        assert hamming_dist == 0

    def test_sharpness_calculation(self, tmp_path: Path) -> None:
        """Verifies Laplacian focus score differentiates sharp and blurry frames."""
        sharp = np.zeros((128, 128, 3), dtype=np.uint8)
        cv2.rectangle(sharp, (30, 30), (90, 90), (255, 255, 255), 2)
        cv2.circle(sharp, (64, 64), 20, (255, 255, 255), -1)

        blurry = cv2.GaussianBlur(sharp, (31, 31), 0)

        sharp_path = tmp_path / "sharp.png"
        blurry_path = tmp_path / "blurry.png"
        cv2.imwrite(str(sharp_path), sharp)
        cv2.imwrite(str(blurry_path), blurry)

        score_sharp = PerceptualHasher.compute_sharpness(sharp_path)
        score_blurry = PerceptualHasher.compute_sharpness(blurry_path)
        assert score_sharp > score_blurry

    def test_sharpness_missing_file(self, tmp_path: Path) -> None:
        """Verifies missing file returns zero sharpness score."""
        missing = tmp_path / "missing.jpg"
        score = PerceptualHasher.compute_sharpness(missing)
        assert score == 0.0


@pytest.mark.unit
class TestDeduplicationPipeline:
    """Tests exact duplicate and burst cluster deduplication execution."""

    def test_missing_manifest_raises_error(self, tmp_path: Path) -> None:
        """Verifies FileNotFoundError when manifest path points to missing file."""
        cfg = DeduplicationConfig(manifest_path=tmp_path / "missing_manifest.csv")
        pipeline = DeduplicationPipeline(cfg)
        with pytest.raises(FileNotFoundError):
            pipeline.run()

    def test_exact_binary_duplicate_detection(self, tmp_path: Path) -> None:
        """Verifies detection of bitwise identical files under distinct names."""
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        manifest_path = tmp_path / "manifest.csv"

        img = np.full((64, 64, 3), 100, dtype=np.uint8)
        cv2.imwrite(str(images_dir / "frame_001.png"), img)
        cv2.imwrite(str(images_dir / "frame_002_clone.png"), img)

        df = pd.DataFrame([
            {"filename": "frame_001.png", "class_name": "001_azn"},
            {"filename": "frame_002_clone.png", "class_name": "001_azn"},
        ])
        df.to_csv(manifest_path, index=False)

        cfg = DeduplicationConfig(
            manifest_path=manifest_path,
            images_dir=images_dir,
            hamming_threshold=4,
        )
        pipeline = DeduplicationPipeline(cfg)
        res = pipeline.run()

        assert res.initial_count == 2
        assert res.exact_duplicate_count == 1
        assert res.final_count == 1
        assert "frame_002_clone.png" in res.pruned_files

    def test_burst_near_duplicate_cluster_pruning(self, tmp_path: Path) -> None:
        """Verifies near-duplicate burst captures retain only the sharpest representative."""
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        manifest_path = tmp_path / "manifest.csv"

        base = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(base, (20, 20), (80, 80), (255, 255, 255), 2)

        sharp_img = base.copy()
        cv2.line(sharp_img, (0, 0), (100, 100), (255, 255, 255), 2)
        blurry_img = cv2.GaussianBlur(sharp_img, (15, 15), 0)

        cv2.imwrite(str(images_dir / "burst_01.png"), blurry_img)
        cv2.imwrite(str(images_dir / "burst_02.png"), sharp_img)

        df = pd.DataFrame([
            {"filename": "burst_01.png", "class_name": "005_azn"},
            {"filename": "burst_02.png", "class_name": "005_azn"},
        ])
        df.to_csv(manifest_path, index=False)

        cfg = DeduplicationConfig(
            manifest_path=manifest_path,
            images_dir=images_dir,
            hamming_threshold=10,
        )
        pipeline = DeduplicationPipeline(cfg)
        res = pipeline.run()

        assert res.initial_count == 2
        assert res.near_duplicate_pruned_count == 1
        assert res.final_count == 1
        # Sharpest image is retained
        assert "burst_02.png" in res.canonical_files
        assert "burst_01.png" in res.pruned_files

    def test_missing_image_on_disk_skipped(self, tmp_path: Path) -> None:
        """Verifies pipeline skips registered manifest records missing on disk without crashing."""
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        manifest_path = tmp_path / "manifest.csv"

        df = pd.DataFrame([{"filename": "ghost_frame.jpg", "class_name": "010_azn"}])
        df.to_csv(manifest_path, index=False)

        cfg = DeduplicationConfig(manifest_path=manifest_path, images_dir=images_dir)
        pipeline = DeduplicationPipeline(cfg)
        res = pipeline.run()
        assert res.initial_count == 1
        assert res.final_count == 1

    def test_apply_pruning_moves_files(self, tmp_path: Path) -> None:
        """Verifies apply_pruning relocates redundant files into quarantine archive."""
        root = tmp_path / "workspace"
        images_split_dir = root / "images"
        labels_split_dir = root / "labels"
        archive_dir = root / "pruned_archive"
        manifest_path = root / "manifest.csv"

        for split in ["train", "val", "test"]:
            (images_split_dir / split).mkdir(parents=True)
            (labels_split_dir / split).mkdir(parents=True)

        # Create dummy image and label in train split
        pruned_name = "redundant_001.jpg"
        lbl_name = "redundant_001.txt"
        (images_split_dir / "train" / pruned_name).write_text("img", encoding="utf-8")
        (labels_split_dir / "train" / lbl_name).write_text("0 0.5 0.5 0.2 0.2", encoding="utf-8")

        df = pd.DataFrame([{"filename": pruned_name, "class_name": "001_azn"}])
        df.to_csv(manifest_path, index=False)

        cfg = DeduplicationConfig(
            manifest_path=manifest_path,
            images_dir=images_split_dir / "train",
            labels_dir=labels_split_dir / "train",
            pruned_archive_dir=archive_dir,
            split_images_dir=images_split_dir,
            split_labels_dir=labels_split_dir,
        )
        pipeline = DeduplicationPipeline(cfg)

        mock_result = DuplicateResult(
            initial_count=1,
            exact_duplicate_count=1,
            near_duplicate_cluster_count=0,
            near_duplicate_pruned_count=0,
            final_count=0,
            exact_duplicates=[(pruned_name, "canonical.jpg")],
            pruned_files={pruned_name: "canonical.jpg"},
            canonical_files={"canonical.jpg"},
        )

        pipeline.apply_pruning(mock_result)

        # Confirm file moved to quarantine archive
        assert (archive_dir / "images" / pruned_name).exists()
        assert (archive_dir / "labels" / lbl_name).exists()
        assert not (images_split_dir / "train" / pruned_name).exists()

        # Confirm manifest updated
        updated_df = pd.read_csv(manifest_path)
        assert bool(updated_df.loc[0, "is_pruned"]) is True
