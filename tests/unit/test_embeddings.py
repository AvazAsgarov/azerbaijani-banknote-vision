"""Unit Tests for Vision Transformer Embedding Extraction and Storage.

Validates BanknoteDataset indexing, image transforms, batch assembly,
HDF5 container serialization, metadata Parquet exports, and dense NumPy saving.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import cv2
import h5py
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from torchvision import transforms

from src.core.config import ProjectPaths
from src.data.embeddings import BanknoteDataset, ExtractionConfig, FeatureExtractor


@pytest.mark.unit
class TestBanknoteDataset:
    """Tests custom PyTorch dataset loading and sample transformations."""

    def test_missing_manifest_raises_error(self, tmp_path: Path) -> None:
        """Verifies FileNotFoundError upon missing manifest CSV path."""
        transform = transforms.ToTensor()
        with pytest.raises(FileNotFoundError):
            BanknoteDataset(
                manifest_path=tmp_path / "absent.csv",
                images_dir=tmp_path / "images",
                transform=transform,
            )

    def test_pruned_samples_filtered(self, tmp_path: Path) -> None:
        """Verifies dataset excludes pruned frames from indexing."""
        manifest_path = tmp_path / "manifest.csv"
        images_dir = tmp_path / "images"
        images_dir.mkdir()

        img = np.zeros((64, 64, 3), dtype=np.uint8)
        cv2.imwrite(str(images_dir / "active.jpg"), img)
        cv2.imwrite(str(images_dir / "pruned.jpg"), img)

        df = pd.DataFrame([
            {"filename": "active.jpg", "class_name": "001_azn", "is_pruned": False},
            {"filename": "pruned.jpg", "class_name": "001_azn", "is_pruned": True},
        ])
        df.to_csv(manifest_path, index=False)

        dataset = BanknoteDataset(manifest_path, images_dir, transforms.ToTensor())
        assert len(dataset) == 1
        tensor, fname, cname, _ = dataset[0]
        assert fname == "active.jpg"
        assert cname == "001_azn"
        assert tensor.shape == (3, 64, 64)

    def test_fallback_to_split_directories(self, tmp_path: Path) -> None:
        """Verifies fallback image discovery across partitioned split subfolders."""
        root = tmp_path / "workspace"
        images_dir = root / "data" / "processed" / "master" / "images"
        split_dir = root / "data" / "processed" / "images" / "train"
        split_dir.mkdir(parents=True)
        images_dir.mkdir(parents=True)

        fname = "split_frame.jpg"
        img = np.full((32, 32, 3), 200, dtype=np.uint8)
        cv2.imwrite(str(split_dir / fname), img)

        manifest_path = root / "manifest.csv"
        df = pd.DataFrame([{"filename": fname, "class_name": "005_azn", "is_pruned": False}])
        df.to_csv(manifest_path, index=False)

        split_images_dir = root / "data" / "processed" / "images"
        dataset = BanknoteDataset(
            manifest_path=manifest_path,
            images_dir=images_dir,
            transform=transforms.ToTensor(),
            split_images_dir=split_images_dir,
        )
        assert len(dataset) == 1
        tensor, ret_fname, _, _ = dataset[0]
        assert ret_fname == fname
        assert tensor.shape == (3, 32, 32)


@pytest.mark.unit
class TestFeatureExtractor:
    """Tests feature extraction pipelines and multi-format serialization."""

    def test_serialization_pipeline(self, tmp_path: Path) -> None:
        """Verifies HDF5, Parquet, and NumPy export mechanisms."""
        cfg = ExtractionConfig(
            manifest_path=tmp_path / "manifest.csv",
            images_dir=tmp_path / "images",
            output_h5_path=tmp_path / "artifacts" / "test_embeddings.h5",
            output_parquet_path=tmp_path / "artifacts" / "test_metadata.parquet",
            output_npy_path=tmp_path / "artifacts" / "test_embeddings.npy",
        )
        extractor = FeatureExtractor(cfg)

        # Synthetic feature tensors: 5 samples, 1024 dimensions
        rng = np.random.RandomState(42)
        raw_feats = rng.randn(5, 1024).astype(np.float32)
        normed_feats = raw_feats / np.linalg.norm(raw_feats, axis=1, keepdims=True)

        fnames = [f"sample_{i:03d}.jpg" for i in range(5)]
        cnames = ["001_azn"] * 5
        fldrs = ["series_a"] * 5

        # Test H5 serialization
        h5_path = extractor.save_h5(normed_feats, fnames, cnames, fldrs)
        assert h5_path.exists()
        with h5py.File(h5_path, "r") as h5f:
            assert h5f["embeddings"].shape == (5, 1024)
            assert bool(h5f.attrs["normalized"]) is True
            saved_fnames = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in h5f["filenames"][:]]
            assert saved_fnames == fnames

        # Test Parquet serialization
        parquet_path = extractor.save_parquet(fnames, cnames, fldrs)
        assert parquet_path.exists()
        df_meta = pd.read_parquet(parquet_path)
        assert len(df_meta) == 5
        assert list(df_meta["filename"]) == fnames

        # Test NumPy serialization
        npy_path = extractor.save_npy(normed_feats)
        assert npy_path.exists()
        loaded_npy = np.load(npy_path)
        assert loaded_npy.shape == (5, 1024)
        np.testing.assert_allclose(loaded_npy, normed_feats)

    def test_mocked_extract_execution(self, tmp_path: Path) -> None:
        """Verifies batched inference flow using a mock vision backbone."""
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        manifest_path = tmp_path / "manifest.csv"

        rows = []
        for i in range(4):
            fname = f"img_{i:02d}.jpg"
            img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            cv2.imwrite(str(images_dir / fname), img)
            rows.append({
                "filename": fname,
                "class_name": "010_azn",
                "original_capture_folder": "folder_1",
                "is_pruned": False,
            })
        pd.DataFrame(rows).to_csv(manifest_path, index=False)

        cfg = ExtractionConfig(
            manifest_path=manifest_path,
            images_dir=images_dir,
            image_size=64,
            batch_size=2,
            use_fp16=False,
            num_workers=0,
        )
        extractor = FeatureExtractor(cfg)

        # Mock model outputting 128-dim features
        class MockVisionModel(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                b = x.shape[0]
                # Return constant mock vector per sample
                return torch.ones((b, 128), dtype=torch.float32)

        mock_model = MockVisionModel()
        with patch.object(extractor, "load_model", return_value=mock_model):
            embeddings, fnames, cnames, fldrs = extractor.extract()

            assert embeddings.shape == (4, 128)
            assert len(fnames) == 4
            assert len(cnames) == 4
            assert len(fldrs) == 4
            # Assert L2 normalization was computed
            norms = np.linalg.norm(embeddings, axis=1)
            np.testing.assert_allclose(norms, np.ones(4), atol=1e-5)

    def test_load_model(self) -> None:
        """Verifies model loading and evaluation mode setting."""
        cfg = ExtractionConfig(use_fp16=False)
        extractor = FeatureExtractor(cfg)
        mock_nn = nn.Linear(10, 10)
        with patch("torch.hub.load", return_value=mock_nn) as mock_hub:
            loaded = extractor.load_model()
            assert loaded is not None
            mock_hub.assert_called_once()
            # Verify parameters have requires_grad disabled
            for p in loaded.parameters():
                assert p.requires_grad is False

