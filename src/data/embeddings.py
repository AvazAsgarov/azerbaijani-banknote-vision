"""Vision Transformer Feature Extraction and Representation Pipeline.

L2-normalized feature representations are extracted from pretrained self-supervised
DINOv2 vision transformers (e.g., ViT-L/14) to characterize high-dimensional image
manifolds. The resulting latent embeddings support semantic community clustering,
zero-leakage split partitioning, and diagnostic topological investigations.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm

from src.core.config import ProjectPaths

_DEFAULT_PATHS = ProjectPaths()
logger = logging.getLogger("DINOv2Extractor")


@dataclass
class ExtractionConfig:
    """Hyperparameters governing DINOv2 feature inference and artifact serialization.

    Attributes:
        model_name: Torch Hub repository identifier specifying the DINOv2 backbone.
        image_size: Square pixel dimension for spatial interpolation.
        batch_size: Number of images processed per forward inference pass.
        use_fp16: Boolean flag activating half-precision floating-point inference.
        num_workers: Number of background subprocesses for image loading.
        manifest_path: CSV dataset manifest path tracking active records.
        images_dir: Base directory housing processed master image files.
        output_h5_path: Primary destination path for the serialized HDF5 archive.
        output_parquet_path: Secondary destination path for metadata in Parquet format.
        output_npy_path: Destination path for uncompressed NumPy feature array export.
        seed: Random seed fixing deterministic PyTorch operations.
    """

    model_name: str = "dinov2_vitl14"
    image_size: int = 224
    batch_size: int = 8
    use_fp16: bool = True
    num_workers: int = 2
    manifest_path: Path = _DEFAULT_PATHS.manifest_path
    images_dir: Path = _DEFAULT_PATHS.master_images_dir
    output_h5_path: Path = _DEFAULT_PATHS.embeddings_h5_path
    output_parquet_path: Path = _DEFAULT_PATHS.embeddings_parquet_path
    output_npy_path: Path = _DEFAULT_PATHS.embeddings_npy_path
    seed: int = 42


class BanknoteDataset(Dataset):
    """PyTorch Dataset loading non-pruned curated banknote images and provenance records.

    Attributes:
        df: Pandas DataFrame containing verified non-pruned sample metadata.
        images_dir: Base directory housing master images.
        transform: Torchvision transformations applied to loaded PIL images.
    """

    def __init__(
        self,
        manifest_path: Path,
        images_dir: Path,
        transform: transforms.Compose,
        split_images_dir: Optional[Path] = None,
    ) -> None:
        """Initializes the dataset from manifest records and disk locations.

        Args:
            manifest_path: Path to dataset manifest CSV file.
            images_dir: Directory containing candidate image files.
            transform: Composed torchvision transformation pipeline.
            split_images_dir: Optional directory containing split subfolders.

        Raises:
            FileNotFoundError: Raised when manifest CSV fails to exist on disk.
        """
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        raw_df = pd.read_csv(manifest_path)
        if "is_pruned" in raw_df.columns:
            self.df = raw_df[~raw_df["is_pruned"].astype(bool)].reset_index(drop=True)
        else:
            self.df = raw_df.copy()

        self.images_dir = images_dir
        self.split_images_dir = split_images_dir or _DEFAULT_PATHS.images_dir
        self.transform = transform
        logger.info("Loaded %d active images from %s", len(self.df), manifest_path.name)

    def _resolve_image_path(self, filename: str) -> Path:
        """Resolves filesystem location for a sample image, checking master and split subfolders.

        Args:
            filename: Target image filename.

        Returns:
            Resolved Path pointing to existing file, or primary master directory path.
        """
        img_path = self.images_dir / filename
        if not img_path.exists():
            for split in ["train", "val", "test"]:
                alt_path = self.split_images_dir / split / filename
                if alt_path.exists():
                    return alt_path
        return img_path

    def __len__(self) -> int:
        """Returns the total number of active images in the dataset."""
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, str, str]:
        """Loads and transforms target sample at specified integer index.

        Args:
            idx: Integer index of requested sample.

        Returns:
            Tuple containing transformed tensor, filename, denomination, and folder string.
        """
        row = self.df.iloc[idx]
        fname = str(row["filename"])
        class_name = str(row.get("class_name", "unknown"))
        capture_folder = str(row.get("original_capture_folder", "unknown"))

        img_path = self._resolve_image_path(fname)
        img = Image.open(img_path).convert("RGB")
        return self.transform(img), fname, class_name, capture_folder


class FeatureExtractor:
    """Batched Vision Transformer inference and feature serialization engine.

    Attributes:
        config: ExtractionConfig supplying hyperparameters and path references.
        device: Torch computing hardware device (CUDA GPU or fallback CPU).
        transform: Image preprocessing transformations scaled to network specifications.
    """

    def __init__(self, config: Optional[ExtractionConfig] = None) -> None:
        """Initializes feature extraction engine and image transformation parameters.

        Args:
            config: Optional extraction configuration; defaults resolve to project paths.
        """
        self.config = config or ExtractionConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose([
            transforms.Resize(
                (self.config.image_size, self.config.image_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def load_model(self) -> nn.Module:
        """Loads and freezes the DINOv2 vision transformer backbone.

        Returns:
            Configured torch.nn.Module set to evaluation mode on target device.
        """
        logger.info("Loading pretrained %s from PyTorch Hub...", self.config.model_name)
        model = torch.hub.load("facebookresearch/dinov2", self.config.model_name)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        if self.config.use_fp16 and torch.cuda.is_available():
            model = model.half()
        return model.to(self.device)

    def extract(self) -> Tuple[np.ndarray, List[str], List[str], List[str]]:
        """Runs the extraction pipeline across all active dataset images.

        Returns:
            Tuple of (embeddings matrix, filenames, class_names, capture_folders).
        """
        dataset = BanknoteDataset(
            manifest_path=self.config.manifest_path,
            images_dir=self.config.images_dir,
            transform=self.transform,
        )
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        model = self.load_model()
        embeddings_list: List[np.ndarray] = []
        fnames_list: List[str] = []
        cnames_list: List[str] = []
        fldrs_list: List[str] = []

        logger.info("Starting inference on %s across %d samples...", self.device, len(dataset))
        start_time = time.perf_counter()

        with torch.inference_mode():
            for batch_imgs, fnames, cnames, fldrs in tqdm(loader, desc="DINOv2 Extraction"):
                batch_imgs = batch_imgs.to(self.device)
                if self.config.use_fp16 and torch.cuda.is_available():
                    batch_imgs = batch_imgs.half()

                feats = model(batch_imgs)
                feats = nn.functional.normalize(feats, p=2, dim=-1)

                embeddings_list.append(feats.cpu().float().numpy())
                fnames_list.extend(fnames)
                cnames_list.extend(cnames)
                fldrs_list.extend(fldrs)

        total_elapsed = time.perf_counter() - start_time
        all_embeddings = np.vstack(embeddings_list)
        logger.info(
            "Extracted %s embeddings in %.2f s (%.2f ms/image)",
            all_embeddings.shape, total_elapsed, (total_elapsed / len(dataset)) * 1000.0,
        )
        return all_embeddings, fnames_list, cnames_list, fldrs_list

    def save_h5(
        self,
        embeddings: np.ndarray,
        fnames: List[str],
        cnames: List[str],
        fldrs: List[str],
    ) -> Path:
        """Persists feature embeddings and metadata attributes into an HDF5 container.

        Args:
            embeddings: Normalized floating-point feature array of shape (N, D).
            fnames: List of sample filenames.
            cnames: List of class denomination strings.
            fldrs: List of source capture folder locations.

        Returns:
            Path pointing to written HDF5 file on disk.
        """
        out_path = self.config.output_h5_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Writing embeddings to %s", out_path)
        with h5py.File(out_path, "w") as h5f:
            h5f.create_dataset("embeddings", data=embeddings, compression="gzip", compression_opts=4)
            h5f.create_dataset("filenames", data=np.array(fnames, dtype=h5py.string_dtype()))
            h5f.create_dataset("class_names", data=np.array(cnames, dtype=h5py.string_dtype()))
            h5f.create_dataset("capture_folders", data=np.array(fldrs, dtype=h5py.string_dtype()))
            h5f.attrs["model_name"] = self.config.model_name
            h5f.attrs["embedding_dim"] = embeddings.shape[1]
            h5f.attrs["num_samples"] = embeddings.shape[0]
            h5f.attrs["normalized"] = True
            h5f.attrs["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

        logger.info("HDF5 file saved successfully (size: %.2f MB)", out_path.stat().st_size / 1024**2)
        return out_path

    def save_parquet(
        self,
        fnames: List[str],
        cnames: List[str],
        fldrs: List[str],
    ) -> Path:
        """Exports sample metadata and manifest attributes into an Apache Parquet file.

        Args:
            fnames: List of sample filenames.
            cnames: List of class denomination strings.
            fldrs: List of source capture folder locations.

        Returns:
            Path pointing to written Parquet file on disk.
        """
        out_path = self.config.output_parquet_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_meta = pd.DataFrame({
            "filename": fnames,
            "class_name": cnames,
            "original_capture_folder": fldrs,
        })
        if self.config.manifest_path.exists():
            df_manifest = pd.read_csv(self.config.manifest_path)
            if "split" in df_manifest.columns:
                df_meta = df_meta.merge(
                    df_manifest[["filename", "split", "meta_cluster"]],
                    on="filename",
                    how="left",
                )
        df_meta.to_parquet(out_path, index=False)
        logger.info("Metadata Parquet saved to: %s", out_path)
        return out_path

    def save_npy(self, embeddings: np.ndarray) -> Path:
        """Serializes dense feature array to uncompressed NumPy array format on disk.

        Args:
            embeddings: Normalized floating-point feature array of shape (N, D).

        Returns:
            Path pointing to written NumPy file on disk.
        """
        out_path = self.config.output_npy_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, embeddings)
        logger.info("NumPy feature array saved to: %s", out_path)
        return out_path

    def run(self) -> Tuple[Path, Path, Path]:
        """Runs complete extraction and serialization pipeline across all artifact formats.

        Extracts L2-normalized feature embeddings from active master images using the
        configured DINOv2 vision transformer, then writes the representations to HDF5,
        Apache Parquet metadata, and uncompressed NumPy formats.

        Returns:
            Tuple containing paths to the written (HDF5, Parquet, NumPy) files.
        """
        embeddings, fnames, cnames, fldrs = self.extract()
        h5_path = self.save_h5(embeddings, fnames, cnames, fldrs)
        parquet_path = self.save_parquet(fnames, cnames, fldrs)
        npy_path = self.save_npy(embeddings)
        return h5_path, parquet_path, npy_path

