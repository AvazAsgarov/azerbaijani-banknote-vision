"""Exact Binary and Perceptual Near-Duplicate Deduplication Engine.

Production-grade exact (SHA-256) and perceptual hash (DCT-based pHash) routines
detect and isolate redundant image captures. Perceptual clusters produced by
camera bursts are resolved by retaining the candidate maximizing Laplacian
gradient variance (sharpness score), while inferior frames move into quarantine.
"""

from collections import defaultdict
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import pandas as pd

from src.core.config import ProjectPaths

_DEFAULT_PATHS = ProjectPaths()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("DeduplicationEngine")


@dataclass
class DeduplicationConfig:
    """Hyperparameters governing exact and perceptual duplicate detection.

    Attributes:
        hash_size: Spatial frequency dimension for discrete cosine transform.
        highfreq_factor: Scaling factor determining intermediate resize resolution.
        hamming_threshold: Maximum bitwise difference establishing near-duplicate identity.
        chunk_size: Block size for vectorized matrix distance calculations.
        manifest_path: Target CSV path tracking dataset provenance.
        images_dir: Source directory containing unpartitioned master images.
        labels_dir: Source directory containing unpartitioned YOLO annotations.
        pruned_archive_dir: Destination quarantine folder for rejected duplicates.
        report_output_path: Destination JSON path recording duplicate audit statistics.
    """

    hash_size: int = 8
    highfreq_factor: int = 4
    hamming_threshold: int = 4
    chunk_size: int = 500
    manifest_path: Path = _DEFAULT_PATHS.manifest_path
    images_dir: Path = _DEFAULT_PATHS.master_images_dir
    labels_dir: Path = _DEFAULT_PATHS.master_labels_dir
    pruned_archive_dir: Path = _DEFAULT_PATHS.pruned_archive_dir
    split_images_dir: Path = _DEFAULT_PATHS.images_dir
    split_labels_dir: Path = _DEFAULT_PATHS.labels_dir
    report_output_path: Path = _DEFAULT_PATHS.audit_report_path



@dataclass
class DuplicateResult:
    """Quantitative summary and record mappings produced by deduplication.

    Attributes:
        initial_count: Total registered images evaluated during execution.
        exact_duplicate_count: Number of exact binary duplicate files identified.
        near_duplicate_cluster_count: Number of distinct perceptual burst clusters.
        near_duplicate_pruned_count: Number of redundant burst frames designated for pruning.
        final_count: Total unique images remaining in active dataset service.
        exact_duplicates: Pairings of redundant files with canonical counterparts.
        pruned_files: Dictionary mapping pruned filenames to canonical targets.
        canonical_files: Set of retained representative filenames.
    """

    initial_count: int
    exact_duplicate_count: int
    near_duplicate_cluster_count: int
    near_duplicate_pruned_count: int
    final_count: int
    exact_duplicates: List[Tuple[str, str]]
    pruned_files: Dict[str, str]
    canonical_files: Set[str]


class PerceptualHasher:
    """Discrete Cosine Transform (DCT) 64-bit perceptual hash engine.

    Attributes:
        hash_size: Spatial frequency cutoff dimension.
        highfreq_factor: Multiplier determining input scaling resolution.
        img_size: Target square pixel dimension for preprocessing.
    """

    def __init__(self, hash_size: int = 8, highfreq_factor: int = 4) -> None:
        """Initializes hasher resolution parameters.

        Args:
            hash_size: Low-frequency grid dimension extracted from DCT matrix.
            highfreq_factor: Scaling factor controlling intermediate resize size.
        """
        self.hash_size = hash_size
        self.highfreq_factor = highfreq_factor
        self.img_size = hash_size * highfreq_factor

    def compute_phash(self, image_path: Path) -> Optional[np.ndarray]:
        """Calculates 64-bit boolean DCT perceptual hash for target image file.

        Args:
            image_path: Filesystem path pointing to the input image.

        Returns:
            One-dimensional boolean array of 64 bits, or None when decoding fails.
        """
        if not Path(image_path).is_file():
            logger.warning("File does not exist: %s", image_path)
            return None

        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            logger.warning("Failed to read image at: %s", image_path)
            return None

        resized = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        dct = cv2.dct(np.float32(resized))
        dct_lowfreq = dct[: self.hash_size, : self.hash_size]
        median_val = np.median(dct_lowfreq)
        return (dct_lowfreq > median_val).flatten()

    @staticmethod
    def compute_sharpness(image_path: Path) -> float:
        """Calculates focus sharpness score via Laplacian operator variance.

        Args:
            image_path: Filesystem path pointing to the input image.

        Returns:
            Variance score where higher values denote sharper edge gradients.
        """
        if not Path(image_path).is_file():
            return 0.0

        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0.0
        return float(cv2.Laplacian(img, cv2.CV_64F).var())


class DeduplicationPipeline:
    """Orchestrates duplicate detection, sharpness filtering, and dataset pruning.

    Attributes:
        config: DeduplicationConfig instance supplying paths and thresholds.
        hasher: PerceptualHasher instance executing DCT hash transforms.
    """

    def __init__(self, config: Optional[DeduplicationConfig] = None) -> None:
        """Initializes the deduplication pipeline with configuration parameters.

        Args:
            config: Optional configuration instance; defaults resolve to project paths.
        """
        self.config = config or DeduplicationConfig()
        self.hasher = PerceptualHasher(
            hash_size=self.config.hash_size,
            highfreq_factor=self.config.highfreq_factor,
        )

    @staticmethod
    def compute_file_sha256(filepath: Path) -> str:
        """Calculates SHA-256 hexadecimal digest for target file.

        Args:
            filepath: Path pointing to input file on disk.

        Returns:
            Hexadecimal SHA-256 digest string.
        """
        hasher = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _find_exact_and_phash(
        self,
        df: pd.DataFrame,
    ) -> Tuple[Dict[str, str], List[Tuple[str, str]], Dict[str, np.ndarray], List[str]]:
        """Identifies exact SHA-256 duplicates and computes DCT perceptual hashes.

        Args:
            df: DataFrame loaded from the manifest.

        Returns:
            Tuple of (exact_pruned map, exact_duplicate_pairs, phash_dict, valid_filenames).
        """
        sha_to_file: Dict[str, str] = {}
        exact_duplicate_pairs: List[Tuple[str, str]] = []
        exact_pruned: Dict[str, str] = {}
        phash_dict: Dict[str, np.ndarray] = {}
        valid_filenames: List[str] = []

        for _, row in df.iterrows():
            fname = str(row["filename"])
            img_path = self.config.images_dir / fname
            if not img_path.exists():
                logger.warning("Image missing on disk: %s", img_path)
                continue

            sha = self.compute_file_sha256(img_path)
            if sha in sha_to_file:
                canonical = sha_to_file[sha]
                exact_duplicate_pairs.append((fname, canonical))
                exact_pruned[fname] = canonical
                logger.info("Exact duplicate identified: %s is identical to %s", fname, canonical)
            else:
                sha_to_file[sha] = fname

            h = self.hasher.compute_phash(img_path)
            if h is not None:
                phash_dict[fname] = h
                valid_filenames.append(fname)

        return exact_pruned, exact_duplicate_pairs, phash_dict, valid_filenames

    def _compute_perceptual_adjacency(
        self,
        phash_matrix: np.ndarray,
        valid_filenames: List[str],
        exact_pruned: Dict[str, str],
    ) -> Dict[str, Set[str]]:
        """Computes chunked pairwise Hamming distances to construct perceptual adjacency graph.

        Args:
            phash_matrix: Boolean array of shape (N, 64) containing perceptual hashes.
            valid_filenames: List of filenames aligned with rows of phash_matrix.
            exact_pruned: Dictionary of already identified exact duplicates to exclude.

        Returns:
            Adjacency dictionary mapping filenames to their near-duplicate neighbor filenames.
        """
        adjacency: Dict[str, Set[str]] = defaultdict(set)
        chunk_size = self.config.chunk_size
        n_images = len(valid_filenames)

        logger.info("Computing pairwise Hamming distance for %d images...", n_images)
        for i in range(0, n_images, chunk_size):
            chunk_a = phash_matrix[i : i + chunk_size]
            for j in range(i, n_images, chunk_size):
                chunk_b = phash_matrix[j : j + chunk_size]
                dists = np.bitwise_xor(chunk_a[:, None, :], chunk_b[None, :, :]).sum(axis=2)
                r_indices, c_indices = np.where(dists <= self.config.hamming_threshold)
                for r, c in zip(r_indices, c_indices):
                    gi, gj = i + r, j + c
                    if gi < gj:
                        f1, f2 = valid_filenames[gi], valid_filenames[gj]
                        if f1 in exact_pruned or f2 in exact_pruned:
                            continue
                        adjacency[f1].add(f2)
                        adjacency[f2].add(f1)
        return adjacency

    @staticmethod
    def _find_connected_components(adjacency: Dict[str, Set[str]]) -> List[List[str]]:
        """Discovers connected components within the adjacency graph via BFS traversal.

        Args:
            adjacency: Mapping from node filenames to adjacent neighbor sets.

        Returns:
            List of connected component clusters (each cluster is a list of filenames).
        """
        visited: Set[str] = set()
        clusters: List[List[str]] = []
        for node in adjacency:
            if node not in visited:
                cluster: List[str] = []
                queue = [node]
                visited.add(node)
                for curr in queue:
                    cluster.append(curr)
                    for neighbor in adjacency[curr]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                clusters.append(cluster)
        return clusters

    def _resolve_canonical_representatives(
        self,
        clusters: List[List[str]],
    ) -> Tuple[Dict[str, str], Set[str]]:
        """Ranks clusters by Laplacian sharpness variance and selects canonical representatives.

        Args:
            clusters: List of connected component filename clusters.

        Returns:
            Tuple of (near_duplicate_pruned map, canonical_representatives set).
        """
        near_dup_pruned: Dict[str, str] = {}
        canonical_representatives: Set[str] = set()

        for cluster in clusters:
            scored = [
                (fname, self.hasher.compute_sharpness(self.config.images_dir / fname))
                for fname in cluster
            ]
            scored.sort(key=lambda x: x[1], reverse=True)
            canonical_img, best_sharpness = scored[0]
            canonical_representatives.add(canonical_img)

            for pruned_img, sharpness in scored[1:]:
                near_dup_pruned[pruned_img] = canonical_img
                logger.debug(
                    "Pruning near-duplicate: %s (var=%.1f) -> Retaining %s (var=%.1f)",
                    pruned_img, sharpness, canonical_img, best_sharpness,
                )

        return near_dup_pruned, canonical_representatives

    def run(self) -> DuplicateResult:
        """Executes exact binary duplicate scan and near-duplicate pHash clustering.

        Pairwise Hamming distances construct adjacency components across perceptual hashes.
        Connected components determine candidate clusters from which the frame with
        maximal Laplacian variance remains selected, while redundant burst frames receive
        pruning flags.

        Returns:
            DuplicateResult containing pruned filenames and statistical breakdown.

        Raises:
            FileNotFoundError: Raised when dataset manifest is missing on disk.
        """
        logger.info("Starting dataset deduplication audit...")
        if not self.config.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.config.manifest_path}")

        df = pd.read_csv(self.config.manifest_path)
        initial_count = len(df)
        logger.info("Auditing %d total registered images in manifest.", initial_count)

        exact_pruned, exact_duplicate_pairs, phash_dict, valid_filenames = self._find_exact_and_phash(df)
        logger.info("Found %d exact binary duplicate files.", len(exact_pruned))

        phash_matrix = np.array([phash_dict[f] for f in valid_filenames], dtype=bool)
        adjacency = self._compute_perceptual_adjacency(phash_matrix, valid_filenames, exact_pruned)
        clusters = self._find_connected_components(adjacency)
        logger.info("Formed %d perceptual burst-shot clusters.", len(clusters))

        near_dup_pruned, canonical_representatives = self._resolve_canonical_representatives(clusters)

        total_pruned = {**exact_pruned, **near_dup_pruned}
        final_count = initial_count - len(total_pruned)
        logger.info(
            "Deduplication complete: %d initial, %d exact pruned, %d near-duplicates pruned, %d remaining.",
            initial_count, len(exact_pruned), len(near_dup_pruned), final_count,
        )

        return DuplicateResult(
            initial_count=initial_count,
            exact_duplicate_count=len(exact_pruned),
            near_duplicate_cluster_count=len(clusters),
            near_duplicate_pruned_count=len(near_dup_pruned),
            final_count=final_count,
            exact_duplicates=exact_duplicate_pairs,
            pruned_files=total_pruned,
            canonical_files=canonical_representatives,
        )

    def apply_pruning(self, result: DuplicateResult) -> None:
        """Updates dataset manifest and transfers pruned images to quarantine storage.

        Args:
            result: DuplicateResult containing identified redundant records.
        """
        logger.info("Applying pruning to dataset directories...")
        pruned_set = set(result.pruned_files.keys())

        df = pd.read_csv(self.config.manifest_path)
        df["is_pruned"] = df["filename"].isin(pruned_set)
        df["canonical_reference"] = df["filename"].map(result.pruned_files).fillna(df["filename"])
        df["prune_reason"] = df["filename"].apply(
            lambda f: "exact_duplicate" if any(f == d[0] for d in result.exact_duplicates)
            else ("near_duplicate_burst" if f in pruned_set else "retained")
        )
        df.to_csv(self.config.manifest_path, index=False)
        logger.info("Updated manifest with pruning flags at: %s", self.config.manifest_path)

        archive_dir = self.config.pruned_archive_dir
        (archive_dir / "images").mkdir(parents=True, exist_ok=True)
        (archive_dir / "labels").mkdir(parents=True, exist_ok=True)

        for split in ["train", "val", "test"]:
            img_split_dir = self.config.split_images_dir / split
            lbl_split_dir = self.config.split_labels_dir / split
            if not img_split_dir.exists():
                continue

            for fname in pruned_set:
                src_img = img_split_dir / fname
                src_lbl = lbl_split_dir / fname.replace(".jpg", ".txt").replace(".png", ".txt")

                if src_img.exists():
                    dest_img = archive_dir / "images" / fname
                    src_img.rename(dest_img)
                if src_lbl.exists():
                    dest_lbl = archive_dir / "labels" / src_lbl.name
                    src_lbl.rename(dest_lbl)

        logger.info("Pruned files archived safely to: %s", archive_dir)


if __name__ == "__main__":
    pipeline = DeduplicationPipeline()
    res = pipeline.run()
    pipeline.apply_pruning(res)
    logger.info("Finished successfully. Clean active dataset count: %d", res.final_count)
