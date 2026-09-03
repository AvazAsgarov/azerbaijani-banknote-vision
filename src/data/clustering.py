"""Semantic Meta-Scene Community Clustering Engine.

Agglomerative hierarchical clustering operates over normalized DINOv2 feature
representations to partition physical capture environments into disjoint semantic
communities. Folder-level mean centroids prevent visual leakage across burst shots
and near-identical backgrounds during downstream dataset splitting.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import h5py
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

from src.core.config import ClusterConfig, ProjectPaths

logger = logging.getLogger("MetaSceneClustering")


@dataclass
class ClusterOutput:
    """Encapsulation of clustering outputs and metadata mappings.

    Attributes:
        folder_to_meta: Mapping from original capture folder strings to cluster integers.
        df_manifest: Active dataset manifest populated with assigned cluster identifiers.
        num_meta_scenes: Total number of discrete meta-scene communities discovered.
        meta_cluster_sizes: Dictionary recording the sample count per cluster.
    """

    folder_to_meta: Dict[str, int]
    df_manifest: pd.DataFrame
    num_meta_scenes: int
    meta_cluster_sizes: Dict[int, int]


class MetaSceneClusterer:
    """Hierarchical agglomerative clustering engine for scene community grouping.

    Attributes:
        paths: Configuration paths resolving dataset files and artifact locations.
        cluster_cfg: Hyperparameters governing distance metrics and linkage criteria.
    """

    def __init__(
        self,
        paths: Optional[ProjectPaths] = None,
        cluster_cfg: Optional[ClusterConfig] = None,
    ) -> None:
        """Initializes the clustering engine with project paths and parameters.

        Args:
            paths: Project configuration paths containing the embedding HDF5 archive.
            cluster_cfg: Clustering configuration specifying distance cutoff and linkage.
        """
        self.paths = paths or ProjectPaths()
        self.cluster_cfg = cluster_cfg or ClusterConfig()

    def _compute_folder_centroids(
        self,
        embeddings: np.ndarray,
        folders: List[str],
        unique_folders: List[str],
    ) -> np.ndarray:
        """Computes L2-normalized mean feature centroid per unique capture folder.

        Args:
            embeddings: Dense feature matrix of shape (N, D).
            folders: List of capture folder names corresponding to each sample.
            unique_folders: Sorted list of distinct capture folder names.

        Returns:
            Normalized centroid matrix of shape (num_folders, D).
        """
        folder_centroids: List[np.ndarray] = []
        for fld in unique_folders:
            folder_indices = [idx for idx, f in enumerate(folders) if f == fld]
            centroid = np.mean(embeddings[folder_indices], axis=0)
            norm = np.linalg.norm(centroid)
            normalized_centroid = centroid / norm if norm > 0 else centroid
            folder_centroids.append(normalized_centroid)
        return np.array(folder_centroids)

    @staticmethod
    def _compute_cosine_distance_matrix(centroid_matrix: np.ndarray) -> np.ndarray:
        """Computes pairwise cosine distance matrix from normalized centroid vectors.

        Distance is defined as D = 1.0 - (u . v), clipped to [0.0, 2.0].

        Args:
            centroid_matrix: Normalized centroid array of shape (K, D).

        Returns:
            Symmetric distance matrix of shape (K, K).
        """
        distance_matrix = 1.0 - (centroid_matrix @ centroid_matrix.T)
        np.clip(distance_matrix, 0.0, 2.0, out=distance_matrix)
        return distance_matrix

    def execute(self) -> ClusterOutput:
        """Executes folder-level mean embedding aggregation and agglomerative clustering.

        Folder centroids are computed by calculating mean normalized DINOv2 feature
        vectors per capture folder. Pairwise cosine distances then guide agglomerative
        clustering until the distance threshold is reached.

        Returns:
            ClusterOutput containing folder-to-cluster mappings and active dataframe.

        Raises:
            FileNotFoundError: Raised when the feature HDF5 archive does not exist on disk.
        """
        with h5py.File(self.paths.embeddings_h5_path, "r") as h5f:
            embeddings: np.ndarray = h5f["embeddings"][:]
            folders: List[str] = [
                x.decode("utf-8") if isinstance(x, bytes) else str(x)
                for x in h5f["capture_folders"][:]
            ]

        df = pd.read_csv(self.paths.manifest_path)
        if "is_pruned" in df.columns:
            active_mask = ~df["is_pruned"].astype(bool)
        else:
            active_mask = np.ones(len(df), dtype=bool)
        df_active = df[active_mask].copy().reset_index(drop=True)

        unique_folders = sorted(list(set(folders)))
        logger.info("%d unique capture folders are detected across active records.", len(unique_folders))

        centroid_matrix = self._compute_folder_centroids(embeddings, folders, unique_folders)
        distance_matrix = self._compute_cosine_distance_matrix(centroid_matrix)

        clustering_model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=self.cluster_cfg.distance_threshold,
            metric=self.cluster_cfg.metric,
            linkage=self.cluster_cfg.linkage,
        )
        cluster_labels = clustering_model.fit_predict(distance_matrix)

        folder_to_meta: Dict[str, int] = {
            fld: int(cluster_labels[idx])
            for idx, fld in enumerate(unique_folders)
        }

        df_active["meta_cluster"] = df_active["original_capture_folder"].map(folder_to_meta)
        num_meta = len(set(cluster_labels))
        cluster_counts = df_active["meta_cluster"].value_counts().to_dict()

        logger.info("%d atomic meta-scene communities are discovered.", num_meta)
        return ClusterOutput(
            folder_to_meta=folder_to_meta,
            df_manifest=df_active,
            num_meta_scenes=num_meta,
            meta_cluster_sizes=cluster_counts,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    clusterer = MetaSceneClusterer()
    result = clusterer.execute()
    print(f"Clustering complete. {result.num_meta_scenes} communities are formed.")
