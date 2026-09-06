"""Unit Tests for Semantic Meta-Scene Clustering Engine.

Validates folder centroid aggregation, cosine distance matrix calculation,
agglomerative hierarchical clustering, and meta-cluster assignment to manifests.
"""

from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import pytest

from src.core.config import ClusterConfig, ProjectPaths
from src.data.clustering import ClusterOutput, MetaSceneClusterer


@pytest.mark.unit
class TestMetaSceneClusterer:
    """Tests hierarchical community grouping across feature representations."""

    def test_missing_embeddings_raises_error(self, tmp_path: Path) -> None:
        """Verifies FileNotFoundError when HDF5 container fails to exist."""
        paths = ProjectPaths(root_dir=tmp_path)
        clusterer = MetaSceneClusterer(paths)
        with pytest.raises(FileNotFoundError):
            clusterer.execute()

    def test_execute_with_mock_environment(self, populated_mock_env: ProjectPaths) -> None:
        """Verifies end-to-end execution of clustering on populated mock setup."""
        # Ensure capture_folders is present in H5
        with h5py.File(populated_mock_env.embeddings_h5_path, "a") as h5f:
            df = pd.read_csv(populated_mock_env.manifest_path)
            if "capture_folders" not in h5f:
                h5f.create_dataset(
                    "capture_folders",
                    data=[f.encode("utf-8") for f in df["original_capture_folder"]],
                )
            if "class_names" not in h5f:
                h5f.create_dataset(
                    "class_names",
                    data=[c.encode("utf-8") for c in df["class_name"]],
                )

        clusterer = MetaSceneClusterer(populated_mock_env)
        out = clusterer.execute()

        assert isinstance(out, ClusterOutput)
        assert out.num_meta_scenes >= 1
        assert len(out.folder_to_meta) == 3
        assert "meta_cluster" in out.df_manifest.columns
        assert not out.df_manifest["meta_cluster"].isna().any()

    def test_identical_folder_embeddings_merge(self, tmp_path: Path) -> None:
        """Verifies identical folder representations merge into a unified cluster."""
        root = tmp_path / "workspace"
        paths = ProjectPaths(root_dir=root)
        paths.embeddings_dir.mkdir(parents=True)
        paths.processed_data_dir.mkdir(parents=True)

        # Create two folders with identical unit vectors
        vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        embeddings = np.array([vec, vec], dtype=np.float32)
        fnames = ["f1.jpg", "f2.jpg"]
        cnames = ["001_azn", "001_azn"]
        folders = ["series_a", "series_b"]

        with h5py.File(paths.embeddings_h5_path, "w") as h5f:
            h5f.create_dataset("embeddings", data=embeddings)
            h5f.create_dataset("filenames", data=[f.encode("utf-8") for f in fnames])
            h5f.create_dataset("class_names", data=[c.encode("utf-8") for c in cnames])
            h5f.create_dataset("capture_folders", data=[f.encode("utf-8") for f in folders])

        df = pd.DataFrame({
            "filename": fnames,
            "class_name": cnames,
            "original_capture_folder": folders,
            "is_pruned": [False, False],
        })
        df.to_csv(paths.manifest_path, index=False)

        cfg = ClusterConfig(distance_threshold=0.20)
        clusterer = MetaSceneClusterer(paths, cfg)
        out = clusterer.execute()

        # Both folders should merge into the same meta-scene cluster
        assert out.folder_to_meta["series_a"] == out.folder_to_meta["series_b"]
        assert out.num_meta_scenes == 1

    def test_orthogonal_folder_embeddings_separate(self, tmp_path: Path) -> None:
        """Verifies orthogonal folder representations remain in separate clusters."""
        root = tmp_path / "workspace"
        paths = ProjectPaths(root_dir=root)
        paths.embeddings_dir.mkdir(parents=True)
        paths.processed_data_dir.mkdir(parents=True)

        # Two orthogonal unit vectors (cosine distance = 1.0)
        v1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        embeddings = np.array([v1, v2], dtype=np.float32)
        fnames = ["f1.jpg", "f2.jpg"]
        cnames = ["001_azn", "005_azn"]
        folders = ["series_a", "series_b"]

        with h5py.File(paths.embeddings_h5_path, "w") as h5f:
            h5f.create_dataset("embeddings", data=embeddings)
            h5f.create_dataset("filenames", data=[f.encode("utf-8") for f in fnames])
            h5f.create_dataset("class_names", data=[c.encode("utf-8") for c in cnames])
            h5f.create_dataset("capture_folders", data=[f.encode("utf-8") for f in folders])

        df = pd.DataFrame({
            "filename": fnames,
            "class_name": cnames,
            "original_capture_folder": folders,
            "is_pruned": [False, False],
        })
        df.to_csv(paths.manifest_path, index=False)

        # Distance threshold 0.20 is below orthogonal distance 1.0
        cfg = ClusterConfig(distance_threshold=0.20)
        clusterer = MetaSceneClusterer(paths, cfg)
        out = clusterer.execute()

        assert out.folder_to_meta["series_a"] != out.folder_to_meta["series_b"]
        assert out.num_meta_scenes == 2
