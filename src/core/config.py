"""Centralized Configuration Module for Banknote Detection Pipeline.

Directory structures, computational paths, hyperparameters for community clustering,
and multi-objective group stratification thresholds are defined within this module.
Every path resolves relative to the project root directory to ensure cross-platform
reproducibility across local workstations and remote GPU execution environments.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple


@dataclass(frozen=True)
class ProjectPaths:
    """Encapsulation of filesystem locations and generated artifact paths.

    Attributes:
        root_dir: Absolute path pointing to the project repository root.
        configs_dir: Directory containing YAML model and dataset declarations.
        data_yaml_path: Target YOLO / RT-DETR dataset specification file.
        data_dir: Base directory housing raw and processed image collections.
        raw_data_dir: Source archive containing initial captures.
        processed_data_dir: Standardized dataset directory after curation.
        images_dir: Target image directory partitioned into split subsets.
        labels_dir: Target YOLO annotation directory partitioned into split subsets.
        master_images_dir: Consolidated unique image pool prior to splitting.
        master_labels_dir: Consolidated annotation labels prior to splitting.
        pruned_archive_dir: Quarantine storage for removed duplicate frames.
        manifest_path: Comprehensive CSV catalog tracking image provenance.
        summary_json_path: Quantitative summary of dataset dimensions.
        artifacts_dir: Root directory housing pipeline execution outputs.
        embeddings_dir: Storage for extracted high-dimensional feature tensors.
        reports_dir: Destination for structured scientific analytical summaries.
        embeddings_h5_path: Primary HDF5 container for Vision Transformer features.
        embeddings_parquet_path: Parquet table pairing sample metadata with splits.
        embeddings_npy_path: Raw NumPy matrix array for dense tensor calculations.
        audit_report_path: Quantitative JSON summary of duplicate investigations.
        split_summary_path: Verification metrics for zero-leakage partitions.
        docs_dir: Documentation files and technical reports.
        figures_dir: Root visualization export directory.
        figures_eda_dir: Destination for distribution and geometry plots.
        figures_embeddings_dir: Destination for latent manifold and metric charts.
        figures_splits_dir: Destination for partition balance and integrity plots.
        notebooks_dir: Interactive Jupyter research environments.
    """

    root_dir: Path = Path(__file__).resolve().parent.parent.parent

    configs_dir: Path = None  # type: ignore
    data_yaml_path: Path = None  # type: ignore

    data_dir: Path = None  # type: ignore
    raw_data_dir: Path = None  # type: ignore
    processed_data_dir: Path = None  # type: ignore

    images_dir: Path = None  # type: ignore
    labels_dir: Path = None  # type: ignore
    master_images_dir: Path = None  # type: ignore
    master_labels_dir: Path = None  # type: ignore
    pruned_archive_dir: Path = None  # type: ignore
    manifest_path: Path = None  # type: ignore
    summary_json_path: Path = None  # type: ignore

    artifacts_dir: Path = None  # type: ignore
    experiments_dir: Path = None  # type: ignore
    embeddings_dir: Path = None  # type: ignore
    reports_dir: Path = None  # type: ignore
    embeddings_h5_path: Path = None  # type: ignore
    embeddings_parquet_path: Path = None  # type: ignore
    embeddings_npy_path: Path = None  # type: ignore
    audit_report_path: Path = None  # type: ignore
    split_summary_path: Path = None  # type: ignore

    docs_dir: Path = None  # type: ignore
    figures_dir: Path = None  # type: ignore
    figures_eda_dir: Path = None  # type: ignore
    figures_embeddings_dir: Path = None  # type: ignore
    figures_splits_dir: Path = None  # type: ignore
    figures_tinyml_dir: Path = None  # type: ignore
    notebooks_dir: Path = None  # type: ignore

    def __post_init__(self) -> None:
        """Derives default subpaths relative to root_dir for attributes set to None."""
        set_attr = object.__setattr__
        root = self.root_dir

        configs = self.configs_dir or (root / "configs")
        set_attr(self, "configs_dir", configs)
        set_attr(self, "data_yaml_path", self.data_yaml_path or (configs / "data.yaml"))

        data = self.data_dir or (root / "data")
        set_attr(self, "data_dir", data)
        set_attr(self, "raw_data_dir", self.raw_data_dir or (data / "raw"))

        processed = self.processed_data_dir or (data / "processed")
        set_attr(self, "processed_data_dir", processed)

        set_attr(self, "images_dir", self.images_dir or (processed / "images"))
        set_attr(self, "labels_dir", self.labels_dir or (processed / "labels"))
        set_attr(self, "master_images_dir", self.master_images_dir or (processed / "master" / "images"))
        set_attr(self, "master_labels_dir", self.master_labels_dir or (processed / "master" / "labels"))
        set_attr(self, "pruned_archive_dir", self.pruned_archive_dir or (processed / "pruned_archive"))
        set_attr(self, "manifest_path", self.manifest_path or (processed / "dataset_manifest.csv"))
        set_attr(self, "summary_json_path", self.summary_json_path or (processed / "dataset_summary.json"))

        artifacts = self.artifacts_dir or (root / "artifacts")
        set_attr(self, "artifacts_dir", artifacts)

        exp_dir = self.experiments_dir or (artifacts / "experiments")
        set_attr(self, "experiments_dir", exp_dir)

        emb_dir = self.embeddings_dir or (artifacts / "embeddings")
        set_attr(self, "embeddings_dir", emb_dir)

        rep_dir = self.reports_dir or (artifacts / "reports")
        set_attr(self, "reports_dir", rep_dir)

        set_attr(self, "embeddings_h5_path", self.embeddings_h5_path or (emb_dir / "embeddings_dinov2_vitl14.h5"))
        set_attr(self, "embeddings_parquet_path", self.embeddings_parquet_path or (emb_dir / "embeddings_metadata.parquet"))
        set_attr(self, "embeddings_npy_path", self.embeddings_npy_path or (emb_dir / "embeddings_dinov2_vitl14.npy"))
        set_attr(self, "audit_report_path", self.audit_report_path or (rep_dir / "dataset_audit_report.json"))
        set_attr(self, "split_summary_path", self.split_summary_path or (rep_dir / "split_summary.json"))

        set_attr(self, "docs_dir", self.docs_dir or (root / "docs"))

        fig_root = self.figures_dir or (root / "reports" / "figures")
        set_attr(self, "figures_dir", fig_root)
        set_attr(self, "figures_eda_dir", self.figures_eda_dir or (fig_root / "eda"))
        set_attr(self, "figures_embeddings_dir", self.figures_embeddings_dir or (fig_root / "embeddings"))
        set_attr(self, "figures_splits_dir", self.figures_splits_dir or (fig_root / "splits"))
        set_attr(self, "figures_tinyml_dir", self.figures_tinyml_dir or (fig_root / "tinyml"))
        set_attr(self, "notebooks_dir", self.notebooks_dir or (root / "notebooks"))


@dataclass(frozen=True)
class ClusterConfig:
    """Hyperparameters for DINOv2 meta-scene community clustering.

    Attributes:
        distance_threshold: Agglomerative clustering cutoff in cosine distance space.
        linkage: Criterion determining distance calculation across cluster pairs.
        metric: Metric designation passed directly to the clustering algorithm.
    """

    distance_threshold: float = 0.20
    linkage: str = "average"
    metric: str = "precomputed"


@dataclass(frozen=True)
class SplitConfig:
    """Target ratios and optimization weights for zero-leakage dataset partitioning.

    Attributes:
        train_ratio: Target proportion of samples assigned to the training split.
        val_ratio: Target proportion of samples assigned to the validation split.
        test_ratio: Target proportion of samples assigned to the test split.
        random_seed: Deterministic integer seed governing stochastic processes.
        optimization_iterations: Iteration budget for local hill-climbing search.
        classes: Ordered tuple of canonical Azerbaijani banknote denomination identifiers.
    """

    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 777
    optimization_iterations: int = 3000
    classes: Tuple[str, ...] = (
        "001_azn",
        "005_azn",
        "010_azn",
        "020_azn",
        "050_azn",
        "100_azn",
        "200_azn",
    )


@dataclass(frozen=True)
class DatasetQualityConfig:
    """Production quality thresholds and volume benchmarks for banknote vision dataset.

    Attributes:
        min_detection_images: Recommended minimum volume for robust vision detection models.
        min_classes: Total canonical denomination classes required.
    """

    min_detection_images: int = 2000
    min_classes: int = 7


@dataclass(frozen=True)
class MobileBridgeConfig:
    """Inference bridge server parameters for mobile companion application.

    Attributes:
        default_host: Network interface binding host address.
        default_port: HTTP server TCP listening port.
        default_conf_threshold: Bounding box detection confidence filter.
        simulation_default_class_id: Fallback class index for simulation testing.
        simulation_default_class_code: Fallback class name code.
        simulation_default_conf: Fallback detection confidence score.
        simulation_default_bbox: Fallback bounding box coordinates in normalized units.
    """

    default_host: str = "0.0.0.0"
    default_port: int = 8000
    default_conf_threshold: float = 0.50
    simulation_default_class_id: int = 2
    simulation_default_class_code: str = "010_azn"
    simulation_default_conf: float = 0.98
    simulation_default_bbox: Tuple[float, float, float, float] = (0.15, 0.25, 0.85, 0.75)


@dataclass(frozen=True)
class RemoteClusterConfig:
    """Connection parameters and authentication for remote A100 GPU compute cluster.

    All network credentials default to environment variables with fallback values
    to support production execution and local development.

    Attributes:
        base_url: Base HTTP endpoint for remote Jupyter/Triton server.
        token: Authentication token for Jupyter REST and WebSocket channels.
    """

    base_url: str = os.getenv("REMOTE_GPU_BASE_URL", "")
    token: str = os.getenv("REMOTE_GPU_TOKEN", "")

    @property
    def ws_url(self) -> str:
        """Derives WebSocket endpoint URL corresponding to base_url."""
        url_clean = self.base_url.replace("http://", "").replace("https://", "").rstrip("/")
        return f"ws://{url_clean}"

    def get_auth_headers(self) -> Dict[str, str]:
        """Constructs HTTP Authorization headers dictionary for Jupyter API."""
        return {"Authorization": f"token {self.token}"}

    def get_websocket_url(self, kernel_id: str) -> str:
        """Constructs full WebSocket URL for a specific active kernel channel."""
        return f"{self.ws_url}/api/kernels/{kernel_id}/channels?token={self.token}"


