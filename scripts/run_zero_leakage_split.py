"""Zero-Leakage Dataset Partitioning and Validation Orchestrator.

Semantic community clustering is computed over DINOv2 feature embeddings,
followed by multi-objective knapsack partitioning and empirical verification.
"""

import logging
import sys
from pathlib import Path

# Project root is appended to system path for module resolution
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.core.config import ClusterConfig, ProjectPaths, SplitConfig
from src.data.clustering import MetaSceneClusterer
from src.data.splitter import ZeroLeakageSplitter
from src.data.validator import SplitValidator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("ZeroLeakageOrchestrator")


def run_pipeline() -> None:
    """
    Execution pipeline for zero-leakage meta-scene splitting and validation.
    """
    paths = ProjectPaths()
    cluster_cfg = ClusterConfig()
    split_cfg = SplitConfig()

    logger.info("Stage 1: Meta-Scene clustering via DINOv2 embeddings is initiated...")
    clusterer = MetaSceneClusterer(paths=paths, cluster_cfg=cluster_cfg)
    cluster_output = clusterer.execute()
    logger.info("%d atomic meta-scene communities are generated.", cluster_output.num_meta_scenes)

    logger.info("Stage 2: Multi-objective combinatorial knapsack routing is executed...")
    splitter = ZeroLeakageSplitter(paths=paths, split_cfg=split_cfg)
    partition_result = splitter.partition(cluster_output)
    logger.info(
        "Partitioning completed: Train=%d, Val=%d, Test=%d.",
        partition_result.train_count,
        partition_result.val_count,
        partition_result.test_count
    )

    logger.info("Stage 3: Empirical zero-leakage and disjointness validation is executed...")
    validator = SplitValidator(paths=paths)
    validation_report = validator.validate()

    logger.info("Pipeline execution summary:")
    logger.info("  - Cluster disjointness confirmed: %s", validation_report.is_disjoint)
    logger.info("  - Max Train-Test cosine similarity: %.4f", validation_report.max_train_test_similarity)
    logger.info("  - Max Train-Val cosine similarity: %.4f", validation_report.max_train_val_similarity)
    logger.info("  - Mean cross-split cosine similarity: %.4f", validation_report.mean_train_test_similarity)
    logger.info("  - Top-5 nearest neighbor contamination rate: %.4f", validation_report.top_k_contamination_rate)
    logger.info("  - Split distribution figure saved to: %s", paths.figures_splits_dir / "split_distribution.png")
    logger.info("  - Validation report saved to: %s", paths.split_summary_path)


if __name__ == "__main__":
    run_pipeline()
