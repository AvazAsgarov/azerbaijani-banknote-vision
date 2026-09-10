"""Data Processing, Deduplication, Feature Extraction, and Split Routing Package.

Pipelines for exact/perceptual deduplication, high-dimensional DINOv2 representation
inference, agglomerative scene community clustering, multi-objective group
stratification, and cross-split empirical leakage validation are exposed.
"""

from .clustering import ClusterOutput, MetaSceneClusterer
from .deduplicate import DeduplicationConfig, DeduplicationPipeline, DuplicateResult
from .embeddings import BanknoteDataset, ExtractionConfig, FeatureExtractor
from .splitter import SplitPartitionResult, ZeroLeakageSplitter
from .validator import SplitValidator, ValidationReport

__all__ = [
    "BanknoteDataset",
    "ClusterOutput",
    "MetaSceneClusterer",
    "DeduplicationConfig",
    "DeduplicationPipeline",
    "DuplicateResult",
    "ExtractionConfig",
    "FeatureExtractor",
    "ZeroLeakageSplitter",
    "SplitPartitionResult",
    "SplitValidator",
    "ValidationReport",
]
