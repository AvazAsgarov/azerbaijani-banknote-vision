"""
Unit tests for manifold latent representation metrics and clustering indices.
Verifies silhouette coefficients, Davies-Bouldin separation, and outlier detection models.
"""

import pytest
from src.eda.manifold import ManifoldMetrics, ManifoldAnalyzer
from src.core.config import ProjectPaths


@pytest.mark.unit
class TestManifoldMetricsDataModel:
    """Test suite verifying ManifoldMetrics dataclass fields and cluster indices."""

    def test_manifold_metrics_instantiation(self):
        """Verify ManifoldMetrics stores valid silhouette and separation statistics.

        Args:
            None.

        Returns:
            None.
        """
        metrics = ManifoldMetrics(
            silhouette_cosine=0.68,
            davies_bouldin=1.12,
            calinski_harabasz=450.0,
            mean_intra_class_sim=0.85,
            mean_inter_class_sim=0.32,
            class_similarity_matrix={"001_azn": {"001_azn": 1.0, "005_azn": 0.35}},
            top_outlier_filenames=["sample_01.jpg", "sample_02.jpg"],
            top_outlier_scores=[-1.85, -1.72],
            top_outlier_classes=["001_azn", "050_azn"]
        )

        assert metrics.silhouette_cosine > 0.50
        assert metrics.davies_bouldin > 0.0
        assert metrics.mean_intra_class_sim > metrics.mean_inter_class_sim
        assert len(metrics.top_outlier_filenames) == 2

    def test_manifold_analyzer_initialization_defaults(self):
        """Verify ManifoldAnalyzer initializes with standardized project paths.

        Args:
            None.

        Returns:
            None.
        """
        paths = ProjectPaths()
        analyzer = ManifoldAnalyzer(paths=paths)

        assert analyzer.paths == paths
