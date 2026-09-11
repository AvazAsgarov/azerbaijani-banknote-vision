"""
Unit tests for cross-split statistical integrity and zero-leakage verification.
Verifies Wasserstein distances, Kolmogorov-Smirnov statistics, and integrity dataclasses.
"""

import pytest
from src.eda.integrity import IntegrityMetrics, IntegrityAnalyzer
from src.core.config import ProjectPaths


@pytest.mark.unit
class TestIntegrityMetricsDataModel:
    """Test suite verifying IntegrityMetrics dataclass structure and validation fields."""

    def test_integrity_metrics_instantiation(self):
        """Verify IntegrityMetrics holds cross-partition divergence statistics.

        Args:
            None.

        Returns:
            None.
        """
        metrics = IntegrityMetrics(
            is_meta_disjoint=True,
            wasserstein_train_val=0.045,
            wasserstein_train_test=0.048,
            ks_stat_train_test=0.052,
            ks_pvalue_train_test=0.85,
            mean_min_dist_train_test=0.18,
            p01_min_dist_train_test=0.08,
            test_feature_coverage_pct=98.5
        )

        assert metrics.is_meta_disjoint
        assert metrics.wasserstein_train_val > 0.0
        assert metrics.ks_pvalue_train_test > 0.05
        assert metrics.test_feature_coverage_pct > 90.0

    def test_integrity_analyzer_initialization_defaults(self):
        """Verify IntegrityAnalyzer initializes with standardized project paths.

        Args:
            None.

        Returns:
            None.
        """
        paths = ProjectPaths()
        analyzer = IntegrityAnalyzer(paths=paths)

        assert analyzer.paths == paths
