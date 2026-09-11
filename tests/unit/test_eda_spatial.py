"""
Unit tests for spatial geometry and bounding annotation metrics.
Verifies aspect ratio calculations, scale distributions, and spatial dataclasses.
"""

import pytest
from src.eda.spatial import SpatialMetrics, SpatialAnalyzer
from src.core.config import ProjectPaths


@pytest.mark.unit
class TestSpatialMetricsDataModel:
    """Test suite verifying SpatialMetrics dataclass structure and field integrity."""

    def test_spatial_metrics_instantiation(self):
        """Verify SpatialMetrics holds valid statistical boundaries and anchor priors.

        Args:
            None.

        Returns:
            None.
        """
        metrics = SpatialMetrics(
            total_annotations=3500,
            mean_aspect_ratio=1.75,
            median_aspect_ratio=1.70,
            aspect_ratio_p05=1.35,
            aspect_ratio_p95=2.45,
            pct_small_objects=2.5,
            pct_medium_objects=15.0,
            pct_large_objects=82.5,
            pct_single_instance=95.0,
            pct_multi_instance=5.0,
            max_instances_per_image=3,
            optimal_anchors_k6=[(0.45, 0.25), (0.60, 0.35)],
            mean_iou_k6=0.82,
            recommended_input_resolution=160,
            recommended_min_stride=16
        )

        assert metrics.total_annotations == 3500
        assert metrics.mean_aspect_ratio > 1.0
        assert metrics.pct_large_objects > metrics.pct_small_objects
        assert len(metrics.optimal_anchors_k6) == 2
        assert metrics.recommended_input_resolution == 160

    def test_spatial_analyzer_initialization_defaults(self):
        """Verify SpatialAnalyzer initializes with standardized project paths.

        Args:
            None.

        Returns:
            None.
        """
        paths = ProjectPaths()
        analyzer = SpatialAnalyzer(paths=paths)

        assert analyzer.paths == paths
