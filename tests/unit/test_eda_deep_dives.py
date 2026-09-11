"""
Unit tests for domain-specific structural vulnerability and deep dive metrics.
Verifies confusion pair tracking, orientation classifications, and loss weight dictionaries.
"""

import pytest
from src.eda.deep_dives import DeepDiveMetrics, DeepDiveAnalyzer
from src.core.config import ProjectPaths


@pytest.mark.unit
class TestDeepDiveMetricsDataModel:
    """Test suite verifying DeepDiveMetrics dataclass attributes and orientation fields."""

    def test_deep_dive_metrics_instantiation(self):
        """Verify DeepDiveMetrics instantiates with valid orientation and loss weighting fields.

        Args:
            None.

        Returns:
            None.
        """
        metrics = DeepDiveMetrics(
            top_confusing_pairs=[("001_azn", "050_azn", 0.42)],
            pct_horizontal_banknotes=88.5,
            pct_vertical_banknotes=9.2,
            pct_square_like=2.3,
            recommended_focal_gamma=2.0,
            recommended_rotation_degrees=15.0,
            class_loss_weights={"001_azn": 1.0, "005_azn": 1.2}
        )

        assert metrics.pct_horizontal_banknotes > metrics.pct_vertical_banknotes
        assert metrics.recommended_focal_gamma == 2.0
        assert len(metrics.top_confusing_pairs) == 1
        assert "001_azn" in metrics.class_loss_weights

    def test_deep_dive_analyzer_initialization_defaults(self):
        """Verify DeepDiveAnalyzer initializes with standardized project paths.

        Args:
            None.

        Returns:
            None.
        """
        paths = ProjectPaths()
        analyzer = DeepDiveAnalyzer(paths=paths)

        assert analyzer.paths == paths
