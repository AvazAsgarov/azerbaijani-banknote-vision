"""
Unit tests for photometric analysis, channel statistics, and focus metrics.
Verifies luminance bounds, blur classifications, and photometric dataclass integrity.
"""

import pytest
from src.eda.photometric import PhotometricMetrics, PhotometricAnalyzer
from src.core.config import ProjectPaths


@pytest.mark.unit
class TestPhotometricMetricsDataModel:
    """Test suite verifying PhotometricMetrics dataclass and field types."""

    def test_photometric_metrics_instantiation(self):
        """Verify PhotometricMetrics instantiates with expected physical boundaries and variance stats.

        Args:
            None.

        Returns:
            None.
        """
        metrics = PhotometricMetrics(
            mean_luminance=128.5,
            std_luminance=45.2,
            luminance_p05=40.0,
            luminance_p95=210.0,
            mean_rgb=(130.0, 125.0, 120.0),
            std_rgb=(48.0, 44.0, 42.0),
            mean_rms_contrast=0.38,
            median_laplacian_var=145.0,
            pct_sharp_frames=78.5,
            pct_moderate_blur=18.0,
            pct_heavy_blur=3.5,
            recommended_brightness_delta=0.20,
            recommended_contrast_delta=0.25,
            recommended_saturation_delta=0.30,
            recommended_hue_delta=0.08,
            recommended_blur_kernel_max=7
        )

        assert metrics.mean_luminance > 0.0
        assert metrics.pct_sharp_frames > metrics.pct_heavy_blur
        assert metrics.recommended_blur_kernel_max == 7

    def test_photometric_analyzer_initialization_defaults(self):
        """Verify PhotometricAnalyzer initializes with standardized project paths.

        Args:
            None.

        Returns:
            None.
        """
        paths = ProjectPaths()
        analyzer = PhotometricAnalyzer(paths=paths)

        assert analyzer.paths == paths
