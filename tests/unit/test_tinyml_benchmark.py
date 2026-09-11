"""
Unit tests for edge simulation and latency profiling utilities.
Verifies runtime latency calculations, frame rate calculations, and safety benchmark runs.
"""

import pytest
import torch

from src.tinyml.model import YOLOFastestV2
from src.tinyml.benchmark_edge import profile_latency


@pytest.mark.unit
class TestEdgeBenchmarkUtilities:
    """Test suite verifying latency profiling and frame rate calculations."""

    def test_profile_latency_returns_positive_metrics(self):
        """Verify profile_latency returns positive duration values and frame rate calculations.

        Args:
            None.

        Returns:
            None.
        """
        device = torch.device("cpu")
        model = YOLOFastestV2(num_classes=7).to(device)

        mean_lat, std_lat, fps = profile_latency(model, device, num_runs=5)

        assert mean_lat > 0.0
        assert std_lat >= 0.0
        assert fps > 0.0
        assert pytest.approx(fps * mean_lat, 1e-2) == 1000.0
