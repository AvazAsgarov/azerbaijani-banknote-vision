"""
Invariant contract tests for TinyML edge deployment on ESP32-S3 Sense hardware.
Verifies non-negotiable memory ceilings, parameter ceilings, and safety invariants.
"""

import pytest
import numpy as np

from src.tinyml.model import YOLOFastestV2
from src.tinyml.dataset import ANCHORS_S16, ANCHORS_S32, DENOMINATIONS
from src.tinyml.safety_guard import DYNAMIC_THRESHOLDS


@pytest.mark.invariant
class TestTinyMLEdgeHardwareInvariants:
    """Test suite verifying architectural and hardware memory invariants."""

    def test_parameter_count_below_hardware_ceiling(self):
        """Verify parameter count remains strictly below 280000 parameters.

        Args:
            None.

        Returns:
            None.
        """
        model = YOLOFastestV2(num_classes=len(DENOMINATIONS))
        param_count = model.count_parameters()

        assert param_count <= 280000
        assert param_count >= 150000

    def test_anchor_dimensions_strictly_positive(self):
        """Verify anchor width and height values remain strictly positive non-zero floats.

        Args:
            None.

        Returns:
            None.
        """
        for a in ANCHORS_S16:
            assert a[0] > 0.0
            assert a[1] > 0.0

        for a in ANCHORS_S32:
            assert a[0] > 0.0
            assert a[1] > 0.0

    def test_high_value_currency_safety_threshold_invariant(self):
        """Verify high-value denomination thresholds remain strictly at or above 0.75.

        Args:
            None.

        Returns:
            None.
        """
        assert DYNAMIC_THRESHOLDS["100_azn"] >= 0.75
        assert DYNAMIC_THRESHOLDS["200_azn"] >= 0.80

    def test_canonical_denomination_count_invariant(self):
        """Verify denomination list contains exactly seven canonical currency classes.

        Args:
            None.

        Returns:
            None.
        """
        assert len(DENOMINATIONS) == 7
        assert "001_azn" in DENOMINATIONS
        assert "200_azn" in DENOMINATIONS
