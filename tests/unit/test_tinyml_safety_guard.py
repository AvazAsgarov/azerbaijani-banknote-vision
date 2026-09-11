"""
Unit tests for 4-tier assistive safety guardrails.
Verifies geometric plausibility filtering, dynamic confidence gating, and temporal smoothing.
"""

import pytest

from src.tinyml.safety_guard import AssistiveSafetyGuard, DYNAMIC_THRESHOLDS


@pytest.mark.unit
class TestAssistiveSafetyGuard:
    """Test suite verifying multi-tier verification guardrails for smart glasses."""

    def test_dynamic_threshold_table_values(self):
        """Verify dynamic threshold table enforces high thresholds on high-value currency.

        Args:
            None.

        Returns:
            None.
        """
        assert DYNAMIC_THRESHOLDS["001_azn"] == 0.50
        assert DYNAMIC_THRESHOLDS["100_azn"] == 0.75
        assert DYNAMIC_THRESHOLDS["200_azn"] == 0.80

    def test_tier1_geometry_rejects_micro_boxes(self):
        """Verify Tier 1 filter rejects candidate boxes occupying less than 8 percent optical area.

        Args:
            None.

        Returns:
            None.
        """
        guard = AssistiveSafetyGuard()
        micro_box = [0.5, 0.5, 0.1, 0.1]

        passed = guard.validate_geometry(micro_box)
        assert not passed

    def test_tier1_geometry_accepts_valid_banknote_ratios(self):
        """Verify Tier 1 filter approves standard banknote aspect ratios.

        Args:
            None.

        Returns:
            None.
        """
        guard = AssistiveSafetyGuard()
        valid_box = [0.5, 0.5, 0.60, 0.35]

        passed = guard.validate_geometry(valid_box)
        assert passed

    def test_tier2_confidence_gate_differentiates_denominations(self):
        """Verify Tier 2 filter enforces escalating confidence requirements by denomination.

        Args:
            None.

        Returns:
            None.
        """
        guard = AssistiveSafetyGuard()

        passed_1 = guard.validate_confidence("001_azn", 0.55)
        rejected_100 = guard.validate_confidence("100_azn", 0.65)
        passed_100 = guard.validate_confidence("100_azn", 0.80)

        assert passed_1
        assert not rejected_100
        assert passed_100

    def test_tier3_temporal_smoothing_requires_majority_votes(self):
        """Verify Tier 3 filter requires 3 of 5 historical frame matches before confirmation.

        Args:
            None.

        Returns:
            None.
        """
        guard = AssistiveSafetyGuard(window_size=5, required_matches=3)

        det = {"box": [0.5, 0.5, 0.60, 0.35], "confidence": 0.90, "class": "010_azn"}

        status1, _ = guard.process_frame_detections([det])
        assert status1 == "AMBIGUOUS"

        status2, _ = guard.process_frame_detections([det])
        assert status2 == "AMBIGUOUS"

        status3, details = guard.process_frame_detections([det])
        assert status3 == "CONFIRMED"
        assert details["denomination"] == "010_azn"

    def test_tier4_guidance_generation_on_rejection(self):
        """Verify Tier 4 emits descriptive assistive audio guidance when frames lack confirmation.

        Args:
            None.

        Returns:
            None.
        """
        guard = AssistiveSafetyGuard()
        status, details = guard.process_frame_detections([])

        assert status == "NO_DETECTION"
        assert details["action"] == "SCANNING"
