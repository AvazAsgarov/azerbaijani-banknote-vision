"""
Integration tests for TinyML detection and assistive safety guardrail pipeline.
Verifies end-to-end dataflow from model inference through multi-tier safety confirmation.
"""

import pytest
import torch

from src.tinyml.model import YOLOFastestV2
from src.tinyml.train import decode_predictions, non_max_suppression
from src.tinyml.safety_guard import AssistiveSafetyGuard, DENOMINATIONS


@pytest.mark.integration
class TestTinyMLAssistivePipeline:
    """Test suite verifying end-to-end TinyML assistive pipeline integration."""

    def test_complete_inference_and_safety_pipeline_flow(self):
        """Verify complete inference and safety filtering pipeline returns valid dispatch decisions.

        Args:
            None.

        Returns:
            None.
        """
        device = torch.device("cpu")
        model = YOLOFastestV2(num_classes=len(DENOMINATIONS)).to(device)
        model.eval()

        guard = AssistiveSafetyGuard(window_size=3, required_matches=2)
        dummy_frame = torch.randn(1, 3, 160, 160, device=device)

        with torch.no_grad():
            p16, p32 = model(dummy_frame)
            batch_preds = decode_predictions(p16, p32, conf_thresh=0.01)
            preds = non_max_suppression(batch_preds[0], iou_thresh=0.45)

        formatted_candidates = []
        for p in preds:
            formatted_candidates.append({
                "box": p[:4],
                "confidence": float(p[4]),
                "class": DENOMINATIONS[int(p[5])]
            })

        status, details = guard.process_frame_detections(formatted_candidates)

        assert status in ["CONFIRMED", "AMBIGUOUS", "NO_DETECTION"]
        assert "action" in details
        assert isinstance(details["message"], str) or "denomination" in details

    def test_safety_guardrail_suppresses_low_confidence_high_denominations(self):
        """Verify pipeline suppresses unconfirmed high-denomination candidate detections.

        Args:
            None.

        Returns:
            None.
        """
        guard = AssistiveSafetyGuard(window_size=3, required_matches=2)

        sketchy_candidate = [{
            "box": [0.5, 0.5, 0.60, 0.35],
            "confidence": 0.60,
            "class": "200_azn"
        }]

        status, details = guard.process_frame_detections(sketchy_candidate)

        assert status == "NO_DETECTION"
        assert details["action"] == "SCANNING"
