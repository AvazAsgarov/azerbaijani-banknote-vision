"""
Assistive Safety Guard and Temporal Confirmation Engine for Smart Glasses.
Enforces geometric bounding box plausibility, denomination dynamic thresholds,
and sliding-window majority voting to eliminate high-value false positives.
"""

from collections import deque
from typing import Any, Dict, List, Sequence, Tuple

DENOMINATIONS = [
    "001_azn", "005_azn", "010_azn", "020_azn", "050_azn", "100_azn", "200_azn"
]

DYNAMIC_THRESHOLDS = {
    "001_azn": 0.50,
    "005_azn": 0.55,
    "010_azn": 0.60,
    "020_azn": 0.65,
    "050_azn": 0.70,
    "100_azn": 0.75,
    "200_azn": 0.80
}

class AssistiveSafetyGuard:
    """Multi-Tier Safety Guardrail evaluating candidate detections before audio dispatch.

    Tiers:
        1. Geometric Plausibility (Aspect ratio and minimum frame area)
        2. Denomination Dynamic Confidence Gate
        3. Temporal Sliding-Window Majority Voting (3 of 5 frames)
        4. BLE Dispatch / Guidance Token Generation
    """

    def __init__(
        self,
        window_size: int = 5,
        required_matches: int = 3,
        iou_threshold: float = 0.45,
    ) -> None:
        """Initialize assistive safety guardrail.

        Args:
            window_size: Length of the temporal history deque.
            required_matches: Number of matching class detections needed to confirm.
            iou_threshold: Intersection-over-Union threshold for spatial matching.
        """
        self.window_size = window_size
        self.required_matches = required_matches
        self.iou_threshold = iou_threshold
        self.history: deque = deque(maxlen=window_size)
        self.confirmed_count = 0
        self.rejected_count = 0

    def validate_geometry(self, box: Sequence[float]) -> bool:
        """Tier 1: Enforce physical banknote aspect ratio constraints.

        Args:
            box: Sequence of [xc, yc, w, h] normalized to [0.0, 1.0].

        Returns:
            True if bounding box satisfies size and aspect ratio criteria.
        """
        _xc, _yc, w, h = box
        if w <= 0 or h <= 0:
            return False
        ratio = w / h
        area = w * h

        # Reject micro-boxes occupying under 8% of the optical canvas
        if area < 0.08:
            return False

        # Physical banknote aspect ratios (landscape: 1.4-2.5, portrait: 0.4-0.7)
        valid_landscape = (1.35 <= ratio <= 2.55)
        valid_portrait = (0.38 <= ratio <= 0.72)
        return (valid_landscape or valid_portrait)

    def validate_confidence(self, class_name: str, confidence: float) -> bool:
        """Tier 2: Apply denomination-specific dynamic confidence gate.

        Args:
            class_name: Target banknote class identifier.
            confidence: Predicted classification confidence probability.

        Returns:
            True if confidence meets or exceeds class-specific threshold.
        """
        min_threshold = DYNAMIC_THRESHOLDS.get(class_name, 0.70)
        return confidence >= min_threshold

    def process_frame_detections(
        self, detections: List[Dict[str, Any]]
    ) -> Tuple[str, Dict[str, Any]]:
        """Process list of candidate detections from current inference frame.

        Args:
            detections: List of candidate detection dicts, each with keys
                'box' ([xc, yc, w, h]), 'confidence' (float), and 'class' (str).

        Returns:
            Tuple containing:
                status_code: 'CONFIRMED' | 'AMBIGUOUS' | 'NO_DETECTION'
                details: Dictionary with action type and confirmed denomination or guidance message.
        """
        valid_candidates = []
        for det in detections:
            box = det["box"]
            cls_name = det["class"]
            conf = det["confidence"]

            if not self.validate_geometry(box):
                continue
            if not self.validate_confidence(cls_name, conf):
                continue

            valid_candidates.append(det)

        if not valid_candidates:
            self.history.append(None)
            if any(self.history):
                return "AMBIGUOUS", {"action": "GUIDANCE", "message": "Banknote detected. Please hold steady under lighting."}
            return "NO_DETECTION", {"action": "SCANNING", "message": "Scanning for banknote..."}

        # Select highest confidence valid detection for this frame
        best_candidate = max(valid_candidates, key=lambda d: d["confidence"])
        self.history.append(best_candidate)

        # Tier 3: Temporal Majority Confirmation
        class_counts: Dict[str, int] = {}
        for past_det in self.history:
            if past_det is not None:
                c = past_det["class"]
                class_counts[c] = class_counts.get(c, 0) + 1

        for c, count in class_counts.items():
            if count >= self.required_matches:
                self.confirmed_count += 1
                return "CONFIRMED", {
                    "action": "AUDIO_DISPATCH",
                    "denomination": c,
                    "confidence": best_candidate["confidence"],
                    "box": best_candidate["box"]
                }

        self.rejected_count += 1
        return "AMBIGUOUS", {"action": "GUIDANCE", "message": "Verifying denomination. Maintain position."}
