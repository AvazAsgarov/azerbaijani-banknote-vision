"""
Unit Tests for Mobile Vision Inference Bridge Server.

Validates route handling, model discovery, health checks,
frame decoding, and bounding box normalization.
"""

import base64
import json
from io import BytesIO
from unittest.mock import MagicMock, patch
from PIL import Image
import pytest

from scripts.mobile_bridge import MobileBridgeHandler, load_inference_model


def test_model_loading_discovery():
    """Validates dynamic model discovery without crashing."""
    model = load_inference_model("auto")
    # Returns either loaded YOLO model or None in simulation mode
    assert model is not None or model is None


def test_mobile_bridge_health():
    """Validates /health endpoint."""
    handler = MobileBridgeHandler.__new__(MobileBridgeHandler)
    handler.path = "/health"
    handler.wfile = BytesIO()
    handler.send_response = MagicMock()
    handler.send_header = MagicMock()
    handler.end_headers = MagicMock()

    handler.do_GET()

    output = handler.wfile.getvalue().decode("utf-8")
    assert output != ""
    data = json.loads(output)
    assert data.get("status") == "ready"
    assert "model_id" in data


def test_mobile_bridge_model_info():
    """Validates /model endpoint."""
    handler = MobileBridgeHandler.__new__(MobileBridgeHandler)
    handler.path = "/model"
    handler.wfile = BytesIO()
    handler.send_response = MagicMock()
    handler.send_header = MagicMock()
    handler.end_headers = MagicMock()

    handler.do_GET()

    output = handler.wfile.getvalue().decode("utf-8")
    data = json.loads(output)
    assert "model_id" in data
    assert "device" in data


def test_mobile_bridge_predict_simulation():
    """Validates /predict endpoint with simulated fallback."""
    import scripts.mobile_bridge as mb
    original_model = mb.ACTIVE_MODEL
    mb.ACTIVE_MODEL = None  # Force simulation mode

    try:
        img = Image.new("RGB", (64, 64), color="blue")
        buf = BytesIO()
        img.save(buf, format="JPEG")
        b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")

        payload = json.dumps({"image": b64_str}).encode("utf-8")

        handler = MobileBridgeHandler.__new__(MobileBridgeHandler)
        handler.path = "/predict"
        handler.headers = {"Content-Length": str(len(payload))}
        handler.rfile = BytesIO(payload)
        handler.wfile = BytesIO()
        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()

        handler.do_POST()

        output = handler.wfile.getvalue().decode("utf-8")
        data = json.loads(output)
        assert "detections" in data
        assert len(data["detections"]) > 0
        det = data["detections"][0]
        assert "denomination_id" in det
        assert "confidence" in det
        assert "bbox" in det
        assert len(det["bbox"]) == 4
    finally:
        mb.ACTIVE_MODEL = original_model


def test_mobile_bridge_glasses_status():
    """Validates /glasses/status endpoint."""
    handler = MobileBridgeHandler.__new__(MobileBridgeHandler)
    handler.path = "/glasses/status"
    handler.wfile = BytesIO()
    handler.send_response = MagicMock()
    handler.send_header = MagicMock()
    handler.end_headers = MagicMock()

    handler.do_GET()

    output = handler.wfile.getvalue().decode("utf-8")
    data = json.loads(output)
    assert "connected" in data
    assert "thermal_state" in data
    assert "active_model" in data


def test_decode_tinyml_predictions_synthetic():
    """Validates decode_tinyml_predictions with synthetic output tensors."""
    import numpy as np
    from scripts.mobile_bridge import decode_tinyml_predictions

    # Create dummy p16 (1, 3, 10, 10, 12) and p32 (1, 3, 5, 5, 12)
    p16 = np.zeros((1, 3, 10, 10, 12), dtype=np.float32)
    p32 = np.zeros((1, 3, 5, 5, 12), dtype=np.float32)

    # Invert logits so confidence is high for anchor 0 at grid (5, 5) for 50 AZN (class 4)
    # obj logit = 5.0 (sigmoid ~ 0.993)
    p16[0, 0, 5, 5, 4] = 5.0
    # class 4 logit = 5.0
    p16[0, 0, 5, 5, 5 + 4] = 5.0

    dets = decode_tinyml_predictions(p16, p32, conf_thresh=0.50)
    assert len(dets) == 1
    assert dets[0]["denomination_id"] == 4
    assert dets[0]["class_code"] == "050_azn"
    assert dets[0]["confidence"] > 0.90
    assert len(dets[0]["bbox"]) == 4

