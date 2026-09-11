"""
Local Inference Bridge Server for Mobile App & Smart Glasses.

Exposes an ultra-low latency HTTP endpoint for the mobile app (Expo Go)
to send camera frames and receive bounding box detections, class predictions,
and inference timings. Powered by local NVIDIA GPU acceleration (or CPU).
Decoupled: dynamically loads whatever model is specified or available,
supporting Ultralytics YOLO models (yolo11m, yolov8m) and TinyML YOLO-FastestV2
(PyTorch / ONNX INT8). Also bridges Seeed Studio XIAO ESP32-S3 Sense live stream.
"""

import argparse
import base64
import json
import logging
import math
import os
import sys
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps

# Ensure project root in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.core.config import MobileBridgeConfig, ProjectPaths
from src.core.constants import CLASS_ID_MAP
from src.hardware.esp32_camera import ESP32CameraBridge, find_esp32s3_port

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("MobileBridge")

PATHS = ProjectPaths()
BRIDGE_CFG = MobileBridgeConfig()

LOADED_MODELS: Dict[str, Tuple[Any, str]] = {}  # id -> (model_instance, type)
ACTIVE_MODEL: Any = None
ACTIVE_MODEL_ID: str = "yolo11m"
ACTIVE_MODEL_TYPE: str = "ultralytics"
CAMERA_BRIDGE: Optional[ESP32CameraBridge] = None
ACTIVE_CAMERA_URL: Optional[str] = os.environ.get("ESP32_CAMERA_URL")

# Anchors for YOLO-FastestV2
ANCHORS_S16 = [(118.0, 58.0), (56.0, 112.0), (84.0, 82.0)]
ANCHORS_S32 = [(62.0, 32.0), (30.0, 60.0), (40.0, 42.0)]


def discover_camera_url() -> Optional[str]:
    """Auto-discovers camera stream URL across environment, mDNS, and known IP subnets."""
    env_url = os.environ.get("ESP32_CAMERA_URL")
    if env_url:
        return env_url

    import socket
    try:
        ip = socket.gethostbyname("aznvision.local")
        if ip:
            return f"http://{ip}:81/stream"
    except Exception:
        pass

    # Fallback to current WPRVT subnet IP
    return "http://192.168.0.6:81/stream"


def get_camera_bridge(target_url: Optional[str] = None) -> Optional[ESP32CameraBridge]:
    """Retrieves or initializes the ESP32CameraBridge singleton (Serial or Wi-Fi)."""
    global CAMERA_BRIDGE, ACTIVE_CAMERA_URL
    if target_url:
        ACTIVE_CAMERA_URL = target_url

    target_addr = ACTIVE_CAMERA_URL or discover_camera_url() or find_esp32s3_port()

    if CAMERA_BRIDGE is None:
        if target_addr:
            CAMERA_BRIDGE = ESP32CameraBridge(port=target_addr)
            CAMERA_BRIDGE.connect()
    elif target_addr and CAMERA_BRIDGE.port_name != target_addr:
        CAMERA_BRIDGE.disconnect()
        CAMERA_BRIDGE = ESP32CameraBridge(port=target_addr)
        CAMERA_BRIDGE.connect()
    elif not CAMERA_BRIDGE.is_connected():
        CAMERA_BRIDGE.connect()
    return CAMERA_BRIDGE


def compute_iou(b1: List[float], b2: List[float]) -> float:
    """Computes IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def decode_tinyml_predictions(
    p16: np.ndarray,
    p32: np.ndarray,
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.35,
    max_det: int = 3,
) -> List[Dict[str, Any]]:
    """Decodes raw YOLO-FastestV2 feature tensors into normalized bounding box detections."""
    candidates: List[Dict[str, Any]] = []

    for head_idx, (p, grid_size, anchors) in enumerate([
        (p16, 10, ANCHORS_S16),
        (p32, 5, ANCHORS_S32),
    ]):
        p_squeeze = p[0]  # (3, grid, grid, 12)
        for a_idx, (aw, ah) in enumerate(anchors):
            head_a = p_squeeze[a_idx]  # (grid, grid, 12)
            tx = 1.0 / (1.0 + np.exp(-head_a[..., 0]))
            ty = 1.0 / (1.0 + np.exp(-head_a[..., 1]))
            tw = np.clip(head_a[..., 2], -4.0, 4.0)
            th = np.clip(head_a[..., 3], -4.0, 4.0)
            obj = 1.0 / (1.0 + np.exp(-head_a[..., 4]))
            cls_probs = 1.0 / (1.0 + np.exp(-head_a[..., 5:]))  # (grid, grid, 7)

            cls_idx = np.argmax(cls_probs, axis=-1)
            cls_conf = np.max(cls_probs, axis=-1)
            scores = obj * cls_conf

            mask = scores >= conf_thresh
            if not np.any(mask):
                continue

            ys, xs = np.where(mask)
            for y, x in zip(ys, xs):
                score = float(scores[y, x])
                c_id = int(cls_idx[y, x])
                xc = (x + tx[y, x]) / float(grid_size)
                yc = (y + ty[y, x]) / float(grid_size)
                w = (aw * np.exp(tw[y, x])) / 160.0
                h = (ah * np.exp(th[y, x])) / 160.0

                x1 = max(0.0, min(1.0, float(xc - w / 2.0)))
                y1 = max(0.0, min(1.0, float(yc - h / 2.0)))
                x2 = max(0.0, min(1.0, float(xc + w / 2.0)))
                y2 = max(0.0, min(1.0, float(yc + h / 2.0)))

                if x2 > x1 and y2 > y1:
                    candidates.append({
                        "denomination_id": c_id,
                        "class_code": CLASS_ID_MAP.get(c_id, f"class_{c_id}"),
                        "confidence": round(score, 3),
                        "bbox": [round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)],
                        "model_id": "yolo_fastestv2",
                    })

    if not candidates:
        return []

    candidates.sort(key=lambda d: d["confidence"], reverse=True)
    kept: List[Dict[str, Any]] = []
    while candidates and len(kept) < max_det:
        best = candidates.pop(0)
        kept.append(best)
        candidates = [
            c for c in candidates
            if compute_iou(best["bbox"], c["bbox"]) < iou_thresh
        ]

    return kept


def decode_fomo_predictions(
    heatmap: np.ndarray,
    conf_thresh: float = 0.40,
    max_det: int = 3,
) -> List[Dict[str, Any]]:
    """Decodes FOMO centroid heatmap of shape (1, num_classes + 1, H, W) or (num_classes + 1, H, W)."""
    if heatmap.ndim == 4:
        heatmap = heatmap[0]  # (8, H, W)
    num_channels, h_grid, w_grid = heatmap.shape

    # Apply softmax across classes
    exp_h = np.exp(heatmap - np.max(heatmap, axis=0, keepdims=True))
    probs = exp_h / np.sum(exp_h, axis=0, keepdims=True)  # (8, H, W)

    candidates: List[Dict[str, Any]] = []
    # Channel 0 is background, channels 1..7 are Azerbaijani banknote classes
    for c_id in range(min(num_channels - 1, 7)):
        channel_prob = probs[c_id + 1]  # (H, W)
        ys, xs = np.where(channel_prob >= conf_thresh)
        for y, x in zip(ys, xs):
            score = float(channel_prob[y, x])
            xc = (x + 0.5) / float(w_grid)
            yc = (y + 0.5) / float(h_grid)
            bw, bh = 0.28, 0.18
            x1 = max(0.0, min(1.0, float(xc - bw / 2.0)))
            y1 = max(0.0, min(1.0, float(yc - bh / 2.0)))
            x2 = max(0.0, min(1.0, float(xc + bw / 2.0)))
            y2 = max(0.0, min(1.0, float(yc + bh / 2.0)))
            candidates.append({
                "denomination_id": c_id,
                "class_code": CLASS_ID_MAP.get(c_id, f"class_{c_id}"),
                "confidence": round(score, 3),
                "bbox": [round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)],
                "model_id": "fomo",
            })

    candidates.sort(key=lambda d: d["confidence"], reverse=True)
    return candidates[:max_det]


def load_single_model(model_name: str) -> Tuple[Optional[Any], str]:
    """Loads a specific model into memory and returns (model, model_type)."""
    tinyml_candidates = [
        PATHS.artifacts_dir / "tinyml" / "yolo_fastestv2_160_int8.onnx",
        PATHS.artifacts_dir / "tinyml" / "yolo_fastestv2_160.onnx",
        PATHS.artifacts_dir / "tinyml" / "yolo_fastestv2_best.pt",
    ]

    fomo_candidates = [
        PATHS.artifacts_dir / "tinyml" / "fomo_mobilenetv1" / "fomo_mobilenetv1_96.onnx",
        PATHS.artifacts_dir / "tinyml" / "fomo_mobilenetv1" / "weights" / "fomo_best.pt",
    ]

    # Champion: exp3_color_space_shortcut/arm1_rgb_full
    # Wins on mAP50-95=92.6% (vs arm4=87.5%) — real-world quality metric
    # Per-class: 001_azn=95.18%, 005_azn=86.08%, all others >98%
    # Second choice: exp2_data_augmentation/arm4_full_composite (mAP50-95=87.5%)
    champion_yolo11m = PATHS.experiments_dir / "exp3_color_space_shortcut" / "arm1_rgb_full" / "weights" / "best.pt"
    runner_up_yolo11m = PATHS.experiments_dir / "exp2_data_augmentation" / "arm4_full_composite" / "weights" / "best.pt"
    baseline_yolo11m = PATHS.experiments_dir / "exp1_architecture_battle" / "yolo11m" / "weights" / "best.pt"

    ultralytics_candidates = {
        "yolo11m": champion_yolo11m if champion_yolo11m.exists() else (runner_up_yolo11m if runner_up_yolo11m.exists() else baseline_yolo11m),
        "yolo11m_champion": champion_yolo11m,
        "yolo11m_runner_up": runner_up_yolo11m,
        "yolo11m_baseline": baseline_yolo11m,
        "yolov8m": PATHS.experiments_dir / "exp1_architecture_battle" / "yolov8m" / "weights" / "best.pt",
    }

    if model_name in ("yolo_fastestv2", "tinyml"):
        for cand in tinyml_candidates:
            if cand.exists():
                if cand.suffix == ".onnx":
                    try:
                        import onnxruntime as ort
                        opts = ort.SessionOptions()
                        opts.log_severity_level = 3
                        session = ort.InferenceSession(str(cand), sess_options=opts, providers=["CPUExecutionProvider"])
                        logger.info("Loaded active ONNX TinyML model: %s", cand.name)
                        return session, "tinyml_onnx"
                    except Exception as exc:
                        logger.warning("Failed loading ONNX %s: %s", cand, exc)
                elif cand.suffix == ".pt":
                    try:
                        import torch
                        from src.tinyml.model import YOLOFastestV2
                        m = YOLOFastestV2(num_classes=7)
                        m.load_state_dict(torch.load(str(cand), map_location="cpu"))
                        m.eval()
                        logger.info("Loaded active PyTorch TinyML model: %s", cand.name)
                        return m, "tinyml_torch"
                    except Exception as exc:
                        logger.warning("Failed loading PyTorch %s: %s", cand, exc)

    if model_name in ("fomo", "fomo_mobilenetv1"):
        for cand in fomo_candidates:
            if cand.exists():
                if cand.suffix == ".onnx":
                    try:
                        import onnxruntime as ort
                        opts = ort.SessionOptions()
                        opts.log_severity_level = 3
                        session = ort.InferenceSession(str(cand), sess_options=opts, providers=["CPUExecutionProvider"])
                        logger.info("Loaded active ONNX FOMO model: %s", cand.name)
                        return session, "fomo_onnx"
                    except Exception as exc:
                        logger.warning("Failed loading ONNX FOMO %s: %s", cand, exc)
                elif cand.suffix == ".pt":
                    try:
                        import torch
                        from src.tinyml.fomo_model import FOMOMobileNetV1
                        m = FOMOMobileNetV1(num_classes=7, width_mult=0.35, input_size=96)
                        m.load_state_dict(torch.load(str(cand), map_location="cpu"))
                        m.eval()
                        logger.info("Loaded active PyTorch FOMO model: %s", cand.name)
                        return m, "fomo_torch"
                    except Exception as exc:
                        logger.warning("Failed loading PyTorch FOMO %s: %s", cand, exc)

    if model_name in ultralytics_candidates:
        path = ultralytics_candidates[model_name]
        if path.exists():
            try:
                import torch
                from ultralytics import YOLO
                device = "cuda" if torch.cuda.is_available() else "cpu"
                m = YOLO(str(path))
                if device == "cuda":
                    m.to("cuda")
                gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
                logger.info("Loaded active Ultralytics YOLO model: %s on %s (%s)", model_name, device.upper(), gpu_name)
                return m, "ultralytics"
            except Exception as exc:
                logger.warning("Failed loading Ultralytics %s: %s", model_name, exc)

    return None, "simulation"


def load_inference_model(model_name: str = "auto") -> Any:
    """Prepares model registry and sets default active model."""
    global ACTIVE_MODEL, ACTIVE_MODEL_ID, ACTIVE_MODEL_TYPE, LOADED_MODELS

    # Always preload Server YOLO11m model first
    if "yolo11m" not in LOADED_MODELS:
        m, mtype = load_single_model("yolo11m")
        if m is not None:
            LOADED_MODELS["yolo11m"] = (m, mtype)

    # Always attempt to preload TinyML model
    if "yolo_fastestv2" not in LOADED_MODELS:
        m, mtype = load_single_model("yolo_fastestv2")
        if m is not None:
            LOADED_MODELS["yolo_fastestv2"] = (m, mtype)

    # Determine which model is designated active (default to champion YOLO11m on local GPU)
    target_id = "yolo11m" if model_name in ("auto", "champion", "best", "yolo11m") else model_name

    if target_id in LOADED_MODELS:
        ACTIVE_MODEL, ACTIVE_MODEL_TYPE = LOADED_MODELS[target_id]
        ACTIVE_MODEL_ID = target_id
    elif "yolo11m" in LOADED_MODELS:
        ACTIVE_MODEL, ACTIVE_MODEL_TYPE = LOADED_MODELS["yolo11m"]
        ACTIVE_MODEL_ID = "yolo11m"
    elif "yolo_fastestv2" in LOADED_MODELS:
        ACTIVE_MODEL, ACTIVE_MODEL_TYPE = LOADED_MODELS["yolo_fastestv2"]
        ACTIVE_MODEL_ID = "yolo_fastestv2"
    else:
        ACTIVE_MODEL = None
        ACTIVE_MODEL_TYPE = "simulation"
        ACTIVE_MODEL_ID = "simulation_mode"

    logger.info("Active default model set to: %s (%s)", ACTIVE_MODEL_ID, ACTIVE_MODEL_TYPE)
    return ACTIVE_MODEL


def switch_active_model(model_id: str) -> bool:
    """Switches active model at runtime without restarting server."""
    global ACTIVE_MODEL, ACTIVE_MODEL_ID, ACTIVE_MODEL_TYPE, LOADED_MODELS
    normalized_id = "yolo_fastestv2" if model_id in ("tinyml", "yolo_fastestv2") else model_id

    if normalized_id not in LOADED_MODELS:
        m, mtype = load_single_model(normalized_id)
        if m is not None:
            LOADED_MODELS[normalized_id] = (m, mtype)
        else:
            return False

    ACTIVE_MODEL, ACTIVE_MODEL_TYPE = LOADED_MODELS[normalized_id]
    ACTIVE_MODEL_ID = normalized_id
    logger.info("Switched active model to: %s (%s)", ACTIVE_MODEL_ID, ACTIVE_MODEL_TYPE)
    return True


class MobileBridgeHandler(SimpleHTTPRequestHandler):
    """Handles mobile inference requests and hardware camera stream."""

    def do_GET(self) -> None:
        """Endpoint status, model info, and hardware streaming."""
        if self.path in ("/health", "/"):
            self._send_json({
                "status": "ready",
                "model_id": ACTIVE_MODEL_ID,
                "model_type": ACTIVE_MODEL_TYPE,
                "loaded_models": list(LOADED_MODELS.keys()),
                "timestamp": time.time(),
            })
        elif self.path == "/model":
            self._send_json({
                "model_id": ACTIVE_MODEL_ID,
                "model_type": ACTIVE_MODEL_TYPE,
                "device": "CPU/ONNX INT8" if "tinyml" in ACTIVE_MODEL_TYPE else ("GPU/CUDA" if ACTIVE_MODEL_TYPE == "ultralytics" else "Mock"),
            })
        elif self.path == "/models":
            models_info = [
                {
                    "id": "yolo_fastestv2",
                    "displayName": "YOLO-FastestV2 INT8 (Smart Glasses)",
                    "architectureType": "Mobile INT8 Edge",
                    "resolution": [160, 160],
                    "sizeKb": 290.6,
                    "target": "ESP32-S3 / Edge",
                    "isActive": ACTIVE_MODEL_ID == "yolo_fastestv2",
                    "isLoaded": "yolo_fastestv2" in LOADED_MODELS,
                },
                {
                    "id": "yolo11m",
                    "displayName": "YOLOv11m (C2PSA Attention)",
                    "architectureType": "Spatial Attention CNN",
                    "resolution": [640, 640],
                    "sizeKb": 40200.0,
                    "target": "Server GPU",
                    "isActive": ACTIVE_MODEL_ID == "yolo11m",
                    "isLoaded": "yolo11m" in LOADED_MODELS,
                },
                {
                    "id": "fomo",
                    "displayName": "FOMO MobileNetV1 0.35x (Centroid Edge)",
                    "architectureType": "Centroid Grid Classifier",
                    "resolution": [96, 96],
                    "sizeKb": 281.0,
                    "target": "ESP32-S3 Micro",
                    "isActive": ACTIVE_MODEL_ID in ("fomo", "fomo_mobilenetv1"),
                    "isLoaded": any(k in LOADED_MODELS for k in ("fomo", "fomo_mobilenetv1")),
                },
            ]
            self._send_json({"active_model": ACTIVE_MODEL_ID, "models": models_info})
        elif self.path == "/glasses/status":
            self._handle_glasses_status()
        elif self.path == "/glasses/frame":
            self._handle_glasses_frame()
        elif self.path == "/glasses/frame/json":
            self._handle_glasses_frame_json()
        elif self.path in ("/stream", "/glasses/stream"):
            self._handle_glasses_stream()
        elif self.path.startswith("/glasses/detect") or self.path.startswith("/glasses/live"):
            self._handle_glasses_detect()
        else:
            self.send_error(404, "Endpoint not found")

    def _handle_glasses_detect(self) -> None:
        """Reads live frame from ESP32 camera, runs active TinyML model, and returns frame + detections."""
        start_time = time.perf_counter()
        bridge = get_camera_bridge()
        raw_jpeg = bridge.read_raw_frame(max_read_time=0.6) if bridge else None
        telemetry = bridge.get_hardware_telemetry() if bridge else {}

        if not raw_jpeg:
            self._send_json({
                "status": "waiting_for_hardware",
                "message": "ESP32-S3 OV2640 camera stream initializing...",
                "detections": [],
                "telemetry": telemetry,
                "model_id": ACTIVE_MODEL_ID,
            })
            return

        try:
            # Return clean raw base64 without prefix so client can format uri cleanly
            b64_frame = base64.b64encode(raw_jpeg).decode("utf-8")
            raw_img = Image.open(BytesIO(raw_jpeg))
            img = ImageOps.exif_transpose(raw_img).convert("RGB")
            target_model, target_type, target_id = ACTIVE_MODEL, ACTIVE_MODEL_TYPE, ACTIVE_MODEL_ID
            detections: List[Dict[str, Any]] = []

            if target_type == "tinyml_onnx" and target_model:
                resized = img.resize((160, 160), Image.BILINEAR)
                tensor = np.transpose(np.array(resized, dtype=np.float32) / 255.0, (2, 0, 1))
                tensor = np.expand_dims(tensor, axis=0)
                input_name = target_model.get_inputs()[0].name
                outputs = target_model.run(None, {input_name: tensor})
                detections = decode_tinyml_predictions(outputs[0], outputs[1], conf_thresh=0.30)

            elif target_type == "fomo_onnx" and target_model:
                resized = img.resize((96, 96), Image.BILINEAR)
                tensor = np.transpose(np.array(resized, dtype=np.float32) / 255.0, (2, 0, 1))
                tensor = np.expand_dims(tensor, axis=0)
                input_name = target_model.get_inputs()[0].name
                outputs = target_model.run(None, {input_name: tensor})
                detections = decode_fomo_predictions(outputs[0], conf_thresh=0.40)

            elif target_type == "fomo_torch" and target_model:
                import torch
                resized = img.resize((96, 96), Image.BILINEAR)
                tensor = np.transpose(np.array(resized, dtype=np.float32) / 255.0, (2, 0, 1))
                tensor = torch.from_numpy(tensor).unsqueeze(0).float()
                with torch.no_grad():
                    out = target_model(tensor)
                detections = decode_fomo_predictions(out.cpu().numpy(), conf_thresh=0.40)

            elif target_type == "ultralytics" and target_model:
                import torch
                dev = "cuda" if torch.cuda.is_available() else "cpu"
                results = target_model.predict(img, conf=0.30, device=dev, verbose=False)
                for res in results:
                    w, h = img.size
                    for box in res.boxes:
                        cls_id = int(box.cls[0].item())
                        conf = float(box.conf[0].item())
                        xyxy = box.xyxy[0].tolist()
                        norm_box = [
                            round(xyxy[0] / max(1, w), 3),
                            round(xyxy[1] / max(1, h), 3),
                            round(xyxy[2] / max(1, w), 3),
                            round(xyxy[3] / max(1, h), 3),
                        ]
                        detections.append({
                            "denomination_id": cls_id,
                            "class_code": CLASS_ID_MAP.get(cls_id, f"class_{cls_id}"),
                            "confidence": round(conf, 3),
                            "bbox": norm_box,
                            "model_id": target_id,
                        })

            latency_ms = round((time.perf_counter() - start_time) * 1000.0, 1)
            for d in detections:
                d["latency_ms"] = latency_ms

            self._send_json({
                "status": "success",
                "frame": b64_frame,
                "detections": detections,
                "model_id": target_id,
                "latency_ms": latency_ms,
                "telemetry": telemetry,
                "image_size": [img.size[0], img.size[1]] if "img" in locals() and img else [320, 240],
            })
        except Exception as exc:
            self._send_json({"status": "error", "message": str(exc), "detections": []}, status=500)

    def _handle_glasses_status(self) -> None:
        bridge = get_camera_bridge()
        if bridge and bridge.is_connected():
            telemetry = bridge.get_hardware_telemetry()
            telemetry["active_model"] = ACTIVE_MODEL_ID
            self._send_json(telemetry)
        else:
            self._send_json({
                "connected": False,
                "port": find_esp32s3_port(),
                "thermal_state": "DISCONNECTED",
                "instant_fps": 0.0,
                "average_fps": 0.0,
                "active_model": ACTIVE_MODEL_ID,
            })

    def _handle_glasses_frame(self) -> None:
        bridge = get_camera_bridge()
        raw_jpeg = bridge.read_raw_frame() if bridge else None
        if raw_jpeg:
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(raw_jpeg)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(raw_jpeg)
        else:
            self.send_error(503, "Camera frame unavailable from smart glasses")

    def _handle_glasses_frame_json(self) -> None:
        bridge = get_camera_bridge()
        raw_jpeg = bridge.read_raw_frame() if bridge else None
        if raw_jpeg:
            b64_frame = base64.b64encode(raw_jpeg).decode("utf-8")
            telemetry = bridge.get_hardware_telemetry() if bridge else {}
            self._send_json({
                "status": "success",
                "frame": b64_frame,
                "telemetry": telemetry,
            })
        else:
            self._send_json({"status": "error", "message": "Camera frame unavailable"}, status=503)

    def _handle_glasses_stream(self) -> None:
        """Proxies real-time MJPEG camera stream directly to web browsers and mobile clients."""
        bridge = get_camera_bridge()
        if not bridge:
            self.send_error(503, "Camera bridge unavailable")
            return

        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        try:
            while True:
                frame = bridge.read_raw_frame(max_read_time=0.25)
                if frame:
                    part = (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        + f"Content-Length: {len(frame)}\r\n\r\n".encode("latin1")
                        + frame
                        + b"\r\n"
                    )
                    self.wfile.write(part)
                    self.wfile.flush()
                else:
                    time.sleep(0.02)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def do_POST(self) -> None:
        """Frame inference endpoint /predict."""
        if self.path == "/predict":
            self._handle_predict()
        elif self.path == "/model":
            self._handle_switch_model()
        elif self.path in ("/glasses/wifi", "/glasses/connect"):
            self._handle_connect_camera()
        else:
            self.send_error(404, "Endpoint not found")

    def _handle_connect_camera(self) -> None:
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")
        try:
            payload = json.loads(body)
            url = payload.get("url") or payload.get("port")
            if url:
                bridge = get_camera_bridge(url)
                self._send_json({
                    "status": "ok",
                    "message": f"Connected to camera at {url}",
                    "telemetry": bridge.get_hardware_telemetry() if bridge else {},
                })
            else:
                self._send_json({"status": "error", "message": "Missing 'url' parameter in body"}, status=400)
        except Exception as exc:
            self._send_json({"status": "error", "message": str(exc)}, status=400)

    def _handle_switch_model(self) -> None:
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")
        try:
            payload = json.loads(body)
            target_model = payload.get("model_id") or payload.get("model")
            if target_model and switch_active_model(target_model):
                self._send_json({"status": "ok", "active_model": ACTIVE_MODEL_ID, "model_type": ACTIVE_MODEL_TYPE})
            else:
                self._send_json({"status": "error", "message": f"Unknown or unloaded model: {target_model}"}, status=400)
        except Exception as exc:
            self._send_json({"status": "error", "message": str(exc)}, status=400)

    def _handle_predict(self) -> None:
        start_time = time.perf_counter()
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")

        try:
            payload = json.loads(body)
            img_b64 = payload.get("image", "")

            # Optional model override per inference call
            req_model = payload.get("model_id") or payload.get("model")
            target_model, target_type, target_id = ACTIVE_MODEL, ACTIVE_MODEL_TYPE, ACTIVE_MODEL_ID

            if req_model:
                norm_req = "yolo_fastestv2" if req_model in ("tinyml", "yolo_fastestv2") else req_model
                if norm_req in LOADED_MODELS:
                    target_model, target_type = LOADED_MODELS[norm_req]
                    target_id = norm_req
                else:
                    m, mtype = load_single_model(norm_req)
                    if m is not None:
                        LOADED_MODELS[norm_req] = (m, mtype)
                        target_model, target_type, target_id = m, mtype, norm_req

            if "," in img_b64:
                img_b64 = img_b64.split(",", 1)[1]

            detections: List[Dict[str, Any]] = []

            if target_model and img_b64:
                img_bytes = base64.b64decode(img_b64)
                raw_img = Image.open(BytesIO(img_bytes))
                img = ImageOps.exif_transpose(raw_img).convert("RGB")

                if target_type == "ultralytics":
                    import torch
                    dev = "cuda" if torch.cuda.is_available() else "cpu"
                    results = target_model.predict(img, conf=0.30, device=dev, verbose=False)
                    for res in results:
                        boxes = res.boxes
                        w, h = img.size
                        for box in boxes:
                            cls_id = int(box.cls[0].item())
                            conf = float(box.conf[0].item())
                            xyxy = box.xyxy[0].tolist()
                            norm_box = [
                                round(xyxy[0] / max(1, w), 3),
                                round(xyxy[1] / max(1, h), 3),
                                round(xyxy[2] / max(1, w), 3),
                                round(xyxy[3] / max(1, h), 3),
                            ]
                            class_code = CLASS_ID_MAP.get(cls_id, f"class_{cls_id}")
                            detections.append({
                                "denomination_id": cls_id,
                                "class_code": class_code,
                                "confidence": round(conf, 3),
                                "bbox": norm_box,
                                "model_id": target_id,
                            })

                elif target_type == "tinyml_onnx":
                    resized = img.resize((160, 160), Image.BILINEAR)
                    tensor = np.transpose(np.array(resized, dtype=np.float32) / 255.0, (2, 0, 1))
                    tensor = np.expand_dims(tensor, axis=0)

                    input_name = target_model.get_inputs()[0].name
                    outputs = target_model.run(None, {input_name: tensor})
                    p16, p32 = outputs[0], outputs[1]
                    detections = decode_tinyml_predictions(p16, p32, conf_thresh=0.30)

                elif target_type == "tinyml_torch":
                    import torch
                    resized = img.resize((160, 160), Image.BILINEAR)
                    tensor = np.transpose(np.array(resized, dtype=np.float32) / 255.0, (2, 0, 1))
                    tensor = torch.from_numpy(tensor).unsqueeze(0).float()
                    with torch.no_grad():
                        p16, p32 = target_model(tensor)
                    detections = decode_tinyml_predictions(p16.cpu().numpy(), p32.cpu().numpy(), conf_thresh=0.30)

                elif target_type == "fomo_onnx":
                    resized = img.resize((96, 96), Image.BILINEAR)
                    tensor = np.transpose(np.array(resized, dtype=np.float32) / 255.0, (2, 0, 1))
                    tensor = np.expand_dims(tensor, axis=0)
                    input_name = target_model.get_inputs()[0].name
                    outputs = target_model.run(None, {input_name: tensor})
                    detections = decode_fomo_predictions(outputs[0], conf_thresh=0.40)

                elif target_type == "fomo_torch":
                    import torch
                    resized = img.resize((96, 96), Image.BILINEAR)
                    tensor = np.transpose(np.array(resized, dtype=np.float32) / 255.0, (2, 0, 1))
                    tensor = torch.from_numpy(tensor).unsqueeze(0).float()
                    with torch.no_grad():
                        out = target_model(tensor)
                    detections = decode_fomo_predictions(out.cpu().numpy(), conf_thresh=0.40)

            if not detections and (not target_model or len(detections) == 0):
                if not target_model:
                    detections = [
                        {
                            "denomination_id": BRIDGE_CFG.simulation_default_class_id,
                            "class_code": BRIDGE_CFG.simulation_default_class_code,
                            "confidence": BRIDGE_CFG.simulation_default_conf,
                            "bbox": list(BRIDGE_CFG.simulation_default_bbox),
                            "model_id": target_id,
                        }
                    ]

            latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
            for d in detections:
                d["latency_ms"] = latency_ms

            img_w, img_h = (img.size[0], img.size[1]) if "img" in locals() and img else (640, 640)
            self._send_json({
                "detections": detections,
                "latency_ms": latency_ms,
                "model_id": target_id,
                "image_size": [img_w, img_h],
            })

        except Exception as exc:
            self._send_json({"error": str(exc), "detections": []}, status=400)

    def _send_json(self, data: Dict[str, Any], status: int = 200) -> None:
        encoded = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(encoded)

    def do_OPTIONS(self) -> None:
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


def run_bridge(
    host: str = BRIDGE_CFG.default_host,
    port: int = BRIDGE_CFG.default_port,
    model: str = "auto",
) -> None:
    """Runs the HTTP mobile inference bridge."""
    load_inference_model(model)
    try:
        bridge = get_camera_bridge()
        if bridge:
            logger.info("Camera bridge pre-warmed on: %s", bridge.port_name)
    except Exception as exc:
        logger.warning("Camera bridge pre-warm notice: %s", exc)
    server_address = (host, port)
    httpd = ThreadingHTTPServer(server_address, MobileBridgeHandler)
    httpd.daemon_threads = True
    logger.info("Mobile Vision Bridge running at http://%s:%d/", host, port)
    logger.info("Ready to accept frames for model: %s (type: %s)", ACTIVE_MODEL_ID, ACTIVE_MODEL_TYPE)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Stopping Mobile Vision Bridge.")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mobile Vision Inference Bridge")
    parser.add_argument("--host", default=BRIDGE_CFG.default_host, help=f"Host binding (default {BRIDGE_CFG.default_host})")
    parser.add_argument("--port", type=int, default=BRIDGE_CFG.default_port, help=f"Port to bind (default {BRIDGE_CFG.default_port})")
    parser.add_argument("--model", default="auto", help="Model to load: auto, yolo11m, yolov8m, yolo_fastestv2")
    args = parser.parse_args()
    run_bridge(host=args.host, port=args.port, model=args.model)
