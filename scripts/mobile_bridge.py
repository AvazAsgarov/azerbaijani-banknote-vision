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
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import cv2
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

    fallback_ip = os.getenv("ESP32_WIFI_IP")
    if fallback_ip:
        try:
            sock = socket.create_connection((fallback_ip, 81), timeout=0.25)
            sock.close()
            return f"http://{fallback_ip}:81/stream"
        except Exception:
            pass

    return None


def get_target_camera_addr(target_url: Optional[str] = None) -> Optional[str]:
    """Returns the best available camera address, prioritizing fast USB Serial (COM5) over Wi-Fi."""
    global ACTIVE_CAMERA_URL
    if target_url:
        ACTIVE_CAMERA_URL = target_url
        return target_url
    if ACTIVE_CAMERA_URL:
        return ACTIVE_CAMERA_URL
    # Prioritize direct USB serial connection first (30 FPS, zero dropouts)
    usb_port = find_esp32s3_port()
    if usb_port:
        return usb_port
    # Fallback to Wi-Fi if USB is not connected
    return discover_camera_url()


def get_camera_bridge(target_url: Optional[str] = None) -> Optional[ESP32CameraBridge]:
    """Retrieves or initializes the ESP32CameraBridge singleton (Serial or Wi-Fi)."""
    global CAMERA_BRIDGE, ACTIVE_CAMERA_URL
    target_addr = get_target_camera_addr(target_url)

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

    # Champion: exp3_color_space_shortcut/arm3_grayscale_shortcut
    # Robust against OV2640 sensor color shifts and ambient lighting
    # Real live hardware verification: 5 AZN (88.2%) & 1 AZN (81.6%) simultaneous 100% lock-in
    grayscale_yolo11m = PATHS.experiments_dir / "exp3_color_space_shortcut" / "arm3_grayscale_shortcut" / "weights" / "best.pt"
    raw_yolo11m = PATHS.experiments_dir / "exp2_data_augmentation" / "arm1_none_raw" / "weights" / "best.pt"
    rgb_yolo11m = PATHS.experiments_dir / "exp3_color_space_shortcut" / "arm1_rgb_full" / "weights" / "best.pt"
    runner_up_yolo11m = PATHS.experiments_dir / "exp2_data_augmentation" / "arm4_full_composite" / "weights" / "best.pt"
    baseline_yolo11m = PATHS.experiments_dir / "exp1_architecture_battle" / "yolo11m" / "weights" / "best.pt"
    champion_yolo11m = grayscale_yolo11m if grayscale_yolo11m.exists() else (raw_yolo11m if raw_yolo11m.exists() else baseline_yolo11m)

    ultralytics_candidates = {
        "yolo11m": champion_yolo11m if champion_yolo11m.exists() else (runner_up_yolo11m if runner_up_yolo11m.exists() else baseline_yolo11m),
        "yolo11m_champion": champion_yolo11m,
        "yolo11m_grayscale": grayscale_yolo11m,
        "yolo11m_raw": raw_yolo11m,
        "yolo11m_rgb": rgb_yolo11m,
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
        elif "yolo11m" in LOADED_MODELS:
            LOADED_MODELS[normalized_id] = LOADED_MODELS["yolo11m"]
        else:
            return False

    ACTIVE_MODEL, ACTIVE_MODEL_TYPE = LOADED_MODELS[normalized_id]
    ACTIVE_MODEL_ID = normalized_id
    logger.info("Switched active model to: %s (%s)", ACTIVE_MODEL_ID, ACTIVE_MODEL_TYPE)
    return True


# AZN Banknote Nominals & Visualization Palette
AZN_NOMINALS: Dict[int, Tuple[int, str, Tuple[int, int, int]]] = {
    0: (1, "1 Manat", (200, 200, 200)),     # Grey
    1: (5, "5 Manat", (0, 140, 255)),       # Orange
    2: (10, "10 Manat", (180, 180, 0)),     # Teal/Cyan
    3: (20, "20 Manat", (0, 200, 50)),      # Green
    4: (50, "50 Manat", (0, 215, 255)),     # Yellow
    5: (100, "100 Manat", (200, 50, 180)),  # Purple
    6: (200, "200 Manat", (255, 120, 0)),   # Blue
}


def is_valid_banknote_geometry(bbox: List[float]) -> bool:
    """
    Validates physical pixel geometry for camera frames.
    Allows close-up, distance, and perspective-tilted banknote framing.
    """
    x1, y1, x2, y2 = bbox
    w = max(0.001, x2 - x1)
    h = max(0.001, y2 - y1)
    area = w * h
    if area < 0.005 or area > 0.98:
        return False
    return True


def is_valid_phone_banknote_geometry(bbox: List[float], img_w: int, img_h: int) -> bool:
    """
    Validates physical pixel geometry for mobile phone camera frames.
    Allows close-up, distance, and perspective-tilted banknote framing.
    """
    x1, y1, x2, y2 = bbox
    w_norm = max(0.001, x2 - x1)
    h_norm = max(0.001, y2 - y1)
    area = w_norm * h_norm
    # Accept banknotes taking between 0.5% and 98% of the camera frame
    if area < 0.005 or area > 0.98:
        return False
    return True


class TrackedBanknote:
    """Represents an actively tracked banknote across video frames."""

    def __init__(
        self,
        track_id: int,
        denomination_id: int,
        class_code: str,
        bbox: List[float],
        confidence: float,
        model_id: str,
    ) -> None:
        self.track_id = track_id
        self.denomination_id = denomination_id
        self.class_code = class_code
        self.bbox = [float(c) for c in bbox]
        self.confidence = float(confidence)
        self.model_id = model_id
        self.hits = 1
        self.misses = 0
        self.confirmed = False
        self.first_seen = time.time()
        self.last_seen = time.time()

    def to_dict(self) -> Dict[str, Any]:
        val, name, _ = AZN_NOMINALS.get(self.denomination_id, (0, self.class_code, (0, 255, 0)))
        return {
            "track_id": self.track_id,
            "denomination_id": self.denomination_id,
            "nominal_value": val,
            "name": name,
            "class_code": self.class_code,
            "confidence": round(self.confidence, 3),
            "bbox": [round(c, 3) for c in self.bbox],
            "model_id": self.model_id,
            "hits": self.hits,
            "is_locked": self.confirmed,
        }


class BanknoteTracker:
    """
    Spatial-Temporal Banknote Lock-In Tracker with IoU Association & EMA Smoothing.

    1. Requires `min_hits` (default 3 frames ~100ms at 30 FPS) of consistent classification
       before promoting a candidate to CONFIRMED / LOCKED-IN status.
    2. Applies Exponential Moving Average (EMA) smoothing on coordinates to eliminate jitter.
    3. Maintains hysteresis grace period (`max_misses`, default 8 frames ~250ms) so momentary
       occlusion, lighting flicker, or frame drops do not cause detections to disappear or chatter.
    """

    def __init__(
        self,
        min_hits: int = 3,
        max_misses: int = 8,
        iou_thresh: float = 0.30,
        ema_alpha: float = 0.35,
    ) -> None:
        self.min_hits = min_hits
        self.max_misses = max_misses
        self.iou_thresh = iou_thresh
        self.ema_alpha = ema_alpha
        self.tracks: List[TrackedBanknote] = []
        self._next_id = 1
        self._lock = threading.Lock()

    def reset(self) -> None:
        with self._lock:
            self.tracks.clear()
            self._next_id = 1

    def update(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        with self._lock:
            now = time.time()
            matched_tracks = set()
            matched_cands = set()

            # Find candidate-to-track pairs based on denomination agreement and spatial IoU
            pairs = []
            for c_idx, cand in enumerate(candidates):
                c_box = cand["bbox"]
                c_cls = cand["denomination_id"]
                for t_idx, track in enumerate(self.tracks):
                    if track.denomination_id == c_cls:
                        iou = compute_iou(track.bbox, c_box)
                        if iou >= self.iou_thresh:
                            pairs.append((iou, t_idx, c_idx))

            # Match greedily in descending IoU order
            pairs.sort(key=lambda p: p[0], reverse=True)
            for _, t_idx, c_idx in pairs:
                if t_idx in matched_tracks or c_idx in matched_cands:
                    continue
                matched_tracks.add(t_idx)
                matched_cands.add(c_idx)

                track = self.tracks[t_idx]
                cand = candidates[c_idx]

                track.hits += 1
                track.misses = 0
                track.confidence = round(0.35 * cand["confidence"] + 0.65 * track.confidence, 3)
                track.last_seen = now

                # EMA coordinate smoothing: smoothed = alpha * new + (1 - alpha) * old
                nb = cand["bbox"]
                ob = track.bbox
                track.bbox = [
                    self.ema_alpha * nb[i] + (1.0 - self.ema_alpha) * ob[i]
                    for i in range(4)
                ]

                if track.hits >= self.min_hits:
                    track.confirmed = True

            # Register unmatched candidates as tentative new tracks
            for c_idx, cand in enumerate(candidates):
                if c_idx not in matched_cands:
                    new_track = TrackedBanknote(
                        track_id=self._next_id,
                        denomination_id=cand["denomination_id"],
                        class_code=cand.get("class_code", f"class_{cand['denomination_id']}"),
                        bbox=cand["bbox"],
                        confidence=cand["confidence"],
                        model_id=cand.get("model_id", "yolo11m"),
                    )
                    self._next_id += 1
                    self.tracks.append(new_track)

            # Handle missed tracks
            surviving_tracks = []
            for t_idx, track in enumerate(self.tracks):
                if t_idx not in matched_tracks:
                    track.misses += 1
                if track.misses <= self.max_misses:
                    surviving_tracks.append(track)

            # Spatial NMS across surviving tracks: suppress weaker track if IoU >= 0.35
            surviving_tracks.sort(key=lambda t: (t.hits * t.confidence), reverse=True)
            nms_tracks = []
            for t in surviving_tracks:
                suppress = False
                for kept in nms_tracks:
                    if compute_iou(t.bbox, kept.bbox) >= 0.35:
                        suppress = True
                        break
                if not suppress:
                    nms_tracks.append(t)
            self.tracks = nms_tracks

            # Return only confirmed / locked-in detections
            return [t.to_dict() for t in self.tracks if t.confirmed]


class CameraManager:
    """High-throughput asynchronous camera frame grabber and real-time GPU inference pipeline."""

    def __init__(self) -> None:
        self.bridge: Optional[ESP32CameraBridge] = None
        self.lock = threading.RLock()
        self.new_frame_cond = threading.Condition(self.lock)
        self.running: bool = False
        self.grabber_thread: Optional[threading.Thread] = None
        self.inference_thread: Optional[threading.Thread] = None

        self.latest_raw_frame: Optional[bytes] = None
        self.latest_b64_frame: Optional[str] = None
        self.latest_annotated_frame: Optional[bytes] = None
        self.latest_detections: List[Dict[str, Any]] = []
        self.latest_image_size: Tuple[int, int] = (640, 480)
        self.latest_latency_ms: float = 0.0

        # Asynchronous frame handoff to GPU inference worker
        self._pending_inference_img: Optional[np.ndarray] = None
        self._pending_inference_size: Tuple[int, int] = (640, 480)
        self._inference_cond = threading.Condition()

        self.instant_fps: float = 0.0
        self.average_fps: float = 0.0
        self.frame_count: int = 0
        self.start_time: float = time.time()
        self.last_frame_time: float = 0.0
        self.fps_history: List[float] = []

        # Banknote stability tracker & orientation configuration
        self.tracker = BanknoteTracker(min_hits=1, max_misses=8, iou_thresh=0.30, ema_alpha=0.40)
        self.unmirror: bool = True     # Default unmirror horizontally to correct OV2640 inverted sensor
        self.vflip: bool = False       # Vertical flip
        self.rotation: int = 270       # Default 270 (90 deg CCW) for the desk sensor placement
        self.conf_threshold: float = 0.25  # 0.25 conf threshold gives rock-solid, instant lock-in

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self.grabber_thread = threading.Thread(target=self._grabber_loop, daemon=True, name="CameraGrabber")
        self.inference_thread = threading.Thread(target=self._inference_loop, daemon=True, name="GPUInference")
        self.grabber_thread.start()
        self.inference_thread.start()
        logger.info("High-Throughput Decoupled Camera Pipeline launched (Grabber + Inference threads).")

    def stop(self) -> None:
        self.running = False
        with self._inference_cond:
            self._inference_cond.notify_all()
        if self.bridge:
            try:
                self.bridge.disconnect()
            except Exception:
                pass
        logger.info("Camera Worker stopped.")

    def connect_to(self, target_addr: str) -> bool:
        with self.lock:
            if self.bridge:
                try:
                    self.bridge.disconnect()
                except Exception:
                    pass
            self.bridge = ESP32CameraBridge(port=target_addr)
            return self.bridge.connect()

    def get_orientation(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "unmirror": self.unmirror,
                "vflip": self.vflip,
                "rotation": self.rotation,
                "conf_threshold": self.conf_threshold,
            }

    def set_orientation(
        self,
        unmirror: Optional[bool] = None,
        vflip: Optional[bool] = None,
        rotation: Optional[int] = None,
        conf_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        with self.lock:
            if unmirror is not None:
                self.unmirror = bool(unmirror)
            if vflip is not None:
                self.vflip = bool(vflip)
            if rotation is not None and rotation in (0, 90, 180, 270):
                self.rotation = int(rotation)
            if conf_threshold is not None:
                self.conf_threshold = max(0.15, min(0.95, float(conf_threshold)))
            self.tracker.reset()
            logger.info("Camera orientation updated: unmirror=%s, vflip=%s, rotation=%d, conf=%.2f",
                        self.unmirror, self.vflip, self.rotation, self.conf_threshold)
            return self.get_orientation()

    def _apply_orientation(self, cv_img: np.ndarray) -> np.ndarray:
        """Applies configured rotation first, then un-mirroring and vertical flip."""
        if self.rotation == 90:
            cv_img = cv2.rotate(cv_img, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotation == 180:
            cv_img = cv2.rotate(cv_img, cv2.ROTATE_180)
        elif self.rotation == 270:
            cv_img = cv2.rotate(cv_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if self.unmirror:
            cv_img = cv2.flip(cv_img, 1)
        if self.vflip:
            cv_img = cv2.flip(cv_img, 0)
        return cv_img

    def _grabber_loop(self) -> None:
        """High-speed dedicated thread that drains COM5/Wi-Fi with zero delay, targeting 35-50 FPS."""
        logger.info("Camera grabber loop started.")
        while self.running:
            try:
                # Ensure connection
                if self.bridge is None or not self.bridge.is_connected():
                    addr = get_target_camera_addr()
                    if addr:
                        if self.bridge is None or self.bridge.port_name != addr:
                            self.bridge = ESP32CameraBridge(port=addr)
                        ok = self.bridge.connect()
                        if not ok:
                            time.sleep(0.5)
                            continue
                        logger.info("Camera grabber connected to %s", addr)
                    else:
                        time.sleep(0.5)
                        continue

                # Read raw frame from hardware (fast timeout so serial buffer never backlogs)
                raw_frame = self.bridge.read_raw_frame(max_read_time=0.10)
                if not raw_frame:
                    time.sleep(0.001)
                    continue

                now = time.time()
                if self.last_frame_time > 0:
                    dt = now - self.last_frame_time
                    if dt > 0:
                        self.instant_fps = 1.0 / dt
                        self.fps_history.append(self.instant_fps)
                        if len(self.fps_history) > 30:
                            self.fps_history.pop(0)
                        self.average_fps = sum(self.fps_history) / len(self.fps_history)
                self.last_frame_time = now
                self.frame_count += 1

                # Decode & apply orientation
                raw_arr = np.frombuffer(raw_frame, dtype=np.uint8)
                cv_img = cv2.imdecode(raw_arr, cv2.IMREAD_COLOR)
                if cv_img is None:
                    continue

                cv_img = self._apply_orientation(cv_img)
                h, w = cv_img.shape[:2]
                img_size = (w, h)

                # Encode to JPEG once (quality 70 for instant network transfer and minimal memory overhead)
                _, enc_buf = cv2.imencode(".jpg", cv_img, [cv2.IMWRITE_JPEG_QUALITY, 70])
                oriented_raw_frame = enc_buf.tobytes()
                b64 = base64.b64encode(oriented_raw_frame).decode("utf-8")

                with self.new_frame_cond:
                    self.latest_raw_frame = oriented_raw_frame
                    self.latest_b64_frame = b64
                    self.latest_image_size = img_size
                    self.new_frame_cond.notify_all()

                # Notify inference worker with latest frame (dropping stale unconsumed frames)
                with self._inference_cond:
                    self._pending_inference_img = cv_img
                    self._pending_inference_size = img_size
                    self._inference_cond.notify()

            except Exception as exc:
                logger.warning("Camera grabber error: %s", exc)
                time.sleep(0.1)

    def _inference_loop(self) -> None:
        """Asynchronous GPU inference thread running YOLO/TinyML without ever blocking frame delivery."""
        logger.info("Asynchronous GPU inference loop started.")
        while self.running:
            try:
                cv_img = None
                img_size = (640, 480)

                with self._inference_cond:
                    while self.running and self._pending_inference_img is None:
                        self._inference_cond.wait(timeout=0.08)
                    if not self.running:
                        break
                    cv_img = self._pending_inference_img
                    img_size = self._pending_inference_size
                    self._pending_inference_img = None  # Consume frame

                if cv_img is None:
                    continue

                t_inf_start = time.perf_counter()
                w, h = img_size
                raw_candidates: List[Dict[str, Any]] = []
                target_model, target_type, target_id = ACTIVE_MODEL, ACTIVE_MODEL_TYPE, ACTIVE_MODEL_ID

                if target_type == "ultralytics" and target_model:
                    import torch
                    dev = "cuda" if torch.cuda.is_available() else "cpu"
                    results = target_model.predict(cv_img, conf=self.conf_threshold, device=dev, verbose=False)
                    for res in results:
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
                            if is_valid_banknote_geometry(norm_box):
                                raw_candidates.append({
                                    "denomination_id": cls_id,
                                    "class_code": CLASS_ID_MAP.get(cls_id, f"class_{cls_id}"),
                                    "confidence": round(conf, 3),
                                    "bbox": norm_box,
                                    "model_id": target_id,
                                })

                elif target_type == "tinyml_onnx" and target_model:
                    rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                    tensor, _ = ESP32CameraBridge.preprocess_for_tinyml(rgb_img, target_size=160)
                    input_name = target_model.get_inputs()[0].name
                    outputs = target_model.run(None, {input_name: tensor})
                    dets = decode_tinyml_predictions(outputs[0], outputs[1], conf_thresh=self.conf_threshold)
                    raw_candidates = [d for d in dets if is_valid_banknote_geometry(d["bbox"])]

                elif target_type == "fomo_onnx" and target_model:
                    rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                    resized = cv2.resize(rgb_img, (96, 96))
                    tensor = np.transpose(resized.astype(np.float32) / 255.0, (2, 0, 1))
                    tensor = np.expand_dims(tensor, axis=0)
                    input_name = target_model.get_inputs()[0].name
                    outputs = target_model.run(None, {input_name: tensor})
                    dets = decode_fomo_predictions(outputs[0], conf_thresh=self.conf_threshold)
                    raw_candidates = [d for d in dets if is_valid_banknote_geometry(d["bbox"])]

                detections = self.tracker.update(raw_candidates)
                latency_ms = round((time.perf_counter() - t_inf_start) * 1000.0, 1)
                for d in detections:
                    d["latency_ms"] = latency_ms

                # Generate annotated frame for web monitor
                annotated_bytes = self._draw_annotations(cv_img, detections, img_size) if detections else None

                with self.lock:
                    self.latest_detections = detections
                    self.latest_latency_ms = latency_ms
                    if annotated_bytes:
                        self.latest_annotated_frame = annotated_bytes

            except Exception as exc:
                logger.warning("Inference loop error: %s", exc)
                time.sleep(0.02)

    def _draw_annotations(
        self,
        cv_img: np.ndarray,
        detections: List[Dict[str, Any]],
        img_size: Tuple[int, int],
    ) -> bytes:
        w, h = img_size
        annotated = cv_img.copy()

        for det in detections:
            bbox = det.get("bbox", [0, 0, 0, 0])
            cid = det.get("denomination_id", 0)
            conf = det.get("confidence", 0.0)
            nom_val, nom_name, color = AZN_NOMINALS.get(cid, (0, det.get("class_code", ""), (0, 255, 0)))

            x1 = max(0, min(w - 1, int(bbox[0] * w)))
            y1 = max(0, min(h - 1, int(bbox[1] * h)))
            x2 = max(0, min(w - 1, int(bbox[2] * w)))
            y2 = max(0, min(h - 1, int(bbox[3] * h)))

            # Draw glowing bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Draw corner lock brackets
            cw = min(22, max(10, int((x2 - x1) * 0.15)))
            ch = min(22, max(10, int((y2 - y1) * 0.15)))
            # Top-left
            cv2.line(annotated, (x1, y1), (x1 + cw, y1), (255, 255, 255), 3)
            cv2.line(annotated, (x1, y1), (x1, y1 + ch), (255, 255, 255), 3)
            # Top-right
            cv2.line(annotated, (x2, y1), (x2 - cw, y1), (255, 255, 255), 3)
            cv2.line(annotated, (x2, y1), (x2, y1 + ch), (255, 255, 255), 3)
            # Bottom-left
            cv2.line(annotated, (x1, y2), (x1 + cw, y2), (255, 255, 255), 3)
            cv2.line(annotated, (x1, y2), (x1, y2 - ch), (255, 255, 255), 3)
            # Bottom-right
            cv2.line(annotated, (x2, y2), (x2 - cw, y2), (255, 255, 255), 3)
            cv2.line(annotated, (x2, y2), (x2, y2 - ch), (255, 255, 255), 3)

            label = f"{nom_name} {int(conf * 100)}% [LOCKED]"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(annotated, (x1, max(0, y1 - th - 8)), (x1 + tw + 8, max(th + 8, y1)), (0, 0, 0), -1)
            cv2.rectangle(annotated, (x1, max(0, y1 - th - 8)), (x1 + tw + 8, max(th + 8, y1)), color, 1)
            cv2.putText(annotated, label, (x1 + 4, max(th + 2, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        banner_text = f"ESP32-S3 | {ACTIVE_MODEL_ID} | {self.instant_fps:.1f} FPS | {self.latest_latency_ms:.1f}ms | Conf >= {int(self.conf_threshold*100)}%"
        cv2.putText(annotated, banner_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(annotated, banner_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1, cv2.LINE_AA)

        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return buf.tobytes()

    def get_detect_payload(self) -> Dict[str, Any]:
        with self.lock:
            if self.latest_b64_frame is None:
                return {
                    "status": "waiting_for_hardware",
                    "message": "ESP32-S3 OV2640 camera initializing...",
                    "detections": [],
                    "telemetry": self.get_telemetry(),
                    "model_id": ACTIVE_MODEL_ID,
                }
            return {
                "status": "success",
                "frame": self.latest_b64_frame,
                "detections": list(self.latest_detections),
                "model_id": ACTIVE_MODEL_ID,
                "latency_ms": self.latest_latency_ms,
                "telemetry": self.get_telemetry(),
                "image_size": list(self.latest_image_size),
            }

    def get_telemetry(self) -> Dict[str, Any]:
        if self.bridge and self.bridge.is_connected():
            t = self.bridge.get_hardware_telemetry()
            t["instant_fps"] = round(self.instant_fps, 1)
            t["average_fps"] = round(self.average_fps, 1)
            t["active_model"] = ACTIVE_MODEL_ID
            t["connected"] = True
            t["orientation"] = self.get_orientation()
            return t
        return {
            "connected": False,
            "port": find_esp32s3_port() or "COM5",
            "thermal_state": "DISCONNECTED",
            "instant_fps": 0.0,
            "average_fps": 0.0,
            "active_model": ACTIVE_MODEL_ID,
            "orientation": self.get_orientation(),
        }


CAMERA_MANAGER = CameraManager()


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
        elif self.path in ("/view", "/web", "/monitor"):
            viewer_file = PATHS.artifacts_dir / "web_camera_viewer.html"
            if viewer_file.exists():
                content = viewer_file.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            else:
                self.send_error(404, "Viewer page not found")
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
        elif self.path == "/glasses/annotated_stream":
            self._handle_glasses_annotated_stream()
        elif self.path.startswith("/glasses/detect") or self.path.startswith("/glasses/live"):
            self._handle_glasses_detect()
        elif self.path == "/camera/orientation":
            self._send_json(CAMERA_MANAGER.get_orientation())
        elif self.path in ("/camera/reset_tracker", "/tracker/reset"):
            CAMERA_MANAGER.tracker.reset()
            self._send_json({"status": "ok", "message": "Tracker reset successfully"})
        else:
            self.send_error(404, "Endpoint not found")

    def _handle_glasses_detect(self) -> None:
        """Returns the freshest frame and GPU detections instantly from CameraManager."""
        parsed_url = urlparse(self.path)
        query_params = parse_qs(parsed_url.query)
        req_model = query_params.get("model", [None])[0]

        if req_model and req_model != ACTIVE_MODEL_ID:
            switch_active_model(req_model)

        payload = CAMERA_MANAGER.get_detect_payload()
        self._send_json(payload)

    def _handle_glasses_status(self) -> None:
        telemetry = CAMERA_MANAGER.get_telemetry()
        self._send_json(telemetry)

    def _handle_glasses_frame(self) -> None:
        raw_jpeg = CAMERA_MANAGER.latest_raw_frame
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
        raw_jpeg = CAMERA_MANAGER.latest_raw_frame
        if raw_jpeg:
            b64_frame = base64.b64encode(raw_jpeg).decode("utf-8")
            self._send_json({
                "status": "success",
                "frame": b64_frame,
                "telemetry": CAMERA_MANAGER.get_telemetry(),
            })
        else:
            self._send_json({"status": "error", "message": "Camera frame unavailable"}, status=503)

    def _handle_glasses_stream(self) -> None:
        """Proxies clean real-time MJPEG camera stream directly to web browsers and mobile clients."""
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        last_ts = 0.0
        try:
            while True:
                with CAMERA_MANAGER.new_frame_cond:
                    CAMERA_MANAGER.new_frame_cond.wait(timeout=0.25)
                    frame = CAMERA_MANAGER.latest_raw_frame
                    ts = CAMERA_MANAGER.last_frame_time

                if frame and ts != last_ts:
                    last_ts = ts
                    part = (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        + f"Content-Length: {len(frame)}\r\n\r\n".encode("latin1")
                        + frame
                        + b"\r\n"
                    )
                    self.wfile.write(part)
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def _handle_glasses_annotated_stream(self) -> None:
        """Proxies real-time annotated MJPEG camera stream directly to web browsers and mobile clients."""
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        last_ts = 0.0
        try:
            while True:
                with CAMERA_MANAGER.new_frame_cond:
                    CAMERA_MANAGER.new_frame_cond.wait(timeout=0.25)
                    frame = CAMERA_MANAGER.latest_annotated_frame
                    ts = CAMERA_MANAGER.last_frame_time

                if frame and ts != last_ts:
                    last_ts = ts
                    part = (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        + f"Content-Length: {len(frame)}\r\n\r\n".encode("latin1")
                        + frame
                        + b"\r\n"
                    )
                    self.wfile.write(part)
                    self.wfile.flush()
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
        elif self.path == "/camera/orientation":
            self._handle_set_orientation()
        elif self.path in ("/camera/reset_tracker", "/tracker/reset"):
            CAMERA_MANAGER.tracker.reset()
            self._send_json({"status": "ok", "message": "Tracker reset successfully"})
        else:
            self.send_error(404, "Endpoint not found")

    def _handle_set_orientation(self) -> None:
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"
        try:
            payload = json.loads(body) if body else {}
            new_cfg = CAMERA_MANAGER.set_orientation(
                unmirror=payload.get("unmirror"),
                vflip=payload.get("vflip"),
                rotation=payload.get("rotation"),
                conf_threshold=payload.get("conf_threshold") or payload.get("conf"),
            )
            self._send_json({"status": "ok", "orientation": new_cfg})
        except Exception as exc:
            self._send_json({"status": "error", "message": str(exc)}, status=400)

    def _handle_connect_camera(self) -> None:
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")
        try:
            payload = json.loads(body)
            url = payload.get("url") or payload.get("port")
            if url:
                ok = CAMERA_MANAGER.connect_to(url)
                self._send_json({
                    "status": "ok" if ok else "error",
                    "message": f"Connected to camera at {url}" if ok else f"Failed connecting to {url}",
                    "telemetry": CAMERA_MANAGER.get_telemetry(),
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
            img_b64 = payload.get("image") or payload.get("frame", "")

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
                    w, h = img.size
                    results = target_model.predict(img, conf=0.25, device=dev, verbose=False)
                    for res in results:
                        boxes = res.boxes
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
                            if not is_valid_phone_banknote_geometry(norm_box, w, h):
                                continue
                            class_code = CLASS_ID_MAP.get(cls_id, f"class_{cls_id}")
                            nom_val, nom_name, _ = AZN_NOMINALS.get(cls_id, (0, class_code, (0, 255, 0)))
                            detections.append({
                                "denomination_id": cls_id,
                                "nominal_value": nom_val,
                                "name": nom_name,
                                "class_code": class_code,
                                "confidence": round(conf, 3),
                                "bbox": norm_box,
                                "model_id": target_id,
                                "is_locked": True,
                            })

                    # If nothing detected at 0 degrees, test 90-degree rotation (common in phone portrait sensors)
                    if not detections:
                        rot_img = img.rotate(90, expand=True)
                        rw, rh = rot_img.size
                        rot_results = target_model.predict(rot_img, conf=0.25, device=dev, verbose=False)
                        for res in rot_results:
                            for box in res.boxes:
                                cls_id = int(box.cls[0].item())
                                conf = float(box.conf[0].item())
                                rxyxy = box.xyxy[0].tolist()
                                orig_x1 = max(0.0, min(1.0, 1.0 - rxyxy[3] / float(rh)))
                                orig_y1 = max(0.0, min(1.0, rxyxy[0] / float(rw)))
                                orig_x2 = max(0.0, min(1.0, 1.0 - rxyxy[1] / float(rh)))
                                orig_y2 = max(0.0, min(1.0, rxyxy[2] / float(rw)))
                                norm_box = [
                                    round(min(orig_x1, orig_x2), 3),
                                    round(min(orig_y1, orig_y2), 3),
                                    round(max(orig_x1, orig_x2), 3),
                                    round(max(orig_y1, orig_y2), 3),
                                ]
                                if not is_valid_phone_banknote_geometry(norm_box, w, h):
                                    continue
                                class_code = CLASS_ID_MAP.get(cls_id, f"class_{cls_id}")
                                nom_val, nom_name, _ = AZN_NOMINALS.get(cls_id, (0, class_code, (0, 255, 0)))
                                detections.append({
                                    "denomination_id": cls_id,
                                    "nominal_value": nom_val,
                                    "name": nom_name,
                                    "class_code": class_code,
                                    "confidence": round(conf, 3),
                                    "bbox": norm_box,
                                    "model_id": target_id,
                                    "is_locked": True,
                                })

                    # If still nothing detected, test 270-degree rotation
                    if not detections:
                        rot270 = img.rotate(270, expand=True)
                        r270w, r270h = rot270.size
                        rot270_res = target_model.predict(rot270, conf=0.25, device=dev, verbose=False)
                        for res in rot270_res:
                            for box in res.boxes:
                                cls_id = int(box.cls[0].item())
                                conf = float(box.conf[0].item())
                                rxyxy = box.xyxy[0].tolist()
                                orig_x1 = max(0.0, min(1.0, rxyxy[1] / float(r270h)))
                                orig_y1 = max(0.0, min(1.0, 1.0 - rxyxy[2] / float(r270w)))
                                orig_x2 = max(0.0, min(1.0, rxyxy[3] / float(r270h)))
                                orig_y2 = max(0.0, min(1.0, 1.0 - rxyxy[0] / float(r270w)))
                                norm_box = [
                                    round(min(orig_x1, orig_x2), 3),
                                    round(min(orig_y1, orig_y2), 3),
                                    round(max(orig_x1, orig_x2), 3),
                                    round(max(orig_y1, orig_y2), 3),
                                ]
                                if not is_valid_phone_banknote_geometry(norm_box, w, h):
                                    continue
                                class_code = CLASS_ID_MAP.get(cls_id, f"class_{cls_id}")
                                nom_val, nom_name, _ = AZN_NOMINALS.get(cls_id, (0, class_code, (0, 255, 0)))
                                detections.append({
                                    "denomination_id": cls_id,
                                    "nominal_value": nom_val,
                                    "name": nom_name,
                                    "class_code": class_code,
                                    "confidence": round(conf, 3),
                                    "bbox": norm_box,
                                    "model_id": target_id,
                                    "is_locked": True,
                                })

                    # If still nothing detected, test 180-degree rotation (upside-down phone)
                    if not detections:
                        rot180 = img.rotate(180)
                        rot180_res = target_model.predict(rot180, conf=0.25, device=dev, verbose=False)
                        for res in rot180_res:
                            for box in res.boxes:
                                cls_id = int(box.cls[0].item())
                                conf = float(box.conf[0].item())
                                rxyxy = box.xyxy[0].tolist()
                                orig_x1 = max(0.0, min(1.0, 1.0 - rxyxy[2] / float(w)))
                                orig_y1 = max(0.0, min(1.0, 1.0 - rxyxy[3] / float(h)))
                                orig_x2 = max(0.0, min(1.0, 1.0 - rxyxy[0] / float(w)))
                                orig_y2 = max(0.0, min(1.0, 1.0 - rxyxy[1] / float(h)))
                                norm_box = [
                                    round(min(orig_x1, orig_x2), 3),
                                    round(min(orig_y1, orig_y2), 3),
                                    round(max(orig_x1, orig_x2), 3),
                                    round(max(orig_y1, orig_y2), 3),
                                ]
                                if not is_valid_phone_banknote_geometry(norm_box, w, h):
                                    continue
                                class_code = CLASS_ID_MAP.get(cls_id, f"class_{cls_id}")
                                nom_val, nom_name, _ = AZN_NOMINALS.get(cls_id, (0, class_code, (0, 255, 0)))
                                detections.append({
                                    "denomination_id": cls_id,
                                    "nominal_value": nom_val,
                                    "name": nom_name,
                                    "class_code": class_code,
                                    "confidence": round(conf, 3),
                                    "bbox": norm_box,
                                    "model_id": target_id,
                                    "is_locked": True,
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
    CAMERA_MANAGER.start()
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
        CAMERA_MANAGER.stop()
        httpd.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mobile Vision Inference Bridge")
    parser.add_argument("--host", default=BRIDGE_CFG.default_host, help=f"Host binding (default {BRIDGE_CFG.default_host})")
    parser.add_argument("--port", type=int, default=BRIDGE_CFG.default_port, help=f"Port to bind (default {BRIDGE_CFG.default_port})")
    parser.add_argument("--model", default="auto", help="Model to load: auto, yolo11m, yolov8m, yolo_fastestv2")
    args = parser.parse_args()
    run_bridge(host=args.host, port=args.port, model=args.model)
