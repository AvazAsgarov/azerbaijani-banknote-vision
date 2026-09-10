/**
 * Model Architecture Registry & Dynamic Selection.
 *
 * Decoupled from hardcoded assumptions: supports whatever architecture
 * is currently trained or designated as champion in experiments.
 */

export interface ModelDescriptor {
  id: string;
  displayName: string;
  architectureType: "Spatial Attention CNN" | "Anchor-Free CNN" | "Vision Transformer" | "Mobile INT8 Edge";
  inputResolution: [number, number];
  typicalLatencyMs: number;
  isEdgeMicrocontroller: boolean;
  modelWeightFormat: string;
}

export const SUPPORTED_MODELS: Record<string, ModelDescriptor> = {
  champion_auto: {
    id: "champion_auto",
    displayName: "Auto-Negotiated Champion",
    architectureType: "Spatial Attention CNN",
    inputResolution: [640, 640],
    typicalLatencyMs: 4.8,
    isEdgeMicrocontroller: false,
    modelWeightFormat: "PyTorch .pt / ONNX",
  },
  yolo11m: {
    id: "yolo11m",
    displayName: "YOLOv11m (C2PSA Attention)",
    architectureType: "Spatial Attention CNN",
    inputResolution: [640, 640],
    typicalLatencyMs: 5.2,
    isEdgeMicrocontroller: false,
    modelWeightFormat: "PyTorch .pt",
  },
  yolov8m: {
    id: "yolov8m",
    displayName: "YOLOv8m (Anchor-Free Baseline)",
    architectureType: "Anchor-Free CNN",
    inputResolution: [640, 640],
    typicalLatencyMs: 6.4,
    isEdgeMicrocontroller: false,
    modelWeightFormat: "PyTorch .pt",
  },
  rtdetr_l: {
    id: "rtdetr_l",
    displayName: "RT-DETR-L (Vision Transformer)",
    architectureType: "Vision Transformer",
    inputResolution: [640, 640],
    typicalLatencyMs: 14.1,
    isEdgeMicrocontroller: false,
    modelWeightFormat: "PyTorch .pt",
  },
  yolo_fastestv2: {
    id: "yolo_fastestv2",
    displayName: "YOLO-FastestV2 INT8 (Smart Glasses)",
    architectureType: "Mobile INT8 Edge",
    inputResolution: [160, 160],
    typicalLatencyMs: 42.0,
    isEdgeMicrocontroller: true,
    modelWeightFormat: "INT8 C++ Header / TFLM",
  },
};

export const DEFAULT_MODEL_ID = "champion_auto";
