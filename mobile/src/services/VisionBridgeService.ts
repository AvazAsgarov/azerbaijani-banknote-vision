/**
 * Computer Vision Bridge Service for Phone Camera Inferences.
 *
 * Transmits camera frames to the local laptop inference bridge (running on RTX GPU)
 * or evaluates built-in sample banknotes for offline standalone testing.
 */

export interface BoundingBoxDetection {
  denominationId: number;
  classCode: string;
  confidence: number;
  bbox: [number, number, number, number]; // [x1, y1, x2, y2] normalized [0, 1]
  modelId: string;
  latencyMs: number;
  imageWidth?: number;
  imageHeight?: number;
}

export interface BridgeConfig {
  bridgeUrl: string; // e.g. "http://192.168.0.142:8000"
  timeoutMs: number;
  isMockMode: boolean;
}

const getInitialBridgeUrl = (): string => {
  // 1. Browser runtime: dynamically use the current server hostname
  if (typeof window !== "undefined" && window.location && window.location.hostname) {
    const host = window.location.hostname;
    if (host && host !== "localhost" && host !== "127.0.0.1") {
      return `http://${host}:8000`;
    }
  }

  // 2. React Native / Expo Go runtime
  try {
    // Dynamic require prevents syntax errors in Node/Jest test runners
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const expConst = require("expo-constants");
    const constants = expConst?.default || expConst;
    const hostUri =
      constants?.expoConfig?.hostUri ||
      constants?.manifest2?.extra?.expoGo?.debuggerHost ||
      constants?.linkingUri;

    if (hostUri) {
      const cleaned = hostUri.replace(/^[a-zA-Z]+:\/\//, "");
      const ip = cleaned.split(":")[0];
      if (ip && ip !== "localhost" && ip !== "127.0.0.1") {
        return `http://${ip}:8000`;
      }
    }
  } catch {
    // Pure node test fallback
  }

  // 3. Fallback: Environment variable or localhost
  return process.env.EXPO_PUBLIC_VISION_BRIDGE_URL || "http://127.0.0.1:8000";
};

export class VisionBridgeService {
  private config: BridgeConfig = {
    bridgeUrl: getInitialBridgeUrl(),
    timeoutMs: 5000,
    isMockMode: false,
  };

  constructor(config?: Partial<BridgeConfig>) {
    if (config) {
      this.config = { ...this.config, ...config };
    }
  }

  public setBridgeUrl(url: string): void {
    this.config.bridgeUrl = url.trim().replace(/\/+$/, "");
  }

  public getBridgeUrl(): string {
    return this.config.bridgeUrl;
  }

  public setMockMode(mock: boolean): void {
    this.config.isMockMode = mock;
  }

  public isMock(): boolean {
    return this.config.isMockMode;
  }

  private activeModelId: string = "yolo11m";

  public setActiveModelId(modelId: string): void {
    this.activeModelId = modelId;
  }

  public getActiveModelId(): string {
    return this.activeModelId;
  }

  /**
   * Switches the active inference model on the bridge server.
   */
  public async switchBridgeModel(modelId: string): Promise<boolean> {
    this.activeModelId = modelId;
    if (this.config.isMockMode) return true;

    try {
      const response = await fetch(`${this.config.bridgeUrl}/model`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_id: modelId }),
      });
      return response.ok;
    } catch {
      return false;
    }
  }

  /**
   * Fetches the list of available models from the inference bridge.
   */
  public async fetchBridgeModels(): Promise<any[]> {
    if (this.config.isMockMode) return [];
    try {
      const response = await fetch(`${this.config.bridgeUrl}/models`);
      if (response.ok) {
        const data = await response.json();
        return data.models || [];
      }
    } catch {}
    return [];
  }

  /**
   * Fetches live camera frame and real-time TinyML detection from connected ESP32-S3.
   */
  public async fetchGlassesLive(): Promise<{
    status: string;
    frame?: string;
    detections: BoundingBoxDetection[];
    modelId: string;
    latencyMs: number;
    telemetry?: {
      port?: string;
      connected?: boolean;
      instant_fps?: number;
      estimated_chip_temp_c?: number;
    };
  } | null> {
    if (this.config.isMockMode) {
      const mockDets = this.generateMockDetection(undefined, this.activeModelId);
      return {
        status: "mock",
        detections: mockDets,
        modelId: this.activeModelId,
        latencyMs: 42.0,
        telemetry: { port: "COM5 (SIMULATED)", connected: true, instant_fps: 12.0 },
      };
    }

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 2000);

      const response = await fetch(`${this.config.bridgeUrl}/glasses/detect`, {
        signal: controller.signal,
      });
      clearTimeout(timeoutId);

      if (!response.ok) return null;
      const data = await response.json();

      const detections: BoundingBoxDetection[] = (data.detections || []).map((det: any) => ({
        denominationId: det.denomination_id,
        classCode: det.class_code,
        confidence: det.confidence,
        bbox: det.bbox,
        modelId: det.model_id || this.activeModelId,
        latencyMs: det.latency_ms || 42.0,
      }));

      return {
        status: data.status,
        frame: data.frame,
        detections,
        modelId: data.model_id || this.activeModelId,
        latencyMs: data.latency_ms || 42.0,
        telemetry: data.telemetry,
      };
    } catch {
      return null;
    }
  }

  /**
   * Sends base64 image frame to bridge or evaluates sample.
   *
   * @param imageBase64 Base64-encoded JPEG image string
   * @param requestedModelId Optional specific model to run ("yolo_fastestv2" | "yolo11m")
   * @returns Array of BoundingBoxDetection
   */
  public async analyzeFrame(imageBase64: string, requestedModelId?: string): Promise<BoundingBoxDetection[]> {
    const targetModel = requestedModelId || this.activeModelId;
    if (this.config.isMockMode) {
      return this.generateMockDetection(undefined, targetModel);
    }

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.config.timeoutMs);

      const response = await fetch(`${this.config.bridgeUrl}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: imageBase64, model_id: targetModel }),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        return [];
      }

      const data = await response.json();
      const imageSize = Array.isArray(data.image_size) && data.image_size.length === 2 ? data.image_size : [640, 640];
      return (data.detections || []).map((det: any) => ({
        denominationId: det.denomination_id,
        classCode: det.class_code,
        confidence: det.confidence,
        bbox: det.bbox,
        modelId: det.model_id || targetModel,
        latencyMs: det.latency_ms || (targetModel === "yolo_fastestv2" ? 42.0 : 5.2),
        imageWidth: imageSize[0],
        imageHeight: imageSize[1],
      }));
    } catch {
      // Fallback gracefully on network error
      return [];
    }
  }

  /**
   * Generates a sample detection for instant zero-hardware testing.
   */
  public generateMockDetection(specificDenominationId?: number, modelId: string = "yolo_fastestv2"): BoundingBoxDetection[] {
    const denomId = specificDenominationId !== undefined ? specificDenominationId : 2; // Default 10 AZN
    const classCodes = ["001_azn", "005_azn", "010_azn", "020_azn", "050_azn", "100_azn", "200_azn"];

    return [
      {
        denominationId: denomId,
        classCode: classCodes[denomId] || "010_azn",
        confidence: 0.97,
        bbox: [0.15, 0.28, 0.85, 0.72],
        modelId,
        latencyMs: modelId === "yolo_fastestv2" ? 42.0 : 5.2,
      },
    ];
  }

  /**
   * Generates multiple sample detections (e.g. 5 AZN and 10 AZN side-by-side)
   * for testing assistive multi-banknote sum recognition and UI overlays.
   */
  public generateMockMultiDetection(
    denominationIds: number[] = [1, 2], // Default 5 AZN + 10 AZN -> 15 AZN
    modelId: string = "yolo_fastestv2"
  ): BoundingBoxDetection[] {
    const classCodes = ["001_azn", "005_azn", "010_azn", "020_azn", "050_azn", "100_azn", "200_azn"];
    const count = denominationIds.length;
    return denominationIds.map((denomId, index) => {
      const xSpan = 0.85 / Math.max(1, count);
      const x1 = 0.08 + index * xSpan;
      const x2 = Math.min(0.95, x1 + xSpan * 0.9);
      return {
        denominationId: denomId,
        classCode: classCodes[denomId] || "010_azn",
        confidence: 0.96 - index * 0.02,
        bbox: [x1, 0.25, x2, 0.75],
        modelId,
        latencyMs: modelId === "yolo_fastestv2" ? 42.0 : 5.2,
      };
    });
  }
}

export const globalVisionBridge = new VisionBridgeService();
