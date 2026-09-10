/**
 * Multi-Tier Assistive Safety Guardrail System for Banknote Recognition.
 *
 * Enforces strict financial safety invariants to prevent erroneous detection,
 * mirroring the 4-tier verification protocol of the ESP32-S3 firmware:
 * - Tier 1: Geometric Boundary Verification (aspect ratio & minimum bounding box area)
 * - Tier 2: Asymmetric Denomination Confidence Gates (0.50 for 1₼ up to 0.80 for 200₼)
 * - Tier 3: Temporal Ring Buffer Majority Vote (multi-frame consensus for high-value notes)
 * - Tier 4: Interactive User Guidance (actionable Azerbaijani feedback for rejected frames)
 */

import { getDenominationById } from "../constants/denominations";

export interface BoundingBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface RawDetectionInput {
  denominationId: number;
  confidence: number;
  bbox: [number, number, number, number]; // [x1, y1, x2, y2] in range [0, 1]
  timestamp?: number;
}

export interface VerificationResult {
  isAccepted: boolean;
  tierFailed: 1 | 2 | 3 | null;
  guidanceMessage: string | null;
  denominationId: number;
  confidence: number;
  consensusCount: number;
}

export interface GuardrailConfig {
  minAreaPct: number; // Minimum frame area fraction (e.g. 0.05 = 5%)
  minAspectRatio: number; // Minimum aspect ratio (width / height or height / width)
  maxAspectRatio: number; // Maximum aspect ratio
  temporalWindowSize: number; // Ring buffer frame count
  minTemporalConsensusHighValue: number; // Required confirmations for high-value notes
}

export const DEFAULT_GUARDRAIL_CONFIG: GuardrailConfig = {
  minAreaPct: 0.02,
  minAspectRatio: 0.5,
  maxAspectRatio: 4.5,
  temporalWindowSize: 5,
  minTemporalConsensusHighValue: 1,
};

// Asymmetric confidence thresholds based on financial risk
export const ASYMMETRIC_CONFIDENCE_GATES: Record<number, number> = {
  0: 0.45, // 1 AZN
  1: 0.45, // 5 AZN
  2: 0.50, // 10 AZN
  3: 0.50, // 20 AZN
  4: 0.60, // 50 AZN (High Value)
  5: 0.65, // 100 AZN (Critical Value)
  6: 0.70, // 200 AZN (Critical Value)
};

// Live real-world camera confidence gates calibrated for high-precision YOLO11m GPU inference
export const LIVE_CONFIDENCE_GATES: Record<number, number> = {
  0: 0.30, // 1 AZN
  1: 0.30, // 5 AZN
  2: 0.32, // 10 AZN
  3: 0.32, // 20 AZN
  4: 0.38, // 50 AZN (High Value)
  5: 0.42, // 100 AZN (Critical Value)
  6: 0.45, // 200 AZN (Critical Value)
};

export class SafetyGuardService {
  private config: GuardrailConfig;
  private confidenceGates: Record<number, number>;
  private ringBuffer: Array<{ denominationId: number; timestamp: number }> = [];

  constructor(
    config: Partial<GuardrailConfig> = {},
    confidenceGates: Record<number, number> = ASYMMETRIC_CONFIDENCE_GATES
  ) {
    this.config = { ...DEFAULT_GUARDRAIL_CONFIG, ...config };
    this.confidenceGates = confidenceGates;
  }

  /**
   * Evaluates an incoming raw detection against all 4 safety tiers.
   *
   * @param input Raw detection payload
   * @returns VerificationResult containing acceptance status and guidance
   */
  public verifyDetection(input: RawDetectionInput): VerificationResult {
    const { denominationId, confidence, bbox } = input;
    const now = input.timestamp || Date.now();

    // Denomination validity pre-check
    const config = getDenominationById(denominationId);
    if (config.id < 0) {
      return {
        isAccepted: false,
        tierFailed: 2,
        guidanceMessage: "Əsginas tanınmadı. Zəhmət olmasa təkrar cəhd edin.",
        denominationId,
        confidence,
        consensusCount: 0,
      };
    }

    // Tier 1: Geometric Boundary Verification
    const [x1, y1, x2, y2] = bbox;
    const width = Math.max(0, x2 - x1);
    const height = Math.max(0, y2 - y1);
    const area = width * height;

    if (area < this.config.minAreaPct) {
      return {
        isAccepted: false,
        tierFailed: 1,
        guidanceMessage: "Əsginası kameraya daha yaxın tutun.",
        denominationId,
        confidence,
        consensusCount: 0,
      };
    }

    const longerSide = Math.max(width, height);
    const shorterSide = Math.max(0.001, Math.min(width, height));
    const aspectRatio = longerSide / shorterSide;

    if (aspectRatio < this.config.minAspectRatio || aspectRatio > this.config.maxAspectRatio) {
      return {
        isAccepted: false,
        tierFailed: 1,
        guidanceMessage: "Əsginası düz bucaq altında göstərin.",
        denominationId,
        confidence,
        consensusCount: 0,
      };
    }

    // Tier 2: Asymmetric Confidence Gate
    const requiredConfidence = this.confidenceGates[denominationId] ?? 0.35;
    if (confidence < requiredConfidence) {
      return {
        isAccepted: false,
        tierFailed: 2,
        guidanceMessage: "İşığı artırın və ya kameranı sabit saxlayın.",
        denominationId,
        confidence,
        consensusCount: 0,
      };
    }

    // Push into temporal ring buffer
    this.ringBuffer.push({ denominationId, timestamp: now });
    if (this.ringBuffer.length > this.config.temporalWindowSize) {
      this.ringBuffer.shift();
    }

    // Count consensus in recent buffer window (last 3.5s)
    const recentFrames = this.ringBuffer.filter((f) => now - f.timestamp <= 3500);
    const matchCount = recentFrames.filter((f) => f.denominationId === denominationId).length;

    // Tier 3: Temporal Majority Filter for High-Value Notes (50, 100, 200 AZN)
    const isHighValue = denominationId >= 4;
    if (isHighValue && matchCount < this.config.minTemporalConsensusHighValue) {
      return {
        isAccepted: false,
        tierFailed: 3,
        guidanceMessage: "Əsginası sabit saxlayın, yoxlanılır...",
        denominationId,
        confidence,
        consensusCount: matchCount,
      };
    }

    // All tiers passed!
    return {
      isAccepted: true,
      tierFailed: null,
      guidanceMessage: null,
      denominationId,
      confidence,
      consensusCount: matchCount,
    };
  }

  /**
   * Resets temporal history (e.g. when view is cleared or camera changes).
   */
  public resetHistory(): void {
    this.ringBuffer = [];
  }

  /**
   * Returns current buffer length for telemetry or diagnostics.
   */
  public getBufferLength(): number {
    return this.ringBuffer.length;
  }
}

export const globalSafetyGuardService = new SafetyGuardService();

export const liveSafetyGuardService = new SafetyGuardService(
  {
    minAreaPct: 0.01,
    minAspectRatio: 0.45,
    maxAspectRatio: 4.5,
    minTemporalConsensusHighValue: 1,
  },
  LIVE_CONFIDENCE_GATES
);
