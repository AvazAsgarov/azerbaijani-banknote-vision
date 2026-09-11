/**
 * Unit Tests for 4-Tier SafetyGuardService.
 */

import { SafetyGuardService } from "../src/services/SafetyGuardService";

describe("SafetyGuardService", () => {
  let guard: SafetyGuardService;

  beforeEach(() => {
    guard = new SafetyGuardService();
  });

  test("accepts valid standard banknote detection with good geometry and confidence", () => {
    const result = guard.verifyDetection({
      denominationId: 2, // 10 AZN
      confidence: 0.92,
      bbox: [0.1, 0.3, 0.9, 0.7], // w=0.8, h=0.4 => AR=2.0, area=0.32
    });

    expect(result.isAccepted).toBe(true);
    expect(result.tierFailed).toBeNull();
    expect(result.guidanceMessage).toBeNull();
  });

  test("rejects detection with tiny area coverage (<5%) under Tier 1", () => {
    const result = guard.verifyDetection({
      denominationId: 2,
      confidence: 0.95,
      bbox: [0.1, 0.1, 0.2, 0.2], // w=0.1, h=0.1 => area=0.01 (<0.05)
    });

    expect(result.isAccepted).toBe(false);
    expect(result.tierFailed).toBe(1);
    expect(result.guidanceMessage).toContain("yaxın tutun");
  });

  test("rejects detection with abnormal aspect ratio under Tier 1", () => {
    const result = guard.verifyDetection({
      denominationId: 3, // 20 AZN
      confidence: 0.95,
      bbox: [0.1, 0.1, 0.6, 0.6], // w=0.5, h=0.5 => AR=1.0 (<1.25)
    });

    expect(result.isAccepted).toBe(false);
    expect(result.tierFailed).toBe(1);
    expect(result.guidanceMessage).toContain("düz bucaq");
  });

  test("enforces asymmetric confidence gates under Tier 2", () => {
    // 100 AZN requires >= 0.75 confidence
    const lowConfResult = guard.verifyDetection({
      denominationId: 5, // 100 AZN
      confidence: 0.70, // Below 0.75 threshold
      bbox: [0.1, 0.3, 0.9, 0.7],
    });

    expect(lowConfResult.isAccepted).toBe(false);
    expect(lowConfResult.tierFailed).toBe(2);
    expect(lowConfResult.guidanceMessage).toContain("İşığı artırın");

    // 10 AZN requires only >= 0.60; conf 0.65 passes Tier 2
    const standardPass = guard.verifyDetection({
      denominationId: 2, // 10 AZN
      confidence: 0.65,
      bbox: [0.1, 0.3, 0.9, 0.7],
    });

    expect(standardPass.isAccepted).toBe(true);
  });

  test("requires multi-frame consensus for high-value notes under Tier 3", () => {
    // Frame 1: 100 AZN with high confidence
    const frame1 = guard.verifyDetection({
      denominationId: 5, // 100 AZN
      confidence: 0.95,
      bbox: [0.1, 0.3, 0.9, 0.7],
      timestamp: 1000,
    });

    // High value note requires 2 consensus frames: frame 1 should wait
    expect(frame1.isAccepted).toBe(false);
    expect(frame1.tierFailed).toBe(3);
    expect(frame1.guidanceMessage).toContain("yoxlanılır");

    // Frame 2: 100 AZN confirmed in next frame
    const frame2 = guard.verifyDetection({
      denominationId: 5,
      confidence: 0.96,
      bbox: [0.1, 0.3, 0.9, 0.7],
      timestamp: 1500,
    });

    expect(frame2.isAccepted).toBe(true);
    expect(frame2.tierFailed).toBeNull();
    expect(frame2.consensusCount).toBeGreaterThanOrEqual(2);
  });

  test("resets temporal buffer cleanly", () => {
    guard.verifyDetection({
      denominationId: 5,
      confidence: 0.95,
      bbox: [0.1, 0.3, 0.9, 0.7],
    });
    expect(guard.getBufferLength()).toBe(1);

    guard.resetHistory();
    expect(guard.getBufferLength()).toBe(0);
  });
});
