/**
 * Unit Tests for VisionBridgeService.
 */

import { VisionBridgeService } from "../src/services/VisionBridgeService";

describe("VisionBridgeService", () => {
  let bridge: VisionBridgeService;

  beforeEach(() => {
    bridge = new VisionBridgeService({
      bridgeUrl: "http://127.0.0.1:8000/",
      isMockMode: true,
    });
  });

  test("normalizes bridge URL trimming trailing slashes", () => {
    bridge.setBridgeUrl("http://192.168.1.50:8000///");
    expect(bridge.getBridgeUrl()).toBe("http://192.168.1.50:8000");
  });

  test("generateMockDetection produces valid normalized bounding box", () => {
    const dets = bridge.generateMockDetection(4); // 50 AZN
    expect(dets).toHaveLength(1);
    const d = dets[0];
    expect(d.denominationId).toBe(4);
    expect(d.classCode).toBe("050_azn");
    expect(d.confidence).toBeGreaterThan(0.9);

    const [x1, y1, x2, y2] = d.bbox;
    expect(x1).toBeGreaterThanOrEqual(0);
    expect(y1).toBeGreaterThanOrEqual(0);
    expect(x2).toBeLessThanOrEqual(1);
    expect(y2).toBeLessThanOrEqual(1);
    expect(x2).toBeGreaterThan(x1);
    expect(y2).toBeGreaterThan(y1);
  });

  test("analyzeFrame in mock mode returns valid detection without network call", async () => {
    bridge.setMockMode(true);
    expect(bridge.isMock()).toBe(true);

    const results = await bridge.analyzeFrame("dummy_base64_data");
    expect(results).toHaveLength(1);
    expect(results[0].denominationId).toBe(2); // default 10 AZN
  });
});
