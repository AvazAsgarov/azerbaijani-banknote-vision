/**
 * Unit Tests for BLEService (ESP32-S3 Binary Packet Protocol).
 */

import { BLEService } from "../src/services/BLEService";

describe("BLEService", () => {
  let bleService: BLEService;

  beforeEach(() => {
    bleService = new BLEService();
  });

  test("parses valid 8-byte packet correctly", () => {
    // Construct packet for 50 AZN (id=4), 95% conf, bbox=[0.10, 0.20, 0.80, 0.90]
    const packet = new Uint8Array(8);
    packet[0] = 0xAA; // Magic
    packet[1] = 4;    // 50 AZN
    packet[2] = 95;   // 95%
    packet[3] = 10;   // x1
    packet[4] = 20;   // y1
    packet[5] = 80;   // x2
    packet[6] = 90;   // y2

    // Checksum: XOR sum 0..6
    let chk = 0;
    for (let i = 0; i < 7; i++) {
      chk ^= packet[i];
    }
    packet[7] = chk;

    const result = bleService.parsePacket(packet);
    expect(result).not.toBeNull();
    expect(result?.denominationId).toBe(4);
    expect(result?.confidence).toBeCloseTo(0.95, 2);
    expect(result?.bbox).toEqual([0.1, 0.2, 0.8, 0.9]);
    expect(result?.batteryPct).toBe(88);
  });

  test("rejects truncated packets with less than 8 bytes", () => {
    const shortPacket = new Uint8Array([0xAA, 1, 90, 10, 20]);
    expect(bleService.parsePacket(shortPacket)).toBeNull();
  });

  test("rejects packets with incorrect magic byte", () => {
    const badMagic = new Uint8Array([0xBB, 2, 90, 10, 20, 80, 90, 0]);
    expect(bleService.parsePacket(badMagic)).toBeNull();
  });

  test("rejects packets with invalid checksum", () => {
    const corrupted = new Uint8Array([0xAA, 2, 90, 10, 20, 80, 90, 0xFF]);
    expect(bleService.parsePacket(corrupted)).toBeNull();
  });

  test("rejects out-of-range denomination IDs", () => {
    const outOfRange = new Uint8Array([0xAA, 9, 90, 10, 20, 80, 90, 0]);
    // Compute proper checksum
    let chk = 0;
    for (let i = 0; i < 7; i++) chk ^= outOfRange[i];
    outOfRange[7] = chk;

    expect(bleService.parsePacket(outOfRange)).toBeNull();
  });
});
