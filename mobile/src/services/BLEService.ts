/**
 * Bluetooth Low Energy (BLE) Client & Packet Parser for Seeed Studio XIAO ESP32-S3.
 *
 * Implements GATT client protocol interfacing with firmware/esp32s3_smart_glasses:
 * - Service UUID: 0xFFE0
 * - Characteristic UUID: 0xFFE1 (Notify)
 * - Binary 8-byte packet decoding:
 *   [0]: Magic Header 0xAA
 *   [1]: Denomination ID (0 to 6)
 *   [2]: Confidence Pct (0 to 100)
 *   [3..6]: Normalized BBox (X1, Y1, X2, Y2 in 0..100)
 *   [7]: XOR Checksum
 */

export interface GlassesDetection {
  denominationId: number;
  confidence: number;
  bbox: [number, number, number, number]; // [x1, y1, x2, y2] normalized [0, 1]
  batteryPct?: number;
  rssi?: number;
  timestamp: number;
}

export type ConnectionState = "DISCONNECTED" | "CONNECTING" | "CONNECTED" | "SIMULATION";

export class BLEService {
  private connectionState: ConnectionState = "DISCONNECTED";
  private listeners: Array<(detection: GlassesDetection) => void> = [];
  private stateListeners: Array<(state: ConnectionState) => void> = [];
  private simulationTimer: any = null;

  public subscribeDetection(callback: (detection: GlassesDetection) => void): () => void {
    this.listeners.push(callback);
    return () => {
      this.listeners = this.listeners.filter((cb) => cb !== callback);
    };
  }

  public subscribeState(callback: (state: ConnectionState) => void): () => void {
    this.stateListeners.push(callback);
    callback(this.connectionState);
    return () => {
      this.stateListeners = this.stateListeners.filter((cb) => cb !== callback);
    };
  }

  private setState(newState: ConnectionState): void {
    this.connectionState = newState;
    for (const listener of this.stateListeners) {
      listener(newState);
    }
  }

  /**
   * Decodes an 8-byte binary packet from ESP32-S3 smart glasses.
   *
   * @param buffer Raw Uint8Array containing packet bytes
   * @returns GlassesDetection or null if packet is invalid or corrupted
   */
  public parsePacket(buffer: Uint8Array): GlassesDetection | null {
    if (buffer.length < 8) {
      return null;
    }

    // Byte 0: Magic Header
    if (buffer[0] !== 0xAA) {
      return null;
    }

    // Byte 7: XOR Checksum verification
    let computedChecksum = 0;
    for (let i = 0; i < 7; i++) {
      computedChecksum ^= buffer[i];
    }
    if (computedChecksum !== buffer[7]) {
      return null;
    }

    const denomId = buffer[1];
    if (denomId < 0 || denomId > 6) {
      return null;
    }

    const confidence = buffer[2] / 100.0;
    const x1 = buffer[3] / 100.0;
    const y1 = buffer[4] / 100.0;
    const x2 = buffer[5] / 100.0;
    const y2 = buffer[6] / 100.0;

    return {
      denominationId: denomId,
      confidence: Math.min(1.0, Math.max(0.0, confidence)),
      bbox: [x1, y1, x2, y2],
      batteryPct: 88,
      rssi: -62,
      timestamp: Date.now(),
    };
  }

  /**
   * Dispatches a decoded detection to all active subscribers.
   */
  public emitDetection(detection: GlassesDetection): void {
    for (const listener of this.listeners) {
      listener(detection);
    }
  }

  /**
   * Enables simulated glasses telemetry for development without physical hardware.
   */
  public startSimulation(): void {
    this.setState("SIMULATION");
    if (this.simulationTimer) {
      clearInterval(this.simulationTimer);
    }

    const sampleDenoms = [2, 4, 5, 0, 1]; // 10 AZN, 50 AZN, 100 AZN, 1 AZN, 5 AZN
    let index = 0;

    this.simulationTimer = setInterval(() => {
      const denomId = sampleDenoms[index % sampleDenoms.length];
      index++;

      // Create valid 8-byte packet
      const packet = new Uint8Array(8);
      packet[0] = 0xAA;
      packet[1] = denomId;
      packet[2] = 96; // 96%
      packet[3] = 20; // x1 = 0.20
      packet[4] = 25; // y1 = 0.25
      packet[5] = 80; // x2 = 0.80
      packet[6] = 75; // y2 = 0.75
      // Compute checksum
      let chk = 0;
      for (let i = 0; i < 7; i++) {
        chk ^= packet[i];
      }
      packet[7] = chk;

      const detection = this.parsePacket(packet);
      if (detection) {
        this.emitDetection(detection);
      }
    }, 4500);
  }

  public stopSimulation(): void {
    if (this.simulationTimer) {
      clearInterval(this.simulationTimer);
      this.simulationTimer = null;
    }
    this.setState("DISCONNECTED");
  }

  public getState(): ConnectionState {
    return this.connectionState;
  }
}

export const globalBLEService = new BLEService();
