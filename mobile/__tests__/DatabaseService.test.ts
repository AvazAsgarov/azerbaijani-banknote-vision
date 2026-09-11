/**
 * Unit Tests for Relational DatabaseService (SQLite & In-Memory Adapter).
 */

import { InMemoryDatabaseAdapter } from "../src/services/db/DatabaseAdapters";
import { DatabaseService } from "../src/services/db/DatabaseService";

describe("DatabaseService", () => {
  let dbService: DatabaseService;
  let adapter: InMemoryDatabaseAdapter;

  beforeEach(async () => {
    adapter = new InMemoryDatabaseAdapter();
    dbService = new DatabaseService(adapter);
    await dbService.init();
  });

  test("initializes cleanly with empty scans and zero balance", async () => {
    const scans = await dbService.getScans();
    expect(scans).toHaveLength(0);

    const balance = await dbService.getWalletBalance();
    expect(balance).toBe(0);

    const stats = await dbService.getStats();
    expect(stats.totalCount).toBe(0);
    expect(stats.totalAmount).toBe(0);
  });

  test("inserts scan record with complete metadata and coordinates", async () => {
    const inserted = await dbService.insertScan({
      denominationId: 2, // 10 AZN
      nominalValue: 10,
      name: "10 Manat",
      confidence: 0.96,
      source: "PHONE_CAM",
      modelId: "yolo11m",
      latencyMs: 5.2,
      bbox: [0.1, 0.2, 0.8, 0.7],
      timestamp: 1720000000000,
    });

    expect(inserted.id).toBeDefined();
    expect(inserted.denominationId).toBe(2);
    expect(inserted.nominalValue).toBe(10);
    expect(inserted.confidence).toBe(0.96);
    expect(inserted.bbox).toEqual([0.1, 0.2, 0.8, 0.7]);

    const scans = await dbService.getScans();
    expect(scans).toHaveLength(1);
    expect(scans[0].name).toBe("10 Manat");
  });

  test("calculates aggregation stats and breakdown across multiple scans", async () => {
    await dbService.insertScan({
      denominationId: 0, // 1 AZN
      nominalValue: 1,
      name: "1 Manat",
      confidence: 0.92,
      source: "SAMPLE",
      modelId: "yolo11m",
      latencyMs: 4.8,
      bbox: [0.1, 0.1, 0.9, 0.9],
      timestamp: 1000,
    });

    await dbService.insertScan({
      denominationId: 4, // 50 AZN
      nominalValue: 50,
      name: "50 Manat",
      confidence: 0.98,
      source: "GLASSES_BLE",
      modelId: "yolo_fastestv2",
      latencyMs: 42.0,
      bbox: [0.2, 0.2, 0.8, 0.8],
      timestamp: 2000,
    });

    await dbService.insertScan({
      denominationId: 5, // 100 AZN
      nominalValue: 100,
      name: "100 Manat",
      confidence: 0.99,
      source: "PHONE_CAM",
      modelId: "yolo11m",
      latencyMs: 5.1,
      bbox: [0.15, 0.15, 0.85, 0.85],
      timestamp: 3000,
    });

    const stats = await dbService.getStats();
    expect(stats.totalCount).toBe(3);
    expect(stats.totalAmount).toBe(151);
    expect(stats.breakdown[0]).toBe(1);
    expect(stats.breakdown[4]).toBe(1);
    expect(stats.breakdown[5]).toBe(1);
  });

  test("filters scans by denomination correctly", async () => {
    await dbService.insertScan({
      denominationId: 1, // 5 AZN
      nominalValue: 5,
      name: "5 Manat",
      confidence: 0.91,
      source: "SAMPLE",
      modelId: "yolo11m",
      latencyMs: 5.0,
      bbox: [0.1, 0.1, 0.8, 0.8],
      timestamp: 1000,
    });

    await dbService.insertScan({
      denominationId: 2, // 10 AZN
      nominalValue: 10,
      name: "10 Manat",
      confidence: 0.94,
      source: "PHONE_CAM",
      modelId: "yolo11m",
      latencyMs: 5.2,
      bbox: [0.1, 0.1, 0.8, 0.8],
      timestamp: 2000,
    });

    const allScans = await dbService.getScans();
    expect(allScans).toHaveLength(2);

    const filtered5 = await dbService.getScans(50, 0, 1);
    expect(filtered5).toHaveLength(1);
    expect(filtered5[0].denominationId).toBe(1);

    const filtered10 = await dbService.getScans(50, 0, 2);
    expect(filtered10).toHaveLength(1);
    expect(filtered10[0].denominationId).toBe(2);
  });

  test("persists and updates wallet balance accurately", async () => {
    await dbService.setWalletBalance(260);
    const balance = await dbService.getWalletBalance();
    expect(balance).toBe(260);

    await dbService.setWalletBalance(0);
    expect(await dbService.getWalletBalance()).toBe(0);
  });

  test("clears all scans cleanly", async () => {
    await dbService.insertScan({
      denominationId: 6, // 200 AZN
      nominalValue: 200,
      name: "200 Manat",
      confidence: 0.99,
      source: "PHONE_CAM",
      modelId: "yolo11m",
      latencyMs: 5.0,
      bbox: [0.1, 0.1, 0.9, 0.9],
      timestamp: 1000,
    });

    expect(await dbService.getScans()).toHaveLength(1);
    await dbService.clearAllScans();
    expect(await dbService.getScans()).toHaveLength(0);
  });

  test("notifies subscribers upon scan changes", async () => {
    const subscriber = jest.fn();
    const unsub = dbService.subscribe(subscriber);

    await dbService.insertScan({
      denominationId: 3, // 20 AZN
      nominalValue: 20,
      name: "20 Manat",
      confidence: 0.95,
      source: "PHONE_CAM",
      modelId: "yolo11m",
      latencyMs: 5.0,
      bbox: [0.1, 0.1, 0.8, 0.8],
      timestamp: 1000,
    });

    expect(subscriber).toHaveBeenCalledTimes(1);

    unsub();
    await dbService.clearAllScans();
    expect(subscriber).toHaveBeenCalledTimes(1); // No call after unsub
  });
});
