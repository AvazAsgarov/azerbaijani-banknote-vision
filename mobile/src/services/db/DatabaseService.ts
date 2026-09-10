/**
 * Relational SQLite Database Service for Banknote Scans & Telemetry.
 *
 * Implements persistent storage for:
 * - Scanned banknote history (denomination, nominal value, confidence, bbox, model, latency, timestamp, image URI)
 * - Persistent wallet balance and audit milestones
 * - Querying, aggregation, and breakdown statistics
 */

import { getDenominationById } from "../../constants/denominations";
import { ExpoSQLiteAdapter, IDatabaseAdapter, InMemoryDatabaseAdapter } from "./DatabaseAdapters";
import { globalImageStorageService, ImageStorageService } from "./ImageStorageService";

export interface ScanRecord {
  id: string;
  denominationId: number;
  nominalValue: number;
  name: string;
  confidence: number;
  source: "PHONE_CAM" | "GLASSES_BLE" | "SAMPLE";
  modelId: string;
  latencyMs: number;
  bbox: [number, number, number, number];
  imageUri: string | null;
  timestamp: number;
}

export interface ScanStats {
  totalCount: number;
  totalAmount: number;
  breakdown: Record<number, number>; // denominationId -> count
}

export class DatabaseService {
  private adapter: IDatabaseAdapter;
  private imageStorage: ImageStorageService;
  private isInitialized: boolean = false;
  private listeners: Array<() => void> = [];

  constructor(adapter?: IDatabaseAdapter, imageStorage?: ImageStorageService) {
    if (adapter) {
      this.adapter = adapter;
    } else {
      const isTestEnv = typeof process !== "undefined" && (process.env.NODE_ENV === "test" || process.env.JEST_WORKER_ID !== undefined);
      const isBrowserEnv = typeof window !== "undefined" && typeof document !== "undefined" && !Boolean((globalThis as any).nativeCallSyncHook);
      this.adapter = isTestEnv || isBrowserEnv ? new InMemoryDatabaseAdapter() : new ExpoSQLiteAdapter();
    }
    this.imageStorage = imageStorage || globalImageStorageService;
  }

  /**
   * Initializes SQLite tables, constraints, and indices.
   */
  public async init(): Promise<void> {
    if (this.isInitialized) return;

    await this.adapter.init();

    await this.adapter.execAsync(`
      CREATE TABLE IF NOT EXISTS scan_records (
        id TEXT PRIMARY KEY,
        denomination_id INTEGER NOT NULL,
        nominal_value REAL NOT NULL,
        name TEXT NOT NULL,
        confidence REAL NOT NULL,
        source TEXT NOT NULL,
        model_id TEXT NOT NULL,
        latency_ms REAL NOT NULL,
        bbox_json TEXT NOT NULL,
        image_uri TEXT,
        timestamp INTEGER NOT NULL
      );
    `);

    await this.adapter.execAsync(`
      CREATE INDEX IF NOT EXISTS idx_scan_records_timestamp ON scan_records (timestamp DESC);
      CREATE INDEX IF NOT EXISTS idx_scan_records_denom ON scan_records (denomination_id);
    `);

    await this.adapter.execAsync(`
      CREATE TABLE IF NOT EXISTS wallet_meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
      );
    `);

    this.isInitialized = true;
  }

  public subscribe(callback: () => void): () => void {
    this.listeners.push(callback);
    return () => {
      this.listeners = this.listeners.filter((cb) => cb !== callback);
    };
  }

  private notify(): void {
    for (const cb of this.listeners) {
      cb();
    }
  }

  /**
   * Inserts a newly verified banknote detection into SQLite.
   *
   * @param record Scan parameters
   * @param imageBase64 Optional frame Base64 JPEG data to save to filesystem
   * @returns Persisted ScanRecord
   */
  public async insertScan(
    record: Omit<ScanRecord, "id" | "imageUri">,
    imageBase64?: string
  ): Promise<ScanRecord> {
    await this.init();

    const id = `${record.timestamp || Date.now()}-${Math.random().toString(36).substring(2, 8)}`;
    const timestamp = record.timestamp || Date.now();
    const config = getDenominationById(record.denominationId);

    let savedImageUri: string | null = null;
    if (imageBase64) {
      const filename = `scan_${timestamp}_${record.denominationId}.jpg`;
      savedImageUri = await this.imageStorage.saveImage(filename, imageBase64);
    }

    const bboxJson = JSON.stringify(record.bbox);

    await this.adapter.runAsync(
      `INSERT INTO scan_records (
        id, denomination_id, nominal_value, name, confidence,
        source, model_id, latency_ms, bbox_json, image_uri, timestamp
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      [
        id,
        record.denominationId,
        config.nominalValue || record.nominalValue,
        config.name || record.name,
        record.confidence,
        record.source,
        record.modelId,
        record.latencyMs,
        bboxJson,
        savedImageUri,
        timestamp,
      ]
    );

    const saved: ScanRecord = {
      id,
      denominationId: record.denominationId,
      nominalValue: config.nominalValue || record.nominalValue,
      name: config.name || record.name,
      confidence: record.confidence,
      source: record.source,
      modelId: record.modelId,
      latencyMs: record.latencyMs,
      bbox: record.bbox,
      imageUri: savedImageUri,
      timestamp,
    };

    this.notify();
    return saved;
  }

  /**
   * Queries scan history with pagination and denomination filtering.
   */
  public async getScans(
    limit: number = 50,
    offset: number = 0,
    denominationFilter?: number
  ): Promise<ScanRecord[]> {
    await this.init();

    let sql = `SELECT * FROM scan_records`;
    const params: any[] = [];

    if (denominationFilter !== undefined && denominationFilter >= 0) {
      sql += ` WHERE denomination_id = ?`;
      params.push(denominationFilter);
    }

    sql += ` ORDER BY timestamp DESC LIMIT ? OFFSET ?`;
    params.push(limit, offset);

    const rows = await this.adapter.getAllAsync(sql, params);

    return rows.map((row: any) => ({
      id: row.id,
      denominationId: row.denomination_id,
      nominalValue: row.nominal_value,
      name: row.name,
      confidence: row.confidence,
      source: row.source,
      modelId: row.model_id,
      latencyMs: row.latency_ms,
      bbox: row.bbox_json ? JSON.parse(row.bbox_json) : [0, 0, 0, 0],
      imageUri: row.image_uri || null,
      timestamp: row.timestamp,
    }));
  }

  /**
   * Computes aggregate statistics across all recorded scans.
   */
  public async getStats(): Promise<ScanStats> {
    await this.init();

    const scans = await this.adapter.getAllAsync(
      `SELECT denomination_id, nominal_value FROM scan_records`
    );

    let totalAmount = 0;
    const breakdown: Record<number, number> = { 0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0 };

    for (const s of scans) {
      totalAmount += Number(s.nominal_value) || 0;
      const dId = Number(s.denomination_id);
      if (breakdown[dId] !== undefined) {
        breakdown[dId]++;
      } else {
        breakdown[dId] = 1;
      }
    }

    return {
      totalCount: scans.length,
      totalAmount,
      breakdown,
    };
  }

  /**
   * Resets and deletes all scan history and image files.
   */
  public async clearAllScans(): Promise<void> {
    await this.init();
    await this.adapter.runAsync(`DELETE FROM scan_records`);
    await this.imageStorage.clearAll();
    this.notify();
  }

  /**
   * Retrieves the persisted wallet balance from SQLite.
   */
  public async getWalletBalance(): Promise<number> {
    await this.init();
    const row = await this.adapter.getFirstAsync(
      `SELECT value FROM wallet_meta WHERE key = ?`,
      ["wallet_balance"]
    );
    return row ? Number(row.value) || 0 : 0;
  }

  /**
   * Persists the current wallet balance to SQLite.
   */
  public async setWalletBalance(balance: number): Promise<void> {
    await this.init();
    await this.adapter.runAsync(
      `INSERT OR REPLACE INTO wallet_meta (key, value) VALUES (?, ?)`,
      ["wallet_balance", balance.toString()]
    );
  }
}

export const globalDatabaseService = new DatabaseService();
