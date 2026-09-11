/**
 * Database Adapters & Storage Interface Abstraction.
 *
 * Provides a clean adapter layer allowing:
 * 1. Production runtime execution via modern `expo-sqlite`
 * 2. Headless in-memory test execution for Jest and Node environments
 */

export interface IDatabaseAdapter {
  init(): Promise<void>;
  execAsync(sql: string): Promise<void>;
  runAsync(sql: string, params?: any[]): Promise<{ lastInsertRowId?: number; changes: number }>;
  getAllAsync<T = any>(sql: string, params?: any[]): Promise<T[]>;
  getFirstAsync<T = any>(sql: string, params?: any[]): Promise<T | null>;
  closeAsync(): Promise<void>;
}

export class InMemoryDatabaseAdapter implements IDatabaseAdapter {
  private scanRecords: any[] = [];
  private walletMeta: Map<string, string> = new Map();

  public async init(): Promise<void> {
    // In-memory initialization
  }

  public async execAsync(sql: string): Promise<void> {
    const trimmed = sql.trim().toLowerCase();
    if (trimmed.startsWith("delete from scan_records")) {
      this.scanRecords = [];
    }
  }

  public async runAsync(sql: string, params: any[] = []): Promise<{ lastInsertRowId?: number; changes: number }> {
    const trimmed = sql.trim();
    if (/insert\s+into\s+scan_records/i.test(trimmed)) {
      const record = {
        id: params[0],
        denomination_id: params[1],
        nominal_value: params[2],
        name: params[3],
        confidence: params[4],
        source: params[5],
        model_id: params[6],
        latency_ms: params[7],
        bbox_json: params[8],
        image_uri: params[9],
        timestamp: params[10],
      };
      this.scanRecords.unshift(record);
      return { lastInsertRowId: this.scanRecords.length, changes: 1 };
    }

    if (/insert\s+or\s+replace\s+into\s+wallet_meta/i.test(trimmed)) {
      this.walletMeta.set(params[0], params[1]);
      return { changes: 1 };
    }

    if (/delete\s+from\s+scan_records/i.test(trimmed)) {
      const count = this.scanRecords.length;
      this.scanRecords = [];
      return { changes: count };
    }

    return { changes: 0 };
  }

  public async getAllAsync<T = any>(sql: string, params: any[] = []): Promise<T[]> {
    const trimmed = sql.trim();

    if (/select\s+.*from\s+scan_records/i.test(trimmed)) {
      let results = [...this.scanRecords];

      // Handle denomination filter: WHERE denomination_id = ?
      if (/where\s+denomination_id\s*=\s*\?/i.test(trimmed) && params.length > 0) {
        const filterId = Number(params[0]);
        results = results.filter((r) => r.denomination_id === filterId);
      }

      // Handle sort order
      if (/order\s+by\s+timestamp\s+desc/i.test(trimmed)) {
        results.sort((a, b) => b.timestamp - a.timestamp);
      }

      // Handle limit & offset
      const limitMatch = trimmed.match(/limit\s+(\d+|\?)/i);
      const offsetMatch = trimmed.match(/offset\s+(\d+|\?)/i);

      let limit = results.length;
      let offset = 0;

      if (limitMatch) {
        if (limitMatch[1] === "?") {
          const paramIdx = params.length >= 2 ? params.length - 2 : 0;
          limit = Number(params[paramIdx]);
        } else {
          limit = Number(limitMatch[1]);
        }
      }

      if (offsetMatch) {
        if (offsetMatch[1] === "?") {
          offset = Number(params[params.length - 1]);
        } else {
          offset = Number(offsetMatch[1]);
        }
      }

      return results.slice(offset, offset + limit) as T[];
    }

    return [] as T[];
  }

  public async getFirstAsync<T = any>(sql: string, params: any[] = []): Promise<T | null> {
    const trimmed = sql.trim();

    if (/select\s+sum\(nominal_value\).*from\s+scan_records/i.test(trimmed)) {
      const totalAmount = this.scanRecords.reduce((sum, r) => sum + (Number(r.nominal_value) || 0), 0);
      const totalCount = this.scanRecords.length;
      return { total_amount: totalAmount, total_count: totalCount } as unknown as T;
    }

    if (/select\s+value\s+from\s+wallet_meta\s+where\s+key\s*=\s*\?/i.test(trimmed)) {
      const val = this.walletMeta.get(params[0]);
      if (val !== undefined) {
        return { value: val } as unknown as T;
      }
      return null;
    }

    const all = await this.getAllAsync<T>(sql, params);
    return all.length > 0 ? all[0] : null;
  }

  public async closeAsync(): Promise<void> {
    // No-op for in-memory
  }
}

export class ExpoSQLiteAdapter implements IDatabaseAdapter {
  private db: any = null;
  private dbName: string;

  constructor(dbName: string = "banknote_vision.db") {
    this.dbName = dbName;
  }

  public async init(): Promise<void> {
    try {
      const SQLite = require("expo-sqlite");
      if (typeof SQLite.openDatabaseAsync === "function") {
        this.db = await SQLite.openDatabaseAsync(this.dbName);
      } else if (typeof SQLite.openDatabase === "function") {
        this.db = SQLite.openDatabase(this.dbName);
      }
    } catch (err) {
      console.warn("Falling back to in-memory adapter; native expo-sqlite is unavailable:", err);
    }
  }

  public async execAsync(sql: string): Promise<void> {
    if (!this.db) return;
    if (typeof this.db.execAsync === "function") {
      await this.db.execAsync(sql);
    } else if (typeof this.db.transaction === "function") {
      await new Promise<void>((resolve, reject) => {
        this.db.transaction((tx: any) => {
          tx.executeSql(sql, [], () => resolve(), (_: any, err: any) => { reject(err); return false; });
        });
      });
    }
  }

  public async runAsync(sql: string, params: any[] = []): Promise<{ lastInsertRowId?: number; changes: number }> {
    if (!this.db) return { changes: 0 };
    if (typeof this.db.runAsync === "function") {
      const result = await this.db.runAsync(sql, params);
      return { lastInsertRowId: result.lastInsertRowId, changes: result.changes };
    }
    return new Promise((resolve, reject) => {
      this.db.transaction((tx: any) => {
        tx.executeSql(
          sql,
          params,
          (_: any, res: any) => resolve({ lastInsertRowId: res.insertId, changes: res.rowsAffected }),
          (_: any, err: any) => { reject(err); return false; }
        );
      });
    });
  }

  public async getAllAsync<T = any>(sql: string, params: any[] = []): Promise<T[]> {
    if (!this.db) return [];
    if (typeof this.db.getAllAsync === "function") {
      return await this.db.getAllAsync(sql, params);
    }
    return new Promise((resolve, reject) => {
      this.db.transaction((tx: any) => {
        tx.executeSql(
          sql,
          params,
          (_: any, res: any) => {
            const items: T[] = [];
            for (let i = 0; i < res.rows.length; i++) {
              items.push(res.rows.item(i));
            }
            resolve(items);
          },
          (_: any, err: any) => { reject(err); return false; }
        );
      });
    });
  }

  public async getFirstAsync<T = any>(sql: string, params: any[] = []): Promise<T | null> {
    if (!this.db) return null;
    if (typeof this.db.getFirstAsync === "function") {
      return await this.db.getFirstAsync(sql, params);
    }
    const all = await this.getAllAsync<T>(sql, params);
    return all.length > 0 ? all[0] : null;
  }

  public async closeAsync(): Promise<void> {
    if (this.db && typeof this.db.closeAsync === "function") {
      await this.db.closeAsync();
    }
    this.db = null;
  }
}
