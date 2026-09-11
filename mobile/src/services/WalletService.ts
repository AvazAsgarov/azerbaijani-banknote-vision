/**
 * Wallet Accumulator & Banknote History Service.
 *
 * Tracks unique banknote recognitions, sums up running wallet total (AZN),
 * maintains an auditable session log of scanned notes, and supports single-tap reset.
 * Automatically synchronizes with SQLite persistence layer.
 */

import { getDenominationById } from "../constants/denominations";
import { DatabaseService, globalDatabaseService } from "./db/DatabaseService";

export interface WalletTransaction {
  id: string;
  denominationId: number;
  nominalValue: number;
  name: string;
  timestamp: number;
}

export class WalletService {
  private totalBalance: number = 0;
  private history: WalletTransaction[] = [];
  private lastAddedSignature: string | null = null;
  private lastAddedId: number | null = null;
  private lastAddedTimestamp: number = 0;
  private minIntervalMs: number = 3000;
  private dbService: DatabaseService;
  private listeners: Array<(balance: number, history: WalletTransaction[]) => void> = [];
  private isInitialized: boolean = false;

  constructor(minIntervalMs: number = 3000, dbService?: DatabaseService) {
    this.minIntervalMs = minIntervalMs;
    this.dbService = dbService || globalDatabaseService;
  }

  /**
   * Asynchronously restores persisted wallet state from SQLite.
   */
  public async init(): Promise<void> {
    if (this.isInitialized) return;
    try {
      const persistedBalance = await this.dbService.getWalletBalance();
      const recentScans = await this.dbService.getScans(20);

      this.totalBalance = persistedBalance;
      this.history = recentScans.map((s) => ({
        id: s.id,
        denominationId: s.denominationId,
        nominalValue: s.nominalValue,
        name: s.name,
        timestamp: s.timestamp,
      }));

      this.isInitialized = true;
      this.notify();
    } catch {
      this.isInitialized = true;
    }
  }

  public subscribe(listener: (balance: number, history: WalletTransaction[]) => void): () => void {
    this.listeners.push(listener);
    listener(this.totalBalance, [...this.history]);
    return () => {
      this.listeners = this.listeners.filter((l) => l !== listener);
    };
  }

  private notify(): void {
    const historyCopy = [...this.history];
    for (const listener of this.listeners) {
      listener(this.totalBalance, historyCopy);
    }
  }

  /**
   * Records multiple detected banknotes into wallet atomically, updating running total balance.
   *
   * @param denominationIds Array of canonical denomination integer IDs (0 to 6)
   * @param nowTimestamp Current epoch timestamp in ms (defaults to Date.now())
   * @returns boolean true if successfully added to balance, false if debounced
   */
  public registerBanknotes(denominationIds: number[], nowTimestamp: number = Date.now()): boolean {
    if (!denominationIds || denominationIds.length === 0) {
      return false;
    }

    const validConfigs = denominationIds
      .map((id) => getDenominationById(id))
      .filter((cfg) => cfg.id >= 0 && cfg.nominalValue > 0);

    if (validConfigs.length === 0) {
      return false;
    }

    const signature = validConfigs.map((c) => c.id).sort((a, b) => a - b).join(",");
    const isSame = this.lastAddedSignature === signature;
    const elapsed = nowTimestamp - this.lastAddedTimestamp;

    if (isSame && elapsed < this.minIntervalMs) {
      return false;
    }

    let addedSum = 0;
    for (const config of validConfigs) {
      addedSum += config.nominalValue;
      const tx: WalletTransaction = {
        id: `${nowTimestamp}-${Math.random().toString(36).substring(2, 7)}`,
        denominationId: config.id,
        nominalValue: config.nominalValue,
        name: config.name,
        timestamp: nowTimestamp,
      };
      this.history.unshift(tx);
    }

    this.totalBalance += addedSum;
    this.lastAddedSignature = signature;
    this.lastAddedId = validConfigs.length === 1 ? validConfigs[0].id : null;
    this.lastAddedTimestamp = nowTimestamp;

    while (this.history.length > 50) {
      this.history.pop();
    }

    // Persist asynchronously in background to avoid UI latency
    this.dbService.setWalletBalance(this.totalBalance).catch(() => {});

    this.notify();
    return true;
  }

  /**
   * Records a detected banknote into wallet if debounce threshold is respected.
   *
   * @param denominationId Canonical denomination integer ID (0 to 6)
   * @param nowTimestamp Current epoch timestamp in ms (defaults to Date.now())
   * @returns boolean true if successfully added to balance, false if debounced
   */
  public registerBanknote(denominationId: number, nowTimestamp: number = Date.now()): boolean {
    return this.registerBanknotes([denominationId], nowTimestamp);
  }

  /**
   * Resets wallet balance and transaction history to zero.
   */
  public resetWallet(): void {
    this.totalBalance = 0;
    this.history = [];
    this.lastAddedSignature = null;
    this.lastAddedId = null;
    this.lastAddedTimestamp = 0;
    this.dbService.setWalletBalance(0).catch(() => {});
    this.notify();
  }

  public getBalance(): number {
    return this.totalBalance;
  }

  public getHistory(): WalletTransaction[] {
    return [...this.history];
  }
}

export const globalWalletService = new WalletService();
