/**
 * Unit Tests for WalletService (Accumulator & History Logic).
 */

import { WalletService } from "../src/services/WalletService";

describe("WalletService", () => {
  let wallet: WalletService;

  beforeEach(() => {
    wallet = new WalletService(3000);
  });

  test("initial balance is 0 and history is empty", () => {
    expect(wallet.getBalance()).toBe(0);
    expect(wallet.getHistory()).toHaveLength(0);
  });

  test("correctly accumulates balance across distinct banknote registrations", () => {
    expect(wallet.registerBanknote(2, 1000)).toBe(true); // 10 AZN
    expect(wallet.getBalance()).toBe(10);

    expect(wallet.registerBanknote(4, 2000)).toBe(true); // 50 AZN
    expect(wallet.getBalance()).toBe(60);

    expect(wallet.registerBanknote(5, 3000)).toBe(true); // 100 AZN
    expect(wallet.getBalance()).toBe(160);

    const history = wallet.getHistory();
    expect(history).toHaveLength(3);
    expect(history[0].nominalValue).toBe(100);
    expect(history[1].nominalValue).toBe(50);
    expect(history[2].nominalValue).toBe(10);
  });

  test("debounces repeated registrations of the same banknote within minimum interval", () => {
    expect(wallet.registerBanknote(3, 1000)).toBe(true); // 20 AZN
    expect(wallet.getBalance()).toBe(20);

    // Duplicate detection within 3000ms
    expect(wallet.registerBanknote(3, 2200)).toBe(false);
    expect(wallet.getBalance()).toBe(20);

    // After 3000ms, allowed
    expect(wallet.registerBanknote(3, 4500)).toBe(true);
    expect(wallet.getBalance()).toBe(40);
  });

  test("resets balance and history cleanly", () => {
    wallet.registerBanknote(6, 1000); // 200 AZN
    expect(wallet.getBalance()).toBe(200);

    wallet.resetWallet();
    expect(wallet.getBalance()).toBe(0);
    expect(wallet.getHistory()).toHaveLength(0);
  });

  test("notifies subscribers when balance changes", () => {
    const listener = jest.fn();
    const unsub = wallet.subscribe(listener);

    // Initial call on subscribe
    expect(listener).toHaveBeenCalledWith(0, []);

    wallet.registerBanknote(1, 1000); // 5 AZN
    expect(listener).toHaveBeenCalledWith(5, expect.any(Array));

    unsub();
    wallet.registerBanknote(0, 5000); // 1 AZN
    expect(listener).toHaveBeenCalledTimes(2); // No new call after unsub
  });

  test("restores persisted balance and history on init", async () => {
    const mockDb: any = {
      getWalletBalance: jest.fn().mockResolvedValue(160),
      getScans: jest.fn().mockResolvedValue([
        { id: "tx-1", denominationId: 5, nominalValue: 100, name: "100 Manat", timestamp: 2000 },
        { id: "tx-2", denominationId: 4, nominalValue: 50, name: "50 Manat", timestamp: 1000 },
      ]),
      setWalletBalance: jest.fn().mockResolvedValue(undefined),
    };

    const persistentWallet = new WalletService(3000, mockDb);
    await persistentWallet.init();

    expect(persistentWallet.getBalance()).toBe(160);
    expect(persistentWallet.getHistory()).toHaveLength(2);
    expect(persistentWallet.getHistory()[0].nominalValue).toBe(100);
  });

  test("registerBanknotes atomically adds multiple banknotes (5 AZN + 10 AZN = 15 AZN)", () => {
    expect(wallet.registerBanknotes([1, 2], 1000)).toBe(true);
    expect(wallet.getBalance()).toBe(15);
    expect(wallet.getHistory()).toHaveLength(2);

    // Debounce duplicate set within 3000ms
    expect(wallet.registerBanknotes([1, 2], 2000)).toBe(false);
    expect(wallet.getBalance()).toBe(15);

    // After 3000ms, allows adding
    expect(wallet.registerBanknotes([1, 2], 4500)).toBe(true);
    expect(wallet.getBalance()).toBe(30);
  });
});

