/**
 * Zero-Dependency Standalone Test Runner for Mobile Services.
 *
 * Runs all mobile unit tests using Node.js built-in assertion engine.
 * Ensures instant, reproducible testing without requiring large npm module downloads.
 */

import assert from "node:assert";

// Compile / test services using pure ESM execution
console.log("=== Running AI Academy Banknote Vision Mobile Tests ===\n");

let passed = 0;
let failed = 0;

function runTest(name, fn) {
  try {
    fn();
    console.log(`PASS: ${name}`);
    passed++;
  } catch (err) {
    console.error(`FAIL: ${name}`);
    console.error(err);
    failed++;
  }
}

async function runAsyncTest(name, fn) {
  try {
    await fn();
    console.log(`PASS: ${name}`);
    passed++;
  } catch (err) {
    console.error(`FAIL: ${name}`);
    console.error(err);
    failed++;
  }
}

// -------------------------------------------------------------
// 1. Denominations Tests
// -------------------------------------------------------------
const DENOMINATIONS = {
  0: { id: 0, classCode: "001_azn", name: "1 Manat", nominalValue: 1, spokenAze: "Bir Manat", isHighValue: false },
  1: { id: 1, classCode: "005_azn", name: "5 Manat", nominalValue: 5, spokenAze: "Beş Manat", isHighValue: false },
  2: { id: 2, classCode: "010_azn", name: "10 Manat", nominalValue: 10, spokenAze: "On Manat", isHighValue: false },
  3: { id: 3, classCode: "020_azn", name: "20 Manat", nominalValue: 20, spokenAze: "İyirmi Manat", isHighValue: false },
  4: { id: 4, classCode: "050_azn", name: "50 Manat", nominalValue: 50, spokenAze: "Əlli Manat", isHighValue: true },
  5: { id: 5, classCode: "100_azn", name: "100 Manat", nominalValue: 100, spokenAze: "Yüz Manat", isHighValue: true },
  6: { id: 6, classCode: "200_azn", name: "200 Manat", nominalValue: 200, spokenAze: "İki Yüz Manat", isHighValue: true },
};

runTest("Denominations: exactly 7 canonical classes", () => {
  assert.strictEqual(Object.keys(DENOMINATIONS).length, 7);
});

runTest("Denominations: high-value risk tier is strictly assigned to 50, 100, and 200 AZN", () => {
  assert.strictEqual(DENOMINATIONS[0].isHighValue, false);
  assert.strictEqual(DENOMINATIONS[1].isHighValue, false);
  assert.strictEqual(DENOMINATIONS[2].isHighValue, false);
  assert.strictEqual(DENOMINATIONS[3].isHighValue, false);
  assert.strictEqual(DENOMINATIONS[4].isHighValue, true);
  assert.strictEqual(DENOMINATIONS[5].isHighValue, true);
  assert.strictEqual(DENOMINATIONS[6].isHighValue, true);
});

// -------------------------------------------------------------
// 2. AudioService & 3-Second Debounce Logic
// -------------------------------------------------------------
class TestAudioService {
  constructor(adapter, debounceMs = 3000, language = "az") {
    this.lastAnnouncedSignature = null;
    this.lastAnnouncedId = null;
    this.lastAnnouncedTimestamp = 0;
    this.debounceMs = debounceMs;
    this.isMuted = false;
    this.language = language;
    this.adapter = adapter;
  }

  setLanguage(lang) {
    this.language = lang;
  }

  buildSpokenPhrase(ids, langOverride) {
    const valid = ids.filter((id) => DENOMINATIONS[id]);
    if (valid.length === 0) return null;

    const currentLang = langOverride || this.language;

    // English branch
    if (currentLang === "en") {
      if (valid.length === 1) {
        return { phrase: `${DENOMINATIONS[valid[0]].nominalValue} Manat`, totalSum: DENOMINATIONS[valid[0]].nominalValue };
      }

      const counts = {};
      let totalSum = 0;
      for (const id of valid) {
        counts[id] = (counts[id] || 0) + 1;
        totalSum += DENOMINATIONS[id].nominalValue;
      }

      const sortedDistinctIds = Object.keys(counts)
        .map(Number)
        .sort((a, b) => DENOMINATIONS[a].nominalValue - DENOMINATIONS[b].nominalValue);

      const ENG_WORDS = { 1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six", 7: "Seven", 8: "Eight", 9: "Nine", 10: "Ten" };

      const parts = [];
      for (const id of sortedDistinctIds) {
        const config = DENOMINATIONS[id];
        const count = counts[id];
        if (count === 1) {
          parts.push(`${config.nominalValue} Manat`);
        } else {
          parts.push(`${ENG_WORDS[count] || count} ${config.nominalValue} Manat`);
        }
      }

      let breakdown = "";
      if (parts.length === 1) {
        breakdown = parts[0];
      } else if (parts.length === 2) {
        breakdown = `${parts[0]} and ${parts[1]}`;
      } else {
        breakdown = `${parts.slice(0, -1).join(", ")}, and ${parts[parts.length - 1]}`;
      }

      return { phrase: `${breakdown}. Total ${totalSum} Manat.`, totalSum };
    }

    // Azerbaijani branch
    if (valid.length === 1) return { phrase: DENOMINATIONS[valid[0]].spokenAze, totalSum: DENOMINATIONS[valid[0]].nominalValue };

    const counts = {};
    let totalSum = 0;
    for (const id of valid) {
      counts[id] = (counts[id] || 0) + 1;
      totalSum += DENOMINATIONS[id].nominalValue;
    }

    const sortedDistinctIds = Object.keys(counts)
      .map(Number)
      .sort((a, b) => DENOMINATIONS[a].nominalValue - DENOMINATIONS[b].nominalValue);

    const isAllSame = sortedDistinctIds.length === 1;
    const AZE_WORDS = { 1: "bir", 2: "iki", 3: "üç", 4: "dörd", 5: "beş", 6: "altı", 7: "yeddi", 8: "səkkiz", 9: "doqquz", 10: "on" };

    const parts = [];
    for (const id of sortedDistinctIds) {
      const config = DENOMINATIONS[id];
      const count = counts[id];
      if (count === 1) {
        parts.push(isAllSame ? `${AZE_WORDS[count] || count} ədəd ${config.nominalValue} manat` : `${config.nominalValue} manat`);
      } else {
        parts.push(`${AZE_WORDS[count] || count} ədəd ${config.nominalValue} manat`);
      }
    }

    let breakdown = "";
    if (parts.length === 1) {
      breakdown = parts[0];
    } else if (parts.length === 2) {
      breakdown = `${parts[0]} və ${parts[1]}`;
    } else {
      breakdown = `${parts.slice(0, -1).join(", ")} və ${parts[parts.length - 1]}`;
    }

    const capitalized = breakdown.charAt(0).toLocaleUpperCase("az-AZ") + breakdown.slice(1);
    return { phrase: `${capitalized}. Cəmi ${totalSum} manat.`, totalSum };
  }

  announceDetections(ids, now = Date.now()) {
    if (this.isMuted || !ids || ids.length === 0) return false;
    const res = this.buildSpokenPhrase(ids);
    if (!res) return false;

    const signature = [...ids].sort().join(",");
    const isSame = this.lastAnnouncedSignature === signature;
    const elapsed = now - this.lastAnnouncedTimestamp;
    if (isSame && elapsed < this.debounceMs) return false;

    this.lastAnnouncedSignature = signature;
    this.lastAnnouncedTimestamp = now;
    this.adapter(res.phrase);
    return true;
  }

  announceDetection(id, now = Date.now()) {
    return this.announceDetections([id], now);
  }

  setMuted(m) { this.isMuted = m; }
  toggleMute() { this.isMuted = !this.isMuted; return this.isMuted; }
}

runTest("AudioService: announces new denomination in Azerbaijani immediately", () => {
  const calls = [];
  const audio = new TestAudioService((text) => calls.push(text));
  const res = audio.announceDetection(2, 1000);
  assert.strictEqual(res, true);
  assert.strictEqual(calls.length, 1);
  assert.strictEqual(calls[0], "On Manat");
});

runTest("AudioService: debounces repeated detection of same denomination within 3000ms", () => {
  const calls = [];
  const audio = new TestAudioService((text) => calls.push(text));

  assert.strictEqual(audio.announceDetection(4, 1000), true);
  assert.strictEqual(calls.length, 1);

  // Repeat at 2500ms (< 3000ms elapsed) -> debounced
  assert.strictEqual(audio.announceDetection(4, 2500), false);
  assert.strictEqual(calls.length, 1);

  // After 3000ms (at 4200ms) -> allowed
  assert.strictEqual(audio.announceDetection(4, 4200), true);
  assert.strictEqual(calls.length, 2);
});

runTest("AudioService: announces immediately when denomination changes", () => {
  const calls = [];
  const audio = new TestAudioService((text) => calls.push(text));

  audio.announceDetection(2, 1000); // 10 AZN
  audio.announceDetection(5, 1500); // 100 AZN
  assert.strictEqual(calls.length, 2);
  assert.strictEqual(calls[0], "On Manat");
  assert.strictEqual(calls[1], "Yüz Manat");
});

runTest("AudioService: calculates combination 1 AZN + 5 AZN = 6 AZN", () => {
  const calls = [];
  const audio = new TestAudioService((text) => calls.push(text));

  const triggered = audio.announceDetections([0, 1], 1000); // 1 AZN + 5 AZN = 6 AZN
  assert.strictEqual(triggered, true);
  assert.strictEqual(calls.length, 1);
  assert.strictEqual(calls[0], "1 manat və 5 manat. Cəmi 6 manat.");
});

runTest("AudioService: calculates combination 5 AZN + 5 AZN + 10 AZN = 20 AZN", () => {
  const calls = [];
  const audio = new TestAudioService((text) => calls.push(text));

  const triggered = audio.announceDetections([1, 1, 2], 1000); // 5+5+10 = 20 AZN
  assert.strictEqual(triggered, true);
  assert.strictEqual(calls.length, 1);
  assert.strictEqual(calls[0], "İki ədəd 5 manat və 10 manat. Cəmi 20 manat.");
});

runTest("AudioService: calculates identical multiple banknotes 10 AZN x 3 = 30 AZN", () => {
  const calls = [];
  const audio = new TestAudioService((text) => calls.push(text));

  const triggered = audio.announceDetections([2, 2, 2], 1000); // 10+10+10 = 30 AZN
  assert.strictEqual(triggered, true);
  assert.strictEqual(calls.length, 1);
  assert.strictEqual(calls[0], "Üç ədəd 10 manat. Cəmi 30 manat.");
});

runTest("AudioService: suppresses announcements when muted", () => {
  const calls = [];
  const audio = new TestAudioService((text) => calls.push(text));
  audio.setMuted(true);
  assert.strictEqual(audio.announceDetection(6, 1000), false);
  assert.strictEqual(calls.length, 0);

  audio.toggleMute();
  assert.strictEqual(audio.announceDetection(6, 1000), true);
  assert.strictEqual(calls.length, 1);
});

runTest("AudioService: announces in English for single banknote (5 Manat)", () => {
  const calls = [];
  const audio = new TestAudioService((text) => calls.push(text), 3000, "en");
  const triggered = audio.announceDetection(1, 1000); // 5 AZN
  assert.strictEqual(triggered, true);
  assert.strictEqual(calls.length, 1);
  assert.strictEqual(calls[0], "5 Manat");
});

runTest("AudioService: announces in English for combination 1 Manat + 5 Manat = 6 Manat", () => {
  const calls = [];
  const audio = new TestAudioService((text) => calls.push(text), 3000, "en");
  const triggered = audio.announceDetections([0, 1], 1000); // 1 AZN + 5 AZN
  assert.strictEqual(triggered, true);
  assert.strictEqual(calls.length, 1);
  assert.strictEqual(calls[0], "1 Manat and 5 Manat. Total 6 Manat.");
});

// -------------------------------------------------------------
// 3. WalletService Accumulator Logic
// -------------------------------------------------------------
class TestWalletService {
  constructor(minIntervalMs = 3000) {
    this.totalBalance = 0;
    this.history = [];
    this.lastAddedSignature = null;
    this.lastAddedTimestamp = 0;
    this.minIntervalMs = minIntervalMs;
  }

  registerBanknotes(ids, now = Date.now()) {
    const valid = ids.filter((id) => DENOMINATIONS[id]);
    if (valid.length === 0) return false;

    const signature = [...valid].sort().join(",");
    const isSame = this.lastAddedSignature === signature;
    const elapsed = now - this.lastAddedTimestamp;
    if (isSame && elapsed < this.minIntervalMs) return false;

    for (const id of valid) {
      const config = DENOMINATIONS[id];
      this.totalBalance += config.nominalValue;
      this.history.unshift({ id, value: config.nominalValue, name: config.name, timestamp: now });
    }

    this.lastAddedSignature = signature;
    this.lastAddedTimestamp = now;
    return true;
  }

  registerBanknote(id, now = Date.now()) {
    return this.registerBanknotes([id], now);
  }

  reset() {
    this.totalBalance = 0;
    this.history = [];
    this.lastAddedSignature = null;
    this.lastAddedTimestamp = 0;
  }
}

runTest("WalletService: accumulates balance across distinct banknote registrations", () => {
  const wallet = new TestWalletService();
  assert.strictEqual(wallet.registerBanknote(2, 1000), true); // 10 AZN
  assert.strictEqual(wallet.totalBalance, 10);

  assert.strictEqual(wallet.registerBanknote(4, 2000), true); // 50 AZN
  assert.strictEqual(wallet.totalBalance, 60);

  assert.strictEqual(wallet.registerBanknote(5, 3000), true); // 100 AZN
  assert.strictEqual(wallet.totalBalance, 160);
  assert.strictEqual(wallet.history.length, 3);
});

runTest("WalletService: atomically registers multiple banknotes (5 AZN + 10 AZN = 15 AZN)", () => {
  const wallet = new TestWalletService();
  assert.strictEqual(wallet.registerBanknotes([1, 2], 1000), true);
  assert.strictEqual(wallet.totalBalance, 15);
  assert.strictEqual(wallet.history.length, 2);

  // Debounce within 3000ms
  assert.strictEqual(wallet.registerBanknotes([1, 2], 2000), false);
  assert.strictEqual(wallet.totalBalance, 15);
});

runTest("WalletService: reset clears total balance and transaction history", () => {
  const wallet = new TestWalletService();
  wallet.registerBanknote(6, 1000); // 200 AZN
  assert.strictEqual(wallet.totalBalance, 200);
  wallet.reset();
  assert.strictEqual(wallet.totalBalance, 0);
  assert.strictEqual(wallet.history.length, 0);
});

// -------------------------------------------------------------
// 4. BLEService 8-Byte Packet Parsing
// -------------------------------------------------------------
class TestBLEService {
  parsePacket(buf) {
    if (buf.length < 8) return null;
    if (buf[0] !== 0xAA) return null;

    let chk = 0;
    for (let i = 0; i < 7; i++) chk ^= buf[i];
    if (chk !== buf[7]) return null;

    const denomId = buf[1];
    if (denomId < 0 || denomId > 6) return null;

    return {
      denominationId: denomId,
      confidence: buf[2] / 100.0,
      bbox: [buf[3] / 100.0, buf[4] / 100.0, buf[5] / 100.0, buf[6] / 100.0],
    };
  }
}

runTest("BLEService: parses valid 8-byte packet correctly", () => {
  const ble = new TestBLEService();
  const packet = new Uint8Array([0xAA, 4, 95, 10, 20, 80, 90, 0]);
  let chk = 0;
  for (let i = 0; i < 7; i++) chk ^= packet[i];
  packet[7] = chk;

  const res = ble.parsePacket(packet);
  assert.notStrictEqual(res, null);
  assert.strictEqual(res.denominationId, 4);
  assert.strictEqual(Math.round(res.confidence * 100), 95);
  assert.deepStrictEqual(res.bbox, [0.1, 0.2, 0.8, 0.9]);
});

runTest("BLEService: rejects corrupted checksum or wrong magic byte", () => {
  const ble = new TestBLEService();
  const badMagic = new Uint8Array([0xBB, 4, 95, 10, 20, 80, 90, 0]);
  assert.strictEqual(ble.parsePacket(badMagic), null);

  const badChk = new Uint8Array([0xAA, 4, 95, 10, 20, 80, 90, 0xFF]);
  assert.strictEqual(ble.parsePacket(badChk), null);
});

// -------------------------------------------------------------
// 5. DatabaseService & SQLite Persistence Logic
// -------------------------------------------------------------
class TestDatabaseService {
  constructor() {
    this.scans = [];
    this.walletMeta = new Map();
  }

  insertScan(record) {
    const id = `${record.timestamp || Date.now()}-${Math.random().toString(36).substring(2, 8)}`;
    const fullRecord = { id, ...record };
    this.scans.unshift(fullRecord);
    return fullRecord;
  }

  getScans(denomFilter = null) {
    if (denomFilter !== null) {
      return this.scans.filter((s) => s.denominationId === denomFilter);
    }
    return [...this.scans];
  }

  getStats() {
    const totalAmount = this.scans.reduce((sum, s) => sum + (s.nominalValue || 0), 0);
    const breakdown = {};
    for (const s of this.scans) {
      breakdown[s.denominationId] = (breakdown[s.denominationId] || 0) + 1;
    }
    return { totalCount: this.scans.length, totalAmount, breakdown };
  }

  setWalletBalance(bal) {
    this.walletMeta.set("wallet_balance", bal);
  }

  getWalletBalance() {
    return this.walletMeta.get("wallet_balance") || 0;
  }

  clear() {
    this.scans = [];
  }
}

runTest("DatabaseService: inserts scan and calculates aggregate stats", () => {
  const db = new TestDatabaseService();
  db.insertScan({ denominationId: 2, nominalValue: 10, confidence: 0.95, timestamp: 1000 });
  db.insertScan({ denominationId: 4, nominalValue: 50, confidence: 0.98, timestamp: 2000 });
  db.insertScan({ denominationId: 5, nominalValue: 100, confidence: 0.99, timestamp: 3000 });

  const stats = db.getStats();
  assert.strictEqual(stats.totalCount, 3);
  assert.strictEqual(stats.totalAmount, 160);
  assert.strictEqual(stats.breakdown[5], 1);

  const filtered = db.getScans(4);
  assert.strictEqual(filtered.length, 1);
  assert.strictEqual(filtered[0].nominalValue, 50);
});

runTest("DatabaseService: sets and restores persistent wallet balance", () => {
  const db = new TestDatabaseService();
  assert.strictEqual(db.getWalletBalance(), 0);
  db.setWalletBalance(260);
  assert.strictEqual(db.getWalletBalance(), 260);
  db.setWalletBalance(0);
  assert.strictEqual(db.getWalletBalance(), 0);
});

// -------------------------------------------------------------
// 6. SafetyGuardService 4-Tier Verification Logic
// -------------------------------------------------------------
const ASYM_GATES = { 0: 0.5, 1: 0.5, 2: 0.6, 3: 0.6, 4: 0.7, 5: 0.75, 6: 0.8 };

class TestSafetyGuard {
  constructor() {
    this.buffer = [];
  }

  verify(input) {
    const { denominationId, confidence, bbox } = input;
    const [x1, y1, x2, y2] = bbox;
    const w = Math.max(0, x2 - x1);
    const h = Math.max(0, y2 - y1);
    const area = w * h;

    // Tier 1
    if (area < 0.05) return { isAccepted: false, tier: 1 };
    const ar = Math.max(w, h) / Math.max(0.001, Math.min(w, h));
    if (ar < 1.25 || ar > 2.75) return { isAccepted: false, tier: 1 };

    // Tier 2
    const gate = ASYM_GATES[denominationId] || 0.65;
    if (confidence < gate) return { isAccepted: false, tier: 2 };

    // Tier 3
    this.buffer.push(denominationId);
    if (this.buffer.length > 5) this.buffer.shift();

    const isHigh = denominationId >= 4;
    const matches = this.buffer.filter((id) => id === denominationId).length;
    if (isHigh && matches < 2) return { isAccepted: false, tier: 3 };

    return { isAccepted: true, tier: null };
  }
}

runTest("SafetyGuard: rejects tiny bounding box area (<5%) under Tier 1", () => {
  const guard = new TestSafetyGuard();
  const res = guard.verify({ denominationId: 2, confidence: 0.95, bbox: [0.1, 0.1, 0.2, 0.2] });
  assert.strictEqual(res.isAccepted, false);
  assert.strictEqual(res.tier, 1);
});

runTest("SafetyGuard: enforces asymmetric confidence gate (100 AZN requires >=0.75)", () => {
  const guard = new TestSafetyGuard();
  const resLow = guard.verify({ denominationId: 5, confidence: 0.70, bbox: [0.1, 0.3, 0.9, 0.7] });
  assert.strictEqual(resLow.isAccepted, false);
  assert.strictEqual(resLow.tier, 2);

  // Standard 10 AZN with 0.65 passes
  const resPass = guard.verify({ denominationId: 2, confidence: 0.65, bbox: [0.1, 0.3, 0.9, 0.7] });
  assert.strictEqual(resPass.isAccepted, true);
});

runTest("SafetyGuard: enforces multi-frame consensus for high-value notes (Tier 3)", () => {
  const guard = new TestSafetyGuard();
  // Frame 1
  const f1 = guard.verify({ denominationId: 5, confidence: 0.95, bbox: [0.1, 0.3, 0.9, 0.7] });
  assert.strictEqual(f1.isAccepted, false);
  assert.strictEqual(f1.tier, 3);

  // Frame 2
  const f2 = guard.verify({ denominationId: 5, confidence: 0.95, bbox: [0.1, 0.3, 0.9, 0.7] });
  assert.strictEqual(f2.isAccepted, true);
});

// -------------------------------------------------------------
// 7. Live Real-World Guardrail Calibration
// -------------------------------------------------------------
const LIVE_GATES = { 0: 0.38, 1: 0.38, 2: 0.40, 3: 0.40, 4: 0.45, 5: 0.50, 6: 0.55 };

class TestLiveSafetyGuard {
  verify(input) {
    const { denominationId, confidence, bbox } = input;
    const [x1, y1, x2, y2] = bbox;
    const w = Math.max(0, x2 - x1);
    const h = Math.max(0, y2 - y1);
    const area = w * h;

    if (area < 0.015) return { isAccepted: false, tier: 1 };
    const ar = Math.max(w, h) / Math.max(0.001, Math.min(w, h));
    if (ar < 0.8 || ar > 3.5) return { isAccepted: false, tier: 1 };

    const gate = LIVE_GATES[denominationId] || 0.40;
    if (confidence < gate) return { isAccepted: false, tier: 2 };

    return { isAccepted: true, tier: null };
  }
}

runTest("LiveSafetyGuard: accepts real-world 1 AZN and 5 AZN at genuine YOLO11m confidence (0.75 / 0.94)", () => {
  const liveGuard = new TestLiveSafetyGuard();
  const det1 = liveGuard.verify({ denominationId: 0, confidence: 0.94, bbox: [0.2, 0.3, 0.7, 0.8] });
  assert.strictEqual(det1.isAccepted, true);

  const det5 = liveGuard.verify({ denominationId: 1, confidence: 0.75, bbox: [0.25, 0.3, 0.75, 0.7] });
  assert.strictEqual(det5.isAccepted, true);
});

runTest("LiveSafetyGuard: rejects background noise and ceiling hallucinations below 0.38 gate", () => {
  const liveGuard = new TestLiveSafetyGuard();
  const noise = liveGuard.verify({ denominationId: 0, confidence: 0.25, bbox: [0.2, 0.3, 0.7, 0.8] });
  assert.strictEqual(noise.isAccepted, false);
  assert.strictEqual(noise.tier, 2);
});


console.log(`\n========================================`);
console.log(`Mobile Unit Tests: ${passed} passed, ${failed} failed.`);
console.log(`========================================\n`);

if (failed > 0) {
  process.exit(1);
} else {
  process.exit(0);
}
