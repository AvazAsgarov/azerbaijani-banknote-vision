/**
 * Unit Tests for Denominations constants & lookups.
 */

import {
  DENOMINATIONS,
  DENOMINATION_LIST,
  getDenominationByCode,
  getDenominationById,
} from "../src/constants/denominations";

describe("Denominations", () => {
  test("contains exactly 7 canonical Azerbaijani banknote classes", () => {
    expect(DENOMINATION_LIST).toHaveLength(7);
  });

  test("nominal values match canonical currency values (1 to 200 AZN)", () => {
    expect(DENOMINATIONS[0].nominalValue).toBe(1);
    expect(DENOMINATIONS[1].nominalValue).toBe(5);
    expect(DENOMINATIONS[2].nominalValue).toBe(10);
    expect(DENOMINATIONS[3].nominalValue).toBe(20);
    expect(DENOMINATIONS[4].nominalValue).toBe(50);
    expect(DENOMINATIONS[5].nominalValue).toBe(100);
    expect(DENOMINATIONS[6].nominalValue).toBe(200);
  });

  test("Azerbaijani spoken names are correctly defined", () => {
    expect(DENOMINATIONS[0].spokenAze).toBe("Bir Manat");
    expect(DENOMINATIONS[2].spokenAze).toBe("On Manat");
    expect(DENOMINATIONS[4].spokenAze).toBe("Əlli Manat");
    expect(DENOMINATIONS[5].spokenAze).toBe("Yüz Manat");
    expect(DENOMINATIONS[6].spokenAze).toBe("İki Yüz Manat");
  });

  test("high-value risk tier is strictly assigned to 50, 100, and 200 AZN", () => {
    expect(DENOMINATIONS[0].isHighValue).toBe(false);
    expect(DENOMINATIONS[1].isHighValue).toBe(false);
    expect(DENOMINATIONS[2].isHighValue).toBe(false);
    expect(DENOMINATIONS[3].isHighValue).toBe(false);
    expect(DENOMINATIONS[4].isHighValue).toBe(true);
    expect(DENOMINATIONS[5].isHighValue).toBe(true);
    expect(DENOMINATIONS[6].isHighValue).toBe(true);
  });

  test("lookup by code works as expected", () => {
    expect(getDenominationByCode("010_azn").nominalValue).toBe(10);
    expect(getDenominationByCode("nonexistent").id).toBe(-1);
  });

  test("lookup by invalid ID returns graceful unknown fallback", () => {
    const unknown = getDenominationById(999);
    expect(unknown.id).toBe(-1);
    expect(unknown.name).toContain("Bilinməyən");
  });
});
