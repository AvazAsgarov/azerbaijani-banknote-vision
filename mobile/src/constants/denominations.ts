/**
 * Azerbaijani Banknote Denomination Canonical Mappings & Metadata.
 *
 * Defines the 7 standard AZN banknote classes, numeric values,
 * high-contrast UI accent colors, Azerbaijani TTS spoken phrases,
 * and high-value risk tiers.
 */

export interface DenominationConfig {
  id: number;
  classCode: string;
  name: string;
  nominalValue: number;
  spokenAze: string;
  colorHex: string;
  isHighValue: boolean;
  securityRiskTier: "STANDARD" | "HIGH_VALUE" | "CRITICAL_VALUE";
}

export const DENOMINATIONS: Record<number, DenominationConfig> = {
  0: {
    id: 0,
    classCode: "001_azn",
    name: "1 Manat",
    nominalValue: 1,
    spokenAze: "Bir Manat",
    colorHex: "#8B949E",
    isHighValue: false,
    securityRiskTier: "STANDARD",
  },
  1: {
    id: 1,
    classCode: "005_azn",
    name: "5 Manat",
    nominalValue: 5,
    spokenAze: "Beş Manat",
    colorHex: "#E67E22",
    isHighValue: false,
    securityRiskTier: "STANDARD",
  },
  2: {
    id: 2,
    classCode: "010_azn",
    name: "10 Manat",
    nominalValue: 10,
    spokenAze: "On Manat",
    colorHex: "#00BCD4",
    isHighValue: false,
    securityRiskTier: "STANDARD",
  },
  3: {
    id: 3,
    classCode: "020_azn",
    name: "20 Manat",
    nominalValue: 20,
    spokenAze: "İyirmi Manat",
    colorHex: "#2ECC71",
    isHighValue: false,
    securityRiskTier: "STANDARD",
  },
  4: {
    id: 4,
    classCode: "050_azn",
    name: "50 Manat",
    nominalValue: 50,
    spokenAze: "Əlli Manat",
    colorHex: "#F1C40F",
    isHighValue: true,
    securityRiskTier: "HIGH_VALUE",
  },
  5: {
    id: 5,
    classCode: "100_azn",
    name: "100 Manat",
    nominalValue: 100,
    spokenAze: "Yüz Manat",
    colorHex: "#9B59B6",
    isHighValue: true,
    securityRiskTier: "CRITICAL_VALUE",
  },
  6: {
    id: 6,
    classCode: "200_azn",
    name: "200 Manat",
    nominalValue: 200,
    spokenAze: "İki Yüz Manat",
    colorHex: "#3498DB",
    isHighValue: true,
    securityRiskTier: "CRITICAL_VALUE",
  },
};

export const DENOMINATION_LIST: DenominationConfig[] = Object.values(DENOMINATIONS);

export function getDenominationById(id: number): DenominationConfig {
  return DENOMINATIONS[id] || {
    id: -1,
    classCode: "unknown",
    name: "Bilinməyən Əsginas",
    nominalValue: 0,
    spokenAze: "Bilinməyən Əsginas",
    colorHex: "#6E7681",
    isHighValue: false,
    securityRiskTier: "STANDARD",
  };
}

export function getDenominationByCode(code: string): DenominationConfig {
  const found = DENOMINATION_LIST.find((d) => d.classCode === code);
  return found || getDenominationById(-1);
}
