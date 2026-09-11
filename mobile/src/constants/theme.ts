/**
 * Visual Design Theme for AI Academy Banknote Vision Mobile App.
 *
 * Implements the official AI Academy visual identity:
 * - Deep dark navy background (#0D0F1A / #131524)
 * - Neon lime / chartreuse signature accent (#D4F938)
 * - High-contrast accessibility typography for visually impaired users.
 */

export const THEME = {
  colors: {
    background: "#0D0F1A",
    backgroundAlt: "#131524",
    surface: "#161928",
    surfaceLight: "#1E2337",
    surfaceBorder: "#262B40",
    accentNeon: "#D4F938",
    accentNeonMuted: "rgba(212, 249, 56, 0.15)",
    textPrimary: "#FFFFFF",
    textSecondary: "#8B949E",
    textMuted: "#626880",
    statusSuccess: "#3FB950",
    statusWarning: "#D29922",
    statusDanger: "#F85149",
    statusInfo: "#58A6FF",
    overlayDark: "rgba(13, 15, 26, 0.75)",
  },
  typography: {
    fontFamilySans: "System",
    fontFamilyMono: "Courier",
    sizes: {
      xs: 11,
      sm: 13,
      base: 15,
      lg: 18,
      xl: 22,
      xxl: 28,
      hero: 36,
    },
    weights: {
      regular: "400" as const,
      medium: "500" as const,
      semibold: "600" as const,
      bold: "700" as const,
      heavy: "800" as const,
    },
  },
  spacing: {
    xs: 4,
    sm: 8,
    md: 12,
    lg: 16,
    xl: 24,
    xxl: 32,
  },
  borderRadius: {
    sm: 6,
    md: 10,
    lg: 16,
    full: 9999,
  },
};
