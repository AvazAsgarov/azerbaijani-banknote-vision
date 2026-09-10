import React from "react";
import { Image, StyleSheet, Text, TouchableOpacity, View } from "react-native";
import { THEME } from "../constants/theme";

interface HeaderRibbonProps {
  walletBalance: number;
  sourceMode: "PHONE_CAM" | "GLASSES_BLE";
  currentLanguage?: "en" | "az";
  onToggleLanguage?: () => void;
  onOpenHistory?: () => void;
}

export const HeaderRibbon: React.FC<HeaderRibbonProps> = ({
  walletBalance,
  sourceMode,
  currentLanguage = "en",
  onToggleLanguage,
  onOpenHistory,
}) => {
  return (
    <View style={styles.container}>
      <View style={styles.brandGroup}>
        <Image
          source={require("../../assets/logo.png")}
          style={styles.logo}
          resizeMode="contain"
        />
        <View>
          <Text style={styles.title}>AZN Vision</Text>
          <Text style={styles.subtitle}>AI Academy · Cohort I</Text>
        </View>
      </View>

      <View style={styles.rightGroup}>
        {/* Language Toggle Pill */}
        {onToggleLanguage && (
          <TouchableOpacity
            style={styles.langPill}
            onPress={onToggleLanguage}
            activeOpacity={0.7}
            accessibilityLabel={`Dil seçimi: ${currentLanguage === "en" ? "İngiliscə" : "Azərbaycanca"}`}
            accessibilityRole="button"
          >
            <Text style={styles.langText}>{currentLanguage === "en" ? "🇬🇧 EN" : "🇦🇿 AZ"}</Text>
          </TouchableOpacity>
        )}

        {/* Clickable Wallet Balance / Audit Pill */}
        <TouchableOpacity
          style={styles.balancePill}
          onPress={onOpenHistory}
          activeOpacity={0.7}
          accessibilityLabel="Cüzdan və skan tarixçəsi"
          accessibilityRole="button"
        >
          <Text style={styles.balanceLabel}>Cüzdan 📜</Text>
          <Text style={styles.balanceValue}>{walletBalance} ₼</Text>
        </TouchableOpacity>

        {/* Source Mode Badge */}
        <View style={styles.modeBadge}>
          <View
            style={[
              styles.modeDot,
              { backgroundColor: sourceMode === "GLASSES_BLE" ? THEME.colors.statusInfo : THEME.colors.accentNeon },
            ]}
          />
          <Text style={styles.modeText}>{sourceMode === "GLASSES_BLE" ? "Eynək" : "Kamera"}</Text>
        </View>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    paddingHorizontal: THEME.spacing.lg,
    paddingTop: THEME.spacing.xl,
    paddingBottom: THEME.spacing.md,
    backgroundColor: THEME.colors.background,
    borderBottomWidth: 1,
    borderBottomColor: THEME.colors.surfaceBorder,
  },
  brandGroup: {
    flexDirection: "row",
    alignItems: "center",
    gap: THEME.spacing.sm,
  },
  logo: {
    width: 38,
    height: 38,
    borderRadius: THEME.borderRadius.sm,
  },
  title: {
    color: THEME.colors.textPrimary,
    fontSize: THEME.typography.sizes.base,
    fontWeight: THEME.typography.weights.bold,
    letterSpacing: 0.5,
  },
  subtitle: {
    color: THEME.colors.textSecondary,
    fontSize: THEME.typography.sizes.xs,
  },
  rightGroup: {
    flexDirection: "row",
    alignItems: "center",
    gap: THEME.spacing.sm,
  },
  langPill: {
    backgroundColor: THEME.colors.surface,
    borderColor: THEME.colors.surfaceBorder,
    borderWidth: 1,
    borderRadius: THEME.borderRadius.md,
    paddingHorizontal: 8,
    paddingVertical: 6,
    alignItems: "center",
    justifyContent: "center",
  },
  langText: {
    color: THEME.colors.textPrimary,
    fontSize: 11,
    fontWeight: THEME.typography.weights.bold,
  },
  balancePill: {
    backgroundColor: THEME.colors.surface,
    borderColor: THEME.colors.accentNeon,
    borderWidth: 1.5,
    borderRadius: THEME.borderRadius.md,
    paddingHorizontal: THEME.spacing.md,
    paddingVertical: 4,
    alignItems: "center",
  },
  balanceLabel: {
    color: THEME.colors.accentNeon,
    fontSize: 9,
    fontWeight: THEME.typography.weights.bold,
    textTransform: "uppercase",
  },
  balanceValue: {
    color: THEME.colors.textPrimary,
    fontSize: THEME.typography.sizes.base,
    fontWeight: THEME.typography.weights.heavy,
  },
  modeBadge: {
    flexDirection: "row",
    alignItems: "center",
    gap: 4,
    backgroundColor: THEME.colors.surface,
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: THEME.borderRadius.sm,
    borderWidth: 1,
    borderColor: THEME.colors.surfaceBorder,
  },
  modeDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
  },
  modeText: {
    color: THEME.colors.textSecondary,
    fontSize: 10,
    fontWeight: THEME.typography.weights.semibold,
  },
});
