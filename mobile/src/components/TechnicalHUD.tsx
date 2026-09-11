import React from "react";
import { StyleSheet, Text, TouchableOpacity, View } from "react-native";
import { THEME } from "../constants/theme";

interface TechnicalHUDProps {
  fps: number;
  latencyMs: number;
  activeModelName: string;
  sourceMode: "PHONE_CAM" | "GLASSES_BLE";
  batteryPct?: number;
  rssi?: number;
  isVisible: boolean;
  onToggle: () => void;
  onSwitchModel?: () => void;
}

export const TechnicalHUD: React.FC<TechnicalHUDProps> = ({
  fps,
  latencyMs,
  activeModelName,
  sourceMode,
  batteryPct = 90,
  rssi = -64,
  isVisible,
  onToggle,
  onSwitchModel,
}) => {
  if (!isVisible) {
    return (
      <TouchableOpacity style={styles.floatingToggle} onPress={onToggle} activeOpacity={0.8}>
        <Text style={styles.toggleText}>HUD</Text>
      </TouchableOpacity>
    );
  }

  const isTinyML = activeModelName.toLowerCase().includes("fastestv2") || activeModelName.toLowerCase().includes("int8");

  return (
    <View style={styles.container}>
      <View style={styles.headerRow}>
        <View style={styles.badge}>
          <View style={styles.liveDot} />
          <Text style={styles.badgeText}>TELEMETRİYA HUD</Text>
        </View>
        <TouchableOpacity onPress={onToggle} style={styles.closeBtn}>
          <Text style={styles.closeText}>Bağla</Text>
        </TouchableOpacity>
      </View>

      <View style={styles.grid}>
        <View style={styles.metricCard}>
          <Text style={styles.metricLabel}>FPS</Text>
          <Text style={styles.metricVal}>{fps}</Text>
        </View>

        <View style={styles.metricCard}>
          <Text style={styles.metricLabel}>GECİKMƏ</Text>
          <Text style={styles.metricVal}>{latencyMs.toFixed(1)} ms</Text>
        </View>

        <View style={styles.metricCard}>
          <Text style={styles.metricLabel}>MƏNBƏ</Text>
          <Text style={styles.metricVal}>{sourceMode === "GLASSES_BLE" ? "ESP32-S3" : "Telefon"}</Text>
        </View>

        {sourceMode === "GLASSES_BLE" && (
          <View style={styles.metricCard}>
            <Text style={styles.metricLabel}>EYNƏK BATAREYA</Text>
            <Text style={styles.metricVal}>{batteryPct}% ({rssi} dBm)</Text>
          </View>
        )}
      </View>

      <TouchableOpacity
        style={styles.modelRow}
        onPress={onSwitchModel}
        activeOpacity={0.7}
        disabled={!onSwitchModel}
      >
        <Text style={styles.modelLabel}>Model:</Text>
        <Text style={styles.modelVal} numberOfLines={1}>
          {activeModelName}
        </Text>
        <View style={[styles.switchBadge, isTinyML ? styles.badgeTinyML : styles.badgeServer]}>
          <Text style={styles.switchBadgeText}>{isTinyML ? "TinyML INT8" : "Server GPU"} ⇄</Text>
        </View>
      </TouchableOpacity>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    backgroundColor: "rgba(13, 15, 26, 0.92)",
    borderWidth: 1,
    borderColor: THEME.colors.surfaceBorder,
    borderRadius: THEME.borderRadius.md,
    padding: THEME.spacing.md,
    marginHorizontal: THEME.spacing.md,
    marginBottom: THEME.spacing.sm,
  },
  headerRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: THEME.spacing.sm,
  },
  badge: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
  },
  liveDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: THEME.colors.accentNeon,
  },
  badgeText: {
    color: THEME.colors.accentNeon,
    fontSize: 10,
    fontWeight: THEME.typography.weights.heavy,
    letterSpacing: 0.5,
  },
  closeBtn: {
    paddingHorizontal: 6,
    paddingVertical: 2,
    backgroundColor: THEME.colors.surface,
    borderRadius: 4,
  },
  closeText: {
    color: THEME.colors.textSecondary,
    fontSize: 10,
  },
  grid: {
    flexDirection: "row",
    gap: THEME.spacing.sm,
    marginBottom: THEME.spacing.sm,
  },
  metricCard: {
    flex: 1,
    backgroundColor: THEME.colors.surface,
    padding: 6,
    borderRadius: 4,
    borderWidth: 1,
    borderColor: THEME.colors.surfaceBorder,
  },
  metricLabel: {
    color: THEME.colors.textMuted,
    fontSize: 9,
    fontWeight: THEME.typography.weights.bold,
    textTransform: "uppercase",
  },
  metricVal: {
    color: THEME.colors.textPrimary,
    fontSize: THEME.typography.sizes.sm,
    fontWeight: THEME.typography.weights.bold,
    marginTop: 2,
  },
  modelRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
    paddingTop: 4,
    borderTopWidth: 1,
    borderTopColor: "rgba(255, 255, 255, 0.05)",
  },
  modelLabel: {
    color: THEME.colors.textMuted,
    fontSize: 10,
  },
  modelVal: {
    color: THEME.colors.accentNeon,
    fontSize: 10,
    fontWeight: THEME.typography.weights.semibold,
    flex: 1,
  },
  floatingToggle: {
    position: "absolute",
    top: 90,
    right: 16,
    backgroundColor: THEME.colors.surface,
    borderColor: THEME.colors.accentNeon,
    borderWidth: 1,
    borderRadius: 6,
    paddingHorizontal: 10,
    paddingVertical: 5,
    zIndex: 50,
  },
  toggleText: {
    color: THEME.colors.accentNeon,
    fontSize: 11,
    fontWeight: THEME.typography.weights.bold,
  },
  switchBadge: {
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 4,
    borderWidth: 1,
  },
  badgeTinyML: {
    backgroundColor: "rgba(0, 240, 255, 0.15)",
    borderColor: THEME.colors.accentNeon,
  },
  badgeServer: {
    backgroundColor: "rgba(168, 85, 247, 0.15)",
    borderColor: "#A855F7",
  },
  switchBadgeText: {
    color: THEME.colors.textPrimary,
    fontSize: 9,
    fontWeight: THEME.typography.weights.bold,
  },
});
