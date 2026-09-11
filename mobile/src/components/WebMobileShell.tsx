import React, { ReactNode } from "react";
import { Platform, StyleSheet, Text, View } from "react-native";
import { THEME } from "../constants/theme";

interface WebMobileShellProps {
  children: ReactNode;
}

export const WebMobileShell: React.FC<WebMobileShellProps> = ({ children }) => {
  if (Platform.OS !== "web") {
    return <>{children}</>;
  }

  return (
    <View style={styles.desktopWorkspace}>
      {/* Desktop Top Ribbon */}
      <View style={styles.desktopHeader}>
        <View>
          <Text style={styles.desktopTitle}>AZN-Vision · Mobil Önizləmə Simulyatoru</Text>
          <Text style={styles.desktopSubtitle}>
            iPhone 15 Pro Çərçivəsi · 393 × 852 pt
          </Text>
        </View>

        <View style={styles.badgePill}>
          <View style={styles.liveDot} />
          <Text style={styles.badgeText}>CANLI ÖNİZLƏMƏ</Text>
        </View>
      </View>

      {/* Realistic Smartphone Mockup Chassis */}
      <View style={styles.phoneFrameWrapper}>
        <View style={styles.phoneFrame}>
          {/* Top Dynamic Island / Notch */}
          <View style={styles.dynamicIslandWrapper}>
            <View style={styles.dynamicIsland}>
              <View style={styles.cameraLens} />
              <View style={styles.speakerPill} />
            </View>
          </View>

          {/* Actual Mobile App Screen Content */}
          <View style={styles.phoneScreen}>
            {children}
          </View>

          {/* Bottom Home Indicator */}
          <View style={styles.homeIndicatorWrapper}>
            <View style={styles.homeIndicator} />
          </View>
        </View>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  desktopWorkspace: {
    flex: 1,
    minHeight: "100vh" as any,
    backgroundColor: "#07080E",
    alignItems: "center",
    justifyContent: "flex-start",
    paddingVertical: 16,
    paddingHorizontal: 16,
  },
  desktopHeader: {
    width: "100%",
    maxWidth: 460,
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 14,
    paddingHorizontal: 8,
  },
  desktopTitle: {
    color: THEME.colors.textPrimary,
    fontSize: THEME.typography.sizes.sm,
    fontWeight: THEME.typography.weights.heavy,
    letterSpacing: 0.5,
  },
  desktopSubtitle: {
    color: THEME.colors.textMuted,
    fontSize: 10,
    marginTop: 2,
  },
  badgePill: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
    backgroundColor: THEME.colors.surface,
    borderColor: THEME.colors.accentNeon,
    borderWidth: 1,
    borderRadius: THEME.borderRadius.full,
    paddingHorizontal: 10,
    paddingVertical: 4,
  },
  liveDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: THEME.colors.accentNeon,
  },
  badgeText: {
    color: THEME.colors.accentNeon,
    fontSize: 9,
    fontWeight: THEME.typography.weights.bold,
  },
  phoneFrameWrapper: {
    padding: 10,
    backgroundColor: "#161924",
    borderRadius: 56,
    borderWidth: 2,
    borderColor: "#2B3147",
    shadowColor: "#000000",
    shadowOffset: { width: 0, height: 16 },
    shadowOpacity: 0.6,
    shadowRadius: 32,
    elevation: 24,
  },
  phoneFrame: {
    width: 393,
    height: 840,
    backgroundColor: THEME.colors.background,
    borderRadius: 46,
    overflow: "hidden",
    position: "relative",
    borderWidth: 4,
    borderColor: "#0B0D15",
  },
  dynamicIslandWrapper: {
    position: "absolute",
    top: 10,
    left: 0,
    right: 0,
    alignItems: "center",
    zIndex: 100,
    pointerEvents: "none",
  },
  dynamicIsland: {
    width: 110,
    height: 30,
    backgroundColor: "#000000",
    borderRadius: 15,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingHorizontal: 10,
  },
  cameraLens: {
    width: 10,
    height: 10,
    borderRadius: 5,
    backgroundColor: "#0F142A",
    borderWidth: 1.5,
    borderColor: "#1E2545",
  },
  speakerPill: {
    width: 50,
    height: 4,
    borderRadius: 2,
    backgroundColor: "#1A1D2B",
  },
  phoneScreen: {
    flex: 1,
    backgroundColor: THEME.colors.background,
    paddingTop: 4,
  },
  homeIndicatorWrapper: {
    position: "absolute",
    bottom: 6,
    left: 0,
    right: 0,
    alignItems: "center",
    zIndex: 100,
    pointerEvents: "none",
  },
  homeIndicator: {
    width: 134,
    height: 5,
    borderRadius: 3,
    backgroundColor: "rgba(255, 255, 255, 0.4)",
  },
});
