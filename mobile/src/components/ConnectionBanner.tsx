import React, { useEffect, useState } from "react";
import { StyleSheet, Text, TouchableOpacity, View } from "react-native";
import { THEME } from "../constants/theme";
import { globalVisionBridge } from "../services/VisionBridgeService";

export const ConnectionBanner: React.FC = () => {
  const [isBridgeOnline, setIsBridgeOnline] = useState<boolean | null>(null);
  const [isChecking, setIsChecking] = useState<boolean>(false);

  const checkBridgeHealth = async () => {
    setIsChecking(true);
    try {
      const url = globalVisionBridge.getBridgeUrl();
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 1200);

      const res = await fetch(`${url}/health`, {
        method: "GET",
        signal: controller.signal,
      });
      clearTimeout(timeoutId);

      setIsBridgeOnline(res.ok);
    } catch {
      setIsBridgeOnline(false);
    } finally {
      setIsChecking(false);
    }
  };

  useEffect(() => {
    checkBridgeHealth();
    const interval = setInterval(checkBridgeHealth, 8000);
    return () => clearInterval(interval);
  }, []);

  // When online or not checked yet, show minimal badge or hide to keep UI focused
  if (isBridgeOnline === null || isBridgeOnline === true) {
    return null;
  }

  return (
    <View style={styles.banner}>
      <View style={styles.leftGroup}>
        <View style={styles.warnDot} />
        <Text style={styles.bannerText}>
          GPU Server Offline · Simulyasiya Rejimi Aktivdir
        </Text>
      </View>

      <TouchableOpacity
        style={styles.retryBtn}
        onPress={checkBridgeHealth}
        disabled={isChecking}
        activeOpacity={0.7}
      >
        <Text style={styles.retryText}>{isChecking ? "..." : "Sına"}</Text>
      </TouchableOpacity>
    </View>
  );
};

const styles = StyleSheet.create({
  banner: {
    backgroundColor: "rgba(210, 153, 34, 0.15)",
    borderColor: THEME.colors.statusWarning,
    borderWidth: 1,
    paddingHorizontal: THEME.spacing.md,
    paddingVertical: 6,
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
  },
  leftGroup: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
    flex: 1,
  },
  warnDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: THEME.colors.statusWarning,
  },
  bannerText: {
    color: THEME.colors.statusWarning,
    fontSize: 10,
    fontWeight: THEME.typography.weights.semibold,
  },
  retryBtn: {
    backgroundColor: THEME.colors.surface,
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderRadius: 4,
    borderWidth: 1,
    borderColor: THEME.colors.statusWarning,
  },
  retryText: {
    color: THEME.colors.statusWarning,
    fontSize: 10,
    fontWeight: THEME.typography.weights.bold,
  },
});
