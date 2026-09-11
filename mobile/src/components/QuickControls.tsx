import React from "react";
import { StyleSheet, Text, TouchableOpacity, View } from "react-native";
import { THEME } from "../constants/theme";

interface QuickControlsProps {
  isTorchOn: boolean;
  onToggleTorch: () => void;
  isMuted: boolean;
  onToggleMute: () => void;
  onResetWallet: () => void;
  sourceMode: "PHONE_CAM" | "GLASSES_BLE";
  onToggleSource: () => void;
}

export const QuickControls: React.FC<QuickControlsProps> = ({
  isTorchOn,
  onToggleTorch,
  isMuted,
  onToggleMute,
  onResetWallet,
  sourceMode,
  onToggleSource,
}) => {
  return (
    <View style={styles.container}>
      {/* Torch Button */}
      <TouchableOpacity
        style={[styles.btn, isTorchOn && styles.btnActive]}
        onPress={onToggleTorch}
        activeOpacity={0.7}
      >
        <Text style={[styles.btnText, isTorchOn && styles.btnTextActive]}>
          {isTorchOn ? "Fənər: AÇIQ" : "Fənər"}
        </Text>
      </TouchableOpacity>

      {/* Mute Button */}
      <TouchableOpacity
        style={[styles.btn, isMuted && styles.btnWarn]}
        onPress={onToggleMute}
        activeOpacity={0.7}
      >
        <Text style={[styles.btnText, isMuted && styles.btnTextWarn]}>
          {isMuted ? "Səs: SUS" : "Səs: AÇIQ"}
        </Text>
      </TouchableOpacity>

      {/* Reset Wallet Button */}
      <TouchableOpacity
        style={styles.btn}
        onPress={onResetWallet}
        activeOpacity={0.7}
      >
        <Text style={styles.btnText}>Sıfırla</Text>
      </TouchableOpacity>

      {/* Source Toggle Button */}
      <TouchableOpacity
        style={[styles.btn, sourceMode === "GLASSES_BLE" && styles.btnInfo]}
        onPress={onToggleSource}
        activeOpacity={0.7}
      >
        <Text style={[styles.btnText, sourceMode === "GLASSES_BLE" && styles.btnTextInfo]}>
          {sourceMode === "GLASSES_BLE" ? "Eynək" : "Kamera"}
        </Text>
      </TouchableOpacity>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    backgroundColor: THEME.colors.background,
    paddingHorizontal: THEME.spacing.lg,
    paddingVertical: THEME.spacing.md,
    borderTopWidth: 1,
    borderTopColor: THEME.colors.surfaceBorder,
    gap: THEME.spacing.sm,
  },
  btn: {
    flex: 1,
    backgroundColor: THEME.colors.surface,
    borderColor: THEME.colors.surfaceBorder,
    borderWidth: 1,
    borderRadius: THEME.borderRadius.md,
    paddingVertical: 12,
    alignItems: "center",
    justifyContent: "center",
  },
  btnActive: {
    backgroundColor: THEME.colors.accentNeon,
    borderColor: THEME.colors.accentNeon,
  },
  btnWarn: {
    borderColor: THEME.colors.statusWarning,
  },
  btnInfo: {
    borderColor: THEME.colors.statusInfo,
  },
  btnText: {
    color: THEME.colors.textPrimary,
    fontSize: THEME.typography.sizes.xs,
    fontWeight: THEME.typography.weights.bold,
  },
  btnTextActive: {
    color: "#000000",
  },
  btnTextWarn: {
    color: THEME.colors.statusWarning,
  },
  btnTextInfo: {
    color: THEME.colors.statusInfo,
  },
});
