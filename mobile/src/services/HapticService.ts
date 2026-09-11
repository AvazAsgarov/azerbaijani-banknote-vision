/**
 * Tactile Feedback (Haptics) Service for Visually Impaired Users.
 *
 * Emits distinct vibration impulses upon confirmed banknote detection.
 */

export type HapticAdapter = (type: "light" | "medium" | "heavy" | "success") => Promise<void> | void;

export class HapticService {
  private hapticAdapter: HapticAdapter;
  private isEnabled: boolean = true;

  constructor(adapter?: HapticAdapter) {
    this.hapticAdapter = adapter || this.defaultHapticAdapter;
  }

  private defaultHapticAdapter: HapticAdapter = async (type) => {
    try {
      const Haptics = require("expo-haptics");
      if (type === "success") {
        await Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
      } else if (type === "heavy") {
        await Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Heavy);
      } else if (type === "medium") {
        await Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
      } else {
        await Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
      }
    } catch {
      // Graceful fallback when running in simulator or non-supported device
    }
  };

  public setEnabled(enabled: boolean): void {
    this.isEnabled = enabled;
  }

  public triggerDetectionFeedback(isHighValue: boolean = false): void {
    if (!this.isEnabled) return;
    if (isHighValue) {
      this.hapticAdapter("success");
    } else {
      this.hapticAdapter("medium");
    }
  }

  public triggerSelection(): void {
    if (!this.isEnabled) return;
    this.hapticAdapter("light");
  }
}

export const globalHapticService = new HapticService();
