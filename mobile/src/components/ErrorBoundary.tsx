import React, { Component, ErrorInfo, ReactNode } from "react";
import { SafeAreaView, StyleSheet, Text, TouchableOpacity, View } from "react-native";
import { THEME } from "../constants/theme";

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null,
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    console.error("ErrorBoundary caught an unhandled error:", error, errorInfo);
  }

  private handleReset = (): void => {
    this.setState({ hasError: false, error: null });
  };

  public render(): ReactNode {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <SafeAreaView style={styles.container}>
          <View style={styles.content}>
            <View style={styles.iconCircle}>
              <Text style={styles.iconText}>⚠️</Text>
            </View>

            <Text style={styles.title}>Gözlənilməz Xəta Baş Verdi</Text>
            <Text style={styles.subtitle}>
              Tətbiqin icrası zamanı xəta qeydə alındı. Demo rejimində davam etmək üçün yenidən başladın.
            </Text>

            <View style={styles.errorBox}>
              <Text style={styles.errorLabel}>Xəta Məlumatı:</Text>
              <Text style={styles.errorText} numberOfLines={4}>
                {this.state.error?.message || "Naməlum xəta"}
              </Text>
            </View>

            <TouchableOpacity style={styles.retryBtn} onPress={this.handleReset} activeOpacity={0.8}>
              <Text style={styles.retryBtnText}>Tətbiqi Yenidən Başlat 🔄</Text>
            </TouchableOpacity>
          </View>
        </SafeAreaView>
      );
    }

    return this.props.children;
  }
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: THEME.colors.background,
  },
  content: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    paddingHorizontal: THEME.spacing.xl,
  },
  iconCircle: {
    width: 64,
    height: 64,
    borderRadius: 32,
    backgroundColor: THEME.colors.surface,
    borderWidth: 1.5,
    borderColor: THEME.colors.statusWarning,
    justifyContent: "center",
    alignItems: "center",
    marginBottom: THEME.spacing.lg,
  },
  iconText: {
    fontSize: 28,
  },
  title: {
    color: THEME.colors.textPrimary,
    fontSize: THEME.typography.sizes.xl,
    fontWeight: THEME.typography.weights.heavy,
    textAlign: "center",
    marginBottom: THEME.spacing.sm,
  },
  subtitle: {
    color: THEME.colors.textSecondary,
    fontSize: THEME.typography.sizes.sm,
    textAlign: "center",
    lineHeight: 20,
    marginBottom: THEME.spacing.xl,
  },
  errorBox: {
    width: "100%",
    backgroundColor: THEME.colors.surface,
    borderWidth: 1,
    borderColor: THEME.colors.surfaceBorder,
    borderRadius: THEME.borderRadius.md,
    padding: THEME.spacing.md,
    marginBottom: THEME.spacing.xl,
  },
  errorLabel: {
    color: THEME.colors.textMuted,
    fontSize: 10,
    fontWeight: THEME.typography.weights.bold,
    textTransform: "uppercase",
    marginBottom: 4,
  },
  errorText: {
    color: THEME.colors.statusDanger,
    fontSize: THEME.typography.sizes.xs,
    fontFamily: THEME.typography.fontFamilyMono,
  },
  retryBtn: {
    width: "100%",
    backgroundColor: THEME.colors.accentNeon,
    paddingVertical: 14,
    borderRadius: THEME.borderRadius.md,
    alignItems: "center",
  },
  retryBtnText: {
    color: "#000000",
    fontSize: THEME.typography.sizes.base,
    fontWeight: THEME.typography.weights.heavy,
  },
});
