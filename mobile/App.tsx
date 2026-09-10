import React from "react";
import { SafeAreaView, StatusBar, StyleSheet } from "react-native";
import { ErrorBoundary } from "./src/components/ErrorBoundary";
import { WebMobileShell } from "./src/components/WebMobileShell";
import { THEME } from "./src/constants/theme";
import { CameraVisionScreen } from "./src/screens/CameraVisionScreen";

export default function App() {
  return (
    <WebMobileShell>
      <SafeAreaView style={styles.container}>
        <StatusBar barStyle="light-content" backgroundColor={THEME.colors.background} />
        <ErrorBoundary>
          <CameraVisionScreen />
        </ErrorBoundary>
      </SafeAreaView>
    </WebMobileShell>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: THEME.colors.background,
  },
});
