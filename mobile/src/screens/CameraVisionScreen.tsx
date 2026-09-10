import React, { useEffect, useRef, useState } from "react";
import { Dimensions, Image, StyleSheet, Text, View } from "react-native";
import { CameraView, useCameraPermissions } from "expo-camera";
import { BoundingBoxOverlay, DetectionOverlayItem } from "../components/BoundingBoxOverlay";
import { ConnectionBanner } from "../components/ConnectionBanner";
import { HeaderRibbon } from "../components/HeaderRibbon";
import { QuickControls } from "../components/QuickControls";
import { ScanHistoryModal } from "../components/ScanHistoryModal";
import { TechnicalHUD } from "../components/TechnicalHUD";
import { getDenominationById } from "../constants/denominations";
import { DEFAULT_MODEL_ID, SUPPORTED_MODELS } from "../constants/models";
import { THEME } from "../constants/theme";
import { globalAudioService } from "../services/AudioService";
import { GlassesDetection, globalBLEService } from "../services/BLEService";
import { globalDatabaseService } from "../services/db/DatabaseService";
import { globalHapticService } from "../services/HapticService";
import { liveSafetyGuardService } from "../services/SafetyGuardService";
import { BoundingBoxDetection, globalVisionBridge } from "../services/VisionBridgeService";
import { globalWalletService } from "../services/WalletService";

const { width: SCREEN_WIDTH, height: SCREEN_HEIGHT } = Dimensions.get("window");

export const CameraVisionScreen: React.FC = () => {
  const [permission, requestPermission] = useCameraPermissions();
  const [isTorchOn, setIsTorchOn] = useState<boolean>(false);
  const [isMuted, setIsMuted] = useState<boolean>(false);
  const [isHUDVisible, setIsHUDVisible] = useState<boolean>(false);
  const [isHistoryVisible, setIsHistoryVisible] = useState<boolean>(false);
  const [sourceMode, setSourceMode] = useState<"PHONE_CAM" | "GLASSES_BLE">("PHONE_CAM");
  const [guidanceMessage, setGuidanceMessage] = useState<string | null>(null);

  // ESP32 Smart Glasses Live Camera state
  const [glassesFrame, setGlassesFrame] = useState<string | null>(null);
  const [glassesStatusText, setGlassesStatusText] = useState<string>("ESP32-S3 kamerası axtarılır...");

  // Telemetry state
  const [walletBalance, setWalletBalance] = useState<number>(0);
  const [activeModelId, setActiveModelId] = useState<string>("yolo11m");
  const [latencyMs, setLatencyMs] = useState<number>(32.0);
  const [fps, setFps] = useState<number>(60);
  const [batteryPct, setBatteryPct] = useState<number>(92);
  const [rssi, setRssi] = useState<number>(-62);

  // Active Detection state (supports both single & multiple simultaneous banknotes)
  const [activeDetections, setActiveDetections] = useState<DetectionOverlayItem[]>([]);
  const [activeBBox, setActiveBBox] = useState<[number, number, number, number] | null>(null);
  const [activeDenomId, setActiveDenomId] = useState<number | null>(null);
  const [activeConfidence, setActiveConfidence] = useState<number | null>(null);

  // Language state (defaults to English as configured in AudioService)
  const [currentLanguage, setCurrentLanguage] = useState<"en" | "az">(globalAudioService.getLanguage());

  // Dynamic Viewfinder layout measurement for responsive bounding box rendering
  const [viewfinderSize, setViewfinderSize] = useState<{ width: number; height: number }>({
    width: SCREEN_WIDTH,
    height: SCREEN_HEIGHT * 0.65,
  });

  // Source camera image dimensions for optical aspect ratio alignment
  const [imageDimensions, setImageDimensions] = useState<{ width: number; height: number } | null>(null);

  const clearTimerRef = useRef<any>(null);
  const guidanceTimerRef = useRef<any>(null);
  const cameraRef = useRef<any>(null);
  const isScanningRef = useRef<boolean>(false);

  // Initialize SQLite database and restore persisted wallet on startup
  useEffect(() => {
    globalDatabaseService.init().catch(() => {});
    globalWalletService.init().catch(() => {});
  }, []);

  // Subscribe to wallet updates
  useEffect(() => {
    const unsubWallet = globalWalletService.subscribe((balance) => {
      setWalletBalance(balance);
    });
    return unsubWallet;
  }, []);

  // Request camera permissions on mount
  useEffect(() => {
    if (!permission?.granted) {
      requestPermission();
    }
  }, [permission]);

  // Continuous camera frame capture loop in PHONE_CAM mode
  useEffect(() => {
    if (sourceMode !== "PHONE_CAM" || !permission?.granted) return;

    const intervalId = setInterval(async () => {
      if (isScanningRef.current) return;
      isScanningRef.current = true;

      try {
        let base64Frame: string | null = null;

        if (cameraRef.current && typeof cameraRef.current.takePictureAsync === "function") {
          try {
            const photo = await cameraRef.current.takePictureAsync({
              quality: 0.25,
              base64: true,
              shutterSound: false,
              skipProcessing: false,
            });
            if (photo?.base64) {
              base64Frame = photo.base64;
            }
          } catch {
            // Native camera capture error or busy
          }
        }

        // Web preview fallback (HTML5 video element in browser)
        if (!base64Frame && typeof document !== "undefined") {
          try {
            const video = document.querySelector("video") as HTMLVideoElement | null;
            if (video && video.videoWidth > 0 && video.videoHeight > 0) {
              const canvas = document.createElement("canvas");
              canvas.width = 320;
              canvas.height = Math.round((video.videoHeight / video.videoWidth) * 320);
              const ctx = canvas.getContext("2d");
              if (ctx) {
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                const dataUrl = canvas.toDataURL("image/jpeg", 0.5);
                base64Frame = dataUrl.replace(/^data:image\/[a-z]+;base64,/, "");
              }
            }
          } catch {
            // Web canvas error
          }
        }

        if (base64Frame) {
          const detections = await globalVisionBridge.analyzeFrame(base64Frame, activeModelId);
          if (detections && detections.length > 0) {
            handleIncomingDetections(detections, "PHONE_CAM", base64Frame);
          }
        }
      } catch {
        // General loop error
      } finally {
        isScanningRef.current = false;
      }
    }, 750);

    return () => clearInterval(intervalId);
  }, [sourceMode, permission, activeModelId]);

  // Continuous live hardware camera & TinyML inference loop in GLASSES_BLE mode
  useEffect(() => {
    if (sourceMode !== "GLASSES_BLE") {
      setGlassesFrame(null);
      return;
    }

    const intervalId = setInterval(async () => {
      if (isScanningRef.current) return;
      isScanningRef.current = true;

      try {
        const liveData = await globalVisionBridge.fetchGlassesLive();
        if (liveData) {
          if (liveData.frame) {
            setGlassesFrame(liveData.frame);
            setGlassesStatusText("ESP32-S3 Canlı Kamera Yayımı Aktivdir");
          }
          if (liveData.telemetry) {
            if (typeof liveData.telemetry.instant_fps === "number") {
              setFps(Math.round(liveData.telemetry.instant_fps));
            }
            if (typeof liveData.telemetry.estimated_chip_temp_c === "number") {
              setBatteryPct(
                Math.min(100, Math.max(20, Math.round(100 - (liveData.telemetry.estimated_chip_temp_c - 25) * 1.5)))
              );
            }
          }
          if (liveData.detections && liveData.detections.length > 0) {
            handleIncomingDetections(liveData.detections, "GLASSES_BLE", liveData.frame);
          }
        } else {
          setGlassesStatusText("ESP32-S3 qoşulmayıb və ya COM port axtarılır...");
        }
      } catch {
        setGlassesStatusText("ESP32-S3 rabitə xətası");
      } finally {
        isScanningRef.current = false;
      }
    }, 1000);

    return () => clearInterval(intervalId);
  }, [sourceMode, activeModelId]);

  /**
   * Evaluates single or multiple incoming detections, verifies safety gates,
   * announces total sum in Azerbaijani via TTS, triggers haptic, and records in wallet/DB.
   */
  const handleIncomingDetections = (
    detections: Array<{
      denominationId: number;
      confidence: number;
      bbox: [number, number, number, number];
      latencyMs?: number;
      imageWidth?: number;
      imageHeight?: number;
    }>,
    source: "PHONE_CAM" | "GLASSES_BLE" | "SAMPLE" = "PHONE_CAM",
    imageBase64?: string
  ) => {
    if (!detections || detections.length === 0) return;

    const first = detections[0];
    if (first?.imageWidth && first?.imageHeight) {
      setImageDimensions({ width: first.imageWidth, height: first.imageHeight });
    }

    // Filter detections through safety guardrails
    const acceptedDetections = detections.filter((det) => {
      const safetyCheck = liveSafetyGuardService.verifyDetection({
        denominationId: det.denominationId,
        confidence: det.confidence,
        bbox: det.bbox,
      });

      if (!safetyCheck.isAccepted && safetyCheck.guidanceMessage) {
        setGuidanceMessage(safetyCheck.guidanceMessage);
        if (guidanceTimerRef.current) clearTimeout(guidanceTimerRef.current);
        guidanceTimerRef.current = setTimeout(() => setGuidanceMessage(null), 2500);
      }
      return safetyCheck.isAccepted;
    });

    if (acceptedDetections.length === 0) return;

    setGuidanceMessage(null);
    setActiveDetections(
      acceptedDetections.map((d) => ({
        denominationId: d.denominationId,
        confidence: d.confidence,
        bbox: d.bbox,
      }))
    );

    const top = acceptedDetections[0];
    setActiveBBox(top.bbox);
    setActiveDenomId(top.denominationId);
    setActiveConfidence(top.confidence);
    setLatencyMs(top.latencyMs || (activeModelId === "yolo_fastestv2" ? 42.0 : 5.2));

    // Calculate sum and trigger multi-banknote Azerbaijani voice announcement
    const denomIds = acceptedDetections.map((d) => d.denominationId);
    const announced = globalAudioService.announceDetections(denomIds);

    // If speech was triggered (not debounced), provide haptic and update wallet
    if (announced) {
      const hasHighValue = denomIds.some((id) => id >= 4);
      globalHapticService.triggerDetectionFeedback(hasHighValue);
      globalWalletService.registerBanknotes(denomIds);

      // Persist detections to SQLite database
      for (const det of acceptedDetections) {
        const denomConfig = getDenominationById(det.denominationId);
        globalDatabaseService
          .insertScan(
            {
              denominationId: det.denominationId,
              nominalValue: denomConfig.nominalValue,
              name: denomConfig.name,
              confidence: det.confidence,
              source,
              modelId: activeModelId,
              latencyMs: det.latencyMs || 42.0,
              bbox: det.bbox,
              timestamp: Date.now(),
            },
            imageBase64
          )
          .catch(() => {});
      }
    }

    // Auto-clear bounding boxes after 2.5s if no new detections arrive
    if (clearTimerRef.current) {
      clearTimeout(clearTimerRef.current);
    }
    clearTimerRef.current = setTimeout(() => {
      setActiveDetections([]);
      setActiveBBox(null);
      setActiveDenomId(null);
      setActiveConfidence(null);
    }, 2500);
  };

  // Backwards-compatible single-detection handler
  const handleIncomingDetection = (
    denomId: number,
    confidence: number,
    bbox: [number, number, number, number],
    detectedLatency: number = 5.2,
    source: "PHONE_CAM" | "GLASSES_BLE" | "SAMPLE" = "PHONE_CAM",
    imageBase64?: string
  ) => {
    handleIncomingDetections(
      [{ denominationId: denomId, confidence, bbox, latencyMs: detectedLatency }],
      source,
      imageBase64
    );
  };

  const handleToggleTorch = () => {
    setIsTorchOn((prev) => !prev);
  };

  const handleToggleMute = () => {
    const nextMuted = globalAudioService.toggleMute();
    setIsMuted(nextMuted);
  };

  const handleResetWallet = () => {
    globalWalletService.resetWallet();
    globalAudioService.resetState();
    liveSafetyGuardService.resetHistory();
    setActiveDetections([]);
    setActiveBBox(null);
    setActiveDenomId(null);
    setActiveConfidence(null);
  };

  const handleToggleSource = () => {
    if (sourceMode === "PHONE_CAM") {
      setSourceMode("GLASSES_BLE");
    } else {
      setSourceMode("PHONE_CAM");
    }
  };

  const handleToggleModel = () => {
    const nextModel = activeModelId === "yolo_fastestv2" ? "yolo11m" : "yolo_fastestv2";
    setActiveModelId(nextModel);
    globalVisionBridge.switchBridgeModel(nextModel).catch(() => {});
  };

  const handleToggleLanguage = () => {
    const nextLang: "en" | "az" = currentLanguage === "en" ? "az" : "en";
    setCurrentLanguage(nextLang);
    globalAudioService.setLanguage(nextLang);
    globalHapticService.triggerSelection();
  };

  const handleViewfinderLayout = (event: any) => {
    const { width, height } = event.nativeEvent.layout;
    if (width > 0 && height > 0) {
      setViewfinderSize({ width, height });
    }
  };

  const activeModelDescriptor = SUPPORTED_MODELS[activeModelId] || SUPPORTED_MODELS.champion_auto;

  return (
    <View style={styles.container}>
      {/* Top Header Ribbon with Clickable Wallet Audit & Language Switcher */}
      <HeaderRibbon
        walletBalance={walletBalance}
        sourceMode={sourceMode}
        currentLanguage={currentLanguage}
        onToggleLanguage={handleToggleLanguage}
        onOpenHistory={() => setIsHistoryVisible(true)}
      />

      {/* Network / GPU Server Connection Status Banner */}
      <ConnectionBanner />

      {/* Main Viewfinder Section */}
      <View style={styles.viewfinder} onLayout={handleViewfinderLayout}>
        {sourceMode === "PHONE_CAM" ? (
          permission?.granted ? (
            <CameraView
              ref={cameraRef}
              style={StyleSheet.absoluteFill}
              enableTorch={isTorchOn}
              facing="back"
              animateShutter={false}
            />
          ) : (
            <View style={styles.simulatedFeed}>
              <View style={styles.reticleCrosshair} />
              <Text style={styles.feedStatusText}>
                Kamera icazəsi gözlənilir və ya simulyasiya rejimindədir
              </Text>
            </View>
          )
        ) : glassesFrame ? (
          <View style={StyleSheet.absoluteFill}>
            <Image
              source={{
                uri: glassesFrame.startsWith("data:")
                  ? glassesFrame
                  : `data:image/jpeg;base64,${glassesFrame}`,
              }}
              style={StyleSheet.absoluteFill}
              resizeMode="cover"
            />
            <View style={styles.liveGlassesBadge}>
              <View style={styles.liveDot} />
              <Text style={styles.liveGlassesBadgeText}>CANLI ESP32-S3 KAMERA</Text>
            </View>
          </View>
        ) : (
          <View style={styles.simulatedFeed}>
            <View style={styles.reticleCrosshair} />
            <Text style={styles.feedStatusText}>{glassesStatusText}</Text>
          </View>
        )}

        {/* Safety Guidance Toast */}
        {guidanceMessage && (
          <View style={styles.guidanceToast}>
            <Text style={styles.guidanceText}>💡 {guidanceMessage}</Text>
          </View>
        )}

        {/* Bounding Box Overlay (Single & Multi-banknote Total Sum) */}
        <BoundingBoxOverlay
          bbox={activeBBox}
          denominationId={activeDenomId}
          confidence={activeConfidence}
          detections={activeDetections}
          containerWidth={viewfinderSize.width}
          containerHeight={viewfinderSize.height}
          imageWidth={imageDimensions?.width}
          imageHeight={imageDimensions?.height}
        />

        {/* Technical HUD Overlay */}
        <View style={styles.hudWrapper}>
          <TechnicalHUD
            fps={fps}
            latencyMs={latencyMs}
            activeModelName={activeModelDescriptor.displayName}
            sourceMode={sourceMode}
            batteryPct={batteryPct}
            rssi={rssi}
            isVisible={isHUDVisible}
            onToggle={() => setIsHUDVisible((prev) => !prev)}
            onSwitchModel={handleToggleModel}
          />
        </View>
      </View>

      {/* Bottom Quick Controls */}
      <QuickControls
        isTorchOn={isTorchOn}
        onToggleTorch={handleToggleTorch}
        isMuted={isMuted}
        onToggleMute={handleToggleMute}
        onResetWallet={handleResetWallet}
        sourceMode={sourceMode}
        onToggleSource={handleToggleSource}
      />

      {/* Scan History & Audit Modal */}
      <ScanHistoryModal
        isVisible={isHistoryVisible}
        onClose={() => setIsHistoryVisible(false)}
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: THEME.colors.background,
  },
  viewfinder: {
    flex: 1,
    overflow: "hidden",
    position: "relative",
    backgroundColor: "#05070D",
  },
  simulatedFeed: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: THEME.colors.backgroundAlt,
  },
  liveGlassesBadge: {
    position: "absolute",
    top: 14,
    left: 14,
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
    backgroundColor: "rgba(13, 15, 26, 0.88)",
    borderWidth: 1,
    borderColor: THEME.colors.statusSuccess,
    borderRadius: THEME.borderRadius.sm,
    paddingHorizontal: 10,
    paddingVertical: 5,
    zIndex: 30,
  },
  liveDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: THEME.colors.statusSuccess,
  },
  liveGlassesBadgeText: {
    color: THEME.colors.statusSuccess,
    fontSize: 10,
    fontWeight: THEME.typography.weights.heavy,
    letterSpacing: 0.5,
  },
  reticleCrosshair: {
    width: 140,
    height: 140,
    borderColor: "rgba(212, 249, 56, 0.3)",
    borderWidth: 1,
    borderStyle: "dashed",
    borderRadius: 70,
    marginBottom: 16,
  },
  feedStatusText: {
    color: THEME.colors.textMuted,
    fontSize: THEME.typography.sizes.xs,
    textAlign: "center",
    paddingHorizontal: 32,
  },
  guidanceToast: {
    position: "absolute",
    top: 20,
    alignSelf: "center",
    backgroundColor: "rgba(13, 15, 26, 0.9)",
    borderWidth: 1,
    borderColor: THEME.colors.statusWarning,
    borderRadius: THEME.borderRadius.md,
    paddingHorizontal: 16,
    paddingVertical: 8,
    zIndex: 40,
  },
  guidanceText: {
    color: THEME.colors.statusWarning,
    fontSize: THEME.typography.sizes.xs,
    fontWeight: THEME.typography.weights.bold,
  },
  hudWrapper: {
    position: "absolute",
    bottom: 0,
    left: 0,
    right: 0,
  },
});
