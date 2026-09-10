import React from "react";
import { StyleSheet, Text, View } from "react-native";
import { getDenominationById } from "../constants/denominations";
import { THEME } from "../constants/theme";

export interface DetectionOverlayItem {
  bbox: [number, number, number, number]; // [x1, y1, x2, y2] normalized [0, 1]
  denominationId: number;
  confidence: number;
}

interface BoundingBoxOverlayProps {
  bbox?: [number, number, number, number] | null;
  denominationId?: number | null;
  confidence?: number | null;
  detections?: DetectionOverlayItem[];
  containerWidth: number;
  containerHeight: number;
  imageWidth?: number;
  imageHeight?: number;
}

export const BoundingBoxOverlay: React.FC<BoundingBoxOverlayProps> = ({
  bbox,
  denominationId,
  confidence,
  detections,
  containerWidth,
  containerHeight,
  imageWidth,
  imageHeight,
}) => {
  // Normalize to items list
  let items: DetectionOverlayItem[] = [];
  if (detections && detections.length > 0) {
    items = detections.filter(
      (d): d is DetectionOverlayItem =>
        Boolean(d.bbox) &&
        typeof d.denominationId === "number" &&
        typeof d.confidence === "number"
    );
  } else if (
    bbox &&
    typeof denominationId === "number" &&
    typeof confidence === "number"
  ) {
    items = [{ bbox, denominationId, confidence }];
  }

  if (items.length === 0) {
    return null;
  }

  const totalSum = items.reduce((sum, item) => {
    const config = getDenominationById(item.denominationId);
    return sum + (config.nominalValue || 0);
  }, 0);

  return (
    <View style={StyleSheet.absoluteFill} pointerEvents="none">
      {/* Floating Multi-Banknote Sum Banner if multiple items detected */}
      {items.length > 1 && (
        <View style={styles.multiSumBanner}>
          <Text style={styles.multiSumIcon}>💰</Text>
          <Text style={styles.multiSumText}>
            CƏMİ: <Text style={styles.multiSumAmount}>{totalSum} AZN</Text>
          </Text>
          <View style={styles.multiSumBadge}>
            <Text style={styles.multiSumBadgeText}>{items.length} Əsginas</Text>
          </View>
        </View>
      )}

      {items.map((item, index) => {
        const [nx1, ny1, nx2, ny2] = item.bbox;
        const imgW = imageWidth || containerWidth;
        const imgH = imageHeight || containerHeight;
        const imgAspect = imgW / Math.max(1, imgH);
        const containerAspect = containerWidth / Math.max(1, containerHeight);

        let scaledW = containerWidth;
        let scaledH = containerHeight;
        let offsetX = 0;
        let offsetY = 0;

        if (imgAspect > containerAspect) {
          scaledW = containerHeight * imgAspect;
          offsetX = (containerWidth - scaledW) / 2;
        } else {
          scaledH = containerWidth / imgAspect;
          offsetY = (containerHeight - scaledH) / 2;
        }

        const left = Math.max(0, offsetX + nx1 * scaledW);
        const top = Math.max(0, offsetY + ny1 * scaledH);
        const width = Math.max(20, (nx2 - nx1) * scaledW);
        const height = Math.max(20, (ny2 - ny1) * scaledH);

        const denom = getDenominationById(item.denominationId);
        const confPct = Math.round(item.confidence * 100);

        return (
          <View
            key={`bbox-${index}-${item.denominationId}`}
            style={[
              styles.box,
              {
                left,
                top,
                width,
                height,
                borderColor: denom.isHighValue ? THEME.colors.statusWarning : THEME.colors.accentNeon,
              },
            ]}
          >
            {/* Corner Brackets */}
            <View style={[styles.corner, styles.topLeft, denom.isHighValue && styles.cornerHighValue]} />
            <View style={[styles.corner, styles.topRight, denom.isHighValue && styles.cornerHighValue]} />
            <View style={[styles.corner, styles.bottomLeft, denom.isHighValue && styles.cornerHighValue]} />
            <View style={[styles.corner, styles.bottomRight, denom.isHighValue && styles.cornerHighValue]} />

            {/* Denomination Tag */}
            <View style={[styles.tag, denom.isHighValue && styles.tagHighValue]}>
              <Text style={[styles.tagName, denom.isHighValue && styles.tagNameHighValue]}>{denom.name}</Text>
              <Text style={styles.tagConf}>{confPct}%</Text>
              {denom.isHighValue && (
                <View style={styles.highValueBadge}>
                  <Text style={styles.highValueText}>DƏYƏRLİ</Text>
                </View>
              )}
            </View>
          </View>
        );
      })}
    </View>
  );
};

const styles = StyleSheet.create({
  box: {
    position: "absolute",
    borderWidth: 2,
    backgroundColor: "rgba(212, 249, 56, 0.08)",
    borderRadius: THEME.borderRadius.sm,
  },
  tag: {
    position: "absolute",
    top: -28,
    left: -2,
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
    backgroundColor: THEME.colors.background,
    borderWidth: 1.5,
    borderColor: THEME.colors.accentNeon,
    borderRadius: THEME.borderRadius.sm,
    paddingHorizontal: 8,
    paddingVertical: 3,
  },
  tagName: {
    color: THEME.colors.accentNeon,
    fontSize: THEME.typography.sizes.sm,
    fontWeight: THEME.typography.weights.heavy,
  },
  tagConf: {
    color: THEME.colors.textPrimary,
    fontSize: THEME.typography.sizes.xs,
    fontWeight: THEME.typography.weights.medium,
  },
  highValueBadge: {
    backgroundColor: THEME.colors.statusWarning,
    paddingHorizontal: 4,
    paddingVertical: 1,
    borderRadius: 3,
  },
  highValueText: {
    color: "#000000",
    fontSize: 9,
    fontWeight: THEME.typography.weights.heavy,
  },
  corner: {
    position: "absolute",
    width: 14,
    height: 14,
    borderColor: THEME.colors.accentNeon,
  },
  topLeft: {
    top: -2,
    left: -2,
    borderTopWidth: 4,
    borderLeftWidth: 4,
  },
  topRight: {
    top: -2,
    right: -2,
    borderTopWidth: 4,
    borderRightWidth: 4,
  },
  bottomLeft: {
    bottom: -2,
    left: -2,
    borderBottomWidth: 4,
    borderLeftWidth: 4,
  },
  bottomRight: {
    bottom: -2,
    right: -2,
    borderBottomWidth: 4,
    borderRightWidth: 4,
  },
  cornerHighValue: {
    borderColor: THEME.colors.statusWarning,
  },
  tagHighValue: {
    borderColor: THEME.colors.statusWarning,
  },
  tagNameHighValue: {
    color: THEME.colors.statusWarning,
  },
  multiSumBanner: {
    position: "absolute",
    top: 14,
    alignSelf: "center",
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    backgroundColor: "rgba(13, 15, 26, 0.92)",
    borderWidth: 1.5,
    borderColor: THEME.colors.accentNeon,
    borderRadius: THEME.borderRadius.md,
    paddingHorizontal: 16,
    paddingVertical: 8,
    zIndex: 50,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.4,
    shadowRadius: 8,
    elevation: 8,
  },
  multiSumIcon: {
    fontSize: 18,
  },
  multiSumText: {
    color: THEME.colors.textSecondary,
    fontSize: THEME.typography.sizes.sm,
    fontWeight: THEME.typography.weights.heavy,
    letterSpacing: 0.5,
  },
  multiSumAmount: {
    color: THEME.colors.accentNeon,
    fontSize: THEME.typography.sizes.base,
    fontWeight: THEME.typography.weights.heavy,
  },
  multiSumBadge: {
    backgroundColor: THEME.colors.accentNeonMuted,
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: THEME.borderRadius.sm,
    borderWidth: 1,
    borderColor: THEME.colors.accentNeon,
  },
  multiSumBadgeText: {
    color: THEME.colors.accentNeon,
    fontSize: 10,
    fontWeight: THEME.typography.weights.bold,
  },
});
