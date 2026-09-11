import React, { useEffect, useState } from "react";
import {
  Alert,
  FlatList,
  Image,
  Modal,
  SafeAreaView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from "react-native";
import { DENOMINATION_LIST, getDenominationById } from "../constants/denominations";
import { THEME } from "../constants/theme";
import { globalAudioService } from "../services/AudioService";
import { DatabaseService, globalDatabaseService, ScanRecord, ScanStats } from "../services/db/DatabaseService";

interface ScanHistoryModalProps {
  isVisible: boolean;
  onClose: () => void;
  databaseService?: DatabaseService;
}

export const ScanHistoryModal: React.FC<ScanHistoryModalProps> = ({
  isVisible,
  onClose,
  databaseService = globalDatabaseService,
}) => {
  const [scans, setScans] = useState<ScanRecord[]>([]);
  const [stats, setStats] = useState<ScanStats>({ totalCount: 0, totalAmount: 0, breakdown: {} });
  const [selectedDenomFilter, setSelectedDenomFilter] = useState<number | null>(null);

  const loadData = async () => {
    const list = await databaseService.getScans(100, 0, selectedDenomFilter ?? undefined);
    const aggregate = await databaseService.getStats();
    setScans(list);
    setStats(aggregate);
  };

  useEffect(() => {
    if (isVisible) {
      loadData();
    }
  }, [isVisible, selectedDenomFilter]);

  // Subscribe to DB changes
  useEffect(() => {
    const unsub = databaseService.subscribe(() => {
      loadData();
    });
    return unsub;
  }, [selectedDenomFilter]);

  const handleSpeakSummary = () => {
    const isEn = globalAudioService.getLanguage() === "en";
    const text = isEn
      ? `Total ${stats.totalCount} banknotes scanned. Total amount is ${stats.totalAmount} Manat.`
      : `Ümumi ${stats.totalCount} əsginas skan edilib. Cəmi məbləğ ${stats.totalAmount} manat təşkil edir.`;
    // Announce through audio service
    globalAudioService.announceDetection(-1); // Reset state
    try {
      const Speech = require("expo-speech");
      Speech.stop();
      Speech.speak(text, { language: isEn ? "en-US" : "az-AZ", rate: 0.95 });
    } catch {
      // Fallback
    }
  };

  const handleClearHistory = () => {
    Alert.alert(
      "Tarixçəni Sil",
      "Bütün skan tarixçəsini və yaddaşda saxlanılan kadrları silmək istədiyinizə əminsiniz?",
      [
        { text: "Ləğv et", style: "cancel" },
        {
          text: "Sil",
          style: "destructive",
          onPress: async () => {
            await databaseService.clearAllScans();
            await loadData();
          },
        },
      ]
    );
  };

  const formatTime = (ts: number): string => {
    const d = new Date(ts);
    const timeStr = d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
    const dateStr = `${d.getDate().toString().padStart(2, "0")}.${(d.getMonth() + 1).toString().padStart(2, "0")}`;
    return `${timeStr} · ${dateStr}`;
  };

  const renderScanItem = ({ item }: { item: ScanRecord }) => {
    const denom = getDenominationById(item.denominationId);
    const confPct = Math.round(item.confidence * 100);

    return (
      <View style={styles.card}>
        <View style={styles.cardHeader}>
          {/* Banknote Badge / Image */}
          <View style={styles.leftInfo}>
            {item.imageUri ? (
              <Image source={{ uri: item.imageUri }} style={styles.thumbnail} />
            ) : (
              <View style={[styles.colorSquare, { backgroundColor: denom.colorHex }]}>
                <Text style={styles.nominalBadgeText}>{denom.nominalValue} ₼</Text>
              </View>
            )}

            <View style={styles.nameGroup}>
              <Text style={styles.cardTitle}>{denom.name}</Text>
              <Text style={styles.cardTimestamp}>{formatTime(item.timestamp)}</Text>
            </View>
          </View>

          {/* Amount Pill */}
          <View style={styles.amountPill}>
            <Text style={styles.amountText}>+{denom.nominalValue} ₼</Text>
          </View>
        </View>

        {/* Card Metadata Footer */}
        <View style={styles.cardFooter}>
          <View style={styles.metaBadge}>
            <Text style={styles.metaLabel}>Güvən:</Text>
            <Text style={styles.metaValue}>{confPct}%</Text>
          </View>

          <View style={styles.metaBadge}>
            <Text style={styles.metaLabel}>Mənbə:</Text>
            <Text style={styles.metaValue}>
              {item.source === "GLASSES_BLE" ? "Eynək" : item.source === "PHONE_CAM" ? "Kamera" : "Test"}
            </Text>
          </View>

          <View style={styles.metaBadge}>
            <Text style={styles.metaLabel}>Gecikmə:</Text>
            <Text style={styles.metaValue}>{item.latencyMs.toFixed(1)} ms</Text>
          </View>

          {denom.isHighValue && (
            <View style={styles.highValueBadge}>
              <Text style={styles.highValueText}>DƏYƏRLİ</Text>
            </View>
          )}
        </View>
      </View>
    );
  };

  return (
    <Modal visible={isVisible} animationType="slide" onRequestClose={onClose}>
      <SafeAreaView style={styles.container}>
        {/* Header */}
        <View style={styles.header}>
          <View>
            <Text style={styles.headerTitle}>Skan Tarixçəsi & Audit</Text>
            <Text style={styles.headerSubtitle}>SQLite Yaddaş Qeydləri</Text>
          </View>
          <TouchableOpacity onPress={onClose} style={styles.closeBtn} activeOpacity={0.7}>
            <Text style={styles.closeBtnText}>Bağla</Text>
          </TouchableOpacity>
        </View>

        {/* Aggregation Summary Banner */}
        <View style={styles.summaryBanner}>
          <View style={styles.summaryStats}>
            <View>
              <Text style={styles.summaryLabel}>ÜMUMİ MƏBLƏĞ</Text>
              <Text style={styles.summaryValue}>{stats.totalAmount} ₼</Text>
            </View>
            <View style={styles.statDivider} />
            <View>
              <Text style={styles.summaryLabel}>SKAN SAYI</Text>
              <Text style={styles.summaryValue}>{stats.totalCount}</Text>
            </View>
          </View>

          <View style={styles.bannerActions}>
            <TouchableOpacity style={styles.audioActionBtn} onPress={handleSpeakSummary} activeOpacity={0.8}>
              <Text style={styles.audioActionText}>Səsləndir 🔊</Text>
            </TouchableOpacity>

            <TouchableOpacity style={styles.clearActionBtn} onPress={handleClearHistory} activeOpacity={0.8}>
              <Text style={styles.clearActionText}>Təmizlə</Text>
            </TouchableOpacity>
          </View>
        </View>

        {/* Denomination Filter Chips */}
        <View style={styles.filterBar}>
          <TouchableOpacity
            style={[styles.filterChip, selectedDenomFilter === null && styles.filterChipActive]}
            onPress={() => setSelectedDenomFilter(null)}
          >
            <Text style={[styles.filterChipText, selectedDenomFilter === null && styles.filterChipTextActive]}>
              Hamısı
            </Text>
          </TouchableOpacity>

          {DENOMINATION_LIST.map((d) => {
            const isSelected = selectedDenomFilter === d.id;
            return (
              <TouchableOpacity
                key={d.id}
                style={[styles.filterChip, isSelected && styles.filterChipActive]}
                onPress={() => setSelectedDenomFilter(d.id)}
              >
                <Text style={[styles.filterChipText, isSelected && styles.filterChipTextActive]}>
                  {d.nominalValue}₼
                </Text>
              </TouchableOpacity>
            );
          })}
        </View>

        {/* List of Scans */}
        {scans.length === 0 ? (
          <View style={styles.emptyContainer}>
            <Text style={styles.emptyTitle}>Hələ heç bir əsginas qeydə alınmayıb</Text>
            <Text style={styles.emptySub}>
              Kameranı əsginasa yaxınlaşdırın və ya test panelindən nümunə seçin.
            </Text>
          </View>
        ) : (
          <FlatList
            data={scans}
            keyExtractor={(item) => item.id}
            renderItem={renderScanItem}
            contentContainerStyle={styles.listContent}
          />
        )}
      </SafeAreaView>
    </Modal>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: THEME.colors.background,
  },
  header: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    paddingHorizontal: THEME.spacing.lg,
    paddingTop: THEME.spacing.md,
    paddingBottom: THEME.spacing.md,
    borderBottomWidth: 1,
    borderBottomColor: THEME.colors.surfaceBorder,
  },
  headerTitle: {
    color: THEME.colors.textPrimary,
    fontSize: THEME.typography.sizes.lg,
    fontWeight: THEME.typography.weights.bold,
  },
  headerSubtitle: {
    color: THEME.colors.textSecondary,
    fontSize: THEME.typography.sizes.xs,
  },
  closeBtn: {
    backgroundColor: THEME.colors.surface,
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: THEME.borderRadius.sm,
    borderWidth: 1,
    borderColor: THEME.colors.surfaceBorder,
  },
  closeBtnText: {
    color: THEME.colors.accentNeon,
    fontSize: THEME.typography.sizes.sm,
    fontWeight: THEME.typography.weights.bold,
  },
  summaryBanner: {
    backgroundColor: THEME.colors.surface,
    marginHorizontal: THEME.spacing.md,
    marginTop: THEME.spacing.md,
    borderRadius: THEME.borderRadius.md,
    padding: THEME.spacing.md,
    borderWidth: 1,
    borderColor: THEME.colors.surfaceBorder,
  },
  summaryStats: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: THEME.spacing.md,
  },
  statDivider: {
    width: 1,
    height: 36,
    backgroundColor: THEME.colors.surfaceBorder,
    marginHorizontal: THEME.spacing.lg,
  },
  summaryLabel: {
    color: THEME.colors.textMuted,
    fontSize: 9,
    fontWeight: THEME.typography.weights.heavy,
    letterSpacing: 0.5,
  },
  summaryValue: {
    color: THEME.colors.accentNeon,
    fontSize: THEME.typography.sizes.xl,
    fontWeight: THEME.typography.weights.heavy,
    marginTop: 2,
  },
  bannerActions: {
    flexDirection: "row",
    gap: THEME.spacing.sm,
  },
  audioActionBtn: {
    flex: 2,
    backgroundColor: THEME.colors.accentNeon,
    paddingVertical: 8,
    borderRadius: THEME.borderRadius.sm,
    alignItems: "center",
  },
  audioActionText: {
    color: "#000000",
    fontSize: THEME.typography.sizes.xs,
    fontWeight: THEME.typography.weights.heavy,
  },
  clearActionBtn: {
    flex: 1,
    backgroundColor: THEME.colors.surfaceLight,
    borderWidth: 1,
    borderColor: THEME.colors.surfaceBorder,
    paddingVertical: 8,
    borderRadius: THEME.borderRadius.sm,
    alignItems: "center",
  },
  clearActionText: {
    color: THEME.colors.statusDanger,
    fontSize: THEME.typography.sizes.xs,
    fontWeight: THEME.typography.weights.bold,
  },
  filterBar: {
    flexDirection: "row",
    paddingHorizontal: THEME.spacing.md,
    paddingVertical: THEME.spacing.sm,
    gap: 6,
    flexWrap: "wrap",
  },
  filterChip: {
    paddingHorizontal: 10,
    paddingVertical: 5,
    backgroundColor: THEME.colors.surface,
    borderRadius: THEME.borderRadius.sm,
    borderWidth: 1,
    borderColor: THEME.colors.surfaceBorder,
  },
  filterChipActive: {
    borderColor: THEME.colors.accentNeon,
    backgroundColor: THEME.colors.accentNeonMuted,
  },
  filterChipText: {
    color: THEME.colors.textSecondary,
    fontSize: 11,
    fontWeight: THEME.typography.weights.bold,
  },
  filterChipTextActive: {
    color: THEME.colors.accentNeon,
  },
  listContent: {
    paddingHorizontal: THEME.spacing.md,
    paddingBottom: THEME.spacing.xxl,
    gap: THEME.spacing.sm,
  },
  card: {
    backgroundColor: THEME.colors.surface,
    borderWidth: 1,
    borderColor: THEME.colors.surfaceBorder,
    borderRadius: THEME.borderRadius.md,
    padding: THEME.spacing.md,
  },
  cardHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: THEME.spacing.sm,
  },
  leftInfo: {
    flexDirection: "row",
    alignItems: "center",
    gap: THEME.spacing.sm,
  },
  colorSquare: {
    width: 38,
    height: 38,
    borderRadius: THEME.borderRadius.sm,
    justifyContent: "center",
    alignItems: "center",
  },
  thumbnail: {
    width: 38,
    height: 38,
    borderRadius: THEME.borderRadius.sm,
    backgroundColor: THEME.colors.backgroundAlt,
  },
  nominalBadgeText: {
    color: "#FFFFFF",
    fontSize: 12,
    fontWeight: THEME.typography.weights.heavy,
  },
  nameGroup: {
    justifyContent: "center",
  },
  cardTitle: {
    color: THEME.colors.textPrimary,
    fontSize: THEME.typography.sizes.base,
    fontWeight: THEME.typography.weights.bold,
  },
  cardTimestamp: {
    color: THEME.colors.textMuted,
    fontSize: 10,
    marginTop: 2,
  },
  amountPill: {
    backgroundColor: THEME.colors.accentNeonMuted,
    borderColor: THEME.colors.accentNeon,
    borderWidth: 1,
    borderRadius: THEME.borderRadius.sm,
    paddingHorizontal: 8,
    paddingVertical: 4,
  },
  amountText: {
    color: THEME.colors.accentNeon,
    fontSize: THEME.typography.sizes.sm,
    fontWeight: THEME.typography.weights.heavy,
  },
  cardFooter: {
    flexDirection: "row",
    alignItems: "center",
    gap: THEME.spacing.sm,
    paddingTop: THEME.spacing.xs,
    borderTopWidth: 1,
    borderTopColor: "rgba(255, 255, 255, 0.05)",
  },
  metaBadge: {
    flexDirection: "row",
    gap: 3,
    alignItems: "center",
  },
  metaLabel: {
    color: THEME.colors.textMuted,
    fontSize: 10,
  },
  metaValue: {
    color: THEME.colors.textSecondary,
    fontSize: 10,
    fontWeight: THEME.typography.weights.bold,
  },
  highValueBadge: {
    backgroundColor: THEME.colors.statusWarning,
    paddingHorizontal: 5,
    paddingVertical: 1,
    borderRadius: 3,
    marginLeft: "auto",
  },
  highValueText: {
    color: "#000000",
    fontSize: 9,
    fontWeight: THEME.typography.weights.heavy,
  },
  emptyContainer: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    paddingHorizontal: 40,
  },
  emptyTitle: {
    color: THEME.colors.textPrimary,
    fontSize: THEME.typography.sizes.base,
    fontWeight: THEME.typography.weights.bold,
    textAlign: "center",
    marginBottom: 8,
  },
  emptySub: {
    color: THEME.colors.textMuted,
    fontSize: THEME.typography.sizes.xs,
    textAlign: "center",
    lineHeight: 18,
  },
});
