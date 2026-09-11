import React from "react";
import { ScrollView, StyleSheet, Text, TouchableOpacity, View } from "react-native";
import { DENOMINATION_LIST } from "../constants/denominations";
import { THEME } from "../constants/theme";

interface SampleBanknotePickerProps {
  onSelectDenomination: (id: number) => void;
  onSelectMultiple?: (ids: number[]) => void;
  selectedId: number | null;
}

export const SampleBanknotePicker: React.FC<SampleBanknotePickerProps> = ({
  onSelectDenomination,
  onSelectMultiple,
  selectedId,
}) => {
  return (
    <View style={styles.container}>
      <Text style={styles.title}>Test Əsginasları (Tək və Cəm Simulyasiyası):</Text>
      <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.scroll}>
        {/* Multi-banknote sum test chips */}
        {onSelectMultiple && (
          <>
            <TouchableOpacity
              style={[styles.chip, styles.multiChip]}
              onPress={() => onSelectMultiple([1, 2])}
              activeOpacity={0.7}
            >
              <Text style={styles.multiChipText}>5₼ + 10₼ (15₼ Cəm)</Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={[styles.chip, styles.multiChip]}
              onPress={() => onSelectMultiple([3, 4])}
              activeOpacity={0.7}
            >
              <Text style={styles.multiChipText}>20₼ + 50₼ (70₼ Cəm)</Text>
            </TouchableOpacity>
          </>
        )}

        {DENOMINATION_LIST.map((denom) => {
          const isSelected = selectedId === denom.id;
          return (
            <TouchableOpacity
              key={denom.id}
              style={[
                styles.chip,
                isSelected && styles.chipSelected,
                { borderColor: isSelected ? THEME.colors.accentNeon : THEME.colors.surfaceBorder },
              ]}
              onPress={() => onSelectDenomination(denom.id)}
              activeOpacity={0.7}
            >
              <Text
                style={[
                  styles.chipText,
                  isSelected && styles.chipTextSelected,
                ]}
              >
                {denom.nominalValue} ₼
              </Text>
            </TouchableOpacity>
          );
        })}
      </ScrollView>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    paddingVertical: THEME.spacing.sm,
    backgroundColor: THEME.colors.surface,
    borderTopWidth: 1,
    borderTopColor: THEME.colors.surfaceBorder,
  },
  title: {
    color: THEME.colors.textMuted,
    fontSize: 10,
    fontWeight: THEME.typography.weights.semibold,
    paddingHorizontal: THEME.spacing.lg,
    marginBottom: 4,
    textTransform: "uppercase",
  },
  scroll: {
    paddingHorizontal: THEME.spacing.lg,
    gap: THEME.spacing.sm,
  },
  chip: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    backgroundColor: THEME.colors.background,
    borderWidth: 1,
    borderRadius: THEME.borderRadius.sm,
    alignItems: "center",
  },
  chipSelected: {
    backgroundColor: THEME.colors.accentNeonMuted,
    borderColor: THEME.colors.accentNeon,
  },
  chipText: {
    color: THEME.colors.textSecondary,
    fontSize: THEME.typography.sizes.xs,
    fontWeight: THEME.typography.weights.bold,
  },
  chipTextSelected: {
    color: THEME.colors.accentNeon,
  },
  multiChip: {
    backgroundColor: "rgba(212, 249, 56, 0.12)",
    borderColor: THEME.colors.accentNeon,
    borderWidth: 1.5,
  },
  multiChipText: {
    color: THEME.colors.accentNeon,
    fontSize: THEME.typography.sizes.xs,
    fontWeight: THEME.typography.weights.heavy,
  },
});
