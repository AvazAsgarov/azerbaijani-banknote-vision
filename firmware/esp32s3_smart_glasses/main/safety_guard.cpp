/**
 * Assistive Safety Guard Implementation for ESP32-S3 Firmware.
 * Implements 4-tier safety guardrail eliminating high-value false positives.
 */

#include "safety_guard.h"
#include <algorithm>

FirmwareSafetyGuard::FirmwareSafetyGuard() : history_index(0), history_count(0) {
    reset();
}

void FirmwareSafetyGuard::reset() {
    for (int i = 0; i < TEMPORAL_WINDOW_SIZE; i++) {
        history_class[i] = -1;
        history_conf[i] = 0.0f;
    }
    history_index = 0;
    history_count = 0;
}

bool FirmwareSafetyGuard::check_geometry(const BoundingBox& box) {
    if (box.w <= 0.0f || box.h <= 0.0f) return false;
    float area = box.w * box.h;
    if (area < 0.08f) return false; // Reject micro-detections below 8% of frame

    float ratio = box.w / box.h;
    bool landscape = (ratio >= 1.35f && ratio <= 2.55f);
    bool portrait  = (ratio >= 0.38f && ratio <= 0.72f);
    return (landscape || portrait);
}

bool FirmwareSafetyGuard::check_confidence(int class_id, float conf) {
    if (class_id < 0 || class_id >= NUM_CLASSES) return false;
    return conf >= DYNAMIC_THRESHOLDS_VAL[class_id];
}

SafetyDecision FirmwareSafetyGuard::evaluate_detections(const std::vector<DetectionResult>& detections) {
    SafetyDecision decision;
    decision.status = STATUS_NO_DETECTION;
    decision.confirmed_class_id = -1;
    decision.confidence = 0.0f;
    decision.message = "Scanning for banknote...";

    // Tier 1 & 2: Filter candidates by geometry and dynamic confidence
    int best_valid_class = -1;
    float best_valid_conf = 0.0f;
    BoundingBox best_valid_box = {0.0f, 0.0f, 0.0f, 0.0f};

    for (const auto& det : detections) {
        if (!check_geometry(det.box)) continue;
        if (!check_confidence(det.class_id, det.confidence)) continue;

        if (det.confidence > best_valid_conf) {
            best_valid_conf = det.confidence;
            best_valid_class = det.class_id;
            best_valid_box = det.box;
        }
    }

    // Update temporal sliding history
    history_class[history_index] = best_valid_class;
    history_conf[history_index] = best_valid_conf;
    history_box[history_index] = best_valid_box;
    history_index = (history_index + 1) % TEMPORAL_WINDOW_SIZE;
    if (history_count < TEMPORAL_WINDOW_SIZE) history_count++;

    if (best_valid_class == -1) {
        // Check if recent frames had detections
        bool had_recent = false;
        for (int i = 0; i < history_count; i++) {
            if (history_class[i] != -1) {
                had_recent = true;
                break;
            }
        }
        if (had_recent) {
            decision.status = STATUS_GUIDANCE;
            decision.message = "Banknote detected. Please hold steady under lighting.";
            return decision;
        }
        return decision;
    }

    // Tier 3: Temporal Majority Confirmation
    int class_votes[NUM_CLASSES] = {0};
    for (int i = 0; i < history_count; i++) {
        int c = history_class[i];
        if (c >= 0 && c < NUM_CLASSES) {
            class_votes[c]++;
        }
    }

    for (int c = 0; c < NUM_CLASSES; c++) {
        if (class_votes[c] >= TEMPORAL_MATCH_REQUIRED) {
            decision.status = STATUS_CONFIRMED;
            decision.confirmed_class_id = c;
            decision.confidence = best_valid_conf;
            decision.box = best_valid_box;
            decision.message = DENOMINATION_NAMES[c];
            return decision;
        }
    }

    // Detected valid candidate, but awaiting temporal confirmation
    decision.status = STATUS_GUIDANCE;
    decision.message = "Verifying denomination. Maintain position.";
    return decision;
}
