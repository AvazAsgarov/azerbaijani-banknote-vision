/**
 * Assistive Safety Guard Interface for Smart Glasses Firmware.
 * Evaluates geometric plausibility, dynamic confidence, and temporal stability.
 */

#ifndef SAFETY_GUARD_H
#define SAFETY_GUARD_H

#include <stdint.h>
#include <stdbool.h>
#include <vector>
#include "model_runner.h"
#include "app_config.h"

enum SafetyStatus {
    STATUS_NO_DETECTION,
    STATUS_GUIDANCE,
    STATUS_CONFIRMED
};

struct SafetyDecision {
    SafetyStatus status;
    int confirmed_class_id;
    float confidence;
    BoundingBox box;
    const char* message;
};

class FirmwareSafetyGuard {
public:
    FirmwareSafetyGuard();

    // Process detections from current inference frame and return safe operational decision
    SafetyDecision evaluate_detections(const std::vector<DetectionResult>& detections);

    // Reset temporal confirmation history
    void reset();

private:
    int history_class[TEMPORAL_WINDOW_SIZE];
    float history_conf[TEMPORAL_WINDOW_SIZE];
    BoundingBox history_box[TEMPORAL_WINDOW_SIZE];
    int history_index;
    int history_count;

    bool check_geometry(const BoundingBox& box);
    bool check_confidence(int class_id, float conf);
};

#endif // SAFETY_GUARD_H
