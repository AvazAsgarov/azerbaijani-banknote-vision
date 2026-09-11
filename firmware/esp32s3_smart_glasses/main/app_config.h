/**
 * System Configuration for Seeed Studio XIAO ESP32S3 Sense Smart Glasses.
 * Defines camera hardware pins, memory arena budgets, and detection thresholds.
 */

#ifndef APP_CONFIG_H
#define APP_CONFIG_H

#include <stdint.h>

// Camera Pin Mapping: Seeed Studio XIAO ESP32S3 Sense (OV2640)
#define CAM_PIN_PWDN    -1
#define CAM_PIN_RESET   -1
#define CAM_PIN_XCLK    10
#define CAM_PIN_SIOD    40
#define CAM_PIN_SIOC    39

#define CAM_PIN_D7      48
#define CAM_PIN_D6      11
#define CAM_PIN_D5      12
#define CAM_PIN_D4      14
#define CAM_PIN_D3      16
#define CAM_PIN_D2      18
#define CAM_PIN_D1      17
#define CAM_PIN_D0      15
#define CAM_PIN_VSYNC   38
#define CAM_PIN_HREF    47
#define CAM_PIN_PCLK    13

// Neural Network Architecture Constraints
#define MODEL_INPUT_WIDTH       160
#define MODEL_INPUT_HEIGHT      160
#define MODEL_INPUT_CHANNELS    3
#define NUM_CLASSES             7
#define NUM_ANCHORS             3

// Tensor Arena Memory Limit: 285 KB allocated strictly in internal SRAM
#define TENSOR_ARENA_SIZE_BYTES (285 * 1024)

// Temporal Confirmation Window
#define TEMPORAL_WINDOW_SIZE    5
#define TEMPORAL_MATCH_REQUIRED 3

// Denomination String Table
static const char* const DENOMINATION_NAMES[NUM_CLASSES] = {
    "001_azn", "005_azn", "010_azn", "020_azn", "050_azn", "100_azn", "200_azn"
};

// Denomination Dynamic Acceptance Thresholds
static const float DYNAMIC_THRESHOLDS_VAL[NUM_CLASSES] = {
    0.50f,  // 001_azn (Grey / Folkloric)
    0.55f,  // 005_azn (Orange / Literature)
    0.60f,  // 010_azn (Cyan / History)
    0.65f,  // 020_azn (Green / Karabakh)
    0.70f,  // 050_azn (Yellow / Education)
    0.75f,  // 100_azn (Purple / Architecture)
    0.80f   // 200_azn (Blue / Modernity)
};

#endif // APP_CONFIG_H
