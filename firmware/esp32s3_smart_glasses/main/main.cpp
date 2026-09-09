/**
 * Main Application Entry Point for ESP32-S3 Sense Smart Assistive Glasses.
 * Coordinates real-time camera capture, TFLite Micro inference, and safety guard verification.
 */

#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "app_config.h"
#include "camera_pipeline.h"
#include "model_runner.h"
#include "safety_guard.h"

// Forward declaration of model byte array exported from quantize_export pipeline
extern const unsigned char g_yolo_fastestv2_model_data[];
extern const unsigned int g_yolo_fastestv2_model_data_len;

static const char* TAG = "SMART_GLASSES";

// Fallback dummy model array for compilation verification before flash burn
__attribute__((weak)) const unsigned char g_yolo_fastestv2_model_data[16] = {0};
__attribute__((weak)) const unsigned int g_yolo_fastestv2_model_data_len = 16;

void vision_pipeline_task(void* pvParameters) {
    ESP_LOGI(TAG, "Starting Assistive Vision Pipeline on ESP32-S3 Sense...");

    CameraPipeline camera;
    if (!camera.init()) {
        ESP_LOGE(TAG, "Fatal: Camera pipeline initialization failure!");
        vTaskDelete(NULL);
        return;
    }

    ModelRunner runner;
    if (g_yolo_fastestv2_model_data_len > 16) {
        if (!runner.init(g_yolo_fastestv2_model_data, g_yolo_fastestv2_model_data_len)) {
            ESP_LOGE(TAG, "Fatal: Neural network runner initialization failure!");
            vTaskDelete(NULL);
            return;
        }
    } else {
        ESP_LOGW(TAG, "Model binary array awaiting burn. Running camera telemetry stream.");
    }

    FirmwareSafetyGuard safety_guard;
    std::vector<DetectionResult> detections;
    detections.reserve(8);

    int64_t last_log_time = esp_timer_get_time();
    int frame_counter = 0;

    while (true) {
        int64_t t_start = esp_timer_get_time();

        // 1. Capture and convert frame directly into SRAM tensor buffer
        int8_t* input_buf = runner.get_input_buffer();
        if (input_buf != nullptr) {
            camera.capture_and_preprocess(input_buf, runner.get_input_scale(), runner.get_input_zero_point());
        }

        // 2. Execute TFLM INT8 Neural Inference
        if (g_yolo_fastestv2_model_data_len > 16) {
            runner.run_inference(detections, 0.25f, 0.45f);

            // 3. Multi-Tier Assistive Safety Guardrail Verification
            SafetyDecision decision = safety_guard.evaluate_detections(detections);

            if (decision.status == STATUS_CONFIRMED) {
                ESP_LOGI(TAG, ">>> BLE AUDIO TRANSMIT: CONFIRMED %s (Conf: %.2f) <<<",
                         decision.message, decision.confidence);
            } else if (decision.status == STATUS_GUIDANCE) {
                ESP_LOGI(TAG, ">>> BLE GUIDANCE: %s <<<", decision.message);
            }
        }

        int64_t t_end = esp_timer_get_time();
        int64_t elapsed_us = t_end - t_start;
        frame_counter++;

        if (t_end - last_log_time >= 2000000) { // Log every 2 seconds
            float fps = float(frame_counter) * 1000000.0f / float(t_end - last_log_time);
            ESP_LOGI(TAG, "Telemetry: %.1f FPS (Inference Latency: %lld ms)", fps, elapsed_us / 1000);
            frame_counter = 0;
            last_log_time = t_end;
        }

        // Paced at ~20-25 FPS to allow thermal dissipation and prevent LDO/CPU overheating
        vTaskDelay(pdMS_TO_TICKS(40));
    }
}

extern "C" void app_main(void) {
    ESP_LOGI(TAG, "========================================================");
    ESP_LOGI(TAG, " Azerbaijani Banknote Assistive Smart Glasses System   ");
    ESP_LOGI(TAG, " Target: Seeed Studio XIAO ESP32S3 Sense (Xtensa LX7)   ");
    ESP_LOGI(TAG, " Model: YOLO-FastestV2 INT8 (160x160 RGB)               ");
    ESP_LOGI(TAG, "========================================================");

    // Launch vision processing task pinned to Core 1 (leaving Core 0 for BLE/Wi-Fi)
    xTaskCreatePinnedToCore(
        vision_pipeline_task,
        "vision_task",
        8192,
        NULL,
        5,
        NULL,
        1
    );
}
