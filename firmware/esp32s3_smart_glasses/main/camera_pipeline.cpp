/**
 * Camera Pipeline Implementation for OV2640 on ESP32-S3.
 * Configures double frame buffers in PSRAM and executes integer streaming conversion into SRAM.
 */

#include "camera_pipeline.h"
#include "app_config.h"
#include "esp_log.h"
#include <algorithm>

static const char* TAG = "CAM_PIPELINE";

CameraPipeline::CameraPipeline() : current_fb(nullptr) {}

CameraPipeline::~CameraPipeline() {
    release_frame();
}

bool CameraPipeline::init() {
    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer = LEDC_TIMER_0;
    config.pin_d0 = CAM_PIN_D0;
    config.pin_d1 = CAM_PIN_D1;
    config.pin_d2 = CAM_PIN_D2;
    config.pin_d3 = CAM_PIN_D3;
    config.pin_d4 = CAM_PIN_D4;
    config.pin_d5 = CAM_PIN_D5;
    config.pin_d6 = CAM_PIN_D6;
    config.pin_d7 = CAM_PIN_D7;
    config.pin_xclk = CAM_PIN_XCLK;
    config.pin_pclk = CAM_PIN_PCLK;
    config.pin_vsync = CAM_PIN_VSYNC;
    config.pin_href = CAM_PIN_HREF;
    config.pin_sccb_sda = CAM_PIN_SIOD;
    config.pin_sccb_scl = CAM_PIN_SIOC;
    config.pin_pwdn = CAM_PIN_PWDN;
    config.pin_reset = CAM_PIN_RESET;
    config.xclk_freq_hz = 20000000;
    config.pixel_format = PIXFORMAT_RGB565;
    config.frame_size = FRAMESIZE_QVGA;     // 320x240 native capture
    config.jpeg_quality = 12;
    config.fb_count = 2;                     // Double buffering in PSRAM
    config.fb_location = CAMERA_FB_IN_PSRAM; // Prevent SRAM starvation
    config.grab_mode = CAMERA_GRAB_LATEST;

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Camera initialization failed with error: 0x%x", err);
        return false;
    }

    ESP_LOGI(TAG, "OV2640 camera successfully initialized in PSRAM mode.");
    return true;
}

void CameraPipeline::release_frame() {
    if (current_fb != nullptr) {
        esp_camera_fb_return(current_fb);
        current_fb = nullptr;
    }
}

bool CameraPipeline::capture_and_preprocess(int8_t* model_input_tensor, float input_scale, int32_t input_zero_point) {
    release_frame();
    current_fb = esp_camera_fb_get();
    if (!current_fb) {
        ESP_LOGE(TAG, "Camera frame grab failed.");
        return false;
    }

    // Source dimensions: 320x240 RGB565 (2 bytes per pixel)
    // Destination dimensions: 160x160x3 INT8 in internal SRAM
    const uint8_t* src = current_fb->buf;
    const int src_w = current_fb->width;
    const int src_h = current_fb->height;

    // Center crop: take 240x240 square out of 320x240
    const int crop_x_offset = (src_w - src_h) / 2; // 40 pixels offset

    for (int dst_y = 0; dst_y < MODEL_INPUT_HEIGHT; dst_y++) {
        int src_y = (dst_y * src_h) / MODEL_INPUT_HEIGHT;
        int row_offset = src_y * src_w * 2;

        for (int dst_x = 0; dst_x < MODEL_INPUT_WIDTH; dst_x++) {
            int src_x = crop_x_offset + (dst_x * src_h) / MODEL_INPUT_WIDTH;
            int pixel_index = row_offset + src_x * 2;

            // Extract RGB565 components (Big-endian format from camera DVP)
            uint8_t b1 = src[pixel_index];
            uint8_t b2 = src[pixel_index + 1];
            uint16_t rgb565 = (uint16_t(b1) << 8) | uint16_t(b2);

            uint8_t r = ((rgb565 >> 11) & 0x1F) << 3;
            uint8_t g = ((rgb565 >> 5) & 0x3F) << 2;
            uint8_t b = (rgb565 & 0x1F) << 3;

            // Direct integer quantization into target SRAM tensor
            int dst_idx = (dst_y * MODEL_INPUT_WIDTH + dst_x) * 3;

            // Formula: quant = clamp(round(val / scale) + zero_point, -128, 127)
            float norm_r = float(r) / 255.0f;
            float norm_g = float(g) / 255.0f;
            float norm_b = float(b) / 255.0f;

            int32_t q_r = int32_t(std::round(norm_r / input_scale)) + input_zero_point;
            int32_t q_g = int32_t(std::round(norm_g / input_scale)) + input_zero_point;
            int32_t q_b = int32_t(std::round(norm_b / input_scale)) + input_zero_point;

            model_input_tensor[dst_idx]     = int8_t(std::clamp(q_r, -128, 127));
            model_input_tensor[dst_idx + 1] = int8_t(std::clamp(q_g, -128, 127));
            model_input_tensor[dst_idx + 2] = int8_t(std::clamp(q_b, -128, 127));
        }
    }

    return true;
}
