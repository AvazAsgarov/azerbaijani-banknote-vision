/**
 * Camera Pipeline Interface for OV2640 Sensor on ESP32-S3 Sense.
 * Manages frame acquisition into PSRAM and zero-copy resizing into SRAM input tensor.
 */

#ifndef CAMERA_PIPELINE_H
#define CAMERA_PIPELINE_H

#include <stdint.h>
#include <stdbool.h>
#include "esp_camera.h"

class CameraPipeline {
public:
    CameraPipeline();
    ~CameraPipeline();

    // Initialize OV2640 camera with double buffering in PSRAM
    bool init();

    // Capture frame and resize/normalize directly into model INT8 tensor in internal SRAM
    bool capture_and_preprocess(int8_t* model_input_tensor, float input_scale, int32_t input_zero_point);

    // Release current frame buffer back to camera DMA driver
    void release_frame();

private:
    camera_fb_t* current_fb;
};

#endif // CAMERA_PIPELINE_H
