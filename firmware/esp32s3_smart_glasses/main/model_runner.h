/**
 * TFLite Micro Model Runner Interface for YOLO-FastestV2 on ESP32-S3.
 * Manages tensor arena in internal SRAM, executes inference, and decodes dual heads.
 */

#ifndef MODEL_RUNNER_H
#define MODEL_RUNNER_H

#include <stdint.h>
#include <stdbool.h>
#include <vector>
#include "app_config.h"

struct BoundingBox {
    float xc;
    float yc;
    float w;
    float h;
};

struct DetectionResult {
    BoundingBox box;
    int class_id;
    float confidence;
    float objectness;
};

class ModelRunner {
public:
    ModelRunner();
    ~ModelRunner();

    // Allocate internal SRAM tensor arena and initialize TFLite interpreter
    bool init(const unsigned char* model_data, size_t model_len);

    // Retrieve pointer to model input buffer
    int8_t* get_input_buffer();
    float get_input_scale() const;
    int32_t get_input_zero_point() const;

    // Run inference and return decoded detections after NMS
    bool run_inference(std::vector<DetectionResult>& out_detections, float conf_threshold = 0.25f, float nms_threshold = 0.45f);

private:
    uint8_t* tensor_arena;
    void* interpreter_handle;
    void* model_handle;
    int8_t* input_tensor_ptr;
    float input_scale;
    int32_t input_zero_point;

    // Decode raw INT8 outputs from dual heads
    void decode_stride16(const int8_t* data, float scale, int32_t zp, float conf_thresh, std::vector<DetectionResult>& candidates);
    void decode_stride32(const int8_t* data, float scale, int32_t zp, float conf_thresh, std::vector<DetectionResult>& candidates);
    void apply_nms(std::vector<DetectionResult>& candidates, float nms_thresh, std::vector<DetectionResult>& results);
};

#endif // MODEL_RUNNER_H
