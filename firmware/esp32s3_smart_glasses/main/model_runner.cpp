/**
 * TFLite Micro Model Runner Implementation for ESP32-S3 Sense.
 * Allocates tensor arena strictly in internal SRAM to prevent PSRAM latency collapse.
 */

#include "model_runner.h"
#include "esp_log.h"
#include "esp_heap_caps.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include <cmath>
#include <algorithm>

static const char* TAG = "MODEL_RUNNER";

// Defined anchors matching PyTorch training setup
static const float ANCHORS_16[3][2] = {{118.0f, 58.0f}, {56.0f, 112.0f}, {84.0f, 82.0f}};
static const float ANCHORS_32[3][2] = {{62.0f, 32.0f},  {30.0f, 60.0f},   {40.0f, 42.0f}};

static float sigmoid_fast(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

static float compute_box_iou(const BoundingBox& b1, const BoundingBox& b2) {
    float x1_min = b1.xc - b1.w / 2.0f;
    float x1_max = b1.xc + b1.w / 2.0f;
    float y1_min = b1.yc - b1.h / 2.0f;
    float y1_max = b1.yc + b1.h / 2.0f;

    float x2_min = b2.xc - b2.w / 2.0f;
    float x2_max = b2.xc + b2.w / 2.0f;
    float y2_min = b2.yc - b2.h / 2.0f;
    float y2_max = b2.yc + b2.h / 2.0f;

    float inter_w = std::max(0.0f, std::min(x1_max, x2_max) - std::max(x1_min, x2_min));
    float inter_h = std::max(0.0f, std::min(y1_max, y2_max) - std::max(y1_min, y2_min));
    float inter = inter_w * inter_h;
    float union_area = (b1.w * b1.h) + (b2.w * b2.h) - inter;
    return (union_area > 0.0f) ? (inter / union_area) : 0.0f;
}

ModelRunner::ModelRunner() 
    : tensor_arena(nullptr), interpreter_handle(nullptr), model_handle(nullptr),
      input_tensor_ptr(nullptr), input_scale(1.0f / 255.0f), input_zero_point(-128) {}

ModelRunner::~ModelRunner() {
    if (tensor_arena != nullptr) {
        heap_caps_free(tensor_arena);
        tensor_arena = nullptr;
    }
}

bool ModelRunner::init(const unsigned char* model_data, size_t model_len) {
    // 1. Allocate tensor arena strictly in Internal SRAM (MALLOC_CAP_INTERNAL)
    tensor_arena = (uint8_t*) heap_caps_malloc(TENSOR_ARENA_SIZE_BYTES, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (!tensor_arena) {
        ESP_LOGE(TAG, "Failed to allocate %d bytes in Internal SRAM!", TENSOR_ARENA_SIZE_BYTES);
        return false;
    }
    ESP_LOGI(TAG, "Allocated %d KB tensor arena in single-cycle internal SRAM.", TENSOR_ARENA_SIZE_BYTES / 1024);

    // 2. Map FlatBuffer Model
    const tflite::Model* model = tflite::GetModel(model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Model schema version %d does not match runtime %d", model->version(), TFLITE_SCHEMA_VERSION);
        return false;
    }
    model_handle = (void*) model;

    // 3. Register Op Resolver (Only ops required by YOLO-FastestV2)
    static tflite::MicroMutableOpResolver<7> resolver;
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddAdd();
    resolver.AddMaxPool2D();
    resolver.AddLogistic();
    resolver.AddReshape();
    resolver.AddQuantize();

    // 4. Construct Micro Interpreter
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, TENSOR_ARENA_SIZE_BYTES
    );

    TfLiteStatus allocate_status = static_interpreter.AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors() failed in TFLM!");
        return false;
    }

    interpreter_handle = (void*) &static_interpreter;
    TfLiteTensor* input = static_interpreter.input(0);
    input_tensor_ptr = input->data.int8;
    input_scale = input->params.scale;
    input_zero_point = input->params.zero_point;

    ESP_LOGI(TAG, "Model runner initialized. Arena used: %d / %d bytes", 
             static_interpreter.arena_used_bytes(), TENSOR_ARENA_SIZE_BYTES);
    return true;
}

int8_t* ModelRunner::get_input_buffer() {
    return input_tensor_ptr;
}

float ModelRunner::get_input_scale() const {
    return input_scale;
}

int32_t ModelRunner::get_input_zero_point() const {
    return input_zero_point;
}

void ModelRunner::decode_stride16(const int8_t* data, float scale, int32_t zp, float conf_thresh, std::vector<DetectionResult>& candidates) {
    // Head Stride 16: Grid 10x10, 3 anchors, 12 attributes (tx, ty, tw, th, obj, 7 classes)
    const int grid_size = 10;
    const int out_channels_per_anchor = 12;

    for (int a = 0; a < NUM_ANCHORS; a++) {
        for (int y = 0; y < grid_size; y++) {
            for (int x = 0; x < grid_size; x++) {
                int base_idx = ((y * grid_size + x) * NUM_ANCHORS + a) * out_channels_per_anchor;

                // Dequantize objectness
                float raw_obj = float(data[base_idx + 4] - zp) * scale;
                float obj_conf = sigmoid_fast(raw_obj);
                if (obj_conf < conf_thresh) continue;

                // Dequantize classes
                int best_cls = 0;
                float best_cls_prob = 0.0f;
                for (int c = 0; c < NUM_CLASSES; c++) {
                    float raw_cls = float(data[base_idx + 5 + c] - zp) * scale;
                    float prob = sigmoid_fast(raw_cls);
                    if (prob > best_cls_prob) {
                        best_cls_prob = prob;
                        best_cls = c;
                    }
                }

                float final_conf = obj_conf * best_cls_prob;
                if (final_conf < conf_thresh) continue;

                // Box coordinates
                float raw_tx = float(data[base_idx + 0] - zp) * scale;
                float raw_ty = float(data[base_idx + 1] - zp) * scale;
                float raw_tw = float(data[base_idx + 2] - zp) * scale;
                float raw_th = float(data[base_idx + 3] - zp) * scale;

                BoundingBox box;
                box.xc = (float(x) + sigmoid_fast(raw_tx)) / 10.0f;
                box.yc = (float(y) + sigmoid_fast(raw_ty)) / 10.0f;
                box.w  = (ANCHORS_16[a][0] * std::exp(std::clamp(raw_tw, -4.0f, 4.0f))) / 160.0f;
                box.h  = (ANCHORS_16[a][1] * std::exp(std::clamp(raw_th, -4.0f, 4.0f))) / 160.0f;

                candidates.push_back({box, best_cls, final_conf, obj_conf});
            }
        }
    }
}

void ModelRunner::decode_stride32(const int8_t* data, float scale, int32_t zp, float conf_thresh, std::vector<DetectionResult>& candidates) {
    // Head Stride 32: Grid 5x5, 3 anchors, 12 attributes
    const int grid_size = 5;
    const int out_channels_per_anchor = 12;

    for (int a = 0; a < NUM_ANCHORS; a++) {
        for (int y = 0; y < grid_size; y++) {
            for (int x = 0; x < grid_size; x++) {
                int base_idx = ((y * grid_size + x) * NUM_ANCHORS + a) * out_channels_per_anchor;

                float raw_obj = float(data[base_idx + 4] - zp) * scale;
                float obj_conf = sigmoid_fast(raw_obj);
                if (obj_conf < conf_thresh) continue;

                int best_cls = 0;
                float best_cls_prob = 0.0f;
                for (int c = 0; c < NUM_CLASSES; c++) {
                    float raw_cls = float(data[base_idx + 5 + c] - zp) * scale;
                    float prob = sigmoid_fast(raw_cls);
                    if (prob > best_cls_prob) {
                        best_cls_prob = prob;
                        best_cls = c;
                    }
                }

                float final_conf = obj_conf * best_cls_prob;
                if (final_conf < conf_thresh) continue;

                float raw_tx = float(data[base_idx + 0] - zp) * scale;
                float raw_ty = float(data[base_idx + 1] - zp) * scale;
                float raw_tw = float(data[base_idx + 2] - zp) * scale;
                float raw_th = float(data[base_idx + 3] - zp) * scale;

                BoundingBox box;
                box.xc = (float(x) + sigmoid_fast(raw_tx)) / 5.0f;
                box.yc = (float(y) + sigmoid_fast(raw_ty)) / 5.0f;
                box.w  = (ANCHORS_32[a][0] * std::exp(std::clamp(raw_tw, -4.0f, 4.0f))) / 160.0f;
                box.h  = (ANCHORS_32[a][1] * std::exp(std::clamp(raw_th, -4.0f, 4.0f))) / 160.0f;

                candidates.push_back({box, best_cls, final_conf, obj_conf});
            }
        }
    }
}

void ModelRunner::apply_nms(std::vector<DetectionResult>& candidates, float nms_thresh, std::vector<DetectionResult>& results) {
    if (candidates.empty()) return;

    std::sort(candidates.begin(), candidates.end(), [](const DetectionResult& a, const DetectionResult& b) {
        return a.confidence > b.confidence;
    });

    std::vector<bool> suppressed(candidates.size(), false);
    for (size_t i = 0; i < candidates.size(); i++) {
        if (suppressed[i]) continue;
        results.push_back(candidates[i]);
        for (size_t j = i + 1; j < candidates.size(); j++) {
            if (!suppressed[j]) {
                if (compute_box_iou(candidates[i].box, candidates[j].box) >= nms_thresh) {
                    suppressed[j] = true;
                }
            }
        }
    }
}

bool ModelRunner::run_inference(std::vector<DetectionResult>& out_detections, float conf_threshold, float nms_threshold) {
    out_detections.clear();
    tflite::MicroInterpreter* interpreter = (tflite::MicroInterpreter*) interpreter_handle;
    if (!interpreter) return false;

    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        ESP_LOGE(TAG, "Interpreter invocation failed!");
        return false;
    }

    TfLiteTensor* out_s16 = interpreter->output(0);
    TfLiteTensor* out_s32 = interpreter->output(1);

    std::vector<DetectionResult> candidates;
    candidates.reserve(16);

    decode_stride16(out_s16->data.int8, out_s16->params.scale, out_s16->params.zero_point, conf_threshold, candidates);
    decode_stride32(out_s32->data.int8, out_s32->params.scale, out_s32->params.zero_point, conf_threshold, candidates);

    apply_nms(candidates, nms_threshold, out_detections);
    return true;
}
