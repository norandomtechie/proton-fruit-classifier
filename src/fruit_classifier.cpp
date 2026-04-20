/*
 * Fruit classifier using TensorFlow Lite Micro.
 * Runs a small quantized CNN on 48x48 grayscale input.
 * Designed for RP2350B with OV7670 camera.
 */

#include "fruit_classifier.h"
#include "fruit_model.h"

// Must include common.h first, then provide a stub for TfLiteTensorDataFree
// which is hidden by TF_LITE_STATIC_MEMORY but referenced by kernel_util.h
#include "tensorflow/lite/core/c/common.h"
#ifdef TF_LITE_STATIC_MEMORY
#ifdef __cplusplus
extern "C" {
#endif
static inline void TfLiteTensorDataFree(TfLiteTensor* t) { (void)t; }
#ifdef __cplusplus
}
#endif
#endif

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "pico/stdlib.h"
#include <cstring>

// Tensor arena — model needs ~70-100 KB for a 4-layer CNN with 48x48 input.
// Align to 16 bytes for CMSIS-NN.
static constexpr int kTensorArenaSize = 192 * 1024;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

// Scratch buffer for downsampled RGB image
static int8_t resized_image[FRUIT_INPUT_SIZE * FRUIT_INPUT_SIZE * 3];

// TFLM objects
static tflite::MicroInterpreter *interpreter = nullptr;
static TfLiteTensor *input_tensor = nullptr;
static TfLiteTensor *output_tensor = nullptr;

// Timing
static uint32_t last_inference_us = 0;

// Op resolver — only include ops used by our model
static tflite::MicroMutableOpResolver<8> resolver;

static const char *class_names[] = {"apple", "banana", "strawberry", "background"};

// =====================================================================
// Image preprocessing: YUY2 → NxN RGB INT8 (3 channels)
// Quantization params are read from input tensor at runtime.
// =====================================================================
static float input_scale = 1.0f / 255.0f;
static int32_t input_zero_point = -128;

static void preprocess_yuy2_to_rgb(const uint8_t *yuy2, uint16_t w, uint16_t h) {
    // YUY2 format: Y0 U0 Y1 V0 Y2 U1 Y3 V1 ...
    // Each pair of pixels shares U and V values.

    const float x_ratio = (float)w / FRUIT_INPUT_SIZE;
    const float y_ratio = (float)h / FRUIT_INPUT_SIZE;
    const int src_stride = w * 2;  // 2 bytes per pixel in YUY2
    const float inv_scale = 1.0f / (255.0f * input_scale);

    for (int dy = 0; dy < FRUIT_INPUT_SIZE; dy++) {
        int sy = (int)(dy * y_ratio);
        if (sy >= h) sy = h - 1;

        for (int dx = 0; dx < FRUIT_INPUT_SIZE; dx++) {
            int sx = (int)(dx * x_ratio);
            if (sx >= w) sx = w - 1;

            // Extract Y, U, V from YUY2
            uint8_t y_val = yuy2[sy * src_stride + sx * 2];
            uint8_t u_val = yuy2[sy * src_stride + (sx & ~1) * 2 + 1];
            uint8_t v_val = yuy2[sy * src_stride + (sx & ~1) * 2 + 3];

            // YUV → RGB (BT.601 integer math)
            int c = y_val - 16;
            int d = u_val - 128;
            int e = v_val - 128;
            int r = (298 * c + 409 * e + 128) >> 8;
            int g = (298 * c - 100 * d - 208 * e + 128) >> 8;
            int b = (298 * c + 516 * d + 128) >> 8;
            if (r < 0) r = 0; if (r > 255) r = 255;
            if (g < 0) g = 0; if (g > 255) g = 255;
            if (b < 0) b = 0; if (b > 255) b = 255;

            // Quantize each channel to INT8
            int idx = (dy * FRUIT_INPUT_SIZE + dx) * 3;
            int32_t qr = (int32_t)(r * inv_scale) + input_zero_point;
            int32_t qg = (int32_t)(g * inv_scale) + input_zero_point;
            int32_t qb = (int32_t)(b * inv_scale) + input_zero_point;
            if (qr < -128) qr = -128; if (qr > 127) qr = 127;
            if (qg < -128) qg = -128; if (qg > 127) qg = 127;
            if (qb < -128) qb = -128; if (qb > 127) qb = 127;
            resized_image[idx]     = (int8_t)qr;
            resized_image[idx + 1] = (int8_t)qg;
            resized_image[idx + 2] = (int8_t)qb;
        }
    }
}

// =====================================================================
// Public API
// =====================================================================

extern "C" bool fruit_classifier_init(void) {
    // Skip tflite::InitializeTarget() — it calls stdio_init_all() which
    // must not run on Core 1 (conflicts with Core 0's CDC USB stdio).

    // Load model
    const tflite::Model *model = tflite::GetModel(fruit_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        return false;
    }

    // Register only the ops our model uses
    resolver.AddConv2D();
    resolver.AddMaxPool2D();
    resolver.AddFullyConnected();
    resolver.AddReshape();
    resolver.AddQuantize();
    resolver.AddDequantize();
    resolver.AddMean();       // for GlobalAveragePooling

    // Build interpreter
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    TfLiteStatus status = interpreter->AllocateTensors();
    if (status != kTfLiteOk) {
        return false;
    }

    input_tensor = interpreter->input(0);
    output_tensor = interpreter->output(0);

    // Read quantization params from the model for correct preprocessing
    input_scale = input_tensor->params.scale;
    input_zero_point = input_tensor->params.zero_point;

    return true;
}

extern "C" bool fruit_classifier_run(const uint8_t *yuy2_frame, uint16_t width, uint16_t height,
                                     fruit_result_t *result) {
    if (!interpreter || !result) return false;

    // Preprocess: downsample to NxN RGB INT8
    preprocess_yuy2_to_rgb(yuy2_frame, width, height);

    // Copy into input tensor
    memcpy(input_tensor->data.int8, resized_image,
           FRUIT_INPUT_SIZE * FRUIT_INPUT_SIZE * 3);

    // Run inference with timing
    uint32_t t0 = time_us_32();
    TfLiteStatus status = interpreter->Invoke();
    uint32_t t1 = time_us_32();
    last_inference_us = t1 - t0;

    if (status != kTfLiteOk) {
        return false;
    }

    // Find class with highest score
    int8_t *output = output_tensor->data.int8;
    int best_idx = 0;
    int8_t best_score = output[0];
    for (int i = 1; i < FRUIT_NUM_CLASSES; i++) {
        if (output[i] > best_score) {
            best_score = output[i];
            best_idx = i;
        }
    }

    result->class_id = best_idx;
    result->class_name = class_names[best_idx];
    result->confidence = best_score;
    for (int i = 0; i < FRUIT_NUM_CLASSES; i++) {
        result->scores[i] = output[i];
    }

    return true;
}

extern "C" uint32_t fruit_classifier_get_inference_time_us(void) {
    return last_inference_us;
}
