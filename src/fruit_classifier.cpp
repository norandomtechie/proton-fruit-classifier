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
static constexpr int kTensorArenaSize = 128 * 1024;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

// Scratch buffer for downsampled image
static int8_t resized_image[FRUIT_INPUT_SIZE * FRUIT_INPUT_SIZE];

// TFLM objects
static tflite::MicroInterpreter *interpreter = nullptr;
static TfLiteTensor *input_tensor = nullptr;
static TfLiteTensor *output_tensor = nullptr;

// Timing
static uint32_t last_inference_us = 0;

// Op resolver — only include ops used by our model
static tflite::MicroMutableOpResolver<8> resolver;

static const char *class_names[] = {"apple", "banana", "orange", "background"};

// =====================================================================
// Image preprocessing: YUY2 → 48x48 grayscale INT8
// =====================================================================
static void preprocess_yuy2_to_gray48(const uint8_t *yuy2, uint16_t w, uint16_t h) {
    // YUY2 format: Y0 U0 Y1 V0 Y2 U1 Y3 V1 ...
    // Y is luminance (grayscale), at every even byte index.
    // We bilinearly downsample by stepping through source coordinates.

    const float x_ratio = (float)w / FRUIT_INPUT_SIZE;
    const float y_ratio = (float)h / FRUIT_INPUT_SIZE;
    const int src_stride = w * 2;  // 2 bytes per pixel in YUY2

    for (int dy = 0; dy < FRUIT_INPUT_SIZE; dy++) {
        int sy = (int)(dy * y_ratio);
        if (sy >= h) sy = h - 1;

        for (int dx = 0; dx < FRUIT_INPUT_SIZE; dx++) {
            int sx = (int)(dx * x_ratio);
            if (sx >= w) sx = w - 1;

            // In YUY2, Y byte is at offset sx*2 in the row
            uint8_t y_val = yuy2[sy * src_stride + sx * 2];

            // Convert uint8 [0,255] to int8 [-128,127] for INT8 quantization
            // (zero_point=-128, scale=1/255 is the standard quantization)
            resized_image[dy * FRUIT_INPUT_SIZE + dx] = (int8_t)(y_val - 128);
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
    resolver.AddSoftmax();

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

    return true;
}

extern "C" bool fruit_classifier_run(const uint8_t *yuy2_frame, uint16_t width, uint16_t height,
                                     fruit_result_t *result) {
    if (!interpreter || !result) return false;

    // Preprocess: downsample to 48x48 grayscale INT8
    preprocess_yuy2_to_gray48(yuy2_frame, width, height);

    // Copy into input tensor
    memcpy(input_tensor->data.int8, resized_image,
           FRUIT_INPUT_SIZE * FRUIT_INPUT_SIZE);

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

    return true;
}

extern "C" uint32_t fruit_classifier_get_inference_time_us(void) {
    return last_inference_us;
}
