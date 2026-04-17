#ifndef FRUIT_CLASSIFIER_H_
#define FRUIT_CLASSIFIER_H_

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Model input dimensions
#define FRUIT_INPUT_SIZE   48
#define FRUIT_NUM_CLASSES  4

// Classification result
typedef struct {
    int class_id;           // 0=apple, 1=banana, 2=orange, 3=background
    const char *class_name; // human-readable label
    int8_t confidence;      // raw INT8 output score (higher = more confident)
} fruit_result_t;

// Initialize the TF Lite interpreter. Call once at startup.
// Returns true on success.
bool fruit_classifier_init(void);

// Run inference on a 160x120 YUY2 frame buffer.
// Internally downsamples to 48x48 grayscale and runs the model.
// Result is written to *result.
// Returns true on success.
bool fruit_classifier_run(const uint8_t *yuy2_frame, uint16_t width, uint16_t height,
                          fruit_result_t *result);

// Get inference time of last run in microseconds.
uint32_t fruit_classifier_get_inference_time_us(void);

#ifdef __cplusplus
}
#endif

#endif  // FRUIT_CLASSIFIER_H_
