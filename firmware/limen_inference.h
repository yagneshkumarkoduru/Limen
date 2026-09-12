/* Quantized LDA inference with exp-LUT confidence and model CRC binding. */
#ifndef LIMEN_INFERENCE_H
#define LIMEN_INFERENCE_H

#include <stdint.h>

typedef struct
{
    int32_t class_idx;
    int32_t confidence_pm;
} limen_inference_result_t;

uint32_t limen_model_crc32(void);
void limen_inference_predict(
    const int64_t            *features,
    limen_inference_result_t *result);

#endif /* LIMEN_INFERENCE_H */
