/* Quantized LDA inference; golden-model mirror of limen Python reference. */
#include "limen_inference.h"
#include "limen_model.h"

static uint32_t crc32_byte(uint32_t crc, uint8_t byte)
{
    uint32_t i;
    crc ^= (uint32_t)byte;
    for (i = 0U; i < 8U; i++)
    {
        if ((crc & 1U) != 0U)
        {
            crc = (crc >> 1U) ^ 0xEDB88320U;
        }
        else
        {
            crc >>= 1U;
        }
    }
    return crc;
}

uint32_t limen_model_crc32(void)
{
    uint32_t crc = 0xFFFFFFFFU;
    uint32_t c;
    uint32_t j;

    for (c = 0U; c < LIMEN_MODEL_N_CLASSES; c++)
    {
        for (j = 0U; j < LIMEN_MODEL_N_FEATURES; j++)
        {
            uint64_t v = (uint64_t)limen_model_coef_q16[c][j];
            uint8_t  b;
            b = (uint8_t)(v & 0xFFU);           crc = crc32_byte(crc, b);
            b = (uint8_t)((v >> 8)  & 0xFFU);   crc = crc32_byte(crc, b);
            b = (uint8_t)((v >> 16) & 0xFFU);   crc = crc32_byte(crc, b);
            b = (uint8_t)((v >> 24) & 0xFFU);   crc = crc32_byte(crc, b);
            b = (uint8_t)((v >> 32) & 0xFFU);   crc = crc32_byte(crc, b);
            b = (uint8_t)((v >> 40) & 0xFFU);   crc = crc32_byte(crc, b);
            b = (uint8_t)((v >> 48) & 0xFFU);   crc = crc32_byte(crc, b);
            b = (uint8_t)((v >> 56) & 0xFFU);   crc = crc32_byte(crc, b);
        }
    }
    for (c = 0U; c < LIMEN_MODEL_N_CLASSES; c++)
    {
        uint64_t v = (uint64_t)limen_model_intercept_q16[c];
        uint8_t  b;
        b = (uint8_t)(v & 0xFFU);           crc = crc32_byte(crc, b);
        b = (uint8_t)((v >> 8)  & 0xFFU);   crc = crc32_byte(crc, b);
        b = (uint8_t)((v >> 16) & 0xFFU);   crc = crc32_byte(crc, b);
        b = (uint8_t)((v >> 24) & 0xFFU);   crc = crc32_byte(crc, b);
        b = (uint8_t)((v >> 32) & 0xFFU);   crc = crc32_byte(crc, b);
        b = (uint8_t)((v >> 40) & 0xFFU);   crc = crc32_byte(crc, b);
        b = (uint8_t)((v >> 48) & 0xFFU);   crc = crc32_byte(crc, b);
        b = (uint8_t)((v >> 56) & 0xFFU);   crc = crc32_byte(crc, b);
    }
    return crc ^ 0xFFFFFFFFU;
}

void limen_inference_predict(
    const int64_t            *features,
    limen_inference_result_t *result)
{
    int64_t  scores[LIMEN_MODEL_N_CLASSES];
    uint32_t c;
    uint32_t j;
    uint32_t best  = 0U;
    int64_t  s_max;
    int64_t  denom = 0;

    for (c = 0U; c < LIMEN_MODEL_N_CLASSES; c++)
    {
        int64_t acc = (int64_t)limen_model_intercept_q16[c];
        for (j = 0U; j < LIMEN_MODEL_N_FEATURES; j++)
        {
            acc += (int64_t)limen_model_coef_q16[c][j] * features[j];
        }
        scores[c] = acc;
    }

    for (c = 1U; c < LIMEN_MODEL_N_CLASSES; c++)
    {
        if (scores[c] > scores[best])
        {
            best = c;
        }
    }
    s_max = scores[best];

    for (c = 0U; c < LIMEN_MODEL_N_CLASSES; c++)
    {
        int64_t diff = s_max - scores[c];
        if (diff > 524288)
        {
            diff = 524288;
        }
        denom += (int64_t)limen_model_exp_lut[(uint32_t)(diff >> 11)];
    }

    result->class_idx = (int32_t)best;
    if (denom <= 0)
    {
        result->confidence_pm = 1000;
    }
    else
    {
        int64_t conf = 4096 * 1000 / denom;
        result->confidence_pm = (conf > 1000) ? 1000 : (int32_t)conf;
    }
}
