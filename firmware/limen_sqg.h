/* Integer signal-quality gate; mirrors limen.safety.SignalQualityGate. */
#ifndef LIMEN_SQG_H
#define LIMEN_SQG_H

#include <stdint.h>

typedef struct
{
    uint16_t reliability_pm;
    uint8_t  flatline_channels;
    uint8_t  artifact_detected;
} limen_sqg_result_t;

uint16_t limen_sqg_assess(const int32_t *emg, limen_sqg_result_t *result);

#endif /* LIMEN_SQG_H */
