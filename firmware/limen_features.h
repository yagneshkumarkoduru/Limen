/* Integer windowed features; mirrors limen.preprocessing.extract_features exactly. */
#ifndef LIMEN_FEATURES_H
#define LIMEN_FEATURES_H

#include <stdint.h>

#define LIMEN_EMG_CHANNELS      (8U)
#define LIMEN_IMU_CHANNELS      (6U)
#define LIMEN_EMG_WINDOW        (200U)
#define LIMEN_IMU_WINDOW        (20U)
#define LIMEN_FEATURE_COUNT     (40U + 12U)
#define LIMEN_ZC_DEADBAND_Q8    (3200)
#define LIMEN_FLATLINE_STD_Q8   (1280)

uint64_t limen_isqrt_u64(uint64_t value);
int64_t  limen_floordiv_i64(int64_t a, int64_t b);

void limen_features_extract(
    const int32_t *emg,
    const int32_t *imu,
    int64_t       *features);

#endif /* LIMEN_FEATURES_H */
