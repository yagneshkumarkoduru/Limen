/* Integer windowed features; golden-model mirror of the Python reference. */
#include "limen_features.h"

uint64_t limen_isqrt_u64(uint64_t value)
{
    uint64_t x;
    uint64_t x1;
    if (value == 0U)
    {
        return 0U;
    }
    x  = value;
    x1 = (x + 1U) / 2U;
    while (x1 < x)
    {
        x  = x1;
        x1 = (x + value / x) / 2U;
    }
    return x;
}

int64_t limen_floordiv_i64(int64_t a, int64_t b)
{
    int64_t q = a / b;
    if ((a % b) != 0 && ((a < 0) != (b < 0)))
    {
        q--;
    }
    return q;
}

void limen_features_extract(
    const int32_t *emg,
    const int32_t *imu,
    int64_t       *features)
{
    uint32_t ch;
    uint32_t i;
    uint32_t f = 0U;
    uint32_t n = LIMEN_EMG_WINDOW;

    for (ch = 0U; ch < LIMEN_EMG_CHANNELS; ch++)
    {
        int64_t abs_sum = 0;
        int64_t sq_sum  = 0;
        int64_t wl_sum  = 0;
        int64_t zc      = 0;
        int64_t ssc     = 0;
        int32_t prev    = emg[ch];

        for (i = 0U; i < n; i++)
        {
            int32_t x  = emg[i * LIMEN_EMG_CHANNELS + ch];
            int64_t ax = (x < 0) ? -(int64_t)x : (int64_t)x;
            abs_sum += ax;
            sq_sum  += (int64_t)x * (int64_t)x;
            if (i > 0U)
            {
                int64_t d = (int64_t)x - (int64_t)prev;
                wl_sum += (d < 0) ? -d : d;
                if (i < n - 1U)
                {
                    int32_t nxt = emg[(i + 1U) * LIMEN_EMG_CHANNELS + ch];
                    int64_t d2  = (int64_t)nxt - (int64_t)x;
                    if (((x >  LIMEN_ZC_DEADBAND_Q8) && (prev < -LIMEN_ZC_DEADBAND_Q8)) ||
                        ((x < -LIMEN_ZC_DEADBAND_Q8) && (prev >  LIMEN_ZC_DEADBAND_Q8)))
                    {
                        zc++;
                    }
                    if ((d * d2 > 0) &&
                        ((d >  LIMEN_ZC_DEADBAND_Q8) || (d < -LIMEN_ZC_DEADBAND_Q8)) &&
                        ((d2 > LIMEN_ZC_DEADBAND_Q8) || (d2 < -LIMEN_ZC_DEADBAND_Q8)))
                    {
                        ssc++;
                    }
                }
            }
            prev = x;
        }
        features[f++] = abs_sum / (int64_t)n;
        features[f++] = (int64_t)limen_isqrt_u64((uint64_t)(sq_sum / (int64_t)n));
        features[f++] = wl_sum / (int64_t)n;
        features[f++] = zc;
        features[f++] = ssc;
    }

    n = LIMEN_IMU_WINDOW;
    for (ch = 0U; ch < LIMEN_IMU_CHANNELS; ch++)
    {
        int64_t sum  = 0;
        int64_t mad  = 0;
        int64_t mean;
        for (i = 0U; i < n; i++)
        {
            sum += (int64_t)imu[i * LIMEN_IMU_CHANNELS + ch];
        }
        mean = limen_floordiv_i64(sum, (int64_t)n);
        for (i = 0U; i < n; i++)
        {
            int64_t dev = (int64_t)imu[i * LIMEN_IMU_CHANNELS + ch] - mean;
            mad += (dev < 0) ? -dev : dev;
        }
        features[f++] = mean;
        features[f++] = mad / (int64_t)n;
    }
}
