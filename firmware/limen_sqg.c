/* Integer signal-quality gate; golden-model mirror of the Python reference. */
#include "limen_sqg.h"
#include "limen_features.h"

uint16_t limen_sqg_assess(const int32_t *emg, limen_sqg_result_t *result)
{
    uint32_t n     = LIMEN_EMG_WINDOW;
    uint32_t ch;
    uint32_t i;
    int64_t  means[LIMEN_EMG_CHANNELS];
    int64_t  variance[LIMEN_EMG_CHANNELS];
    int64_t  rel   = 1000;
    uint32_t flat  = 0U;
    int64_t  peak  = 0;
    int64_t  clipped = 0;
    int64_t  common_num = 0;
    int64_t  den_all    = 0;
    uint32_t total = LIMEN_EMG_WINDOW * LIMEN_EMG_CHANNELS;

    for (ch = 0U; ch < LIMEN_EMG_CHANNELS; ch++)
    {
        int64_t sum  = 0;
        int64_t mean;
        for (i = 0U; i < n; i++)
        {
            sum += (int64_t)emg[i * LIMEN_EMG_CHANNELS + ch];
        }
        mean = limen_floordiv_i64(sum, (int64_t)n);
        means[ch]    = mean;
        variance[ch] = 0;
        for (i = 0U; i < n; i++)
        {
            int64_t dev = (int64_t)emg[i * LIMEN_EMG_CHANNELS + ch] - mean;
            variance[ch] += dev * dev;
        }
    }

    for (ch = 0U; ch < LIMEN_EMG_CHANNELS; ch++)
    {
        if ((int64_t)limen_isqrt_u64((uint64_t)(variance[ch] / (int64_t)n)) < LIMEN_FLATLINE_STD_Q8)
        {
            flat++;
        }
    }

    if (flat > 0U)
    {
        int64_t penalty = 1000 - 250 * (int64_t)flat;
        rel = rel * penalty / 1000;
        if (rel < 0)
        {
            rel = 0;
        }
    }

    for (i = 0U; i < total; i++)
    {
        int64_t ax = (emg[i] < 0) ? -(int64_t)emg[i] : (int64_t)emg[i];
        if (ax > peak)
        {
            peak = ax;
        }
    }
    {
        int64_t clip_threshold = peak * 98 / 100;
        for (i = 0U; i < total; i++)
        {
            int64_t ax = (emg[i] < 0) ? -(int64_t)emg[i] : (int64_t)emg[i];
            if (ax > clip_threshold)
            {
                clipped++;
            }
        }
    }
    if (clipped * 100 > (int64_t)total)
    {
        rel = rel * 600 / 1000;
    }

    for (i = 0U; i < n; i++)
    {
        int64_t c_sum  = 0;
        int64_t common;
        for (ch = 0U; ch < LIMEN_EMG_CHANNELS; ch++)
        {
            int64_t dev = (int64_t)emg[i * LIMEN_EMG_CHANNELS + ch] - means[ch];
            c_sum   += dev;
            den_all += dev * dev;
        }
        common = limen_floordiv_i64(c_sum, (int64_t)LIMEN_EMG_CHANNELS);
        common_num += common * common;
    }
    result->artifact_detected = 0U;
    if (den_all > 0)
    {
        int64_t ratio_q16 = common_num * (int64_t)LIMEN_EMG_CHANNELS * 65536 / den_all;
        if (ratio_q16 > 163840)
        {
            result->artifact_detected = 1U;
            rel = rel * 300 / 1000;
        }
    }

    result->flatline_channels = (uint8_t)flat;
    result->reliability_pm    = (uint16_t)rel;
    return result->reliability_pm;
}
