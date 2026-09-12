/* Host conformance harness: firmware must reproduce every golden vector exactly. */
#include <stdio.h>
#include <string.h>
#include <inttypes.h>
#include "limen_pipeline.h"
#include "limen_inference.h"
#include "limen_model.h"
#include "golden_conformance.h"

static int compare_expected(
    const golden_expected_t         *exp,
    const limen_pipeline_output_t   *out)
{
    return (exp->class_idx      == out->class_idx)
        && (exp->conf_pm        == out->confidence_pm)
        && (exp->rel_pm         == out->reliability_pm)
        && (exp->state          == (int32_t)out->state)
        && (exp->torque_mnm     == out->torque_mnm)
        && (exp->velocity_urad_s == out->velocity_urad_s)
        && (exp->veto           == out->veto);
}

static void report_case(
    const char                      *suite,
    uint32_t                         index,
    const golden_expected_t         *exp,
    const limen_pipeline_output_t   *out)
{
    printf("FAIL %s[%" PRIu32 "] expected {cls=%d conf=%d rel=%d state=%d tq=%d vel=%d veto=%d} "
           "got {cls=%d conf=%d rel=%d state=%d tq=%d vel=%d veto=%d}\n",
           suite, index,
           exp->class_idx, exp->conf_pm, exp->rel_pm, exp->state,
           exp->torque_mnm, exp->velocity_urad_s, exp->veto,
           out->class_idx, out->confidence_pm, out->reliability_pm, (int)out->state,
           out->torque_mnm, out->velocity_urad_s, out->veto);
}

int main(void)
{
    limen_trace_t            trace;
    limen_pipeline_t         pipeline;
    limen_pipeline_output_t  out;
    uint32_t i;
    uint32_t failures = 0U;
    uint32_t checks   = 0U;
    uint32_t model_crc;

    model_crc = limen_model_crc32();
    if (model_crc != LIMEN_MODEL_CRC32)
    {
        printf("FAIL model crc: computed 0x%08" PRIX32 " expected 0x%08X\n",
               model_crc, (unsigned)LIMEN_MODEL_CRC32);
        return 1;
    }
    printf("model crc: OK (0x%08" PRIX32 ")\n", model_crc);

    for (i = 0U; i < GOLDEN_SINGLE_COUNT; i++)
    {
        limen_trace_init(&trace);
        limen_pipeline_init(&pipeline, &trace);
        limen_pipeline_step(&pipeline, golden_single_emg[i], golden_single_imu[i],
                            i * 20000U, &out);
        checks++;
        if (!compare_expected(&golden_single_expected[i], &out))
        {
            failures++;
            report_case("single", i, &golden_single_expected[i], &out);
        }
    }

    limen_trace_init(&trace);
    limen_pipeline_init(&pipeline, &trace);
    for (i = 0U; i < GOLDEN_SEQUENCE_COUNT; i++)
    {
        limen_pipeline_step(&pipeline, golden_sequence_emg[i], golden_sequence_imu[i],
                            i * 20000U, &out);
        checks++;
        if (!compare_expected(&golden_sequence_expected[i], &out))
        {
            failures++;
            report_case("sequence", i, &golden_sequence_expected[i], &out);
        }
    }

    {
        limen_trace_entry_t entry;
        limen_trace_entry_t fetched;
        uint32_t k;
        limen_trace_init(&trace);
        memset(&entry, 0, sizeof(entry));
        for (k = 0U; k < 700U; k++)
        {
            entry.sequence    = k + 1U;
            entry.timestamp_us = k * 20U;
            entry.kind        = LIMEN_TRACE_KIND_DECISION;
            entry.a           = (int32_t)k;
            limen_trace_push(&trace, &entry);
        }
        checks++;
        if (trace.count != LIMEN_TRACE_CAPACITY || trace.dropped != 700U - LIMEN_TRACE_CAPACITY)
        {
            failures++;
            printf("FAIL trace wrap: count=%" PRIu32 " dropped=%" PRIu32 "\n",
                   trace.count, trace.dropped);
        }
        fetched = trace.entries[trace.head];
        checks++;
        if (!limen_trace_validate(&fetched) || fetched.sequence != 189U)
        {
            failures++;
            printf("FAIL trace oldest: valid=%d seq=%" PRIu32 "\n",
                   (int)limen_trace_validate(&fetched), fetched.sequence);
        }
        fetched.a ^= 1;
        checks++;
        if (limen_trace_validate(&fetched) != 0U)
        {
            failures++;
            printf("FAIL trace corruption not detected\n");
        }
    }

    printf("conformance: %" PRIu32 "/%" PRIu32 " checks passed\n", checks - failures, checks);
    return (failures == 0U) ? 0 : 1;
}
