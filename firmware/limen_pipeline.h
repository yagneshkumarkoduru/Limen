/* Full integer decision loop: window -> features -> SQG -> inference -> supervisor. */
#ifndef LIMEN_PIPELINE_H
#define LIMEN_PIPELINE_H

#include <stdint.h>
#include "limen_sqg.h"
#include "limen_supervisor.h"
#include "limen_trace.h"

typedef struct
{
    limen_supervisor_t supervisor;
    limen_trace_t     *trace;
    uint32_t           sequence;
} limen_pipeline_t;

typedef struct
{
    int32_t        class_idx;
    int32_t        confidence_pm;
    int32_t        reliability_pm;
    limen_state_t  state;
    int32_t        torque_mnm;
    int32_t        velocity_urad_s;
    int32_t        veto;
    uint32_t       effective_confidence_pm;
} limen_pipeline_output_t;

void limen_pipeline_init(limen_pipeline_t *pipeline, limen_trace_t *trace);
void limen_pipeline_step(
    limen_pipeline_t        *pipeline,
    const int32_t           *emg,
    const int32_t           *imu,
    uint32_t                 timestamp_us,
    limen_pipeline_output_t *output);

#endif /* LIMEN_PIPELINE_H */
