/* Full integer decision loop wiring the golden-model stages together. */
#include "limen_pipeline.h"
#include "limen_features.h"
#include "limen_inference.h"
#include "limen_model.h"

void limen_pipeline_init(limen_pipeline_t *pipeline, limen_trace_t *trace)
{
    limen_supervisor_init(&pipeline->supervisor);
    pipeline->trace    = trace;
    pipeline->sequence = 0U;
}

void limen_pipeline_step(
    limen_pipeline_t        *pipeline,
    const int32_t           *emg,
    const int32_t           *imu,
    uint32_t                 timestamp_us,
    limen_pipeline_output_t *output)
{
    int64_t                   features[LIMEN_FEATURE_COUNT];
    limen_sqg_result_t        sqg;
    limen_inference_result_t  inference;
    limen_supervisor_input_t  sup_in;
    limen_supervisor_output_t sup_out;
    limen_trace_entry_t       entry;

    limen_features_extract(emg, imu, features);
    limen_sqg_assess(emg, &sqg);
    limen_inference_predict(features, &inference);

    sup_in.valid                    = 1U;
    sup_in.intent_valid             = 1U;
    sup_in.confidence_permille      = (uint16_t)inference.confidence_pm;
    sup_in.reliability_permille     = sqg.reliability_pm;
    sup_in.requested_torque_mnm     = limen_model_request_torque_mnm[inference.class_idx];
    sup_in.requested_velocity_urad_s = limen_model_request_velocity_urad_s[inference.class_idx];
    (void)limen_supervisor_step(&pipeline->supervisor, &sup_in, &sup_out);

    pipeline->sequence++;
    output->class_idx               = inference.class_idx;
    output->confidence_pm           = inference.confidence_pm;
    output->reliability_pm          = sqg.reliability_pm;
    output->state                   = sup_out.state;
    output->torque_mnm              = sup_out.torque_mnm;
    output->velocity_urad_s         = sup_out.velocity_urad_s;
    output->veto                    = (int32_t)sup_out.supervisor_veto;
    output->effective_confidence_pm = sup_out.effective_confidence_permille;

    if (pipeline->trace != (limen_trace_t *)0)
    {
        entry.sequence    = pipeline->sequence;
        entry.timestamp_us = timestamp_us;
        entry.kind        = LIMEN_TRACE_KIND_DECISION;
        entry.reserved    = 0U;
        entry.reserved2   = 0U;
        entry.a           = sup_out.torque_mnm;
        entry.b           = (int32_t)sup_out.state;
        entry.c           = (int32_t)sup_out.effective_confidence_permille;
        limen_trace_push(pipeline->trace, &entry);
    }
}
