/* Limen MCU reference implementation.
 * Fixed-point reference for comparison with Python and RTL models. */
#include "limen_supervisor.h"

#define LIMEN_T_HIGH_PERMILLE                   (750U)
#define LIMEN_T_LOW_PERMILLE                    (450U)
#define LIMEN_PERSISTENCE_WINDOWS               (3U)
#define LIMEN_DEGRADED_TORQUE_SCALE_PERMILLE    (400U)
#define LIMEN_DEGRADED_VELOCITY_SCALE_PERMILLE  (500U)
#define LIMEN_HARD_MAX_TORQUE_MNM               (40000)
#define LIMEN_HARD_MAX_VELOCITY_URAD_S          (6000000)

static void limen_zero_output(limen_supervisor_output_t *output, limen_state_t state)
{
    output->state                         = state;
    output->torque_mnm                    = 0;
    output->velocity_urad_s               = 0;
    output->effective_confidence_permille = 0U;
    output->supervisor_veto               = 0U;
}

void limen_supervisor_init(limen_supervisor_t *supervisor)
{
    if (supervisor != (limen_supervisor_t *)0)
    {
        supervisor->state                = LIMEN_STATE_SAFE_HALT;
        supervisor->low_confidence_streak = 0U;
    }
}

uint8_t limen_supervisor_step(
    limen_supervisor_t              *supervisor,
    const limen_supervisor_input_t  *input,
    limen_supervisor_output_t       *output)
{
    uint32_t product;
    uint32_t effective;
    int64_t  torque;
    int64_t  velocity;
    uint8_t  veto;

    if ((supervisor == (limen_supervisor_t *)0) ||
        (input      == (const limen_supervisor_input_t *)0) ||
        (output     == (limen_supervisor_output_t *)0))
    {
        return 0U;
    }

    if ((input->valid == 0U) || (input->intent_valid == 0U) ||
        (input->confidence_permille  > 1000U) ||
        (input->reliability_permille > 1000U))
    {
        supervisor->state                = LIMEN_STATE_SAFE_HALT;
        supervisor->low_confidence_streak = 0U;
        limen_zero_output(output, LIMEN_STATE_SAFE_HALT);
        return 1U;
    }

    product   = (uint32_t)input->confidence_permille *
                (uint32_t)input->reliability_permille;
    effective = product / 1000U;

    if (effective < LIMEN_T_LOW_PERMILLE)
    {
        supervisor->low_confidence_streak++;
        if (supervisor->low_confidence_streak >= LIMEN_PERSISTENCE_WINDOWS)
        {
            supervisor->state = LIMEN_STATE_SAFE_HALT;
        }
        else
        {
            supervisor->state = LIMEN_STATE_DEGRADED;
        }
    }
    else
    {
        supervisor->low_confidence_streak = 0U;
        if (effective >= LIMEN_T_HIGH_PERMILLE)
        {
            supervisor->state = LIMEN_STATE_NOMINAL;
        }
        else
        {
            supervisor->state = LIMEN_STATE_DEGRADED;
        }
    }

    torque   = input->requested_torque_mnm;
    velocity = input->requested_velocity_urad_s;

    if (supervisor->state == LIMEN_STATE_SAFE_HALT)
    {
        torque   = 0;
        velocity = 0;
    }
    else if (supervisor->state == LIMEN_STATE_DEGRADED)
    {
        torque   = (torque   * (int64_t)LIMEN_DEGRADED_TORQUE_SCALE_PERMILLE)   / 1000;
        velocity = (velocity * (int64_t)LIMEN_DEGRADED_VELOCITY_SCALE_PERMILLE) / 1000;
    }

    veto = 0U;
    if ((torque > LIMEN_HARD_MAX_TORQUE_MNM) || (torque < -LIMEN_HARD_MAX_TORQUE_MNM))
    {
        torque = LIMEN_HARD_MAX_TORQUE_MNM;
        veto   = 1U;
    }
    if ((velocity > LIMEN_HARD_MAX_VELOCITY_URAD_S) ||
        (velocity < -LIMEN_HARD_MAX_VELOCITY_URAD_S))
    {
        velocity = LIMEN_HARD_MAX_VELOCITY_URAD_S;
        veto     = 1U;
    }

    output->state                         = supervisor->state;
    output->torque_mnm                    = (int32_t)torque;
    output->velocity_urad_s               = (int32_t)velocity;
    output->effective_confidence_permille = effective;
    output->supervisor_veto               = veto;
    return 1U;
}
