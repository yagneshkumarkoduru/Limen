/* Limen MCU reference interface.
 * Reference implementation only: no safety certification is claimed. */
#ifndef LIMEN_SUPERVISOR_H
#define LIMEN_SUPERVISOR_H

#include <stdint.h>

typedef enum
{
    LIMEN_STATE_SAFE_HALT = 0U,
    LIMEN_STATE_DEGRADED  = 1U,
    LIMEN_STATE_NOMINAL   = 2U
} limen_state_t;

typedef struct
{
    uint8_t  valid;
    uint8_t  intent_valid;
    uint16_t confidence_permille;
    uint16_t reliability_permille;
    int32_t  requested_torque_mnm;
    int32_t  requested_velocity_urad_s;
} limen_supervisor_input_t;

typedef struct
{
    limen_state_t state;
    int32_t       torque_mnm;
    int32_t       velocity_urad_s;
    uint32_t      effective_confidence_permille;
    uint8_t       supervisor_veto;
} limen_supervisor_output_t;

typedef struct
{
    limen_state_t state;
    uint32_t      low_confidence_streak;
} limen_supervisor_t;

void limen_supervisor_init(limen_supervisor_t *supervisor);
uint8_t limen_supervisor_step(
    limen_supervisor_t              *supervisor,
    const limen_supervisor_input_t  *input,
    limen_supervisor_output_t       *output);

#endif /* LIMEN_SUPERVISOR_H */
