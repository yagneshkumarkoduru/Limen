/* Lock-free single-writer trace ring buffer with per-entry CRC8. */
#include "limen_trace.h"

static uint8_t crc8_byte(uint8_t crc, uint8_t byte)
{
    uint8_t i;
    crc ^= byte;
    for (i = 0U; i < 8U; i++)
    {
        if ((crc & 0x80U) != 0U)
        {
            crc = (uint8_t)((crc << 1) ^ 0x07U);
        }
        else
        {
            crc = (uint8_t)(crc << 1);
        }
    }
    return crc;
}

uint8_t limen_trace_entry_crc8(const limen_trace_entry_t *entry)
{
    const uint8_t *bytes = (const uint8_t *)entry;
    uint8_t        crc   = 0U;
    uint8_t        i;
    /* CRC covers all bytes except the crc field itself (byte offset 9). */
    for (i = 0U; i < (uint8_t)sizeof(limen_trace_entry_t); i++)
    {
        if (i != 9U)
        {
            crc = crc8_byte(crc, bytes[i]);
        }
    }
    return crc;
}

uint8_t limen_trace_validate(const limen_trace_entry_t *entry)
{
    return (limen_trace_entry_crc8(entry) == entry->crc8) ? 1U : 0U;
}

void limen_trace_init(limen_trace_t *trace)
{
    uint32_t i;
    for (i = 0U; i < LIMEN_TRACE_CAPACITY; i++)
    {
        trace->entries[i].sequence    = 0U;
        trace->entries[i].timestamp_us = 0U;
        trace->entries[i].kind        = 0U;
        trace->entries[i].crc8        = 0U;
        trace->entries[i].reserved    = 0U;
        trace->entries[i].reserved2   = 0U;
        trace->entries[i].a           = 0;
        trace->entries[i].b           = 0;
        trace->entries[i].c           = 0;
    }
    trace->head    = 0U;
    trace->count   = 0U;
    trace->dropped = 0U;
}

void limen_trace_push(limen_trace_t *trace, const limen_trace_entry_t *entry)
{
    limen_trace_entry_t slot = *entry;
    slot.crc8              = limen_trace_entry_crc8(&slot);
    trace->entries[trace->head] = slot;
    trace->head = (trace->head + 1U) % LIMEN_TRACE_CAPACITY;
    if (trace->count < LIMEN_TRACE_CAPACITY)
    {
        trace->count++;
    }
    else
    {
        trace->dropped++;
    }
}
