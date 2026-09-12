/* Lock-free single-writer trace ring buffer with per-entry CRC8. */
#ifndef LIMEN_TRACE_H
#define LIMEN_TRACE_H

#include <stdint.h>

#define LIMEN_TRACE_CAPACITY        (512U)
#define LIMEN_TRACE_KIND_DECISION   (1U)
#define LIMEN_TRACE_KIND_FAULT      (2U)

typedef struct
{
    uint32_t sequence;
    uint32_t timestamp_us;
    uint8_t  kind;
    uint8_t  crc8;
    uint8_t  reserved;
    uint8_t  reserved2;
    int32_t  a;
    int32_t  b;
    int32_t  c;
} limen_trace_entry_t;

typedef struct
{
    limen_trace_entry_t entries[LIMEN_TRACE_CAPACITY];
    uint32_t            head;
    uint32_t            count;
    uint32_t            dropped;
} limen_trace_t;

void    limen_trace_init(limen_trace_t *trace);
void    limen_trace_push(limen_trace_t *trace, const limen_trace_entry_t *entry);
uint8_t limen_trace_entry_crc8(const limen_trace_entry_t *entry);
uint8_t limen_trace_validate(const limen_trace_entry_t *entry);

#endif /* LIMEN_TRACE_H */
