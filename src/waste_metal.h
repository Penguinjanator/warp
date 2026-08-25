/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/*
 * waste_metal.h — the one thing the Metal backend offers that is not a
 * waste_kernels slot.
 *
 * The dispatch table is per row range, which is the right shape for the
 * trunk matvec and the wrong one for the expert apply: a single apply is
 * 3072 rows and measures 12 GB/s of index on this GPU against the CPU's
 * 24.5, while a layer's worth of them in one command buffer measures 126.
 * The unit that pays is a batch, so the batch has to reach the backend as
 * a batch. See docs/EXP1.md §5a.
 */
#ifndef WASTE_METAL_H
#define WASTE_METAL_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* One VQ3R apply. `rec` is the expert record's page-aligned base and the
 * index and per-row scales are byte offsets inside it; `lut` is the
 * page-aligned table base and `lut_off` a float offset into it. Both
 * alignments are what newBufferWithBytesNoCopy requires, and both are
 * already true — expert records come from the O_DIRECT allocator and the
 * LUT scratch was moved to it for this. */
typedef struct {
    float         *y;            /* host destination, m floats            */
    const uint8_t *rec;
    size_t         rec_bytes;
    uint32_t       idx_off, sc_off;
    const float   *lut;
    size_t         lut_bytes;
    uint32_t       lut_off;
    int            m, nv, stages, entries;
} waste_vq_job;

/* 0 if every job ran on the device, non-zero if the caller must do them
 * itself. Never partially applied: on failure nothing was written. */
int waste_metal_vq3r(const waste_vq_job *jobs, int n);

#ifdef __cplusplus
}
#endif
#endif /* WASTE_METAL_H */
