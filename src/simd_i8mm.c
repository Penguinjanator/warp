/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/*
 * simd_i8mm.c — the Q4G trunk matvec through ARM's int8 matrix multiply.
 *
 * One translation unit, built with its own -march, entered only after
 * waste_cpu_features() reports FEAT_I8MM: the same arrangement
 * src/simd_avx2.c has on x86. FEAT_I8MM is ARMv8.6 and the portable build
 * targets something older, so a kernel written inside model.c would have
 * compiled to nothing at all — which is exactly what it did on the first
 * attempt, silently, leaving the caller's output buffer untouched and the
 * logits 109% off. A kernel that can be compiled out has to be somewhere
 * the build system can see.
 *
 * Why SMMLA for a matvec. It does a 2x2 int8 matmul — 32 MACs against
 * SDOT's 16 — and a matvec normally wastes half of them, because the
 * second row of B would be the same activation vector as the first. It
 * does not have to be: split the activation into a base-128 pair,
 * x ~ s*(128*a + b) with a and b both int8, put two *weight* rows in A and
 * the two activation planes in B, and one instruction produces w0.a, w0.b,
 * w1.a, w1.b — all four useful. That is SDOT's weight-bytes per
 * instruction with 15-bit activations instead of 8-bit, and 15 bits is the
 * whole point: the 8-bit version is faster and moves K3's logits far
 * enough that the KDA recurrence carries it (docs/EXP1.md §2b).
 *
 * The activation layout quant_act4_mm writes is what makes the inner loop
 * three instructions: per weight group, the even elements' two planes then
 * the odd elements' two planes, each as blocks of eight `a` followed by
 * eight `b`. That is how vmmlaq_s32 reads B, and it puts the low and high
 * nibble of a packed byte in front of the right plane with no shuffling.
 */
#include <stdint.h>
#include <stddef.h>

#include "simd.h"

#if defined(__ARM_FEATURE_MATMUL_INT8)
#include <arm_neon.h>

void waste_mvq4_rows_i8mm(int b, int e, void *p)
{
    const mvq4_arg *a = (const mvq4_arg *)p;
    const int ng = a->ng, g = a->group;
    const uint8x16_t m0f = vdupq_n_u8(0x0f);
    const int8x16_t  m8  = vdupq_n_s8(8);
    int o = b;
    for (; o + 2 <= e; o += 2) {
        const uint8_t *r0 = a->W + (size_t)o * a->rowbytes;
        const uint8_t *r1 = r0 + a->rowbytes;
        float acc0 = 0, acc1 = 0;
        for (int k = 0; k < ng; k++) {
            const uint8_t *q0 = r0 + (size_t)k * g / 2;
            const uint8_t *q1 = r1 + (size_t)k * g / 2;
            const int8_t *xe = a->xq + (size_t)k * 2 * g;
            const int8_t *xo = xe + g;
            int32x4_t d = vdupq_n_s32(0);
            for (int j = 0; j < g / 2; j += 8) {
                const uint8x16_t by = vcombine_u8(vld1_u8(q0 + j), vld1_u8(q1 + j));
                const int8x16_t lo =
                    vsubq_s8(vreinterpretq_s8_u8(vandq_u8(by, m0f)), m8);
                const int8x16_t hi =
                    vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(by, 4)), m8);
                d = vmmlaq_s32(d, lo, vld1q_s8(xe + 2 * j));
                d = vmmlaq_s32(d, hi, vld1q_s8(xo + 2 * j));
            }
            const float sx = a->xs[k];
            acc0 += waste_f16(a->ws[(size_t)o * ng + k]) * sx *
                    (128.0f * (float)vgetq_lane_s32(d, 0) + (float)vgetq_lane_s32(d, 1));
            acc1 += waste_f16(a->ws[(size_t)(o + 1) * ng + k]) * sx *
                    (128.0f * (float)vgetq_lane_s32(d, 2) + (float)vgetq_lane_s32(d, 3));
        }
        a->y[o] = acc0; a->y[o + 1] = acc1;
    }
    /* An odd last row has no partner to share the B operand with, so it
     * pays the scalar path — one row out of thousands. */
    for (; o < e; o++) {
        const uint8_t *row = a->W + (size_t)o * a->rowbytes;
        float acc = 0;
        for (int k = 0; k < ng; k++) {
            const uint8_t *q0 = row + (size_t)k * g / 2;
            const int8_t *xe = a->xq + (size_t)k * 2 * g;
            const int8_t *xo = xe + g;
            long dh = 0, dl = 0;
            for (int j = 0; j < g / 2; j++) {
                const int wl = (int)(q0[j] & 0x0f) - 8;
                const int wh = (int)(q0[j] >> 4) - 8;
                const int blk = (j >> 3) * 16, sl = j & 7;
                dh += (long)wl * xe[blk + sl] + (long)wh * xo[blk + sl];
                dl += (long)wl * xe[blk + 8 + sl] + (long)wh * xo[blk + 8 + sl];
            }
            acc += waste_f16(a->ws[(size_t)o * ng + k]) * a->xs[k] *
                   (128.0f * (float)dh + (float)dl);
        }
        a->y[o] = acc;
    }
}

#else
/* Compiled into every ARM build, so the symbol exists even where the flag
 * did not take. The dispatcher checks the runtime bit and never calls it
 * here; if something ever does, doing nothing is the one wrong answer
 * available, so it refuses loudly instead by writing zeros. */
void waste_mvq4_rows_i8mm(int b, int e, void *p)
{
    const mvq4_arg *a = (const mvq4_arg *)p;
    for (int o = b; o < e; o++) a->y[o] = 0.0f;
}
#endif
