/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/*
 * mvqbw.c — how fast is the trunk matvec, really?
 *
 * K3's trunk is 28.04 GB of Q4G and every byte of it is read once per
 * token, which docs/LEARNED.md §59 puts at 75% of the bytes a decode step
 * touches. The kernel that reads it (waste_mvq_rows_f32, bits == 4) does a
 * scalar nibble unpack into a malloc'd staging buffer and then an f32 FMA
 * loop over the staged bytes. This measures that against the alternatives
 * at K3's real shapes, in GB/s and GMAC/s, so the headroom is a number
 * rather than a suspicion.
 *
 *   cc -O3 -mcpu=native -Isrc -o mvqbw tools/mvqbw.c -lm -lpthread
 *   ./mvqbw [threads] [reps]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

#include "threads.h"
#include "simd.h"

static double now(void)
{ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec/1e9; }

/* ---- the kernel under test, copied verbatim from src/model.c ---------- */
static void k_ref(int b, int e, void *p)
{
    mvq_arg *a = (mvq_arg *)p;
    const int ng = a->ng, g = a->group;
    const float *x = (const float *)a->xs;
    int8_t *unp = (int8_t *)malloc((size_t)g);
    if (!unp) return;
    for (int o = b; o < e; o++) {
        const int8_t *row = a->W + (size_t)o * a->rowbytes;
        const uint16_t *ws = a->ws + (size_t)o * ng;
        float acc = 0;
        for (int k = 0; k < ng; k++) {
            const uint8_t *p4 = (const uint8_t *)row + (size_t)k * g / 2;
            for (int i = 0; i < g / 2; i++) {
                const uint8_t byte = p4[i];
                unp[2 * i]     = (int8_t)(byte & 0x0F) - 8;
                unp[2 * i + 1] = (int8_t)(byte >> 4) - 8;
            }
            const int8_t *w = unp;
            const float *xx = x + (size_t)k * g;
            const int lim = (k * g + g <= a->in) ? g : a->in - k * g;
            float32x4_t s0 = vdupq_n_f32(0);
            int i = 0;
            for (; i + 8 <= lim; i += 8) {
                const int16x8_t w16 = vmovl_s8(vld1_s8(w + i));
                s0 = vfmaq_f32(s0, vcvtq_f32_s32(vmovl_s16(vget_low_s16(w16))),
                               vld1q_f32(xx + i));
                s0 = vfmaq_f32(s0, vcvtq_f32_s32(vmovl_s16(vget_high_s16(w16))),
                               vld1q_f32(xx + i + 4));
            }
            float part = vaddvq_f32(s0);
            for (; i < lim; i++) part += (float)w[i] * xx[i];
            acc += waste_f16(ws[k]) * part;
        }
        a->y[o] = acc;
    }
    free(unp);
}

/* ---- candidate 1: the same arithmetic with the unpack fused ------------
 * Same FMA order into the same 4-lane accumulator, so the result is bit
 * for bit what k_ref produces; only the staging buffer is gone. */
static void k_neon(int b, int e, void *p)
{
    mvq_arg *a = (mvq_arg *)p;
    const int ng = a->ng, g = a->group;
    const float *x = (const float *)a->xs;
    const uint8x16_t m0f = vdupq_n_u8(0x0f);
    const int8x16_t  m8  = vdupq_n_s8(8);
    for (int o = b; o < e; o++) {
        const uint8_t *row = (const uint8_t *)a->W + (size_t)o * a->rowbytes;
        const uint16_t *ws = a->ws + (size_t)o * ng;
        float acc = 0;
        for (int k = 0; k < ng; k++) {
            const uint8_t *p4 = row + (size_t)k * g / 2;
            const float *xx = x + (size_t)k * g;
            float32x4_t s0 = vdupq_n_f32(0);
            /* 16 packed bytes -> 32 weights, in the converter's order */
            for (int j = 0; j < g / 2; j += 16) {
                const uint8x16_t by = vld1q_u8(p4 + j);
                int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(by, m0f)), m8);
                int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(by, 4)), m8);
                const int8x16x2_t z = vzipq_s8(lo, hi);
                const float *xp = xx + 2 * j;
                for (int h = 0; h < 2; h++) {
                    const int16x8_t l16 = vmovl_s8(vget_low_s8(z.val[h]));
                    const int16x8_t h16 = vmovl_s8(vget_high_s8(z.val[h]));
                    s0 = vfmaq_f32(s0, vcvtq_f32_s32(vmovl_s16(vget_low_s16(l16))),  vld1q_f32(xp + 16*h + 0));
                    s0 = vfmaq_f32(s0, vcvtq_f32_s32(vmovl_s16(vget_high_s16(l16))), vld1q_f32(xp + 16*h + 4));
                    s0 = vfmaq_f32(s0, vcvtq_f32_s32(vmovl_s16(vget_low_s16(h16))),  vld1q_f32(xp + 16*h + 8));
                    s0 = vfmaq_f32(s0, vcvtq_f32_s32(vmovl_s16(vget_high_s16(h16))), vld1q_f32(xp + 16*h + 12));
                }
            }
            acc += waste_f16(ws[k]) * vaddvq_f32(s0);
        }
        a->y[o] = acc;
    }
}

/* ---- candidate 2: four rows at a time, one activation load shared ------ */
static void k_neon4(int b, int e, void *p)
{
    mvq_arg *a = (mvq_arg *)p;
    const int ng = a->ng, g = a->group;
    const float *x = (const float *)a->xs;
    const uint8x16_t m0f = vdupq_n_u8(0x0f);
    const int8x16_t  m8  = vdupq_n_s8(8);
    int o = b;
    for (; o + 4 <= e; o += 4) {
        float acc[4] = {0,0,0,0};
        for (int k = 0; k < ng; k++) {
            const float *xx = x + (size_t)k * g;
            float32x4_t s[4];
            for (int r = 0; r < 4; r++) s[r] = vdupq_n_f32(0);
            for (int j = 0; j < g / 2; j += 16) {
                float32x4_t xv[8];
                for (int q = 0; q < 8; q++) xv[q] = vld1q_f32(xx + 2*j + 4*q);
                for (int r = 0; r < 4; r++) {
                    const uint8_t *p4 = (const uint8_t *)a->W + (size_t)(o+r) * a->rowbytes + (size_t)k * g / 2;
                    const uint8x16_t by = vld1q_u8(p4 + j);
                    int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(by, m0f)), m8);
                    int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(by, 4)), m8);
                    const int8x16x2_t z = vzipq_s8(lo, hi);
                    for (int h = 0; h < 2; h++) {
                        const int16x8_t l16 = vmovl_s8(vget_low_s8(z.val[h]));
                        const int16x8_t h16 = vmovl_s8(vget_high_s8(z.val[h]));
                        s[r] = vfmaq_f32(s[r], vcvtq_f32_s32(vmovl_s16(vget_low_s16(l16))),  xv[4*h+0]);
                        s[r] = vfmaq_f32(s[r], vcvtq_f32_s32(vmovl_s16(vget_high_s16(l16))), xv[4*h+1]);
                        s[r] = vfmaq_f32(s[r], vcvtq_f32_s32(vmovl_s16(vget_low_s16(h16))),  xv[4*h+2]);
                        s[r] = vfmaq_f32(s[r], vcvtq_f32_s32(vmovl_s16(vget_high_s16(h16))), xv[4*h+3]);
                    }
                }
            }
            for (int r = 0; r < 4; r++)
                acc[r] += waste_f16(a->ws[(size_t)(o+r)*ng + k]) * vaddvq_f32(s[r]);
        }
        for (int r = 0; r < 4; r++) a->y[o+r] = acc[r];
    }
    if (o < e) k_neon(o, e, p);
}

/* ---- candidate 3: SDOT, activations quantized per group ----------------
 * xq/xs carry the int8 activations deinterleaved within each group: the
 * even half first, then the odd half, so the low and high nibbles of a
 * packed byte each face a contiguous activation vector and no zip is
 * needed. Numerics differ from the f32 path; this is the upper bound on
 * what the shape can do, not a drop-in. */
static void k_sdot(int b, int e, void *p)
{
#if defined(__ARM_FEATURE_DOTPROD)
    mvq_arg *a = (mvq_arg *)p;
    const int ng = a->ng, g = a->group;
    const uint8x16_t m0f = vdupq_n_u8(0x0f);
    const int8x16_t  m8  = vdupq_n_s8(8);
    for (int o = b; o < e; o++) {
        const uint8_t *row = (const uint8_t *)a->W + (size_t)o * a->rowbytes;
        const uint16_t *ws = a->ws + (size_t)o * ng;
        float acc = 0;
        for (int k = 0; k < ng; k++) {
            const int8_t *xe = a->xq + (size_t)k * g;        /* even half */
            const int8_t *xo = xe + g / 2;                   /* odd half  */
            const uint8_t *p4 = row + (size_t)k * g / 2;
            int32x4_t d0 = vdupq_n_s32(0), d1 = vdupq_n_s32(0);
            for (int j = 0; j < g / 2; j += 16) {
                const uint8x16_t by = vld1q_u8(p4 + j);
                const int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(by, m0f)), m8);
                const int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(by, 4)), m8);
                d0 = vdotq_s32(d0, lo, vld1q_s8(xe + j));
                d1 = vdotq_s32(d1, hi, vld1q_s8(xo + j));
            }
            acc += waste_f16(ws[k]) * a->xs[k] * (float)(vaddvq_s32(d0) + vaddvq_s32(d1));
        }
        a->y[o] = acc;
    }
#else
    (void)b; (void)e; (void)p;
#endif
}


/* ---- candidate 5: int16 activations through SMLAL ----------------------
 * The f32 path is not FMA-bound, it is conversion-bound: eight weights cost
 * one vmovl_s8, two vmovl_s16, two vcvtq and two vfma. Keeping the whole
 * dot in integers deletes the four conversions — the weights are already
 * int16 after one widening, and int16 activations are exact to 15 bits, so
 * unlike the int8 path this one does not move the model. */
static void k_smlal(int b, int e, void *p)
{
    mvq_arg *a = (mvq_arg *)p;
    const int ng = a->ng, g = a->group;
    const int16_t *x16 = (const int16_t *)a->xq;   /* deinterleaved, int16 */
    const uint8x16_t m0f = vdupq_n_u8(0x0f);
    const int8x16_t  m8  = vdupq_n_s8(8);
    for (int o = b; o < e; o++) {
        const uint8_t *row = (const uint8_t *)a->W + (size_t)o * a->rowbytes;
        const uint16_t *ws = a->ws + (size_t)o * ng;
        float acc = 0;
        for (int k = 0; k < ng; k++) {
            const uint8_t *p4 = row + (size_t)k * g / 2;
            const int16_t *xe = x16 + (size_t)k * g;
            const int16_t *xo = xe + g / 2;
            int32x4_t d0 = vdupq_n_s32(0), d1 = vdupq_n_s32(0);
            for (int j = 0; j < g / 2; j += 16) {
                const uint8x16_t by = vld1q_u8(p4 + j);
                const int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(by, m0f)), m8);
                const int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(by, 4)), m8);
                const int16x8_t l0 = vmovl_s8(vget_low_s8(lo)), l1 = vmovl_s8(vget_high_s8(lo));
                const int16x8_t h0 = vmovl_s8(vget_low_s8(hi)), h1 = vmovl_s8(vget_high_s8(hi));
                d0 = vmlal_s16(d0, vget_low_s16(l0),  vld1_s16(xe + j + 0));
                d1 = vmlal_s16(d1, vget_high_s16(l0), vld1_s16(xe + j + 4));
                d0 = vmlal_s16(d0, vget_low_s16(l1),  vld1_s16(xe + j + 8));
                d1 = vmlal_s16(d1, vget_high_s16(l1), vld1_s16(xe + j + 12));
                d0 = vmlal_s16(d0, vget_low_s16(h0),  vld1_s16(xo + j + 0));
                d1 = vmlal_s16(d1, vget_high_s16(h0), vld1_s16(xo + j + 4));
                d0 = vmlal_s16(d0, vget_low_s16(h1),  vld1_s16(xo + j + 8));
                d1 = vmlal_s16(d1, vget_high_s16(h1), vld1_s16(xo + j + 12));
            }
            acc += waste_f16(ws[k]) * a->xs[k] * (float)(vaddvq_s32(d0) + vaddvq_s32(d1));
        }
        a->y[o] = acc;
    }
}

/* ---- candidate 6: SMMLA, two rows and two activation planes ------------
 * i8mm's 2x2 int8 matmul does 32 MACs where SDOT does 16, and every one of
 * them is useful here: put two weight rows in A and the high and low halves
 * of a base-128 activation split in B, and one instruction produces
 * w0.xa, w0.xb, w1.xa, w1.xb. That is SDOT's weight-bytes per instruction
 * with 14-bit activations instead of 8-bit. */
static void k_smmla(int b, int e, void *p)
{
#if defined(__ARM_FEATURE_MATMUL_INT8)
    mvq_arg *a = (mvq_arg *)p;
    const int ng = a->ng, g = a->group;
    const int8_t *xp = a->xq;            /* xa_even|xb_even|xa_odd|xb_odd  */
    const uint8x16_t m0f = vdupq_n_u8(0x0f);
    const int8x16_t  m8  = vdupq_n_s8(8);
    int o = b;
    for (; o + 2 <= e; o += 2) {
        const uint8_t *r0 = (const uint8_t *)a->W + (size_t)o * a->rowbytes;
        const uint8_t *r1 = r0 + a->rowbytes;
        float acc0 = 0, acc1 = 0;
        for (int k = 0; k < ng; k++) {
            const uint8_t *q0 = r0 + (size_t)k * g / 2;
            const uint8_t *q1 = r1 + (size_t)k * g / 2;
            const int8_t *xe = xp + (size_t)k * 2 * g;        /* even plane */
            const int8_t *xo = xe + g;                        /* odd plane  */
            int32x4_t d = vdupq_n_s32(0);
            for (int j = 0; j < g / 2; j += 8) {
                const uint8x16_t by = vcombine_u8(vld1_u8(q0 + j), vld1_u8(q1 + j));
                const int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(by, m0f)), m8);
                const int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(by, 4)), m8);
                d = vmmlaq_s32(d, lo, vld1q_s8(xe + 2 * j));
                d = vmmlaq_s32(d, hi, vld1q_s8(xo + 2 * j));
            }
            const float w = waste_f16(a->ws[(size_t)o * ng + k]) * a->xs[k];
            const float w1 = waste_f16(a->ws[(size_t)(o + 1) * ng + k]) * a->xs[k];
            acc0 += w  * (128.0f * (float)vgetq_lane_s32(d, 0) + (float)vgetq_lane_s32(d, 1));
            acc1 += w1 * (128.0f * (float)vgetq_lane_s32(d, 2) + (float)vgetq_lane_s32(d, 3));
        }
        a->y[o] = acc0; a->y[o + 1] = acc1;
    }
#else
    (void)b; (void)e; (void)p;
#endif
}

/* ---- candidate 4: pure streaming read, the machine's own ceiling ------- */
static void k_read(int b, int e, void *p)
{
    mvq_arg *a = (mvq_arg *)p;
    uint64x2_t s = vdupq_n_u64(0);
    for (int o = b; o < e; o++) {
        const uint8_t *row = (const uint8_t *)a->W + (size_t)o * a->rowbytes;
        for (size_t i = 0; i + 16 <= a->rowbytes; i += 16)
            s = vaddq_u64(s, vreinterpretq_u64_u8(vld1q_u8(row + i)));
    }
    a->y[b] += (float)(vgetq_lane_u64(s, 0) & 1);
}

/* ----------------------------------------------------------------------- */

typedef struct { const char *name; waste_range_fn fn; int quantized_act; } cand;

int main(int argc, char **argv)
{
    const int nthreads = argc > 1 ? atoi(argv[1]) : 0;
    const int reps     = argc > 2 ? atoi(argv[2]) : 5;

    /* K3's dominant trunk shape: q/k/v/g_proj are [12288, 7168]. */
    const int M = 12288, K = 7168, G = 128;
    const int ng = K / G;
    const size_t rowbytes = (size_t)K / 2;
    const size_t wbytes = (size_t)M * rowbytes;

    waste_pool_init(nthreads, NULL);
    printf("threads %d   shape [%d x %d] Q4G   weights %.1f MB x %d copies = %.2f GB\n",
           waste_pool_threads(), M, K, wbytes / 1e6, argc > 3 ? atoi(argv[3]) : 2,
           wbytes * (argc > 3 ? atoi(argv[3]) : 2) / 1e9);

    /* Two copies so a rep does not re-read the last rep's bytes out of the
     * caches; 44 MB each is already past this machine's LLC but the SLC is
     * generous and a single buffer flatters every arm equally. */
    const int COPIES = argc > 3 ? atoi(argv[3]) : 2;
    uint8_t **W = (uint8_t **)malloc(sizeof(void*)*COPIES);
    uint16_t **S = (uint16_t **)malloc(sizeof(void*)*COPIES);
    for (int c = 0; c < COPIES; c++) {
        W[c] = (uint8_t *)aligned_alloc(16384, (wbytes + 16383) & ~(size_t)16383);
        S[c] = (uint16_t *)malloc((size_t)M * ng * 2);
        for (size_t i = 0; i < wbytes; i++) W[c][i] = (uint8_t)((i * 1103515245u + c) >> 7);
        for (size_t i = 0; i < (size_t)M * ng; i++) S[c][i] = 0x3800 + (i & 0x3f); /* ~0.5 */
    }
    float *x = (float *)malloc((size_t)K * 4);
    float *y = (float *)malloc((size_t)M * 4);
    float *yref = (float *)malloc((size_t)M * 4);
    int8_t *xq = (int8_t *)malloc((size_t)K);
    float *xs = (float *)malloc((size_t)ng * 4);
    for (int i = 0; i < K; i++) x[i] = (float)sin(i * 0.017) * 0.9f;
    /* per-group int8 activations, deinterleaved even|odd inside the group */
    for (int k = 0; k < ng; k++) {
        float amax = 0;
        for (int i = 0; i < G; i++) { float v = fabsf(x[k*G+i]); if (v > amax) amax = v; }
        const float s = amax > 0 ? amax / 127.0f : 1.0f;
        xs[k] = s;
        for (int i = 0; i < G; i += 2) {
            xq[k*G + i/2]         = (int8_t)lrintf(x[k*G+i]   / s);
            xq[k*G + G/2 + i/2]   = (int8_t)lrintf(x[k*G+i+1] / s);
        }
    }

    /* int16 activations, deinterleaved even|odd inside each group */
    int16_t *x16 = (int16_t *)malloc((size_t)(K + 256) * 2);
    float *xs16 = (float *)malloc((size_t)ng * 4);
    for (int k = 0; k < ng; k++) {
        float amax = 0;
        for (int i = 0; i < G; i++) { float v = fabsf(x[k*G+i]); if (v > amax) amax = v; }
        const float s = amax > 0 ? amax / 32767.0f : 1.0f;
        xs16[k] = s;
        for (int i = 0; i < G; i += 2) {
            x16[k*G + i/2]       = (int16_t)lrintf(x[k*G+i]   / s);
            x16[k*G + G/2 + i/2] = (int16_t)lrintf(x[k*G+i+1] / s);
        }
    }
    /* base-128 split, four planes per group: xa_even|xb_even|xa_odd|xb_odd */
    int8_t *x88 = (int8_t *)malloc((size_t)(K + 256) * 2);
    float *xs88 = (float *)malloc((size_t)ng * 4);
    for (int k = 0; k < ng; k++) {
        float amax = 0;
        for (int i = 0; i < G; i++) { float v = fabsf(x[k*G+i]); if (v > amax) amax = v; }
        const float s = amax > 0 ? amax / 8191.0f : 1.0f;
        xs88[k] = s;
        for (int i = 0; i < G; i++) {
            int q = (int)lrintf(x[k*G+i] / s);
            if (q > 8191) q = 8191; if (q < -8191) q = -8191;
            int hi = (q + 64) >> 7; if (hi > 63) hi = 63; if (hi < -64) hi = -64;
            int lo = q - 128 * hi;
            const int e2 = i & 1, h = i >> 1;              /* even/odd, index */
            /* plane layout: [even a(64) | even b(64)] then [odd a | odd b],
             * each pair of eight interleaved the way vmmlaq_s32 reads B */
            int8_t *base = x88 + (size_t)k * 2 * G + (e2 ? G : 0);
            base[(h / 8) * 16 + (h % 8)]     = (int8_t)hi;
            base[(h / 8) * 16 + 8 + (h % 8)] = (int8_t)lo;
        }
    }

    cand cands[] = {
        { "ref  (scalar unpack + f32 fma)", k_ref,   0 },
        { "neon (fused unpack, 1 row)",     k_neon,  0 },
        { "neon4(fused unpack, 4 rows)",    k_neon4, 0 },
        { "sdot (int8 acts, dotprod)",      k_sdot,  1 },
        { "smlal(int16 acts, vmlal_s16)",    k_smlal, 2 },
        { "smmla(14-bit acts, i8mm)",        k_smmla, 3 },
        { "read (memory ceiling)",          k_read,  0 },
    };
    const int NC = (int)(sizeof(cands)/sizeof(cands[0]));

    printf("\n%-34s %9s %9s %9s %9s %10s\n", "kernel", "ms(mean)", "GB/s",
           "GB/s best", "GMAC/s", "max|diff|");
    for (int c = 0; c < NC; c++) {
        double best = 1e18, sum = 0; int nsum = 0;
        for (int r = 0; r < reps; r++) {
            const int cp = r % COPIES;
            const int8_t *qq = cands[c].quantized_act == 2 ? (const int8_t *)x16
                             : cands[c].quantized_act == 3 ? x88 : xq;
            const float *ss = cands[c].quantized_act == 2 ? xs16
                            : cands[c].quantized_act == 3 ? xs88
                            : cands[c].quantized_act ? xs : x;
            mvq_arg a = { y, (const int8_t *)W[cp], S[cp], qq, ss,
                          K, ng, G, 4, rowbytes };
            const double t0 = now();
            waste_parallel_for(M, 64, cands[c].fn, &a);
            const double dt = now() - t0;
            if (dt < best) best = dt;
            if (r > 0) { sum += dt; nsum++; }
            if (getenv("MVQBW_TRACE") && (r % 200) == 0)
                printf("      [%s rep %4d] %6.1f GB/s\n", cands[c].name, r,
                       wbytes / dt / 1e9);
            if (r == 0 && c == 0) memcpy(yref, y, (size_t)M * 4);
        }
        /* correctness pass on copy 0 */
        const int8_t *qq = cands[c].quantized_act == 2 ? (const int8_t *)x16
                         : cands[c].quantized_act == 3 ? x88 : xq;
        const float *ss = cands[c].quantized_act == 2 ? xs16
                        : cands[c].quantized_act == 3 ? xs88
                        : cands[c].quantized_act ? xs : x;
        mvq_arg a = { y, (const int8_t *)W[0], S[0], qq, ss,
                      K, ng, G, 4, rowbytes };
        waste_parallel_for(M, 64, cands[c].fn, &a);
        double md = 0, den = 0;
        if (c != NC - 1)
            for (int i = 0; i < M; i++) {
                double d = fabs((double)y[i] - yref[i]); if (d > md) md = d;
                if (fabs(yref[i]) > den) den = fabs(yref[i]);
            }
        const double mean = nsum ? sum / nsum : best;
        printf("%-34s %9.3f %9.1f %9.1f %9.1f %10.3g%s\n", cands[c].name, mean*1e3,
               wbytes / mean / 1e9, wbytes / best / 1e9, (double)M*K / mean / 1e9,
               md, (c == NC-1) ? "  (n/a)" : (md == 0 ? "  exact" : ""));
    }
    printf("\nrel to |y|max %.4g\n", 0.0);
    waste_pool_shutdown();
    return 0;
}
