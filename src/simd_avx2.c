/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/*
 * simd_avx2.c — AVX2 versions of the two range kernels (see simd.h).
 *
 * Built with -mavx2 -mfma, so nothing here may be called before
 * waste_cpu_features() confirms the machine has them; backend.c is the only
 * caller of the registration function and checks first. The file compiles
 * to nothing on non-x86 targets.
 */

#include "simd.h"
#include "waste_backend.h"

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#include <stdlib.h>

/* Horizontal sum of eight floats. Used once per group of 128, not per
 * element, so the shuffle chain is amortized ~16x. */
static inline float hsum256(__m256 v)
{
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 0x55));
    return _mm_cvtss_f32(lo);
}

static void mvq_rows_f32_avx2(int b, int e, void *p)
{
    mvq_arg *a = (mvq_arg *)p;
    const int ng = a->ng, g = a->group;
    const float *x = (const float *)a->xs;
    int8_t *unp = NULL;
    if (a->bits != 8) {
        unp = (int8_t *)malloc((size_t)g);
        if (!unp) return;
    }
    for (int o = b; o < e; o++) {
        const int8_t *row = a->W + (size_t)o * a->rowbytes;
        const uint16_t *ws = a->ws + (size_t)o * ng;
        float acc = 0;
        for (int k = 0; k < ng; k++) {
            const int8_t *w;
            if (a->bits == 3) {
                for (int i = 0; i < g; i++)
                    unp[i] = (int8_t)waste_q3_at((const uint8_t *)row, (long)k * g + i);
                w = unp;
            } else if (a->bits == 4) {
                const uint8_t *p4 = (const uint8_t *)row + (size_t)k * g / 2;
                /* 16 packed bytes -> 32 nibbles, low first, biased by 8 */
                int i = 0;
                for (; i + 32 <= g; i += 32) {
                    const __m128i by = _mm_loadu_si128((const __m128i *)(p4 + i / 2));
                    const __m128i lo = _mm_and_si128(by, _mm_set1_epi8(0x0F));
                    const __m128i hi = _mm_and_si128(_mm_srli_epi16(by, 4),
                                                     _mm_set1_epi8(0x0F));
                    /* interleave so the order is byte0.lo, byte0.hi, ... */
                    const __m128i e0 = _mm_unpacklo_epi8(lo, hi);
                    const __m128i e1 = _mm_unpackhi_epi8(lo, hi);
                    const __m128i bias = _mm_set1_epi8(8);
                    _mm_storeu_si128((__m128i *)(unp + i), _mm_sub_epi8(e0, bias));
                    _mm_storeu_si128((__m128i *)(unp + i + 16), _mm_sub_epi8(e1, bias));
                }
                for (; i < g; i++) {
                    const uint8_t byte = p4[i >> 1];
                    unp[i] = (int8_t)((i & 1) ? (byte >> 4) - 8 : (byte & 0x0F) - 8);
                }
                w = unp;
            } else {
                w = row + (size_t)k * g;
            }
            const float *xx = x + (size_t)k * g;
            const int lim = (k * g + g <= a->in) ? g : a->in - k * g;
            __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
            int i = 0;
            for (; i + 16 <= lim; i += 16) {
                /* int8 -> int32 -> float, eight at a time */
                const __m256i w0 = _mm256_cvtepi8_epi32(
                    _mm_loadl_epi64((const __m128i *)(w + i)));
                const __m256i w1 = _mm256_cvtepi8_epi32(
                    _mm_loadl_epi64((const __m128i *)(w + i + 8)));
                s0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(w0),
                                     _mm256_loadu_ps(xx + i), s0);
                s1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(w1),
                                     _mm256_loadu_ps(xx + i + 8), s1);
            }
            float part = hsum256(_mm256_add_ps(s0, s1));
            for (; i < lim; i++) part += (float)w[i] * xx[i];
            acc += waste_f16(ws[k]) * part;
        }
        a->y[o] = acc;
    }
    free(unp);
}

static void lutb_range_avx2(int lo, int hi, void *p)
{
    const lutb_arg *a = (const lutb_arg *)p;
    const int en = a->entries, vd = a->vec_dim, st = a->stages;
    for (int v = lo; v < hi; v++) {
        const float *xv = a->x + (size_t)v * vd;
        for (int s = 0; s < st; s++) {
            const float *CT = a->booksT + (size_t)(a->cb_base + s) * en * vd;
            float *dst = a->lut + ((size_t)v * st + s) * en;
            int c = 0;
            for (; c + 32 <= en; c += 32) {
                __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps(),
                       a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
                for (int d = 0; d < vd; d++) {
                    const __m256 xd = _mm256_set1_ps(xv[d]);
                    const float *cr = CT + (size_t)d * en + c;
                    a0 = _mm256_fmadd_ps(xd, _mm256_loadu_ps(cr), a0);
                    a1 = _mm256_fmadd_ps(xd, _mm256_loadu_ps(cr + 8), a1);
                    a2 = _mm256_fmadd_ps(xd, _mm256_loadu_ps(cr + 16), a2);
                    a3 = _mm256_fmadd_ps(xd, _mm256_loadu_ps(cr + 24), a3);
                }
                _mm256_storeu_ps(dst + c, a0);
                _mm256_storeu_ps(dst + c + 8, a1);
                _mm256_storeu_ps(dst + c + 16, a2);
                _mm256_storeu_ps(dst + c + 24, a3);
            }
            for (; c < en; c++) {
                float t = 0;
                for (int d = 0; d < vd; d++) t += xv[d] * CT[(size_t)d * en + c];
                dst[c] = t;
            }
        }
    }
}

const char *waste_register_avx2(waste_kernels *t)
{
    t->mvq_rows_f32 = mvq_rows_f32_avx2;
    t->lutb_range = lutb_range_avx2;
    return "AVX2";
}

#endif /* x86 */
