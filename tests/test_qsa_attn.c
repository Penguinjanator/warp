/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/*
 * test_qsa_attn.c — QSA's attention, against the loops it replaced.
 *
 *   ./test_qsa_attn
 *
 * waste_qwen_qsa_attn_heads scores four selected tokens at a time and sums
 * the values through NEON, where it used to do both one element at a time.
 * Neither is allowed to change a bit, and the way to get that wrong is not
 * the order of the sums — it is the rounding of each product. The loop this
 * replaced compiles to four independent products in one vector and a scalar
 * chain of adds, so every product is rounded on its own; written as
 * `s += q * k` the same arithmetic contracts to a fused multiply-add, which
 * rounds once. Both versions of that mistake produced logits that differed
 * in the last bits and generated different text a few hundred tokens in.
 * The reference below is the kernel's old one-token-at-a-time shape, with
 * each product and add written as waste_qwen_qsa_mac — the same definition
 * the kernel uses. Copied verbatim, `s += q * k` was not a fixed reference:
 * its rounding was the compiler's, so this test passed on clang at -O2,
 * failed on clang at -O1 (make asan on arm64) and failed on gcc on arm64
 * at every level with the vectorizer on, against a kernel that had not
 * changed. What it checks now is the claim the kernel makes: that four
 * tokens at a time, and values sixteen lanes at a time, give exactly what
 * one token at a time gives.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/qwen_qsa.h"

static void old_attn(int h0, int h1, const float *q, int Hq, int D,
                     const float *k, const float *v, int Hkv, int T,
                     const int *sel, int n_sel, float scale,
                     float *out, float *scratch)
{
    const int n_rep = Hkv > 0 ? Hq / Hkv : 1;
    float *scores = scratch;
    if (h1 > h0)
        memset(out + (size_t)h0 * D, 0, (size_t)(h1 - h0) * D * sizeof(float));
    if (!q || !k || !v || !sel || !scratch || n_sel < 1) return;
    for (int h = h0; h < h1; h++) {
        const int hv = h / (n_rep > 0 ? n_rep : 1);
        const float *qh = q + (size_t)h * D;
        float m = -1e30f;
        for (int i = 0; i < n_sel; i++) {
            const int t = sel[i];
            if (t < 0 || t >= T) { scores[i] = -1e30f; continue; }
            const float *kh = k + ((size_t)t * Hkv + hv) * D;
            float s = 0.0f;
            for (int d = 0; d < D; d++) s = waste_qwen_qsa_mac(s, qh[d], kh[d]);
            s *= scale;
            scores[i] = s;
            if (s > m) m = s;
        }
        float z = 0.0f;
        for (int i = 0; i < n_sel; i++) {
            scores[i] = expf(scores[i] - m);
            z += scores[i];
        }
        if (z < 1e-20f) z = 1e-20f;
        float *oh = out + (size_t)h * D;
        for (int i = 0; i < n_sel; i++) {
            const int t = sel[i];
            if (t < 0 || t >= T) continue;
            const float w = scores[i] / z;
            const float *vh = v + ((size_t)t * Hkv + hv) * D;
            for (int d = 0; d < D; d++) oh[d] = waste_qwen_qsa_mac(oh[d], w, vh[d]);
        }
    }
}

static unsigned rng = 99991u;
static float frnd(void) { rng = rng * 1103515245u + 12345u; return (float)((int)(rng >> 9) - 4194304) / 4194304.0f; }

int main(void)
{
    enum { D = 256, Hq = 8, Hkv = 2, T = 600, NSEL = 512 };
    static float q[Hq * D], k[T * Hkv * D], v[T * Hkv * D];
    static float o1[Hq * D], o2[Hq * D], sc1[NSEL], sc2[NSEL];
    static int sel[NSEL];
    int bad = 0;
    for (int it = 0; it < 40; it++) {
        for (size_t i = 0; i < sizeof q / sizeof *q; i++) q[i] = frnd();
        for (size_t i = 0; i < sizeof k / sizeof *k; i++) k[i] = frnd();
        for (size_t i = 0; i < sizeof v / sizeof *v; i++) v[i] = frnd();
        const int n_sel = 1 + (int)(rng % NSEL);
        for (int i = 0; i < n_sel; i++) {
            rng = rng * 1103515245u + 12345u;
            /* mostly valid, some out of range, as a short context gives */
            sel[i] = (rng % 50 == 0) ? -1 : (int)(rng % (T + 4));
        }
        old_attn(0, Hq, q, Hq, D, k, v, Hkv, T, sel, n_sel, 0.125f, o1, sc1);
        waste_qwen_qsa_attn_heads(0, Hq, q, Hq, D, k, v, Hkv, T, sel, n_sel, 0.125f, o2, sc2);
        if (memcmp(o1, o2, sizeof o1)) {
            int first = -1;
            for (int i = 0; i < Hq * D; i++) if (memcmp(&o1[i], &o2[i], 4)) { first = i; break; }
            printf("iter %d n_sel %d: first differing output %d: %.9g vs %.9g\n",
                   it, n_sel, first, o1[first], o2[first]);
            bad++;
        }
    }
    printf("%s: %d of 40 cases differ\n", bad ? "FAIL" : "ok", bad);
    return bad ? 1 : 0;
}
