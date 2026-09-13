/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/* qwen_qsa.c — see qwen_qsa.h. Official QSA indexer, not MLA. */

#include "qwen_qsa.h"

#include <math.h>
#include <string.h>

void waste_qwen_mrope_interleave(const float *freqs_t, const float *freqs_h,
                                 const float *freqs_w, const int *section,
                                 int half, float *out)
{
    memcpy(out, freqs_t, (size_t)half * sizeof(float));
    const float *fh[3] = { freqs_t, freqs_h, freqs_w };
    for (int dim = 1; dim <= 2; dim++) {
        const int length = section[dim] * 3;
        for (int i = dim; i < length && i < half; i += 3)
            out[i] = fh[dim][i];
    }
}

int waste_qwen_rope_apply(float *x, int dim, const float *cos, const float *sin,
                          int rotary_dim)
{
    if (rotary_dim > dim) rotary_dim = dim;
    if (rotary_dim < 0) rotary_dim = 0;
    if (rotary_dim > 256) return -1;
    const int half = rotary_dim / 2;
    float tmp[256];
    memcpy(tmp, x, (size_t)rotary_dim * sizeof(float));
    for (int i = 0; i < rotary_dim; i++) {
        const float rh = (i < half) ? -tmp[i + half] : tmp[i - half];
        x[i] = tmp[i] * cos[i] + rh * sin[i];
    }
    (void)dim;
    return 0;
}

static void rmsnorm_1p(float *o, const float *x, const float *w, int n, float eps)
{
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += x[i] * x[i];
    const float r = 1.0f / sqrtf(s / (float)n + eps);
    for (int i = 0; i < n; i++) o[i] = x[i] * r * (1.0f + w[i]);
}

void waste_qwen_qsa_pool_block(const float *raw_k, int compress, int Dk,
                               const float *k_ln_w, float eps, float *out)
{
    if (!out || Dk < 1) return;
    memset(out, 0, (size_t)Dk * sizeof(float));
    if (!raw_k || !k_ln_w || compress < 1) return;
    for (int t = 0; t < compress; t++) {
        const float *row = raw_k + (size_t)t * Dk;
        for (int d = 0; d < Dk; d++) out[d] += row[d];
    }
    for (int d = 0; d < Dk; d++) out[d] /= (float)compress;
    rmsnorm_1p(out, out, k_ln_w, Dk, eps);
}

int waste_qwen_qsa_select(const float *q_heads, int Hq, int Dk,
                          const float *raw_k, int T, int query_pos,
                          const float *full_cos, const float *full_sin,
                          int rotary_dim, const float *k_ln_w, float eps,
                          int compress, int block_topk,
                          int *sel, float *work, int *taken)
{
    if (!sel || compress < 1 || Dk < 1) return 0;
    if (query_pos < 0) query_pos = 0;
    if (T < 1) return 0;
    if (query_pos >= T) query_pos = T - 1;
    const int vis = query_pos + 1;
    const int n_complete = vis / compress;
    const int n_tail = vis - n_complete * compress;
    int nsel = 0;

    if (n_complete > 0) {
        if (!work || !taken || !q_heads || !raw_k || !k_ln_w) return 0;
        waste_qwen_qsa_score_blocks(0, n_complete, q_heads, Hq, Dk, raw_k,
                                    full_cos, full_sin, rotary_dim, k_ln_w, eps,
                                    compress, work, work + (size_t)n_complete * Dk);
    }
    (void)nsel;
    return waste_qwen_qsa_pick(n_complete > 0 ? work + (size_t)n_complete * Dk : NULL,
                               n_complete, block_topk, compress, n_tail, sel, taken);
}

void waste_qwen_qsa_score_blocks(int b0, int b1, const float *q_heads, int Hq,
                                 int Dk, const float *raw_k,
                                 const float *full_cos, const float *full_sin,
                                 int rotary_dim, const float *k_ln_w, float eps,
                                 int compress, float *pooled, float *scores)
{
    const float inv_sqrt = 1.0f / sqrtf((float)Dk);
    for (int b = b0; b < b1; b++) {
        float *po = pooled + (size_t)b * Dk;
        waste_qwen_qsa_pool_block(raw_k + (size_t)b * compress * Dk,
                                  compress, Dk, k_ln_w, eps, po);
        if (full_cos && full_sin && rotary_dim > 0)
            waste_qwen_rope_apply(po, Dk,
                                  full_cos + (size_t)(b * compress) * rotary_dim,
                                  full_sin + (size_t)(b * compress) * rotary_dim,
                                  rotary_dim);
        float s = 0.0f;
        for (int h = 0; h < Hq; h++) {
            float dot = 0.0f;
            const float *qh = q_heads + (size_t)h * Dk;
            for (int d = 0; d < Dk; d++) dot += qh[d] * po[d];
            if (dot < 0.0f) dot = 0.0f;
            s += dot;
        }
        scores[b] = s * inv_sqrt;
    }
}

/* Whether block a comes before block b in the selection: the higher score,
 * and on a tie the earlier block. That is the order the repeated argmax this
 * replaced produced — it took the first strictly greater score each pass. */
static int qsa_before(const float *s, int a, int b)
{
    return s[a] > s[b] || (s[a] == s[b] && a < b);
}

static void qsa_sift(const float *s, int *o, int i, int n)
{
    for (;;) {
        int w = i;
        const int l = 2 * i + 1, r = l + 1;
        if (l < n && qsa_before(s, o[w], o[l])) w = l;
        if (r < n && qsa_before(s, o[w], o[r])) w = r;
        if (w == i) return;
        const int tmp = o[i]; o[i] = o[w]; o[w] = tmp;
        i = w;
    }
}

int waste_qwen_qsa_pick(const float *scores, int n_complete, int block_topk,
                        int compress, int n_tail, int *sel, int *order)
{
    int nsel = 0;
    if (n_complete > 0 && scores && order) {
        /* The argmax only ever took a score above -1e30, which also leaves
         * out a NaN; the ordering below is total over what remains. */
        int n = 0;
        for (int b = 0; b < n_complete; b++)
            if (scores[b] > -1e30f) order[n++] = b;
        /* Heapsort with the latest block in selection order at the root:
         * each pass moves the worst remaining block to the end, so the
         * array finishes best first — O(n log n), where the argmax was a
         * pass over every block for every block kept. */
        for (int i = n / 2 - 1; i >= 0; i--) qsa_sift(scores, order, i, n);
        for (int end = n - 1; end > 0; end--) {
            const int tmp = order[0]; order[0] = order[end]; order[end] = tmp;
            qsa_sift(scores, order, 0, end);
        }
        const int keep = n_complete < block_topk ? n_complete : block_topk;
        const int take = keep < n ? keep : n;
        for (int j = 0; j < take; j++)
            for (int t = 0; t < compress; t++)
                sel[nsel++] = order[j] * compress + t;
    }
    for (int t = 0; t < n_tail; t++)
        sel[nsel++] = n_complete * compress + t;
    return nsel;
}

void waste_qwen_qsa_attn(const float *q, int Hq, int D,
                         const float *k, const float *v, int Hkv, int T,
                         const int *sel, int n_sel, float scale,
                         float *out, float *scratch)
{
    waste_qwen_qsa_attn_heads(0, Hq, q, Hq, D, k, v, Hkv, T, sel, n_sel, scale,
                              out, scratch);
}

void waste_qwen_qsa_attn_heads(int h0, int h1, const float *q, int Hq, int D,
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
            for (int d = 0; d < D; d++) s += qh[d] * kh[d];
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
            for (int d = 0; d < D; d++) oh[d] += w * vh[d];
        }
    }
}
