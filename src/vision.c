/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/*
 * vision.c — K3's vision tower: pixels in, text-embedding space out.
 *
 * A 27-layer ViT, hidden 1024, 12 heads of 128, patch 14, with 2D rotary
 * position embedding and RMSNorm; then a 2x2 spatial merge and a two-layer
 * projector into the language model's 7168 dimensions. Diffed against
 * tools/vision_ref.py, which transcribes the reference implementation.
 *
 * Two details the reference gets right and a reimplementation easily gets
 * wrong. The encoder MLP uses the tanh approximation of GELU while the
 * projector uses the exact one — different functions, both called "gelu"
 * in the config. And the rotary table interleaves the *width* axis at even
 * indices with the height axis at odd ones; the reference's own docstring
 * says the opposite of what its code does, and the weights were trained
 * against the code.
 *
 * The tower is loaded only when a caller asks for it (waste_cfg.vision):
 * it is 438 MB, and on a machine where the expert cache decides throughput
 * that is not a rounding error.
 */

#include "model.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VT_PATCH   14
#define VT_PIXELS  (3 * VT_PATCH * VT_PATCH)      /* 588 */

int waste_vision_available(const waste_model *m)
{
    return waste_find(m, "vision_tower.patch_embed.proj.weight") != NULL;
}

static const waste_tensor *vt(waste_model *m, const char *fmt, int i)
{
    char name[160];
    if (i < 0) snprintf(name, sizeof name, "%s", fmt);
    else       snprintf(name, sizeof name, fmt, i);
    const waste_tensor *t = waste_find(m, name);
    if (!t) fprintf(stderr, "waste: vision tensor missing: %s\n", name);
    return t;
}

/* Bilinear resize of the stored [64][64][D] position grid onto [h][w].
 * align_corners=False, which is what F.interpolate defaults to and what
 * the tower was trained with. */
static void pos_resize(const float *src, int sh, int sw, int D,
                       float *dst, int h, int w)
{
    for (int y = 0; y < h; y++) {
        const float fy = ((float)y + 0.5f) * (float)sh / (float)h - 0.5f;
        int y0 = (int)floorf(fy);
        const float wy = fy - (float)y0;
        int y1 = y0 + 1;
        if (y0 < 0) y0 = 0;
        if (y1 < 0) y1 = 0;
        if (y0 > sh - 1) y0 = sh - 1;
        if (y1 > sh - 1) y1 = sh - 1;
        for (int x = 0; x < w; x++) {
            const float fx = ((float)x + 0.5f) * (float)sw / (float)w - 0.5f;
            int x0 = (int)floorf(fx);
            const float wx = fx - (float)x0;
            int x1 = x0 + 1;
            if (x0 < 0) x0 = 0;
            if (x1 < 0) x1 = 0;
            if (x0 > sw - 1) x0 = sw - 1;
            if (x1 > sw - 1) x1 = sw - 1;
            float *o = dst + ((size_t)y * w + x) * D;
            const float *a = src + ((size_t)y0 * sw + x0) * D;
            const float *b = src + ((size_t)y0 * sw + x1) * D;
            const float *c = src + ((size_t)y1 * sw + x0) * D;
            const float *e = src + ((size_t)y1 * sw + x1) * D;
            const float w00 = (1 - wy) * (1 - wx), w01 = (1 - wy) * wx;
            const float w10 = wy * (1 - wx), w11 = wy * wx;
            for (int d = 0; d < D; d++)
                o[d] = w00 * a[d] + w01 * b[d] + w10 * c[d] + w11 * e[d];
        }
    }
}

/* cos/sin for one position, interleaved: slot 2i follows x (width), slot
 * 2i+1 follows y (height), matching _precompute_freqs_cis. */
static void rope_pair(int x, int y, int hd, int j, float *cs, float *sn)
{
    const int i = j >> 1;                    /* which frequency */
    const float f = 1.0f / powf(10000.0f, (float)(4 * i) / (float)hd);
    const float ang = (j & 1) ? (float)y * f : (float)x * f;
    *cs = cosf(ang);
    *sn = sinf(ang);
}

static void apply_rope(float *v, int heads, int hd, int x, int y)
{
    const int half = hd / 2;
    for (int j = 0; j < half; j++) {
        float cs, sn;
        rope_pair(x, y, hd, j, &cs, &sn);
        for (int h = 0; h < heads; h++) {
            float *p = v + (size_t)h * hd + 2 * j;
            const float a = p[0], b = p[1];
            p[0] = a * cs - b * sn;
            p[1] = a * sn + b * cs;
        }
    }
}

static inline float gelu_tanh(float x)
{
    const float c = 0.7978845608028654f;      /* sqrt(2/pi) */
    return 0.5f * x * (1.0f + tanhf(c * (x + 0.044715f * x * x * x)));
}

static inline float gelu_erf(float x)
{
    return 0.5f * x * (1.0f + erff(x * 0.70710678118654752f));
}

static void softmax_row(float *a, int n)
{
    float mx = a[0];
    for (int i = 1; i < n; i++) if (a[i] > mx) mx = a[i];
    float s = 0;
    for (int i = 0; i < n; i++) { a[i] = expf(a[i] - mx); s += a[i]; }
    const float inv = 1.0f / s;
    for (int i = 0; i < n; i++) a[i] *= inv;
}

/* WASTE_VIS_STAGE=embed|blockN|encoder stops early and writes the
 * intermediate instead, so a divergence can be bisected against the
 * oracle's --stage. */
static int stage_is(const char *want)
{
    const char *e = getenv("WASTE_VIS_STAGE");
    return e && !strcmp(e, want);
}

int waste_vision_encode(waste_model *m, const float *pixels, int h, int w,
                        float *out)
{
    if (!waste_vision_available(m)) return -1;
    if (h <= 0 || w <= 0 || (h & 1) || (w & 1)) {
        fprintf(stderr, "waste: vision grid must be even in both axes\n");
        return -1;
    }
    const waste_vision_cfg *c = &m->vcfg;
    const int D = c->hidden, heads = c->heads, hd = c->qkv_hidden / heads;
    const int L = h * w, qkv = c->qkv_hidden;

    float *x    = (float *)malloc((size_t)L * D * sizeof(float));
    float *y    = (float *)malloc((size_t)L * D * sizeof(float));
    float *pos  = (float *)malloc((size_t)L * D * sizeof(float));
    float *qkvb = (float *)malloc((size_t)L * 3 * qkv * sizeof(float));
    float *ob   = (float *)malloc((size_t)L * qkv * sizeof(float));
    float *ff   = (float *)malloc((size_t)L * c->inter * sizeof(float));
    float *att  = (float *)malloc((size_t)L * sizeof(float));
    if (!x || !y || !pos || !qkvb || !ob || !ff || !att) {
        free(x); free(y); free(pos); free(qkvb); free(ob); free(ff); free(att);
        return -1;
    }

    /* --- patch embedding: the 14x14 conv is a matmul over 588 ----------- */
    const waste_tensor *pw = vt(m, "vision_tower.patch_embed.proj.weight", -1);
    if (!pw) goto fail;
    waste_matmul_t(m, x, pw, pixels, D, VT_PIXELS, L);

    /* The stored grid is quantized like everything else, so materialize it
     * once — 64*64*1024 floats, 16 MB, and only when an image arrives. */
    const waste_tensor *pe = vt(m, "vision_tower.patch_embed.pos_emb.weight", -1);
    if (!pe) goto fail;
    float *grid = (float *)malloc((size_t)c->pos_h * c->pos_w * D * sizeof(float));
    if (!grid) goto fail;
    if (pe->data) {
        memcpy(grid, pe->data, (size_t)c->pos_h * c->pos_w * D * sizeof(float));
    } else {
        for (long r = 0; r < (long)c->pos_h * c->pos_w; r++)
            waste_deq_row(pe, r, D, grid + (size_t)r * D);
    }
    if (h == c->pos_h && w == c->pos_w)
        memcpy(pos, grid, (size_t)L * D * sizeof(float));
    else
        pos_resize(grid, c->pos_h, c->pos_w, D, pos, h, w);
    free(grid);
    for (size_t i = 0; i < (size_t)L * D; i++) x[i] += pos[i];
    if (stage_is("embed")) {
        memcpy(out, x, (size_t)L * D * sizeof(float));
        goto done;
    }

    /* --- encoder -------------------------------------------------------- */
    const float scale = 1.0f / sqrtf((float)hd);
    for (int b = 0; b < c->layers; b++) {
        char nm[160];
        snprintf(nm, sizeof nm, "vision_tower.encoder.blocks.%d.norm0.weight", b);
        const waste_tensor *n0 = waste_find(m, nm);
        snprintf(nm, sizeof nm, "vision_tower.encoder.blocks.%d.norm1.weight", b);
        const waste_tensor *n1 = waste_find(m, nm);
        snprintf(nm, sizeof nm, "vision_tower.encoder.blocks.%d.wqkv.weight", b);
        const waste_tensor *wq = waste_find(m, nm);
        snprintf(nm, sizeof nm, "vision_tower.encoder.blocks.%d.wo.weight", b);
        const waste_tensor *wo = waste_find(m, nm);
        snprintf(nm, sizeof nm, "vision_tower.encoder.blocks.%d.mlp.fc0.weight", b);
        const waste_tensor *f0 = waste_find(m, nm);
        snprintf(nm, sizeof nm, "vision_tower.encoder.blocks.%d.mlp.fc1.weight", b);
        const waste_tensor *f1 = waste_find(m, nm);
        if (!n0 || !n1 || !wq || !wo || !f0 || !f1) goto fail;

        for (int i = 0; i < L; i++)
            waste_rmsnorm(y + (size_t)i * D, x + (size_t)i * D, n0->data, D, c->eps);
        waste_matmul_t(m, qkvb, wq, y, 3 * qkv, D, L);

        /* rotate q and k in place; the packed layout is [3][heads][hd] */
        for (int i = 0; i < L; i++) {
            float *base = qkvb + (size_t)i * 3 * qkv;
            apply_rope(base,       heads, hd, i % w, i / w);
            apply_rope(base + qkv, heads, hd, i % w, i / w);
        }

        for (int hh = 0; hh < heads; hh++) {
            for (int i = 0; i < L; i++) {
                const float *q = qkvb + (size_t)i * 3 * qkv + (size_t)hh * hd;
                for (int j = 0; j < L; j++) {
                    const float *k = qkvb + (size_t)j * 3 * qkv + qkv + (size_t)hh * hd;
                    float s = 0;
                    for (int d = 0; d < hd; d++) s += q[d] * k[d];
                    att[j] = s * scale;
                }
                softmax_row(att, L);
                float *o = ob + (size_t)i * qkv + (size_t)hh * hd;
                memset(o, 0, (size_t)hd * sizeof(float));
                for (int j = 0; j < L; j++) {
                    const float *v = qkvb + (size_t)j * 3 * qkv + 2 * qkv + (size_t)hh * hd;
                    const float a = att[j];
                    for (int d = 0; d < hd; d++) o[d] += a * v[d];
                }
            }
        }
        waste_matmul_t(m, y, wo, ob, D, qkv, L);
        for (size_t i = 0; i < (size_t)L * D; i++) x[i] += y[i];
        {
            char want[32];
            snprintf(want, sizeof want, "attn%d", b);
            if (stage_is(want)) {
                memcpy(out, x, (size_t)L * D * sizeof(float));
                goto done;
            }
        }

        for (int i = 0; i < L; i++)
            waste_rmsnorm(y + (size_t)i * D, x + (size_t)i * D, n1->data, D, c->eps);
        {   char want[32]; snprintf(want, sizeof want, "n1_%d", b);
            if (stage_is(want)) { memcpy(out, y, (size_t)L * D * sizeof(float)); goto done; } }

        waste_matmul_t(m, ff, f0, y, c->inter, D, L);
        {   char want[32]; snprintf(want, sizeof want, "fc0_%d", b);
            if (stage_is(want)) {          /* wide: first D columns per row */
                for (int i = 0; i < L; i++)
                    memcpy(out + (size_t)i * D, ff + (size_t)i * c->inter,
                           (size_t)D * sizeof(float));
                goto done; } }

        for (size_t i = 0; i < (size_t)L * c->inter; i++) ff[i] = gelu_tanh(ff[i]);
        {   char want[32]; snprintf(want, sizeof want, "act_%d", b);
            if (stage_is(want)) {
                for (int i = 0; i < L; i++)
                    memcpy(out + (size_t)i * D, ff + (size_t)i * c->inter,
                           (size_t)D * sizeof(float));
                goto done; } }

        waste_matmul_t(m, y, f1, ff, D, c->inter, L);
        {   char want[32]; snprintf(want, sizeof want, "fc1_%d", b);
            if (stage_is(want)) { memcpy(out, y, (size_t)L * D * sizeof(float)); goto done; } }
        for (size_t i = 0; i < (size_t)L * D; i++) x[i] += y[i];
        {
            char want[32];
            snprintf(want, sizeof want, "block%d", b);
            if (stage_is(want)) {
                memcpy(out, x, (size_t)L * D * sizeof(float));
                goto done;
            }
        }
    }

    const waste_tensor *fn = vt(m, "vision_tower.encoder.final_layernorm.weight", -1);
    if (!fn) goto fail;
    for (int i = 0; i < L; i++)
        waste_rmsnorm(y + (size_t)i * D, x + (size_t)i * D, fn->data, D, c->eps);

    if (stage_is("encoder")) {
        memcpy(out, y, (size_t)L * D * sizeof(float));
        goto done;
    }

    /* --- 2x2 merge: [h][w][D] -> [h/2 * w/2][4D], row-major within the tile */
    const int nh = h / 2, nw = w / 2, merged = 4 * D;
    float *mg = (float *)malloc((size_t)nh * nw * merged * sizeof(float));
    if (!mg) goto fail;
    for (int a = 0; a < nh; a++)
        for (int bb = 0; bb < nw; bb++) {
            float *dst = mg + ((size_t)a * nw + bb) * merged;
            for (int dy = 0; dy < 2; dy++)
                for (int dx = 0; dx < 2; dx++)
                    memcpy(dst + ((size_t)dy * 2 + dx) * D,
                           y + ((size_t)(2 * a + dy) * w + (2 * bb + dx)) * D,
                           (size_t)D * sizeof(float));
        }

    /* --- projector ------------------------------------------------------- */
    const waste_tensor *p0 = vt(m, "mm_projector.proj.0.weight", -1);
    const waste_tensor *p2 = vt(m, "mm_projector.proj.2.weight", -1);
    const waste_tensor *pn = vt(m, "mm_projector.post_norm.weight", -1);
    if (!p0 || !p2 || !pn) { free(mg); goto fail; }
    const int N = nh * nw;
    float *h0 = (float *)malloc((size_t)N * merged * sizeof(float));
    if (!h0) { free(mg); goto fail; }
    waste_matmul_t(m, h0, p0, mg, merged, merged, N);
    for (size_t i = 0; i < (size_t)N * merged; i++) h0[i] = gelu_erf(h0[i]);
    waste_matmul_t(m, out, p2, h0, c->text_hidden, merged, N);
    for (int i = 0; i < N; i++)
        waste_rmsnorm(out + (size_t)i * c->text_hidden,
                      out + (size_t)i * c->text_hidden,
                      pn->data, c->text_hidden, c->proj_eps);

    free(h0); free(mg);
done:
    free(x); free(y); free(pos); free(qkvb); free(ob); free(ff); free(att);
    return 0;

fail:
    free(x); free(y); free(pos); free(qkvb); free(ob); free(ff); free(att);
    return -1;
}
