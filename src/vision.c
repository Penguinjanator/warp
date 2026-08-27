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
 * it is 434 MB, and on a machine where the expert cache decides throughput
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

/* ---- GLM-5.3-Flash's tower ---------------------------------------------
 *
 * A second tower rather than a branch inside the one above, because the two
 * agree on the block *shape* and on nothing inside it. K3 has a learned
 * 64x64 position grid, GELU, and no biases; GLM has 2D rope only, per-head
 * RMSNorms on q and k, biases on every projection, a clamped SwiGLU, a
 * patch that spans two temporal slots, and a gated merger where K3 has a
 * two-layer projector.
 *
 * Four things are easy to get wrong here, and each is why this was written
 * against tools/glm_vision_ref.py rather than against the config:
 *
 *   - **Patch order is block-major.** Rows arrive so that each consecutive
 *     merge*merge of them is one spatial block, which is what lets the
 *     downsample be a reshape. The rotary indices are built the same way;
 *     a row order that disagrees rotates every patch by someone else's
 *     position, and the output stays finite and plausible.
 *   - **The rotation is rotate_half over a doubled table**, not the
 *     interleaved pairing K3's tower uses.
 *   - **q and k are RMSNormed per head**, over head_dim, before rotating.
 *   - **The SwiGLU is clamped** in the encoder MLP and again in the merger,
 *     and the merger's first activation is the exact GELU.
 */

/* gate <- silu(clamp(gate)) * clamp(up), the activation both the encoder
 * MLP and the merger use. The same clamp the language model's experts get,
 * and the same reason: at limit 10 it fires on real activations. */
static void vt_swiglu(float *gate, const float *up, size_t n, float limit)
{
    for (size_t i = 0; i < n; i++) {
        float g = gate[i], u = up[i];
        if (limit > 0.0f) {
            if (g > limit) g = limit;
            u = u > limit ? limit : (u < -limit ? -limit : u);
        }
        gate[i] = (g / (1.0f + expf(-g))) * u;
    }
}

/* One row of a bias vector that may have been quantized like everything
 * else. Small enough that materializing it per call costs nothing. */
static const float *vt_bias(waste_model *m, const char *name, float *tmp, int n)
{
    const waste_tensor *t = waste_find(m, name);
    if (!t) return NULL;
    if (t->data) return t->data;
    waste_deq_row(t, 0, n, tmp);
    return tmp;
}

/* cos/sin for one patch's head_dim-wide rotary table.
 *
 * dim = head_dim/2 frequencies, each axis contributing dim/2 of them:
 * inv_freq[j] = theta^(-2j/dim), the row index scaling the first half and
 * the column index the second, then the whole thing doubled so that
 * rotate_half pairs component i with component i + head_dim/2. */
static void glm_rope_row(int row, int col, int hd, float *cs, float *sn)
{
    const int dim = hd / 2, nf = dim / 2;
    for (int j = 0; j < nf; j++) {
        const float inv = powf(10000.0f, -(float)(2 * j) / (float)dim);
        const float a = (float)row * inv, b = (float)col * inv;
        cs[j] = cosf(a);           sn[j] = sinf(a);
        cs[nf + j] = cosf(b);      sn[nf + j] = sinf(b);
    }
    for (int j = 0; j < dim; j++) { cs[dim + j] = cs[j]; sn[dim + j] = sn[j]; }
}

/* v[0..hd) <- v*cos + rotate_half(v)*sin */
static void glm_rope_apply(float *v, int hd, const float *cs, const float *sn)
{
    float t[256];
    const int h = hd / 2;
    for (int i = 0; i < hd; i++) t[i] = v[i];
    for (int i = 0; i < h; i++) {
        v[i]     = t[i]     * cs[i]     + (-t[h + i]) * sn[i];
        v[h + i] = t[h + i] * cs[h + i] + ( t[i]    ) * sn[h + i];
    }
}

static void rmsnorm_rows(float *dst, const float *src, const float *w,
                         int rows, int n, float eps)
{
    for (int r = 0; r < rows; r++) {
        const float *x = src + (size_t)r * n;
        float ss = 0;
        for (int i = 0; i < n; i++) ss += x[i] * x[i];
        const float inv = 1.0f / sqrtf(ss / (float)n + eps);
        float *o = dst + (size_t)r * n;
        for (int i = 0; i < n; i++) o[i] = x[i] * inv * w[i];
    }
}

int waste_vision_encode_glm(waste_model *m, const float *pixels, int h, int w,
                            float *out)
{
    const waste_vision_cfg *c = &m->vcfg;
    const int D = c->hidden, heads = c->heads, hd = D / heads;
    const int L = h * w, mg = c->merge, OD = c->out_hidden;
    const int npix = 3 * c->temporal * c->patch * c->patch;
    if (h <= 0 || w <= 0 || h % mg || w % mg) {
        fprintf(stderr, "waste: vision grid must be a multiple of %d\n", mg);
        return -1;
    }
    if (hd > 256) { fprintf(stderr, "waste: vision head_dim %d too wide\n", hd); return -1; }

    float *x   = (float *)malloc((size_t)L * D * sizeof(float));
    float *y   = (float *)malloc((size_t)L * D * sizeof(float));
    float *qkv = (float *)malloc((size_t)L * 3 * D * sizeof(float));
    float *ob  = (float *)malloc((size_t)L * D * sizeof(float));
    float *ff  = (float *)malloc((size_t)2 * L * c->inter * sizeof(float));
    float *att = (float *)malloc((size_t)L * sizeof(float));
    float *cs  = (float *)malloc((size_t)L * hd * sizeof(float));
    float *sn  = (float *)malloc((size_t)L * hd * sizeof(float));
    float *bias = (float *)malloc((size_t)(c->inter > OD ? c->inter : OD) * sizeof(float));
    if (!x || !y || !qkv || !ob || !ff || !att || !cs || !sn || !bias) goto oom;

    /* --- patch embedding: the Conv3d is a matmul over 3*T*14*14 --------- */
    {
        const waste_tensor *pw = waste_find(m, "vision_tower.patch_embed.proj.weight");
        if (!pw) goto fail;
        waste_matmul_t(m, x, pw, pixels, D, npix, L);
        const float *pb = vt_bias(m, "vision_tower.patch_embed.proj.bias", bias, D);
        if (pb) for (int r = 0; r < L; r++)
            for (int i = 0; i < D; i++) x[(size_t)r * D + i] += pb[i];
    }
    if (getenv("WASTE_VIS_STAGE") &&
        strcmp(getenv("WASTE_VIS_STAGE"), "embed") == 0) {
        memcpy(out, x, (size_t)L * D * sizeof(float));
        goto done;
    }

    /* --- the rotary tables, block-major over merge blocks --------------- */
    {
        int r = 0;
        for (int by = 0; by < h / mg; by++)
            for (int bx = 0; bx < w / mg; bx++)
                for (int dy = 0; dy < mg; dy++)
                    for (int dx = 0; dx < mg; dx++, r++)
                        glm_rope_row(by * mg + dy, bx * mg + dx, hd,
                                     cs + (size_t)r * hd, sn + (size_t)r * hd);
    }

    /* --- encoder -------------------------------------------------------- */
    {
    const float scale = 1.0f / sqrtf((float)hd);
    for (int b = 0; b < c->layers; b++) {
        char nm[160];
        snprintf(nm, sizeof nm, "vision_tower.blocks.%d.norm1.weight", b);
        const waste_tensor *n1 = waste_find(m, nm);
        if (!n1 || !n1->data) goto fail;
        rmsnorm_rows(y, x, n1->data, L, D, c->eps);

        snprintf(nm, sizeof nm, "vision_tower.blocks.%d.attn.qkv.weight", b);
        const waste_tensor *wq = waste_find(m, nm);
        if (!wq) goto fail;
        waste_matmul_t(m, qkv, wq, y, 3 * D, D, L);
        snprintf(nm, sizeof nm, "vision_tower.blocks.%d.attn.qkv.bias", b);
        {
            float qb[3 * 4096];
            const float *pb = vt_bias(m, nm, qb, 3 * D);
            if (pb) for (int r = 0; r < L; r++)
                for (int i = 0; i < 3 * D; i++) qkv[(size_t)r * 3 * D + i] += pb[i];
        }

        /* q and k are normalized per head and then rotated; v is neither. */
        snprintf(nm, sizeof nm, "vision_tower.blocks.%d.attn.q_norm.weight", b);
        const waste_tensor *qn = waste_find(m, nm);
        snprintf(nm, sizeof nm, "vision_tower.blocks.%d.attn.k_norm.weight", b);
        const waste_tensor *kn = waste_find(m, nm);
        if (!qn || !kn || !qn->data || !kn->data) goto fail;
        for (int r = 0; r < L; r++) {
            float *row = qkv + (size_t)r * 3 * D;
            for (int hh = 0; hh < heads; hh++) {
                float *q = row + (size_t)hh * hd;
                float *k = row + D + (size_t)hh * hd;
                rmsnorm_rows(q, q, qn->data, 1, hd, c->eps);
                rmsnorm_rows(k, k, kn->data, 1, hd, c->eps);
                glm_rope_apply(q, hd, cs + (size_t)r * hd, sn + (size_t)r * hd);
                glm_rope_apply(k, hd, cs + (size_t)r * hd, sn + (size_t)r * hd);
            }
        }

        /* full attention over the image, one head at a time */
        for (int hh = 0; hh < heads; hh++) {
            for (int i = 0; i < L; i++) {
                const float *q = qkv + (size_t)i * 3 * D + (size_t)hh * hd;
                for (int j = 0; j < L; j++) {
                    const float *k = qkv + (size_t)j * 3 * D + D + (size_t)hh * hd;
                    float acc = 0;
                    for (int t = 0; t < hd; t++) acc += q[t] * k[t];
                    att[j] = acc * scale;
                }
                softmax_row(att, L);
                float *o = ob + (size_t)i * D + (size_t)hh * hd;
                for (int t = 0; t < hd; t++) o[t] = 0;
                for (int j = 0; j < L; j++) {
                    const float a = att[j];
                    const float *v = qkv + (size_t)j * 3 * D + 2 * D + (size_t)hh * hd;
                    for (int t = 0; t < hd; t++) o[t] += a * v[t];
                }
            }
        }

        snprintf(nm, sizeof nm, "vision_tower.blocks.%d.attn.proj.weight", b);
        const waste_tensor *wo = waste_find(m, nm);
        if (!wo) goto fail;
        waste_matmul_t(m, y, wo, ob, D, D, L);
        snprintf(nm, sizeof nm, "vision_tower.blocks.%d.attn.proj.bias", b);
        {
            const float *pb = vt_bias(m, nm, bias, D);
            for (int r = 0; r < L; r++)
                for (int i = 0; i < D; i++)
                    x[(size_t)r * D + i] += y[(size_t)r * D + i] + (pb ? pb[i] : 0.0f);
        }

        snprintf(nm, sizeof nm, "vision_tower.blocks.%d.norm2.weight", b);
        const waste_tensor *n2 = waste_find(m, nm);
        if (!n2 || !n2->data) goto fail;
        rmsnorm_rows(y, x, n2->data, L, D, c->eps);

        float *gate = ff, *up = ff + (size_t)L * c->inter;
        snprintf(nm, sizeof nm, "vision_tower.blocks.%d.mlp.gate_proj.weight", b);
        const waste_tensor *wg = waste_find(m, nm);
        snprintf(nm, sizeof nm, "vision_tower.blocks.%d.mlp.up_proj.weight", b);
        const waste_tensor *wu = waste_find(m, nm);
        if (!wg || !wu) goto fail;
        waste_matmul_t(m, gate, wg, y, c->inter, D, L);
        waste_matmul_t(m, up, wu, y, c->inter, D, L);
        snprintf(nm, sizeof nm, "vision_tower.blocks.%d.mlp.gate_proj.bias", b);
        {
            const float *gb = vt_bias(m, nm, bias, c->inter);
            if (gb) for (int r = 0; r < L; r++)
                for (int i = 0; i < c->inter; i++) gate[(size_t)r * c->inter + i] += gb[i];
        }
        snprintf(nm, sizeof nm, "vision_tower.blocks.%d.mlp.up_proj.bias", b);
        {
            const float *ub = vt_bias(m, nm, bias, c->inter);
            if (ub) for (int r = 0; r < L; r++)
                for (int i = 0; i < c->inter; i++) up[(size_t)r * c->inter + i] += ub[i];
        }
        vt_swiglu(gate, up, (size_t)L * c->inter, c->swiglu_limit);

        snprintf(nm, sizeof nm, "vision_tower.blocks.%d.mlp.down_proj.weight", b);
        const waste_tensor *wd = waste_find(m, nm);
        if (!wd) goto fail;
        waste_matmul_t(m, y, wd, gate, D, c->inter, L);
        snprintf(nm, sizeof nm, "vision_tower.blocks.%d.mlp.down_proj.bias", b);
        {
            const float *db = vt_bias(m, nm, bias, D);
            for (int r = 0; r < L; r++)
                for (int i = 0; i < D; i++)
                    x[(size_t)r * D + i] += y[(size_t)r * D + i] + (db ? db[i] : 0.0f);
        }
        {
            const char *st = getenv("WASTE_VIS_STAGE");
            char want[32];
            snprintf(want, sizeof want, "block%d", b);
            if (st && strcmp(st, want) == 0) {
                memcpy(out, x, (size_t)L * D * sizeof(float));
                goto done;
            }
        }
    }
    }

    {
        const waste_tensor *pn = waste_find(m, "vision_tower.post_layernorm.weight");
        if (!pn || !pn->data) goto fail;
        rmsnorm_rows(x, x, pn->data, L, D, c->eps);
    }
    if (getenv("WASTE_VIS_STAGE") &&
        strcmp(getenv("WASTE_VIS_STAGE"), "post") == 0) {
        memcpy(out, x, (size_t)L * D * sizeof(float));
        goto done;
    }

    /* --- downsample: the Conv2d over a merge block is a matmul over
     * (channel, dy, dx) — the weight's own axis order, and not the
     * (dy, dx, channel) a reader of the reshape above would assume. */
    {
        const int nb = L / (mg * mg), win = D * mg * mg;
        float *blk = (float *)malloc((size_t)nb * win * sizeof(float));
        if (!blk) goto oom;
        for (int b2 = 0; b2 < nb; b2++)
            for (int ch = 0; ch < D; ch++)
                for (int dy = 0; dy < mg; dy++)
                    for (int dx = 0; dx < mg; dx++)
                        blk[(size_t)b2 * win + ((size_t)ch * mg + dy) * mg + dx] =
                            x[((size_t)b2 * mg * mg + (size_t)dy * mg + dx) * D + ch];
        const waste_tensor *dw = waste_find(m, "vision_tower.downsample.weight");
        if (!dw) { free(blk); goto fail; }
        waste_matmul_t(m, y, dw, blk, OD, win, nb);
        free(blk);
        const float *db = vt_bias(m, "vision_tower.downsample.bias", bias, OD);
        if (db) for (int r = 0; r < nb; r++)
            for (int i = 0; i < OD; i++) y[(size_t)r * OD + i] += db[i];
        if (getenv("WASTE_VIS_STAGE") &&
            strcmp(getenv("WASTE_VIS_STAGE"), "downsample") == 0) {
            memcpy(out, y, (size_t)nb * OD * sizeof(float));
            goto done;
        }

        /* --- merger: proj -> LayerNorm -> GELU -> clamped SwiGLU -------- */
        const waste_tensor *pj = waste_find(m, "vision_tower.merger.proj.weight");
        if (!pj) goto fail;
        waste_matmul_t(m, x, pj, y, OD, OD, nb);
        {
            const waste_tensor *lw = waste_find(m, "vision_tower.merger.post_projection_norm.weight");
            float lb[4096];
            const float *lbias = vt_bias(m, "vision_tower.merger.post_projection_norm.bias", lb, OD);
            if (!lw || !lw->data) goto fail;
            for (int r = 0; r < nb; r++) {
                float *row = x + (size_t)r * OD;
                float mean = 0;
                for (int i = 0; i < OD; i++) mean += row[i];
                mean /= (float)OD;
                float var = 0;
                for (int i = 0; i < OD; i++) { const float d = row[i] - mean; var += d * d; }
                const float inv = 1.0f / sqrtf(var / (float)OD + c->proj_eps);
                for (int i = 0; i < OD; i++)
                    row[i] = gelu_erf((row[i] - mean) * inv * lw->data[i]
                                            + (lbias ? lbias[i] : 0.0f));
            }
        }
        float *gate = ff, *up = ff + (size_t)nb * c->proj_inter;
        const waste_tensor *mg1 = waste_find(m, "vision_tower.merger.gate_proj.weight");
        const waste_tensor *mu1 = waste_find(m, "vision_tower.merger.up_proj.weight");
        const waste_tensor *md1 = waste_find(m, "vision_tower.merger.down_proj.weight");
        if (!mg1 || !mu1 || !md1) goto fail;
        waste_matmul_t(m, gate, mg1, x, c->proj_inter, OD, nb);
        waste_matmul_t(m, up, mu1, x, c->proj_inter, OD, nb);
        vt_swiglu(gate, up, (size_t)nb * c->proj_inter, c->swiglu_limit);
        waste_matmul_t(m, out, md1, gate, OD, c->proj_inter, nb);
    }

done:
    free(x); free(y); free(qkv); free(ob); free(ff); free(att);
    free(cs); free(sn); free(bias);
    return 0;
oom:
fail:
    free(x); free(y); free(qkv); free(ob); free(ff); free(att);
    free(cs); free(sn); free(bias);
    return -1;
}
