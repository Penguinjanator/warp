/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/*
 * image.c — a file on disk to the patch tensor the vision tower wants.
 *
 * Decode (stb_image), resize to a whole number of 14-pixel patches with an
 * even count on both axes because the tower merges 2x2, normalize, and lay
 * the pixels out as [patches][3*14*14] in the channel-major order the patch
 * embedding's conv kernel expects.
 *
 * The mean and std are not chosen here: they come from vision.json, which
 * the converter fills from the release's preprocessor_config.json —
 * `media_proc_cfg` carries mean = std = 0.5, i.e. [-1, 1], and
 * kimi_k3_vision_processing.py applies exactly those. This file's own
 * fallback (see the caller in model.c) matches, and is only reached for a
 * container converted without the file present.
 *
 * That paragraph used to say the release shipped no preprocessor config
 * and that the normalization here was the CLIP convention, a guess. It
 * ships one; the values were corrected and this comment was not, which is
 * the worse of the two errors — it sent a reader debugging the tower to
 * question the one number that had just been verified against the
 * release. tests/run.sh checks it now.
 */

#include "model.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_HDR
#define STBI_NO_LINEAR
#define STBI_NO_PIC
#define STBI_NO_PNM
#define STBI_ASSERT(x) ((void)0)
#include "../third_party/stb_image.h"

#define PATCH 14

/* Bilinear sample of an 8-bit RGB image, matching the resize convention the
 * position grid already uses: half-pixel centres, no antialiasing. */
static float sample(const unsigned char *src, int sw, int sh, int ch,
                    float fx, float fy, int c)
{
    int x0 = (int)floorf(fx), y0 = (int)floorf(fy);
    const float wx = fx - (float)x0, wy = fy - (float)y0;
    int x1 = x0 + 1, y1 = y0 + 1;
    if (x0 < 0) x0 = 0; if (y0 < 0) y0 = 0;
    if (x1 < 0) x1 = 0; if (y1 < 0) y1 = 0;
    if (x0 > sw - 1) x0 = sw - 1; if (y0 > sh - 1) y0 = sh - 1;
    if (x1 > sw - 1) x1 = sw - 1; if (y1 > sh - 1) y1 = sh - 1;
    const float a = src[((size_t)y0 * sw + x0) * ch + c];
    const float b = src[((size_t)y0 * sw + x1) * ch + c];
    const float d = src[((size_t)y1 * sw + x0) * ch + c];
    const float e = src[((size_t)y1 * sw + x1) * ch + c];
    return (1 - wy) * ((1 - wx) * a + wx * b) + wy * ((1 - wx) * d + wx * e);
}

int waste_image_size(const char *path, int *w, int *h)
{
    int c = 0;
    if (!path || !w || !h) return -1;
    return stbi_info(path, w, h, &c) ? 0 : -1;
}

float *waste_image_load(const char *path, int max_patches,
                        const float *mean, const float *std,
                        int *out_h, int *out_w)
{
    int sw = 0, sh = 0, ch = 0;
    if (!stbi_info(path, &sw, &sh, &ch) || sw <= 0 || sh <= 0 ||
        (uint64_t)sw * (uint64_t)sh > WASTE_MAX_SOURCE_PIXELS) {
        fprintf(stderr, "waste: image dimensions are invalid or exceed the %u-pixel limit: %s\n",
                WASTE_MAX_SOURCE_PIXELS, path ? path : "(null)");
        return NULL;
    }
    unsigned char *img = stbi_load(path, &sw, &sh, &ch, 3);
    if (!img) {
        fprintf(stderr, "waste: cannot decode %s: %s\n", path, stbi_failure_reason());
        return NULL;
    }

    /* Choose a patch grid that keeps the aspect ratio, stays under the
     * budget, and is even on both axes for the 2x2 merge. */
    double scale = sqrt((double)max_patches / ((double)sw * sh / (PATCH * PATCH)));
    if (scale > 1.0) scale = 1.0;
    int gw = (int)((double)sw * scale / PATCH + 0.5);
    int gh = (int)((double)sh * scale / PATCH + 0.5);
    if (gw < 2) gw = 2;
    if (gh < 2) gh = 2;
    gw &= ~1;
    gh &= ~1;
    while (gw * gh > max_patches && (gw > 2 || gh > 2)) {
        if (gw >= gh && gw > 2) gw -= 2; else if (gh > 2) gh -= 2; else break;
    }

    const int W = gw * PATCH, H = gh * PATCH;
    float *px = (float *)malloc((size_t)gw * gh * 3 * PATCH * PATCH * sizeof(float));
    if (!px) { stbi_image_free(img); return NULL; }

    const float rx = (float)sw / (float)W, ry = (float)sh / (float)H;
    for (int py = 0; py < gh; py++) {
        for (int pxi = 0; pxi < gw; pxi++) {
            float *dst = px + ((size_t)py * gw + pxi) * 3 * PATCH * PATCH;
            for (int c = 0; c < 3; c++)
                for (int j = 0; j < PATCH; j++)
                    for (int i = 0; i < PATCH; i++) {
                        const float fx = ((float)(pxi * PATCH + i) + 0.5f) * rx - 0.5f;
                        const float fy = ((float)(py * PATCH + j) + 0.5f) * ry - 0.5f;
                        const float v = sample(img, sw, sh, 3, fx, fy, c) / 255.0f;
                        dst[((size_t)c * PATCH + j) * PATCH + i] =
                            (v - mean[c]) / std[c];
                    }
        }
    }
    stbi_image_free(img);
    *out_h = gh;
    *out_w = gw;
    return px;
}

/* ---- GLM-5.3-Flash's preprocessing --------------------------------------
 *
 * A different grid rule and a different patch order from K3's above, and
 * both are stated by the release rather than chosen here.
 *
 * **The grid.** K3 scales to a patch budget. GLM aligns to `patch * merge`
 * — 28 pixels, so that the 2x2 merge always has whole blocks — and then
 * fits the result into a budget expressed in *merged tokens*, with a
 * binary search over the content height when it does not fit. Transcribed
 * from `smart_resize` in image_processing_glm5_next.py, including the two
 * details that look like accidents and are not: the frame count rounds to
 * a multiple of the temporal factor before it multiplies the budget, and
 * the search moves `low` to `content_height + 1` rather than to the
 * aligned one.
 *
 * **The order.** K3 lays patches out in raster order. GLM's are
 * block-major over merge blocks — block row, block column, then the
 * merge x merge inside it — which is what lets the tower's downsample be a
 * reshape, and what `get_vision_position_ids` assumes when it builds the
 * rotary indices. Getting this wrong rotates every patch by someone else's
 * position and leaves the output finite and plausible.
 *
 * **The pixels inside a row** are (channel, temporal, y, x), and the
 * temporal axis is a *copy*: a still image is shown to the two-slot patch
 * embedding twice. That is `expand`, not new content.
 *
 * The resampling is antialiased bicubic, which is what the release's
 * processor uses. A bilinear sample matched torch's bilinear exactly and
 * the release's bicubic to 7.7% relative on the patch tensor — a different
 * image to the tower, not a rounding difference — so the kernel is
 * implemented rather than approximated.
 */

/* ---- antialiased bicubic resize ----------------------------------------
 *
 * What GLM's processor uses, and the one place this pipeline was knowingly
 * not the reference's: a bilinear sample of the source matched torch's
 * bilinear exactly and the release's bicubic to 7.7% relative on the patch
 * tensor, which is a different image to the tower.
 *
 * Separable, and antialiased in the sense PIL and torchvision mean:
 * downsampling widens the kernel's support by the scale factor instead of
 * point-sampling through it, which is what stops a shrink from aliasing.
 * `a` is -0.5, PIL's choice and the one torch ported.
 *
 * Two passes rather than one 2-D kernel: the weights of a separable filter
 * factor, and computing them once per output row and column instead of
 * once per output pixel is the difference between this being free and
 * being the slowest thing in the image path.
 */
static float cubic_w(float x)
{
    const float a = -0.5f;
    x = x < 0 ? -x : x;
    if (x < 1.0f) return ((a + 2.0f) * x - (a + 3.0f)) * x * x + 1.0f;
    if (x < 2.0f) return (((x - 5.0f) * x + 8.0f) * x - 4.0f) * a;
    return 0.0f;
}

/* One axis' weights: for each output index, the source range and the
 * normalized taps over it. Returns the widest range, which sizes the
 * caller's buffer. */
static int cubic_axis(int in, int out, int **first, int **count, float **wts)
{
    const double scale = (double)in / (double)out;
    const double support = (scale > 1.0 ? scale : 1.0) * 2.0;
    const int span = (int)ceil(support) * 2 + 2;
    int *f = (int *)malloc((size_t)out * sizeof(int));
    int *n = (int *)malloc((size_t)out * sizeof(int));
    float *w = (float *)malloc((size_t)out * span * sizeof(float));
    if (!f || !n || !w) { free(f); free(n); free(w); return 0; }
    for (int i = 0; i < out; i++) {
        const double centre = ((double)i + 0.5) * scale;
        int lo = (int)floor(centre - support + 0.5);
        int hi = (int)floor(centre + support + 0.5);
        if (lo < 0) lo = 0;
        if (hi > in) hi = in;
        float *row = w + (size_t)i * span;
        double sum = 0;
        int k = 0;
        for (int s = lo; s < hi && k < span; s++, k++) {
            const float t = cubic_w((float)(((double)s + 0.5 - centre) /
                                            (scale > 1.0 ? scale : 1.0)));
            row[k] = t;
            sum += t;
        }
        /* A degenerate range would divide by zero; it cannot happen for
         * in,out >= 1, and saying so costs one branch. */
        if (sum == 0.0) { row[0] = 1.0f; k = k ? k : 1; sum = 1.0; }
        for (int j = 0; j < k; j++) row[j] = (float)(row[j] / sum);
        f[i] = lo;
        n[i] = k;
    }
    *first = f; *count = n; *wts = w;
    return span;
}

/* src is 8-bit RGB, interleaved. dst is float, planar [3][H][W], still in
 * 0..255 — the caller rescales and normalizes. */
static int resize_bicubic(const unsigned char *src, int sw, int sh,
                          float *dst, int dw, int dh)
{
    int *xf = NULL, *xn = NULL, *yf = NULL, *yn = NULL;
    float *xw = NULL, *yw = NULL, *mid = NULL;
    int rc = -1;
    const int xspan = cubic_axis(sw, dw, &xf, &xn, &xw);
    const int yspan = cubic_axis(sh, dh, &yf, &yn, &yw);
    if (!xspan || !yspan) goto out;
    mid = (float *)malloc((size_t)3 * sh * dw * sizeof(float));
    if (!mid) goto out;

    for (int c = 0; c < 3; c++)
        for (int y = 0; y < sh; y++)
            for (int x = 0; x < dw; x++) {
                const float *w = xw + (size_t)x * xspan;
                float acc = 0;
                for (int k = 0; k < xn[x]; k++)
                    acc += w[k] * (float)src[((size_t)y * sw + xf[x] + k) * 3 + c];
                mid[((size_t)c * sh + y) * dw + x] = acc;
            }
    for (int c = 0; c < 3; c++)
        for (int y = 0; y < dh; y++) {
            const float *w = yw + (size_t)y * yspan;
            for (int x = 0; x < dw; x++) {
                float acc = 0;
                for (int k = 0; k < yn[y]; k++)
                    acc += w[k] * mid[((size_t)c * sh + yf[y] + k) * dw + x];
                dst[((size_t)c * dh + y) * dw + x] = acc;
            }
        }
    rc = 0;
out:
    free(xf); free(xn); free(xw); free(yf); free(yn); free(yw); free(mid);
    return rc;
}

static int glm_align(int v, int f) { return (v + f - 1) / f * f; }

/* smart_resize for a single frame. Returns the aligned pixel height and
 * width; the caller divides by `patch` for the grid. */
static void glm_smart_resize(int sh, int sw, int patch, int merge,
                             int temporal, int min_tok, int max_tok,
                             int *out_h, int *out_w)
{
    const int factor = patch * merge;
    const long long ppt = (long long)temporal * factor * factor;
    const long long min_px = (long long)min_tok * ppt;
    const long long max_px = (long long)max_tok * ppt;
    /* round(num_frames / temporal) * temporal with num_frames = 1 is 0 for
     * every temporal above 2, and 0 for 2 as well under round-half-even;
     * the max() is what makes it one aligned frame pair either way. */
    const int frames = temporal;

    int ah = glm_align(sh, factor), aw = glm_align(sw, factor);
    long long budget = (long long)frames * ah * aw;

    if (budget < min_px) {
        const double scale = sqrt((double)min_px / ((double)sh * sw));
        ah = glm_align((int)ceil(sh * scale) < 1 ? 1 : (int)ceil(sh * scale), factor);
        aw = glm_align((int)ceil(sw * scale) < 1 ? 1 : (int)ceil(sw * scale), factor);
        budget = (long long)frames * ah * aw;
    }
    if (budget > max_px) {
        int low = 1, high = sh, bh = factor, bw = factor;
        while (low <= high) {
            const int ch = (low + high) / 2;
            int cw = (int)floor((double)sw * ch / (double)sh);
            if (cw < 1) cw = 1;
            const int cah = glm_align(ch, factor), caw = glm_align(cw, factor);
            if ((long long)frames * cah * caw <= max_px) {
                bh = cah; bw = caw; low = ch + 1;
            } else {
                high = ch - 1;
            }
        }
        ah = bh; aw = bw;
    }
    *out_h = ah;
    *out_w = aw;
}

float *waste_image_load_glm(const char *path, const waste_vision_cfg *v,
                            int *out_h, int *out_w)
{
    int sw = 0, sh = 0, ch = 0;
    if (!stbi_info(path, &sw, &sh, &ch) || sw <= 0 || sh <= 0 ||
        (uint64_t)sw * (uint64_t)sh > WASTE_MAX_SOURCE_PIXELS) {
        fprintf(stderr, "waste: image dimensions are invalid or exceed the %u-pixel limit: %s\n",
                WASTE_MAX_SOURCE_PIXELS, path ? path : "(null)");
        return NULL;
    }
    unsigned char *img = stbi_load(path, &sw, &sh, &ch, 3);
    if (!img) {
        fprintf(stderr, "waste: cannot decode %s: %s\n", path, stbi_failure_reason());
        return NULL;
    }

    const int patch = v->patch, mg = v->merge, T = v->temporal;
    int min_tok = v->min_tokens > 0 ? v->min_tokens : 1;
    int max_tok = v->max_patches > 0 ? v->max_patches : 1024;
    if (min_tok > max_tok) min_tok = max_tok;
    int H = 0, W = 0;
    glm_smart_resize(sh, sw, patch, mg, T, min_tok, max_tok, &H, &W);
    const int gh = H / patch, gw = W / patch;
    const int npix = 3 * T * patch * patch;

    float *px = (float *)malloc((size_t)gh * gw * npix * sizeof(float));
    float *rs = (float *)malloc((size_t)3 * H * W * sizeof(float));
    if (!px || !rs || resize_bicubic(img, sw, sh, rs, W, H)) {
        stbi_image_free(img); free(px); free(rs);
        return NULL;
    }
    stbi_image_free(img);

    size_t row = 0;
    for (int by = 0; by < gh / mg; by++)
        for (int bx = 0; bx < gw / mg; bx++)
            for (int dy = 0; dy < mg; dy++)
                for (int dx = 0; dx < mg; dx++, row++) {
                    const int py = by * mg + dy, pxi = bx * mg + dx;
                    float *dst = px + row * npix;
                    for (int c = 0; c < 3; c++)
                        for (int j = 0; j < patch; j++)
                            for (int i = 0; i < patch; i++) {
                                const size_t si = ((size_t)c * H + py * patch + j)
                                                  * W + pxi * patch + i;
                                const float val =
                                    (rs[si] / 255.0f - v->mean[c]) / v->std[c];
                                /* (channel, temporal, y, x), the temporal
                                 * axis a copy of the same patch */
                                for (int t = 0; t < T; t++)
                                    dst[(((size_t)c * T + t) * patch + j) * patch + i] = val;
                            }
                }
    free(rs);
    *out_h = gh;
    *out_w = gw;
    return px;
}
