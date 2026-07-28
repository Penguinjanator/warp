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
 * A caveat that matters more than the code: **K3 ships no preprocessor
 * config**. There is no preprocessor_config.json in the release and no
 * image mean/std anywhere in the modeling code or config, so the
 * normalization here is the CLIP convention every tower of this lineage
 * uses — and it is a choice, not a transcription. If image understanding
 * comes out subtly wrong while the tower itself matches its oracle to
 * 2.3e-06, this is the first thing to question. It is in vision.json so it
 * can be corrected without a rebuild.
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

float *waste_image_load(const char *path, int max_patches,
                        const float *mean, const float *std,
                        int *out_h, int *out_w)
{
    int sw = 0, sh = 0, ch = 0;
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
