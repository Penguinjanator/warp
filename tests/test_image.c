/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/* The image loader, checked against arithmetic rather than against a
 * reference implementation: for a solid colour the answer is known in
 * closed form, so a wrong resize, a wrong channel order or a wrong
 * normalization all show up as a number that isn't (v - mean) / std.
 *
 * Needs no model and no container — the pixels come from a TGA this test
 * writes itself, which is the one image format simple enough to emit in a
 * dozen lines and which stb_image reads. */
#include "../src/model.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PATCH 14

static const float MEAN[3] = { 0.48145466f, 0.4578275f, 0.40821073f };
static const float STD[3]  = { 0.26862954f, 0.26130258f, 0.27577711f };

static int fails = 0;

static void ok(int cond, const char *what)
{
    if (!cond) { printf("FAIL %s\n", what); fails++; }
}

static void close_to(float got, float want, float tol, const char *what)
{
    if (!(fabsf(got - want) <= tol)) {
        printf("FAIL %s: %+.5f vs %+.5f\n", what, (double)got, (double)want);
        fails++;
    }
}

/* Uncompressed 24-bit TGA. Bottom-up by default, which is why the header's
 * origin bit is set: we want row 0 at the top, like every other format. */
static int write_tga(const char *path, int w, int h,
                     unsigned char r, unsigned char g, unsigned char b)
{
    unsigned char hdr[18];
    memset(hdr, 0, sizeof hdr);
    hdr[2] = 2;                                  /* uncompressed true-colour */
    hdr[12] = (unsigned char)(w & 0xff); hdr[13] = (unsigned char)(w >> 8);
    hdr[14] = (unsigned char)(h & 0xff); hdr[15] = (unsigned char)(h >> 8);
    hdr[16] = 24;
    hdr[17] = 0x20;                              /* origin top-left */
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    fwrite(hdr, 1, sizeof hdr, f);
    const unsigned char px[3] = { b, g, r };     /* TGA stores BGR */
    for (long i = 0; i < (long)w * h; i++) fwrite(px, 1, 3, f);
    fclose(f);
    return 0;
}

int main(int argc, char **argv)
{
    const char *dir = (argc > 1) ? argv[1] : ".";
    char path[512];
    snprintf(path, sizeof path, "%s/waste_test_img.tga", dir);

    /* --- a solid colour: every pixel is predictable ---------------------- */
    if (write_tga(path, 56, 28, 255, 0, 0)) { printf("FAIL cannot write %s\n", path); return 1; }

    int gh = 0, gw = 0;
    float *px = waste_image_load(path, 1024, MEAN, STD, &gh, &gw);
    ok(px != NULL, "decode");
    if (!px) return 1;

    /* 56x28 at 14 pixels per patch is 4 wide by 2 high, and both are already
     * even so nothing is rounded away. */
    ok(gw == 4 && gh == 2, "grid 4x2");
    ok(gw % 2 == 0 && gh % 2 == 0, "grid even on both axes");

    /* Channel-major inside a patch: R plane, then G, then B. Sampling the
     * centre avoids the edge clamp in the bilinear filter. */
    const int P = PATCH, mid = (7 * P) + 7;
    for (int p = 0; p < gw * gh; p++) {
        const float *q = px + (size_t)p * 3 * P * P;
        close_to(q[0 * P * P + mid], (1.0f - MEAN[0]) / STD[0], 1e-4f, "R");
        close_to(q[1 * P * P + mid], (0.0f - MEAN[1]) / STD[1], 1e-4f, "G");
        close_to(q[2 * P * P + mid], (0.0f - MEAN[2]) / STD[2], 1e-4f, "B");
    }
    free(px);

    /* --- the patch budget is a ceiling, not a suggestion ----------------- */
    if (write_tga(path, 700, 700, 128, 128, 128)) { printf("FAIL write\n"); return 1; }
    px = waste_image_load(path, 64, MEAN, STD, &gh, &gw);
    ok(px != NULL, "decode large");
    if (px) {
        ok(gw * gh <= 64, "patch budget respected");
        ok(gw % 2 == 0 && gh % 2 == 0, "large grid even");
        ok(gw == gh, "square image gives a square grid");
        free(px);
    }

    /* --- an image smaller than one patch still yields a legal grid ------- */
    if (write_tga(path, 3, 5, 0, 255, 0)) { printf("FAIL write\n"); return 1; }
    px = waste_image_load(path, 1024, MEAN, STD, &gh, &gw);
    ok(px != NULL, "decode tiny");
    if (px) {
        ok(gw >= 2 && gh >= 2, "tiny image padded up to the 2x2 merge");
        close_to(px[1 * P * P + mid], (1.0f - MEAN[1]) / STD[1], 1e-4f, "tiny G");
        free(px);
    }

    /* --- a file that is not an image is an error, not a crash ------------ */
    snprintf(path, sizeof path, "%s/waste_test_img.tga", dir);
    FILE *f = fopen(path, "wb");
    if (f) { fputs("this is not an image", f); fclose(f); }
    ok(waste_image_load(path, 1024, MEAN, STD, &gh, &gw) == NULL, "garbage rejected");
    ok(waste_image_load("/nonexistent/x.png", 1024, MEAN, STD, &gh, &gw) == NULL,
       "missing file rejected");
    remove(path);

    printf(fails ? "IMAGE FAILED (%d)\n" : "IMAGE OK\n", fails);
    return fails ? 1 : 0;
}
