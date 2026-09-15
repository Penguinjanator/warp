/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/* test_vision_ds41.c — DeepSeek-V4.1's tower and its preprocessing, for the
 * diff against tools/ds41_vision_ref.py.
 *
 * Two modes, because the two halves fail differently. `tower` runs the
 * encoder on a patch tensor the caller supplies, so the comparison is
 * against the reference on identical input and nothing about image
 * decoding is in it. `pixels` runs the preprocessing on a real file and
 * dumps what it produced, so the patch order and the normalization can be
 * checked against the reference's reshape without the tower's arithmetic
 * on top.
 *
 *   test_vision_ds41 tower  CONTAINER H W pixels.bin [out.bin]
 *   test_vision_ds41 pixels CONTAINER image.png      [out.bin]
 *
 * WASTE_VIS_STAGE stops the tower early — embed, blockN, post —
 * so a divergence can be bisected to the stage that introduces it rather
 * than read off the merged output.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/model.h"

static int open_model(waste_model *m, const char *dir)
{
    waste_load_opts o;
    memset(&o, 0, sizeof o);
    o.cache_bytes = 0;            /* the tower is trunk; no experts needed */
    o.want_vision = 1;
    o.direct_io = 1;
    const int rc = waste_model_load(m, dir, 256, &o);
    if (rc) fprintf(stderr, "load failed rc=%d\n", rc);
    return rc;
}

static void stats(const char *what, const float *v, size_t n, int rows, int dim)
{
    double mean = 0, ss = 0, mx = 0;
    for (size_t i = 0; i < n; i++) {
        mean += v[i];
        ss += (double)v[i] * v[i];
        const double a = v[i] < 0 ? -v[i] : v[i];
        if (a > mx) mx = a;
    }
    mean /= (double)n;
    double var = ss / (double)n - mean * mean;
    printf("%s (%d, %d)  mean %.6f  std %.6f  absmax %.6f\n",
           what, rows, dim, mean, var > 0 ? sqrt(var) : 0.0, mx);
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr,
                "usage: %s tower  CONTAINER H W pixels.bin [out.bin]\n"
                "       %s pixels CONTAINER image [out.bin]\n"
                "       %s plan   CONTAINER W H\n",
                argv[0], argv[0], argv[0]);
        return 2;
    }
    const char *mode = argv[1];
    waste_model m;
    if (open_model(&m, argv[2])) return 1;
    const waste_vision_cfg *c = &m.vcfg;
    if (c->tower != WASTE_TOWER_DS41) {
        fprintf(stderr, "this container's tower is not DeepSeek-V4.1's\n");
        waste_model_free(&m);
        return 1;
    }
    const int npix = 3 * c->patch * c->patch;
    int rc = 1;

    /* The geometry on its own, with no image and no decoding: the pixel
     * box and the token grid an image of this size resolves to. It is a
     * pure function and the reference has the same one, so the two can be
     * asked the same question without a file in between. */
    if (!strcmp(mode, "plan")) {
        if (argc < 5) { fprintf(stderr, "plan: need W H\n"); goto out; }
        int bh = 0, bw = 0, nh = 0, nw = 0;
        waste_image_plan_ds41(atoi(argv[3]), atoi(argv[4]), c, &bh, &bw,
                              &nh, &nw);
        printf("box %dx%d  patches %dx%d  tokens %dx%d  span %d\n",
               bw, bh, bw / c->patch, bh / c->patch, nw, nh,
               nh * (nw + 1) + 2);
        rc = 0;
        goto out;
    }

    if (!strcmp(mode, "pixels")) {
        if (argc < 4) { fprintf(stderr, "pixels: need an image\n"); goto out; }
        int gh = 0, gw = 0;
        float *px = waste_image_load_ds41(argv[3], c, &gh, &gw);
        if (!px) { fprintf(stderr, "image load failed\n"); goto out; }
        const int nh = (gh + c->merge - 1) / c->merge;
        const int nw = (gw + c->merge - 1) / c->merge;
        printf("grid %dx%d patches, %dx%d tokens, span %d, row %d\n",
               gh, gw, nh, nw, nh * (nw + 1) + 2, npix);
        if (argc > 4) {
            FILE *f = fopen(argv[4], "wb");
            if (f) {
                fwrite(px, sizeof(float), (size_t)gh * gw * npix, f);
                fclose(f);
            }
        }
        free(px);
        rc = 0;
        goto out;
    }

    if (strcmp(mode, "tower") || argc < 6) {
        fprintf(stderr, "tower: need H W pixels.bin\n");
        goto out;
    }
    {
        const int h = atoi(argv[3]), w = atoi(argv[4]);
        if (h <= 0 || w <= 0) { fprintf(stderr, "bad grid\n"); goto out; }
        const size_t L = (size_t)h * w;
        float *px = (float *)malloc(L * npix * sizeof(float));
        /* Wide enough for whichever stage was asked for: the merged output
         * is the smallest of them. */
        /* Wide enough for whichever stage was asked for, and for the span,
         * which has more rows than the token grid. */
        const size_t widest = (size_t)(c->hidden > c->out_hidden
                                       ? c->hidden : c->out_hidden);
        float *out = (float *)malloc((L + (size_t)h + 4) * widest * sizeof(float));
        if (!px || !out) { free(px); free(out); goto out; }
        FILE *f = fopen(argv[5], "rb");
        if (!f || fread(px, sizeof(float), L * npix, f) != L * npix) {
            fprintf(stderr, "pixels file short\n");
            if (f) fclose(f);
            free(px); free(out);
            goto out;
        }
        fclose(f);
        if (waste_vision_encode_ds41(&m, px, h, w, out)) {
            fprintf(stderr, "encode failed\n");
            free(px); free(out);
            goto out;
        }
        const char *st = getenv("WASTE_VIS_STAGE");
        const int merged = !st || !strcmp(st, "span");
        /* The whole span, delimiters included — one newline per token row
         * and two at the ends, which is what the LLM sees. */
        const int nh = (h + c->merge - 1) / c->merge;
        const int nw = (w + c->merge - 1) / c->merge;
        const int rows = merged ? nh * (nw + 1) + 2 : (int)L;
        const int dim = merged ? c->out_hidden : c->hidden;
        stats("tower ->", out, (size_t)rows * dim, rows, dim);
        if (argc > 6) {
            FILE *o = fopen(argv[6], "wb");
            if (o) { fwrite(out, sizeof(float), (size_t)rows * dim, o); fclose(o); }
        }
        free(px); free(out);
        rc = 0;
    }
out:
    waste_model_free(&m);
    return rc;
}
