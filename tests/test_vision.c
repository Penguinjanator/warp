/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/* Runs the C vision tower on pixels dumped by tools/vision_ref.py and
 * writes the embeddings, so the two can be diffed. */
#include "../src/model.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    if (argc < 6) {
        fprintf(stderr, "usage: %s MODEL pixels.bin H W out.bin\n", argv[0]);
        return 2;
    }
    const int h = atoi(argv[3]), w = atoi(argv[4]);
    waste_model m;
    if (waste_model_load(&m, argv[1], 512, 0, 1)) { fprintf(stderr, "load\n"); return 1; }
    if (!waste_vision_available(&m)) { fprintf(stderr, "no vision tower\n"); return 1; }

    const int L = h * w, PX = 3 * 14 * 14;
    float *px = (float *)malloc((size_t)L * PX * sizeof(float));
    FILE *f = fopen(argv[2], "rb");
    if (!f || fread(px, sizeof(float), (size_t)L * PX, f) != (size_t)L * PX) {
        fprintf(stderr, "pixels\n"); return 1;
    }
    fclose(f);

    /* a staged run writes L x vt_hidden instead of N x text_hidden */
    const char *st = getenv("WASTE_VIS_STAGE");
    const int N = st ? L : (h / 2) * (w / 2);
    const int TH = st ? m.vcfg.hidden : m.vcfg.text_hidden;
    float *out = (float *)malloc((size_t)N * TH * sizeof(float));
    if (waste_vision_encode(&m, px, h, w, out)) { fprintf(stderr, "encode\n"); return 1; }

    double mean = 0, sq = 0;
    for (size_t i = 0; i < (size_t)N * TH; i++) { mean += out[i]; sq += out[i] * out[i]; }
    mean /= (double)N * TH;
    printf("grid %dx%d -> (%d, %d)  mean %+.5f  std %.5f\n", h, w, N, TH,
           mean, sqrt(sq / ((double)N * TH) - mean * mean));

    f = fopen(argv[5], "wb");
    fwrite(out, sizeof(float), (size_t)N * TH, f);
    fclose(f);
    waste_model_free(&m);
    free(px); free(out);
    return 0;
}
