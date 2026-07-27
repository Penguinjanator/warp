/*
 * test_kda.c — run the KDA kernel over a fixture produced by
 * tools/kda_ref.py and write the outputs back for comparison against
 * fla's naive_recurrent_kda (the reference shipped with Kimi-Linear).
 *
 *   cc -O2 -o test_kda tests/test_kda.c src/kda.c -lm
 *   ./test_kda fixture.bin out.bin
 *
 * Fixture layout (little-endian f32, dims as int32):
 *   T H K V | q[T][H][K] k[T][H][K] v[T][H][V] g[T][H][K] beta[T][H] S0[H][K][V]
 * Output: o[T][H][V] then S_final[H][K][V]
 */

#include <stdio.h>
#include <stdlib.h>

#include "../src/kda.h"

static void *xread(FILE *f, size_t n)
{
    void *p = malloc(n);
    if (!p || fread(p, 1, n, f) != n) { fprintf(stderr, "short read\n"); exit(1); }
    return p;
}

int main(int argc, char **argv)
{
    if (argc < 3) { fprintf(stderr, "usage: %s fixture.bin out.bin\n", argv[0]); return 2; }
    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror("fixture"); return 1; }

    int32_t d[4];
    if (fread(d, sizeof(int32_t), 4, f) != 4) { fprintf(stderr, "bad header\n"); return 1; }
    const int T = d[0], H = d[1], K = d[2], V = d[3];
    printf("T=%d H=%d K=%d V=%d\n", T, H, K, V);

    const size_t nqk = (size_t)T * H * K, nv = (size_t)T * H * V;
    float *q = xread(f, nqk * sizeof(float));
    float *k = xread(f, nqk * sizeof(float));
    float *v = xread(f, nv * sizeof(float));
    float *g = xread(f, nqk * sizeof(float));
    float *beta = xread(f, (size_t)T * H * sizeof(float));
    float *S = xread(f, (size_t)H * K * V * sizeof(float));
    fclose(f);

    float *o = calloc(nv, sizeof(float));
    float *scratch = calloc((size_t)V, sizeof(float));
    if (!o || !scratch) { fprintf(stderr, "oom\n"); return 1; }

    waste_kda_forward(T, H, K, V, q, k, v, g, beta, S, o, scratch);

    FILE *out = fopen(argv[2], "wb");
    if (!out) { perror("out"); return 1; }
    fwrite(o, sizeof(float), nv, out);
    fwrite(S, sizeof(float), (size_t)H * K * V, out);
    fclose(out);
    printf("wrote %s\n", argv[2]);
    return 0;
}
