/*
 * test_forward.c — run the C forward pass and dump logits for the oracle diff.
 *
 *   cc -O2 -fopenmp -o test_forward tests/test_forward.c src/model.c src/kda.c \
 *      src/kda_neon.c src/backend.c -lm
 *   ./test_forward /Users/marco/models/kimi-linear.waste 1008,10484,318,15383,387 out.bin
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

#include "../src/model.h"
#include "../src/waste_backend.h"

static double now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: %s container ids[,..] [out.bin] [n_gen]\n", argv[0]);
        return 2;
    }
    const char *dir = argv[1];
    int ids[512], n = 0;
    for (char *p = strtok(argv[2], ","); p && n < 512; p = strtok(NULL, ","))
        ids[n++] = atoi(p);
    const char *out = argc > 3 ? argv[3] : NULL;
    const int n_gen = argc > 4 ? atoi(argv[4]) : 0;

    waste_model m;
    double t0 = now();
    if (waste_model_load(&m, dir, 4096)) { fprintf(stderr, "load failed\n"); return 1; }
    printf("loaded in %.1fs — backend %s, %d layers, %d experts, top-%d, vocab %d\n",
           now() - t0, waste_backend_name(), m.cfg.n_layers, m.cfg.n_experts,
           m.cfg.top_k, m.cfg.vocab);

    const float *lg = NULL;
    t0 = now();
    for (int i = 0; i < n; i++) lg = waste_model_step(&m, ids[i], i, NULL);
    const double tp = now() - t0;

    int best = 0;
    for (int v = 1; v < m.cfg.vocab; v++) if (lg[v] > lg[best]) best = v;
    printf("prefill %d tok in %.2fs (%.2f tok/s); argmax %d, max %.4f\n",
           n, tp, n / tp, best, lg[best]);

    if (out) {
        FILE *f = fopen(out, "wb");
        fwrite(lg, sizeof(float), (size_t)m.cfg.vocab, f);
        fclose(f);
        printf("wrote %s\n", out);
    }

    int cur = best;
    for (int i = 0; i < n_gen; i++) {
        t0 = now();
        lg = waste_model_step(&m, cur, n + i, NULL);
        best = 0;
        for (int v = 1; v < m.cfg.vocab; v++) if (lg[v] > lg[best]) best = v;
        printf("  [%3d] %6d  (%.2fs, %llu expert reads)\n", i, best, now() - t0,
               (unsigned long long)m.expert_reads);
        cur = best;
    }

    extern double waste_prof[8];
    if (getenv("WASTE_PROFILE")) {
        const char *names[8] = {"proj","kda","mla","moe(all)","  expert deq",
                                "  expert mm","lm_head","other"};
        double tot = 0;
        for (int i = 0; i < 8; i++) tot += (i == 4 || i == 5) ? 0 : waste_prof[i];
        printf("\n-- profile (s, %d steps) --\n", n + n_gen);
        for (int i = 0; i < 8; i++)
            if (waste_prof[i] > 0)
                printf("  %-14s %7.2f  %5.1f%%\n", names[i], waste_prof[i],
                       100.0 * waste_prof[i] / tot);
        printf("  %-14s %7.2f\n", "accounted", tot);
    }
    waste_model_free(&m);
    return 0;
}
