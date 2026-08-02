/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/*
 * sweep.c — one model load, many configurations, one machine state.
 *
 * A K3 measurement costs about 50 seconds of which 48 are model load and
 * prefill, and that is the smaller problem. The larger one is that every
 * heavy run changes the machine the next one lands on, so two arms measured
 * in two processes are measured on two computers: docs/LEARNED.md §32 and
 * §33 are both records of a conclusion that came out wrong for exactly that
 * reason, and §16's "sweep upward, never downward" exists because of it.
 *
 * So: load once, then run the arms back to back, interleaved, with the
 * expert cache cleared and the session reset between each. What is left
 * varying between two adjacent measurements is the setting and roughly
 * nothing else.
 *
 * The one thing it still cannot vary is the context length, which sizes the
 * KV and KDA state at load.
 *
 *   sweep CONTAINER ids,.. n_gen lookahead=0,6 [repeat]
 *   sweep CONTAINER ids,.. n_gen iodepth=2,4,8 [repeat]
 *   sweep CONTAINER ids,.. n_gen cache=3400,17736,23879 [repeat]
 *
 * `cache` is in MB and re-makes the expert cache in place. The trunk is what
 * a load costs, not the cache, so a budget sweep no longer needs a process
 * per budget — which is what made §32 and §33 come out wrong, each process
 * meeting a machine the one before it had changed. The footprint at each arm
 * is what that budget would have made it: the same trunk plus the cache
 * being asked for.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "../src/model.h"

static double now(void)
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec / 1e6;
}

#define MAX_ARMS 16
#define MAX_IDS 512

int main(int argc, char **argv)
{
    if (argc < 5) {
        fprintf(stderr,
                "usage: %s CONTAINER ids,.. n_gen KEY=v1,v2,.. [repeat]\n"
                "  KEY is lookahead, iodepth or cache (MB)\n", argv[0]);
        return 2;
    }
    int ids[MAX_IDS], n = 0;
    for (char *p = strtok(argv[2], ","); p && n < MAX_IDS; p = strtok(NULL, ","))
        ids[n++] = atoi(p);
    const int n_gen = atoi(argv[3]);

    char *eq = strchr(argv[4], '=');
    if (!eq) { fprintf(stderr, "expected KEY=v1,v2,..\n"); return 2; }
    *eq = 0;
    const char *key = argv[4];
    int arm[MAX_ARMS], n_arms = 0;
    for (char *p = strtok(eq + 1, ","); p && n_arms < MAX_ARMS; p = strtok(NULL, ","))
        arm[n_arms++] = atoi(p);
    const int repeat = argc > 5 ? atoi(argv[5]) : 3;

    const int is_look = !strcmp(key, "lookahead");
    const int is_depth = !strcmp(key, "iodepth");
    const int is_cache = !strcmp(key, "cache");
    if (!is_look && !is_depth && !is_cache) {
        fprintf(stderr, "unknown key %s\n", key);
        return 2;
    }

    waste_model m;
    waste_load_opts lo;
    memset(&lo, 0, sizeof lo);
    const char *cmb = getenv("WASTE_CACHE_MB");
    lo.cache_bytes = (size_t)(cmb ? atoi(cmb) : 0) << 20;
    lo.direct_io = 1;
    double t0 = now();
    if (waste_model_load(&m, argv[1], 4096, &lo)) {
        fprintf(stderr, "load failed\n");
        return 1;
    }
    printf("loaded in %.1fs — cache %d slots; %d arms x %d repeats, "
           "%d prompt + %d generated\n\n",
           now() - t0, m.cache.n_slots, n_arms, repeat, n, n_gen);

    /* Interleaved rather than grouped: if the machine drifts over the run —
     * and it does — a grouped sweep charges the drift to the last arm and
     * an interleaved one spreads it across all of them. */
    printf("%8s %6s %7s %9s %9s %9s\n", key, "rep", "slots", "tok/s", "hit",
           "GB read");
    for (int r = 0; r < repeat; r++) {
        for (int a = 0; a < n_arms; a++) {
            if (is_look) {
                waste_model_set_lookahead(arm[a]);
            } else if (is_depth) {
                m.cache.depth = arm[a] < 1 ? 1 : arm[a];
            } else if (waste_model_resize_cache(&m, (size_t)arm[a] << 20)) {
                fprintf(stderr, "resize to %d MB failed\n", arm[a]);
                return 1;
            }

            waste_model_reset(&m);
            waste_ecache_clear(&m.cache);

            const float *lg = NULL;
            for (int i = 0; i < n; i++) lg = waste_model_step(&m, ids[i], i, NULL);
            if (!lg) { fprintf(stderr, "prompt failed\n"); return 1; }

            int cur = 0;
            for (int v = 1; v < m.cfg.vocab; v++) if (lg[v] > lg[cur]) cur = v;
            const double s = now();
            for (int i = 0; i < n_gen; i++) {
                lg = waste_model_step(&m, cur, n + i, NULL);
                if (!lg) { fprintf(stderr, "step failed\n"); return 1; }
                cur = 0;
                for (int v = 1; v < m.cfg.vocab; v++) if (lg[v] > lg[cur]) cur = v;
            }
            const double dt = now() - s;
            const unsigned long long h = m.cache.hits, mi = m.cache.misses;
            printf("%8d %6d %7d %8.3f %8.1f%% %8.1f\n", arm[a], r + 1,
                   m.cache.n_slots, n_gen / dt,
                   100.0 * (double)h / (double)(h + mi ? h + mi : 1),
                   (double)m.cache.bytes_read / 1073741824.0);
            fflush(stdout);
        }
    }
    waste_model_free(&m);
    return 0;
}
