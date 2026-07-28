/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/* test_state.c — a saved session must resume exactly where it left off.
 *
 *   ./test_state MODEL
 * Runs a prompt, saves, continues; then reloads the save and continues
 * again. The two continuations must be identical.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/waste.h"

typedef struct { int32_t ids[64]; int n; } sink;

static int on_tok(const waste_token_info *i, const char *piece, void *user)
{
    (void)piece;
    sink *s = (sink *)user;
    if (s->n < 64) s->ids[s->n++] = i->token;
    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 2) { fprintf(stderr, "usage: %s MODEL\n", argv[0]); return 2; }
    waste_cfg cfg;
    waste_cfg_init(&cfg);
    cfg.ram_budget_bytes = 6ULL << 30;
    waste_ctx *c;
    if (waste_open(argv[1], &cfg, &c) != WASTE_OK) { fprintf(stderr, "open\n"); return 1; }

    int32_t prompt[64];
    size_t n = 0;
    if (waste_tokenize(c, "The capital of France is Paris, and the capital of Italy is",
                       0, prompt, 64, &n) != WASTE_OK) { fprintf(stderr, "tok\n"); return 1; }

    waste_gen_params p;
    waste_gen_params_init(&p);
    p.max_tokens = 6;

    /* run the prompt, save, then continue */
    const char *path = "/tmp/waste_state_test.bin";
    sink a = {{0}, 0};
    const float *lg;
    if (waste_eval(c, prompt, n, &lg, NULL) != WASTE_OK) return 1;
    if (waste_state_save(c, path) != WASTE_OK) { fprintf(stderr, "save\n"); return 1; }
    int32_t nxt[1] = { 0 };
    for (int v = 1; v < 163840; v++) if (lg[v] > lg[nxt[0]]) nxt[0] = v;
    waste_generate(c, nxt, 1, &p, on_tok, &a);

    /* fresh context, load the save, continue from the same token */
    waste_close(c);
    if (waste_open(argv[1], &cfg, &c) != WASTE_OK) return 1;
    const waste_status st = waste_state_load(c, path);
    if (st != WASTE_OK) { fprintf(stderr, "load: %s\n", waste_strerror(st)); return 1; }
    sink b = {{0}, 0};
    waste_generate(c, nxt, 1, &p, on_tok, &b);
    waste_close(c);

    int same = (a.n == b.n);
    for (int i = 0; i < a.n && same; i++) same = (a.ids[i] == b.ids[i]);
    printf("fresh   :");
    for (int i = 0; i < a.n; i++) printf(" %d", a.ids[i]);
    printf("\nrestored:");
    for (int i = 0; i < b.n; i++) printf(" %d", b.ids[i]);
    printf("\n%s\n", same ? "STATE OK — restored session continues identically"
                          : "STATE MISMATCH");
    remove(path);
    return same ? 0 : 1;
}
