/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/*
 * test_qsa_pick.c — QSA's block top-k, against the loop it replaced.
 *
 *   ./test_qsa_pick
 *
 * waste_qwen_qsa_pick sorts where the selection used to take a repeated
 * argmax, and the order it writes is the order attention sums the selected
 * tokens in — so "the same set" is not enough, it has to be the same
 * sequence, bit for bit downstream. The reference check covers the usual
 * case; this covers the ones that decide an ordering: ties everywhere,
 * scores the argmax could never take (NaN, -1e30, -inf), and a budget below,
 * at and above the block count. The old loop is copied verbatim.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/qwen_qsa.h"

static int old_pick(const float *scores, int n_complete, int block_topk,
                    int compress, int n_tail, int *sel, int *taken)
{
    int nsel = 0;
    if (n_complete > 0) {
        for (int b = 0; b < n_complete; b++) taken[b] = 0;
        const int keep = n_complete < block_topk ? n_complete : block_topk;
        for (int j = 0; j < keep; j++) {
            int best = -1;
            float bv = -1e30f;
            for (int b = 0; b < n_complete; b++) {
                if (taken[b]) continue;
                if (scores[b] > bv) { bv = scores[b]; best = b; }
            }
            if (best < 0) break;
            taken[best] = 1;
            for (int t = 0; t < compress; t++)
                sel[nsel++] = best * compress + t;
        }
    }
    for (int t = 0; t < n_tail; t++)
        sel[nsel++] = n_complete * compress + t;
    return nsel;
}

static unsigned rng = 12345u;
static unsigned rnd(void) { rng = rng * 1103515245u + 12345u; return rng >> 8; }

int main(void)
{
    enum { MAXB = 1200, MAXSEL = MAXB * 4 + 8 };
    static float scores[MAXB];
    static int sel_a[MAXSEL], sel_b[MAXSEL], taken[MAXB], order[MAXB];
    int cases = 0;
    for (int it = 0; it < 4000; it++) {
        const int n = (int)(rnd() % (it < 200 ? 16 : MAXB));
        const int compress = 1 + (int)(rnd() % 4);
        const int n_tail = (int)(rnd() % compress);
        const int budget_kind = (int)(rnd() % 3);
        const int topk = budget_kind == 0 ? (n > 0 ? (int)(rnd() % n) : 0)
                       : budget_kind == 1 ? n : n + 1 + (int)(rnd() % 50);
        const int levels = 1 + (int)(rnd() % 6);   /* few distinct scores: ties */
        for (int b = 0; b < n; b++) {
            const unsigned r = rnd() % 100;
            if (r < 3)      scores[b] = NAN;
            else if (r < 5) scores[b] = -1e30f;
            else if (r < 6) scores[b] = -INFINITY;
            else if (r < 50) scores[b] = (float)(rnd() % levels);
            else            scores[b] = (float)(rnd() % 100000) / 7.0f;
        }
        const int na = old_pick(scores, n, topk, compress, n_tail, sel_a, taken);
        const int nb = waste_qwen_qsa_pick(scores, n, topk, compress, n_tail, sel_b, order);
        cases++;
        if (na != nb || memcmp(sel_a, sel_b, (size_t)na * sizeof(int)) != 0) {
            printf("FAIL case %d: n=%d topk=%d compress=%d tail=%d -> old %d, new %d\n",
                   it, n, topk, compress, n_tail, na, nb);
            return 1;
        }
    }
    printf("ok: %d cases, the pick matches the argmax it replaced\n", cases);
    return 0;
}
