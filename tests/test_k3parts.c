/*
 * test_k3parts.c — exercise the three K3 components whose maths is new,
 * and dump inputs + outputs so tools/k3parts_ref.py can recompute them
 * from the released reference source and diff.
 *
 * These cannot be validated end to end until the weights land, so they are
 * validated in isolation now, on synthetic inputs.
 *
 *   cc -O2 -o test_k3parts tests/test_k3parts.c libwaste.a -lm -lpthread
 *   ./test_k3parts out.bin
 *
 * Layout of out.bin (little-endian f32 unless noted):
 *   [situ]   i32 n, f32 beta, f32 linear_beta, gate[n], up[n], out[n]
 *   [gate]   i32 H, i32 D, f32 lower_bound, g_in[H*D], A_log[H],
 *            dt[H*D], g_k3[H*D], g_linear[H*D]
 *   [ares]   i32 nb, i32 hid, f32 eps, blocks[nb*hid], prefix[hid],
 *            norm_w[hid], proj_w[hid], out[hid]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/model.h"

static uint64_t rng = 0x243F6A8885A308D3ULL;
static float frand(void)
{
    rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
    return (float)((double)((rng >> 11) & 0x1FFFFFFFFFFFFFULL) / 9007199254740992.0
                   * 2.0 - 1.0);
}

static void wr(FILE *f, const float *v, int n) { fwrite(v, sizeof(float), (size_t)n, f); }
static void wi(FILE *f, int v) { fwrite(&v, sizeof(int), 1, f); }
static void wf(FILE *f, float v) { fwrite(&v, sizeof(float), 1, f); }

int main(int argc, char **argv)
{
    const char *out = argc > 1 ? argv[1] : "k3parts.bin";
    FILE *f = fopen(out, "wb");
    if (!f) { perror("open"); return 1; }

    /* ---- 1. SiTU ------------------------------------------------------ */
    {
        const int n = 512;
        const float beta = 4.0f, lbeta = 25.0f;    /* K3's config values */
        float *g = malloc((size_t)n * 4), *u = malloc((size_t)n * 4),
              *o = malloc((size_t)n * 4);
        for (int i = 0; i < n; i++) { g[i] = frand() * 30.0f; u[i] = frand() * 30.0f; }
        for (int i = 0; i < n; i++) o[i] = waste_situ_pair(g[i], u[i], beta, lbeta);
        wi(f, n); wf(f, beta); wf(f, lbeta);
        wr(f, g, n); wr(f, u, n); wr(f, o, n);
        free(g); free(u); free(o);
    }

    /* ---- 2. decay gate, both forms ------------------------------------ */
    {
        const int H = 8, D = 16, n = H * D;
        const float lb = -5.0f;
        float *gin = malloc((size_t)n * 4), *A = malloc((size_t)H * 4),
              *dt = malloc((size_t)n * 4), *g1 = malloc((size_t)n * 4),
              *g2 = malloc((size_t)n * 4);
        for (int i = 0; i < n; i++) { gin[i] = frand() * 4.0f; dt[i] = frand(); }
        for (int i = 0; i < H; i++) A[i] = frand();
        memcpy(g1, gin, (size_t)n * 4);
        waste_kda_decay_gate(g1, A, dt, H, D, lb);          /* K3 form */
        memcpy(g2, gin, (size_t)n * 4);
        waste_kda_decay_gate(g2, A, dt, H, D, 0.0f);        /* Kimi-Linear form */
        wi(f, H); wi(f, D); wf(f, lb);
        wr(f, gin, n); wr(f, A, H); wr(f, dt, n); wr(f, g1, n); wr(f, g2, n);
        free(gin); free(A); free(dt); free(g1); free(g2);
    }

    /* ---- 3. attention residuals --------------------------------------- */
    {
        const int nb = 5, hid = 128;
        const float eps = 1e-5f;
        waste_model m;
        memset(&m, 0, sizeof m);
        m.cfg.hidden = hid;
        m.cfg.eps = eps;
        m.n_blockres = nb;
        m.blockres = malloc((size_t)nb * hid * 4);
        m.att = malloc(4096 * 4);
        float *ps = malloc((size_t)hid * 4), *nw = malloc((size_t)hid * 4),
              *pw = malloc((size_t)hid * 4), *o = malloc((size_t)hid * 4);
        for (int i = 0; i < nb * hid; i++) m.blockres[i] = frand() * 2.0f;
        for (int i = 0; i < hid; i++) { ps[i] = frand() * 2.0f; nw[i] = frand(); pw[i] = frand(); }
        waste_apply_attn_res(&m, ps, nw, pw, o);
        wi(f, nb); wi(f, hid); wf(f, eps);
        wr(f, m.blockres, nb * hid); wr(f, ps, hid);
        wr(f, nw, hid); wr(f, pw, hid); wr(f, o, hid);
        free(m.blockres); free(m.att); free(ps); free(nw); free(pw); free(o);
    }

    fclose(f);
    printf("wrote %s\n", out);
    return 0;
}
