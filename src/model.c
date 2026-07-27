/*
 * model.c — container loading + forward pass. See model.h.
 *
 * One token per call: prefill is just repeated steps, which keeps the
 * decode path (the one that matters for a streaming engine) as the only
 * path, and makes the KDA/conv/KV state handling uniform.
 */

#define _GNU_SOURCE
#include "model.h"

#include <math.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

#include "json.h"
#include "kda.h"
#include "waste_backend.h"
#include "waste_format.h"

#define MAXP 512

static char *slurp(const char *path, size_t *len)
{
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *b = (char *)malloc((size_t)n + 1);
    if (!b || fread(b, 1, (size_t)n, f) != (size_t)n) { free(b); fclose(f); return NULL; }
    b[n] = 0;
    fclose(f);
    if (len) *len = (size_t)n;
    return b;
}

const waste_tensor *waste_find(const waste_model *m, const char *name)
{
    for (int i = 0; i < m->n_tensors; i++)
        if (strcmp(m->t[i].name, name) == 0) return &m->t[i];
    return NULL;
}

__attribute__((format(printf, 2, 3)))
static const float *T(const waste_model *m, const char *fmt, ...)
{
    char buf[160];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    const waste_tensor *t = waste_find(m, buf);
    return t ? t->data : NULL;
}

/* ---- kernels used only here (dispatchable later) ----------------------- */

static void rmsnorm(float *o, const float *x, const float *w, int n, float eps)
{
    float s = 0;
    for (int i = 0; i < n; i++) s += x[i] * x[i];
    const float r = 1.0f / sqrtf(s / (float)n + eps);
    for (int i = 0; i < n; i++) o[i] = x[i] * r * w[i];
}

/* y[out] = W[out][in] . x[in] */
static void matvec(float *y, const float *W, const float *x, int out, int in)
{
#pragma omp parallel for schedule(static)
    for (int o = 0; o < out; o++) {
        const float *row = W + (size_t)o * in;
        float s = 0;
        for (int i = 0; i < in; i++) s += row[i] * x[i];
        y[o] = s;
    }
}

static inline float silu(float v) { return v / (1.0f + expf(-v)); }

static void softmax(float *x, int n)
{
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float s = 0;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); s += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= s;
}

/* ---- loading ----------------------------------------------------------- */

static int load_trunk(waste_model *m, const char *dir, const js_doc *d, int trunk)
{
    char path[MAXP];
    snprintf(path, sizeof path, "%s/trunk.bin", dir);
    size_t blen;
    char *blob = slurp(path, &blen);
    if (!blob) return -1;

    m->n_tensors = d->tok[trunk].size;
    m->t = (waste_tensor *)calloc((size_t)m->n_tensors, sizeof *m->t);

    for (int i = 0; i < m->n_tensors; i++) {
        int e = js_at(d, trunk, i);
        waste_tensor *t = &m->t[i];
        js_str(d, js_get(d, e, "name"), t->name, sizeof t->name);
        const int fmt = (int)js_int(d, js_get(d, e, "fmt"), 0);
        const long off = js_int(d, js_get(d, e, "off"), 0);
        int sh = js_get(d, e, "shape");
        t->ndim = d->tok[sh].size;
        t->n = 1;
        for (int k = 0; k < t->ndim && k < 4; k++) {
            t->shape[k] = (int)js_int(d, js_at(d, sh, k), 1);
            t->n *= (size_t)t->shape[k];
        }
        t->data = (float *)malloc(t->n * sizeof(float));
        if (!t->data) { free(blob); return -1; }

        if (fmt == 0) {                                   /* F32 */
            memcpy(t->data, blob + off, t->n * sizeof(float));
        } else {                                          /* Q8G */
            const int g = (int)js_int(d, js_get(d, e, "group"), 128);
            const long soff = js_int(d, js_get(d, e, "scale_off"), 0);
            const int N = t->shape[t->ndim - 1];
            const long rows = (long)(t->n / (size_t)N);
            const int ng = (N + g - 1) / g;
            const int8_t *q = (const int8_t *)(blob + off);
            const uint16_t *sc = (const uint16_t *)(blob + soff);
            for (long r = 0; r < rows; r++) {
                for (int b = 0; b < ng; b++) {
                    /* fp16 -> float without <arm_fp16.h> assumptions */
                    const uint16_t h = sc[r * ng + b];
                    const uint32_t sign = (uint32_t)(h >> 15) << 31;
                    uint32_t exp = (h >> 10) & 0x1f, man = h & 0x3ff, bits;
                    if (exp == 0) { bits = sign; }
                    else { bits = sign | ((exp + 112u) << 23) | (man << 13); }
                    float s;
                    memcpy(&s, &bits, 4);
                    for (int k = 0; k < g; k++) {
                        const int col = b * g + k;
                        if (col >= N) break;
                        t->data[r * N + col] = (float)q[(r * ng + b) * g + k] * s;
                    }
                }
            }
        }
    }
    free(blob);
    return 0;
}

static void cfg_from_json(waste_config *c, const js_doc *d, int cfg)
{
    c->n_layers = (int)js_int(d, js_get(d, cfg, "num_hidden_layers"), 0);
    c->hidden = (int)js_int(d, js_get(d, cfg, "hidden_size"), 0);
    c->n_experts = (int)js_int(d, js_get(d, cfg, "num_experts"), 0);
    c->top_k = (int)js_int(d, js_get(d, cfg, "num_experts_per_token"), 0);
    c->moe_inter = (int)js_int(d, js_get(d, cfg, "moe_intermediate_size"), 0);
    c->dense_inter = (int)js_int(d, js_get(d, cfg, "intermediate_size"), 0);
    c->n_shared = (int)js_int(d, js_get(d, cfg, "num_shared_experts"), 0);
    c->first_dense = (int)js_int(d, js_get(d, cfg, "first_k_dense_replace"), 0);
    c->vocab = (int)js_int(d, js_get(d, cfg, "vocab_size"), 0);
    c->n_heads = (int)js_int(d, js_get(d, cfg, "num_attention_heads"), 0);
    c->kv_lora = (int)js_int(d, js_get(d, cfg, "kv_lora_rank"), 0);
    c->qk_nope = (int)js_int(d, js_get(d, cfg, "qk_nope_head_dim"), 0);
    c->qk_rope = (int)js_int(d, js_get(d, cfg, "qk_rope_head_dim"), 0);
    c->v_head = (int)js_int(d, js_get(d, cfg, "v_head_dim"), 0);
    c->eps = (float)js_num(d, js_get(d, cfg, "rms_norm_eps"), 1e-5);
    c->routed_scale = (float)js_num(d, js_get(d, cfg, "routed_scaling_factor"), 1.0);
    c->renorm = js_get(d, cfg, "moe_renormalize") >= 0;

    int lac = js_get(d, cfg, "linear_attn_config");
    c->kda_heads = (int)js_int(d, js_get(d, lac, "num_heads"), 0);
    c->kda_dim = (int)js_int(d, js_get(d, lac, "head_dim"), 0);
    c->conv_k = (int)js_int(d, js_get(d, lac, "short_conv_kernel_size"), 4);
    memset(c->kda_layer, 0, sizeof c->kda_layer);
    int kl = js_get(d, lac, "kda_layers");
    for (int i = 0; i < d->tok[kl].size; i++) {
        int v = (int)js_int(d, js_at(d, kl, i), -1) - 1;   /* list is 1-based */
        if (v >= 0 && v < 128) c->kda_layer[v] = 1;
    }
}

int waste_model_load(waste_model *m, const char *dir, int kv_cap)
{
    memset(m, 0, sizeof *m);
    m->kv_cap = kv_cap;
    waste_backend_init(WASTE_BE_AUTO);

    char path[MAXP];
    snprintf(path, sizeof path, "%s/manifest.json", dir);
    char *src = slurp(path, NULL);
    if (!src) return -1;
    js_doc d;
    if (js_parse(&d, src) < 0) { free(src); return -1; }

    cfg_from_json(&m->cfg, &d, js_get(&d, 0, "config"));
    const waste_config *c = &m->cfg;

    int eq = js_get(&d, 0, "expert_quant");
    m->stages = (int)js_int(&d, js_get(&d, eq, "stages"), 3);
    m->vec_dim = (int)js_int(&d, js_get(&d, eq, "vec_dim"), 8);
    m->cb_entries = (int)js_int(&d, js_get(&d, eq, "entries"), 256);

    if (load_trunk(m, dir, &d, js_get(&d, 0, "trunk")) < 0) { js_free(&d); free(src); return -1; }

    /* codebooks */
    snprintf(path, sizeof path, "%s/codebooks.bin", dir);
    size_t cblen;
    char *cb = slurp(path, &cblen);
    if (!cb) { js_free(&d); free(src); return -1; }
    const size_t rec = 16 + (size_t)m->cb_entries * m->vec_dim * 2;
    m->n_books = (int)(cblen / rec);
    m->codebooks = (float *)malloc((size_t)m->n_books * m->cb_entries * m->vec_dim * sizeof(float));
    for (int b = 0; b < m->n_books; b++) {
        const uint16_t *h = (const uint16_t *)(cb + b * rec + 16);
        for (int i = 0; i < m->cb_entries * m->vec_dim; i++) {
            const uint16_t v = h[i];
            const uint32_t sign = (uint32_t)(v >> 15) << 31;
            uint32_t exp = (v >> 10) & 0x1f, man = v & 0x3ff, bits;
            bits = exp ? (sign | ((exp + 112u) << 23) | (man << 13)) : sign;
            memcpy(&m->codebooks[(size_t)b * m->cb_entries * m->vec_dim + i], &bits, 4);
        }
    }
    free(cb);

    /* expert banks */
    m->expert_m[0] = m->expert_m[1] = c->moe_inter; m->expert_n[0] = m->expert_n[1] = c->hidden;
    m->expert_m[2] = c->hidden; m->expert_n[2] = c->moe_inter;
    int layers = js_get(&d, 0, "layers");
    for (int L = 0; L < c->n_layers; L++) {
        char key[16];
        snprintf(key, sizeof key, "%d", L);
        int e = js_get(&d, layers, key);
        if (e < 0) continue;
        char fn[64];
        js_str(&d, js_get(&d, e, "file"), fn, sizeof fn);
        snprintf(path, sizeof path, "%s/%s", dir, fn);
        m->bank[L].f = fopen(path, "rb");
        m->bank[L].n_experts = (int)js_int(&d, js_get(&d, e, "experts"), 0);
        m->bank[L].cb_base = (int)js_int(&d, js_get(&d, e, "codebook_base"), 0);
        const long bytes = js_int(&d, js_get(&d, e, "bytes"), 0);
        m->bank[L].rec_bytes = m->bank[L].n_experts ? bytes / m->bank[L].n_experts : 0;
    }
    js_free(&d);
    free(src);

    /* state + scratch */
    const int H = c->kda_heads, D = c->kda_dim, C = H * D;
    for (int L = 0; L < c->n_layers; L++) {
        if (c->kda_layer[L]) {
            m->S[L] = (float *)calloc((size_t)H * D * D, sizeof(float));
            m->conv[L] = (float *)calloc((size_t)3 * C * (c->conv_k - 1), sizeof(float));
        } else {
            const int kd = c->qk_nope + c->qk_rope;
            m->kcache[L] = (float *)calloc((size_t)kv_cap * c->n_heads * kd, sizeof(float));
            m->vcache[L] = (float *)calloc((size_t)kv_cap * c->n_heads * c->v_head, sizeof(float));
        }
    }
    const int big = c->hidden > C ? c->hidden : C;
    m->x = (float *)calloc((size_t)c->hidden, sizeof(float));
    m->h = (float *)calloc((size_t)c->hidden, sizeof(float));
    m->tmp = (float *)calloc((size_t)8 * big + 8 * c->moe_inter + 8 * c->dense_inter, sizeof(float));
    m->att = (float *)calloc((size_t)kv_cap * c->n_heads + 1024, sizeof(float));
    m->logits = (float *)calloc((size_t)c->vocab, sizeof(float));
    m->ff = (float *)calloc((size_t)2 * (c->dense_inter > c->moe_inter ? c->dense_inter : c->moe_inter), sizeof(float));
    m->e_gate = (float *)malloc((size_t)c->moe_inter * c->hidden * sizeof(float));
    m->e_up = (float *)malloc((size_t)c->moe_inter * c->hidden * sizeof(float));
    m->e_down = (float *)malloc((size_t)c->hidden * c->moe_inter * sizeof(float));
    return (m->x && m->logits && m->e_gate && m->e_up && m->e_down) ? 0 : -1;
}

void waste_model_free(waste_model *m)
{
    for (int i = 0; i < m->n_tensors; i++) free(m->t[i].data);
    free(m->t);
    free(m->codebooks);
    for (int L = 0; L < 128; L++) {
        free(m->S[L]); free(m->conv[L]); free(m->kcache[L]); free(m->vcache[L]);
        if (m->bank[L].f) fclose(m->bank[L].f);
    }
    free(m->x); free(m->h); free(m->tmp); free(m->att); free(m->logits);
    free(m->ff); free(m->e_gate); free(m->e_up); free(m->e_down);
}

/* ---- expert dequant ---------------------------------------------------- */

static void dequant_expert(waste_model *m, int L, int eid)
{
    waste_bank *b = &m->bank[L];
    static uint8_t *buf = NULL;
    static size_t bufsz = 0;
    if (bufsz < (size_t)b->rec_bytes) {
        free(buf);
        buf = (uint8_t *)malloc((size_t)b->rec_bytes);
        bufsz = (size_t)b->rec_bytes;
    }
    fseek(b->f, (long)eid * b->rec_bytes, SEEK_SET);
    if (fread(buf, 1, (size_t)b->rec_bytes, b->f) != (size_t)b->rec_bytes) return;
    m->expert_reads++;

    const waste_expert_hdr *h = (const waste_expert_hdr *)buf;
    const uint32_t off[3] = { h->gate_off, h->up_off, h->down_off };
    float *dst[3] = { m->e_gate, m->e_up, m->e_down };
    size_t sc_cur = h->chan_corr_off;

    for (int k = 0; k < 3; k++) {
        const int M = m->expert_m[k], N = m->expert_n[k];
        const long nvec = (long)M * N / m->vec_dim;
        const uint8_t *idx = buf + off[k];
        const uint16_t *sc = (const uint16_t *)(buf + sc_cur);
        sc_cur += (size_t)M * 2;
        for (long v = 0; v < nvec; v++) {
            float acc[16] = {0};
            for (int s = 0; s < m->stages; s++) {
                const int book = h->codebook_id + k * m->stages + s;
                const float *C = m->codebooks +
                    ((size_t)book * m->cb_entries + idx[v * m->stages + s]) * m->vec_dim;
                for (int j = 0; j < m->vec_dim; j++) acc[j] += C[j];
            }
            const long base = v * m->vec_dim;
            const int row = (int)(base / N);
            const uint16_t hs = sc[row];
            const uint32_t sign = (uint32_t)(hs >> 15) << 31;
            uint32_t e = (hs >> 10) & 0x1f, mn = hs & 0x3ff, bits;
            bits = e ? (sign | ((e + 112u) << 23) | (mn << 13)) : sign;
            float s;
            memcpy(&s, &bits, 4);
            for (int j = 0; j < m->vec_dim; j++) dst[k][base + j] = acc[j] * s;
        }
    }
}

/* ---- layers ------------------------------------------------------------ */

static void kda_layer(waste_model *m, int L, const float *in, float *out)
{
    const waste_config *c = &m->cfg;
    const int H = c->kda_heads, D = c->kda_dim, C = H * D, hid = c->hidden;
    float *q = m->tmp, *k = q + C, *v = k + C, *g = v + C, *o = g + C;
    float *beta = o + C, *lo = beta + H, *gate = lo + C;

    const char *nm[3] = { "q", "k", "v" };
    float *dstv[3] = { q, k, v };
    for (int i = 0; i < 3; i++) {
        char b[128];
        snprintf(b, sizeof b, "model.layers.%d.self_attn.%s_proj.weight", L, nm[i]);
        matvec(dstv[i], waste_find(m, b)->data, in, C, hid);
        snprintf(b, sizeof b, "model.layers.%d.self_attn.%s_conv1d.weight", L, nm[i]);
        waste_k.short_conv_step(C, c->conv_k, waste_find(m, b)->data, NULL,
                                m->conv[L] + (size_t)i * C * (c->conv_k - 1),
                                dstv[i], dstv[i]);
    }

    matvec(lo, T(m, "model.layers.%d.self_attn.f_a_proj.weight", L), in, D, hid);
    matvec(g, T(m, "model.layers.%d.self_attn.f_b_proj.weight", L), lo, C, D);
    const float *A_log = T(m, "model.layers.%d.self_attn.A_log", L);
    const float *dt = T(m, "model.layers.%d.self_attn.dt_bias", L);
    for (int h = 0; h < H; h++) {
        const float a = -expf(A_log[h]);
        for (int j = 0; j < D; j++) {
            const int i = h * D + j;
            const float z = g[i] + dt[i];
            /* softplus, numerically safe */
            const float sp = z > 20.0f ? z : log1pf(expf(z));
            g[i] = a * sp;
        }
    }
    matvec(beta, T(m, "model.layers.%d.self_attn.b_proj.weight", L), in, H, hid);
    for (int h = 0; h < H; h++) beta[h] = 1.0f / (1.0f + expf(-beta[h]));

    waste_k.kda_step(H, D, D, q, k, v, g, beta, m->S[L], o, m->att);

    matvec(lo, T(m, "model.layers.%d.self_attn.g_a_proj.weight", L), in, D, hid);
    matvec(gate, T(m, "model.layers.%d.self_attn.g_b_proj.weight", L), lo, C, D);
    const float *onw = T(m, "model.layers.%d.self_attn.o_norm.weight", L);
    for (int h = 0; h < H; h++)
        waste_k.rmsnorm_gated(D, o + h * D, gate + h * D, onw, c->eps, o + h * D);

    matvec(out, T(m, "model.layers.%d.self_attn.o_proj.weight", L), o, hid, C);
}

static void mla_layer(waste_model *m, int L, const float *in, float *out, int pos)
{
    const waste_config *c = &m->cfg;
    const int nh = c->n_heads, qd = c->qk_nope + c->qk_rope, vh = c->v_head;
    const int hid = c->hidden;
    float *q = m->tmp, *ckv = q + nh * qd, *kb = ckv + c->kv_lora + c->qk_rope;
    float *o = kb + nh * (c->qk_nope + vh);

    matvec(q, T(m, "model.layers.%d.self_attn.q_proj.weight", L), in, nh * qd, hid);
    matvec(ckv, T(m, "model.layers.%d.self_attn.kv_a_proj_with_mqa.weight", L), in,
           c->kv_lora + c->qk_rope, hid);
    float *kpass = ckv, *krot = ckv + c->kv_lora;
    rmsnorm(kpass, kpass, T(m, "model.layers.%d.self_attn.kv_a_layernorm.weight", L),
            c->kv_lora, c->eps);
    matvec(kb, T(m, "model.layers.%d.self_attn.kv_b_proj.weight", L), kpass,
           nh * (c->qk_nope + vh), c->kv_lora);

    float *kc = m->kcache[L] + (size_t)pos * nh * qd;
    float *vc = m->vcache[L] + (size_t)pos * nh * vh;
    for (int h = 0; h < nh; h++) {
        const float *src = kb + h * (c->qk_nope + vh);
        memcpy(kc + h * qd, src, (size_t)c->qk_nope * sizeof(float));
        memcpy(kc + h * qd + c->qk_nope, krot, (size_t)c->qk_rope * sizeof(float));
        memcpy(vc + h * vh, src + c->qk_nope, (size_t)vh * sizeof(float));
    }
    m->n_kv[L] = pos + 1;

    const float scale = 1.0f / sqrtf((float)qd);
    const int S = m->n_kv[L];
    for (int h = 0; h < nh; h++) {
        float *a = m->att;
        for (int s = 0; s < S; s++) {
            const float *kk = m->kcache[L] + ((size_t)s * nh + h) * qd;
            float acc = 0;
            for (int i = 0; i < qd; i++) acc += q[h * qd + i] * kk[i];
            a[s] = acc * scale;
        }
        softmax(a, S);
        float *oh = o + h * vh;
        memset(oh, 0, (size_t)vh * sizeof(float));
        for (int s = 0; s < S; s++) {
            const float *vv = m->vcache[L] + ((size_t)s * nh + h) * vh;
            const float w = a[s];
            for (int i = 0; i < vh; i++) oh[i] += w * vv[i];
        }
    }
    matvec(out, T(m, "model.layers.%d.self_attn.o_proj.weight", L), o, hid, nh * vh);
}

static void ffn(waste_model *m, const float *W1, const float *W3, const float *W2,
                const float *in, float *out, int inter, int hid, float w, int accum)
{
    float *a = m->ff, *b = a + inter;
    matvec(a, W1, in, inter, hid);
    matvec(b, W3, in, inter, hid);
    for (int i = 0; i < inter; i++) a[i] = silu(a[i]) * b[i];
    float *dst = accum ? m->h : out;
    matvec(dst, W2, a, hid, inter);
    if (accum) for (int i = 0; i < hid; i++) out[i] += w * dst[i];
}

static void moe_layer(waste_model *m, int L, const float *in, float *out, int *routed)
{
    const waste_config *c = &m->cfg;
    const int E = c->n_experts, K = c->top_k, hid = c->hidden;
    float *sc = m->att + 4096;
    matvec(sc, T(m, "model.layers.%d.block_sparse_moe.gate.weight", L), in, E, hid);
    const float *bias = T(m, "model.layers.%d.block_sparse_moe.gate.e_score_correction_bias", L);
    float *score = sc + E;
    for (int e = 0; e < E; e++) score[e] = 1.0f / (1.0f + expf(-sc[e]));

    int idx[64];
    float w[64];
    for (int j = 0; j < K; j++) {
        int best = -1;
        float bv = -1e30f;
        for (int e = 0; e < E; e++) {
            int taken = 0;
            for (int p = 0; p < j; p++) if (idx[p] == e) { taken = 1; break; }
            if (taken) continue;
            const float v = score[e] + (bias ? bias[e] : 0.0f);
            if (v > bv) { bv = v; best = e; }
        }
        idx[j] = best;
        w[j] = score[best];
    }
    if (c->renorm && K > 1) {
        float s = 0;
        for (int j = 0; j < K; j++) s += w[j];
        for (int j = 0; j < K; j++) w[j] /= (s + 1e-20f);
    }
    for (int j = 0; j < K; j++) w[j] *= c->routed_scale;
    if (routed) for (int j = 0; j < K; j++) routed[j] = idx[j];

    memset(out, 0, (size_t)hid * sizeof(float));
    for (int j = 0; j < K; j++) {
        dequant_expert(m, L, idx[j]);
        ffn(m, m->e_gate, m->e_up, m->e_down, in, out, c->moe_inter, hid, w[j], 1);
    }
    /* shared expert */
    float *tmp = m->h;
    ffn(m, T(m, "model.layers.%d.block_sparse_moe.shared_experts.gate_proj.weight", L),
        T(m, "model.layers.%d.block_sparse_moe.shared_experts.up_proj.weight", L),
        T(m, "model.layers.%d.block_sparse_moe.shared_experts.down_proj.weight", L),
        in, tmp, c->moe_inter * (c->n_shared ? c->n_shared : 1), hid, 1.0f, 0);
    for (int i = 0; i < hid; i++) out[i] += tmp[i];
}

/* ---- forward ----------------------------------------------------------- */

const float *waste_model_step(waste_model *m, int token, int pos, int *routed)
{
    const waste_config *c = &m->cfg;
    const int hid = c->hidden;
    const float *emb = waste_find(m, "model.embed_tokens.weight")->data;
    memcpy(m->x, emb + (size_t)token * hid, (size_t)hid * sizeof(float));

    float *resid = (float *)malloc((size_t)hid * sizeof(float));
    float *norm = (float *)malloc((size_t)hid * sizeof(float));
    for (int L = 0; L < c->n_layers; L++) {
        char b[128];
        snprintf(b, sizeof b, "model.layers.%d.input_layernorm.weight", L);
        rmsnorm(norm, m->x, waste_find(m, b)->data, hid, c->eps);
        if (c->kda_layer[L]) kda_layer(m, L, norm, resid);
        else mla_layer(m, L, norm, resid, pos);
        for (int i = 0; i < hid; i++) m->x[i] += resid[i];

        snprintf(b, sizeof b, "model.layers.%d.post_attention_layernorm.weight", L);
        rmsnorm(norm, m->x, waste_find(m, b)->data, hid, c->eps);
        snprintf(b, sizeof b, "model.layers.%d.block_sparse_moe.gate.weight", L);
        if (waste_find(m, b))
            moe_layer(m, L, norm, resid, routed ? routed + (size_t)L * c->top_k : NULL);
        else
            ffn(m, T(m, "model.layers.%d.mlp.gate_proj.weight", L),
                T(m, "model.layers.%d.mlp.up_proj.weight", L),
                T(m, "model.layers.%d.mlp.down_proj.weight", L),
                norm, resid, c->dense_inter, hid, 1.0f, 0);
        for (int i = 0; i < hid; i++) m->x[i] += resid[i];
    }
    rmsnorm(norm, m->x, waste_find(m, "model.norm.weight")->data, hid, c->eps);
    matvec(m->logits, waste_find(m, "lm_head.weight")->data, norm, c->vocab, hid);
    free(resid);
    free(norm);
    return m->logits;
}
