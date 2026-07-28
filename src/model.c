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

#include <fcntl.h>
#include <unistd.h>

#include "ecache.h"
#include "json.h"
#include "threads.h"
#include "kda.h"
#include "waste_backend.h"
#include "waste_format.h"

#define MAXP 512

/* ---- lightweight phase profiling (WASTE_PROFILE=1) --------------------- */
#include <time.h>
double waste_prof[8];
uint64_t waste_prof_n[8];
enum { P_LUTB, P_KDA, P_MLA, P_ROUTE, P_EDEQ, P_EMM, P_HEAD, P_LUTA };
static int prof_on = -1;
static double pnow(void)
{
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
}
#define PROF_START(b) double _t##b = prof_on ? pnow() : 0
#define PROF_END(b)   do { if (prof_on) { waste_prof[b] += pnow() - _t##b; waste_prof_n[b]++; } } while (0)

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

/* Formats a tensor name. Rotates over several buffers because callers pass
 * two or three of these to the same function, and C does not order
 * argument evaluation — a single static buffer would make them all alias. */
__attribute__((format(printf, 1, 2)))
static const char *tname(const char *fmt, ...)
{
    static char buf[8][160];
    static int turn = 0;
    char *b = buf[turn++ & 7];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(b, 160, fmt, ap);
    va_end(ap);
    return b;
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

static inline float f16_to_f32(uint16_t h)
{
    const uint32_t sign = (uint32_t)(h >> 15) << 31;
    const uint32_t e = (h >> 10) & 0x1f, mn = h & 0x3ff;
    const uint32_t bits = e ? (sign | ((e + 112u) << 23) | (mn << 13)) : sign;
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

/* ---- kernels used only here (dispatchable later) ----------------------- */

static void rmsnorm(float *o, const float *x, const float *w, int n, float eps)
{
    float s = 0;
    for (int i = 0; i < n; i++) s += x[i] * x[i];
    const float r = 1.0f / sqrtf(s / (float)n + eps);
    for (int i = 0; i < n; i++) o[i] = x[i] * r * w[i];
}

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
static inline float dotf(const float *a, const float *b, int n)
{
    float32x4_t s0 = vdupq_n_f32(0), s1 = vdupq_n_f32(0);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        s0 = vfmaq_f32(s0, vld1q_f32(a + i), vld1q_f32(b + i));
        s1 = vfmaq_f32(s1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
    }
    float acc = vaddvq_f32(vaddq_f32(s0, s1));
    for (; i < n; i++) acc += a[i] * b[i];
    return acc;
}
#else
static inline float dotf(const float *a, const float *b, int n)
{
    float acc = 0;
    for (int i = 0; i < n; i++) acc += a[i] * b[i];
    return acc;
}
#endif

static int q8_off = 1;     /* 1 = keep the trunk stored as int8          */
static int sdot_on = 0;    /* 1 = also quantize activations (SDOT path)  */

/* ---- int8 x int8 matvec (SDOT / dotprod) --------------------------------
 * The trunk is already stored Q8G: int8 weights with one fp16 scale per
 * group of 128 inputs. Keeping it that way (instead of expanding to f32 at
 * load) saves ~6 GB of RAM and lets the dot run on ARM SDOT / x86 VNNI.
 * Activations are quantized per group with the same geometry, so a group
 * contributes scale_w * scale_x * <int32 dot>.
 */
typedef struct {
    float *y; const int8_t *W; const uint16_t *ws;
    const int8_t *xq; const float *xs; int in, ng, group, bits;
} mvq_arg;

static inline int32_t idot(const int8_t *a, const int8_t *b, int n)
{
#if defined(__ARM_FEATURE_DOTPROD)
    int32x4_t acc = vdupq_n_s32(0);
    int i = 0;
    for (; i + 16 <= n; i += 16)
        acc = vdotq_s32(acc, vld1q_s8(a + i), vld1q_s8(b + i));
    int32_t s = vaddvq_s32(acc);
    for (; i < n; i++) s += (int32_t)a[i] * b[i];
    return s;
#else
    int32_t s = 0;
    for (int i = 0; i < n; i++) s += (int32_t)a[i] * b[i];
    return s;
#endif
}

/* int8 weights, f32 activations: dequantize the row inline. Keeps the
 * 4x memory saving of int8 storage while producing exactly the numbers the
 * f32 path does — no activation quantization error. */
static void mvq_rows_f32(int b, int e, void *p)
{
    mvq_arg *a = (mvq_arg *)p;
    const int ng = a->ng, g = a->group;
    const float *x = (const float *)a->xs;      /* raw activations */
    int8_t *unp = NULL;
    if (a->bits == 4) {
        unp = (int8_t *)malloc((size_t)g);
        if (!unp) return;
    }
    for (int o = b; o < e; o++) {
        const int8_t *row = a->W + (size_t)o * ng * g * a->bits / 8;
        const uint16_t *ws = a->ws + (size_t)o * ng;
        float acc = 0;
        for (int k = 0; k < ng; k++) {
            const int8_t *w;
            if (a->bits == 4) {
                /* two signed nibbles per byte, low first */
                const uint8_t *p4 = (const uint8_t *)row + (size_t)k * g / 2;
                for (int i = 0; i < g / 2; i++) {
                    const uint8_t byte = p4[i];
                    unp[2 * i]     = (int8_t)(byte & 0x0F) - 8;
                    unp[2 * i + 1] = (int8_t)(byte >> 4) - 8;
                }
                w = unp;
            } else {
                w = row + (size_t)k * g;
            }
            const float *xx = x + (size_t)k * g;
            const int lim = (k * g + g <= a->in) ? g : a->in - k * g;
#if defined(__ARM_NEON) || defined(__aarch64__)
            float32x4_t s0 = vdupq_n_f32(0);
            int i = 0;
            for (; i + 8 <= lim; i += 8) {
                const int16x8_t w16 = vmovl_s8(vld1_s8(w + i));
                s0 = vfmaq_f32(s0, vcvtq_f32_s32(vmovl_s16(vget_low_s16(w16))),
                               vld1q_f32(xx + i));
                s0 = vfmaq_f32(s0, vcvtq_f32_s32(vmovl_s16(vget_high_s16(w16))),
                               vld1q_f32(xx + i + 4));
            }
            float part = vaddvq_f32(s0);
            for (; i < lim; i++) part += (float)w[i] * xx[i];
#else
            float part = 0;
            for (int i = 0; i < lim; i++) part += (float)w[i] * xx[i];
#endif
            acc += f16_to_f32(ws[k]) * part;
        }
        a->y[o] = acc;
    }
    free(unp);
}

static void mvq_rows(int b, int e, void *p)
{
    mvq_arg *a = (mvq_arg *)p;
    const int ng = a->ng, g = a->group;
    for (int o = b; o < e; o++) {
        const int8_t *row = a->W + (size_t)o * ng * g;
        const uint16_t *ws = a->ws + (size_t)o * ng;
        float acc = 0;
        for (int k = 0; k < ng; k++)
            acc += f16_to_f32(ws[k]) * a->xs[k] *
                   (float)idot(row + (size_t)k * g, a->xq + (size_t)k * g, g);
        a->y[o] = acc;
    }
}

typedef struct { float *y; const float *W, *x; int in; } mv_arg;

static void mv_rows(int b, int e, void *p)
{
    mv_arg *a = (mv_arg *)p;
    for (int o = b; o < e; o++)
        a->y[o] = dotf(a->W + (size_t)o * a->in, a->x, a->in);
}

/* y[out] = W[out][in] . x[in]; row split, so results do not depend on
 * the thread count. */
static void matvec(float *y, const float *W, const float *x, int out, int in)
{
    mv_arg a = { y, W, x, in };
    waste_parallel_for(out, 64, mv_rows, &a);
}

/* Quantize x into per-group int8 (same grouping as the weights). */
static void quant_act(const float *x, int n, int g, int8_t *q, float *sc)
{
    const int ng = (n + g - 1) / g;
    for (int k = 0; k < ng; k++) {
        const int beg = k * g, end = (beg + g < n) ? beg + g : n;
        float amax = 0;
        for (int i = beg; i < end; i++) {
            const float v = fabsf(x[i]);
            if (v > amax) amax = v;
        }
        const float s = amax > 0 ? amax / 127.0f : 1.0f;
        sc[k] = s;
        const float inv = 1.0f / s;
        for (int i = beg; i < end; i++) {
            int v = (int)lrintf(x[i] * inv);
            q[i] = (int8_t)(v > 127 ? 127 : (v < -127 ? -127 : v));
        }
        for (int i = end; i < beg + g; i++) q[i] = 0;
    }
}

/* Matvec against a trunk tensor, quantized path when available. */
static void matvec_t(waste_model *m, float *y, const waste_tensor *t,
                     const float *x, int out, int in)
{
    if (!t || (!t->q && !t->data)) { memset(y, 0, (size_t)out * sizeof(float)); return; }
    if (!t->q) { matvec(y, t->data, x, out, in); return; }
    const int g = t->group, ng = (in + g - 1) / g;
    if (sdot_on && t->bits == 8) {
        quant_act(x, in, g, m->xq, m->xs);
        mvq_arg a = { y, t->q, t->qs, m->xq, m->xs, in, ng, g, 8 };
        waste_parallel_for(out, 64, mvq_rows, &a);
    } else {
        mvq_arg a = { y, t->q, t->qs, NULL, x, in, ng, g, t->bits };
        waste_parallel_for(out, 64, mvq_rows_f32, &a);
    }
}

static inline float silu(float v) { return v / (1.0f + expf(-v)); }

/* SiTU (K3): beta*tanh(g/beta)*sigmoid(g) * [linear_beta*tanh(u/linear_beta)]
 * — replaces SiLU-and-multiply, and unlike it the "up" half is squashed too. */
float waste_situ_pair(float g, float u, float beta, float lbeta)
{
    const float a = beta * tanhf(g / beta) / (1.0f + expf(-g));
    return a * (lbeta > 0.0f ? lbeta * tanhf(u / lbeta) : u);
}

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
        const int g = (int)js_int(d, js_get(d, e, "group"), 128);
        const long soff = js_int(d, js_get(d, e, "scale_off"), 0);
        int sh = js_get(d, e, "shape");
        t->ndim = d->tok[sh].size;
        t->n = 1;
        for (int k = 0; k < t->ndim && k < 4; k++) {
            t->shape[k] = (int)js_int(d, js_at(d, sh, k), 1);
            t->n *= (size_t)t->shape[k];
        }
        if (fmt == 0) {                                   /* F32 */
            t->data = (float *)malloc(t->n * sizeof(float));
            if (!t->data) { free(blob); return -1; }
            memcpy(t->data, blob + off, t->n * sizeof(float));
        } else if (!q8_off) {                             /* Q8G -> f32 */
            const int N = t->shape[t->ndim - 1];
            const long rows = (long)(t->n / (size_t)N);
            const int ng = (N + g - 1) / g;
            t->data = (float *)malloc(t->n * sizeof(float));
            if (!t->data) { free(blob); return -1; }
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
        } else {                             /* Q8G or Q4G, kept quantized */
            const int N = t->shape[t->ndim - 1];
            const long rows = (long)(t->n / (size_t)N);
            const int ng = (N + g - 1) / g;
            t->group = g;
            t->bits = (fmt == 3) ? 4 : 8;   /* FMT_Q4G = 3 */
            const size_t payload = (size_t)rows * ng * g * t->bits / 8;
            t->q = (int8_t *)malloc(payload);
            t->qs = (uint16_t *)malloc((size_t)rows * ng * sizeof(uint16_t));
            if (!t->q || !t->qs) { free(blob); return -1; }
            memcpy(t->q, blob + off, payload);
            memcpy(t->qs, blob + soff, (size_t)rows * ng * sizeof(uint16_t));
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
    c->q_lora = (int)js_int(d, js_get(d, cfg, "q_lora_rank"), 0);
    c->qk_nope = (int)js_int(d, js_get(d, cfg, "qk_nope_head_dim"), 0);
    c->qk_rope = (int)js_int(d, js_get(d, cfg, "qk_rope_head_dim"), 0);
    c->v_head = (int)js_int(d, js_get(d, cfg, "v_head_dim"), 0);
    c->eps = (float)js_num(d, js_get(d, cfg, "rms_norm_eps"), 1e-5);
    c->routed_scale = (float)js_num(d, js_get(d, cfg, "routed_scaling_factor"), 1.0);
    c->renorm = js_get(d, cfg, "moe_renormalize") >= 0;

    c->latent_dim = (int)js_int(d, js_get(d, cfg, "routed_expert_hidden_size"), 0);
    c->latent_norm = js_get(d, cfg, "latent_moe_use_norm") >= 0;
    c->attn_res_block = (int)js_int(d, js_get(d, cfg, "attn_res_block_size"), 0);
    c->mla_output_gate = js_get(d, cfg, "mla_use_output_gate") >= 0;
    {
        char act[32];
        js_str(d, js_get(d, cfg, "hidden_act"), act, sizeof act);
        c->act_situ = strcmp(act, "situ") == 0;
    }
    c->situ_beta = (float)js_num(d, js_get(d, cfg, "activation_situ_beta"), 1.0);
    c->situ_linear_beta = (float)js_num(d, js_get(d, cfg, "activation_situ_linear_beta"), 0.0);

    int lac = js_get(d, cfg, "linear_attn_config");
    c->full_rank_gate = js_get(d, lac, "use_full_rank_gate") >= 0;
    c->gate_lower_bound = (float)js_num(d, js_get(d, lac, "gate_lower_bound"), 0.0);
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

int waste_model_load(waste_model *m, const char *dir, int kv_cap,
                     size_t cache_bytes)
{
    memset(m, 0, sizeof *m);
    if (prof_on < 0) { const char *e = getenv("WASTE_PROFILE"); prof_on = e && *e != '0'; }
    m->kv_cap = kv_cap;
    waste_backend_init(WASTE_BE_AUTO);
    {
        const char *e = getenv("WASTE_THREADS");
        waste_pool_init(e ? atoi(e) : 0);
    }

    char path[MAXP];
    snprintf(path, sizeof path, "%s/manifest.json", dir);
    char *src = slurp(path, NULL);
    if (!src) return -1;
    { const char *e = getenv("WASTE_Q8"); if (e && *e == '0') q8_off = 0; }
    { const char *e = getenv("WASTE_SDOT"); sdot_on = e && *e != '0'; }
    js_doc d;
    if (js_parse(&d, src) < 0) { free(src); return -1; }

    {
        int cfg = js_get(&d, 0, "config");
        /* The converter flattens K3's nested text_config into `config` and
         * records the tensor-name prefix separately, so read that — probing
         * for a nested text_config here finds nothing and silently leaves
         * every tensor lookup one prefix short. */
        js_str(&d, js_get(&d, 0, "tensor_prefix"), m->cfg.prefix,
               sizeof m->cfg.prefix);
        const int tc = js_get(&d, cfg, "text_config");   /* raw HF config */
        if (tc >= 0) {
            cfg = tc;
            if (!m->cfg.prefix[0])
                snprintf(m->cfg.prefix, sizeof m->cfg.prefix, "language_model.");
        }
        cfg_from_json(&m->cfg, &d, cfg);
    }
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
        m->bank[L].fd = open(path, O_RDONLY);
#ifdef __APPLE__
        if (m->bank[L].fd >= 0) {
            fcntl(m->bank[L].fd, F_NOCACHE, 1);   /* the engine owns caching */
            fcntl(m->bank[L].fd, F_RDAHEAD, 0);
        }
#endif
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
    m->tmp = (float *)calloc((size_t)8 * big + 8 * c->moe_inter + 8 * c->dense_inter
                             + (size_t)4 * c->n_heads * (c->v_head + c->qk_nope + c->qk_rope)
                             + (size_t)2 * (c->q_lora ? c->q_lora : 1) + 256,
                             sizeof(float));
    m->att = (float *)calloc((size_t)kv_cap * c->n_heads + 1024, sizeof(float));
    m->logits = (float *)calloc((size_t)c->vocab, sizeof(float));
    m->ff = (float *)calloc((size_t)2 * (c->dense_inter > c->moe_inter ? c->dense_inter : c->moe_inter), sizeof(float));
    m->e_gate = (float *)malloc((size_t)c->moe_inter * c->hidden * sizeof(float));
    m->e_up = (float *)malloc((size_t)c->moe_inter * c->hidden * sizeof(float));
    m->e_down = (float *)malloc((size_t)c->hidden * c->moe_inter * sizeof(float));
    {   /* AttnRes history + the latent-MoE staging buffers */
        const int nb = c->attn_res_block ? c->n_layers / c->attn_res_block + 2 : 0;
        m->blockres = nb ? (float *)calloc((size_t)nb * c->hidden, sizeof(float)) : NULL;
        m->prefix_sum = (float *)calloc((size_t)c->hidden, sizeof(float));
        m->ares = (float *)calloc((size_t)(2 * c->hidden + 2 * (c->latent_dim
                                  ? c->latent_dim : c->hidden)), sizeof(float));
    }
    {
        const int nmax = c->hidden > c->dense_inter ? c->hidden : c->dense_inter;
        m->xq = (int8_t *)calloc((size_t)nmax + 256, 1);
        m->xs = (float *)calloc((size_t)nmax / 32 + 64, sizeof(float));
    }

    {   /* LUT: [max_nv][stages][entries] */
        const int nmax = (c->hidden > c->moe_inter ? c->hidden : c->moe_inter);
        m->lut = (float *)malloc((size_t)3 * (nmax / 8 + 1) * m->stages
                                 * m->cb_entries * sizeof(float));
    }
    {   /* expert cache, sized by the caller's budget */
        long rec = 0;
        for (int L = 0; L < c->n_layers; L++)
            if (m->bank[L].rec_bytes > rec) rec = m->bank[L].rec_bytes;
        if (waste_ecache_init(&m->cache, cache_bytes, (size_t)rec, 0)) return -1;
        m->miss_buf = (uint8_t *)malloc((size_t)rec);
        if (!m->miss_buf) return -1;
    }
    return (m->x && m->logits && m->e_gate && m->e_up && m->e_down) ? 0 : -1;
}

void waste_model_free(waste_model *m)
{
    for (int i = 0; i < m->n_tensors; i++) {
        free(m->t[i].data); free(m->t[i].q); free(m->t[i].qs);
    }
    free(m->t);
    free(m->codebooks);
    for (int L = 0; L < 128; L++) {
        free(m->S[L]); free(m->conv[L]); free(m->kcache[L]); free(m->vcache[L]);
        if (m->bank[L].fd > 0) close(m->bank[L].fd);
    }
    free(m->x); free(m->h); free(m->tmp); free(m->att); free(m->logits);
    free(m->ff); free(m->e_gate); free(m->e_up); free(m->e_down); free(m->lut);
    free(m->xq); free(m->xs); free(m->miss_buf);
    free(m->blockres); free(m->prefix_sum); free(m->ares);
    free(m->cx); free(m->cnorm); free(m->cresid); free(m->cq); free(m->ckv);
    free(m->clat); free(m->cff); free(m->cexp); free(m->cblockres);
    free(m->cprefix); free(m->croute); free(m->crw);
    waste_ecache_free(&m->cache);
}

/* ---- expert dequant ---------------------------------------------------- */

/* One pread of a 4 KiB-aligned record — what the layout exists for. */
static int bank_fetch(void *user, int layer, int expert, uint8_t *dst)
{
    waste_model *m = (waste_model *)user;
    waste_bank *b = &m->bank[layer];
    const ssize_t got = pread(b->fd, dst, (size_t)b->rec_bytes,
                              (off_t)expert * b->rec_bytes);
    if (got != (ssize_t)b->rec_bytes) return -1;
    m->expert_reads++;
    return 0;
}

static const uint8_t *read_expert(waste_model *m, int L, int eid)
{
    if (m->cache.n_slots > 0)
        return waste_ecache_get(&m->cache, L, eid, bank_fetch, m);
    m->cache.misses++;
    m->cache.bytes_read += (size_t)m->bank[L].rec_bytes;
    return bank_fetch(m, L, eid, m->miss_buf) == 0 ? m->miss_buf : NULL;
}

/* ---- fused VQ matvec ---------------------------------------------------
 * Never materializes the weights. For a matrix stored as `stages` codebook
 * indices per 8-dim vector, y[row] = scale[row] * sum_v sum_s C_s[i]. x_v.
 * The inner term depends only on (stage, code, vector position), not on
 * the row — so it is tabulated once per matrix and reused by every row,
 * and (for gate/up, whose input is the layer's hidden state) by every
 * routed expert in the layer. Same idea as sqlite-vector's turbo LUT.
 *
 * lut layout: [v][stage][code] — the 3 values a row needs for vector v sit
 * in one contiguous 3 KiB block.
 */
static void vq_build_lut(float *lut, const float *books, int cb_base,
                         const float *x, int N, int stages, int entries,
                         int vec_dim)
{
    PROF_START(P_LUTB);
    const int nv = N / vec_dim;
    for (int v = 0; v < nv; v++) {
        const float *xv = x + (size_t)v * vec_dim;
        for (int s = 0; s < stages; s++) {
            const float *C = books + (size_t)(cb_base + s) * entries * vec_dim;
            float *dst = lut + ((size_t)v * stages + s) * entries;
            for (int c = 0; c < entries; c++)
                dst[c] = dotf(C + (size_t)c * vec_dim, xv, vec_dim);
        }
    }
    PROF_END(P_LUTB);
}

typedef struct {
    float *y; const uint8_t *idx; const uint16_t *scale; const float *lut;
    int nv, stages, entries;
} vq_arg;

/* Row-tiled so the per-position table block is loaded once for a whole
 * tile instead of once per row. The naive row-outer/vector-inner order
 * streams the entire table (884 KB for a 2304-wide matrix) M times; at
 * M=1024 that is ~900 MB of traffic per matrix. With a tile of 64 rows the
 * table is read M/64 times and the tile's indices (55 KB) stay in L1.
 * This is a cache-blocking win, not a SIMD one: the inner op is a gather,
 * which NEON cannot vectorize. */
#define VQ_TILE 64          /* must equal the container's index_block */
#ifndef VQ_SUPER
#define VQ_SUPER 2          /* index blocks handled per pass (swept: 2 wins) */
#endif

/* What this loop actually costs, measured rather than assumed:
 *   - it is NOT table bandwidth. Re-reading the 884 KB table once per
 *     64-row tile works out to 8.2 GB/token, which would need 165 GB/s —
 *     suspiciously exactly this machine's ceiling — but raising VQ_SUPER
 *     to cut that traffic made it *slower* (0.25 -> 0.40 s at SUPER=8),
 *     because the table is shared read-only across threads and stays
 *     cached, while the extra index streams and accumulators do not.
 *   - it is NOT index locality either. Blocking the index layout so a
 *     tile's indices are contiguous measured 1.44x in isolation and
 *     changed nothing here.
 *   - it IS the load -> address -> load dependency of each gather.
 *     Interleaving four independent rows keeps four chains in flight and
 *     is what finally moved it.
 * Swept: VQ_SUPER 1 and 2 tie within noise, 4+ is worse. */
static void vq_rows(int b, int e, void *p)
{
    vq_arg *a = (vq_arg *)p;
    const int nv = a->nv, st = a->stages, en = a->entries;
    float acc[VQ_TILE * VQ_SUPER];

    for (int r0 = b; r0 < e; r0 += VQ_TILE * VQ_SUPER) {
        const int rows = (r0 + VQ_TILE * VQ_SUPER < e) ? VQ_TILE * VQ_SUPER : e - r0;
        const int nblk = (rows + VQ_TILE - 1) / VQ_TILE;
        memset(acc, 0, (size_t)rows * sizeof(float));

        for (int v = 0; v < nv; v++) {
            const float *blk = a->lut + (size_t)v * st * en;
            for (int j = 0; j < nblk; j++) {
                const int nr = (j + 1) * VQ_TILE <= rows ? VQ_TILE
                                                         : rows - j * VQ_TILE;
                const uint8_t *ix = a->idx +
                    ((size_t)(r0 / VQ_TILE + j) * nv + v) * VQ_TILE * st;
                float *ac = acc + (size_t)j * VQ_TILE;
                /* Each gather is load -> address -> load, a ~5-cycle chain.
                 * Four rows are independent, so interleaving them keeps
                 * four chains in flight instead of one. */
                int r = 0;
                if (st == 3) {
                    for (; r + 4 <= nr; r += 4, ix += 4 * 3) {
                        const float t0 = blk[ix[0]] + blk[en + ix[1]] + blk[2 * en + ix[2]];
                        const float t1 = blk[ix[3]] + blk[en + ix[4]] + blk[2 * en + ix[5]];
                        const float t2 = blk[ix[6]] + blk[en + ix[7]] + blk[2 * en + ix[8]];
                        const float t3 = blk[ix[9]] + blk[en + ix[10]] + blk[2 * en + ix[11]];
                        ac[r] += t0; ac[r + 1] += t1; ac[r + 2] += t2; ac[r + 3] += t3;
                    }
                }
                for (; r < nr; r++, ix += st) {
                    float t = blk[ix[0]];
                    for (int s = 1; s < st; s++) t += blk[s * en + ix[s]];
                    ac[r] += t;
                }
            }
        }
        for (int r = 0; r < rows; r++)
            a->y[r0 + r] = acc[r] * f16_to_f32(a->scale[r0 + r]);
    }
}

static void vq_apply(waste_model *m, float *y, const uint8_t *idx,
                     const uint16_t *scale, int M, int N, const float *lut)
{
    PROF_START(P_LUTA);
    vq_arg a = { y, idx, scale, lut, N / m->vec_dim, m->stages, m->cb_entries };
    /* min_chunk = VQ_TILE keeps every thread's range block-aligned, which
     * the blocked index layout requires. */
    waste_parallel_for(M, VQ_TILE * VQ_SUPER, vq_rows, &a);
    PROF_END(P_LUTA);
}

static void vq_matvec(waste_model *m, float *y, const uint8_t *idx,
                      const uint16_t *scale, const float *x, int M, int N,
                      int cb_base, float *lut)
{
    vq_build_lut(lut, m->codebooks, cb_base, x, N, m->stages, m->cb_entries,
                 m->vec_dim);
    vq_apply(m, y, idx, scale, M, N, lut);
}

/* ---- layers ------------------------------------------------------------ */

/* Log-space decay gate, in place over [H][D].
 *
 * Kimi-Linear:  g = -exp(A_log) * softplus(z)         unbounded below
 * K3:           g = lower_bound * sigmoid(exp(A_log) * z)
 *
 * The second is not a clamp of the first — it is a different function, and
 * it confines the decay exp(g) to (exp(lower_bound), 1).
 */
/* `per_channel`: K3 ships A_log with head_dim entries (one per channel,
 * broadcast over heads); Kimi-Linear ships one per head. The tensor's own
 * length says which, so the caller passes that rather than guessing. */
void waste_kda_decay_gate_ex(float *g, const float *A_log, const float *dt_bias,
                             int H, int D, float lower_bound, int per_channel)
{
    for (int h = 0; h < H; h++) {
        const float ea_head = per_channel ? 0.0f : expf(A_log[h]);
        for (int j = 0; j < D; j++) {
            const int i = h * D + j;
            const float ea = per_channel ? expf(A_log[j]) : ea_head;
            const float z = g[i] + (dt_bias ? dt_bias[i] : 0.0f);
            if (lower_bound < 0.0f)
                g[i] = lower_bound / (1.0f + expf(-ea * z));
            else {
                const float sp = z > 20.0f ? z : log1pf(expf(z));
                g[i] = -ea * sp;
            }
        }
    }
}

void waste_kda_decay_gate(float *g, const float *A_log, const float *dt_bias,
                          int H, int D, float lower_bound)
{
    waste_kda_decay_gate_ex(g, A_log, dt_bias, H, D, lower_bound, 0);
}

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
        snprintf(b, sizeof b, "%smodel.layers.%d.self_attn.%s_proj.weight", c->prefix, L, nm[i]);
        matvec_t(m, dstv[i], waste_find(m, b), in, C, hid);
        snprintf(b, sizeof b, "%smodel.layers.%d.self_attn.%s_conv1d.weight", c->prefix, L, nm[i]);
        waste_k.short_conv_step(C, c->conv_k, waste_find(m, b)->data, NULL,
                                m->conv[L] + (size_t)i * C * (c->conv_k - 1),
                                dstv[i], dstv[i]);
    }

    matvec_t(m, lo, waste_find(m, tname("%smodel.layers.%d.self_attn.f_a_proj.weight", c->prefix, L)), in, D, hid);
    matvec_t(m, g, waste_find(m, tname("%smodel.layers.%d.self_attn.f_b_proj.weight", c->prefix, L)), lo, C, D);
    const waste_tensor *At = waste_find(m, tname("%smodel.layers.%d.self_attn.A_log",
                                                 c->prefix, L));
    const float *dt = T(m, "%smodel.layers.%d.self_attn.dt_bias", c->prefix, L);
    waste_kda_decay_gate_ex(g, At->data, dt, H, D, c->gate_lower_bound,
                            (int)At->n == D && D != H);
    matvec_t(m, beta, waste_find(m, tname("%smodel.layers.%d.self_attn.b_proj.weight", c->prefix, L)), in, H, hid);
    for (int h = 0; h < H; h++) beta[h] = 1.0f / (1.0f + expf(-beta[h]));

    waste_k.kda_step(H, D, D, q, k, v, g, beta, m->S[L], o, m->att);

    if (c->full_rank_gate) {
        matvec_t(m, gate, waste_find(m, tname("%smodel.layers.%d.self_attn.g_proj.weight",
                                              c->prefix, L)), in, C, hid);
    } else {
        matvec_t(m, lo, waste_find(m, tname("%smodel.layers.%d.self_attn.g_a_proj.weight",
                                            c->prefix, L)), in, D, hid);
        matvec_t(m, gate, waste_find(m, tname("%smodel.layers.%d.self_attn.g_b_proj.weight",
                                              c->prefix, L)), lo, C, D);
    }
    const float *onw = T(m, "%smodel.layers.%d.self_attn.o_norm.weight", c->prefix, L);
    for (int h = 0; h < H; h++)
        waste_k.rmsnorm_gated(D, o + h * D, gate + h * D, onw, c->eps, o + h * D);

    matvec_t(m, out, waste_find(m, tname("%smodel.layers.%d.self_attn.o_proj.weight", c->prefix, L)), o, hid, C);
}

static void mla_layer(waste_model *m, int L, const float *in, float *out, int pos)
{
    const waste_config *c = &m->cfg;
    const int nh = c->n_heads, qd = c->qk_nope + c->qk_rope, vh = c->v_head;
    const int hid = c->hidden;
    float *q = m->tmp, *ckv = q + nh * qd, *kb = ckv + c->kv_lora + c->qk_rope;
    float *o = kb + nh * (c->qk_nope + vh);

    if (c->q_lora) {
        /* K3 LoRAs the query too: q_a -> RMSNorm -> q_b */
        float *qa = o + (size_t)nh * vh;
        matvec_t(m, qa, waste_find(m, tname("%smodel.layers.%d.self_attn.q_a_proj.weight",
                                            c->prefix, L)), in, c->q_lora, hid);
        rmsnorm(qa, qa, waste_find(m, tname("%smodel.layers.%d.self_attn.q_a_layernorm.weight",
                                            c->prefix, L))->data, c->q_lora, c->eps);
        matvec_t(m, q, waste_find(m, tname("%smodel.layers.%d.self_attn.q_b_proj.weight",
                                           c->prefix, L)), qa, nh * qd, c->q_lora);
    } else {
        matvec_t(m, q, waste_find(m, tname("%smodel.layers.%d.self_attn.q_proj.weight",
                                           c->prefix, L)), in, nh * qd, hid);
    }
    matvec_t(m, ckv, waste_find(m, tname("%smodel.layers.%d.self_attn.kv_a_proj_with_mqa.weight", c->prefix, L)),
             in, c->kv_lora + c->qk_rope, hid);
    float *kpass = ckv, *krot = ckv + c->kv_lora;
    rmsnorm(kpass, kpass, T(m, "%smodel.layers.%d.self_attn.kv_a_layernorm.weight", c->prefix, L),
            c->kv_lora, c->eps);
    matvec_t(m, kb, waste_find(m, tname("%smodel.layers.%d.self_attn.kv_b_proj.weight", c->prefix, L)),
             kpass, nh * (c->qk_nope + vh), c->kv_lora);

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
    if (c->mla_output_gate) {
        /* sigmoid gate on the attention output, before o_proj */
        float *g = o + (size_t)nh * vh + (c->q_lora ? c->q_lora : 0);
        matvec_t(m, g, waste_find(m, tname("%smodel.layers.%d.self_attn.g_proj.weight",
                                           c->prefix, L)), in, nh * vh, hid);
        for (int i = 0; i < nh * vh; i++) o[i] *= 1.0f / (1.0f + expf(-g[i]));
    }
    matvec_t(m, out, waste_find(m, tname("%smodel.layers.%d.self_attn.o_proj.weight", c->prefix, L)), o, hid, nh * vh);
}

static void ffn(waste_model *m, const waste_tensor *W1, const waste_tensor *W3,
                const waste_tensor *W2, const float *in, float *out,
                int inter, int hid, float w, int accum)
{
    float *a = m->ff, *b = a + inter;
    matvec_t(m, a, W1, in, inter, hid);
    matvec_t(m, b, W3, in, inter, hid);
    if (m->cfg.act_situ)
        for (int i = 0; i < inter; i++)
            a[i] = waste_situ_pair(a[i], b[i], m->cfg.situ_beta, m->cfg.situ_linear_beta);
    else
        for (int i = 0; i < inter; i++) a[i] = silu(a[i]) * b[i];
    float *dst = accum ? m->h : out;
    matvec_t(m, dst, W2, a, hid, inter);
    if (accum) for (int i = 0; i < hid; i++) out[i] += w * dst[i];
}

static void moe_layer(waste_model *m, int L, const float *in, float *out, int *routed)
{
    const waste_config *c = &m->cfg;
    const int E = c->n_experts, K = c->top_k, hid = c->hidden;
    /* K3's Stable LatentMoE: experts run on a narrower projection of the
     * hidden state. `in` still drives the router and the shared experts. */
    const int lat = c->latent_dim ? c->latent_dim : hid;
    float *sc = m->att + 4096;
    matvec_t(m, sc, waste_find(m, tname("%smodel.layers.%d.block_sparse_moe.gate.weight", c->prefix, L)), in, E, hid);
    const float *bias = T(m, "%smodel.layers.%d.block_sparse_moe.gate.e_score_correction_bias", c->prefix, L);
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

    const int inter = c->moe_inter;
    float *ga = m->ff, *ub = ga + inter, *acc = m->e_gate;
    const float *xin = in;
    float *lat_in = m->ares + hid, *lat_out = lat_in + lat;
    if (c->latent_dim) {
        matvec_t(m, lat_in, waste_find(m, tname(
                     "%smodel.layers.%d.block_sparse_moe.routed_expert_down_proj.weight",
                     c->prefix, L)), in, lat, hid);
        xin = lat_in;
    }
    float *ysum = c->latent_dim ? lat_out : out;
    memset(ysum, 0, (size_t)lat * sizeof(float));
    const int lut_sz = ((hid > lat ? hid : lat) / m->vec_dim) * m->stages * m->cb_entries;
    float *lut_gate = m->lut, *lut_up = lut_gate + lut_sz, *lut_down = lut_up + lut_sz;
    int lut_ready = 0;
    for (int j = 0; j < K; j++) {
        PROF_START(P_EDEQ);
        const uint8_t *rec = read_expert(m, L, idx[j]);
        PROF_END(P_EDEQ);
        if (!rec) continue;
        const waste_expert_hdr *h = (const waste_expert_hdr *)rec;
        const uint16_t *sc = (const uint16_t *)(rec + h->chan_corr_off);

        PROF_START(P_EMM);
        /* gate/up see the same input and the same per-layer codebooks for
         * every routed expert, so their tables are built once per token. */
        if (!lut_ready) {
            vq_build_lut(lut_gate, m->codebooks, h->codebook_id + 0 * m->stages,
                         xin, lat, m->stages, m->cb_entries, m->vec_dim);
            vq_build_lut(lut_up, m->codebooks, h->codebook_id + 1 * m->stages,
                         xin, lat, m->stages, m->cb_entries, m->vec_dim);
            lut_ready = 1;
        }
        vq_apply(m, ga, rec + h->gate_off, sc, inter, lat, lut_gate);
        vq_apply(m, ub, rec + h->up_off, sc + inter, inter, lat, lut_up);
        if (c->act_situ)
            for (int i = 0; i < inter; i++)
                ga[i] = waste_situ_pair(ga[i], ub[i], c->situ_beta, c->situ_linear_beta);
        else
            for (int i = 0; i < inter; i++) ga[i] = silu(ga[i]) * ub[i];
        vq_matvec(m, acc, rec + h->down_off, sc + 2 * inter, ga, lat, inter,
                  h->codebook_id + 2 * m->stages, lut_down);
        const float wj = w[j];
        for (int i = 0; i < lat; i++) ysum[i] += wj * acc[i];
        PROF_END(P_EMM);
    }
    if (c->latent_dim) {
        if (c->latent_norm)
            rmsnorm(ysum, ysum, waste_find(m, tname(
                        "%smodel.layers.%d.block_sparse_moe.routed_expert_norm.weight",
                        c->prefix, L))->data, lat, c->eps);
        matvec_t(m, out, waste_find(m, tname(
                     "%smodel.layers.%d.block_sparse_moe.routed_expert_up_proj.weight",
                     c->prefix, L)), ysum, hid, lat);
    }

    /* shared expert — on the original hidden state, not the latent */
    float *tmp = m->h;
    ffn(m, waste_find(m, tname("%smodel.layers.%d.block_sparse_moe.shared_experts.gate_proj.weight", c->prefix, L)),
        waste_find(m, tname("%smodel.layers.%d.block_sparse_moe.shared_experts.up_proj.weight", c->prefix, L)),
        waste_find(m, tname("%smodel.layers.%d.block_sparse_moe.shared_experts.down_proj.weight", c->prefix, L)),
        in, tmp, c->moe_inter * (c->n_shared ? c->n_shared : 1), hid, 1.0f, 0);
    for (int i = 0; i < hid; i++) out[i] += tmp[i];
}

/* ---- Attention Residuals (K3) ------------------------------------------
 * Every layer mixes its running sum with a history of block residuals via a
 * learned softmax attention over that history; every attn_res_block_size
 * layers the current sum is appended to the history.
 *
 *   v      = [block_residual..., prefix_sum]        (nb+1) x hidden
 *   k      = rmsnorm(v)                             (no weight yet)
 *   scores = sum(k * (norm.weight * proj.weight))   (nb+1)
 *   out    = softmax(scores) . v
 */
void waste_apply_attn_res(waste_model *m, const float *blockres, int nb,
                          const float *prefix_sum, const float *norm_w,
                          const float *proj_w, float *out)
{
    const int hid = m->cfg.hidden;
    float *sc = m->att;
    for (int i = 0; i <= nb; i++) {
        const float *v = (i < nb) ? blockres + (size_t)i * hid : prefix_sum;
        float ss = 0;
        for (int j = 0; j < hid; j++) ss += v[j] * v[j];
        const float r = 1.0f / sqrtf(ss / (float)hid + m->cfg.eps);
        float acc = 0;
        for (int j = 0; j < hid; j++) acc += v[j] * r * norm_w[j] * proj_w[j];
        sc[i] = acc;
    }
    softmax(sc, nb + 1);
    memset(out, 0, (size_t)hid * sizeof(float));
    for (int i = 0; i <= nb; i++) {
        const float *v = (i < nb) ? blockres + (size_t)i * hid : prefix_sum;
        const float p = sc[i];
        for (int j = 0; j < hid; j++) out[j] += p * v[j];
    }
}

int waste_model_warm_cache(waste_model *m, const char *dir)
{
    if (m->cache.n_slots <= 0) return 0;
    char p[512];
    snprintf(p, sizeof p, "%s/usage.waste", dir);
    return waste_ecache_warm(&m->cache, p, bank_fetch, m);
}

int waste_model_save_usage(const waste_model *m, const char *dir)
{
    if (m->cache.n_slots <= 0) return 0;
    char p[512];
    snprintf(p, sizeof p, "%s/usage.waste", dir);
    return waste_ecache_save_usage(&m->cache, p, m->cache.clock);
}

/* ---- session state ------------------------------------------------------
 * Written with the shapes it depends on, so a state file that no longer
 * matches the model is rejected rather than silently producing nonsense.
 */

typedef struct {
    uint32_t magic, version;
    int32_t  n_layers, hidden, kda_heads, kda_dim, conv_k, n_heads;
    int32_t  qk_nope, qk_rope, v_head, attn_res_block;
    int32_t  pos, n_blockres;
    uint32_t reserved[2];
} waste_state_hdr;

static void state_fill(const waste_model *m, waste_state_hdr *h, int pos)
{
    const waste_config *c = &m->cfg;
    memset(h, 0, sizeof *h);
    h->magic = WASTE_MAGIC_KDASTATE;
    h->version = 1;
    h->n_layers = c->n_layers; h->hidden = c->hidden;
    h->kda_heads = c->kda_heads; h->kda_dim = c->kda_dim; h->conv_k = c->conv_k;
    h->n_heads = c->n_heads; h->qk_nope = c->qk_nope; h->qk_rope = c->qk_rope;
    h->v_head = c->v_head; h->attn_res_block = c->attn_res_block;
    h->pos = pos; h->n_blockres = m->n_blockres;
}

int waste_model_state_save(const waste_model *m, const char *path, int pos)
{
    const waste_config *c = &m->cfg;
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    waste_state_hdr h;
    state_fill(m, &h, pos);
    int rc = fwrite(&h, sizeof h, 1, f) == 1 ? 0 : -1;

    const int H = c->kda_heads, D = c->kda_dim, C = H * D;
    const int qd = c->qk_nope + c->qk_rope;
    for (int L = 0; L < c->n_layers && !rc; L++) {
        if (c->kda_layer[L]) {
            if (fwrite(m->S[L], sizeof(float), (size_t)H * D * D, f) != (size_t)H * D * D) rc = -1;
            const size_t cn = (size_t)3 * C * (c->conv_k - 1);
            if (!rc && fwrite(m->conv[L], sizeof(float), cn, f) != cn) rc = -1;
        } else {
            const int32_t nkv = m->n_kv[L];
            if (fwrite(&nkv, sizeof nkv, 1, f) != 1) { rc = -1; break; }
            const size_t kn = (size_t)nkv * c->n_heads * qd;
            const size_t vn = (size_t)nkv * c->n_heads * c->v_head;
            if (kn && fwrite(m->kcache[L], sizeof(float), kn, f) != kn) rc = -1;
            if (!rc && vn && fwrite(m->vcache[L], sizeof(float), vn, f) != vn) rc = -1;
        }
    }
    if (!rc && c->attn_res_block && m->n_blockres > 0) {
        const size_t n = (size_t)m->n_blockres * c->hidden;
        if (fwrite(m->blockres, sizeof(float), n, f) != n) rc = -1;
    }
    if (!rc && fwrite(m->x, sizeof(float), (size_t)c->hidden, f) != (size_t)c->hidden) rc = -1;
    fclose(f);
    if (rc) remove(path);
    return rc;
}

int waste_model_state_load(waste_model *m, const char *path, int *pos)
{
    const waste_config *c = &m->cfg;
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    waste_state_hdr h, want;
    state_fill(m, &want, 0);
    if (fread(&h, sizeof h, 1, f) != 1) { fclose(f); return -1; }
    /* every shape must match; pos and n_blockres are the payload */
    if (h.magic != want.magic || h.version != want.version ||
        h.n_layers != want.n_layers || h.hidden != want.hidden ||
        h.kda_heads != want.kda_heads || h.kda_dim != want.kda_dim ||
        h.conv_k != want.conv_k || h.n_heads != want.n_heads ||
        h.qk_nope != want.qk_nope || h.qk_rope != want.qk_rope ||
        h.v_head != want.v_head || h.attn_res_block != want.attn_res_block) {
        fclose(f);
        return -2;                       /* state does not belong to this model */
    }

    const int H = c->kda_heads, D = c->kda_dim, C = H * D;
    const int qd = c->qk_nope + c->qk_rope;
    int rc = 0;
    for (int L = 0; L < c->n_layers && !rc; L++) {
        if (c->kda_layer[L]) {
            if (fread(m->S[L], sizeof(float), (size_t)H * D * D, f) != (size_t)H * D * D) rc = -1;
            const size_t cn = (size_t)3 * C * (c->conv_k - 1);
            if (!rc && fread(m->conv[L], sizeof(float), cn, f) != cn) rc = -1;
        } else {
            int32_t nkv = 0;
            if (fread(&nkv, sizeof nkv, 1, f) != 1) { rc = -1; break; }
            if (nkv < 0 || nkv > m->kv_cap) { rc = -1; break; }
            const size_t kn = (size_t)nkv * c->n_heads * qd;
            const size_t vn = (size_t)nkv * c->n_heads * c->v_head;
            if (kn && fread(m->kcache[L], sizeof(float), kn, f) != kn) rc = -1;
            if (!rc && vn && fread(m->vcache[L], sizeof(float), vn, f) != vn) rc = -1;
            m->n_kv[L] = nkv;
        }
    }
    m->n_blockres = h.n_blockres;
    if (!rc && c->attn_res_block && h.n_blockres > 0) {
        const size_t n = (size_t)h.n_blockres * c->hidden;
        if (fread(m->blockres, sizeof(float), n, f) != n) rc = -1;
    }
    if (!rc && fread(m->x, sizeof(float), (size_t)c->hidden, f) != (size_t)c->hidden) rc = -1;
    fclose(f);
    if (!rc && pos) *pos = h.pos;
    return rc;
}

/* ---- chunked prefill ---------------------------------------------------
 * Decode and prefill want opposite strategies for the experts.
 *
 * Decoding one token, an expert's weights are used for a single vector, so
 * expanding them is pure waste and the LUT wins: never dequantize.
 *
 * Prefilling a chunk, the same expert serves many tokens, so it pays to
 * expand it once and run a real GEMM — dense FMA runs ~50x faster than the
 * gathers the LUT needs, and the expansion amortizes over the chunk.
 *
 * The bigger win is upstream of that: tokens in a chunk route to
 * overlapping expert sets, so the union is far smaller than n * top_k, and
 * each distinct expert is read from disk exactly once.
 */

#define CHUNK_MAX 64

int waste_model_chunk_max(const waste_model *m) { (void)m; return CHUNK_MAX; }

/* Y[T][out] = X[T][in] . W^T, parallel over output rows. */
typedef struct {
    float *Y; const float *W, *X; int in, out, T;
} mm_arg;

static void mm_rows(int b, int e, void *p)
{
    mm_arg *a = (mm_arg *)p;
    for (int o = b; o < e; o++) {
        const float *row = a->W + (size_t)o * a->in;
        for (int t = 0; t < a->T; t++)
            a->Y[(size_t)t * a->out + o] = dotf(row, a->X + (size_t)t * a->in, a->in);
    }
}

static void matmul_f32(float *Y, const float *W, const float *X,
                       int out, int in, int T)
{
    mm_arg a = { Y, W, X, in, out, T };
    waste_parallel_for(out, 32, mm_rows, &a);
}

/* Tensor-aware batched matmul; dequantizes a Q8G row on the fly. */
typedef struct {
    float *Y; const int8_t *W; const uint16_t *ws; const float *X;
    int in, out, T, ng, group, bits;
} mmq_arg;

static void mmq_rows(int b, int e, void *p)
{
    mmq_arg *a = (mmq_arg *)p;
    const int g = a->group, ng = a->ng;
    float *row = (float *)malloc((size_t)a->in * sizeof(float));
    if (!row) return;
    for (int o = b; o < e; o++) {
        const int8_t *q = a->W + (size_t)o * ng * g * a->bits / 8;
        const uint16_t *sc = a->ws + (size_t)o * ng;
        for (int k = 0; k < ng; k++) {
            const float s = f16_to_f32(sc[k]);
            const int lim = (k * g + g <= a->in) ? g : a->in - k * g;
            if (a->bits == 4) {
                const uint8_t *p4 = (const uint8_t *)q + (size_t)k * g / 2;
                for (int i = 0; i < lim; i++) {
                    const uint8_t byte = p4[i / 2];
                    const int v = (i & 1) ? (byte >> 4) - 8 : (byte & 0x0F) - 8;
                    row[k * g + i] = (float)v * s;
                }
            } else {
                for (int i = 0; i < lim; i++)
                    row[k * g + i] = (float)q[(size_t)k * g + i] * s;
            }
        }
        for (int t = 0; t < a->T; t++)
            a->Y[(size_t)t * a->out + o] = dotf(row, a->X + (size_t)t * a->in, a->in);
    }
    free(row);
}

static void matmul_t(waste_model *m, float *Y, const waste_tensor *t,
                     const float *X, int out, int in, int T)
{
    (void)m;
    if (!t || (!t->q && !t->data)) {
        memset(Y, 0, (size_t)T * out * sizeof(float));
        return;
    }
    if (!t->q) { matmul_f32(Y, t->data, X, out, in, T); return; }
    const int g = t->group, ng = (in + g - 1) / g;
    mmq_arg a = { Y, t->q, t->qs, X, in, out, T, ng, g, t->bits };
    waste_parallel_for(out, 32, mmq_rows, &a);
}

/* Expand one expert's VQ indices into f32 [M][N]. */
static void vq_expand(waste_model *m, float *W, const uint8_t *idx,
                      const uint16_t *scale, int M, int N, int cb_base)
{
    const int nv_row = N / m->vec_dim, st = m->stages, en = m->cb_entries,
              vd = m->vec_dim;
    for (int r0 = 0; r0 < M; r0 += VQ_TILE) {
        const int nr = (r0 + VQ_TILE <= M) ? VQ_TILE : M - r0;
        const uint8_t *base = idx + (size_t)(r0 / VQ_TILE) * nv_row * VQ_TILE * st;
        for (int v = 0; v < nv_row; v++) {
            const uint8_t *ix = base + (size_t)v * VQ_TILE * st;
            for (int r = 0; r < nr; r++, ix += st) {
                float *dst = W + (size_t)(r0 + r) * N + (size_t)v * vd;
                for (int d = 0; d < vd; d++) dst[d] = 0.0f;
                for (int s = 0; s < st; s++) {
                    const float *C = m->codebooks +
                        ((size_t)(cb_base + s) * en + ix[s]) * vd;
                    for (int d = 0; d < vd; d++) dst[d] += C[d];
                }
            }
        }
    }
    for (int r = 0; r < M; r++) {
        const float sc = f16_to_f32(scale[r]);
        float *row = W + (size_t)r * N;
        for (int i = 0; i < N; i++) row[i] *= sc;
    }
}

/* ---- forward ----------------------------------------------------------- */

static int prefill_alloc(waste_model *m, int T)
{
    if (m->chunk_cap >= T) return 0;
    const waste_config *c = &m->cfg;
    const int hid = c->hidden, lat = c->latent_dim ? c->latent_dim : hid;
    const int wide = c->kda_heads * c->kda_dim;
    const int big = hid > wide ? hid : wide;
    const int inter = c->dense_inter > c->moe_inter ? c->dense_inter : c->moe_inter;
    const int nb = c->attn_res_block ? c->n_layers / c->attn_res_block + 2 : 1;

    free(m->cx); free(m->cnorm); free(m->cresid); free(m->cq); free(m->ckv);
    free(m->clat); free(m->cff); free(m->cexp); free(m->cblockres);
    free(m->cprefix); free(m->croute); free(m->crw);

    m->cx     = (float *)calloc((size_t)T * hid, sizeof(float));
    m->cnorm  = (float *)calloc((size_t)T * hid, sizeof(float));
    m->cresid = (float *)calloc((size_t)T * hid, sizeof(float));
    {   /* cq holds 2 LUTs per token plus one for `down` */
        const size_t lut_sz = (size_t)((hid > lat ? hid : lat) / 8) * 3 * 256;
        m->cq = (float *)calloc((size_t)(2 * T + 1) * lut_sz + 64, sizeof(float));
    }
    {   /* ckv holds the shared-expert staging: gate, up, out */
        const int si = c->moe_inter * (c->n_shared ? c->n_shared : 1);
        m->ckv = (float *)calloc((size_t)T * (size_t)(2 * si + hid) + 64, sizeof(float));
    }
    m->clat   = (float *)calloc((size_t)T * (size_t)(2 * lat + 2 * hid), sizeof(float));
    {   /* cff serves two callers: the per-token MoE staging (2*moe_inter +
         * lat) and the batched dense FFN (2 * T * dense_inter). Size for
         * whichever is larger — getting this wrong overflows silently. */
        const size_t moe_need = (size_t)2 * c->moe_inter + (size_t)(hid > lat ? hid : lat);
        const size_t dense_need = (size_t)2 * T * c->dense_inter;
        m->cff = (float *)calloc((moe_need > dense_need ? moe_need : dense_need) + 64,
                                 sizeof(float));
    }
    /* one expanded expert: gate/up [inter][lat] and down [hid][inter] */
    m->cexp   = (float *)calloc((size_t)2 * c->moe_inter * lat + (size_t)lat * c->moe_inter,
                                sizeof(float));
    m->cblockres = (float *)calloc((size_t)T * nb * hid, sizeof(float));
    m->cprefix   = (float *)calloc((size_t)T * hid, sizeof(float));
    m->croute = (int *)calloc((size_t)T * 64, sizeof(int));
    m->crw    = (float *)calloc((size_t)T * 64, sizeof(float));
    m->chunk_cap = T;
    return (m->cx && m->cnorm && m->cresid && m->cq && m->ckv && m->clat &&
            m->cff && m->cexp && m->cblockres && m->cprefix && m->croute &&
            m->crw) ? 0 : -1;
}

/* MoE over a whole chunk.
 *
 * The first attempt expanded each expert once and ran GEMMs, on the theory
 * that the cost amortizes over the chunk. Measured, it does not: a chunk of
 * 16 tokens spreads over ~1200 distinct experts, i.e. under 3 tokens each,
 * so expanding 7 M weights to serve 3 vectors is far worse than the LUT.
 *
 * What the chunk *does* buy is I/O: those 3328 (token, expert) pairs are
 * only 1210 distinct experts, so the disk sees 2.75x fewer reads. So keep
 * the decode-style LUT maths and reorganize purely to read each expert
 * once. gate/up tables depend on the token but not the expert (codebooks
 * are per layer), so they are built once per token and reused across every
 * expert that token routes to.
 */
static void moe_chunk(waste_model *m, int L, const float *in, float *out, int nT)
{
    const waste_config *c = &m->cfg;
    const int E = c->n_experts, K = c->top_k, hid = c->hidden;
    const int lat = c->latent_dim ? c->latent_dim : hid, inter = c->moe_inter;
    int *route = m->croute;
    float *rw = m->crw;

    float *sc = m->att + 4096, *score = sc + E;
    const float *bias = T(m, "%smodel.layers.%d.block_sparse_moe.gate.e_score_correction_bias",
                          c->prefix, L);
    for (int t = 0; t < nT; t++) {
        matvec_t(m, sc, waste_find(m, tname("%smodel.layers.%d.block_sparse_moe.gate.weight",
                                            c->prefix, L)), in + (size_t)t * hid, E, hid);
        for (int e = 0; e < E; e++) score[e] = 1.0f / (1.0f + expf(-sc[e]));
        int *idx = route + (size_t)t * K;
        float *w = rw + (size_t)t * K;
        for (int j = 0; j < K; j++) {
            int best = -1; float bv = -1e30f;
            for (int e = 0; e < E; e++) {
                int taken = 0;
                for (int p = 0; p < j; p++) if (idx[p] == e) { taken = 1; break; }
                if (taken) continue;
                const float v = score[e] + (bias ? bias[e] : 0.0f);
                if (v > bv) { bv = v; best = e; }
            }
            idx[j] = best; w[j] = score[best];
        }
        if (c->renorm && K > 1) {
            float s = 0;
            for (int j = 0; j < K; j++) s += w[j];
            for (int j = 0; j < K; j++) w[j] /= (s + 1e-20f);
        }
        for (int j = 0; j < K; j++) w[j] *= c->routed_scale;
    }

    const float *xin = in;
    float *lat_in = m->clat, *ysum = lat_in + (size_t)nT * lat;
    if (c->latent_dim) {
        matmul_t(m, lat_in, waste_find(m, tname(
                     "%smodel.layers.%d.block_sparse_moe.routed_expert_down_proj.weight",
                     c->prefix, L)), in, lat, hid, nT);
        xin = lat_in;
    }
    memset(ysum, 0, (size_t)nT * lat * sizeof(float));

    /* gate/up tables: one per token, shared by every expert it routes to */
    const int lut_sz = ((hid > lat ? hid : lat) / m->vec_dim) * m->stages * m->cb_entries;
    float *lut_gu = m->cq;                    /* [nT][2][lut_sz] */
    float *lut_down = lut_gu + (size_t)nT * 2 * lut_sz;
    int lut_ready = 0;

    float *ga = m->cff, *ub = ga + inter, *acc = m->cff + 2 * inter;

    for (int e = 0; e < E; e++) {
        int used = 0;
        for (int i = 0; i < nT * K; i++) if (route[i] == e) { used = 1; break; }
        if (!used) continue;

        PROF_START(P_EDEQ);
        const uint8_t *rec = read_expert(m, L, e);
        PROF_END(P_EDEQ);
        if (!rec) continue;
        const waste_expert_hdr *h = (const waste_expert_hdr *)rec;
        const uint16_t *s16 = (const uint16_t *)(rec + h->chan_corr_off);

        PROF_START(P_EMM);
        if (!lut_ready) {
            for (int t = 0; t < nT; t++) {
                vq_build_lut(lut_gu + (size_t)(2 * t) * lut_sz, m->codebooks,
                             h->codebook_id + 0 * m->stages,
                             xin + (size_t)t * lat, lat, m->stages,
                             m->cb_entries, m->vec_dim);
                vq_build_lut(lut_gu + (size_t)(2 * t + 1) * lut_sz, m->codebooks,
                             h->codebook_id + 1 * m->stages,
                             xin + (size_t)t * lat, lat, m->stages,
                             m->cb_entries, m->vec_dim);
            }
            lut_ready = 1;
        }
        for (int t = 0; t < nT; t++) {
            float wj = 0;
            for (int j = 0; j < K; j++)
                if (route[(size_t)t * K + j] == e) { wj = rw[(size_t)t * K + j]; break; }
            if (wj == 0.0f) continue;

            vq_apply(m, ga, rec + h->gate_off, s16, inter, lat,
                     lut_gu + (size_t)(2 * t) * lut_sz);
            vq_apply(m, ub, rec + h->up_off, s16 + inter, inter, lat,
                     lut_gu + (size_t)(2 * t + 1) * lut_sz);
            if (c->act_situ)
                for (int i = 0; i < inter; i++)
                    ga[i] = waste_situ_pair(ga[i], ub[i], c->situ_beta, c->situ_linear_beta);
            else
                for (int i = 0; i < inter; i++) ga[i] = silu(ga[i]) * ub[i];
            vq_matvec(m, acc, rec + h->down_off, s16 + 2 * inter, ga, lat, inter,
                      h->codebook_id + 2 * m->stages, lut_down);
            float *dst = ysum + (size_t)t * lat;
            for (int i = 0; i < lat; i++) dst[i] += wj * acc[i];
        }
        PROF_END(P_EMM);
    }

    if (c->latent_dim) {
        if (c->latent_norm) {
            const float *nw = waste_find(m, tname(
                "%smodel.layers.%d.block_sparse_moe.routed_expert_norm.weight",
                c->prefix, L))->data;
            for (int t = 0; t < nT; t++)
                rmsnorm(ysum + (size_t)t * lat, ysum + (size_t)t * lat, nw, lat, c->eps);
        }
        matmul_t(m, out, waste_find(m, tname(
                     "%smodel.layers.%d.block_sparse_moe.routed_expert_up_proj.weight",
                     c->prefix, L)), ysum, hid, lat, nT);
    } else {
        memcpy(out, ysum, (size_t)nT * hid * sizeof(float));
    }

    /* shared experts, on the full hidden state */
    const int si = inter * (c->n_shared ? c->n_shared : 1);
    float *sa = m->ckv, *sb = sa + (size_t)nT * si, *sh = sb + (size_t)nT * si;
    matmul_t(m, sa, waste_find(m, tname(
                 "%smodel.layers.%d.block_sparse_moe.shared_experts.gate_proj.weight",
                 c->prefix, L)), in, si, hid, nT);
    matmul_t(m, sb, waste_find(m, tname(
                 "%smodel.layers.%d.block_sparse_moe.shared_experts.up_proj.weight",
                 c->prefix, L)), in, si, hid, nT);
    for (int i = 0; i < nT * si; i++)
        sa[i] = c->act_situ ? waste_situ_pair(sa[i], sb[i], c->situ_beta, c->situ_linear_beta)
                            : silu(sa[i]) * sb[i];
    matmul_t(m, sh, waste_find(m, tname(
                 "%smodel.layers.%d.block_sparse_moe.shared_experts.down_proj.weight",
                 c->prefix, L)), sa, hid, si, nT);
    for (int i = 0; i < nT * hid; i++) out[i] += sh[i];
}

/* Prefill a chunk. KDA and MLA still walk the tokens in order — the
 * recurrence and the causal mask demand it, and both are cheap — but every
 * projection is a GEMM and the MoE sees the whole chunk at once. */
const float *waste_model_prefill(waste_model *m, const int *tokens, int n,
                                 int pos0)
{
    const waste_config *c = &m->cfg;
    const int hid = c->hidden;
    if (n <= 0) return m->logits;
    if (n == 1) return waste_model_step(m, tokens[0], pos0, NULL);
    if (n > CHUNK_MAX) n = CHUNK_MAX;
    if (prefill_alloc(m, n)) return NULL;

    const waste_tensor *emb = waste_find(m, tname("%smodel.embed_tokens.weight", c->prefix));
    for (int t = 0; t < n; t++) {
        float *dst = m->cx + (size_t)t * hid;
        if (emb->data) memcpy(dst, emb->data + (size_t)tokens[t] * hid, (size_t)hid * 4);
        else {
            const int g = emb->group, ng = (hid + g - 1) / g;
            const int8_t *row = emb->q + (size_t)tokens[t] * ng * g * emb->bits / 8;
            const uint16_t *sc = emb->qs + (size_t)tokens[t] * ng;
            for (int k = 0; k < ng; k++) {
                const float sv = f16_to_f32(sc[k]);
                for (int i = 0; i < g && k * g + i < hid; i++) {
                    int v;
                    if (emb->bits == 4) {
                        const uint8_t byte = ((const uint8_t *)row)[(k * g + i) / 2];
                        v = (i & 1) ? (byte >> 4) - 8 : (byte & 0x0F) - 8;
                    } else v = row[k * g + i];
                    dst[k * g + i] = (float)v * sv;
                }
            }
        }
    }

    const int ares_on = c->attn_res_block > 0;
    int nb = 0;
    int ps_live = 0;

    for (int L = 0; L < c->n_layers; L++) {
        if (ares_on) {
            memcpy(m->cprefix, m->cx, (size_t)n * hid * sizeof(float));
            ps_live = 1;
            if (nb > 0) {
                const float *nw = waste_find(m, tname("%smodel.layers.%d.self_attention_res_norm.weight", c->prefix, L))->data;
                const float *pw = waste_find(m, tname("%smodel.layers.%d.self_attention_res_proj.weight", c->prefix, L))->data;
                const int stride = (c->n_layers / c->attn_res_block + 2) * hid;
                for (int t = 0; t < n; t++)
                    waste_apply_attn_res(m, m->cblockres + (size_t)t * stride, nb,
                                         m->cprefix + (size_t)t * hid, nw, pw,
                                         m->cx + (size_t)t * hid);
            }
            if (L % c->attn_res_block == 0) {
                const int stride = (c->n_layers / c->attn_res_block + 2) * hid;
                for (int t = 0; t < n; t++)
                    memcpy(m->cblockres + (size_t)t * stride + (size_t)nb * hid,
                           m->cprefix + (size_t)t * hid, (size_t)hid * sizeof(float));
                nb++;
                ps_live = 0;
            }
        }

        const float *iln = waste_find(m, tname("%smodel.layers.%d.input_layernorm.weight", c->prefix, L))->data;
        for (int t = 0; t < n; t++)
            rmsnorm(m->cnorm + (size_t)t * hid, m->cx + (size_t)t * hid, iln, hid, c->eps);

        /* attention: per token, but on the batched norm buffer */
        for (int t = 0; t < n; t++) {
            if (c->kda_layer[L]) kda_layer(m, L, m->cnorm + (size_t)t * hid,
                                           m->cresid + (size_t)t * hid);
            else mla_layer(m, L, m->cnorm + (size_t)t * hid,
                           m->cresid + (size_t)t * hid, pos0 + t);
        }

        if (ares_on) {
            if (ps_live) for (int i = 0; i < n * hid; i++) m->cprefix[i] += m->cresid[i];
            else { memcpy(m->cprefix, m->cresid, (size_t)n * hid * sizeof(float)); ps_live = 1; }
            const float *nw = waste_find(m, tname("%smodel.layers.%d.mlp_res_norm.weight", c->prefix, L))->data;
            const float *pw = waste_find(m, tname("%smodel.layers.%d.mlp_res_proj.weight", c->prefix, L))->data;
            const int stride = (c->n_layers / c->attn_res_block + 2) * hid;
            for (int t = 0; t < n; t++)
                waste_apply_attn_res(m, m->cblockres + (size_t)t * stride, nb,
                                     m->cprefix + (size_t)t * hid, nw, pw,
                                     m->cx + (size_t)t * hid);
        } else {
            for (int i = 0; i < n * hid; i++) m->cx[i] += m->cresid[i];
        }

        const float *pln = waste_find(m, tname("%smodel.layers.%d.post_attention_layernorm.weight", c->prefix, L))->data;
        for (int t = 0; t < n; t++)
            rmsnorm(m->cnorm + (size_t)t * hid, m->cx + (size_t)t * hid, pln, hid, c->eps);

        if (waste_find(m, tname("%smodel.layers.%d.block_sparse_moe.gate.weight", c->prefix, L))) {
            PROF_START(P_ROUTE);
            moe_chunk(m, L, m->cnorm, m->cresid, n);
            PROF_END(P_ROUTE);
        } else {
            const int inter = c->dense_inter;
            float *a = m->cff, *b = a + (size_t)n * inter;
            matmul_t(m, a, waste_find(m, tname("%smodel.layers.%d.mlp.gate_proj.weight", c->prefix, L)), m->cnorm, inter, hid, n);
            matmul_t(m, b, waste_find(m, tname("%smodel.layers.%d.mlp.up_proj.weight", c->prefix, L)), m->cnorm, inter, hid, n);
            for (int i = 0; i < n * inter; i++)
                a[i] = c->act_situ ? waste_situ_pair(a[i], b[i], c->situ_beta, c->situ_linear_beta)
                                   : silu(a[i]) * b[i];
            matmul_t(m, m->cresid, waste_find(m, tname("%smodel.layers.%d.mlp.down_proj.weight", c->prefix, L)), a, hid, inter, n);
        }

        if (ares_on) {
            for (int i = 0; i < n * hid; i++) m->cprefix[i] += m->cresid[i];
            memcpy(m->cx, m->cprefix, (size_t)n * hid * sizeof(float));
        } else {
            for (int i = 0; i < n * hid; i++) m->cx[i] += m->cresid[i];
        }
    }

    /* hand the chunk's block-residual history to the per-token path: decode
     * continues from the last token's row */
    if (ares_on) {
        const int stride = (c->n_layers / c->attn_res_block + 2) * hid;
        memcpy(m->blockres, m->cblockres + (size_t)(n - 1) * stride,
               (size_t)nb * hid * sizeof(float));
    }
    m->n_blockres = nb;
    const float *fnw = waste_find(m, tname("%smodel.norm.weight", c->prefix))->data;
    float *last = m->cnorm;
    rmsnorm(last, m->cx + (size_t)(n - 1) * hid, fnw, hid, c->eps);
    matvec_t(m, m->logits, waste_find(m, tname("%slm_head.weight", c->prefix)), last,
             c->vocab, hid);
    memcpy(m->x, m->cx + (size_t)(n - 1) * hid, (size_t)hid * sizeof(float));
    return m->logits;
}

const float *waste_model_step(waste_model *m, int token, int pos, int *routed)
{
    const waste_config *c = &m->cfg;
    const int hid = c->hidden;
    /* one embedding row; the table may be kept quantized */
    const waste_tensor *emb = waste_find(m, tname("%smodel.embed_tokens.weight", c->prefix));
    if (emb->data) {
        memcpy(m->x, emb->data + (size_t)token * hid, (size_t)hid * sizeof(float));
    } else {
        const int g = emb->group, ng = (hid + g - 1) / g;
        const int8_t *row = emb->q + (size_t)token * ng * g * emb->bits / 8;
        const uint16_t *sc = emb->qs + (size_t)token * ng;
        for (int k = 0; k < ng; k++) {
            const float s = f16_to_f32(sc[k]);
            for (int i = 0; i < g && k * g + i < hid; i++) {
                int v;
                if (emb->bits == 4) {
                    const uint8_t byte = ((const uint8_t *)row)[(k * g + i) / 2];
                    v = (i & 1) ? (byte >> 4) - 8 : (byte & 0x0F) - 8;
                } else v = row[k * g + i];
                m->x[k * g + i] = (float)v * s;
            }
        }
    }

    float *resid = (float *)malloc((size_t)hid * sizeof(float));
    float *norm = (float *)malloc((size_t)hid * sizeof(float));
    const int ares_on = c->attn_res_block > 0;
    float *ps = m->prefix_sum;
    int ps_live = 0;
    m->n_blockres = 0;

    for (int L = 0; L < c->n_layers; L++) {
        char b[128];
        if (ares_on) {
            memcpy(ps, m->x, (size_t)hid * sizeof(float));
            ps_live = 1;
            if (m->n_blockres > 0) {
                waste_apply_attn_res(m, m->blockres, m->n_blockres, ps,
                    waste_find(m, tname("%smodel.layers.%d.self_attention_res_norm.weight", c->prefix, L))->data,
                    waste_find(m, tname("%smodel.layers.%d.self_attention_res_proj.weight", c->prefix, L))->data,
                    m->x);
            }
            if (L % c->attn_res_block == 0) {
                memcpy(m->blockres + (size_t)m->n_blockres * hid, ps,
                       (size_t)hid * sizeof(float));
                m->n_blockres++;
                ps_live = 0;
            }
        }

        snprintf(b, sizeof b, "%smodel.layers.%d.input_layernorm.weight", c->prefix, L);
        rmsnorm(norm, m->x, waste_find(m, b)->data, hid, c->eps);
        if (c->kda_layer[L]) { PROF_START(P_KDA); kda_layer(m, L, norm, resid); PROF_END(P_KDA); }
        else { PROF_START(P_MLA); mla_layer(m, L, norm, resid, pos); PROF_END(P_MLA); }

        if (ares_on) {
            if (ps_live) for (int i = 0; i < hid; i++) ps[i] += resid[i];
            else { memcpy(ps, resid, (size_t)hid * sizeof(float)); ps_live = 1; }
            waste_apply_attn_res(m, m->blockres, m->n_blockres, ps,
                waste_find(m, tname("%smodel.layers.%d.mlp_res_norm.weight", c->prefix, L))->data,
                waste_find(m, tname("%smodel.layers.%d.mlp_res_proj.weight", c->prefix, L))->data,
                m->x);
        } else {
            for (int i = 0; i < hid; i++) m->x[i] += resid[i];
        }

        snprintf(b, sizeof b, "%smodel.layers.%d.post_attention_layernorm.weight", c->prefix, L);
        rmsnorm(norm, m->x, waste_find(m, b)->data, hid, c->eps);
        snprintf(b, sizeof b, "%smodel.layers.%d.block_sparse_moe.gate.weight", c->prefix, L);
        if (waste_find(m, b)) {
            PROF_START(P_ROUTE);
            moe_layer(m, L, norm, resid, routed ? routed + (size_t)L * c->top_k : NULL);
            PROF_END(P_ROUTE);
        }
        else
            ffn(m, waste_find(m, tname("%smodel.layers.%d.mlp.gate_proj.weight", c->prefix, L)),
                waste_find(m, tname("%smodel.layers.%d.mlp.up_proj.weight", c->prefix, L)),
                waste_find(m, tname("%smodel.layers.%d.mlp.down_proj.weight", c->prefix, L)),
                norm, resid, c->dense_inter, hid, 1.0f, 0);

        if (ares_on) {
            for (int i = 0; i < hid; i++) ps[i] += resid[i];
            memcpy(m->x, ps, (size_t)hid * sizeof(float));
        } else {
            for (int i = 0; i < hid; i++) m->x[i] += resid[i];
        }
    }
    rmsnorm(norm, m->x, waste_find(m, tname("%smodel.norm.weight", c->prefix))->data, hid, c->eps);
    PROF_START(P_HEAD);
    matvec_t(m, m->logits, waste_find(m, tname("%slm_head.weight", c->prefix)), norm, c->vocab, hid);
    PROF_END(P_HEAD);
    free(resid);
    free(norm);
    return m->logits;
}
