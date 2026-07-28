/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/*
 * waste.c — the public API, implemented over model.c.
 *
 * Everything the CLI can do goes through here; the CLI links this and
 * nothing private. Errors are returned, never printed, and nothing calls
 * exit().
 */

#include "waste.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif
#include <time.h>

#include "json.h"
#include "model.h"
#include "tokenizer.h"
#include "waste_backend.h"

/* Enough for any prompt a person assembles by hand; a batch host that
 * wants more should run more contexts. */
#define WASTE_MAX_IMAGES 32

struct waste_ctx {
    waste_model m;
    waste_tok *tok;
    waste_cfg cfg;
    waste_memplan plan;
    char path[512];
    int pos;                 /* next position in the sequence */
    int warmed;              /* experts preloaded from the hotlist */
    char quant[64];          /* composed at open, reported by get_info */
    waste_stats stats;

    /* Queued image embeddings, concatenated: img_each[] is how many rows
     * each queued image contributed, which is what expand needs to know
     * to size each placeholder's run. */
    float   *img;
    size_t   img_rows;
    size_t   img_each[WASTE_MAX_IMAGES];
    int      img_n;
};



/* What the container is actually stored as, composed once at open. It used
 * to be one string constant, which was true of the first model converted
 * and wrong about the second: K3's trunk is mostly Q4G where Kimi-Linear's
 * is Q8G. Narrowest quantization first, f32 last. */
static void quant_summary(waste_ctx *c)
{
    static const struct { int fmt; const char *name; } tbl[] = {
        { 7, "Q3G" }, { 3, "Q4G" }, { 2, "Q8G" }, { 1, "F16" }, { 0, "F32" },
    };
    int n = snprintf(c->quant, sizeof c->quant, "experts VQ%dR, trunk",
                     c->m.stages);
    size_t off = (n > 0 && (size_t)n < sizeof c->quant) ? (size_t)n : 0;
    const char *sep = " ";
    for (size_t i = 0; i < sizeof tbl / sizeof *tbl; i++) {
        if (!(c->m.trunk_fmts & (1u << tbl[i].fmt))) continue;
        n = snprintf(c->quant + off, sizeof c->quant - off, "%s%s",
                     sep, tbl[i].name);
        if (n < 0 || (size_t)n >= sizeof c->quant - off) break;
        off += (size_t)n;
        sep = "/";
    }
}

const char *waste_strerror(waste_status s)
{
    switch (s) {
    case WASTE_OK:             return "ok";
    case WASTE_E_IO:           return "I/O error";
    case WASTE_E_FORMAT:       return "malformed container";
    case WASTE_E_RAM_BUDGET:   return "RAM budget below the model's floor";
    case WASTE_E_OOM:          return "out of memory";
    case WASTE_E_ARG:          return "invalid argument";
    case WASTE_E_UNSUPPORTED:  return "unsupported";
    case WASTE_E_CANCELLED:    return "cancelled by callback";
    }
    return "unknown error";
}

void waste_cfg_init(waste_cfg *cfg)
{
    if (!cfg) return;
    memset(cfg, 0, sizeof *cfg);
    cfg->cache_policy = WASTE_CACHE_LFRU;
    cfg->use_direct_io = 1;
    cfg->ctx_tokens = 4096;
}

void waste_gen_params_init(waste_gen_params *p)
{
    if (!p) return;
    memset(p, 0, sizeof *p);
    p->temperature = 0.0f;      /* greedy */
    p->top_p = 1.0f;
    p->max_tokens = 256;
}

static double nowf(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
}

/* ---- memory planning ---------------------------------------------------- */

static char *slurp_all(const char *path)
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
    return b;
}

/* Physical RAM, or 0 when it cannot be determined. */
uint64_t waste_physical_ram(void)
{
#if defined(__APPLE__)
    uint64_t v = 0;
    size_t n = sizeof v;
    if (sysctlbyname("hw.memsize", &v, &n, NULL, 0) == 0) return v;
    return 0;
#elif defined(__linux__)
    const long p = sysconf(_SC_PHYS_PAGES), z = sysconf(_SC_PAGESIZE);
    return (p > 0 && z > 0) ? (uint64_t)p * (uint64_t)z : 0;
#else
    return 0;
#endif
}

waste_status waste_plan_memory(const char *model_path, uint32_t ctx_tokens,
                               waste_memplan *out)
{
    if (!model_path || !out) return WASTE_E_ARG;
    char p[512];
    snprintf(p, sizeof p, "%s/manifest.json", model_path);
    char *src = slurp_all(p);
    if (!src) return WASTE_E_IO;
    js_doc d;
    if (js_parse(&d, src) < 0) { free(src); return WASTE_E_FORMAT; }

    memset(out, 0, sizeof *out);

    /* trunk: sum the tensor payloads as they are stored, minus the ones the
     * engine deliberately leaves on disk (embed_tokens — one row per token) */
    const int trunk = js_get(&d, 0, "trunk");
    char prefix[64];
    js_str(&d, js_get(&d, 0, "tensor_prefix"), prefix, sizeof prefix);
    for (int i = 0; i < js_size(&d, trunk); i++) {
        const int e = js_at(&d, trunk, i);
        char nm[160];
        js_str(&d, js_get(&d, e, "name"), nm, sizeof nm);
        const int fmt = (int)js_int(&d, js_get(&d, e, "fmt"), 0);
        if (fmt != 0 && strstr(nm, "embed_tokens.weight")) continue;
        const uint64_t nb = (uint64_t)js_int(&d, js_get(&d, e, "bytes"), 0);
        /* The vision tower and projector sit outside tensor_prefix. They
         * are loaded only when a caller asks for images, so they are
         * counted apart and folded in by waste_open — counting them here
         * would overstate the floor for every text-only run, and leaving
         * them out entirely understated it for every run with one. */
        if (prefix[0] && strncmp(nm, prefix, strlen(prefix)) != 0) {
            out->vision_bytes += nb;
            continue;
        }
        out->trunk_bytes += nb;
    }

    const int cfg = js_get(&d, 0, "config");
    const int layers = (int)js_int(&d, js_get(&d, cfg, "num_hidden_layers"), 0);
    const int hidden = (int)js_int(&d, js_get(&d, cfg, "hidden_size"), 0);
    const int nheads = (int)js_int(&d, js_get(&d, cfg, "num_attention_heads"), 0);
    const int kv_lora = (int)js_int(&d, js_get(&d, cfg, "kv_lora_rank"), 0);
    const int qk_rope = (int)js_int(&d, js_get(&d, cfg, "qk_rope_head_dim"), 0);
    const int qk_nope = (int)js_int(&d, js_get(&d, cfg, "qk_nope_head_dim"), 0);
    const int v_head = (int)js_int(&d, js_get(&d, cfg, "v_head_dim"), 0);
    const int lac = js_get(&d, cfg, "linear_attn_config");
    const int kh = (int)js_int(&d, js_get(&d, lac, "num_heads"), 0);
    const int kd = (int)js_int(&d, js_get(&d, lac, "head_dim"), 0);
    const int ck = (int)js_int(&d, js_get(&d, lac, "short_conv_kernel_size"), 4);
    const int kl = js_get(&d, lac, "kda_layers");
    const int n_kda = js_size(&d, kl);
    const int n_mla = layers - n_kda;

    /* MLA caches the latent plus the rope dims, not the expanded per-head
     * K and V — kv_b_proj is absorbed into the query and the output. That
     * is 576 floats per token per layer here rather than 30720. */
    out->state_bytes = (uint64_t)n_kda * kh * kd * kd * 4                /* S */
                     + (uint64_t)n_kda * 3 * (ck - 1) * kh * kd * 4      /* conv */
                     + (uint64_t)n_mla * ctx_tokens *
                       ((uint64_t)kv_lora + qk_rope) * 4;                /* KV */
    (void)qk_nope; (void)v_head;

    /* Scratch, counted rather than guessed. The old flat 64 MB was out by
     * 4x on the decode buffers alone (e_gate/e_up/e_down are 252 MB on K3)
     * and ignored the chunked-prefill buffers entirely, which is why peak
     * RSS ran over the budget near the floor. */
    const int moe_inter = (int)js_int(&d, js_get(&d, cfg, "moe_intermediate_size"), 0);
    const int dense_inter = (int)js_int(&d, js_get(&d, cfg, "intermediate_size"), moe_inter);
    const int vocab = (int)js_int(&d, js_get(&d, cfg, "vocab_size"), 0);
    const int lat = (int)js_int(&d, js_get(&d, cfg, "routed_expert_hidden_size"), hidden);
    const int n_shared = (int)js_int(&d, js_get(&d, cfg, "num_shared_experts"), 1);
    const int ares = (int)js_int(&d, js_get(&d, cfg, "attn_res_block_size"), 0);
    const int nb = ares ? layers / ares + 2 : 1;
    const int big = hidden > kh * kd ? hidden : kh * kd;
    const int wide = hidden > lat ? hidden : lat;
    const int T = WASTE_CHUNK_MAX;

    uint64_t sc = 0;
    sc += (uint64_t)3 * moe_inter * hidden * 4;             /* expert staging  */
    sc += (uint64_t)2 * (dense_inter > moe_inter ? dense_inter : moe_inter) * 4;
    sc += ((uint64_t)ctx_tokens * nheads + 1024) * 4;       /* attention rows  */
    sc += (uint64_t)vocab * 4;                              /* logits          */
    sc += ((uint64_t)8 * big + 8 * moe_inter + 8 * dense_inter + 512) * 4;
    sc += (uint64_t)3 * nheads * (kv_lora ? kv_lora : 1) * 4;   /* MLA absorb  */
    sc += (uint64_t)(nb + 4) * hidden * 4;                  /* AttnRes buffers */
    /* chunked prefill, allocated on first use and never freed */
    sc += (uint64_t)T * hidden * 4 * 3;                     /* cx/cnorm/cresid */
    sc += ((uint64_t)(2 * T + 1) * (wide / 8) * 3 * 256 + 64) * 4;   /* LUTs   */
    sc += ((uint64_t)T * (2 * moe_inter * n_shared + hidden) + 64) * 4;
    sc += (uint64_t)T * (2 * lat + 2 * hidden) * 4;
    sc += (uint64_t)2 * T * dense_inter * 4;
    sc += (uint64_t)3 * moe_inter * lat * 4;                /* one expert      */
    sc += (uint64_t)T * nb * hidden * 4 + (uint64_t)T * hidden * 4;
    sc += (uint64_t)T * 64 * 8;
    out->scratch_bytes = sc;

    /* one layer's top-k experts, double buffered */
    const int top_k = (int)js_int(&d, js_get(&d, cfg, "num_experts_per_token"), 8);
    const int lyr = js_get(&d, 0, "layers");
    uint64_t rec = 0;
    if (js_size(&d, lyr) > 0) {
        const int first = lyr + 2;   /* first member's value */
        const uint64_t bytes = (uint64_t)js_int(&d, js_get(&d, first, "bytes"), 0);
        const uint64_t n = (uint64_t)js_int(&d, js_get(&d, first, "experts"), 1);
        rec = n ? bytes / n : 0;
    }
    out->min_expert_cache = rec * (uint64_t)top_k * 2;
    out->floor_bytes = out->trunk_bytes + out->state_bytes +
                       out->scratch_bytes + out->min_expert_cache;
    /* A cache below one token's working set keeps nothing alive to the
     * next token (Gate 5), so "recommended" starts at 3x that. */
    out->recommended_bytes = out->floor_bytes +
                             rec * (uint64_t)top_k * (uint64_t)layers * 3;

    js_free(&d);
    free(src);
    return WASTE_OK;
}

/* ---- lifecycle ---------------------------------------------------------- */

waste_status waste_open(const char *model_path, const waste_cfg *cfg_in,
                        waste_ctx **out)
{
    if (!model_path || !out) return WASTE_E_ARG;
    *out = NULL;

    waste_cfg cfg;
    if (cfg_in) cfg = *cfg_in;
    else waste_cfg_init(&cfg);
    if (!cfg.ctx_tokens) cfg.ctx_tokens = 4096;

    waste_ctx *c = (waste_ctx *)calloc(1, sizeof *c);
    if (!c) return WASTE_E_OOM;
    c->cfg = cfg;
    snprintf(c->path, sizeof c->path, "%s", model_path);

    waste_status st = waste_plan_memory(model_path, cfg.ctx_tokens, &c->plan);
    if (st != WASTE_OK) { free(c); return st; }

    /* The tower is real memory the moment it is asked for, so it has to be
     * in the floor before the budget is resolved against it. */
    if (cfg.vision && c->plan.vision_bytes) {
        c->plan.trunk_bytes += c->plan.vision_bytes;
        c->plan.floor_bytes += c->plan.vision_bytes;
        c->plan.recommended_bytes += c->plan.vision_bytes;
    }

    /* Resolve the budget. A budget the caller set is a contract and is used
     * as given. A zero means "you decide", and that decision has to know the
     * machine: recommended_bytes comes from the model alone — 80.85 GB on K3
     * — so defaulting to it on a 64 GB laptop would ask for 51 GB of expert
     * cache and swap, which is the one thing the budget exists to prevent.
     * Cap the default at what the machine can hold, never below the floor. */
    const uint64_t phys = waste_physical_ram();
    const uint64_t cap = phys ? phys - phys / 8 : 0;   /* 12% left to the OS */
    uint64_t budget = cfg.ram_budget_bytes;

    if (!budget) {
        budget = c->plan.recommended_bytes;
        if (cap && budget > cap)
            budget = cap > c->plan.floor_bytes ? cap : c->plan.floor_bytes;
    }
    if (budget < c->plan.floor_bytes) { free(c); return WASTE_E_RAM_BUDGET; }

    /* A budget close to physical RAM backfires: the OS starts paging out
     * the engine's own expert cache, and a "hit" then costs a page fault
     * instead of the disk read the engine was managing. Measured on K3:
     * 29.1 GB of cache on a 64 GB machine ran at 0.04 tok/s against 0.32
     * with 28.0 GB — a better hit rate and eight times slower. This tests
     * the resolved budget, not the caller's field: the default used to skip
     * the check entirely by being zero here, which is exactly the case that
     * needed it. What survives the clamp above is a floor that does not fit
     * this machine — worth saying out loud before it crawls. */
    if (cap && budget > cap)
        fprintf(stderr,
                "waste: budget %.1f GB leaves under 12%% of this machine's "
                "%.1f GB free\n       the OS will page out the expert cache "
                "and throughput collapses\n",
                budget / 1073741824.0, phys / 1073741824.0);

    c->cfg.ram_budget_bytes = budget;   /* what we actually run under */

    const uint64_t cache_bytes = budget - c->plan.floor_bytes +
                                 c->plan.min_expert_cache;
    {
        const int rc = waste_model_load(&c->m, model_path, (int)cfg.ctx_tokens,
                                        (size_t)cache_bytes, cfg.vision);
        if (rc) {
            free(c);
            return rc == -2 ? WASTE_E_FORMAT : WASTE_E_IO;
        }
    }
    quant_summary(c);
    c->tok = waste_tok_open(model_path);      /* optional */
    /* warm the cache from what previous runs learned, if anything */
    c->warmed = waste_model_warm_cache(&c->m, model_path);
    *out = c;
    return WASTE_OK;
}

void waste_close(waste_ctx *c)
{
    if (!c) return;
    waste_model_free(&c->m);
    waste_tok_free(c->tok);
    free(c->img);
    free(c);
}

waste_status waste_memory_used(const waste_ctx *c, waste_memplan *out)
{
    if (!c || !out) return WASTE_E_ARG;
    *out = c->plan;
    out->min_expert_cache = (uint64_t)c->m.cache.n_slots * c->m.cache.rec_bytes;
    return WASTE_OK;
}

/* ---- tokenizer ---------------------------------------------------------- */

waste_status waste_tokenize(waste_ctx *c, const char *text, int add_bos,
                            int32_t *out, size_t cap, size_t *n_out)
{
    if (!c || !text || !out || !n_out) return WASTE_E_ARG;
    if (!c->tok) return WASTE_E_UNSUPPORTED;
    size_t n = 0;
    if (add_bos && cap > 0) out[n++] = (int32_t)waste_tok_bos(c->tok);
    const int got = waste_tok_encode(c->tok, text, out + n, (int)(cap - n));
    if (got < 0) return WASTE_E_ARG;
    *n_out = n + (size_t)got;
    return WASTE_OK;
}

waste_status waste_detokenize(waste_ctx *c, const int32_t *ids, size_t n,
                              char *out, size_t cap, size_t *n_out)
{
    if (!c || !ids || !out || !n_out) return WASTE_E_ARG;
    if (!c->tok) return WASTE_E_UNSUPPORTED;
    const int got = waste_tok_decode(c->tok, ids, (int)n, out, (int)cap - 1);
    out[got < 0 ? 0 : got] = 0;
    *n_out = (size_t)(got < 0 ? 0 : got);
    return WASTE_OK;
}

/* ---- generation --------------------------------------------------------- */

/* Token ids come from the caller and index the embedding table directly,
 * so an out-of-range one is an out-of-bounds read, not a wrong answer.
 * Nothing checked them until a synthetic container with a 256-entry
 * vocabulary was fed ids from a 163840-entry one and quietly returned
 * whatever was past the end of the table. */
static waste_status check_ids(const waste_ctx *c, const int32_t *ids, size_t n)
{
    const int32_t vocab = (int32_t)c->m.cfg.vocab;
    for (size_t i = 0; i < n; i++)
        if (ids[i] < 0 || ids[i] >= vocab) return WASTE_E_ARG;
    return WASTE_OK;
}


/* ---- images ------------------------------------------------------------- */

void waste_image_clear(waste_ctx *c)
{
    if (!c) return;
    free(c->img);
    c->img = NULL;
    c->img_rows = 0;
    c->img_n = 0;
}

waste_status waste_image_add(waste_ctx *c, const char *path, size_t *n_out)
{
    if (!c || !path) return WASTE_E_ARG;
    if (!c->m.want_vision || !c->m.vcfg.layers) return WASTE_E_UNSUPPORTED;
    if (c->img_n >= WASTE_MAX_IMAGES) return WASTE_E_ARG;

    int gh = 0, gw = 0;
    float *px = waste_image_load(path, c->m.vcfg.max_patches,
                                 c->m.vcfg.mean, c->m.vcfg.std, &gh, &gw);
    if (!px) return WASTE_E_IO;

    const size_t rows = (size_t)(gh / 2) * (size_t)(gw / 2);
    const size_t th = (size_t)c->m.vcfg.text_hidden;
    float *nb = (float *)realloc(c->img, (c->img_rows + rows) * th * sizeof(float));
    if (!nb) { free(px); return WASTE_E_OOM; }
    c->img = nb;

    const int rc = waste_vision_encode(&c->m, px, gh, gw,
                                       c->img + c->img_rows * th);
    free(px);
    if (rc) return WASTE_E_IO;

    c->img_each[c->img_n++] = rows;
    c->img_rows += rows;
    if (n_out) *n_out = rows;
    return WASTE_OK;
}

waste_status waste_image_expand(waste_ctx *c, const int32_t *tokens, size_t n,
                                int32_t *out, size_t cap, size_t *n_out)
{
    if (!c || !tokens || !out) return WASTE_E_ARG;
    if (!c->m.want_vision || !c->m.vcfg.layers) return WASTE_E_UNSUPPORTED;
    const int32_t pad = (int32_t)c->m.vcfg.media_token;

    size_t seen = 0;
    for (size_t i = 0; i < n; i++) if (tokens[i] == pad) seen++;
    /* A mismatch is a prompt-assembly bug, and a silent one would show up
     * much later as an image the model never saw. Refuse it here. */
    if (seen != (size_t)c->img_n) return WASTE_E_ARG;

    size_t k = 0, img = 0;
    for (size_t i = 0; i < n; i++) {
        const size_t rep = (tokens[i] == pad) ? c->img_each[img++] : 1;
        if (k + rep > cap) { if (n_out) *n_out = 0; return WASTE_E_ARG; }
        for (size_t r = 0; r < rep; r++) out[k++] = tokens[i];
    }
    if (n_out) *n_out = k;
    return WASTE_OK;
}

/* The model reads embeddings from m.media as it walks the prompt, so the
 * queue has to be armed before any prefill and released after, or a
 * follow-up turn would splice the same picture in again. */
static void media_arm(waste_ctx *c)
{
    c->m.media = c->img;
    c->m.media_n = (int)c->img_rows;
    c->m.media_used = 0;
    c->m.cfg_media_token = c->m.vcfg.media_token;
}

static void media_release(waste_ctx *c)
{
    if (c->m.media_used > 0) waste_image_clear(c);
    c->m.media = NULL;
    c->m.media_n = c->m.media_used = 0;
}

waste_status waste_eval(waste_ctx *c, const int32_t *tokens, size_t n,
                        const float **logits_out, size_t *vocab_out)
{
    if (!c || !tokens || !n) return WASTE_E_ARG;
    { const waste_status st = check_ids(c, tokens, n); if (st) return st; }
    const float *lg = NULL;
    const int cmax = waste_model_chunk_max(&c->m);
    media_arm(c);
    for (size_t i = 0; i < n; ) {
        int k = (int)(n - i);
        if (k > cmax) k = cmax;
        lg = (k > 1) ? waste_model_prefill(&c->m, tokens + i, k, c->pos)
                     : waste_model_step(&c->m, tokens[i], c->pos, NULL);
        c->pos += k;
        i += (size_t)k;
    }
    media_release(c);
    if (logits_out) *logits_out = lg;
    if (vocab_out) *vocab_out = (size_t)c->m.cfg.vocab;
    return WASTE_OK;
}

static int sample(const float *logits, int vocab, const waste_gen_params *p,
                  uint64_t *rng)
{
    if (p->temperature <= 0.0f) {
        int best = 0;
        for (int i = 1; i < vocab; i++) if (logits[i] > logits[best]) best = i;
        return best;
    }
    /* top-k / top-p over a temperature-scaled softmax */
    int k = p->top_k > 0 && p->top_k < vocab ? p->top_k : vocab;
    int *idx = (int *)malloc((size_t)vocab * sizeof(int));
    float *pr = (float *)malloc((size_t)vocab * sizeof(float));
    if (!idx || !pr) { free(idx); free(pr); return 0; }
    for (int i = 0; i < vocab; i++) idx[i] = i;

    /* partial selection of the top k */
    for (int i = 0; i < k; i++) {
        int best = i;
        for (int j = i + 1; j < vocab; j++)
            if (logits[idx[j]] > logits[idx[best]]) best = j;
        const int t = idx[i]; idx[i] = idx[best]; idx[best] = t;
    }
    float mx = logits[idx[0]], sum = 0;
    for (int i = 0; i < k; i++) {
        pr[i] = expf((logits[idx[i]] - mx) / p->temperature);
        sum += pr[i];
    }
    float cum = 0;
    int last = k;
    for (int i = 0; i < k; i++) {
        cum += pr[i] / sum;
        if (cum >= p->top_p) { last = i + 1; break; }
    }
    *rng = *rng * 6364136223846793005ULL + 1442695040888963407ULL;
    const float r = (float)((*rng >> 11) * 0x1.0p-53) * cum;
    float acc = 0;
    int pick = idx[0];
    for (int i = 0; i < last; i++) {
        acc += pr[i] / sum;
        if (r <= acc) { pick = idx[i]; break; }
    }
    free(idx); free(pr);
    return pick;
}

waste_status waste_generate(waste_ctx *c, const int32_t *prompt, size_t n,
                            const waste_gen_params *params,
                            waste_token_cb cb, void *user)
{
    if (!c || !prompt || !n) return WASTE_E_ARG;
    { const waste_status st = check_ids(c, prompt, n); if (st) return st; }
    waste_gen_params p;
    if (params) p = *params; else waste_gen_params_init(&p);
    uint64_t rng = p.seed ? p.seed : 0x853c49e6748fea9bULL;

    const uint64_t h0 = c->m.cache.hits, m0 = c->m.cache.misses;
    const float *lg = NULL;
    double t0 = nowf();
    /* Prefill in chunks: tokens in a chunk route to overlapping expert
     * sets, so each distinct expert is read once instead of once per
     * token. Measured 2.35x fewer reads on a 16-token prompt. */
    {
        const int cmax = waste_model_chunk_max(&c->m);
        size_t i = 0;
        media_arm(c);
        while (i < n) {
            int k = (int)(n - i);
            if (k > cmax) k = cmax;
            lg = (k > 1) ? waste_model_prefill(&c->m, prompt + i, k, c->pos)
                         : waste_model_step(&c->m, prompt[i], c->pos, NULL);
            c->pos += k;
            i += (size_t)k;
        }
        media_release(c);
    }
    c->stats.sec_total += nowf() - t0;

    char piece[64];
    int cur = sample(lg, c->m.cfg.vocab, &p, &rng);
    for (uint32_t t = 0; t < p.max_tokens; t++) {
        const uint64_t hb = c->m.cache.hits, mb = c->m.cache.misses;
        const uint64_t bb = c->m.cache.bytes_read;
        t0 = nowf();
        lg = waste_model_step(&c->m, cur, c->pos++, NULL);
        const double dt = nowf() - t0;
        c->stats.sec_total += dt;
        c->stats.tokens_generated++;

        int stop = 0;
        for (size_t s = 0; s < p.n_stop; s++) if (p.stop_tokens[s] == cur) stop = 1;
        if (c->tok && cur == waste_tok_eos(c->tok)) stop = 1;

        if (cb) {
            waste_token_info info;
            memset(&info, 0, sizeof info);
            info.token_index = t;
            info.token = cur;
            info.experts_hit = (uint32_t)(c->m.cache.hits - hb);
            info.experts_missed = (uint32_t)(c->m.cache.misses - mb);
            info.bytes_read = c->m.cache.bytes_read - bb;
            info.ms_total = dt * 1000.0;
            int pn = 0;
            if (c->tok) pn = waste_tok_decode1(c->tok, cur, piece, (int)sizeof piece - 1);
            piece[pn] = 0;
            if (cb(&info, piece, user) != 0) return WASTE_E_CANCELLED;
        }
        if (stop) break;
        cur = sample(lg, c->m.cfg.vocab, &p, &rng);
    }
    c->stats.experts_hit += c->m.cache.hits - h0;
    c->stats.experts_missed += c->m.cache.misses - m0;
    c->stats.bytes_read = c->m.cache.bytes_read;
    return WASTE_OK;
}

/* ---- state & introspection ---------------------------------------------- */

void waste_state_reset(waste_ctx *c)
{
    if (!c) return;
    c->pos = 0;
    const waste_config *cf = &c->m.cfg;
    for (int L = 0; L < cf->n_layers; L++) {
        if (c->m.S[L])
            memset(c->m.S[L], 0, (size_t)cf->kda_heads * cf->kda_dim * cf->kda_dim * sizeof(float));
        if (c->m.conv[L])
            memset(c->m.conv[L], 0,
                   (size_t)3 * cf->kda_heads * cf->kda_dim * (cf->conv_k - 1) * sizeof(float));
        c->m.n_kv[L] = 0;
    }
}

waste_status waste_state_save(waste_ctx *c, const char *path)
{
    if (!c || !path) return WASTE_E_ARG;
    return waste_model_state_save(&c->m, path, c->pos) == 0 ? WASTE_OK : WASTE_E_IO;
}

waste_status waste_state_load(waste_ctx *c, const char *path)
{
    if (!c || !path) return WASTE_E_ARG;
    int pos = 0;
    const int rc = waste_model_state_load(&c->m, path, &pos);
    if (rc == -2) return WASTE_E_FORMAT;      /* built for a different model */
    if (rc) return WASTE_E_IO;
    c->pos = pos;
    return WASTE_OK;
}

waste_status waste_model_get_info(const waste_ctx *c, waste_model_info *out)
{
    if (!c || !out) return WASTE_E_ARG;
    const waste_config *cf = &c->m.cfg;
    memset(out, 0, sizeof *out);
    out->n_layers = (uint32_t)cf->n_layers;
    out->n_experts = (uint32_t)cf->n_experts;
    out->top_k = (uint32_t)cf->top_k;
    out->hidden = (uint32_t)cf->hidden;
    out->ctx_max = c->cfg.ctx_tokens;
    /* An expert's matrices are as wide as its input, and in a latent MoE
     * that is not the hidden: K3 projects 7168 down to a 3584-wide latent
     * and runs the experts there, so counting them at hidden width reports
     * exactly twice the parameters the container actually stores. */
    const uint64_t ew = cf->latent_dim ? (uint64_t)cf->latent_dim
                                       : (uint64_t)cf->hidden;
    const uint64_t pe = 3ULL * ew * cf->moe_inter;
    const uint64_t moe_layers = (uint64_t)(cf->n_layers - cf->first_dense);

    /* The routed experts are the model's bulk but not the model: the trunk
     * carries attention, the shared experts, the latent projections, the
     * norms and the embeddings, and every token runs all of it. So it
     * counts in both figures — the exception being the embedding table, of
     * which a token reads one row. Tensors outside `prefix` are the vision
     * tower, which this reports on as little as it runs it. */
    uint64_t trunk = 0, trunk_active = 0;
    const size_t plen = strlen(cf->prefix);
    for (int i = 0; i < c->m.n_tensors; i++) {
        const waste_tensor *t = &c->m.t[i];
        if (plen && strncmp(t->name, cf->prefix, plen) != 0) continue;
        trunk += (uint64_t)t->n;
        if (!strstr(t->name, "embed_tokens.weight")) trunk_active += (uint64_t)t->n;
    }

    out->params_total  = pe * (uint64_t)cf->n_experts * moe_layers + trunk;
    out->params_active = pe * (uint64_t)cf->top_k * moe_layers + trunk_active;
    /* Both containers declare model_type "kimi_linear", so the name comes
     * from the architecture they were converted from. One the engine does
     * not recognize is reported verbatim rather than guessed at. */
    out->arch = strstr(cf->arch, "KimiK3")     ? "kimi-k3"
              : strstr(cf->arch, "KimiLinear") ? "kimi-linear"
              : cf->arch[0]                    ? cf->arch
                                               : "unknown";
    out->quant_summary = c->quant;
    return WASTE_OK;
}

waste_status waste_save_usage(waste_ctx *c)
{
    if (!c) return WASTE_E_ARG;
    return waste_model_save_usage(&c->m, c->path) < 0 ? WASTE_E_IO : WASTE_OK;
}

waste_status waste_get_stats(const waste_ctx *c, waste_stats *out)
{
    if (!c || !out) return WASTE_E_ARG;
    *out = c->stats;
    out->experts_hit = c->m.cache.hits;
    out->experts_missed = c->m.cache.misses;
    out->bytes_read = c->m.cache.bytes_read;
    out->direct_io = c->m.direct_io;
    return WASTE_OK;
}
