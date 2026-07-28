/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/*
 * waste.h — public API of the WASTE inference engine.
 *
 * The engine is a library first: everything the CLI does is done through
 * the functions below, so the same capabilities are available to any host
 * program (editor plugin, server, app). The CLI in cli/ is a client of
 * this header and gets no private access.
 *
 * Design rules
 *   - C11, no dependencies, no global state: every instance is a
 *     waste_ctx, so a host can hold several models at once.
 *   - The caller owns the RAM budget. waste_cfg.ram_budget_bytes is a hard
 *     ceiling on everything the engine allocates (weights, caches, KV,
 *     scratch). The engine sizes its expert cache to fit and never
 *     exceeds the budget; if the budget is below the floor
 *     (waste_ram_floor), loading fails with WASTE_E_RAM_BUDGET instead of
 *     swapping the machine to death.
 *   - No hidden I/O: expert streaming happens on the caller's thread or
 *     on the engine's own pool, never via page-fault surprises.
 *   - Errors are returned, never printed. Nothing calls exit().
 *
 * Threading: a waste_ctx is not thread-safe; use one per thread, or
 * serialize calls. Distinct contexts are independent.
 */

#ifndef WASTE_H
#define WASTE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WASTE_API_VERSION 1

/* ---- engine version ----------------------------------------------------
 * Semantic versioning. The numeric form is MAJOR*10000 + MINOR*100 + PATCH,
 * so it compares with a plain integer test (SQLite's convention):
 *     #if WASTE_VERSION_NUMBER >= 10200   // needs >= 1.2.0
 * The compile-time macros describe the headers the caller built against;
 * the functions report the library actually linked, which is what matters
 * for an embeddable engine that may be updated independently.
 */
#define WASTE_VERSION_MAJOR  0
#define WASTE_VERSION_MINOR  1
#define WASTE_VERSION_PATCH  0
#define WASTE_VERSION_STRING "0.1.0"
#define WASTE_VERSION_NUMBER (WASTE_VERSION_MAJOR * 10000 + \
                              WASTE_VERSION_MINOR * 100 + \
                              WASTE_VERSION_PATCH)

const char *waste_version(void);         /* e.g. "0.1.0"                    */
int         waste_version_number(void);  /* e.g. 100                        */
/* Build details: backend, SIMD, container format version. Never NULL. */
const char *waste_build_info(void);

/* ---- errors ------------------------------------------------------------ */

typedef enum {
    WASTE_OK = 0,
    WASTE_E_IO = -1,           /* file missing, short read, bad checksum   */
    WASTE_E_FORMAT = -2,       /* container malformed or wrong version     */
    WASTE_E_RAM_BUDGET = -3,   /* budget below the model's floor           */
    WASTE_E_OOM = -4,
    WASTE_E_ARG = -5,
    WASTE_E_UNSUPPORTED = -6,  /* arch/quant combination not built in      */
    WASTE_E_CANCELLED = -7,    /* callback asked to stop                   */
} waste_status;

/* Human-readable, static storage; never NULL. */
const char *waste_strerror(waste_status s);

/* ---- memory planning (usable before loading anything) ------------------ */

typedef struct {
    uint64_t trunk_bytes;        /* resident weights: attn, routers,
                                    shared experts, embeddings, head      */
    uint64_t state_bytes;        /* KDA recurrent state + MLA latent KV    */
    uint64_t scratch_bytes;      /* activations, dequant staging, threads  */
    uint64_t min_expert_cache;   /* smallest cache that can run a layer    */
    uint64_t floor_bytes;        /* sum of the above: the hard minimum     */
    uint64_t recommended_bytes;  /* floor + a cache worth having           */
} waste_memplan;

/* Compute the memory floor for a container without loading weights.
 * ctx_tokens sizes the KV/state part. Cheap: reads the manifest only. */
waste_status waste_plan_memory(const char *model_path, uint32_t ctx_tokens,
                               waste_memplan *out);

/* ---- configuration ----------------------------------------------------- */

typedef enum {
    WASTE_CACHE_LFRU = 0,  /* frequency-first, recency tiebreak (default)  */
    WASTE_CACHE_LRU = 1,
    WASTE_CACHE_PINNED = 2, /* honour the baked/learned hotlist only       */
} waste_cache_policy;

typedef struct {
    /* Hard ceiling on all engine allocations, used exactly as given.
     * 0 = the engine picks: waste_memplan.recommended_bytes, capped at 88%
     * of physical RAM so a model whose recommendation exceeds the machine
     * does not swap it. Loading fails if the value (or, at 0, the capped
     * default) is below floor_bytes. waste_memory_used reports what was
     * actually resolved. */
    uint64_t ram_budget_bytes;

    uint32_t ctx_tokens;        /* 0 = container default                   */
    int      n_threads;         /* 0 = hardware concurrency                */
    int      io_threads;        /* expert-fetch pool; 0 = auto             */

    waste_cache_policy cache_policy;
    int      use_direct_io;     /* bypass page cache (F_NOCACHE/O_DIRECT)  */
    int      vision;            /* load the vision tower (438 MB on K3)    */
    int      allow_substitutes; /* low-bit expert on cache miss (HOBBIT);
                                   breaks bit-exactness, off by default    */
    int      expert_deferral;   /* overlap expert fetch with next layer    */

    const char *usage_path;     /* learned hotlist; NULL = <model>/usage   */
    const char *state_path;     /* KDA/KV checkpoint dir; NULL = disabled  */
} waste_cfg;

/* Fills cfg with defaults. Always call this before setting fields, so
 * new fields added in later versions stay sane. */
void waste_cfg_init(waste_cfg *cfg);

/* ---- lifecycle --------------------------------------------------------- */

typedef struct waste_ctx waste_ctx;

waste_status waste_open(const char *model_path, const waste_cfg *cfg,
                        waste_ctx **out);
void waste_close(waste_ctx *ctx);

/* What the engine actually allocated, after open. */
waste_status waste_memory_used(const waste_ctx *ctx, waste_memplan *out);

/* ---- tokenizer --------------------------------------------------------- */

waste_status waste_tokenize(waste_ctx *ctx, const char *text, int add_bos,
                            int32_t *out_tokens, size_t cap, size_t *n_out);
waste_status waste_detokenize(waste_ctx *ctx, const int32_t *tokens, size_t n,
                              char *out, size_t cap, size_t *n_out);

/* ---- generation -------------------------------------------------------- */

typedef struct {
    float temperature;      /* 0 = greedy                                  */
    float top_p;
    int   top_k;
    uint64_t seed;
    uint32_t max_tokens;
    const char *grammar;    /* GBNF text for constrained output, or NULL   */
    const int32_t *stop_tokens;
    size_t n_stop;
} waste_gen_params;

void waste_gen_params_init(waste_gen_params *p);

/* Per-token statistics handed to the callback: enough for a host to draw
 * a progress UI and for us to see whether the cache is doing its job. */
typedef struct {
    uint32_t token_index;
    int32_t  token;
    float    logprob;
    uint32_t experts_hit;      /* served from RAM cache                    */
    uint32_t experts_missed;   /* fetched from disk this token             */
    uint64_t bytes_read;
    double   ms_total;
    double   ms_io;
} waste_token_info;

/* Return 0 to continue, non-zero to stop generation (WASTE_E_CANCELLED). */
typedef int (*waste_token_cb)(const waste_token_info *info, const char *piece,
                              void *user);

waste_status waste_generate(waste_ctx *ctx, const int32_t *prompt, size_t n,
                            const waste_gen_params *params,
                            waste_token_cb cb, void *user);

/* Lower-level: one decode step, for hosts driving their own sampling. */
waste_status waste_eval(waste_ctx *ctx, const int32_t *tokens, size_t n,
                        const float **logits_out, size_t *vocab_out);

/* ---- conversation state ------------------------------------------------ */

/* KDA state is O(1) in context length and MLA KV is compressed, so a whole
 * session checkpoint is small — worth persisting when a cold re-prefill
 * costs minutes at streaming speeds. */
waste_status waste_state_save(waste_ctx *ctx, const char *path);
waste_status waste_state_load(waste_ctx *ctx, const char *path);
void         waste_state_reset(waste_ctx *ctx);

/* ---- introspection ----------------------------------------------------- */

typedef struct {
    uint32_t n_layers, n_experts, top_k, hidden, ctx_max;
    /* The language model: every routed expert plus the trunk, which is
     * attention, the shared experts, the latent projections, the norms and
     * the embeddings. A multimodal container's vision tower is not in
     * either figure. `params_active` is what one token touches — the whole
     * trunk except the embedding table, of which it reads a single row,
     * plus `top_k` experts per MoE layer. */
    uint64_t params_total, params_active;
    const char *arch;           /* e.g. "kimi-k3"                          */
    const char *quant_summary;  /* e.g. "experts VQ2R/VQ3R, trunk Q4G/Q8G" */
} waste_model_info;

waste_status waste_model_get_info(const waste_ctx *ctx, waste_model_info *out);

/* Persist which experts this workload used, so the next open can warm the
 * cache instead of starting cold. Written next to the container. */
waste_status waste_save_usage(waste_ctx *ctx);

/* Aggregate counters since open; a CLI `--stats` prints these. */
typedef struct {
    uint64_t tokens_generated, experts_hit, experts_missed, bytes_read;
    double   sec_total, sec_io;
    /* 0 when a bank could not bypass the page cache (no O_DIRECT support on
     * the filesystem, or a container whose records are not page multiples).
     * The hit rate is then partly the kernel's, not the engine's, and the
     * numbers do not transfer to a machine that cannot cache the model. */
    int      direct_io;
} waste_stats;

waste_status waste_get_stats(const waste_ctx *ctx, waste_stats *out);

/* Physical RAM of this machine, or 0 if it cannot be determined. A budget
 * near this number is counterproductive: the OS pages out the engine's own
 * expert cache, and a hit then costs a page fault. */
uint64_t waste_physical_ram(void);

#ifdef __cplusplus
}
#endif
#endif /* WASTE_H */
