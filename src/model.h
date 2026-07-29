/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/*
 * model.h — WASTE model loading and forward pass (Kimi-family).
 *
 * Trunk tensors are dequantized to f32 at load; routed experts are read
 * one 4 KiB-aligned record at a time and dequantized on demand, which is
 * the streaming path the engine is built around.
 */

#ifndef WASTE_MODEL_H
#define WASTE_MODEL_H

#include <stdint.h>
#include <stdio.h>

#include "ecache.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    char name[128];
    float *data;          /* F32 tensors, or NULL when kept quantized       */
    int8_t *q;            /* quantized payload, row-major                   */
    uint16_t *qs;         /* scales: one fp16 per group of `group`          */
    int group;
    int bits;             /* 8, 4 (two per byte) or 3 (packed bitstream)    */
    size_t rowbytes;      /* stride between rows of `q`                     */
    int shape[4], ndim;
    size_t n;
    /* Set when the tensor is deliberately left on disk instead of being
     * made resident (embed_tokens: 1.11 GB of which one row per token is
     * ever read). q and qs are NULL in that case. */
    long   file_off, file_scale_off;
    int    on_disk;
} waste_tensor;

typedef struct {
    int n_layers, hidden, n_experts, top_k, moe_inter, dense_inter;
    int n_shared, first_dense, vocab, n_heads;
    int kv_lora, q_lora, qk_nope, qk_rope, v_head;
    int kda_heads, kda_dim, conv_k;
#define WASTE_MAX_LAYERS 128
    int kda_layer[WASTE_MAX_LAYERS]; /* 1 if layer is KDA                   */
    float eps, routed_scale;
    int renorm;

    /* --- K3 additions (all absent/0 for Kimi-Linear) ------------------- */
    int   latent_dim;                /* routed_expert_hidden_size; 0 = none */
    int   latent_norm;               /* latent_moe_use_norm                 */
    int   attn_res_block;            /* attn_res_block_size; 0 = no AttnRes */
    int   full_rank_gate;            /* KDA g_proj instead of g_a/g_b       */
    float gate_lower_bound;          /* 0 = softplus form, else bounded     */
    int   mla_output_gate;           /* MLA sigmoid output gate             */
    int   act_situ;                  /* 1 = SiTU instead of SiLU            */
    float situ_beta, situ_linear_beta;
    char  prefix[64];                /* "" or "language_model." (K3)        */
    /* generation_config.json's eos_token_id, mirrored into the container
     * config. The tokenizer used to derive this positionally as
     * base_vocab + 2, which is right on both Kimi models by luck of the
     * reserved-block layout and is a guess everywhere else. Read it. */
    int   eos_token_id;              /* 0 = not stated, keep the default    */
    /* The HF architecture the container was built from. Both models call
     * themselves model_type "kimi_linear", so this is the only field that
     * tells them apart by name rather than by feature. */
    char  arch[64];
} waste_config;

typedef struct {
    int fd;                          /* pread + F_NOCACHE: no page cache    */
    long rec_bytes;
    int n_experts, cb_base;
} waste_bank;

/* Read from vision.json when the container has a tower. */
typedef struct {
    int hidden, heads, qkv_hidden, inter, layers;
    int pos_h, pos_w, text_hidden, patch;
    float eps, proj_eps;
    float mean[3], std[3];       /* pixel normalization — see image.c      */
    int   media_token;           /* the id an image expands from           */
    int   max_patches;
} waste_vision_cfg;

typedef struct {
    waste_config cfg;
    waste_vision_cfg vcfg;
    int  want_vision;                /* load the tower's 434 MB or not     */
    /* Image embeddings for the prefill about to run: one row per merged
     * patch, consumed in order at each media placeholder. */
    const float *media;
    int      media_n, media_used, cfg_media_token;
    waste_tensor *t;
    int n_tensors;
    float *codebooks;                /* [n_books][256][8]                   */
    float *codebooksT;               /* [n_books][8][256], for the LUT build*/
    int n_books, vec_dim, cb_entries, stages;
    /* 1 << fmt for every trunk format the language model actually uses,
     * recorded at load because the tensors left on disk never reach the
     * branch that fills in t->bits. */
    uint32_t trunk_fmts;
    waste_bank bank[WASTE_MAX_LAYERS];
    int expert_m[3], expert_n[3];    /* gate, up, down shapes               */

    /* per-layer state */
    float *S[WASTE_MAX_LAYERS];                   /* KDA recurrent state                 */
    float *conv[WASTE_MAX_LAYERS];                /* KDA short-conv rings (3 x C x K-1)  */
    /* MLA caches the latent, not the expanded per-head K and V: kv_b_proj
     * is absorbed into the query and the output instead (see mla_layer). */
    float *latcache[WASTE_MAX_LAYERS];            /* [kv_cap][kv_lora + qk_rope]         */
    int n_kv[WASTE_MAX_LAYERS], kv_cap;

    /* scratch */
    float *x, *h, *tmp, *att, *logits;
    /* chunked prefill scratch (allocated on first use) */
    float *cx, *cnorm, *cresid, *cq, *ckv, *clat, *cff, *cexp;
    float *cblockres, *cprefix;
    int   *croute;
    float *crw;
    int    chunk_cap;

    float *blockres;                 /* AttnRes history: [nblocks][hidden]  */
    int    n_blockres;
    float *prefix_sum, *ares;
    int8_t  *mmxq;                  /* int8 activations for the SMMLA path */
    float   *mmxs;
    size_t   mmx_cap, mms_cap;
    int      trunk_fd;              /* stays open for the on-disk tensors  */
    int8_t  *embrow;                /* one embedding row, read per token   */
    uint16_t *embsc;
    float *qabs, *cacc, *mrow;      /* MLA absorption scratch, per head    */
    float *e_gate, *e_up, *e_down, *ff, *lut, *xs;
    int8_t *xq;
    uint64_t expert_reads;
    waste_ecache cache;
    uint8_t *miss_buf;               /* used when the cache is disabled     */
    int      direct_io;              /* 0 = a bank fell back to page cache  */
} waste_model;

/* cache_bytes: hard ceiling for the expert cache; 0 = no cache. */
/* want_vision loads the 434 MB vision tower; it has to be a parameter
 * because the first thing load does is zero the struct. */
int  waste_model_load(waste_model *m, const char *dir, int kv_cap,
                      size_t cache_bytes, int want_vision);
void waste_model_free(waste_model *m);
/* Runs one token; returns logits (vocab floats, owned by the model).
 * `pos` is the position in the sequence (0-based). */
const float *waste_model_step(waste_model *m, int token, int pos, int *routed);

/* Prefill a chunk of `n` tokens starting at `pos0`, returning the logits of
 * the last one. Equivalent to n successive waste_model_step calls, but the
 * projections become GEMMs and — the part that matters for a streaming
 * engine — each distinct expert the chunk routes to is read from disk once
 * instead of once per token. */
const float *waste_model_prefill(waste_model *m, const int *tokens, int n,
                                 int pos0);
/* Prefill chunk size. Declared here because waste_plan_memory has to size
 * the chunk scratch into the RAM budget, and that must be the same number
 * the engine actually allocates. */
#define WASTE_CHUNK_MAX 64
int waste_model_chunk_max(const waste_model *m);

/* Session state: KDA recurrent state + short-conv rings + MLA KV + the
 * AttnRes history. Saving it turns a cold re-prefill into a file read,
 * which at streaming speeds is minutes versus milliseconds. */
/* Learned hotlist: which experts this workload uses, so the next run does
 * not start with an empty cache. Stored next to the container. */
int waste_model_warm_cache(waste_model *m, const char *dir);
int waste_model_save_usage(const waste_model *m, const char *dir);

int waste_model_state_save(const waste_model *m, const char *path, int pos);
int waste_model_state_load(waste_model *m, const char *path, int *pos);
const waste_tensor *waste_find(const waste_model *m, const char *name);

/* Internal primitives the vision tower reuses: same numerics, same
 * threading, no second implementation to keep in step. */
void waste_rmsnorm(float *o, const float *x, const float *w, int n, float eps);
void waste_matmul_t(waste_model *m, float *Y, const waste_tensor *t,
                    const float *X, int out, int in, int T);
void waste_deq_row(const waste_tensor *t, long r, int cols, float *dst);

/* One token's embedding row into dst (hidden floats), dequantizing if the
 * table was left quantized. */
int waste_embed_row(waste_model *m, int token, float *dst);

/* Vision: encodes one image's patches into text-embedding space.
 * `pixels` is [h*w][3*14*14] already patchified and normalized, the result
 * [h/2 * w/2][hidden]. Returns 0, or -1 when the container has no vision
 * tower or it was not loaded. */
int waste_vision_encode(waste_model *m, const float *pixels, int h, int w,
                        float *out);
int waste_vision_available(const waste_model *m);

/* Decodes an image and lays it out as [gh*gw][3*14*14], normalized. The
 * grid is chosen to keep the aspect ratio under `max_patches` and to be
 * even on both axes, because the tower merges 2x2. Caller frees. */
/* Source dimensions without decoding the pixels. K3 writes them into the
 * media block as text, so the wrapper needs them before the encode. */
int waste_image_size(const char *path, int *w, int *h);

float *waste_image_load(const char *path, int max_patches,
                        const float *mean, const float *std,
                        int *out_h, int *out_w);

/* Exposed for unit tests (tests/test_k3parts.c) — these are the pieces of
 * K3 whose maths is new, so they are checked against the reference
 * implementation directly rather than only end to end. */
float waste_situ_pair(float gate, float up, float beta, float linear_beta);
void  waste_kda_decay_gate(float *g, const float *A_log, const float *dt_bias,
                           int H, int D, float lower_bound);
void  waste_kda_decay_gate_ex(float *g, const float *A_log, const float *dt_bias,
                              int H, int D, float lower_bound, int per_channel);
void  waste_apply_attn_res(waste_model *m, const float *blockres, int nb,
                           const float *prefix_sum, const float *norm_w,
                           const float *proj_w, float *out);

#ifdef __cplusplus
}
#endif
#endif /* WASTE_MODEL_H */
