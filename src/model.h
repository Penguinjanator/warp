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
    int8_t *q;            /* Q8G payload: int8, row-major                   */
    uint16_t *qs;         /* Q8G scales: one fp16 per group of `group`      */
    int group;
    int shape[4], ndim;
    size_t n;
} waste_tensor;

typedef struct {
    int n_layers, hidden, n_experts, top_k, moe_inter, dense_inter;
    int n_shared, first_dense, vocab, n_heads;
    int kv_lora, qk_nope, qk_rope, v_head;
    int kda_heads, kda_dim, conv_k;
    int kda_layer[128];              /* 1 if layer is KDA                   */
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
} waste_config;

typedef struct {
    int fd;                          /* pread + F_NOCACHE: no page cache    */
    long rec_bytes;
    int n_experts, cb_base;
} waste_bank;

typedef struct {
    waste_config cfg;
    waste_tensor *t;
    int n_tensors;
    float *codebooks;                /* [n_books][256][8]                   */
    int n_books, vec_dim, cb_entries, stages;
    waste_bank bank[128];
    int expert_m[3], expert_n[3];    /* gate, up, down shapes               */

    /* per-layer state */
    float *S[128];                   /* KDA recurrent state                 */
    float *conv[128];                /* KDA short-conv rings (3 x C x K-1)  */
    float *kcache[128], *vcache[128];
    int n_kv[128], kv_cap;

    /* scratch */
    float *x, *h, *tmp, *att, *logits;
    float *blockres;                 /* AttnRes history: [nblocks][hidden]  */
    int    n_blockres;
    float *prefix_sum, *ares;
    float *e_gate, *e_up, *e_down, *ff, *lut, *xs;
    int8_t *xq;
    uint64_t expert_reads;
    waste_ecache cache;
    uint8_t *miss_buf;               /* used when the cache is disabled     */
} waste_model;

/* cache_bytes: hard ceiling for the expert cache; 0 = no cache. */
int  waste_model_load(waste_model *m, const char *dir, int kv_cap,
                      size_t cache_bytes);
void waste_model_free(waste_model *m);
/* Runs one token; returns logits (vocab floats, owned by the model).
 * `pos` is the position in the sequence (0-based). */
const float *waste_model_step(waste_model *m, int token, int pos, int *routed);
const waste_tensor *waste_find(const waste_model *m, const char *name);

#ifdef __cplusplus
}
#endif
#endif /* WASTE_MODEL_H */
