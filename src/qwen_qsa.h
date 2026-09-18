/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/*
 * qwen_qsa.h — Qwen Sparse Attention. Not MLA.
 *
 * Indexer: FP32 mean of four keys, MRoPE, ReLU-sum scores, top-512 blocks,
 * plus the 0–3 tail tokens, then attention over the original K/V.
 * Production decode calls waste_qwen_qsa_select; there is one pooling path.
 */

#ifndef WASTE_QWEN_QSA_H
#define WASTE_QWEN_QSA_H

#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* One product added into a running sum, rounded the same way everywhere QSA
 * forms one: the four-wide score loop, the one-token loop, the value sum,
 * and the reference tests/test_qsa_attn.c checks them against.
 *
 * Written as `acc += a * b`, or as two statements, the rounding belongs to
 * the compiler rather than to the language. Clang contracts within an
 * expression and gcc across statements, and gcc's vectorizer turns a
 * one-accumulator dot product into in-order vector products, each rounded
 * on its own, while it leaves a four-accumulator loop scalar and fused. So
 * the same kernel matched its reference on clang at -O2 and not at -O1, and
 * on gcc at no optimization level with the vectorizer on — linux-arm64 in
 * CI, 40 cases of 40 — whatever -ffp-contract said.
 *
 * Where the target can fuse, fmaf says so: one rounding by the language's
 * definition, one instruction (fmadd, vfmadd), and nothing a flag or a
 * vectorizer can round a second time. Where it cannot, no compiler can
 * fuse either, and two statements are two roundings. */
static inline float waste_qwen_qsa_mac(float acc, float a, float b)
{
#if defined(__ARM_FEATURE_FMA) || defined(__FMA__)
    return fmaf(a, b, acc);
#else
    const float p = a * b;
    return acc + p;
#endif
}

/* Interleaved MRoPE: freqs [3][half] -> out [half]. section is 3 ints. */
void waste_qwen_mrope_interleave(const float *freqs_t, const float *freqs_h,
                                 const float *freqs_w, const int *section,
                                 int half, float *out);

/* Apply RoPE to the first rotary_dim components (cos/sin length).
 * rotary_dim > 256 is refused (no write). */
int waste_qwen_rope_apply(float *x, int dim, const float *cos, const float *sin,
                          int rotary_dim);

/* Mean of `compress` consecutive raw keys, then RMSNorm (1+w).
 * raw_k is [compress][Dk]. */
void waste_qwen_qsa_pool_block(const float *raw_k, int compress, int Dk,
                               const float *k_ln_w, float eps, float *out);

/* Select tokens for one query. raw_k is [T][Dk] (one indexer KV head).
 * Writes up to budget+compress-1 ids into sel, returns the count.
 * q_heads is [Hq][Dk] already layernormed and RoPE'd.
 * work >= n_complete*Dk + n_complete + Dk floats; taken >= n_complete ints.
 * work/taken may be NULL when n_complete == 0. */
int waste_qwen_qsa_select(const float *q_heads, int Hq, int Dk,
                          const float *raw_k, int T, int query_pos,
                          const float *full_cos, const float *full_sin,
                          int rotary_dim, const float *k_ln_w, float eps,
                          int compress, int block_topk,
                          int *sel, float *work, int *taken);

/* The two halves of waste_qwen_qsa_select, which is score_blocks over every
 * complete block followed by pick.
 *
 * score_blocks pools, rotates and scores blocks [b0, b1): block b writes only
 * pooled[b][Dk] and scores[b], so disjoint ranges may run at once — the
 * select call lays pooled at work and scores at work + n_complete * Dk.
 *
 * pick writes the selection in order: complete blocks by score, highest
 * first and a tie to the earlier block, at most block_topk of them, each as
 * its compress token indices; then the n_tail tail tokens. order needs room
 * for n_complete ints. Returns the count. */
void waste_qwen_qsa_score_blocks(int b0, int b1, const float *q_heads, int Hq,
                                 int Dk, const float *raw_k,
                                 const float *full_cos, const float *full_sin,
                                 int rotary_dim, const float *k_ln_w, float eps,
                                 int compress, float *pooled, float *scores);
int waste_qwen_qsa_pick(const float *scores, int n_complete, int block_topk,
                        int compress, int n_tail, int *sel, int *order);

/* Causal softmax attention over selected tokens. q [Hq][D], k/v [T][Hkv][D],
 * n_rep = Hq/Hkv, scaling = 1/sqrt(D). sel[n_sel] are token indices. */
void waste_qwen_qsa_attn(const float *q, int Hq, int D,
                         const float *k, const float *v, int Hkv, int T,
                         const int *sel, int n_sel, float scale,
                         float *out, float *scratch);

/* Query heads [h0, h1) of the same attention; waste_qwen_qsa_attn is this
 * over [0, Hq). A head reads its own query row and its KV head's keys and
 * values and writes only its own row of out, so disjoint ranges may run at
 * once — each with its own scratch of >= n_sel. */
void waste_qwen_qsa_attn_heads(int h0, int h1, const float *q, int Hq, int D,
                               const float *k, const float *v, int Hkv, int T,
                               const int *sel, int n_sel, float scale,
                               float *out, float *scratch);

#ifdef __cplusplus
}
#endif
#endif /* WASTE_QWEN_QSA_H */
