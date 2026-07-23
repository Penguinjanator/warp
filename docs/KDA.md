# Kimi Delta Attention (KDA) — analysis & C kernel plan

Status: draft. Based on the public Kimi Linear / Gated DeltaNet lineage
(arXiv 2412.06464 Gated DeltaNet; Kimi Linear tech report). Exact K3
hyperparameters (head counts, dims, KDA:MLA layer ratio) are **TBD until the
July 27 weights drop** — everything below is written so only constants change.

## Why this matters for WASTE

K3 pairs KDA (linear attention) with Gated MLA. For a memory-starved local
engine this is the single best architectural gift:

- **KDA state is O(1) in sequence length**: one matrix-valued state per head,
  `S ∈ R^{d_k × d_v}`. At e.g. 64 heads × 128×128 × f32 that is ~4 MB/layer
  — versus gigabytes of KV for full attention at long context.
- The remaining MLA layers store only the compressed latent
  (~576 floats/token in the GLM/DeepSeek family — earlier work already has this
  kernel, including the absorption trick and disk KV persistence).
- Net effect: 1M-context capability without 1M-context memory. RAM stays
  reserved for the expert cache, which is where tokens/sec comes from.

## The recurrence

KDA is the delta rule with **per-channel (diagonal) decay gating** — a
finer-grained Gated DeltaNet:

```
S_t = S_{t-1} · Diag(α_t) + β_t · k_t (v_t − S_{t-1}ᵀ k_t)ᵀ
o_t = S_tᵀ q_t
```

- `α_t ∈ (0,1)^{d_k}`: per-channel forget gate (data-dependent, from x_t).
- `β_t ∈ (0,1)`: write strength.
- The `(v_t − Sᵀk_t)` term is the "delta" correction — it *replaces* the old
  association for key k_t instead of accumulating, which is what keeps the
  state usable at long range.

Decode cost per token per head: two GEMVs (`Sᵀk`, `Sᵀq`) + one rank-1
update + one diagonal scale ≈ `3·d_k·d_v` MACs. For 64×128×128:
~3.1 MFLOP/token/layer — trivially CPU-bound-friendly; the engine stays
NVMe-bound, which is what we want.

## Kernel plan (C11, earlier work kernel style)

### Decode (batch 1): fused recurrent step

One pass over the state per token, all three ops fused so `S` streams
through cache exactly once:

```c
// per head: S [dk][dv] f32, contiguous rows
// 1. u = Sᵀ k      (GEMV, accumulate per dv-column)
// 2. delta = beta * (v - u)
// 3. S = Diag(alpha) * S + k · deltaᵀ   (row-scale + rank-1, fused)
// 4. o = Sᵀ q      (GEMV — fusable into pass 3's row loop)
```

- NEON: process `dv` in 4×f32 lanes (`vfmaq_f32`); `alpha` row-scale is one
  `vmulq` per lane. AVX-512: 16-lane, same shape.
- State precision: **f32 accumulate**. Ablate f16/bf16 storage with f32
  accumulate later (halves state memory; risk: delta-rule error compounding
  over 100k+ tokens — needs the oracle harness before enabling).
- Layout: `S` row-major by `d_k` so the rank-1 update (`k_i · deltaᵀ`) writes
  contiguous rows; both GEMVs then read rows sequentially (`Sᵀx` becomes
  per-row dot-accumulate into `o[dv]`).

### Prefill: chunkwise parallel form

Recurrent prefill would be O(T) sequential — too slow for long prompts.
Use the standard chunked formulation (chunk C=64/128): intra-chunk attention
computed as small GEMMs (parallel over chunks via OpenMP), inter-chunk state
carried through the decayed cumulative products. This is the same
structure Kimi contributed to vLLM ("KDA prefix caching"), which we can
consult as a reference implementation for correctness, not code.

Prefill compute: O(T·C·d) — fine on CPU, and it batches beautifully with
the expert batch-union reads earlier work already does for MTP verification.

### Gating projections

`α_t`, `β_t`, plus q/k/v projections are ordinary dense matmuls — they live
in `trunk.bin` at Q8G/Q4G and use the existing IDOT int8-activation kernels
(NEON SDOT / AVX-VNNI) from earlier work `quant.h`. No new matmul work.

## Gated MLA layers

Reuse earlier work's MLA path (latent KV 576 f/token, absorption, `.coli_kv`
persistence) plus the **gate**: K3's Gated MLA adds an output/head gate —
implementation detail TBD from released code, expected to be an elementwise
sigmoid gate on attention output (cheap, fusable into the o-proj).

If K3 ships a DSA-style sparse indexer for the MLA layers, earlier work's
lightning-indexer implementation carries over.

## State persistence

KDA state + MLA latent KV together are a few hundred MB at most →
checkpoint to disk per conversation (extend `.coli_kv` container with a
`KDAS` section) so sessions reopen warm with zero re-prefill. This matters
*more* for WASTE than for earlier work: at 1–3 tok/s, re-prefilling a long agent
transcript is minutes; restoring state is milliseconds.

## Validation bar

Token-exact against the HF reference (earlier work standard): run the released
modeling code on a short prompt, dump per-layer S states and outputs, and
diff. The delta rule is numerically touchy (subtraction of near-equal
terms); f32 accumulation order must match the reference within tolerance,
and greedy tokens must match exactly over ≥1k tokens before any
quantization of the trunk is allowed into the KDA path.

## Open questions (weights drop)

1. Exact KDA:MLA interleave ratio (Kimi Linear used 3:1) and head/dim sizes.
2. Is `α` per-channel on d_k only, or factored differently in K3?
3. Does Gated MLA gate per-head or per-channel?
4. Conv1d short-filter (à la Mamba/GLA) in front of q/k/v? Some DeltaNet
   variants ship one — affects the decode hot loop.
