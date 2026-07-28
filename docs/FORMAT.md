# WASTE Container Format v0 (draft)

Status: **draft** — field sizes marked TBD are frozen after the K3 weights
drop (July 27, 2026) when real tensor shapes and routing statistics are known.

## Design goals

1. **One coalesced read per expert.** An expert's gate/up/down matrices are
   adjacent on disk and loaded with a single `pread` — measured as the
   difference between usable and unusable NVMe throughput.
2. **Placement decides speed, never precision.** The invariant: output is
   bit-identical whether an expert came from RAM cache or disk.
   The *substitute* path (below) is the one deliberate, bounded exception,
   and it is off by default.
3. **O_DIRECT-friendly.** Every independently-readable record is aligned to
   4 KiB and sized in 4 KiB multiples, so the page cache can be bypassed
   (measured wins from `F_NOCACHE`/O_DIRECT on some drives).
4. **Sub-4-bit without collapse.** Per-expert weights are vector-quantized:
   multi-stage (residual) VQ over 8-dim vectors with per-channel scales.
   Gate 3 measured this on real Kimi experts — VQ decisively beats RTN
   below 4 bits, and **3 bits (19.4% error) is the operating point**; 2-bit
   VQ (33%) is more than double the known-good production int4 baseline.

   The KBVQ-MoE shared-low-rank component (arXiv 2602.11184) is
   **specified but NOT implemented in v0** — see "Shared low-rank: on
   probation" below.
5. **Non-uniform bits.** Global per-expert bit allocation à la GEMQ
   (arXiv 2605.23078): important experts get 3 bit, unimportant get 2 (or
   1-bit substitute only). Attention / router / norms / shared experts /
   MTP head stay at 4–8 bit (asymmetric recipe; the MTP head must be int8+).

## File layout

A WASTE model is a directory (not a single file — shard-friendly, resumable
conversion, multi-drive splitting):

```
model.waste/
  manifest.json        # config, tensor index, bit-allocation map, checksums
  trunk.bin            # resident dense part, mmap-friendly
  experts-L{layer}.bin # one expert bank per layer (or grouped, TBD)
  subs-L{layer}.bin    # 1-bit substitute bank (optional)
  codebooks.bin        # VQ codebooks, resident
  lowrank.bin          # shared low-rank factors — NOT written in v0, see below
  usage.waste          # runtime-appended routing stats / learned hotlist
```

### manifest.json

JSON (hardened parser, treat as untrusted). Contains:

- `config`: K3 hyperparameters (layers, n_experts=896, top_k=16, KDA/MLA
  layer pattern, dims — TBD from release config.json).
- `tensors[]`: name, file, offset, aligned_size, quant fmt, shape.
- `bits[]`: per-(layer, expert) bit class from the GEMQ-style allocator.
- `blake3` per file section for integrity.

### Quantization formats (`fmt`)

| fmt | name | bits/weight | use |
|---|---|---|---|
| 0 | F32 | 32 | norms, router, `e_score_correction_bias` |
| 1 | F16/BF16 | 16 | low-rank factors, codebooks |
| 2 | Q8G | 8 | shared experts, attention projections, MTP head |
| 3 | Q4G | 4 (+f16 scale /g128) | trunk fallback, embeddings |
| 4 | **VQ3R** | 3.01 (3 codebooks x 256, dim 8) | default for experts (Gate 3) |
| 5 | **VQ2R** | 2.01 (2 codebooks x 256, dim 8) | only where Gate 3 quality allows |
| 6 | **SUB1** | ~1.0 direct VQ | cache-miss substitutes |

**VQxR record** (per expert matrix): N stages of 8-dim VQ indices into
per-layer codebooks of 256 entries each (N=3 for VQ3R, N=2 for VQ2R), plus
one FP16 scale per output channel:

```
W_expert ≈ scale_per_channel * sum_{s=1..N} codebook_s[index_s]
```

Residual (multi-stage) VQ: each successive codebook quantizes what the
previous stages left over. Bits/weight = N, plus 16/n_in for the channel
scale. Measured on real Kimi experts in Gate 3.

**Index layout is blocked by 64 rows** (`index_block` in the manifest):
`[row_block][vector_position][row_in_block][stage]`. The engine walks a
tile of rows for one vector position at a time; in plain row-major order
those rows sit `n_in/8 * stages` bytes apart, so each is a separate cache
line. Blocked, a tile's indices for one position are contiguous — measured
**1.44x** on the gather loop, which is ~40% of a token. The block size
matches `VQ_TILE` in the engine; a reader must honour `index_block` (0 =
plain row-major).

### Shared low-rank: on probation — specified, NOT implemented in v0

`lowrank.bin` and the `lowrank_id` field exist in the spec but the
converter does not emit them and the engine does not read them;
`lowrank_id` must be 0 in v0.

**Why parked** (Gate 3 plus a follow-up subspace measurement, 2026-07-27,
on real Kimi-Linear-48B experts):

- at rank N/128 the shared basis costs 0.12 bits/weight and reduces error
  by 0.3 pp — noise;
- at equal budget it loses badly: kbvq2 at 4.01 bits = 28.87% error, plain
  per-row INT4 at 4.01 bits = 15.20%;
- structurally, Kimi's experts are nearly mutually orthogonal — pairwise
  overlap of their rank-72 dominant subspaces is **0.046 against a random
  baseline of 0.031** (identical = 1.0); a shared basis captures 7.1% of
  energy vs 3.1% for random directions and 20.0% for each expert's own.

**Why not deleted.** Those measurements are in the *unweighted* weight
metric. "KLT-guided" in the paper most likely means a basis chosen after
whitening by the activation covariance, and there is a credible mechanism
by which that flips the result: every expert sees the same hidden-state
distribution, and LLM hidden states concentrate in a few dominant
directions, so in the activation-weighted metric the useful directions may
be shared *by construction* even though the weights are orthogonal here.

**What settles it:** rerun the Gate 3 comparison with an importance matrix
from real activations, in the same rented GPU session as Gate 2 and the
Gate 4 oracle. Revive if a whitened shared basis buys >1.5 pp of error at
≤0.15 bits/weight; otherwise delete the section and the field. No data has
been written in this format yet, so either way it is a cheap change.

### Expert bank record (`experts-L{n}.bin`)

```
[4 KiB-aligned]
ExpertRec {
  u32 magic 'WEXP', u16 layer, u16 expert_id
  u8  fmt (VQ3R|VQ2R), u8 flags, u16 codebook_id
  u32 gate_off, up_off, down_off, correction_off   // within record
  u32 record_4k_blocks
  -- gate indices | up indices | down indices | per-channel corrections --
}
```

One `pread` of `record_4k_blocks * 4096` bytes yields the whole expert.
Records for the same layer are contiguous and sorted by expert id;
`subs-L{n}.bin` mirrors the layout at SUB1 precision (~5× smaller reads,
used only when the engine's miss-latency budget is exceeded, HOBBIT-style,
arXiv 2411.01433 — flag-gated because it breaks bit-exactness).

### trunk.bin

Everything needed for a forward pass with zero expert reads: embeddings,
KDA/MLA attention weights, routers, shared experts, norms, LM head, MTP
head. mmap-able as one region; target ≤ 25 GB at the fmt mix above so a
64 GB machine keeps ≥ 35 GB for the expert cache.

### usage.waste

Append-only runtime log: per-(layer, expert) hit counts + decayed recency
for the LFRU policy, plus cross-layer routing pairs for the pilot/COUPLE
prefetcher. The converter can also bake an initial hotlist measured on a
calibration corpus.

## Converter pipeline (tools/, to build)

1. Stream K3 release shards (MXFP4) one at a time — never needs the full
   1.5 TB locally beyond the shard in flight + output.
2. Dequant MXFP4 → f32 blocks. (No shared-basis pass in v0 — see "Shared
   low-rank: on probation".)
3. GEMQ-style bit allocation from an importance matrix (imatrix-style
   pipeline; fallback: weight-energy heuristic Σrow²).
4. Emit VQ codebooks (k-means in 8-dim space), indices, corrections;
   verify with `--compare-tensor` byte/logit checks against the reference.

## Open questions (blocked on weights drop)

- Real per-layer expert count/shape; whether 896 is flat or grouped.
- MXFP4 → VQ requantization error compounding (research open question #2).
- Whether SUB1 substitutes measurably hurt K3 quality (KDA layers may be
  more sensitive than FFN literature suggests).
- Optimal codebook granularity: per-layer vs per-expert-group.
