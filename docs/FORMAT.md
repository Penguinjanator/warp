# WASTE Container Format v0 (draft)

Status: **draft** — field sizes marked TBD are frozen after the K3 weights
drop (July 27, 2026) when real tensor shapes and routing statistics are known.

## Design goals

1. **One coalesced read per expert.** An expert's gate/up/down matrices are
   adjacent on disk and loaded with a single `pread` (earlier work measured this
   as the difference between usable and unusable NVMe throughput).
2. **Placement decides speed, never precision.** Same invariant as earlier work:
   output is bit-identical whether an expert came from RAM cache or disk.
   The *substitute* path (below) is the one deliberate, bounded exception,
   and it is off by default.
3. **O_DIRECT-friendly.** Every independently-readable record is aligned to
   4 KiB and sized in 4 KiB multiples, so the page cache can be bypassed
   (earlier work both measured wins from `F_NOCACHE`/O_DIRECT on some drives).
4. **Sub-4-bit without collapse.** Per-expert weights are vector-quantized:
   multi-stage (residual) VQ over 8-dim vectors with per-channel scales.
   Gate 3 measured this on real Kimi experts — VQ decisively beats RTN
   below 4 bits, and **3 bits (19.4% error) is the operating point**; 2-bit
   VQ (33%) is more than double earlier work's known-good production int4.

   The KBVQ-MoE shared-low-rank component (arXiv 2602.11184) is
   **specified but NOT implemented in v0** — see "Shared low-rank: on
   probation" below.
5. **Non-uniform bits.** Global per-expert bit allocation à la GEMQ
   (arXiv 2605.23078): important experts get 3 bit, unimportant get 2 (or
   1-bit substitute only). Attention / router / norms / shared experts /
   MTP head stay at 4–8 bit (earlier work's asymmetric recipe; earlier work issue #8:
   MTP must be int8+).

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
  lowrank.bin          # KBVQ shared low-rank factors (FP16), resident
  usage.waste          # runtime-appended routing stats / learned hotlist
```

### manifest.json

JSON (hardened parser, treat as untrusted — earlier work st.h lesson). Contains:

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

**VQxR record** (per expert matrix): indices into a per-layer codebook
(8-dim vectors; codebook size 2^12 for VQ3R, 2^8 for VQ2R — TBD via
ablation), per-channel FP16 scale+bias correction, plus a residual header
naming the shared low-rank block it composes with:

```
W_expert ≈ (U_shared · V_shared^T) ⊙ per-channel-correction + dequant(VQ indices)
```

`U/V` shared factors (rank ~1/128 of dim, ~0.1 bit/param amortized overhead)
live in `lowrank.bin`, always resident in RAM.

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
(earlier work `.coli_usage` / `tier.h` LFRU), plus cross-layer routing pairs for
the pilot/COUPLE prefetcher. The converter can also bake an initial hotlist
measured on a calibration corpus (earlier work `the precompiled hotlist` approach).

## Converter pipeline (tools/, to build)

1. Stream K3 release shards (MXFP4) one at a time — never needs the full
   1.5 TB locally beyond the shard in flight + output (earlier work
   `convert_fp8_to_int4.py` discipline).
2. Dequant MXFP4 → f32 blocks; accumulate KLT/SVD stats per layer for the
   shared low-rank extraction (two-pass or online PCA, TBD).
3. GEMQ-style bit allocation from an importance matrix (earlier work imatrix
   pipeline; fallback: weight-energy heuristic Σrow²).
4. Emit VQ codebooks (k-means in 8-dim space), indices, corrections;
   verify with `--compare-tensor` byte/logit checks against the reference.

## Open questions (blocked on weights drop)

- Real per-layer expert count/shape; whether 896 is flat or grouped.
- MXFP4 → VQ requantization error compounding (research open question #2).
- Whether SUB1 substitutes measurably hurt K3 quality (KDA layers may be
  more sensitive than FFN literature suggests).
- Optimal codebook granularity: per-layer vs per-expert-group.
