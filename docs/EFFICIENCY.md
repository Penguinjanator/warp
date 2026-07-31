# Where the remaining speed is (2026-07-31)

K3 decodes at 0.33 tok/s on this machine. This document is the measured
answer to "what is left", written after an outside article
([AirLLM](https://github.com/lyogavin/airllm) running K3 at ~5 minutes per
token) prompted a review of the streaming path. The article itself has
nothing this engine wants — WASTE is ~100x faster and never expands an
expert at all — but one line of it does: **AirLLM overlaps loading with
compute and WASTE does not.**

Everything below was measured on the 64 GB M5 Pro with the container on the
internal SSD, unless marked as an estimate. Estimates are labelled and their
inputs named, because the point of this document is to decide what to build
next and a projection dressed as a measurement is how that decision goes
wrong.

## 1. The engine is not I/O-bound. It is half I/O-bound.

This is the finding everything else follows from, and it is the opposite of
what the offloading literature assumes.

Measured with `waste chat … /stats` (cumulative counters, so they include
prefill) at a budget on the floor, so the cache cannot flatter the count.
Chunk dedup is structural — `moe_chunk` iterates over the *distinct* experts
of the chunk — so a miss is a record, and the miss count is the number of
distinct records the chunk needed.

| prompt | cumulative reads | wall |
|---|---|---|
| N=1  | 14 794 | 57 s |
| N=8  | 18 836 | 73 s |
| N=16 | 22 289 | 88 s |
| N=32 | 27 881 | 120 s |

A standalone token costs 92 x 16 = **1472 reads**. The marginal token
*inside a chunk* costs far less — but only in I/O:

| interval | reads/token | vs standalone | I/O | compute | total |
|---|---|---|---|---|---|
| N=1→8   | 577 | **0.39x** | 0.64 s | 1.65 s | 2.29 s |
| N=8→16  | 432 | **0.29x** | 0.48 s | 1.40 s | 1.88 s |
| N=16→32 | 350 | **0.24x** | 0.39 s | 1.62 s | 2.00 s |
| *standalone* | 1472 | 1.00x | **1.62 s** | ~1.5 s | ~3.1 s |

(I/O derived from the read count at the measured 10.73 GB/s; compute is the
remainder of the wall clock, which is why it carries the noise.)

**Grouping tokens removes 70–76% of the I/O and 0% of the compute.** The
compute column is flat at ~1.5 s/token in every row, because `vq_apply`
costs one pass per (token, expert) pair — `vq_rows` does exactly `stages`
gathers per row per vector position — and the number of pairs is T x K x 92
however the tokens are grouped.

Independent check: this puts the ceiling of any batching scheme at
3.1 / 1.9 = **1.63x**, and chunked prefill measures 0.47 tok/s against 0.29
sequential = **1.62x**. The model reproduces a number nobody fitted it to.

## 2. The disk is nearly saturated at queue depth 1

`tools/diskbench.c`, 11.83 MB records (K3's real expert size), cache
bypassed, internal SSD:

```
seq write   :  8.30 GB/s
seq read    :  9.06 GB/s
rand  1 thr : 10.73 GB/s   <- what the engine gets today
rand  2 thr : 12.79 GB/s
rand  4 thr : 12.89 GB/s
rand 16 thr : 12.89 GB/s
```

One read in flight already gets 83% of the maximum. **A second read buys
20% of bandwidth and nothing more.** So the value of prefetching is not
bandwidth — it is that the read stops blocking the arithmetic.

## 3. The MoE loop serializes reads it already knows about

`moe_layer` picks its top-K, then:

```c
for (j = 0; j < K; j++) {
    rec = read_expert(m, L, idx[j]);   /* blocking 11.83 MB pread */
    ...                                 /* only then does it compute */
}
```

`waste_ecache_get` calls `fetch` inline on a miss. Queue depth 1, no read in
flight, no overlap.

But **all 16 expert ids are known before the first read**, from the routing
loop directly above. These are sixteen independent reads the engine has in
hand and issues one at a time.

Implementation constraint worth recording: the compute pool in
`src/threads.h` is fork-join with a single global job descriptor —
`g_pool_run_mu` serializes whole jobs — so the prefetcher cannot ride on it.
It needs its own thread(s) and its own queue.

## 4. What each lever is worth

Baseline **3.03 s/token (0.33 tok/s)** at the default budget (17.56 GB
cache, 13% hit). Decomposed at the margin: **I/O 1.41 s, expert matmul
1.03 s, everything else 0.50 s**.

### A. Pipeline I/O against compute — **built, 1.5–1.6x measured**

Turns a sum into a maximum. With two reads in flight the I/O also runs at
12.89 GB/s rather than 10.73, so 1.41 → 1.17 s.

    before:     1.41 + 1.03 + 0.50 = 2.94 s
    pipelined:  max(1.17, 1.03) + 0.50 = 1.67 s   ->  1.81x projected

**Shipped in 0.7.0 and measured at 1.5–1.6x**: 16 tokens in 31.09 s against
47.90 s synchronous, 0.51 tok/s against 0.33, with `3357 hit / 20195 miss`
identical in every run. Chunked prefill gains ~1.35x, because a chunk
already spreads each expert over several tokens and has less I/O to hide.
The projection was 1.81x; the gap is the per-layer LUT build, which is
serialized ahead of the applies and cannot hide the layer's first read.
[LEARNED.md](LEARNED.md) §22 has the defects it took to get there.

Two reader threads with their own queue (`WASTE_IO_THREADS`, `WASTE_IO_DEPTH`;
0 restores the synchronous path exactly). No format change, no reconversion,
output bit-identical — `tests/run.sh` checks read-ahead against synchronous
reads byte for byte.

Shape confirmed in the literature: SP-MoE ([arXiv:2510.10302](https://arxiv.org/abs/2510.10302))
reports 1.07–3.5x from asynchronous prefetch with a cutoff-layer policy;
MoE-SpeQ ([arXiv:2511.14102](https://arxiv.org/abs/2511.14102)) similar.

### B. The paging cliff — up to ~2.0x, but it is research

[LEARNED.md](LEARNED.md) §16: 17.32 GB of cache gives 13% hit at 0.32 tok/s;
29.32 GB gives 37% hit at **0.04 tok/s**. The engine caps itself at
`floor + 1x working set` to stay away from the cliff, and pays for it by
running at 13% hit when 37% is technically reachable.

If 37% were safe, I/O drops 1.41 → 0.85 s, and with (A) the step reaches
**1.53 s → 1.98x → ~0.65 tok/s**.

Idea to try: allocate cache slots as **purgeable** memory — on macOS
`vm_allocate` with `VM_FLAGS_PURGABLE` plus
`vm_purgable_control(VM_PURGABLE_SET_STATE, VM_PURGABLE_VOLATILE)` while a
slot is idle. Under pressure the kernel *discards* those pages instead of
compressing and swapping them; the engine finds the slot empty and treats it
as a miss, which is exactly the `pread` it already knows how to do. The 8x
cliff becomes graceful degradation and the default budget can go back to
being aggressive. Linux equivalent is `MADV_FREE` plus a sentinel.

The most speculative item here, and the one that attacks the worst number
the project has measured.

### C. Stage-major records — the only lever that cuts both buckets

Today a record is `[hdr][gate][up][down][scales]`, and within a matrix
`[row_block][vec_pos][row_in_block][stage]`: **stages are interleaved at the
innermost level**, so reading 2 of 3 stages would mean reading two bytes in
every three and saving nothing.

Proposed for format v1:

    [hdr][scales][stage0: gate|up|down][stage1: ...][stage2: ...]

each plane padded to 4 KiB. Then reading `s` stages is **one contiguous
pread of a prefix of the record** — design goal 1 ("one coalesced read per
expert") survives intact, and reading all three costs ~0.1% of padding.

What it enables:

- **per-activation precision**: read two stages for the tail of the top-16,
  the ones whose renormalized routing weight is small. Error by stage count
  is 57.5% / 33.2% / 19.5% ([LEARNED.md](LEARNED.md) §20).
- **a two-stage cache**: 1.5x more experts in the same RAM, and a hit that
  needs full precision reads only the missing stage — a third of a record.
- graceful degradation under I/O pressure instead of a stall.

It cuts compute in the same proportion, because `vq_rows` does `stages`
gathers per row. If one stage in three is dropped for half the experts, both
buckets go x0.833 and the step reaches **1.36 s → 2.23x → ~0.74 tok/s**.

**This is not what §20 refuted.** §20 measured *static per-expert*
allocation and found the delta flat to 1.01–1.15x, which is correct and
settled. This is *per-activation* allocation keyed on the current token's
routing weight — the one signal §20 identifies as not flat. And it differs
from §20's closing note (demoting the cold tail saves disk and ~0% of the
reads, because cold experts are not read) in that it demotes what is being
read *now*.

**The measurement that gates it**, and it is an afternoon rather than a
982 GB / 4.7 h reconversion: the distribution of renormalized routing weight
across the top-16. If ranks 9–16 carry under 10% of the mass the lever is
worth ~17% on both buckets; if they carry 30% it is not worth building.

### D. Batching and speculative decoding — refuted for this machine

The offloading literature — SpecMoEOff (2.5x), SP-MoE, MoE-SpeQ — all
assumes **compute is free and the transfer is the wall**. On a GPU behind
PCIe that holds. Here it does not: at the margin compute is 1.5 s/token
against 1.62 s of I/O. They are nearly equal.

From §1, grouping tokens removes 76% of the I/O and none of the compute.
So:

- **Batching** tops out at 1.63x on its own, and it does *not* compose with
  (A), because it removes I/O that (A) has already hidden underneath the
  compute. After (A), B=8 lands on the same ~1.98x that (B) reaches alone.
- **Speculative decoding** at d=4 costs 1 + 3(0.39) = 2.17 token-equivalents
  of I/O but **four tokens of compute**. Accept two of four and it has spent
  more time than generating them in sequence. K3 also ships no MTP head
  ([K3.md](K3.md)), so it would need an external draft on top of that.

Recorded here because it is the first thing anyone who reads that literature
will try to build.

### E. Past ~2x the bottleneck is arithmetic

(A) and (B) together leave `max(0.85 I/O, 1.03 matmul)` — the matmul wins,
and the engine is compute-bound. From there *every* I/O-side lever — bigger
cache, batching, speculation, expert pruning — hits the same wall at ~2x.

What is left past it: (C), and a faster `vq_rows`. That loop is three
dependent gathers per row unrolled by four; `VQ_SUPER` has already been
swept (1 and 2 tie, 4+ is worse). Stage-major planes would at least turn one
interleaved stream into three sequential ones, which the hardware prefetcher
can see.

## 5. Do not rebuild these

Refuted with measurements, in this repo:

- 2-bit experts — 34% error ([LEARNED.md](LEARNED.md) §3, [K3.md](K3.md))
- static per-expert bit allocation — Gate 6 / §20, delta flat 1.01–1.15x
- KBVQ shared low-rank — §3, pending only the activation-weighted rerun
- a 3-bit trunk — §13, the quality wall sits in front of the speed wall
- streaming `lm_head` — §13, a net loss of ~0.8 GB/token

## 6. Order

1. **Asynchronous expert prefetch** (A). Bit-exact, no reconversion, no
   quality risk. The rest of this document is optional; this is not.
2. **Measure the top-16 routing weight distribution** — an afternoon, and it
   decides (C) before committing 4.7 h of reconversion.
3. **Prototype the purgeable cache** (B).
4. **Stage-major layout** (C), if 2 clears it.
