# Feasibility gates

Working rule (Marco, 2026-07-24): before every long/expensive operation, run
a cheap real test that could kill it. Each gate below names the expensive
step it protects, the test, and the recorded verdict.

## Gate 0 — does the trace→simulate methodology work, and what does real
## batch-1 routing look like? ✅ PASSED (with a sobering data point)

*Protects:* everything downstream — the whole premise that we can measure
routing risk cheaply before building.

*Test (run 2026-07-24):* real per-token routing trace from OLMoE-1B-7B
(64 experts, top-8, 16 layers — already in HF cache), 299 decode tokens at
temp 0.7 on a C-coding prompt, via `tools/trace_hf.py` hooks; then
`tools/routing_stats.py simulate`. Wall time: ~15 min including two script
fixes. Trace fixture: `tests/trace_olmoe_299.jsonl`.

*Results (real, not synthetic):*

- **Next-token expert reuse: 43.5%** — moderate, not the strong locality
  the (refuted) literature claim assumed, and well below the 60.7% our
  synthetic Zipf trace showed. The adversarial verification was right to
  be skeptical.
- Concentration is modest: top 21% of (layer,expert) slots cover 50% of
  activations; top 50% cover 80%; top 73% cover 95%.
- LFRU cache hit-rate at K3-analog cache fractions:
  ~6% of expert bytes cached → **~23%** hit; 12.5% → 35%; 25% → 53%;
  53% → 81%. LRU thrashes to 0% at small caches (each token touches 128
  slots); LFRU degrades gracefully — the earlier work-style
  frequency-first policy is the right one.

*Implication for K3 on 64 GB (40 GB cache ≈ 6% of a ~700 GB expert set):*
if K3 routes like OLMoE, expect ~20-25% hit → ~9.5 GB misses/token at
2.12 bit → **~0.8-1.3 tok/s on a 12 GB/s NVMe**, below the optimistic
2-3 tok/s. The pruning plan B gains weight: half the experts cover 80% of
activations even in this flat-ish model.

*Caveats (why this doesn't decide K3 yet):* different scale and expert
granularity (K3: 896 fine-grained experts + quantile-balanced training —
could route flatter or sharper); 299 tokens is short (LFRU barely warms
up; earlier work's learned pin sets improve with hours of workload); single
prompt/domain. Gate 2 reruns this exact pipeline on K3's real trace.

## Gate H — is the storage fast enough to stream experts? ✅ RUN 2026-07-27
## VERDICT: external USB disk is 13.6x too slow for inference; internal is fine

*Protects:* 1.5 TB download + conversion onto the wrong device.

*Test:* `tools/diskbench.c` — the engine's real access pattern (12 MB
records, `F_NOCACHE`, `pread`, 1-8 threads), 8 GB working file.

| device | seq write | seq read | random 12 MB (8 thr) | tok/s @12.5 GB/token |
|---|---|---|---|---|
| `/Volumes/WasteDisk` (APFS, USB, ASM246X bridge) | 0.91 GB/s | 0.92 GB/s | **0.94 GB/s** | **0.075** |
| internal SSD (MacBook Pro M5 Pro, 64 GB) | 9.85 GB/s | 9.52 GB/s | **12.78 GB/s** | **1.02** |

The external enclosure saturates at ~0.94 GB/s and does not scale with
threads: an ASMedia ASM246X is a USB 10 Gbps bridge, so the bottleneck is
the *bridge*, not the NVMe inside it. At that rate one cold token takes
~13 s; with the Gate-0-measured 23% LFRU hit rate, ~10 s/token
(**0.1 tok/s** — a 2000-token answer would take 5.5 hours).

The internal SSD hits 12.8 GB/s on random expert-sized reads — exactly the
top of the range every earlier projection assumed, and it has 1.7 TB free.

*Resulting placement decision:*

- **`/Volumes/WasteDisk/k3/` = download + staging** for the ~1.5 TB of raw
  MXFP4 shards (sequential writes at 0.9 GB/s are perfectly adequate, and
  it keeps the raw download off the internal disk permanently).
- **internal SSD = the converted WASTE container** (~700-900 GB, fits in
  1.7 TB free). This is what the engine streams experts from at runtime.
- Conversion reads shards sequentially from external → writes container to
  internal. Neither step is bottlenecked by the USB bridge.
- *Optional upgrade:* the machine has three free Thunderbolt 5 buses
  (120 Gb/s). Moving the same NVMe into a TB5/USB4 enclosure would give
  ~5-6 GB/s and make external-disk inference viable (~0.4-0.5 tok/s at the
  measured hit rate) — worth it only if the internal disk must stay free.

## Gate 1 — real K3 dimensions vs our estimates. ⏳ waiting for weights
## (release countdown: 2026-07-27 ~17:00 Europe/Rome)

*Protects:* buying/dedicating a 2 TB NVMe; all format TBDs.
*Test:* `routing_stats.py fetch + math` on the released config/index.
*Kill criterion:* per-token I/O or disk footprint far above estimates
(>20 GB/token @2 bit, or >1 TB at 2.5 bit).

## Gate 2 — real batch-1 routing on a Kimi MoE. ✅ RUN 2026-07-27 on
## Kimi-Linear-48B (no GPU rental needed). Hit rates are BETTER than the
## OLMoE stand-in; K3 itself still to confirm

*Protects:* the 1.5 TB download and months of engine work.

*Test:* `tools/kimi_ref.py --trace` — the pure-PyTorch oracle running off
the 3-bit container, hooking the router each decode step. 300 tokens,
coding prompt, batch 1. Fixture: `tests/trace_kimi_300.jsonl`. This is the
same router family as K3 (sigmoid + grouped top-k, `routed_scaling_factor`)
at 256 experts instead of 896.

*Results:*

- 208 unique (layer, expert) slots per token out of 6656 — **3.12% of the
  expert set touched per token**;
- **next-token reuse 33.6%** (OLMoE gave 43.5%: reuse *falls* as experts
  get finer-grained, which is the direction that matters for K3's 896);
- concentration is much sharper than OLMoE: top 8.7% of slots cover 50% of
  activations, top 28% cover 80%, top 51% cover 95%;
- LFRU hit rate vs cache fraction: 3% → 29.4%, 6% → 40.6%, 12% → 54.9%,
  24% → 71.9%, 48% → 87.4%. **LRU collapses to 5.1% at the smallest cache
  where LFRU still gets 29.4%** — frequency-first is not a nicety.

*Consequence:* `memplan.py`'s hit curve now comes from this measurement
instead of OLMoE. Projected K3 throughput at 3.01 bits (the Gate 3
operating point) on the 12.78 GB/s internal SSD:

| budget | cache | frac of experts | hit | GB/token | tok/s |
|---|---|---|---|---|---|
| 16 GB | 4.9 GB | 0.6% | 6% | 13.7 | 0.93 |
| 32 GB | 20.9 GB | 2.6% | 25% | 10.9 | 1.17 |
| **64 GB** | **52.9 GB** | **6.5%** | **42%** | **8.5** | **1.50** |
| 128 GB | 116.9 GB | 14.3% | 58% | 6.1 | 2.09 |

So ~1.5 tok/s on the target machine at 3 bits — the higher bit-width from
Gate 3 costs less than feared, because the better-measured hit rate pays
part of it back.

*Caveat:* 256 experts, not 896. The trend from OLMoE (64) to Kimi-Linear
(256) is *falling* per-token reuse but *rising* concentration; which
dominates at 896 is exactly what Gate 2b on K3 itself must answer.

## Gate 3 — quantization quality at 2-2.5 bit. ⏳ needs weights (partial
## download: a few expert shards only)

*Protects:* full conversion + engine integration.
*Test:* KBVQ-style VQ2R/VQ3R on 2-3 layers' experts from downloaded
shards; measure per-layer output MSE vs MXFP4 reference on calibration
activations; compare against the same measurement on GLM-5.2 layers
(known-good baseline from earlier work).
*Kill criterion:* reconstruction error ≫ GLM-5.2-at-int4 levels →
raise bits (disk grows) or stop.

## Gate 4 — KDA kernel correctness. ⏳ after config known

*Protects:* full engine build.
*Test:* standalone C kernel vs the released reference implementation on
random weights, then token-exact oracle run on real weights (earlier work bar).
