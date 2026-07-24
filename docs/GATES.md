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

## Gate 1 — real K3 dimensions vs our estimates. ⏳ waiting for weights
## (routine scheduled 2026-07-27 09:00)

*Protects:* buying/dedicating a 2 TB NVMe; all format TBDs.
*Test:* `routing_stats.py fetch + math` on the released config/index.
*Kill criterion:* per-token I/O or disk footprint far above estimates
(>20 GB/token @2 bit, or >1 TB at 2.5 bit).

## Gate 2 — K3 real routing trace at batch 1. ⏳ needs weights + a rented
## high-RAM box (hours, tens of €)

*Protects:* the 1.5 TB download and months of engine work.
*Test:* `trace_hf.py` on K3 (adapt gate-module path), ≥2k tokens across
3-4 domains; `simulate` with real record sizes.
*Kill criterion:* LFRU hit < 40% at 40 GB cache AND flat coverage (no
prunable cold tail) → projected < 0.5 tok/s on PCIe5 → stop or pivot.

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
