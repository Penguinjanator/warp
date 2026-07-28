# WASTE — Weight-Aware Streaming Tensor Engine

Run **Kimi K3** (2.8T params, 896 experts / 16 active) locally on a 64 GB machine.

WASTE is a highly optimized model file format plus a pure-C inference engine.

**Scope:**

- an **embeddable** engine — [src/waste.h](src/waste.h) is the whole
  contract: opaque context, no global state, errors returned not printed,
  no dependencies beyond C11 + libc;
- a **fully featured CLI** built *on* that API, with no private access —
  if the CLI can do it, an embedding host can too;
- a **configurable RAM ceiling**: `waste_cfg.ram_budget_bytes` bounds
  every allocation, and the engine refuses to open below the model's
  floor instead of swapping. See [docs/ENGINE.md](docs/ENGINE.md) — the
  measured floor for a K3-shaped config is **~11.4 GB**, and RAM above
  the floor buys cache hit rate, not headroom;
- **cross-platform** (macOS, Windows, Linux) with **pluggable acceleration
  backends** — BLAS, CUDA, Metal, NEON, AVX-512 — over a universal CPU
  implementation that is always present and always correct. Selection is
  conditional compilation plus runtime CPU detection, following
  sqlite-vector's discipline (one function-pointer table, baseline first,
  best backend partially overrides, force-CPU escape hatch) and, like it,
  **no dynamic loading**: [docs/BACKENDS.md](docs/BACKENDS.md).

## Why K3 is possible on 64 GB

Weights released 2026-07-27; the numbers below are read from the actual
files, not the announcement — see [docs/K3.md](docs/K3.md).

| Quantity | Value |
|---|---|
| Layers / experts / top-k | 93 / 896 / 16 |
| **MoE is latent**: experts run on a 3584-wide projection of hidden 7168 | 33.0 M params/expert |
| Routed parameters | 2.72 T (matches the announced 2.8 T total) |
| Native weights | MXFP4, one E8M0 scale per 32 — **1.42 TB, 96 shards** |
| WASTE target (3 bit experts, per Gate 3) | **954 GB**, fits a 1.7 TB internal SSD |
| Resident trunk + state | ~18 GB RAM (the floor) |
| One token's expert working set — the real cache floor | **17.4 GB** |
| Expert bytes read per cold token @3 bit | 17.1 GB |
| Measured hit rate at 64 GB budget (Gate 5 curve) | ~40% |
| Expected decode, internal SSD @12.78 GB/s measured | **~1 tok/s** |

The binding constraints are **disk capacity and bandwidth**, not RAM — but
RAM has a hard floor that is not the model's size: a cache below one
token's working set keeps nothing alive between tokens and delivers a 0%
hit rate. That floor is 17.4 GB for K3.

Everything above is measured. [docs/LEARNED.md](docs/LEARNED.md) collects
what was learned, including the two optimization theories that were
plausible, well-motivated, and wrong.

Two K3 architecture choices work in our favor:

1. **Kimi Delta Attention (KDA)** is linear attention: O(1) state w.r.t.
   context length. The KV cache that would kill 1M-context inference mostly
   disappears; residual Gated MLA layers use latent compression (~576
   floats/token).
2. **MXFP4-native QAT** means the model was trained to survive 4-bit;
   requantizing to 2–3 bit starts from a far better point than BF16 models.

## Architecture (see docs/)

- [docs/FORMAT.md](docs/FORMAT.md) — the WASTE container: KBVQ-style
  decomposition (shared FP16 low-rank across experts + per-expert 2–3-bit VQ
  residuals), GEMQ-style global bit allocation, dual full/substitute expert
  records for HOBBIT-style cache-miss fallback, 4 KiB-aligned coalesced
  expert blocks for O_DIRECT single-read loads.
- [docs/KDA.md](docs/KDA.md) — Kimi Delta Attention math and the C kernel plan
  (chunked prefill, recurrent decode, NEON/AVX-512 mapping).
- [docs/RESEARCH.md](docs/RESEARCH.md) — condensed deep-research report
  (arXiv 2024–2026) with adversarially verified claims and the two refuted ones.
- [src/waste_format.h](src/waste_format.h) — on-disk layout, C11 structs.

## Roadmap

1. **Format spec + header** (this skeleton) — done.
2. **Weights drop (July 27, 2026)**: pull config + 1–2 shards; measure real
   routing statistics at batch 1 (the #1 open question: expert reuse rate
   across decode tokens with 896 experts) and validate the 12.5 GB/token math.
3. **Converter**: MXFP4 shards → WASTE container, shard-by-shard, disk-safe.
4. **Engine bring-up**: dense trunk + KDA/Gated-MLA kernels, oracle-validated
   token-exact against HF transformers.
5. **Streaming tier**: LFRU + hotlist + pilot lookahead + Expert Deferral.
6. **Fallback plan** if throughput disappoints: workload-driven expert
   pruning (with 896 experts the cold tail is likely huge) — halves disk
   footprint and raises cache hit rate.

## Hard requirements

- 64 GB RAM (min), fast NVMe: 1–2 TB, PCIe5 or 2×PCIe4 RAID0 (8–14 GB/s).
- A single mid-range SSD (~3 GB/s) will work but decode sub-1 tok/s.

**Storage layout (measured — see [docs/GATES.md](docs/GATES.md) Gate H):**

| path | role | measured random-read |
|---|---|---|
| `/Volumes/WasteDisk/k3/` | raw MXFP4 download + conversion staging (~1.5 TB) | 0.94 GB/s (USB bridge — fine for staging) |
| internal SSD | converted WASTE container, streamed at runtime (~700–900 GB) | 12.78 GB/s |

USB-bridged external enclosures (10 Gbps class) are ~13× too slow to
stream experts: they cap at ~0.94 GB/s regardless of thread count, i.e.
~0.1 tok/s. Keep the *download* there, run inference off internal NVMe
(or a Thunderbolt 5 / USB4 enclosure at ~5–6 GB/s).

## Non-goals

- Interactive chat-speed inference. WASTE targets usable agentic/batch
  throughput (1–3 tok/s) on hardware people actually own.
- General GGUF/safetensors compatibility. The container is K3-shaped.
