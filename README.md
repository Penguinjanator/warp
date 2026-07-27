# WASTE — Weight-Aware Streaming Tensor Engine

Run **Kimi K3** (2.8T params, 896 experts / 16 active) locally on a 64 GB machine.

WASTE is a highly optimized model file format plus a pure-C inference engine,
built on the lessons of two prior engines:

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
  the floor buys cache hit rate, not headroom.

- earlier work — three-tier expert streaming
  (VRAM → RAM → NVMe), LFRU learned expert cache, router-lookahead pilot,
  MLA compressed KV, MTP speculative decoding, token-exact oracle validation.
- earlier work — asymmetric 2-bit
  per-expert quantization (imatrix-guided), precompiled expert hotlists,
  overlapped SSD prefetch, proven 284B-MoE inference on 64 GB MacBooks.

## Why K3 is (barely) possible on 64 GB

| Quantity | Value |
|---|---|
| Total parameters | 2.8 T (896 experts, 16 active/token, ~50 B active) |
| Native weights | MXFP4 + QAT from SFT stage (~1.5 TB) |
| WASTE target (~2.2 bit avg experts) | **~700–900 GB on NVMe** |
| Resident dense trunk (attn, routers, shared, embeddings @4–8 bit) | ~15–25 GB RAM |
| KDA state + MLA KV (1M ctx capable) | ~1–3 GB RAM |
| Hot expert cache (LFRU, pinned) | ~30–40 GB RAM |
| Expert bytes read per cold token @2 bit | ~12.5 GB |
| Expected decode on PCIe5 NVMe (×1–2) | **~1–3 tok/s** |

The binding constraints are **disk capacity and NVMe bandwidth**, not RAM.
K3's extreme sparsity (1.8% active) means per-token expert traffic at 2 bit
matches GLM-5.2 at 4 bit — the regime earlier work already handles.

Two K3 architecture choices work in our favor:

1. **Kimi Delta Attention (KDA)** is linear attention: O(1) state w.r.t.
   context length. The KV cache that would kill 1M-context inference mostly
   disappears; residual Gated MLA layers use latent compression (~576
   floats/token, earlier work-style).
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
   token-exact against HF transformers (earlier work's standard bar).
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
