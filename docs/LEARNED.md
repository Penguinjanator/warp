# What we know, and how we know it

Everything below was measured on this machine (MacBook Pro M5 Pro, 64 GB,
18 logical / 6 performance cores) unless marked otherwise. Where a belief
turned out wrong, the wrong version is kept — the refutations were worth
more than the confirmations.

---

## 1. The model (see [K3.md](K3.md) for the full read)

K3 is 93 layers, 896 experts, top-16, hidden 7168 — but its MoE is a
**latent MoE**: experts operate on a 3584-wide projection of the hidden
state, not on the full width. That halves both expert size and per-token
I/O against the naive reading, and is what reconciles the announced 2.8T
with the 1.42 TB download. Weights ship **MXFP4, one E8M0 scale per 32**.

New since Kimi-Linear and still to implement: latent MoE projections,
Attention Residuals (`attn_res_block_size: 12`), SiTU activation, and a
full-rank KDA gate. No MTP head. K3 is also multimodal; the text path is
self-contained under `language_model.*`.

## 2. Storage: the enclosure matters more than the disk

`tools/diskbench.c` measures the engine's actual pattern — 12 MB records,
`F_NOCACHE`, `pread`, N threads — not sequential `dd`.

| device | random 12 MB reads |
|---|---|
| external USB SSD (ASM246X bridge) | **0.94 GB/s** |
| internal SSD | **12.78 GB/s** |

The external enclosure saturates at 0.94 GB/s and **does not scale with
threads**: the USB 10 Gbps bridge is the ceiling, not the NVMe inside it.
Hence the split: external disk holds the download and conversion staging,
the internal SSD holds the container the engine streams from.

## 3. Quantization: 3 bits, VQ, and no shared basis

Measured on real Kimi experts (`tools/quant_lab.py`), matched bit budgets:

| scheme | bits | weight error |
|---|---|---|
| rtn2-g64 | 2.25 | **71.8%** — the naive 2-bit strawman collapses |
| vq2 | 2.01 | 33.1% |
| **vq3** | **3.01** | **19.4%** |
| rtn3-g64 | 3.25 | 25.2% |
| rtn4-row (earlier work's production default) | 4.01 | 15.2% |

VQ beats round-to-nearest decisively below 4 bits. 3 bits is the
operating point: 19.4% is within reach of the known-good int4 baseline,
and the 3-bit container **answers factual questions correctly** (see §6).

**The KBVQ shared low-rank component does not pay for itself.** It was the
centrepiece of the original format design. At rank N/128 it costs 0.12
bits and buys 0.3 pp — noise. At equal budget it loses badly: 28.9% error
against plain per-row INT4's 15.2% at the same 4.01 bits. The structural
reason, measured separately: **Kimi's experts are nearly mutually
orthogonal** — pairwise overlap of their rank-72 dominant subspaces is
0.046 against a random baseline of 0.031. It is parked, not deleted,
because the measurement is in the unweighted metric and "KLT-guided"
probably means activation-whitened; the revive/delete criterion is written
into [FORMAT.md](FORMAT.md).

## 4. Routing and caching: the cache floor is one token

Two independent measurements, simulated then real.

**Gate 2 (simulated, from a real trace).** 300 batch-1 decode tokens of
Kimi-Linear: 208 of 6656 (layer, expert) slots touched per token = 3.1%;
next-token reuse **33.6%**, *down* from OLMoE's 43.5% — reuse falls as
experts get finer-grained, the direction that matters for K3's 896. But
concentration rises: the top 8.7% of slots cover half the activations.

**Gate 5 (measured, by the engine's own cache).** `src/ecache.c` with
`pread` + `F_NOCACHE`, so the kernel's page cache cannot flatter the
result — essential, because with a 17 GB container on a 64 GB machine the
kernel was silently caching everything.

| cache % of expert set | 3.0 | 6.0 | 12.1 | 24.2 | 48.4 |
|---|---|---|---|---|---|
| **measured hit** | 13.2% | **40.3%** | 61.9% | 84.8% | 93.9% |
| simulated | 29.4% | 40.6% | 54.9% | 71.9% | 87.4% |

Agreement is close at the K3-relevant 6%, and the real cache does better
above 12%.

**The disagreement at the bottom is the most useful result.** At 1.5% the
hit rate is *exactly zero* — 2604 evictions in 2704 accesses. One token
touches 208 experts; a 100-slot cache keeps nothing alive to the next
token. **The cache floor is one token's working set.** For K3 that is
16 x 92 x 11.8 MB = **17.4 GB**; a 64 GB machine clears it 2.6x, a 32 GB
machine does not.

Policy matters at the margin: LRU collapses to 5.1% where LFRU still gets
29.4%. Frequency-first is load-bearing, not a refinement.

## 5. The KDA kernel

`fla`'s `naive_recurrent_kda` is plain PyTorch, so the *official*
reference runs on Apple Silicon even though the production Triton kernels
do not — `tools/kda_ref.py` loads it by file path around the package
`__init__`. The C kernel matches it to **4.1e-08** at Kimi-Linear's real
shape.

Writing it corrected the drafted recurrence: the delta term uses the
**decayed** state `S' = Diag(exp(g))·S`, not `S_{t-1}`, and `g` is
log-space. With the draft version the model would have produced plausible
but wrong output — the expensive kind of bug.

## 6. The engine works

C forward pass vs the pure-PyTorch oracle on the same container: max abs
logit diff **4.2e-05**, relative **1.5e-06**, argmax and top-10 identical.
End to end, the 3-bit container generates:

> The capital of France is **Paris, and the capital of Italy is Rome. The
> capital of Spain is Madrid, and the capital of Germany is Berlin.**

That one sentence validates converter, container layout, quantization and
four layer types at once.

## 7. Optimization: 17.9x, and two refuted theories

2.15 → 0.12 s/token, logits unchanged at every step.

| step | s/token | what the profile said next |
|---|---|---|
| first correct version | 2.15 | MoE 71% |
| NEON dot + thread pool | 0.93 | expert **dequantization 87.5%** |
| fused VQ matvec (no dequant) | 0.22 | expert matmul 67% |
| hoist gate/up tables | 0.18 | apply 40% |
| unrolled gather chains | **0.12** | apply 36% |

**Never dequantize an expert.** `sum_s C_s[i]·x_v` depends only on
(stage, code, vector position), never on the output row — tabulate it once
per matrix and a row costs 3 lookups per 8 weights. This is
sqlite-vector's turbo-LUT idea applied to a weight matrix. Dequantization
went from 87.5% of the time to zero.

**Two hypotheses measured and refuted**, both plausible:

- *Index locality.* Blocking the index layout so a tile's rows are
  contiguous measured **1.44x in isolation** — and changed nothing in the
  real engine. The microbenchmark did not model 12 threads sharing L2. It
  cost a full reconversion of 19 GB.
- *Table bandwidth.* Re-reading the 884 KB table per 64-row tile is
  8.2 GB/token = 165 GB/s, suspiciously exactly this machine's ceiling.
  Cutting that traffic made it **slower** — the table is shared read-only
  and stays cached; the extra index streams do not.

What actually moved it: each gather is load → address → load, ~5 cycles,
and the loop ran one chain at a time. Interleaving four rows fixed it.

**int8 and SDOT, honestly.** SDOT does not apply to the expert matmul at
all — that inner loop is a gather, and ARM has no gather instruction. It
does apply to the trunk, and delivers 1.9x there (0.30 → 0.16 s), but the
trunk is ~16% of a token so Amdahl caps the end-to-end gain at 13% — which
the exact f32 path matches without quantizing activations. Quantizing them
costs four orders of magnitude of accuracy and reorders the top-10.
Default is therefore int8 *storage* with f32 arithmetic: same numbers,
**5.6 GB of RAM freed**, and by Gate 5 that RAM is worth more as cache
than the 13% was.

## 8. Projection for K3 on 64 GB

Composing the measured pieces at 3 bits: 954 GB of experts on the internal
SSD, 17.1 GB read per cold token, ~46 GB of cache = 2.6x one token's
working set, which Gate 5 puts at ~40% hit → ~10 GB/token from disk at
12.78 GB/s ≈ **0.8 s/token of I/O**, plus compute.

So **~1 tok/s**, in the same range as the earlier 1.5 tok/s estimate but
now built entirely from measurements rather than from literature. The
remaining unknown is whether K3's 896-expert latent routing caches like
Kimi-Linear's 256 — that is the first thing to measure once the download
lands.

## 9. Method notes worth keeping

- **Test before long operations.** Gate H saved a 1.4 TB download onto a
  disk that cannot stream it; Gate 3 changed the format before any data
  was written in it.
- **Microbenchmarks lie about systems.** The 1.44x index-layout result was
  real in isolation and worthless in place.
- **Re-verify after every optimization.** Two bugs were caught only by the
  oracle diff: a thread-pool chunk size that broke block alignment, and a
  `tname()` static buffer that aliased when two names were passed to one
  call.
- **Numbers that look too neat deserve suspicion.** "165 GB/s, exactly the
  machine's ceiling" was a coincidence, not an explanation.

## 10. Where a K3 decode step actually goes (2026-07-28)

First real profile, 5 decode steps, `--budget 46G`:

| stage | share | note |
|---|---|---|
| expert I/O | 39.0% | 17.0 GB/token at ~9.9 GB/s |
| expert matmul | 26.1% | LUT build 8.2 + apply 17.5 |
| KDA layers | 19.1% | dominated by q/k/v/o/gate projections |
| MLA layers | 3.3% | |
| lm_head | 1.4% | |

17.0 GB/token is exactly the working set Gate 5 predicted, and 9.9 GB/s
is close to the internal SSD's measured 12.78 GB/s: **the I/O is already
near the hardware limit**, so it only gets cheaper by being read less
often, which means cache, which means RAM.

And RAM is where the real finding is. `waste plan` at a 46 GB budget:

```
resident trunk       28.61 GB
KDA state + KV       11.68 GB
minimum expert cache  0.38 GB
budget 46 GB -> expert cache 5.64 GB
```

5.64 GB against a 17.0 GB per-token working set is 0.33x — well under
the Gate 5 floor, which is exactly why the measured hit rate is 0.0%
across 8832 accesses. The cache is not underperforming; it was never
given enough room to hold anything.

**The KV cache is 53x larger than MLA requires.** The engine caches K and
V expanded per head — 96 x 192 + 96 x 128 floats per token per layer,
120 KB — when MLA's whole point is that only the 512-wide latent plus
the 64 rope dims need storing, 2.25 KB. Absorbing `kv_b_proj` into the
query (score against the latent directly, accumulate in latent space,
project once at the end) gives:

| ctx | cached now | absorbed |
|---|---|---|
| 4,096 | 11.25 GB | 0.21 GB |
| 32,768 | 90.00 GB | 1.69 GB |
| 131,072 | 360.00 GB | 6.75 GB |
| 1,048,576 | 2.81 TB | 54.00 GB |

It costs 3.2x the attention arithmetic (3.32 -> 10.57 GMAC/token, against
48.6 GMAC for the experts) and saves 53x the attention memory traffic
(11.25 -> 0.21 GB/token).

**Implemented, and it behaves as predicted.** Measured on K3:

| | expert cache | hit rate | tok/s |
|---|---|---|---|
| expanded, budget 46G | 5.64 GB | 0% | 0.28 |
| latent, budget 46G | 16.68 GB | 11% | 0.30 |
| latent, budget 56G | 26.68 GB | 32% | 0.34 |

The 0% was never the cache underperforming — 5.64 GB against a 17.0 GB
working set is a third of the floor, so nothing could survive from one
token to the next. Given room, it behaves exactly as Gate 2's simulation
said it would. Logits are unchanged (max|diff| 1.19e-05 against the
expanded path, argmax and top-5 identical).

And long context stops being impossible: at 128K tokens the expanded
layout wanted 360 GB of KV, the latent wants 7.18 GB and still leaves
20.14 GB of expert cache inside a 56 GB budget.

**The report says the trunk should not be 4-bit.** QAT covered the expert
weights at MXFP4 with everything else in higher precision, so the model
has no trained tolerance for a quantized trunk — and mine is 26.33 GiB
of Q4G measuring 11.8% off the source weights (Q8G measures 0.64%). This
is a quality risk, not a speed one, and it trades directly against the
cache: raising the whole trunk to Q8G would cost the 11 GB that
absorbing the KV cache frees.

**No speculative decoding is available.** The report fine-tunes K3's MTP
layer into an EAGLE-3 draft whose input fuses the 1st, 4th and final
AttnRes blocks — representations this engine already materializes in
`m->blockres`. But the open release ships no MTP, draft, or fusion
tensors, so there is nothing to run.

Method note to add to §9: **WASTE_THREADS=1 is not a serial baseline on
this machine.** 6 of 18 logical cores are performance cores, and a
single-threaded process lands on an E-core; the same KDA kernel reads
17.21s that way against 4.23s with the pool merely available. The
parallel-vs-serial comparison has to be made with the pool alive.


## 11. The RAM budget was not a ceiling (2026-07-28)

`waste.h` calls `ram_budget_bytes` a hard ceiling on everything the engine
allocates. Measuring it on K3 — rather than on the small model the test
uses — showed peak RSS running 1.4 to 3.4 GB over a budget set near the
floor. Two independent causes, neither visible on Kimi-Linear.

**The trunk was loaded twice.** `load_trunk` slurped `trunk.bin` whole and
then copied each tensor out of the buffer, so loading wanted twice the
trunk resident: 57 GB on K3, before a single token. Every tensor knows
its own offset, so there was never a reason to hold the file. Reading
them one at a time with `pread` also cut load from 34s to 20s — most of
that being the memory pressure the second copy caused, not the I/O.

**Scratch was a guess.** The plan used a flat 64 MB + `hidden*64*4`. The
decode buffers alone are 252 MB on K3 (`e_gate`/`e_up`/`e_down` are
`moe_inter * hidden` floats each), and the chunked-prefill buffers — up
to ~500 MB at `WASTE_CHUNK_MAX`, allocated on first use and never freed —
were not counted at all. K3's floor is 30.38 GB, not the 29.69 the
planner claimed.

Two method notes worth keeping:

- **A test on the small model does not test the big one.** The budget
  check has been green since it was written; it runs on Kimi-Linear,
  where the scratch it fails to count is measured in single megabytes.
  Errors that scale with the model need a check that scales with it.
- **Peak RSS is a noisy detector on macOS.** Under pressure the kernel
  compresses anonymous pages, so the same pre-fix overrun measured
  anywhere between 1.3 and 3.4 GB depending on the run and the prompt.
  It is good enough as a guard and not good enough as a proof, which is
  why the accounting is now derived from the config rather than inferred
  from a measurement.
