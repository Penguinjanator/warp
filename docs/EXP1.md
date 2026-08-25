# exp1 — the working log of a throughput branch (2026-08-25)

Branch `exp1`. The goal set for it: **make K3 decode drastically faster on
this laptop**, not by a few per cent. Everything below was measured on the
64 GB M5 Pro this repo develops on, on the commit it is written next to.

This file is the branch's crash log as much as its notebook. **A kernel
panic on 2026-08-25 took the machine down mid-experiment and
`/private/tmp` with it** — the same failure mode `docs/LEARNED.md` §64
records, and the same lesson: on this machine an experiment that adds
gigabytes to a process already holding 45 GB is a liveness risk, and its
results have to be written down before the next one is started, not after.
Append to this file as things are measured. Commit often.

---

## 0. Status board

| # | lever | state | measured |
|---|---|---|---|
| 1 | fused NEON unpack for Q4G (bit-identical) | **landed** | 1.24x on the kernel, ~1.08x end to end |
| 2 | Q4G through SDOT (int8 activations) | **built, off by default** | 1.22x end to end, KL 0.29 on K3 — too much |
| 3 | Q4G through SMLAL / SMMLA (int16 / 14-bit activations) | benchmarked only | 1.44x / 1.85x on the kernel, 43-141x more accurate than 2 |
| 4 | Metal trunk matvec, rewritten | **built, loses in place** | 133-159 GB/s standalone, **8 GB/s inside the engine** — unexplained, see §5 |
| 5 | Metal VQ3R apply | benchmarked only | **123 GB/s of index against the CPU's ~29** once enough rows are in flight |
| 6 | VQ3R with a register-resident int8 table | benchmarked only | 1.76x single-thread over the current gather |
| 7 | GPU LUT build | not started | §61 says this is where a coherent-memory GPU pays most |

Instruments added on this branch: `tools/mvqbw.c`, `tools/metalbw.m`, a new
kernel in `tools/lutbw.c`, and — the one that matters most — `P_TMV`, a
profile bucket for the dense trunk matvec, plus teacher-forced KL scoring
and `sdot4=` / `devkb=` arms in `tests/sweep.c`.

---

## 1. The bucket nobody was looking at

`WASTE_PROFILE` has always had `kda`, `mla`, `moe(all)`, `expert I/O`,
`expert mm`, `LUT apply` and `lm_head`. It has never had a bucket for
**the dense trunk matvec**, so "kda 29%" was being read as if it were the
KDA recurrence. It is not: `matvec_t` is called from the KDA path, the MLA
path, the shared experts and the latent projections, and on K3 it moves
**28.5 GB per token** — the figure §59 puts at three quarters of the bytes
a decode step touches.

Measured with the new bucket, K3, top-8, cache at one working set:

| bucket | s/token | share |
|---|---|---|
| **trunk matvec** | **0.47** | **46%** |
| LUT apply (VQ3R) | 0.34 | 33% |
| LUT build | 0.13 | 13% |
| everything else incl. expert I/O | ~0.09 | 9% |

The trunk matvec is the largest single item in a K3 decode step and was
not visible in any profile this project has published.

## 2. The Q4G kernel was running at a third of the machine

`tools/mvqbw.c`, K3's dominant trunk shape `[12288 x 7168]` Q4G, 18
threads, working set swept from 0.35 to 17.6 GB (it makes no difference —
so this is not a cache artefact):

| kernel | GB/s | GMAC/s | vs the f32 path | accuracy |
|---|---|---|---|---|
| `waste_mvq_rows_f32` as shipped | 65 | 131 | 1.00x | exact |
| fused NEON unpack | **78** | 156 | **1.20x** | **bit-identical** |
| int16 activations, `vmlal_s16` | 113 | 225 | 1.74x | max\|d\| 0.005 |
| 14-bit activations, i8mm `vmmlaq_s32` | 144 | 289 | 2.22x | max\|d\| 0.016 |
| int8 activations, `vdotq_s32` | **173** | 346 | 2.66x | max\|d\| 0.695 |
| pure streaming read (the CPU's ceiling) | 188 | — | — | — |

Two things fall out. The shipped kernel is **compute-bound, not
bandwidth-bound**: 65 against a 188 GB/s ceiling, because a 4-bit weight
has to become a float before it can be multiplied and the shipped path
does that through a scalar nibble unpack into a `malloc`'d staging buffer.
And the int8 path sits at 92% of the ceiling, i.e. **there is nothing
faster than SDOT here, only more accurate**.

### 2a. The fused unpack, landed

Same FMAs into the same four-lane accumulator in the same order, with the
staging buffer deleted — `vld1q_u8`, `vand`/`vshr`, `vzipq_s8`, then the
existing widen-and-FMA chain. Verified **bit-identical** against the
previous build on a real container (`cmp` clean on the logits of a
5-token Kimi-Linear prefill), which is the standard §43 sets for anything
touching these kernels.

### 2b. SDOT, and why it is off

`WASTE_SDOT4=1`, `WASTE_SDOT4_SG` = how many activations share one int8
scale (32 by default; the weights' own group is 128). Activations are
written **deinterleaved** inside each group — even elements then odd — so
the low and high nibble of a packed byte each face a contiguous int8
vector and no interleave is needed in the kernel.

| | tok/s | trunk matvec | teacher-forced KL vs the f32 path |
|---|---|---|---|
| K3 top-8, f32 path | 0.96-0.99 | 59-61 GB/s | — |
| K3 top-8, `WASTE_SDOT4=1` sg=32 | **1.12-1.16** | 106-110 GB/s | **0.289** |
| K3 top-8, `WASTE_SDOT4=1` sg=128 | **1.18** | 120-125 GB/s | 0.441 |
| Kimi-Linear, sg=32 | 11.3 → 11.4 | | 0.0013 |

**1.22x, and the quality is not payable.** §56 records top-4 truncation at
KL 0.118 as *broken*; §50 records VQ2R at logit rel L2 9.4% as a
quantization this repo rejects. The K3 rows above are KL 0.29 and rel L2
23.6%. Note the shape of it: at one position the error is rel L2 0.03,
and over 22 teacher-forced positions it reaches 0.24 — **the KDA
recurrence accumulates it**, which is why the same kernel measures
KL 0.0013 on Kimi-Linear's 27 layers and 0.29 on K3's 93.

So SDOT stays behind an env var, and the interesting row is the one that
is 43x more accurate for 83% of its speed: **i8mm**. Two weight rows in
`A`, the high and low halves of a base-128 activation split in `B`, and
one `vmmlaq_s32` produces `w0.xa, w0.xb, w1.xa, w1.xb` — SDOT's
weight-bytes per instruction with 14-bit activations instead of 8-bit.
Benchmarked, not yet in the engine. **This is the next thing to build.**

## 3. `tests/sweep.c` grew a quality column

Two arms that generate greedily diverge as soon as one logit crosses
another, and after that they answer different questions. `WASTE_SWEEP_KL=1`
makes the first arm generate and **teacher-forces every arm after it on
that same sequence**, scoring KL, logit rel L2, top-10 overlap and argmax
agreement position by position. The reference arm scores exactly 0 against
itself, which is the check that the engine is deterministic and that the
instrument is not lying.

Also added: `sdot4=`, `devkb=`, `WASTE_SWEEP_TOPK` (pin top_k for every
arm, so a kernel sweep runs at §56's recommended operating point rather
than the container's declared top-16), `WASTE_SWEEP_IDS` (print the
generated ids, i.e. §56's continuation gate for free), and a per-arm
report of the trunk-matvec rate.

## 4. The cache stopped mattering at top-8

Same harness, one process, arms interleaved:

| cache | slots | hit | tok/s |
|---|---|---|---|
| 3400 MB | 287 | 52.1% | 0.907 / 0.958 |
| 9000 MB | 760 | 56.6% | 0.943 / 0.982 |
| 17736 MB | 1498 | 62.3% | 0.959 / 0.991 |

**A fifth of a working set is within 4% of a whole one.** §63 killed that
claim at top-16 over 200 tokens; at top-8 over 12 it is back, for the
reason §55 gives — expert I/O is a small share now, so the hit rate buys
less. Not a recommendation: 12 tokens is exactly the length §63 says
flatters a small cache. Recorded because it changes what a RAM lever is
worth, and because a smaller resident set is what keeps this machine
alive (§64, and the panic at the top of this file).

## 5. Metal: the 2026-07-28 conclusion does not survive its own kernel

`docs/BACKENDS.md` measured Metal at 53 GB/s on `lm_head` against the
CPU's 195 and concluded the *shape* of this engine was the problem.
§61 then found that conclusion stated from a mechanism rather than
measured on a coherent-memory part — which is exactly what this machine
is. The kernel it measured did three things:

- converted a group scale from half to float **per weight**;
- reduced through threadgroup memory with a log2(n) barrier ladder;
- put one threadgroup on one row and read one byte at a time.

Rewritten (`tools/metalbw.m`, and the same kernels now in `src/metal.m`):
one simdgroup per row, eight rows a threadgroup, `uint4` loads, the scale
applied once per group of 128, `simd_sum` instead of the ladder.

| | |
|---|---|
| GPU streaming read, 2 GB | **284 GB/s** — 1.5x what the CPU cluster reaches |
| Q4G matvec `[12288 x 7168]` | **133 GB/s mean, 159 best** (was 53) |
| empty command buffer | 11.0 us |

Correct in the engine: rel L2 5.2e-07 against the CPU path, argmax and
top-10 identical.

### 5a. And the VQ3R apply was never tried, which is where the finding is

| rows in one dispatch | GB/s of index |
|---|---|
| 3,072 (one expert matrix) | 12.0 |
| 12,288 | 41.0 |
| 49,152 (gate+up for eight experts) | 89.7 |
| 73,728 | **123.2** |

It was **pure occupancy**. One apply is 3072 rows = 3072 threads, about 6%
of what this GPU wants in flight; a MoE layer's worth of applies is
73,728 rows and the same kernel runs **4.2x the CPU's ~29 GB/s**. Per
token that is 9.1 GB of index at 123 GB/s = **0.074 s against the CPU's
0.31 s**.

That is the largest single measured opportunity on this branch, and it
needs the MoE restructured so gate+up for all routed experts is one
dispatch and down is another — which is also §44's "task granularity that
decouples the parallelism width from the barrier width", arriving from the
GPU side.

A codebook-reconstruction variant (24 KB of centroids in threadgroup
memory, rebuild the weights, FMA) is level with the table lookup at
73,728 rows and ahead of it below that — §50 found the same crossover on
CUDA for the same reason.

### 5b. The open defect: in the engine the same kernel gets 8 GB/s

| | trunk matvec |
|---|---|
| `devkb=-1` (CPU) | 59-61 GB/s |
| `devkb=1024` (Metal) | **7.5-8.6 GB/s** |

20,016 dispatches, 60.7 s inside `commit`+`waitUntilCompleted`, 0.04 s in
the `x`/`y` copies. So it is the dispatch, not the plumbing around it.

What has been **ruled out**, by a self-test that runs the identical kernel
inside the engine's own process (`WASTE_METAL_SELFTEST=1`):

| mode | GB/s before decode |
|---|---|
| driver buffer, reused | 79 |
| driver buffer, 32 rotating | 104 |
| no-copy host buffer, 32 rotating | 104 |
| no-copy + an `x`/`y` memcpy per dispatch | **132** |

— so it is not `newBufferWithBytesNoCopy`, not the copies, not the
process, and not the footprint (`mtres` holds 44 GB of no-copy host
buffers and still measures 100 GB/s mean).

**The last measurement taken before the panic is the lead.** The same
self-test run *after* three decode steps:

| mode | GB/s before | GB/s after decode |
|---|---|---|
| driver buffer, reused | 79 | 53 |
| driver buffer, 32 rotating | 104 | **9.0** (best 141) |
| no-copy host, 32 rotating | 104 | **13.6** (best 142) |
| no-copy + memcpy | 132 | 124 |

Two things to read here. The rotating arms collapse to a *mean* of 9-14
GB/s while their *best* stays at 140 — so it is not a steady-state
slowdown, it is a small number of very slow dispatches dragging the mean.
And the arm that collapses hardest is the one that touches 32 different
44 MB buffers, i.e. the one whose working set is 2.8 GB of pages the
process has not looked at since it allocated them. That points at page
residency for GPU access on a machine holding 45 GB, not at the kernel.

**Do not re-run that experiment as written.** It allocates 2.8 GB inside a
process already at 45 GB, and it is the last thing the machine did before
the panic. The safe version pins the K3 cache low (`WASTE_CACHE_MB=3400`,
which §4 above says costs 4%) and rotates over far fewer buffers.

## 6. VQ3R can be made faster on the CPU, but not VQ4P-fast

`tools/lutbw.c` gained kernel `E`: quantize the fp32 table to int8 and look
it up in registers. `vqtbl4q_s8` answers 0 past index 63, so a 256-entry
table is four of them over `c`, `c-64`, `c-128`, `c-192` added together —
no select, no mask; `vld3q_u8` deinterleaves the `[row][stage]` index
layout 16 rows at a time, which is what makes a stage-major loop
affordable at all.

Single thread, 126 MB of index, K3's gate shape:

| kernel | ms | GB/s | vs VQ3R |
|---|---|---|---|
| A — VQ3R as shipped | 18.6 | 6.6 | 1.00x |
| **E — VQ3R, int8 table in registers** | **10.6** | **11.6** | **1.76x** |
| D — VQ4P (4 stages x 64) | 4.7 | 26.0 | 3.93x |

§41 measured this shape at 1.24x and left it; the difference here is the
row super-tile, and the honest answer is that it does not rescue VQ3R:
**a 256-entry table needs four `vqtbl4q` where a 64-entry one needs one**,
and that factor is the whole gap to VQ4P. Widening the super-tile past one
index block made it *worse*, not better (11.6 → 9.1 → 10.1 → 10.4 GB/s at
super = 1, 2, 4, 8), so the register pressure bites before the amortization
pays.

## 7. Where this leaves the arithmetic

Per token, K3, top-8, on this machine:

| term | today | CPU best available | GPU measured |
|---|---|---|---|
| trunk matvec, 28.5 GB | 0.47 s | 0.26 s (i8mm) | 0.20 s |
| VQ3R apply, 9.1 GB of index | 0.34 s | 0.19 s (kernel E, threaded) | **0.074 s** |
| LUT build | 0.13 s | 0.13 s | ~0.02 s (§61) |
| the rest | 0.09 s | 0.09 s | 0.09 s |
| **total** | **1.03 s → 0.97 tok/s** | **0.67 s → 1.5 tok/s** | **0.38 s → 2.6 tok/s** |

Against the 0.45-0.62 tok/s the README publishes for K3 at top-16, the
GPU column is **4-6x**. The floor under all of it is bytes: 28.5 GB of
trunk plus ~9 GB of expert index at the GPU's 284 GB/s is 132 ms, i.e.
**7.6 tok/s is the ceiling this machine has** for this container, and
nothing measured here gets past it without reading fewer bytes.

## 8. Next, in order

1. **i8mm trunk matvec in the engine** (§2b). Cheap, and the accuracy is
   already benchmarked at 43x SDOT's.
2. **Find §5b.** No Metal work is worth anything until a dispatch in the
   engine costs what the same dispatch costs in the self-test. Run it with
   a small cache first.
3. **Metal VQ3R apply with a fused per-layer dispatch** (§5a). The single
   largest measured opportunity, and it needs `moe_layer` restructured.
4. **Metal LUT build** (§7 row 3, and §61's strongest result).

Refuted or parked on this branch, with the numbers: SDOT for the trunk
(§2b, quality), VQ3R in registers as a way to reach VQ4P (§6), a bigger
expert cache at top-8 (§4).
