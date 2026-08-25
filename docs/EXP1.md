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
| 2 | Q4G through SDOT (int8 activations) | **built, off by default** | 1.22x end to end, and 4700x the error of 3 — not worth it |
| 3 | Q4G through i8mm / SMLAL | **built, `WASTE_TRUNK_KERNEL=2`** | **1.18x end to end**, per-matvec rel L2 **1.4e-08** |
| 4 | Metal trunk matvec, rewritten | **built, loses in place** | 133-159 GB/s standalone, 76 in the engine, **8 at 45 GB RSS** — §5b |
| 5 | Metal VQ3R apply, batched per layer | **built, `WASTE_METAL_MOE=1`** | **1.14x on Kimi-Linear**, a wash on K3 — §5c |
| 6 | VQ3R with a register-resident int8 table | **built, `WASTE_VQ8=1`** | 1.76x on the kernel, **1.19x on the bucket, 1.05x end to end** |
| 7 | GPU LUT build | not started | §61 says this is where a coherent-memory GPU pays most |
| 8 | a pool that knows which cores are fast | **built, on by default** | **1.28x on Kimi-Linear, 1.08-1.11x on K3, bit-identical** |

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


---

## 2c. The three kernels ranked, and the instrument it took to rank them

*(2026-08-25, after the panic.)*

The i8mm and SMLAL kernels are in the engine. `WASTE_TRUNK_KERNEL` picks
one — 0 f32, 1 SDOT, 2 i8mm, 3 SMLAL — and `tests/sweep.c` has a `trunk=`
arm so all four are measured in one process.

The i8mm one lives in **`src/simd_i8mm.c`**, its own translation unit with
its own `-march`, entered only after `waste_cpu_features()` reports
FEAT_I8MM — the arrangement `simd_avx2.c` has on x86, for a reason this
branch learned the hard way. Written inside `model.c` it compiled to
**nothing at all**: FEAT_I8MM is ARMv8.6, the portable ARM build targets
older, `arm_neon.h` hides `vmmlaq_s32` behind
`__ARM_FEATURE_MATMUL_INT8`, and the `#if` around the body left an empty
function that never wrote to the caller's output buffer. The runtime
feature bit said yes, the compiler had said no, and nothing connected the
two. Logits came out 109% off. **A kernel that can be compiled out has to
be somewhere the build system can see** — and `mmq_rows_i8mm`, which has
been in `model.c` behind the same `#if` since before this branch, has the
same property: it is live only under `WASTE_NATIVE=1`.

### The measurement that ranks them is not the one that looked obvious

Teacher-forced logit KL over 8 generated tokens said SDOT 0.43, i8mm
0.008, SMLAL 0.49 — i.e. that the *most* accurate kernel of the three was
the second worst. It is not. **On K3 that metric is measuring when the
first expert flip happens, and after one flip the trajectories are
unrelated.** Two kernels that each agree with the f32 path to 4e-5 on
Kimi-Linear differ from *each other* by rel L2 0.13 on K3 — a number that
has nothing to do with either one's arithmetic.

Three instruments were added to get past that, and the order matters:

- **route agreement, set and rank kept apart.** The engine renormalizes the
  selected weights and sums the experts, so which expert sits at rank 3
  changes nothing and two near-tied scores swapping is invisible in the
  output. Comparing the ordered list reports a harmless swap as two
  mismatches: SDOT scores 56% by rank and 88% by set. The set is the
  number that means something.
- **`WASTE_TRUNK_CHECK=1`**, which runs the f32 reference beside whichever
  kernel is selected, on the real activations, and accumulates the
  relative error per matvec. This is the one that settled it.
- a short horizon. At three generated tokens the chaos has not had time to
  start and the ranking is legible; at eight it is not.

| trunk kernel | mean rel L2 per matvec (3954 real matvecs) | route set @3 tok | KL @3 tok | tok/s |
|---|---|---|---|---|
| f32 (the reference) | — | 100% | 0 | 0.93-0.99 |
| SDOT | **6.50e-05** | 86.5% | 1.2e+00 | 1.10 |
| **i8mm** | **1.39e-08** | **99.91%** | 1.6e-04 | **1.10-1.13** |
| SMLAL | **3.52e-09** | **100.00%** | 6.0e-07 | 1.06-1.09 |

Monotone in the arithmetic, as it should have been all along. And the two
top rows are **below f32 re-association noise**: rearranging a float sum
moves a result by ~1e-7, and these move it by 1e-8 and 3e-9.

**i8mm is the pick**: 1.18x end to end on K3 at top-8, the trunk matvec
from 56 to 105 GB/s, and an error four orders of magnitude under SDOT's
for 97% of SDOT's speed. It is **opt-in**, not default: it is not
bit-identical, and on a model whose router lives on near-ties, "not
bit-identical" eventually means "different text". Whether 1.18x buys that
is a shipping decision, and §56 is the precedent either way — that section
recommends top-8, which is KL 0.037, 230x more damage than this, for
1.49x.

### The methodological note, which is the durable part

**On K3, a logit norm between two arms over more than a few generated
tokens is not a quality measurement.** The router's top-k boundary sits on
near-ties — §52 already recorded that the score-correction bias reorders
the selection on 368 of 368 K3 routing lines — so a perturbation of 1e-9
and one of 1e-4 both eventually flip an expert, and after that the two
runs are answering different questions. What separates them is the error
*before* the flip, and that has to be measured where it happens.

This is §43's finding one level up. There, an int8 lookup table made the
kernel discontinuous, so "numerically equivalent" was not good enough and
bit-identity became the bar. Here the discontinuity is the router itself,
and it applies to every kernel in the engine at once.


---

## 5b (answered). The GPU's throughput in this process is a function of RSS

*(2026-08-25.)*

The 8 GB/s was not a mystery about Metal. It is memory pressure, and one
row settles it — K3, top-8, the trunk matvec, changing nothing but the
expert cache:

| expert cache | process RSS | trunk matvec on the GPU |
|---|---|---|
| 17,736 MB | ~45 GB | **7.5-8.6 GB/s** |
| 3,400 MB | ~31 GB | **71-76 GB/s** |

Standalone the same kernel measures 133-159 GB/s. So the GPU loses about
1.8x at 31 GB of resident set and about 17x at 45 GB, on a 64 GB machine,
**for host memory it is not even reading** — in that run the trunk stayed
on the CPU and only 3.4 GB of cache slots were wrapped for the device.

What it is *not*, each ruled out by a measurement rather than an argument:

- not `newBufferWithBytesNoCopy` — a self-test inside the engine's own
  process reaches 104 GB/s over 32 rotating no-copy host buffers, and 132
  with an `x`/`y` memcpy around every dispatch;
- not the footprint of the *Metal* allocations — `mtres` holds 44 GB of
  no-copy host buffers and still averages 100 GB/s;
- not the engine's habits — rewriting the 1.34 MB table between
  dispatches, reading the output straight back, and idling 1 ms of CPU in
  between measure 127, 125 and 139 GB/s against a 127 GB/s baseline
  (`tools/metalbw.m` §10);
- not dispatch shape — see §5c.

**This is §16 and §24 arriving in a third place.** Those found that a cache
above what the machine will leave resident cannot be bought at any price,
and that a "hit" on such memory is a page fault. This finds that the same
pressure taxes the GPU's access to memory that is not paged at all, and by
more than it taxes the CPU's. The practical form: **on this machine, a
budget that is good for the expert cache is bad for the GPU, and they
cannot both be satisfied.**

## 5c. The VQ3R apply on the device, built

`WASTE_METAL_MOE=1`. Per layer, in batches of `WASTE_XPAR_BATCH` experts:
one command buffer with a **concurrent** compute encoder holding the
batch's gate and up applies, then the CPU's SiTU and down tables, then a
second command buffer for the batch's down applies. `waste_metal_vq3r()`
in `src/waste_metal.h` is the entry point — the one thing the Metal
backend offers that is not a `waste_kernels` slot, because the slot shape
is per row range and the unit that pays here is a batch.

Correct: rel L2 3.6e-07 against the CPU path, argmax and top-10 identical.

**Concurrency inside one command buffer is the whole design, and it had to
be measured.** N separate command buffers serialize; N dispatches inside
one concurrent encoder do not:

| | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| separate command buffers | 127 | 97 | 73 | 41 | 21 | — |
| one concurrent encoder | 124 | 127 | 123 | 126 | 126 | 90 |

(GB/s of index, 49,152 rows total split N ways.) So the engine can bind
each routed expert's own cache slot and still get the occupancy §5a needed.

**On Kimi-Linear it wins:**

| | tok/s | LUT apply |
|---|---|---|
| CPU | 11.11 / 11.18 | 0.99 s |
| device batches | **12.41 / 12.70** | **0.59 s** |

**On K3 it does not.** Two independent reasons, both measured:

| K3, top-16, cache 3400, 8 steps | expert I/O | LUT apply | accounted |
|---|---|---|---|
| CPU | 2.69 | 5.78 | **13.81** |
| device, whole layer at once | 8.19 | 4.31 | 18.47 |
| device, batches of 4 | 4.55 | 4.13 | 14.49 |
| device, batches of 8 | 5.99 | 4.09 | 15.88 |

The first is **§44, exactly as written**: holding a batch's records before
any arithmetic starts is a barrier against the read-ahead, and on K3 it
costs more than the apply saves. Batching bounds it — 8.19 down to 4.55 —
and does not remove it. `WASTE_IO_DEPTH=8` changes nothing, which is what
says the barrier is structural rather than a queue depth.

The second is that **the apply itself only reaches 36 GB/s in the engine
against 126 standalone**, at every batch size tried. That is §5b's
mechanism again and it is the ceiling on this whole direction: until a
Metal dispatch inside this process runs at the speed the same dispatch
runs outside it, no amount of restructuring pays.

At the recommended operating point — top-8, cache 17,736 — the two arms
measure 0.976 and 0.900 tok/s. A wash, on the wrong side.

## 7 (revised). What the branch actually delivers, and the honest ceiling

Clean measurement, one process, arms interleaved, three repeats, K3, cache
17,736 MB. Discard repeat 1 of each arm — §63 records the first repeat
being reproducibly the slowest on three different hosts, and it is again:

| | top-16 (container default) | top-8 (§56's recommendation) |
|---|---|---|
| f32 trunk (this branch, fused unpack) | 0.658 / 0.661 | 0.970 / 0.988 |
| **i8mm trunk** | 0.644 / 0.715 | **1.120 / 1.133** |
| main, same settings (§56) | — | 0.885 |

**1.27x over main at the operating point it recommends, and 1.8-2.5x over
the 0.45-0.62 tok/s the README publishes for the default.** i8mm pays at
top-8 and is a wash at top-16, which is Amdahl and nothing else: the LUT
apply is 7.1 s of a 12-token top-16 run and 4.2 s of a top-8 one, so the
trunk is a much smaller share of the second.

### The ceiling, from the bytes

Per token at top-8 this engine touches 28.5 GB of Q4G trunk and 9.1 GB of
expert index. At the CPU's 188 GB/s streaming ceiling that is 0.20 s, i.e.
**5 tok/s**; at the GPU's 284 GB/s, 0.13 s, i.e. **7.6 tok/s**. Those are
the walls. Today's 1.13 tok/s is 22% of the first.

What the remaining 4.5x is made of, and what is known about each:

| term | today | best measured | what stands in the way |
|---|---|---|---|
| trunk matvec, 28.5 GB | 0.27 s (i8mm, 108 GB/s) | 0.15 s at the CPU's ceiling | SMMLA is already at 77% of it; the rest is the gather in the scale path |
| VQ3R apply, 9.1 GB of index | 0.36 s (25 GB/s) | 0.074 s on the GPU | §5b, and §44's barrier |
| LUT build | 0.13 s | ~0.02 s on the GPU | §5b; §61 says this is the biggest single GPU win there is |
| expert I/O | ~0.28 s, mostly hidden | — | becomes the wall the moment the applies get faster |
| everything else | ~0.14 s | — | |

**So the honest statement is that 2 tok/s is reachable on this machine and
5 is not, without reading fewer bytes.** Every byte-reduction lever this
repo has measured has failed on quality: a 3-bit trunk (§13), per-expert
bit allocation (§20), demoting the routing tail (§23), merging experts
(§53), clustering them (§54), FFN contextual sparsity (§59, §60) and
2-stage experts (§50). The one that worked is `top_k`, and it is taken.

### Next, in order

1. **The GPU LUT build.** It is the one term above with a 6x and no known
   obstacle other than §5b, and §61 measured it as the difference between
   3.8 and 9.1 tok/s on a coherent-memory part.
2. **§5b itself.** Everything Metal is capped at a third of its standalone
   speed until this is understood. The next experiment is the cheapest one
   not yet run: the same self-test at several cache sizes, to get the curve
   rather than the two points above — in a memory-limited cgroup or with
   the cache pinned small, not on a bare 45 GB laptop (§64, and the panic
   at the top of this file).
3. **The read barrier**, if 1 and 2 land: an async submit so the CPU can
   hold the next batch while the device runs the current one.


---

## 6b. The register-table VQ3R apply, built — and the scaling wall again

`WASTE_VQ8=1`. The kernel is `vq_rows_e_neon` in `src/kda_neon.c`; the
int8 shadow of the table is the one VQ4P already builds, with the same one
scale per 32 vector positions, so nothing new was needed on the write side.

| | kernel, 1 thread | the `LUT apply` bucket | end to end |
|---|---|---|---|
| Kimi-Linear | — | 0.98 → 0.88 s (1.11x) | 10.65 → 11.20 (1.05x) |
| K3, top-8 | 6.6 → 11.6 GB/s (**1.76x**) | 4.42 → 3.71 s (**1.19x**) | 1.067 → 1.120 (**1.05x**) |

**1.76x on the kernel and 1.19x on the bucket it is 100% of.** That gap is
§46's "scaling failure nobody has explained" and §47's answer to it,
reproduced by a third kernel: a wide, fast, SIMD-heavy loop does not
scale on this machine the way the latency-bound one it replaces does,
because `waste_parallel_for` cuts work into one chunk per thread and the
twelve efficiency cores are stragglers the barrier waits for.

Measured here directly, K3 top-8, the `LUT apply` bucket:

| threads | fp32 table | int8 register table | ratio |
|---|---|---|---|
| 6 | 3.06-3.30 s | **2.38-2.39 s** | **1.32x** |
| 18 | 3.00-4.41 | 2.53-3.72 | 1.19x |

The wide kernel gains more of its isolated speedup at the performance-core
count, exactly as §47 predicted — and six threads is 24% worse *overall*,
because the trunk matvec loses more than the apply gains (60 → 32 GB/s).
**Neither thread count is right for both kernels, and that is now three
kernels deep**: VQ4P (§47), this, and the i8mm trunk matvec.

So §47's parked recommendation is the standing one, with a third instance
behind it: **a pool that knows which cores are performance cores, and
sizes SIMD-heavy work to them while leaving the latency-bound kernels the
whole machine.** On macOS that is QoS classes rather than affinity —
`QOS_CLASS_USER_INTERACTIVE` prefers P-cores, `QOS_CLASS_BACKGROUND` is
confined to E-cores — so it is buildable here, and it is the cheapest
remaining item on this list that does not depend on §5b.

**Quality**: route set agreement 89-97%, teacher-forced. That is worse
than the i8mm trunk's 98.2% for a fifth of the gain, so it is off by
default and stays a switch.

---

## 9. Where the branch ends, for now

K3, `tests/sweep.c`, one process, arms interleaved, three repeats, cache
17,736 MB, top-8 — §56's recommended operating point. Repeat 1 of each arm
discarded (§63: the first repeat is reproducibly the slowest, on three
hosts and two operating systems).

| | tok/s | vs main |
|---|---|---|
| main at the same settings (§56) | 0.885 | 1.00x |
| exp1, f32 trunk (the fused unpack alone, bit-identical) | 0.970-0.988 | 1.11x |
| exp1, **i8mm trunk** | 1.064-1.133 | **1.24x** |
| exp1, i8mm + int8 VQ table | 1.115-1.124 | **1.27x** |
| exp1, all of it + the fast-core pool (§8) | **1.100-1.111**, spread 1% | **1.25x** |

(The last row is a different sitting and its `WASTE_WIDE=0` control read
0.855-1.002 rather than 1.115-1.124 — §33's rule, and the reason the
within-run ratio is the number to carry: 1.11x for the pool, on top of
1.24x for the kernel. The best stable reading taken on this branch is
**1.178 tok/s**.)

And against what `README.md` publishes for the default configuration —
top-16, automatic budget, **0.45-0.62 tok/s** — the same build measures
**1.12 tok/s, i.e. 1.8 to 2.5x**, of which §56's `top_k` truncation is the
larger half and this branch is the rest.

**What is not here is a factor of five.** §7 has the arithmetic: 37.6 GB
per token at top-8, a 188 GB/s CPU ceiling and a 284 GB/s GPU one, so 5
and 7.6 tok/s are the walls and 1.12 is 22% of the first. Closing that gap
needs the three things §7 lists, and two of them are blocked on §5b, which
is a property of this machine's memory rather than of any kernel.

The one honest summary: **the trunk matvec was 46% of a decode step and
nobody had looked, and it is now 21%.** Everything else on this branch is
either a switch with its numbers written down or a negative result with
the same.


---

## 8. The pool that knows which cores are fast

*(2026-08-25.)*

Three kernels on this branch wanted the same thing and §47 had already
named it: *"a pool that knows which cores are performance cores and sizes
SIMD-heavy work to them while leaving the latency-bound kernels the whole
machine."* Built.

`waste_perf_cpu_count()` in `platform.h` answers 6 here from
`hw.perflevel0.logicalcpu` — level 0 is the fastest level on Apple Silicon
by definition, and a machine with one level answers "no split to make".
Linux gets the big.LITTLE `cpu_capacity` reading; Windows gets 0 until
someone has a machine to measure `EfficiencyClass` on.

The pool then has a **fast group**: the calling thread plus workers
0..n_fast-2, created at `QOS_CLASS_USER_INTERACTIVE`, which is the whole
placement mechanism available on macOS — there is no affinity API, and
`QOS_CLASS_UTILITY` for the rest keeps them off the P cluster's back.
`waste_parallel_for_fast()` cuts the work for that group only.

`WASTE_WIDE` is a bitmask over call sites, because §47's finding is per
kernel and not per machine: 1 the VQ apply, 2 the trunk matvec, 4 the LUT
build. **The default is 5** — see below for why 2 is not in it.

### The first version measured nothing, and that is the interesting part

With one condition variable the fast job still `pthread_cond_broadcast`
to every worker; the twelve efficiency-core threads woke, found no chunk
to take, and decremented the same `active` the barrier waits on. So the
job did not wait for their *work* and did wait for their *wake-up*, which
on a busy machine is the same critical path. Measured: 1.181 against 1.182
tok/s, i.e. exactly nothing.

Two start signals fixes it — the slow group sleeps on its own condvar and
a fast job never touches it, and `active` counts participants rather than
threads. **A barrier is only as narrow as what it is allowed not to wake.**

### What it is worth

Bit-identical, always: the split is by row, so the result does not depend
on how many threads take it. `cmp`-clean against `WASTE_WIDE=0` on a real
container.

| | LUT apply | tok/s |
|---|---|---|
| Kimi-Linear, pool | 0.99 s | 10.01-10.08 |
| Kimi-Linear, **fast group** | **0.51 s** | **12.80-12.82** |
| | **1.94x** | **1.28x** |

| K3, top-8, i8mm + int8 VQ table | LUT apply | tok/s | spread |
|---|---|---|---|
| pool (`WASTE_WIDE=0`) | 3.61-4.00 s | 0.855-1.002 | **17%** |
| **fast group** (default) | **3.10-3.17 s** | **1.100-1.111** | **1%** |

Five repeats each, interleaved, one process. **1.11x on the median — and
the spread goes from 17% to 1%.** That second column is worth as much as
the first: the efficiency cores were not only the straggler, they were the
variance. Everything in this repo that has had to be reported as a range
on this machine was partly reporting this.

Per kernel, K3, measured separately:

| mask | what it moves |
|---|---|
| 1 — VQ apply | 4.4 → 4.0 s with the fp32 table, **3.2 → 2.7 s with the int8 one** |
| 2 — trunk matvec | nothing, on either model |
| 4 — LUT build | ~1.03x on K3, nothing on Kimi-Linear |

The ordering is exactly §47's mechanism. The int8 table kernel is the
widest of the three and gains most (1.20x); the fp32 gather is
latency-bound, so a slow core costs it proportionally less and it gains
1.08x; the trunk matvec is memory-bound at every thread count and gains
nothing, which is why bit 2 is a switch and not a default.

**And it is why this is a default where §47's version was not.** §47
measured *capping* the pool — 25% on one model, -34% on the other, because
the kernel that wants all eighteen cores loses twelve of them. This takes
nothing away from anything: the two kernels whose barrier was waiting on
an efficiency core stop waiting, and every other kernel still gets the
whole machine.
