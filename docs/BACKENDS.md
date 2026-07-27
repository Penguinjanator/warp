# Portability and acceleration backends

Scope requirement (Marco, 2026-07-27): the engine runs on **macOS, Windows
and Linux**; acceleration backends (BLAS, CUDA, Metal, NEON, AVX-512, …)
are selected without burdening a build that does not want them, over a
**universal CPU version always available**. The dispatch discipline follows
[sqlite-vector](file:///Users/marco/GitHub/sqlite-vector).

**No dynamic loading.** An earlier draft resolved accelerators with
`dlopen`/`LoadLibrary`; that was complexity without a matching problem, and
sqlite-vector — the model here — uses none. Backends are chosen by
**conditional compilation plus runtime feature detection**: one binary
still adapts to the CPU it runs on, and a build without CUDA simply has no
CUDA code in it.

## The sqlite-vector pattern, and what we take from it

sqlite-vector keeps one global table of function pointers
(`dispatch_distance_table[VECTOR_DISTANCE_MAX][VECTOR_TYPE_MAX]`,
`src/distance-cpu.c`). At init it `memcpy`s a fully-populated CPU table in,
then the best backend detected at runtime **overwrites the entries it
implements** — `init_distance_functions()` tries AVX-512 → AVX2 → SSE2 on
x86, NEON on ARM, RVV on RISC-V, with a `force_cpu` escape hatch and the
selected backend's name exposed for introspection (`vector_backend()`).
Detection is careful: `cpu_supports_avx512()` checks CPUID *and* XGETBV, so
a CPU whose OS has not enabled ZMM state is correctly rejected.

WASTE adopts all of it:

- one dispatch table, filled with a baseline that is **always compiled in**;
- backends **partially override** — an unimplemented kernel keeps the CPU
  version, so a new backend can start with one hot kernel and grow;
- **runtime detection with OS-support checks**, best-first with fallback;
- **force-CPU escape hatch** (`WASTE_BACKEND=cpu`, or
  `waste_backend_init(WASTE_BE_FORCE_CPU)`) — indispensable for bisecting
  numeric differences;
- **name introspection** (`waste_backend_name()`), surfaced by the CLI.

One difference: sqlite-vector's distance functions all share a signature,
so a 2-D array works. WASTE's kernels do not, so the table is a **struct of
function pointers** (`waste_kernels` in
[src/waste_backend.h](../src/waste_backend.h)). Same idea, C-idiomatic for
heterogeneous ops.

## How a backend gets in

**SIMD** (NEON, dotprod/i8mm, AVX2, AVX-512, SVE, RVV): one translation
unit per ISA (`kda.c`, `kda_neon.c`, `kda_avx2.c`, …), each guarded by
`#if defined(__ARM_NEON)`-style fences so a build for another architecture
compiles them away, and each built with its own flags. All the ones valid
for the target architecture are compiled in, and `waste_cpu_features()`
picks at runtime — that is what lets a single x86 binary use AVX-512 on a
machine that has it and AVX2 on one that does not.

**Accelerators** (CUDA, Metal, BLAS, ROCm): build-time options.

```
make                     # CPU + SIMD only, zero extra dependencies
make WASTE_ENABLE_METAL=1
make WASTE_ENABLE_CUDA=1
```

A build without `WASTE_ENABLE_CUDA` contains no CUDA code and no link
dependency — which is the whole reason dlopen looked tempting, solved more
simply by not linking it. A build *with* it still calls
`waste_register_cuda()` at init, which probes for a usable device and
**returns NULL to decline** if there is none; the engine then keeps the
backend it already had. Declining is normal, not an error.

Metal deserves a note: it is present on every Mac that can run this engine,
so on macOS it is a plain `#ifdef __APPLE__` decision with no runtime
uncertainty beyond device selection.

## Platform abstraction

Beyond kernels, three areas differ per OS and are isolated behind thin
wrappers rather than sprinkled through the engine:

| concern | macOS | Linux | Windows |
|---|---|---|---|
| cache-bypass read | `fcntl(F_NOCACHE)` | `O_DIRECT` | `FILE_FLAG_NO_BUFFERING` |
| positional read | `pread` | `pread` | `ReadFile` + `OVERLAPPED` |
| mapping | `mmap` | `mmap` | `CreateFileMapping`/`MapViewOfFile` |
| CPU features | `sysctlbyname("hw.optional.arm.*")` | `getauxval(AT_HWCAP/2)` | `IsProcessorFeaturePresent` / CPUID |
| threads | pthreads | pthreads | Win32 or C11 `threads.h` |

The expert-streaming path is the one that really cares: Gate H showed
throughput is set by random 12 MB reads with the page cache bypassed, and
that call differs on all three platforms.

## Status

Implemented and verified today:

- `waste_cpu_features()` — x86 CPUID + XGETBV (checks OS-enabled AVX/AVX-512
  state, as sqlite-vector does), aarch64 with per-OS dot-product/i8mm
  detection (macOS `sysctlbyname`, Linux `getauxval`, Windows
  `IsProcessorFeaturePresent`);
- `waste_backend_init()` — CPU baseline, then the best SIMD backend for the
  machine, then any accelerator compiled into this build;
- first kernel family (KDA) wired through the table:
  [src/kda.c](../src/kda.c) is the universal baseline,
  [src/kda_neon.c](../src/kda_neon.c) the NEON specialization.

Verified on this machine (MacBook Pro M5 Pro): detection reports
**`NEON+i8mm`**, and `tools/kda_ref.py` passes against the official
reference on *both* paths —

| path | output max\|diff\| | state max\|diff\| |
|---|---|---|
| `WASTE_BACKEND=cpu` (baseline) | 4.10e-08 | 1.79e-07 |
| auto (NEON+i8mm) | 4.47e-08 | 1.19e-07 |

That equivalence is the contract every future backend must meet: **same
results, only faster.** The Windows branches are written but not yet
exercised — they need a CI runner before we claim Windows support.

## Machine-specific optimization: measured, not guessed (2026-07-27)

Optimizing started from a profile of the C forward pass on Kimi-Linear
(`WASTE_PROFILE=1`), not from intuition. The order the profile dictated:

| step | s/token | what the profile then said |
|---|---|---|
| first correct version | 2.15 | MoE 71%, of which dequant 37% + matmul 30% |
| NEON f32 dot + thread pool | 0.93 | expert **dequantization 87.5%** |
| fused VQ matvec (no dequant) | 0.22 | expert matmul 67%, KDA 12%, I/O 12% |
| hoist gate/up tables out of the expert loop | **0.18** | expert matmul 56%, I/O 17%, KDA 15% |

**11.9x, and the logits still match the oracle** (max abs diff 4.6e-05,
relative 1.5e-06, argmax and top-10 identical) — the same check is rerun
after every step, because an optimization that changes results is not an
optimization.

The two that mattered:

1. **Never dequantize an expert.** The first version expanded VQ indices
   into f32 weights and then multiplied — 87% of the time. Instead, note
   that `sum_s C_s[i] . x_v` depends only on (stage, code, vector
   position), never on the output row: tabulate it once per matrix and
   every row becomes 3 table lookups per 8 weights. This is
   sqlite-vector's turbo-LUT idea applied to a weight matrix rather than a
   distance. Dequantization dropped from 87.5% to nothing, and the
   remaining 16% under "expert deq" is now purely file I/O.
2. **Hoist what does not vary.** Every routed expert in a layer sees the
   same input and the same per-layer codebooks for its gate and up
   matrices, so those two tables are built once per token instead of once
   per expert — 8x less table-building on two of the three matrices.

**Thread scaling** on this M5 Pro (18 logical cores, 6 performance):

| threads | 1 | 4 | 8 | 12 | 18 |
|---|---|---|---|---|---|
| s/token | 0.45 | 0.20 | 0.20 | 0.18 | 0.18 |

2.5x, flattening after ~4 — consistent with the performance-core count and
with a workload that is becoming memory-bound. The pool splits by row, so
results are bit-identical at any thread count; `WASTE_THREADS` overrides.

### int8 and SDOT: where the instruction actually fits

The expert matmul's inner loop is a **gather** (`acc += lut[block + s*256 +
code]`), with no multiply — SDOT cannot vectorize it, and ARM has no gather
instruction. Cache-blocking it (`VQ_TILE`, swept: 64 and 128 tie, larger is
worse) bought 0.18 -> 0.15 s/token; the loop is latency-bound on dependent
loads, not bandwidth-bound.

Where SDOT *does* fit is the trunk: those are dense dots, and the trunk is
already stored Q8G (int8 + one fp16 scale per 128 inputs). Keeping it int8
instead of expanding to f32 at load gives three modes:

| mode | s/token | RSS | logits vs oracle | top-10 |
|---|---|---|---|---|
| f32 weights (expand at load) | 0.15 | 9.5 GB | rel 1.6e-06 | identical |
| **int8 stored, f32 math** | **0.13** | **3.9 GB** | **rel 1.5e-06** | **identical** |
| int8 stored, int8 acts + SDOT | 0.13 | 3.9 GB | rel 1.2e-02 | **reordered** |

SDOT does what it promises on its own slice — the trunk phases drop from
0.30 s to 0.16 s, ~1.9x — but the trunk is only ~16% of a token, so Amdahl
caps the end-to-end gain at ~13%, which the f32-math path matches without
quantizing activations. Quantizing them costs four orders of magnitude of
accuracy and reorders the top-10; argmax survived here, but that is luck,
not a guarantee.

**Default: int8 storage with f32 arithmetic.** It keeps the container's
precision exactly, and the memory saving is the part that matters — 5.6 GB
freed is 5.6 GB more expert cache, and Gate 2 says cache is what buys
tokens/sec. `WASTE_SDOT=1` enables the activation-quantized path for anyone
who wants to measure the trade on their own workload.

Still on the table: i8mm/SMMLA for batched prefill (where activations are a
matrix and the accuracy trade is amortized over more work), a NEON pass
over the LUT accumulation, and Metal for the prefill GEMMs.

## Not yet done

AVX2/AVX-512 modules, the CUDA/Metal/BLAS backends themselves, the platform
I/O wrapper (currently `pread`/`F_NOCACHE` only, in `tools/diskbench.c`),
and a CI matrix across the three OSes.
