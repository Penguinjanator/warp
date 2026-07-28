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

*Design, not inventory.* What exists today is the portable C baseline plus
`kda_neon.c`, and inline NEON in `model.c` and `vq.c` for the quantized
matvec and the VQ tables. There is no x86 SIMD, no SVE, no RVV and no
accelerator: on anything but ARM the engine runs the baseline. The rest of
this section describes where new backends plug in.

**SIMD** (NEON, dotprod/i8mm, AVX2, AVX-512, SVE, RVV): one translation
unit per ISA (`kda.c`, `kda_neon.c`, `kda_avx2.c`, …), each guarded by
`#if defined(__ARM_NEON)`-style fences so a build for another architecture
compiles them away, and each built with its own flags. All the ones valid
for the target architecture are compiled in, and `waste_cpu_features()`
picks at runtime — that is what lets a single x86 binary use AVX-512 on a
machine that has it and AVX2 on one that does not.

**Accelerators** (CUDA, Metal, BLAS, ROCm): build-time options. None is
implemented yet — the dispatch table has hooks for them and the Makefile
has the flags, but there is no `src/metal.m`, `src/cuda.cu` or
`src/blas.c`. Setting a flag now stops the build with a message saying so
rather than failing at link time on an undefined `waste_register_*`.

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

Verified on this machine (MacBook Pro M5 Pro): `tools/kda_ref.py` passes
against the official reference on *both* paths —

| path | output max\|diff\| | state max\|diff\| |
|---|---|---|
| `WASTE_BACKEND=cpu` (baseline) | 4.10e-08 | 1.79e-07 |
| auto (NEON) | 4.47e-08 | 1.19e-07 |

`waste_cpu_features()` also detects dotprod and i8mm on this CPU, and the
build used to name itself `NEON+i8mm` on the strength of that. Nothing in
the engine emits SDOT or SMMLA, so the name has been cut back to `NEON`:
the backend string reports what the binary *uses*, not what the silicon
offers. Put the suffix back in the commit that adds the kernel.

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

## i8mm/SMMLA for the batched matmul: 2x on its own work, 1.2% overall

SMMLA multiplies a 2x8 int8 tile by an 8x2 tile into a 2x2 int32
accumulator — 32 MACs per instruction against 4 for an fp32 FMA. The
natural target is `mmq_rows`, the batched trunk matmul the chunked
prefill uses for the latent projections, the shared experts and the
dense FFN. `src/model.c` now has `mmq_rows_i8mm`, tiling two weight rows
by two tokens and accumulating per quantization group so each group's
pair of scales applies once.

It works, and it is off by default. Measured on a 19-token chunked
prefill of K3:

| | batched mm | total prefill |
|---|---|---|
| f32 path | 2.38 s (6.0%) | 46.20 s |
| SMMLA | 1.15 s (3.0%) | 45.66 s |

**2.07x on the kernel, 1.2% end to end** — because the batched matmul is
only 6% of prefill to begin with. The time is in the VQ path: LUT apply
37.0% plus LUT build 17.7%, then expert I/O 37.6%. That is where the
next SIMD work belongs, and it is gather-shaped rather than GEMM-shaped,
so SMMLA does not reach it.

It also is not free numerically. SMMLA needs int8 on both sides, so the
activations get quantized per group, which the f32 path deliberately
avoids. Logits move by 6.8e-02 relative — argmax and top-5 held on the
prompt tested, but that is a real change, not fp noise.

Hence two switches, both off: build with `make WASTE_NATIVE=1` (the
default build targets baseline ARM, where `__ARM_FEATURE_MATMUL_INT8` is
not defined and the kernel compiles away) and run with `WASTE_I8MM=1`.
Turning it on by default would trade measurable accuracy for 1.2%. If
the VQ path ever gets fast enough that the batched matmul's share grows,
revisit — and at that point move the kernel into its own translation
unit so runtime dispatch can pick it, instead of requiring a native
build.

## First Linux runs (2026-07-28)

The engine had never been built on Linux. Docker, both architectures,
`tests/run.sh` against the Kimi-Linear container:

| | build | suite | backend | generation |
|---|---|---|---|---|
| Linux arm64 | ok | 12 pass, 0 fail, 4 skip | `NEON` | correct |
| Linux x86_64 | ok | 12 pass, 0 fail, 4 skip | `CPU` | correct |
| macOS arm64 | ok | 17 pass | `NEON` | correct |

Both Linux targets produce "The capital of France is Paris, and the
capital of Italy is Rome" — the same continuation as macOS — and both
pass *engine matches the PyTorch oracle*, so the numerics carry across
platforms and architectures. The four skips need `uv` or the source
weights, neither of which is in the image.

x86_64 names itself `CPU`, not AVX2, even though `waste_cpu_features()
`detects AVX2 there. That is the version string doing its job: detection
is not a kernel, and there is no x86 SIMD in this engine.

Three defects turned up, none in the engine and all invisible from macOS:

- The Makefile added `kda_neon.c` when `uname -m` contained "arm". Linux
  on aarch64 reports **aarch64**, which does not, so the translation unit
  was dropped while `backend.c` — which tests `__aarch64__` — still
  emitted the call. Undefined `waste_kda_register_neon` at link.
- `check_budget.sh` measured peak RSS with `/usr/bin/time -l`: a BSD-only
  flag, and the tool is not in a plain debian image at all. It reported
  the budget as exceeded when it had simply measured nothing. Now
  `getrusage(RUSAGE_CHILDREN)` through python3.
- The first attempt bind-mounted the working tree, so the Linux build
  overwrote the macOS objects and `./waste` became "Exec format error"
  on the host. `Dockerfile.test` copies instead.

O_DIRECT works on Docker's volumes: `waste_stats.direct_io` stays 1 and
no warning prints, while `WASTE_DIRECT=0` produces the fallback and the
"hit rate is partly the kernel's" note. Both directions of that path are
now exercised, which is more than the macOS-only build could do.

CUDA remains untested and untestable here on two counts: there is no
NVIDIA GPU on this machine, and there is still no CUDA source to compile
— `WASTE_ENABLE_CUDA=1` stops the build with a message saying so.

## AVX2 and AVX-512 (2026-07-28)

x86 ran pure scalar C until now. Two translation units, `src/simd_avx2.c`
and `src/simd_avx512.c`, each built with its own `-mavx*` flags and
selected by `waste_backend_init` from CPUID — so one binary adapts, which
is why they are separate files rather than `#ifdef`s inside model.c.

They implement the two range kernels that carry the arithmetic:
`mvq_rows_f32` (every trunk projection) and `lutb_range` (the VQ table).
Those moved behind the dispatch table for this, with their argument
structs and the two shared inlines in a new `src/simd.h`. The third hot
path, the VQ gather, gets nothing — no x86 SIMD helps it either, for the
same reason NEON does not.

**AVX2 is verified.** On Linux/x86_64 the engine reports `backend AVX2`,
the suite is 12 passed / 0 failed, and *SIMD backend matches the CPU
baseline* passes — that check runs `WASTE_BACKEND=cpu` against the
dispatched path and compares logits, so it is exactly the claim that
matters. It is now "within fp noise" rather than bit-identical, because
AVX2 accumulates in a different order.

**AVX-512 is compiled and dispatched, never executed.** No machine here
has it: the container's CPU reports `AVX512F=0` in CPUID leaf 7, and so
does `qemu-x86_64 -cpu max` under Rosetta, whose XCR0 leaves the opmask
and ZMM bits clear. The detection is doing the right thing by declining —
that much is confirmed — but the kernels themselves have never run an
instruction. Treat the first AVX-512 machine as the test, and expect the
same "matches the CPU baseline" check to be the thing that decides it.

No performance numbers from any of this: x86 here is emulated, so timings
would measure Rosetta.
