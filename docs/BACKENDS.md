# Portability and acceleration backends

Scope requirement (Marco, 2026-07-27): the engine runs on **macOS, Windows
and Linux**; acceleration backends (BLAS, CUDA, Metal, NEON, AVX-512, …)
are **loaded dynamically**, with a **universal CPU version always
available**. The loading and dispatch discipline follows
[sqlite-vector](file:///Users/marco/GitHub/sqlite-vector).

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

## Two tiers

**Tier 1 — in-process SIMD** (NEON, dotprod/i8mm, AVX2, AVX-512, SVE, RVV).
Compiled into the library, one translation unit per ISA
(`kda.c`, `kda_neon.c`, `kda_avx2.c`, …), each built with its own flags and
guarded by `#if defined(__ARM_NEON)`-style fences so a build for another
architecture simply compiles them away. Selected by
`waste_cpu_features()`; no loading step, no failure mode.

**Tier 2 — external accelerators** (CUDA, Metal, ROCm, Vulkan, BLAS).
These *must* be genuinely dynamic: a binary that links CUDA cannot start on
a machine without it. Each ships as a separate shared object exporting one
symbol:

```c
const char *waste_backend_register(waste_kernels *table);
```

`waste_backend_load()` resolves it with `dlopen`/`LoadLibraryA`, calls it,
and the plugin overwrites the slots it can accelerate — **or returns NULL
to decline** (no compatible GPU, driver too old). Declining is normal, not
an error: the engine keeps whatever it already had. A missing plugin file
is equally non-fatal.

Plugin naming: `libwaste_<name>.{so,dylib,dll}`, so `WASTE_BACKEND=cuda`
finds `libwaste_cuda.so` without the caller knowing platform conventions.

## Platform abstraction

Beyond kernels, three areas differ per OS and are isolated behind thin
wrappers rather than sprinkled through the engine:

| concern | macOS | Linux | Windows |
|---|---|---|---|
| dynamic load | `dlopen`/`dlsym` | `dlopen`/`dlsym` | `LoadLibraryA`/`GetProcAddress` |
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
- `waste_backend_init()` — CPU baseline, then NEON override, then optional
  plugin via `WASTE_BACKEND`;
- `waste_backend_load()` — dlopen/LoadLibrary wrapper with the
  `libwaste_<name>` naming convention;
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

## Not yet done

AVX2/AVX-512 modules, the CUDA/Metal/BLAS plugins themselves, the platform
I/O wrapper (currently `pread`/`F_NOCACHE` only, in `tools/diskbench.c`),
and a CI matrix across the three OSes.
