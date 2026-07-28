# WASTE — Weight-Aware Streaming Tensor Engine

**Run a frontier model on the computer you already own.**

WASTE is a C inference engine and an on-disk format built around one
constraint: the model does not fit in RAM, and never will. A 2.78-trillion
parameter mixture-of-experts activates about 4% of itself per token, so
almost all of that weight is idle at any instant. WASTE keeps the idle
part on disk, streams what a token actually needs, and spends every
available byte of RAM on the part that repeats.

```
$ waste run k3.waste "The capital of France is" -n 16 --budget 46G
The capital of France is Paris."}
{"role": "user", "  content": "What is the capital of France
[16 tokens, 50.4 s, 0.32 tok/s | experts 2784 hit / 20768 miss = 12%]
```

That is Kimi K3 — 2.78T parameters, 93 layers, 896 experts per layer —
generating on a laptop, from a 982 GB container, inside a 46 GB budget.

**This is the first draft.** It runs, its numbers match a PyTorch
reference to five decimal places, and it is slow. Everything below is the
foundation for a long series of optimizations aimed at running this model
at the highest efficiency the hardware allows. The measurements are a
starting line, published so the next ones have something to beat.

## Why the name

Every token answered by a cloud service is paid for twice: once on the
invoice, and once in the electricity of a datacenter running a model that
would fit — barely, awkwardly, but genuinely — on hardware already sitting
on a desk. WASTE means to be the first concrete step toward ending that
waste of tokens. The acronym came second.

## What it is

- **Self-contained.** One `libwaste.a`, one `waste` binary, nothing at run
  time beyond libc and pthreads.
- **Zero dependencies.** No BLAS, no ONNX, no Python in the inference
  path, nothing to install. The Python under `tools/` converts models and
  validates the engine; it never runs alongside it.
- **Fully embeddable.** Eighteen public functions in
  [src/waste.h](src/waste.h): open a model under a RAM ceiling, generate,
  save the session, close. The CLI is a client of that API and touches
  nothing private — if the CLI can do it, so can an embedding host.

```c
waste_cfg cfg;
waste_cfg_init(&cfg);
cfg.ram_budget_bytes = 46ULL << 30;      /* a hard ceiling, not a hint */

waste_ctx *ctx;
if (waste_open("k3.waste", &cfg, &ctx) != WASTE_OK) return 1;
waste_generate(ctx, ids, n, &params, on_token, user);
waste_close(ctx);
```

## How it works

### Placement decides the speed

A model is converted once into a `.waste` container: a JSON manifest, a
resident trunk, and one expert bank per layer. Each expert record is
4 KiB-aligned with its gate, up and down matrices adjacent, so routing to
an expert costs exactly **one `pread`** — not three, not a seek per
matrix. The arithmetic was never the bottleneck.

Reads bypass the page cache (`F_NOCACHE` on macOS, `O_DIRECT` on Linux).
That is deliberate: with a container smaller than RAM the kernel would
cache everything, and the hit rates measured that way are a fiction that
does not survive contact with a 982 GB model.

### Three bits per expert weight

Experts are stored as residual vector quantization — three stages of
256-entry codebooks over 8-dimensional vectors, 3.00 bits per weight — and
the matrix is never materialized. For each token the engine builds a table
of partial dot products, one per codebook entry per vector position, after
which every expert row is three table reads and two adds.

The trunk stays at 4 and 8 bits. The model was trained with
quantization-aware training on the *experts* only, so it has no trained
tolerance for a squeezed trunk: a 3-bit trunk was built and measured, the
cache prediction held, the throughput did not, and the output collapsed.

### The cache floor is one token's working set

The most predictive number in this project. K3 touches 16 experts in each
of 92 layers per token: **17.0 GB**. Below that, an expert cached for one
token is evicted before the next token asks for it, and the hit rate is
not low — it is zero. Above it the curve bends sharply.

| budget | expert cache | hit rate | decode |
|---|---|---|---|
| 32 GB | 3.1 GB | 0% | 0.31 tok/s |
| 46 GB | 17.1 GB | 12% | 0.32 tok/s |
| 52 GB | 23 GB | 25% | 0.33 tok/s |
| 58 GB | 29.1 GB | 37% | **0.04 tok/s** |

Everything in the memory design exists to get above that line, which is
why the engine works to free RAM rather than to save it.

**And there is a ceiling on the other side.** At 58 GB on a 64 GB machine
the hit rate is the best of the run and the throughput is eight times
worse — reproduced twice. The engine is inside its budget; the *machine*
is not, so the OS pages out the expert cache, and a "hit" becomes a page
fault instead of the disk read the engine was managing. One extra
gigabyte of cache turned 0.32 tok/s into 0.04. The engine now warns when
a budget leaves the machine less than 12% of its RAM, but the real lesson
is that a cache you do not control is not a cache.

### Linear attention, and an absorbed KV cache

K3's attention is a 3:1 hybrid: Kimi Delta Attention, which carries a
fixed-size recurrent state instead of a growing KV cache, and gated
multi-head latent attention. The MLA layers cache the 512-wide latent
rather than expanded per-head keys and values, with `kv_b_proj` absorbed
into the query and the output:

```
q_nope · (W_kb c)    ==  (W_kbᵀ q_nope) · c
Σ_s a_s (W_vb c_s)   ==  W_vb (Σ_s a_s c_s)
```

Identical logits to 1.2e-05, and **53× less cache**: 11.25 GB becomes
0.21 GB at 4K context. It is also what makes long context possible at all
— the expanded layout wants 360 GB at 128K tokens, the latent one 7.2.

## Performance and memory

MacBook Pro M5 Pro, 64 GB, container on the internal SSD. Every figure was
measured on the commit it is published with.

### Kimi K3 — 2.78T parameters, 982 GB container

| | |
|---|---|
| minimum RAM | **29.3 GB** at 4K context |
| | 31.9 GB at 32K, 37.0 GB at 128K |
| resident trunk | 27.5 GB |
| read per token | 17.0 GB at ~9.9 GB/s, near the SSD's measured ceiling |
| model load | 20 s |
| prefill | 0.47 tok/s chunked, 0.29 sequential |
| decode | 0.31–0.33 tok/s, best around a 52 GB budget |

The floor is almost entirely the resident trunk. Useful throughput starts
above ~46 GB, where the expert cache finally clears one token's working
set, and ends around 52 GB, where the machine starts paging. Below the
first line extra RAM buys nothing; above the second it costs.

### Kimi-Linear — 48B parameters, 19 GB container

| | |
|---|---|
| minimum RAM | 1.86 GB |
| decode | **7.25 tok/s** at an 8 GB budget, 77% cache hit |

The same engine and the same format, on a model that fits comfortably.
This is what WASTE looks like when it is not fighting.

### Where the time goes

Decode on K3, cold cache: expert I/O 48.5%, expert matmul 23.8%, KDA
layers 14.9%, MLA 3.2%. The I/O already runs near the hardware limit, so
it only gets cheaper by happening less often — which means cache, which
means RAM. That is the whole optimization story so far, and the reason
the next steps are about memory rather than arithmetic.

## Getting started

```bash
git clone <this repo> && cd waste
make                          # libwaste.a, waste, libwastevq
make check                    # 18 checks, about three minutes
```

No configure step and no dependency resolution. `make check` needs no
model: it builds a small synthetic container and runs the engine against
it.

Converting a model needs Python once:

```bash
uv run --with torch --with safetensors python tools/convert.py \
    --src /path/to/hf-model --out model.waste --jobs 3
```

Then:

```bash
waste run   model.waste "The capital of France is" -n 32 --budget 46G
waste chat  model.waste --budget 46G           # multi-turn, state kept
waste eval  model.waste "2 + 2 =" --top-k 5    # next-token distribution
waste plan  model.waste --budget 46G           # what fits, and what does not
echo "prompt" | waste run model.waste          # stdin works too
```

`waste --help` lists all nine commands. `--json` makes `eval`, `tokenize`,
`plan`, `info` and `bench` machine-readable.

## Platforms

| | build | tests | backend |
|---|---|---|---|
| macOS arm64 | yes | 18/18 | NEON |
| Linux arm64 | yes | 12/12 | NEON |
| Linux x86_64 | yes | 12/12 | AVX2 |
| Windows | branches written, never compiled | — | — |

SIMD is selected at run time from CPUID, so a single x86 binary uses
AVX-512 where it exists and AVX2 where it does not. Accelerator backends
are build-time options. A Metal backend exists and is off by default
because it is correct and 22% slower: this engine issues several hundred
small dependent matvecs per token, the worst possible shape for an
accelerator, and the CPU path already runs at the machine's memory
bandwidth.

## Repository

```
src/        the engine — 6,600 lines of C, no dependencies
  model.c     forward pass, MoE routing, KDA and MLA layers
  ecache.c    bounded LFRU expert cache over the banks
  waste.c     the public API
  simd_*.c    per-ISA kernels, selected at run time
cli/        the CLI, a client of the public API
tools/      conversion and validation (Python, never at run time)
docs/       format, engine, backends, and what was learned
tests/      18 checks, including a diff against a PyTorch oracle
```

[docs/LEARNED.md](docs/LEARNED.md) is the one to read before contributing.
It records what was measured, including the optimizations that were
refuted — index-layout blocking, a 3-bit trunk, GPU offload — with the
numbers that killed them.

## Status

Version 0.1.0. Output is correct, validated layer by layer against a
PyTorch reference on both models. The API is not frozen.

Known gaps, plainly: AVX-512 compiles but has never executed on hardware
that has it; Windows has never been built; there is no container checksum;
and chat templates are applied from a declarative `chat.json` rather than
by interpreting Jinja, so a model whose format is published only as a
Jinja template needs that translated by hand.

## License

Apache 2.0 — see [LICENSE](LICENSE). Copyright 2026 SQLite Cloud, Inc.
