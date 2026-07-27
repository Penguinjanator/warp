# Engine shape: embeddable library + CLI, under a hard RAM ceiling

Three scope requirements (Marco, 2026-07-27):

1. the C engine is **embeddable** — a library any host program can link;
2. it ships a **fully featured CLI** that is itself a *client* of that
   library, with no private back door;
3. the engine runs under a **configured maximum RAM**, which first requires
   knowing the **minimum RAM** the model needs at all.

## 1. Library first

The public surface is [src/waste.h](../src/waste.h): opaque `waste_ctx`,
no global state (several models can be open at once), errors returned
never printed, nothing calls `exit()`, no dependencies beyond C11 + libc.

Capability set exposed to hosts: memory planning before load, open/close,
tokenize/detokenize, `waste_generate` with a per-token callback (carrying
cache hit/miss and I/O timing so a host can draw a real progress UI),
lower-level `waste_eval` for hosts that do their own sampling, session
state save/load, model introspection and aggregate stats.

Deliberately *not* in the API: logging to stdout, signal handlers, config
files, argument parsing. Those belong to the host — the CLI included.

## 2. CLI as a first-class client

`cli/` links the library and adds only host concerns: argv parsing, a
terminal renderer for the token callback, REPL/history, an OpenAI-shaped
server mode, and file I/O. **Rule: if the CLI needs a capability, it goes
into `waste.h` first.** That keeps the embedded path honest — anything a
user can do from the shell, a host program can do from C.

Planned commands: `run` (one-shot / piped), `chat` (REPL with state
persistence), `serve` (HTTP), `bench` (throughput + cache stats),
`plan` (print the memory plan for a model and budget — the CLI face of
`waste_plan_memory`), `convert` (MXFP4 → WASTE container), `inspect`
(container/manifest dump).

## 3. RAM budget: the floor, then the ceiling

`waste_cfg.ram_budget_bytes` is a hard ceiling on **everything** the
engine allocates — trunk, state, scratch, expert cache. The engine sizes
its expert cache to fit inside what remains after the mandatory parts, and
refuses to open with `WASTE_E_RAM_BUDGET` if the budget is under the
floor, rather than thrashing the machine into swap.

### What the floor is made of

| part | scales with | can it shrink? |
|---|---|---|
| trunk (embeddings, LM head, attention, routers, shared experts, norms) | model, trunk bit-width | only by quantizing harder |
| KDA recurrent state | layers × heads × d_state² — **not** context | no |
| MLA latent KV | context length (compressed) | shorter ctx |
| scratch / activations | threads, hidden, vocab | fewer threads |
| minimum expert cache | top_k × expert record × 2 (double-buffered) | no |

Everything above the floor is expert cache, and that is the only knob
that buys speed.

### Measured for a K3-shaped config (60L, H=7168, 896 experts, top_k=16,
### experts @2.12 bit, trunk @4.25 bit, ctx 32k — `tools/memplan.py`)

```
trunk                     10.32 GB   (attention 6.25, routers 1.41,
                                      shared 1.29, emb+head 1.16)
KDA state (O(1) in ctx)    0.18 GB
MLA latent KV @32k         0.53 GB
scratch                    0.07 GB
min expert cache           0.35 GB   (16 x 11.1 MB x 2)
-------------------------------------------------
RAM FLOOR                 11.44 GB
```

**So the model can technically run in ~12 GB of RAM** — it just reads
~10.3 GB from disk per token. RAM above the floor converts to hit rate:

| budget | expert cache | cache fraction | hit rate* | GB read/token | tok/s* |
|---|---|---|---|---|---|
| 16 GB | 4.9 GB | 0.9% | 3% | 9.9 | 1.29 |
| 32 GB | 20.9 GB | 3.6% | 14% | 8.8 | 1.45 |
| **64 GB** | **52.9 GB** | **9.2%** | **29%** | **7.3** | **1.75** |
| 128 GB | 116.9 GB | 20.3% | 46% | 5.5 | 2.32 |

\* hit rate interpolated from the Gate 0 OLMoE curve; tok/s counts disk
I/O only (12.78 GB/s internal SSD, Gate H) and ignores compute. Both get
replaced by measurements at Gate 2.

Two things this changes:

- **64 GB is not a cliff, it is a point on a gentle curve.** Going from
  16 GB to 128 GB moves throughput less than 2×, because with 575 GB of
  experts even 117 GB of cache holds only a fifth of them. The dominant
  lever is not RAM, it is *bytes read per token* — i.e. bit-width and
  expert pruning.
- Numbers assume the analytic trunk estimate. Gate 1 replaces it with
  exact tensor sizes from the shard headers.

## Verification bar

The CLI must never be able to do something the library cannot, and
`waste_plan_memory` must agree with `waste_memory_used` after open (a
test asserts it). Any allocation path that can exceed the budget is a bug,
not a tuning issue.
