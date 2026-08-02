# Changelog

Every release since 0.6.1. Numbers here were measured on the commit they
ship with, the same rule the rest of the documentation follows — where a
change was measured and *not* adopted, that is recorded too, because the
measurement is the useful part.

`docs/LEARNED.md` carries the full reasoning; this file carries what
changed. Each entry names the section to read for the numbers behind it.

## 0.6.3 — 2026-08-02

### Fixed

- **The automatic budget sized against the host's RAM inside a container**
  ([#14](https://github.com/sqliteai/waste/issues/14)). `waste_physical_ram()`
  is `sysconf(_SC_PHYS_PAGES)` on Linux, which reports the host's `MemTotal`
  from inside a cgroup that is allowed a fraction of it, so a `--budget`-less
  open resolved `floor + 3x` against memory the kernel would never hand over:
  K3 in a 32 GiB cgroup on a large host asks for ~80 GB and is killed. Unlike
  the paging cliff of [LEARNED.md](docs/LEARNED.md) §16 this has no gradual
  form and no cache policy softens it. The ceiling is now
  `min(physical, cgroup limit)` — the smallest finite `memory.max` or
  `memory.high` across the cgroup and its ancestors, since the limit is
  hierarchical — and the rest of the resolver is unchanged. See §40.

  Current pressure (`MemAvailable`, `memory.current`) was considered and
  deliberately left out: a budget is resolved once and held for a whole run,
  so bounding it by an instantaneous sample would make the same command on
  the same machine two different runs. Whether it should trim the working-set
  multiplier instead is #14, still open.

### Added

- `waste_usable_ram()`: physical RAM, or a smaller cgroup-v2 limit when one
  applies — what a budget of 0 sizes against, and what an embedding host
  should size its own ceiling from. `waste plan --json` reports it beside
  `physical_ram_bytes`, which stays what it always was.

## 0.6.2 — 2026-08-01

### Fixed

- **`waste info` and `waste run` crashed on K3 on every x86 build**
  ([#10](https://github.com/sqliteai/waste/issues/10)). The tensors the
  loader skips — the vision tower, and anything outside `tensor_prefix` —
  kept `group` at 0, and the row-scratch sizing divided by it. The
  architecture decided what that meant: arm64's `sdiv` answers 0 and the
  run continues, x86's `idiv` raises `#DE`. `waste plan` was unaffected
  because it does not load. §37.
- **`WASTE_Q8=0` could not load a 4-bit trunk**
  ([#6](https://github.com/sqliteai/waste/issues/6)) — that is, any
  container a default `tools/convert.py` run produces. The dequantizer
  read one byte per weight, true of Q8G alone, while catching every
  quantized format. It now decodes through `waste_deq_row`, the one place
  that knows all three widths. The same lines also predated `waste_f16`'s
  subnormal fix and flushed group scales below 6.1e-05 to zero.
- **`embed_tokens` stays on disk under `WASTE_Q8=0`**, as it does
  otherwise: 7.93 → 6.52 GiB of peak RSS on Kimi-Linear, identical logits.
  The f32-equivalence check now differs from the default path in the
  storage width alone, which is what it claims to compare.

### Added

- **Router lookahead in the decode path.** At the end of a MoE layer, once
  its reads are consumed and the disk is about to idle through the next
  layer's attention, layer L+1's router runs on layer L's hidden state and
  issues speculative reads for its top 6. Demand hit rate 14–19% → 38–40%
  with total bytes read unchanged (254.2 → 254.5 GB): the records were
  going to be read anyway, and only *when* changes. Nine paired runs,
  median 1.17x. `WASTE_LOOKAHEAD=0` disables. §34, §35.
- **`WASTE_MLOCK`** wires the trunk and the expert cache; `WASTE_MLOCK=cache`
  wires the cache alone. Off by default — Linux's `RLIMIT_MEMLOCK` is
  commonly 8 MB. Wiring the trunk is worth 3x in the transition zone around
  52 GiB and nothing below it; it does not move the knee. §30, §31, §32.

### Changed

- **`tests/run.sh` generates its own PyTorch oracle** from the container
  under test (16.9 s) instead of diffing against a shipped fixture, which
  can only ever be valid for the container that produced it: expert
  codebooks are k-means, and the same seed on a different `--device`
  trains different books — one layer of 26 moves the logits by 1.24
  against a 1e-3 threshold. The fixture remains as the fallback where `uv`
  is absent, with its provenance recorded beside it.
  ([#7](https://github.com/sqliteai/waste/issues/7)), §33.
- **`tools/make_test_container.py` emits what a real conversion does**: a
  `Q4G/Q8G/F32` trunk rather than Q8G throughout, and `--prefix` for a
  container whose tensors are not all under its `tensor_prefix`. Both are
  shapes the suite could not previously reach, and both had a live bug
  behind them.
- Checks that cannot run now say why instead of reporting a refusal as a
  divergence: `WASTE_Q8=0` on K3 wants 211 GB of f32 trunk on a 64 GB
  machine, the oracle prompt is Kimi-Linear's, and a cold hotlist run that
  already missed nothing demonstrates neither outcome
  ([#5](https://github.com/sqliteai/waste/pull/5)).

### Measured and not adopted

- **Cross-layer prefetch from `next_layer_top`** — gated before building
  and refused: 29.0% recall against a 60% break-even. §29, revisited and
  superseded by the lookahead above in §34.
- **The lookahead in the prefill path.** Built, measured, removed: a chunk
  layer's disk is busy continuously, so a prefetch there does not move a
  read into idle time, it moves it in front of another read and pays an
  eviction for it — 7% more bytes. Decode keeps it. §36.

## 0.6.1 and earlier

Not covered here; see `docs/LEARNED.md`, which is dated and append-only,
and the git history.
