# Changelog

Every release since 0.6.1. Numbers here were measured on the commit they
ship with, the same rule the rest of the documentation follows — where a
change was measured and *not* adopted, that is recorded too, because the
measurement is the useful part.

`docs/LEARNED.md` carries the full reasoning; this file carries what
changed. Each entry names the section to read for the numbers behind it.

## 0.6.3 — 2026-08-03

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

- **`tools/convert.py` spawned `--jobs` × cores threads**
  ([#13](https://github.com/sqliteai/waste/pull/13), contributed by
  @andrewwhitecdw). torch sizes its intra-op pool from `os.cpu_count()`, and
  the native VQ encoder reads `nthreads=0` as "every core" (capped at 64), so
  N worker processes meant N×cpus threads competing for the machine: on a
  224-core box `--jobs 8` spawned ~1792 threads and the codebook phase ran
  ~20x slower than the measured baseline. Each worker is now capped at its
  fair share, `cpu_count // jobs`, for both pools — set in the parent before
  the workers spawn, since that is the only point torch reads it, and with
  `setdefault`, so an explicit `OMP_NUM_THREADS` stays the caller's.

- **The trace simulator modelled a different cache than the engine.**
  `tools/routing_stats.py simulate` kept a frequency count across evictions
  that `ec_claim` resets and sampled 32 victims where `EC_SAMPLE` is 16:
  against the same trace it read **36.6% where the engine measured 30.4%** —
  optimistic, plausible and wrong. Both constants now come from `ecache.c`,
  which brings it within 1.5 points across a 0–30% range, and `tests/run.sh`
  asserts the agreement rather than remembering it. Two smaller things went
  with it: the route dump writes the absolute position of the token each row
  belongs to, so readers stop re-deriving token boundaries from where the
  layer index wraps (a heuristic that is simply wrong on the chunked path,
  where rows group by layer), and `simulate --data` takes a container, whose
  manifest states the record size the engine actually `pread`s. See §37,
  *The simulator was modelling a different cache*.

### Added

- `waste_usable_ram()`: physical RAM, or a smaller cgroup-v2 limit when one
  applies — what a budget of 0 sizes against, and what an embedding host
  should size its own ceiling from. `waste plan --json` reports it beside
  `physical_ram_bytes`, which stays what it always was.

- **`tests/sweep.c`, a one-process measurement harness.** It loads a
  container once and runs the arms back to back, interleaved, resetting the
  session and clearing the expert cache between each — a warm cache would
  hand the second arm the first one's work and measure the order instead of
  the setting. Kimi-Linear, two arms, three repeats: spreads of 2.6% and
  0.5%, where nine paired runs of the same comparison across processes
  spanned 0.79x to 1.79x. The variance was the harness, not the feature. On
  K3 the deterministic columns come out exact and the clock still drifts,
  which is the machine's memory system rather than the process — what the
  harness buys there is that the noise is visible as noise. §38.

- **`docs/TECHNICAL.md` and `examples/`.** The measurement tables move out of
  `README.md` into TECHNICAL.md, and `examples/` carries three compilable
  programs against the public header — `api_plan.c` (budget arithmetic
  without loading), `api_text.c`, `api_vision.c` — with a README that walks
  through them.

### Changed

- **§4's cache floor still reproduces exactly, and has stopped binding.**
  The oldest load-bearing measurement here — below one token's working set
  the hit rate is zero, not low — was re-measured across four cache sizes in
  one process, two repeats, hit rate and bytes read identical to the digit
  across both:

  | budget | expert cache | slots | hit | decode |
  |---|---|---|---|---|
  | 32 GB | 3.32 GB | 287 | 29.1% | 0.56–0.58 tok/s |
  | 46 GB | 17.32 GB | 1498 | 36.2% | **0.63 tok/s** |
  | 52 GB | 23.32 GB | 2018 | 38.4% | 0.07–0.09 tok/s |
  | 58 GB | 29.32 GB | 2537 | 41.3% | 0.07–0.08 tok/s |

  The 29.1% at 287 slots is not a refutation of §4: with the lookahead off
  the same 287 slots give **0.0%**, exactly §4's zero. What breaks it is that
  a speculative record has to survive one attention rather than one token, so
  a cache far too small for a token's working set is ample to hold six
  experts. A 3.32 GB cache is now within 10% of a 17.32 GB one, which means
  the premise the default resolver is built on — that cache is only worth
  buying in whole multiples of a working set — no longer holds. **The
  resolver is unchanged**: that is a decision, not a measurement, and it is
  [GATES.md](docs/GATES.md) Gate 7, open. The cliff is exactly where it was,
  and the last two rows say what it is — throughput falls eightfold while the
  hit rate rises and the bytes read fall, because the engine is inside its
  budget and the machine is not. §39.

- **0.6.2's "total bytes read unchanged" for the router lookahead was the
  harness.** Measured with both arms starting from an identically cleared
  cache, the byte economics depend on the cache size: **6.6% fewer** bytes at
  1498 slots, **8% more** at 287, where speculative records are evicted
  before use often enough to be re-read. It is a prefetch at small caches and
  a scheduling change at large ones, and 0.6.2 measured only the large end.
  The feature and its default are unchanged. §38, §39.

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
