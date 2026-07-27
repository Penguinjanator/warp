#!/usr/bin/env python3
"""
routing_stats.py — answer WASTE open question #1 without downloading 1.5 TB.

Three subcommands, stdlib only (no deps, earlier work discipline):

  fetch     Download config.json + model.safetensors.index.json from the
            HF repo (a few KB/MB — NOT the weights). Works the moment the
            K3 repo goes public (2026-07-27).

  math      From the fetched config/index: exact per-expert record sizes,
            per-token expert I/O at 2/2.5/3/4 bits, trunk residency size,
            and the frozen values for every TBD field in docs/FORMAT.md.

  simulate  Feed a routing trace (JSONL, one line per token:
            {"tok": n, "layers": [[e0..e15], ...]}) and get: unique
            experts/token, cross-token reuse, heavy-tail coverage, and
            simulated LRU vs LFRU cache hit-rate curves vs cache GB —
            i.e. the number that decides tok/s on NVMe.

The trace itself requires running the model once somewhere (rented box or
any instrumented reference impl: hook the router topk and log indices —
see trace_hook_snippet() at the bottom for a transformers/vLLM hook).

Usage:
  tools/routing_stats.py fetch  [--repo moonshotai/Kimi-K3] [--out data/]
  tools/routing_stats.py math   [--data data/] [--bits 2.12,2.5,3.06,4.25]
  tools/routing_stats.py simulate trace.jsonl [--data data/] \
      [--cache-gb 8,16,24,32,40] [--bits 2.12]
"""

import argparse
import json
import math
import os
import sys
import urllib.request
from collections import Counter, OrderedDict

HF_BASE = "https://huggingface.co/{repo}/resolve/main/{fname}"
FILES = ("config.json", "model.safetensors.index.json")


# ---------------------------------------------------------------- fetch ----

def cmd_fetch(args):
    os.makedirs(args.out, exist_ok=True)
    for fname in FILES:
        url = HF_BASE.format(repo=args.repo, fname=fname)
        dst = os.path.join(args.out, fname)
        print(f"GET {url}")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "waste-recon/0"})
            with urllib.request.urlopen(req, timeout=60) as r, open(dst, "wb") as f:
                f.write(r.read())
            print(f"  -> {dst} ({os.path.getsize(dst)} bytes)")
        except Exception as e:  # repo not public yet, gated, renamed…
            print(f"  !! {e}\n     (repo not live yet? try --repo with the exact id)")
            return 1
    return 0


# ----------------------------------------------------------------- math ----

def _cfg(d, *keys, default=None):
    """First matching key wins — DeepSeek-family configs vary in naming."""
    for k in keys:
        if k in d:
            return d[k]
    return default


def load_config(data_dir):
    with open(os.path.join(data_dir, "config.json")) as f:
        cfg = json.load(f)
    # unwrap nested text_config style wrappers if present
    if "num_hidden_layers" not in cfg:
        for v in cfg.values():
            if isinstance(v, dict) and "num_hidden_layers" in v:
                cfg = v
                break
    return cfg


def model_shape(cfg):
    s = {
        "layers":        _cfg(cfg, "num_hidden_layers"),
        "hidden":        _cfg(cfg, "hidden_size"),
        "n_experts":     _cfg(cfg, "n_routed_experts", "num_experts", "num_local_experts"),
        "top_k":         _cfg(cfg, "num_experts_per_tok", "num_experts_per_token", "moe_top_k"),
        "moe_inter":     _cfg(cfg, "moe_intermediate_size"),
        "dense_inter":   _cfg(cfg, "intermediate_size"),
        "n_shared":      _cfg(cfg, "n_shared_experts", "num_shared_experts", default=0),
        "first_dense":   _cfg(cfg, "first_k_dense_replace", default=0),
        "vocab":         _cfg(cfg, "vocab_size"),
        "kv_lora":       _cfg(cfg, "kv_lora_rank"),
        "q_lora":        _cfg(cfg, "q_lora_rank"),
    }
    if s["moe_inter"] is None and s["n_experts"]:
        # Mixtral/OLMoE-style configs size experts with plain intermediate_size.
        # (DeepSeek-family always has moe_intermediate_size — there
        # intermediate_size is the dense FFN and would be wrong here.)
        s["moe_inter"] = s["dense_inter"]
        print("!! moe_intermediate_size missing: falling back to "
              "intermediate_size (Mixtral/OLMoE convention)", file=sys.stderr)
    missing = [k for k, v in s.items() if v is None and k not in ("kv_lora", "q_lora")]
    if missing:
        print(f"!! config keys not recognized for: {missing} — inspect config.json "
              f"and extend _cfg() aliases", file=sys.stderr)
    return s


def cmd_math(args):
    s = model_shape(load_config(args.data))
    print(json.dumps(s, indent=2))
    if any(s[k] is None for k in ("layers", "hidden", "n_experts", "top_k", "moe_inter")):
        return 1

    moe_layers = s["layers"] - s["first_dense"]
    # one expert = gate + up + down = 3 * hidden * moe_inter params
    p_expert = 3 * s["hidden"] * s["moe_inter"]
    p_routed_total = moe_layers * s["n_experts"] * p_expert
    p_active_routed = moe_layers * s["top_k"] * p_expert

    print(f"\nMoE layers: {moe_layers}   params/expert/layer: {p_expert/1e6:.1f} M")
    print(f"Routed total: {p_routed_total/1e12:.2f} T   "
          f"routed active/token: {p_active_routed/1e9:.1f} B")

    print(f"\n{'bits/w':>7} {'expert rec':>11} {'4KiB blks':>9} "
          f"{'disk (routed)':>13} {'GB/token cold':>13}")
    for bits in [float(b) for b in args.bits.split(",")]:
        rec = p_expert * bits / 8
        blks = math.ceil((rec + 48) / 4096)          # + waste_expert_hdr
        disk = p_routed_total * bits / 8
        tok = p_active_routed * bits / 8
        print(f"{bits:>7.2f} {rec/2**20:>9.1f}MB {blks:>9} "
              f"{disk/2**30:>11.0f}GB {tok/2**30:>11.1f}GB")

    for nvme in (3, 7, 12):
        bits = float(args.bits.split(",")[0])
        tok = p_active_routed * bits / 8 / 2**30
        print(f"NVMe {nvme:>2} GB/s, 0% hit: {nvme/tok:.2f} tok/s ceiling"
              f"   (50% hit: {nvme/tok*2:.2f}, 80% hit: {nvme/tok*5:.2f})")
    print("\nFreeze these into docs/FORMAT.md + waste_format.h TBDs.")
    return 0


# ------------------------------------------------------------- simulate ----

def cmd_simulate(args):
    s = model_shape(load_config(args.data))
    bits = float(args.bits)
    rec_gb = 3 * s["hidden"] * s["moe_inter"] * bits / 8 / 2**30

    tokens = []           # list of set((layer, expert)) per token
    freq = Counter()
    with open(args.trace) as f:
        for line in f:
            if not line.strip():
                continue
            t = json.loads(line)
            acts = {(li, e) for li, layer in enumerate(t["layers"]) for e in layer}
            tokens.append(acts)
            freq.update(acts)
    n = len(tokens)
    if n < 2:
        print("need >= 2 tokens in trace", file=sys.stderr)
        return 1

    uniq = sum(len(t) for t in tokens) / n
    reuse1 = sum(len(tokens[i] & tokens[i - 1]) / len(tokens[i])
                 for i in range(1, n)) / (n - 1)
    print(f"tokens: {n}   avg unique (layer,expert)/token: {uniq:.0f} "
          f"({uniq*rec_gb:.1f} GB @ {bits}b)")
    print(f"next-token reuse (the refuted-in-literature number): {reuse1:.1%}")

    total = sum(freq.values())
    ranked = [c for _, c in freq.most_common()]
    for cov in (0.5, 0.8, 0.95):
        acc = k = 0
        for c in ranked:
            acc += c; k += 1
            if acc >= cov * total:
                break
        print(f"top {k} expert-slots ({k*rec_gb:.0f} GB) cover {cov:.0%} of activations")

    # LRU vs LFRU hit-rate at various cache budgets.
    # LFRU uses sampled eviction (min freq over 32 sampled keys, recency
    # tiebreak) — same approximation a real C engine would use; exact
    # min-scan is O(n^2) and pointless precision for a feasibility read.
    import random
    rng = random.Random(0)
    print(f"\n{'cache GB':>8} {'slots':>7} {'LRU hit':>8} {'LFRU hit':>9} "
          f"{'GB/tok miss':>11}")
    for gb in [float(g) for g in args.cache_gb.split(",")]:
        slots = int(gb / rec_gb)
        row = f"{gb:>8.0f} {slots:>7}"
        miss_gb = 0.0
        for policy in ("lru", "lfru"):
            cache = OrderedDict()      # key -> last-access clock
            f2, hits, total_acc, clock = Counter(), 0, 0, 0
            for t in tokens:
                for key in t:
                    total_acc += 1
                    clock += 1
                    f2[key] += 1
                    if key in cache:
                        hits += 1
                        cache.move_to_end(key)
                        cache[key] = clock
                    elif slots > 0:
                        if len(cache) >= slots:
                            if policy == "lru":
                                cache.popitem(last=False)
                            else:
                                keys = list(cache)
                                sample = (keys if len(keys) <= 32 else
                                          rng.sample(keys, 32))
                                victim = min(sample,
                                             key=lambda k: (f2[k], cache[k]))
                                cache.pop(victim)
                        cache[key] = clock
            hr = hits / total_acc
            row += f" {hr:>7.1%}" if policy == "lru" else f" {hr:>8.1%}"
            miss_gb = (1 - hr) * uniq * rec_gb
        print(row + f" {miss_gb:>9.1f}")
    print("\nGB/tok miss ÷ NVMe GB/s = seconds/token. That's the verdict.")
    return 0


def trace_hook_snippet():
    return '''
# transformers hook to produce trace.jsonl (run on any box that fits the
# model, or a cheap short rental — a few hundred tokens of agentic-style
# decode is enough for a first read):
#
#   topk_ids per MoE layer is the tensor to log. Pseudocode:
#
#   trace = []
#   def hook(module, inp, out):            # register on each router/gate
#       ids = out[1] if isinstance(out, tuple) else out.topk_ids
#       layer_ids.append(ids[0, -1].tolist())   # batch 1, last position
#   # after each generated token: trace.append({"tok": i, "layers": layer_ids})
'''


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch")
    f.add_argument("--repo", default="moonshotai/Kimi-K3")
    f.add_argument("--out", default="data")
    f.set_defaults(fn=cmd_fetch)

    m = sub.add_parser("math")
    m.add_argument("--data", default="data")
    m.add_argument("--bits", default="2.12,2.5,3.06,4.25")
    m.set_defaults(fn=cmd_math)

    s = sub.add_parser("simulate")
    s.add_argument("trace")
    s.add_argument("--data", default="data")
    s.add_argument("--cache-gb", default="8,16,24,32,40")
    s.add_argument("--bits", default="2.12")
    s.set_defaults(fn=cmd_simulate)

    args = p.parse_args()
    sys.exit(args.fn(args))


if __name__ == "__main__":
    main()
