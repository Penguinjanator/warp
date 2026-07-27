#!/usr/bin/env python3
"""
memplan.py — what is the minimum RAM needed to run this model, and what
does a given RAM budget buy?

The engine takes a hard RAM ceiling (waste_cfg.ram_budget_bytes), so the
first question is the floor: below it the model cannot run at all. This
computes both, from config.json alone — no weights needed.

  tools/memplan.py --data data/            # floor + budget ladder
  tools/memplan.py --data data/ --budget 64 --ctx 131072
  tools/memplan.py --data data/ --trunk-bits 4.25 --expert-bits 2.12

Everything is derived from the config with explicit, printed assumptions.
Gate 1 replaces the analytic trunk estimate with exact tensor sizes read
from the safetensors shard headers.
"""

import argparse
import json
import math
import os
import sys

GB = 1 << 30
MB = 1 << 20


def _cfg(d, *keys, default=None):
    for k in keys:
        if k in d:
            return d[k]
    return default


def load_shape(data_dir):
    with open(os.path.join(data_dir, "config.json")) as f:
        cfg = json.load(f)
    if "num_hidden_layers" not in cfg:
        for v in cfg.values():
            if isinstance(v, dict) and "num_hidden_layers" in v:
                cfg = v
                break
    s = {
        "layers":      _cfg(cfg, "num_hidden_layers"),
        "hidden":      _cfg(cfg, "hidden_size"),
        "n_experts":   _cfg(cfg, "n_routed_experts", "num_experts", "num_local_experts"),
        # Kimi spells these differently from the DeepSeek family
        "top_k":       _cfg(cfg, "num_experts_per_tok", "num_experts_per_token", "moe_top_k"),
        "moe_inter":   _cfg(cfg, "moe_intermediate_size"),
        "dense_inter": _cfg(cfg, "intermediate_size"),
        "n_shared":    _cfg(cfg, "n_shared_experts", "num_shared_experts", default=0) or 0,
        "first_dense": _cfg(cfg, "first_k_dense_replace", default=0) or 0,
        "vocab":       _cfg(cfg, "vocab_size"),
        "n_heads":     _cfg(cfg, "num_attention_heads"),
        "kv_lora":     _cfg(cfg, "kv_lora_rank"),
        "q_lora":      _cfg(cfg, "q_lora_rank"),
        "qk_rope":     _cfg(cfg, "qk_rope_head_dim", default=64),
        "qk_nope":     _cfg(cfg, "qk_nope_head_dim", default=128),
        "v_head":      _cfg(cfg, "v_head_dim", "head_dim", default=128),
        "tie_emb":     bool(_cfg(cfg, "tie_word_embeddings", default=False)),
    }
    if s["moe_inter"] is None:
        s["moe_inter"] = s["dense_inter"]
    # Kimi-Linear/K3 declare the KDA vs full-attention split explicitly
    lac = cfg.get("linear_attn_config") or {}
    s["kda_layers_list"] = lac.get("kda_layers")
    s["kda_heads"] = lac.get("num_heads")
    s["kda_head_dim"] = lac.get("head_dim")
    s["conv_k"] = lac.get("short_conv_kernel_size")
    return s, cfg


def plan(s, args):
    L = s["layers"]
    H = s["hidden"]
    moe_layers = L - s["first_dense"]
    dense_layers = s["first_dense"]
    tb = args.trunk_bits / 8.0     # bytes per weight, trunk
    eb = args.expert_bits / 8.0    # bytes per weight, experts

    p_expert = 3 * H * s["moe_inter"]
    expert_rec = p_expert * eb

    # --- trunk: everything that must stay resident ------------------------
    emb = s["vocab"] * H * tb
    head = 0 if s["tie_emb"] else s["vocab"] * H * tb

    # attention: MLA-style if the LoRA ranks are present, else 4*H^2
    if s["kv_lora"]:
        nh = s["n_heads"] or 64
        qk = s["qk_nope"] + s["qk_rope"]
        p_attn = (H * (s["q_lora"] or H)          # q down-proj
                  + (s["q_lora"] or H) * nh * qk  # q up-proj
                  + H * (s["kv_lora"] + s["qk_rope"])          # kv down
                  + s["kv_lora"] * nh * (s["qk_nope"] + s["v_head"])  # kv up
                  + nh * s["v_head"] * H)         # o-proj
        attn_note = "MLA (LoRA ranks from config)"
    else:
        nh = s["n_heads"] or 32
        p_attn = 4 * H * H
        attn_note = "generic 4*H^2 (no MLA ranks in config)"
    attn = L * p_attn * tb

    router = moe_layers * s["n_experts"] * H * 4          # f32, always
    shared = moe_layers * s["n_shared"] * p_expert * tb
    dense_ffn = dense_layers * 3 * H * s["dense_inter"] * tb
    norms = L * 4 * H * 4
    trunk = emb + head + attn + router + shared + dense_ffn + norms

    # --- state: KDA is O(1) in ctx, MLA KV is per token -------------------
    ctx = args.ctx
    # prefer the config's explicit KDA layer list over the --kda-ratio guess
    if s.get("kda_layers_list"):
        kda_layers = len(s["kda_layers_list"])
        heads = s["kda_heads"] or s["n_heads"] or 64
        d_state = s["kda_head_dim"] or args.kda_dstate
        src = "config"
    elif args.kda_ratio > 0:
        kda_layers = int(L * args.kda_ratio / (args.kda_ratio + 1))
        heads = s["n_heads"] or 64
        d_state = args.kda_dstate
        src = f"--kda-ratio {args.kda_ratio}"
    else:
        kda_layers, heads, d_state, src = 0, 0, 0, "all-MLA"
    mla_layers = L - kda_layers
    kda_state = kda_layers * heads * d_state * d_state * 4
    # short-conv ring buffers: 3 projections x (k-1) taps x proj width
    conv_k = s.get("conv_k") or 4
    kda_state += kda_layers * 3 * (conv_k - 1) * heads * d_state * 4
    per_tok = (s["kv_lora"] + s["qk_rope"]) if s["kv_lora"] else 2 * H
    mla_kv = mla_layers * ctx * per_tok * 2                # f16 latent
    state = kda_state + mla_kv

    # --- scratch: activations, dequant staging, per-thread buffers --------
    scratch = (16 * H * 4 * max(args.threads, 1)          # residual/attn temps
               + s["moe_inter"] * 4 * 8 * max(args.threads, 1)
               + s["vocab"] * 4                            # logits
               + 64 * MB)                                  # slabs, misc

    # --- minimum expert cache: one layer's top-k, double-buffered ---------
    min_cache = s["top_k"] * expert_rec * 2

    floor = trunk + state + scratch + min_cache
    return {
        "moe_layers": moe_layers, "p_expert": p_expert, "expert_rec": expert_rec,
        "emb": emb, "head": head, "attn": attn, "attn_note": attn_note,
        "router": router, "shared": shared, "dense_ffn": dense_ffn,
        "norms": norms, "trunk": trunk,
        "kda_layers": kda_layers, "mla_layers": mla_layers, "kda_src": src,
        "kda_state": kda_state, "mla_kv": mla_kv, "state": state,
        "scratch": scratch, "min_cache": min_cache, "floor": floor,
        "routed_total_disk": moe_layers * s["n_experts"] * p_expert * eb,
        "per_token_io": moe_layers * s["top_k"] * p_expert * eb,
    }


# Gate 0 measured LFRU hit rate vs cache fraction on a real MoE trace
# (OLMoE, 299 decode tokens): fraction of all expert bytes held in cache
# -> hit rate. Used only to project tok/s; replaced by Gate 2 on K3.
HIT_CURVE = [(0.00, 0.00), (0.06, 0.23), (0.125, 0.35), (0.25, 0.53),
             (0.53, 0.81), (1.00, 1.00)]


def hit_rate(frac):
    for i in range(1, len(HIT_CURVE)):
        x0, y0 = HIT_CURVE[i - 1]
        x1, y1 = HIT_CURVE[i]
        if frac <= x1:
            t = (frac - x0) / (x1 - x0) if x1 > x0 else 0
            return y0 + t * (y1 - y0)
    return 1.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--trunk-bits", type=float, default=4.25)
    ap.add_argument("--expert-bits", type=float, default=2.12)
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--budget", type=float, default=None,
                    help="RAM budget in GB; omit for the ladder")
    ap.add_argument("--disk-gbs", type=float, default=12.78,
                    help="measured random-read GB/s of the runtime disk")
    ap.add_argument("--kda-ratio", type=float, default=3.0,
                    help="KDA:MLA layer ratio (Kimi Linear used 3:1); 0=all MLA")
    ap.add_argument("--kda-dstate", type=int, default=128)
    args = ap.parse_args()

    s, raw = load_shape(args.data)
    if not all(s[k] for k in ("layers", "hidden", "n_experts", "top_k", "moe_inter")):
        print("!! config missing required keys:", s, file=sys.stderr)
        return 1
    p = plan(s, args)

    print(f"model: {s['layers']}L hidden={s['hidden']} experts={s['n_experts']} "
          f"top_k={s['top_k']} moe_inter={s['moe_inter']} vocab={s['vocab']}")
    print(f"assumptions: trunk {args.trunk_bits}b, experts {args.expert_bits}b, "
          f"ctx {args.ctx}, {args.threads} threads, attention = {p['attn_note']}")
    print(f"             KDA:MLA = {p['kda_layers']}:{p['mla_layers']} layers "
          f"(from {p['kda_src']})")

    print("\n-- resident trunk (must always be in RAM) --")
    for k, label in (("emb", "embeddings"), ("head", "lm head"),
                     ("attn", "attention"), ("router", "routers (f32)"),
                     ("shared", "shared experts"), ("dense_ffn", "dense FFN layers"),
                     ("norms", "norms")):
        print(f"   {label:<22} {p[k]/GB:8.2f} GB")
    print(f"   {'TRUNK':<22} {p['trunk']/GB:8.2f} GB")

    print("\n-- per-session state --")
    print(f"   {'KDA state (O(1))':<22} {p['kda_state']/GB:8.2f} GB")
    print(f"   {'MLA latent KV @ctx':<22} {p['mla_kv']/GB:8.2f} GB")
    print(f"   {'scratch/activations':<22} {p['scratch']/GB:8.2f} GB")
    print(f"   {'min expert cache':<22} {p['min_cache']/GB:8.2f} GB "
          f"(top_k={s['top_k']} x {p['expert_rec']/MB:.1f} MB x2)")

    print(f"\n== RAM FLOOR: {p['floor']/GB:.2f} GB ==")
    print(f"   (expert set on disk: {p['routed_total_disk']/GB:.0f} GB; "
          f"cold I/O {p['per_token_io']/GB:.1f} GB/token)")

    budgets = [args.budget] if args.budget else [8, 16, 24, 32, 48, 64, 96, 128]
    print(f"\n{'budget':>7} {'cache':>8} {'frac':>6} {'hit*':>6} "
          f"{'GB/token':>9} {'tok/s*':>7}")
    for b in budgets:
        avail = b * GB - p["floor"] + p["min_cache"]
        if avail <= 0:
            print(f"{b:>6.0f}G {'--':>8} {'--':>6} {'--':>6} {'--':>9} "
                  f"{'BELOW FLOOR':>7}")
            continue
        frac = min(avail / p["routed_total_disk"], 1.0)
        h = hit_rate(frac)
        io = p["per_token_io"] * (1 - h)
        tps = args.disk_gbs / (io / GB) if io > 0 else float("inf")
        print(f"{b:>6.0f}G {avail/GB:>7.1f}G {frac:>5.1%} {h:>5.0%} "
              f"{io/GB:>8.1f} {tps:>7.2f}")
    print("\n* hit rate interpolated from the Gate 0 OLMoE curve — a stand-in "
          "until Gate 2 measures K3's own routing. tok/s counts disk I/O only.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
