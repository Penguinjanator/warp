#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""
ds41_preflight.py — every shape the engine will demand, against the
checkpoint's own index, before the conversion writes a byte.

docs/GLM.md on the same check for that release: "caught before the first
conversion rather than after it, by checking every name and shape the
engine will demand against the checkpoint's own index — 1246 tensors, 0
missing, 0 mismatched". The reason is the arithmetic of the failure: a load
refuses on the FIRST missing name, and on a 510 GB checkpoint the hours are
already spent by then.

It runs the other direction from `ds41_check_names` in convert.py. That one
asks whether every name in the checkpoint maps to something the engine
looks up; this one asks whether everything the engine looks up is in the
checkpoint, at the shape the config implies. Both are silent when wrong and
neither implies the other.

  python3 tools/ds41_preflight.py /path/to/DeepSeek-V4.1-Flash

Reads only the safetensors headers — seconds, not a pass over the weights.
"""

import json
import os
import struct
import sys


def check(src):
    """(missing, wrong, expert_missing, n_checked). Prints nothing."""
    cfg = json.load(open(os.path.join(src, "config.json")))
    tc, vc = cfg["text_config"], cfg["vision_config"]
    with open(os.path.join(src, "model.safetensors.index.json")) as f:
        wm = json.load(f)["weight_map"]

    headers, shapes = {}, {}

    def src_shape(name):
        if name not in shapes:
            fn = wm[name]
            if fn not in headers:
                with open(os.path.join(src, fn), "rb") as fh:
                    (n,) = struct.unpack("<Q", fh.read(8))
                    headers[fn] = json.loads(fh.read(n))
            shapes[name] = tuple(headers[fn][name]["shape"])
        return shapes[name]

    hid, nh, hd = tc["hidden_size"], tc["num_attention_heads"], tc["head_dim"]
    ql, G, R = tc["q_lora_rank"], tc["o_groups"], tc["o_lora_rank"]
    E, mi, H = tc["n_routed_experts"], tc["moe_intermediate_size"], tc["hc_mult"]
    ID, IH = tc["index_head_dim"], tc["index_n_heads"]
    NL = tc["num_hidden_layers"]
    ratios = tc["compress_ratios"]
    kv_src = set(tc["kv_source_layer_ids"])
    ix_src = set(tc["index_source_layer_ids"])
    eg = tc["engram_layer_ids"]
    eg_dim = tc["engram_head_dim"]
    eg_h, eg_ng = tc["engram_n_heads"], tc["engram_max_ngram_size"]

    want = {}

    def w(name, *dims):
        want[name] = tuple(dims)

    w("embed.weight", tc["vocab_size"], hid)
    w("head.weight", tc["vocab_size"], hid)
    w("norm.weight", hid)
    for L in range(NL):
        p = f"layers.{L}."
        w(p + "attn_norm.weight", hid)
        w(p + "ffn_norm.weight", hid)
        for site in ("attn", "ffn"):
            w(p + f"hc_{site}_fn", (2 + H) * H, H * hid)
            w(p + f"hc_{site}_base", (2 + H) * H)
            w(p + f"hc_{site}_scale", 3)
        w(p + "attn.wq_a.weight", ql, hid)
        w(p + "attn.q_norm.weight", ql)
        w(p + "attn.wq_b.weight", nh * hd, ql)
        w(p + "attn.wkv.weight", hd, hid)
        w(p + "attn.kv_norm.weight", hd)
        # block-diagonal over o_groups: rows are the groups' ranks, columns
        # are ONE group's heads
        w(p + "attn.wo_a.weight", G * R, nh * hd // G)
        w(p + "attn.wo_b.weight", hid, G * R)
        w(p + "attn.attn_sink", nh)
        if L in kv_src:
            w(p + "attn.compressor.wkv.weight", hd, hid)
            w(p + "attn.compressor.norm.weight", hd)
            # at ratio 1 there is nothing to pool, so no gate — which is
            # why the release ships four wkv and three wgate
            if ratios[L] > 1:
                w(p + "attn.compressor.wgate.weight", hd, hid)
        if L in ix_src:
            w(p + "attn.indexer.wq_b.weight", IH * ID, ql)
            w(p + "attn.indexer.weights_proj.weight", IH, hid)
            if L in kv_src:
                w(p + "attn.indexer.wk.weight", ID, hd)
                w(p + "attn.indexer.k_norm.weight", ID)
        if L in eg:
            w(p + "engram.wkv.weight", hid * (H + 1),
              (eg_ng - 1) * eg_h * eg_dim)
            w(p + "engram.q_weight", H, hid)
            w(p + "engram.k_weight", H, hid)
        w(p + "ffn.gate.weight", E, hid)
        w(p + "ffn.gate.bias", E)
        w(p + "ffn.gate.bias_vl", E)
        w(p + "ffn.shared_experts.w1.weight", mi, hid)
        w(p + "ffn.shared_experts.w3.weight", mi, hid)
        w(p + "ffn.shared_experts.w2.weight", hid, mi)

    vd, vi = vc["hidden_size"], vc["intermediate_size"]
    w("vision.patch_embed.proj.weight", vd, 3 * vc["patch_size"] ** 2)
    w("vision.patch_embed.proj.bias", vd)
    for b in range(vc["num_hidden_layers"]):
        a = f"vision.blocks.{b}."
        w(a + "norm1.weight", vd)
        w(a + "norm2.weight", vd)
        w(a + "attn.wqkv.weight", 3 * vd, vd)
        w(a + "attn.wqkv.bias", 3 * vd)
        w(a + "attn.wo.weight", vd, vd)
        w(a + "attn.wo.bias", vd)
        w(a + "mlp.w1.weight", 2 * vi, vd)      # gate and up in one weight
        w(a + "mlp.w2.weight", vd, vi)
    w("vision.norm.weight", vd)
    w("aligner.w1.weight", hid, vd * vc["downsample_ratio"] ** 2)
    w("aligner.w1.bias", hid)
    w("aligner.w2.weight", hid, hid)
    w("aligner.w2.bias", hid)
    for nm in ("image_start", "image_end", "image_newline"):
        w(nm, hid)

    missing, wrong = [], []
    for name, dims in sorted(want.items()):
        if name not in wm:
            missing.append(name)
        elif src_shape(name) != dims:
            wrong.append((name, src_shape(name), dims))

    # The Engram tables, whose row count the config states twice — here and
    # in the primes tools/ds41_engram.py draws.
    for i, L in enumerate(eg):
        n = f"layers.{L}.engram.embed.weight"
        if n not in wm:
            missing.append(n)
        else:
            dims = (tc["engram_num_embeddings"][i], eg_dim)
            if src_shape(n) != dims:
                wrong.append((n, src_shape(n), dims))

    exp_missing = sum(
        1 for L in range(NL) for e in range(E) for t in ("w1", "w2", "w3")
        if f"layers.{L}.ffn.experts.{e}.{t}.weight" not in wm)
    return missing, wrong, exp_missing, len(want) + len(eg)


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "/Volumes/WasteDisk/ds41"
    missing, wrong, exp_missing, n = check(src)
    print(f"{n} named tensors the engine will demand")
    print(f"  missing  {len(missing)}")
    for m in missing[:10]:
        print(f"    {m}")
    print(f"  wrong shape {len(wrong)}")
    for name, got, dims in wrong[:10]:
        print(f"    {name}: index says {got}, engine wants {dims}")
    print(f"  routed expert tensors missing: {exp_missing}")
    return 1 if (missing or wrong or exp_missing) else 0


if __name__ == "__main__":
    sys.exit(main())
