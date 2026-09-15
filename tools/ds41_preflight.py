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

SRC = sys.argv[1] if len(sys.argv) > 1 else "/Volumes/WasteDisk/ds41"

cfg = json.load(open(os.path.join(SRC, "config.json")))
tc = cfg["text_config"]
vc = cfg["vision_config"]
wm = json.load(open(os.path.join(SRC, "model.safetensors.index.json")))["weight_map"]

# shapes, read from the safetensors headers of the shards that hold them
shards = {}
shape = {}


def header(fn):
    if fn not in shards:
        with open(os.path.join(SRC, fn), "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            shards[fn] = json.loads(f.read(n))
    return shards[fn]


def src_shape(name):
    if name not in shape:
        h = header(wm[name])
        shape[name] = tuple(h[name]["shape"])
    return shape[name]


hid = tc["hidden_size"]
nh = tc["num_attention_heads"]
hd = tc["head_dim"]
rd = tc["qk_rope_head_dim"]
ql = tc["q_lora_rank"]
G, R = tc["o_groups"], tc["o_lora_rank"]
E = tc["n_routed_experts"]
mi = tc["moe_intermediate_size"]
H = tc["hc_mult"]
ID, IH = tc["index_head_dim"], tc["index_n_heads"]
NL = tc["num_hidden_layers"]
ratios = tc["compress_ratios"]
kv_src = set(tc["kv_source_layer_ids"])
ix_src = set(tc["index_source_layer_ids"])
eg = tc["engram_layer_ids"]
eg_dim, eg_h, eg_ng = tc["engram_head_dim"], tc["engram_n_heads"], tc["engram_max_ngram_size"]

want = {}          # source name -> expected shape


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
    w(p + "attn.wo_a.weight", G * R, nh * hd // G)
    w(p + "attn.wo_b.weight", hid, G * R)
    w(p + "attn.attn_sink", nh)
    if L in kv_src:
        w(p + "attn.compressor.wkv.weight", hd, hid)
        w(p + "attn.compressor.norm.weight", hd)
        if ratios[L] > 1:
            w(p + "attn.compressor.wgate.weight", hd, hid)
    if L in ix_src:
        w(p + "attn.indexer.wq_b.weight", IH * ID, ql)
        w(p + "attn.indexer.weights_proj.weight", IH, hid)
        if L in kv_src:
            w(p + "attn.indexer.wk.weight", ID, hd)
            w(p + "attn.indexer.k_norm.weight", ID)
    if L in eg:
        cols = (eg_ng - 1) * eg_h * eg_dim
        w(p + "engram.wkv.weight", hid * (H + 1), cols)
        w(p + "engram.q_weight", H, hid)
        w(p + "engram.k_weight", H, hid)
    w(p + "ffn.gate.weight", E, hid)
    w(p + "ffn.gate.bias", E)
    w(p + "ffn.gate.bias_vl", E)
    w(p + "ffn.shared_experts.w1.weight", mi, hid)
    w(p + "ffn.shared_experts.w3.weight", mi, hid)
    w(p + "ffn.shared_experts.w2.weight", hid, mi)
# the tower
vd, vh, vi = vc["hidden_size"], vc["num_attention_heads"], vc["intermediate_size"]
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
    w(a + "mlp.w1.weight", 2 * vi, vd)
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
        continue
    got = src_shape(name)
    # fp4 experts and fp8 weights are stored packed; the trunk ones here are
    # all fp8 or bf16, which keep their logical shape.
    if got != dims:
        wrong.append((name, got, dims))

# the experts, one layer's worth sampled and the rest checked for presence
exp_missing = 0
for L in range(NL):
    for e in range(E):
        for t in ("w1", "w2", "w3"):
            if f"layers.{L}.ffn.experts.{e}.{t}.weight" not in wm:
                exp_missing += 1
# the engram tables
for i, L in enumerate(eg):
    n = f"layers.{L}.engram.embed.weight"
    if n not in wm:
        missing.append(n)
    else:
        got = src_shape(n)
        if got != (tc["engram_num_embeddings"][i], eg_dim):
            wrong.append((n, got, (tc["engram_num_embeddings"][i], eg_dim)))

print(f"{len(want) + 2} named tensors the engine will demand")
print(f"  missing  {len(missing)}")
for m in missing[:10]:
    print(f"    {m}")
print(f"  wrong shape {len(wrong)}")
for n, g, d in wrong[:10]:
    print(f"    {n}: index says {g}, engine wants {d}")
print(f"  routed expert tensors missing: {exp_missing} of {NL * E * 3}")
sys.exit(1 if (missing or wrong or exp_missing) else 0)
