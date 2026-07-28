#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""
vision_ref.py — K3's vision tower in pure PyTorch, off a WASTE container.

The oracle src/vision.c is diffed against, the same role kimi_ref.py plays
for the text path. Reimplemented from the reference `modeling_kimi_k3.py`
rather than imported from it: that module pulls in flash-attention and the
full multimodal wrapper, neither of which exists on this machine, and the
pieces that matter are small enough to transcribe exactly.

The pipeline, for one image (t=1):

  patchify 14x14  ->  Conv2d(3,1024,14,stride 14)   (a matmul over 588)
  + 64x64 learned position embedding, bilinearly resized to the grid
  27 x [ RMSNorm -> packed QKV -> 2D RoPE -> attention -> Wo -> +res
         RMSNorm -> fc0 -> gelu_tanh -> fc1 -> +res ]
  final RMSNorm
  2x2 spatial merge (1024 -> 4096)
  proj.0 -> GELU(erf) -> proj.2  ->  RMSNorm            (4096 -> 7168)

Two details worth stating because they are easy to get wrong: the encoder
MLP uses the tanh approximation of GELU and the projector uses the exact
one, and the RoPE frequency table interleaves the width axis at even
indices with the height axis at odd ones — the reference's docstring says
the opposite of what its code does, and the code is what the weights were
trained against.

  uv run --with torch python tools/vision_ref.py --container k3.waste \\
      --grid 4 6 --dump vis_ref.bin
"""

import argparse
import json
import math
import os
import struct
import sys

import torch
import torch.nn.functional as F

PREFIX_V = "vision_tower."
PREFIX_P = "mm_projector."


class Trunk:
    """Dequantizes trunk tensors on demand — F32 verbatim, Q8G int8 with one
    fp16 scale per group of 128."""

    def __init__(self, path):
        self.man = json.load(open(os.path.join(path, "manifest.json")))
        self.meta = {e["name"]: e for e in self.man["trunk"]}
        self.f = open(os.path.join(path, "trunk.bin"), "rb")
        self.cache = {}

    def __getitem__(self, name):
        if name in self.cache:
            return self.cache[name]
        e = self.meta[name]
        shape = e["shape"]
        n = 1
        for s in shape:
            n *= s
        if e["fmt"] == 0:
            self.f.seek(e["off"])
            t = torch.frombuffer(bytearray(self.f.read(n * 4)),
                                 dtype=torch.float32).view(*shape).clone()
        elif e["fmt"] == 2:
            rows, N = n // shape[-1], shape[-1]
            g = e["group"]
            ng = (N + g - 1) // g
            self.f.seek(e["off"])
            q = torch.frombuffer(bytearray(self.f.read(rows * ng * g)),
                                 dtype=torch.int8).view(rows, ng, g).float()
            self.f.seek(e["scale_off"])
            sc = torch.frombuffer(bytearray(self.f.read(rows * ng * 2)),
                                  dtype=torch.float16).view(rows, ng, 1).float()
            t = (q * sc).view(rows, ng * g)[:, :N].reshape(*shape).clone()
        else:
            raise RuntimeError(f"{name}: fmt {e['fmt']} not supported here — "
                               f"run tools/requant_vision.py first")
        self.cache[name] = t
        return t


def rms_norm(x, w, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def gelu_tanh(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) *
                                       (x + 0.044715 * x.pow(3))))


def rope_freqs(h, w, head_dim, theta=10000.0):
    """(h*w, head_dim/2) complex. Even slots follow the width axis, odd the
    height axis — matching the reference implementation, not its docstring."""
    n = h * w
    flat = torch.arange(n, dtype=torch.float32)
    x_pos, y_pos = flat % w, torch.div(flat, w, rounding_mode="floor")
    dim_range = torch.arange(0, head_dim, 4, dtype=torch.float32)[:head_dim // 4]
    freqs = 1.0 / (theta ** (dim_range / head_dim))
    x_cis = torch.polar(torch.ones(n, len(freqs)), torch.outer(x_pos, freqs))
    y_cis = torch.polar(torch.ones(n, len(freqs)), torch.outer(y_pos, freqs))
    return torch.cat([x_cis.unsqueeze(-1), y_cis.unsqueeze(-1)], -1).reshape(n, -1)


def apply_rope(x, cis):
    """x: (L, heads, head_dim) -> pairs of adjacent components as complex."""
    c = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(c * cis.unsqueeze(-2)).flatten(-2)


def vision_forward(t, pixels, h, w, cfg, dump_stage=None):
    """pixels: (L, 3, 14, 14) with L = h*w. Returns (h/2 * w/2, 7168)."""
    D = cfg["vt_hidden_size"]
    heads = cfg["vt_num_attention_heads"]
    qkv_h = cfg["qkv_hidden_size"]
    hd = qkv_h // heads
    eps = cfg.get("vt_rms_eps", 1e-6)
    L = h * w

    # --- patch embedding: the conv is a matmul over 3*14*14 -----------------
    pw = t[PREFIX_V + "patch_embed.proj.weight"].reshape(D, -1)      # (D, 588)
    x = pixels.reshape(L, -1) @ pw.T                                  # (L, D)

    # --- position embedding, bilinear from the stored 64x64 grid -----------
    # requant_vision.py stores it flattened, as the engine uses it
    pos = t[PREFIX_V + "patch_embed.pos_emb.weight"]
    ph, pw_ = cfg["init_pos_emb_height"], cfg["init_pos_emb_width"]
    pos = pos.reshape(ph, pw_, D)
    if (h, w) != (ph, pw_):
        pos = F.interpolate(pos.permute(2, 0, 1).unsqueeze(0), size=(h, w),
                            mode="bilinear").squeeze(0).permute(1, 2, 0)
    x = x + pos.reshape(L, D)
    if dump_stage == "embed":
        return x

    cis = rope_freqs(h, w, hd)
    scale = 1.0 / math.sqrt(hd)

    for b in range(cfg["vt_num_hidden_layers"]):
        p = f"{PREFIX_V}encoder.blocks.{b}."
        res = x
        y = rms_norm(x, t[p + "norm0.weight"], eps)
        qkv = (y @ t[p + "wqkv.weight"].T).view(L, 3, heads, hd)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        q, k = apply_rope(q, cis), apply_rope(k, cis)
        att = torch.einsum("lhd,mhd->hlm", q, k) * scale
        att = att.softmax(-1)
        o = torch.einsum("hlm,mhd->lhd", att, v).reshape(L, heads * hd)
        x = res + o @ t[p + "wo.weight"].T
        if dump_stage == f"attn{b}":
            return x

        res = x
        y = rms_norm(x, t[p + "norm1.weight"], eps)
        if dump_stage == f"n1_{b}":
            return y
        y = y @ t[p + "mlp.fc0.weight"].T
        if dump_stage == f"fc0_{b}":
            return y[:, :D]                       # wide: first D columns
        y = gelu_tanh(y)
        if dump_stage == f"act_{b}":
            return y[:, :D]
        y = y @ t[p + "mlp.fc1.weight"].T
        if dump_stage == f"fc1_{b}":
            return y
        x = res + y
        if dump_stage == f"block{b}":
            return x

    x = rms_norm(x, t[PREFIX_V + "encoder.final_layernorm.weight"], eps)
    if dump_stage == "encoder":
        return x

    # --- 2x2 spatial merge: (h*w, D) -> (h/2*w/2, 4D) -----------------------
    nh, nw = h // 2, w // 2
    x = x.view(nh, 2, nw, 2, D).permute(0, 2, 1, 3, 4).reshape(nh * nw, 4 * D)

    # --- projector ----------------------------------------------------------
    x = x @ t[PREFIX_P + "proj.0.weight"].T
    x = F.gelu(x)                                   # exact, not the tanh one
    x = x @ t[PREFIX_P + "proj.2.weight"].T
    return rms_norm(x, t[PREFIX_P + "post_norm.weight"],
                    cfg.get("projector_ln_eps", 1e-5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--container", required=True)
    ap.add_argument("--grid", nargs=2, type=int, default=[4, 6],
                    metavar=("H", "W"), help="patch grid, both even")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stage", default=None,
                    help="embed | blockN | encoder — stop early, for bisecting")
    ap.add_argument("--dump", default=None)
    ap.add_argument("--dump-pixels", default=None)
    args = ap.parse_args()

    h, w = args.grid
    if h % 2 or w % 2:
        print("grid must be even in both axes (2x2 merge)", file=sys.stderr)
        return 2

    t = Trunk(args.container)
    vcfg = json.load(open(os.path.join(args.container, "vision.json")))

    torch.manual_seed(args.seed)
    pixels = torch.randn(h * w, 3, 14, 14)
    if args.dump_pixels:
        with open(args.dump_pixels, "wb") as f:
            v = pixels.flatten().tolist()
            f.write(struct.pack(f"<{len(v)}f", *v))

    out = vision_forward(t, pixels, h, w, vcfg, args.stage)
    print(f"grid {h}x{w} -> {tuple(out.shape)}  "
          f"mean {out.mean():+.5f}  std {out.std():.5f}  "
          f"absmax {out.abs().max():.4f}")
    if args.dump:
        with open(args.dump, "wb") as f:
            v = out.flatten().tolist()
            f.write(struct.pack(f"<{len(v)}f", *v))
        print(f"dumped {out.numel()} floats -> {args.dump}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
