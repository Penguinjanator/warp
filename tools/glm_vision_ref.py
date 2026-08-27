#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""
glm_vision_ref.py — GLM-5.3-Flash's vision tower in pure PyTorch, off a
WASTE container.

The oracle `src/vision.c`'s GLM path is diffed against, the same role
`vision_ref.py` plays for K3's tower. A second file rather than a branch in
that one, because the two towers agree on almost nothing below the block
level and a shared function full of `if tower == ...` would be harder to
check than two that each say one thing.

Transcribed from `modeling_glm5_next.py` rather than imported from it: that
module pulls in the whole multimodal wrapper and a flash-attention path
that does not exist here, and the pieces that matter are small.

The pipeline, for one image at grid (h, w) in patches:

  patchify 14x14 x 2 temporal   ->  Conv3d(3,1024,(2,14,14))  (a matmul
                                    over 1176, plus a bias)
  24 x [ RMSNorm -> qkv+bias -> per-head RMSNorm on q and k -> 2D RoPE
         -> full attention -> proj+bias -> +res
         RMSNorm -> clamped SwiGLU (gate,up,down, all with bias) -> +res ]
  RMSNorm
  2x2 merge block -> Conv2d(1024,4096,2,stride 2)      (a matmul over 4096)
  merger: proj -> LayerNorm -> GELU -> clamped SwiGLU (4096->10240->4096)

Four details that are easy to get wrong, and all four are why this file
exists rather than a reading of the config:

  - **Patch order is block-major.** The rows are laid out so that each
    consecutive four are one 2x2 merge block, which is what lets the
    downsample be a plain reshape. `get_vision_position_ids` builds the
    rotary indices the same way, and a row order that disagrees with it
    rotates every patch by someone else's position.
  - **RoPE is rotate_half over a doubled table**, not the interleaved
    pairing K3's tower uses. `emb = cat(rot, rot)` and then the usual
    `q*cos + rotate_half(q)*sin`.
  - **q and k are RMSNormed per head** before the rotation, over head_dim
    and not over the whole row.
  - **The SwiGLU is clamped** — gate above, up on both sides — in the
    encoder MLP and again in the merger, at `swiglu_limit`. The merger's
    first activation is a plain GELU, exact rather than tanh.

  uv run --with torch python tools/glm_vision_ref.py \\
      --container glm53.waste --grid 4 6 --dump vis_ref.bin
"""

import argparse
import json
import math
import os
import struct
import sys

import torch
import torch.nn.functional as F

PREFIX = "vision_tower."


class Trunk:
    """Dequantizes trunk tensors on demand.

    F32 verbatim, Q8G as int8 with one fp16 scale per group of 128, Q4G as
    two signed nibbles per byte — low nibble first, biased by +8 — which is
    what a default conversion writes for a tensor this size. K3's oracle
    needed `requant_vision.py` to turn its tower into Q8G first; reading the
    packing here is a few lines and removes the step."""

    def __init__(self, path):
        self.man = json.load(open(os.path.join(path, "manifest.json")))
        self.meta = {e["name"]: e for e in self.man["trunk"]}
        self.f = open(os.path.join(path, "trunk.bin"), "rb")
        self.cache = {}

    def __contains__(self, name):
        return name in self.meta

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
        elif e["fmt"] in (2, 3):
            rows, N = n // shape[-1], shape[-1]
            g = e["group"]
            ng = (N + g - 1) // g
            self.f.seek(e["off"])
            if e["fmt"] == 2:
                q = torch.frombuffer(bytearray(self.f.read(rows * ng * g)),
                                     dtype=torch.int8).view(rows, ng, g).float()
            else:
                raw = torch.frombuffer(
                    bytearray(self.f.read(rows * ng * g // 2)),
                    dtype=torch.uint8)
                lo = (raw & 0x0F).to(torch.int16) - 8
                hi = (raw >> 4).to(torch.int16) - 8
                q = torch.stack([lo, hi], dim=-1).flatten()
                q = q.view(rows, ng, g).float()
            self.f.seek(e["scale_off"])
            sc = torch.frombuffer(bytearray(self.f.read(rows * ng * 2)),
                                  dtype=torch.float16).view(rows, ng, 1).float()
            t = (q * sc).view(rows, ng * g)[:, :N].reshape(*shape).clone()
        else:
            raise RuntimeError(f"{name}: fmt {e['fmt']} not supported here")
        self.cache[name] = t
        return t


def rms_norm(x, w, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def clamped_swiglu(gate, up, limit):
    return F.silu(gate.clamp(max=limit)) * up.clamp(-limit, limit)


def position_ids(h, w, merge):
    """(h*w, 2) of (row, column), laid out block-major over merge x merge
    blocks. Transcribed from transformers' get_vision_position_ids."""
    hp, wp = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    block = (h // merge, merge, w // merge, merge)
    hp = hp.reshape(block).transpose(1, 2).flatten()
    wp = wp.reshape(block).transpose(1, 2).flatten()
    return torch.stack([hp, wp], dim=-1)


def rotate_half(x):
    a, b = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-b, a), dim=-1)


def rope_tables(h, w, merge, head_dim, theta=10000.0):
    """(cos, sin), each (h*w, head_dim)."""
    dim = head_dim // 2
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
    rot = (position_ids(h, w, merge).float().unsqueeze(-1) * inv).flatten(1)
    emb = torch.cat((rot, rot), dim=-1)
    return emb.cos(), emb.sin()


def tower_forward(t, pixels, h, w, cfg, dump_stage=None):
    """pixels: (h*w, 1176) in block-major patch order.

    Returns (h*w/merge^2, out_hidden_size)."""
    D = cfg["hidden_size"]
    heads = cfg["num_heads"]
    hd = D // heads
    eps = cfg.get("rms_norm_eps", 1e-5)
    merge = cfg.get("merge_size") or cfg["spatial_merge_size"]
    limit = cfg.get("swiglu_limit", 10.0)
    out_dim = cfg["out_hidden_size"]
    L = h * w

    pw = t[PREFIX + "patch_embed.proj.weight"].reshape(D, -1)
    x = pixels.reshape(L, -1) @ pw.T + t[PREFIX + "patch_embed.proj.bias"]
    if dump_stage == "embed":
        return x

    cos, sin = rope_tables(h, w, merge, hd)
    scale = 1.0 / math.sqrt(hd)
    for b in range(cfg["depth"]):
        p = f"{PREFIX}blocks.{b}."
        y = rms_norm(x, t[p + "norm1.weight"], eps)
        qkv = y @ t[p + "attn.qkv.weight"].T + t[p + "attn.qkv.bias"]
        q, k, v = qkv.reshape(L, 3, heads, hd).permute(1, 0, 2, 3).unbind(0)
        q = rms_norm(q, t[p + "attn.q_norm.weight"], eps)
        k = rms_norm(k, t[p + "attn.k_norm.weight"], eps)
        c, s = cos.unsqueeze(-2), sin.unsqueeze(-2)
        q = q * c + rotate_half(q) * s
        k = k * c + rotate_half(k) * s
        att = torch.einsum("lhd,mhd->hlm", q, k) * scale
        att = att.softmax(-1)
        o = torch.einsum("hlm,mhd->lhd", att, v).reshape(L, D)
        x = x + o @ t[p + "attn.proj.weight"].T + t[p + "attn.proj.bias"]

        y = rms_norm(x, t[p + "norm2.weight"], eps)
        gate = y @ t[p + "mlp.gate_proj.weight"].T + t[p + "mlp.gate_proj.bias"]
        up = y @ t[p + "mlp.up_proj.weight"].T + t[p + "mlp.up_proj.bias"]
        hcat = clamped_swiglu(gate, up, limit)
        x = x + hcat @ t[p + "mlp.down_proj.weight"].T + t[p + "mlp.down_proj.bias"]
        if dump_stage == f"block{b}":
            return x

    x = rms_norm(x, t[PREFIX + "post_layernorm.weight"], eps)
    if dump_stage == "post":
        return x

    # The rows arrive block-major, so a merge block is four consecutive
    # rows and the Conv2d is a matmul over (in_channels, kh, kw) — that
    # order, which is the weight's own layout and not (kh, kw, channels).
    dw = t[PREFIX + "downsample.weight"]                 # (out, D, m, m)
    blocks = x.view(-1, merge, merge, D).permute(0, 3, 1, 2).reshape(-1, D * merge * merge)
    x = blocks @ dw.reshape(out_dim, -1).T + t[PREFIX + "downsample.bias"]
    if dump_stage == "downsample":
        return x

    m = PREFIX + "merger."
    x = x @ t[m + "proj.weight"].T
    x = F.layer_norm(x, (out_dim,), t[m + "post_projection_norm.weight"],
                     t[m + "post_projection_norm.bias"])
    x = F.gelu(x)
    gate = x @ t[m + "gate_proj.weight"].T
    up = x @ t[m + "up_proj.weight"].T
    return clamped_swiglu(gate, up, limit) @ t[m + "down_proj.weight"].T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--container", required=True)
    ap.add_argument("--grid", nargs=2, type=int, default=[4, 6],
                    metavar=("H", "W"), help="patch grid, both even")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dump", default="")
    ap.add_argument("--pixels", default="",
                    help="read the patch tensor from this file instead of "
                         "generating it, so the engine and this see the "
                         "same input rather than the same generator")
    ap.add_argument("--stage", default="",
                    help="stop after embed | blockN | post | downsample")
    args = ap.parse_args()

    vpath = os.path.join(args.container, "vision.json")
    if not os.path.exists(vpath):
        sys.exit(f"{args.container} has no vision.json")
    cfg = json.load(open(vpath))
    if cfg.get("tower") != "glm5-next":
        sys.exit(f"vision.json says tower={cfg.get('tower')!r}; this file is "
                 f"GLM's tower, tools/vision_ref.py is K3's")
    t = Trunk(args.container)

    h, w = args.grid
    merge = cfg.get("merge_size") or cfg["spatial_merge_size"]
    if h % merge or w % merge:
        sys.exit(f"grid must be a multiple of {merge} in both axes")
    npix = (cfg["in_channels"] * cfg.get("temporal_patch_size", 2)
            * cfg["patch_size"] ** 2)
    if args.pixels:
        raw = open(args.pixels, "rb").read()
        pixels = torch.frombuffer(bytearray(raw), dtype=torch.float32)
        pixels = pixels.view(h * w, npix)
    else:
        g = torch.Generator().manual_seed(args.seed)
        pixels = torch.randn(h * w, npix, generator=g) * 0.5

    with torch.no_grad():
        out = tower_forward(t, pixels, h, w, cfg,
                            dump_stage=args.stage or None)
    print(f"grid {h}x{w} -> {tuple(out.shape)}  "
          f"mean {out.mean():.6f}  std {out.std():.6f}  "
          f"absmax {out.abs().max():.6f}")
    if args.dump:
        b = out.contiguous().float().flatten()
        with open(args.dump, "wb") as f:
            f.write(struct.pack(f"<{b.numel()}f", *b.tolist()))
        print(f"dumped -> {args.dump}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
