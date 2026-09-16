#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""
ds41_vision_ref.py — DeepSeek-V4.1's vision tower, off a WASTE container.

The oracle `waste_vision_encode_ds41` is diffed against. Transcribed from
the release's `inference/vision.py`, with the weights read from the
container so the diff measures ARITHMETIC and not quantization — the same
contract as tools/vision_ref.py and tools/glm_vision_ref.py.

Three things it exists to catch, none of which shows up as a wrong shape:

  - **the rotation is split-halves.** Each head's dims are halved and the
    first half rotated against the second, where every other rotation in
    this engine pairs adjacent elements. And the 32-wide table is itself
    two halves — sixteen frequencies of the row, sixteen of the column —
    so getting the interleave wrong scrambles the positions and nothing
    else.
  - **the projector is a 3x3 pixel-unshuffle** with zero padding on the
    right and bottom, and the unfold's element order is
    (channel, dy, dx) — the weight's own axis order, not the (dy, dx,
    channel) a reader of the reshape would assume.
  - **the span is not the image**: one newline per token row and two
    delimiters, all three learned embeddings that live in the TEXT trunk.

  ./test_vision_ds41 tower CONTAINER H W pixels.bin out.bin
  uv run --with torch python tools/ds41_vision_ref.py \\
      --container CONTAINER --pixels pixels.bin --grid HxW --engine out.bin
"""

import argparse
import math
import os
import struct
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kimi_ref import Container                                  # noqa: E402


def num_image_tokens(nh, nw):
    return nh * (nw + 1) + 2


def llm_grid(bh, bw, patch, r):
    return math.ceil((bh // patch) / r), math.ceil((bw // patch) / r)


def solve_resize_ratio(height, width, patch, r, max_tok):
    ratio = height / width
    max_w = math.sqrt((max_tok - 2) / ratio + 0.25) - 0.5
    max_h = max_w * ratio
    cell = patch * r
    if max_w < 1.0:
        return (max_tok - 2) // 2 * cell, cell
    if max_h < 1.0:
        return cell, (max_tok - 3) * cell
    beta = min(math.floor(max_w) * cell / width,
               math.floor(max_h) * cell / height)
    return (math.floor(height * beta / patch) * patch,
            math.floor(width * beta / patch) * patch)


def plan_image_grid(width, height, vj):
    """The release's plan_image_grid, on vision.json's own key names.
    Returns (best_h, best_w, n_llm_h, n_llm_w)."""
    p = vj["patch_size"]
    r = vj["downsample_ratio"]
    max_tok = vj.get("max_image_tokens", 1024)
    min_px = vj.get("min_pixels") or 0
    if 0 < width * height < min_px:
        s = (min_px / (width * height)) ** 0.5
        width = int(width * s)
        height = int(height * s)
    bw = math.ceil(width / p) * p
    bh = math.ceil(height / p) * p
    nh, nw = llm_grid(bh, bw, p, r)
    if num_image_tokens(nh, nw) > max_tok:
        bh, bw = solve_resize_ratio(height, width, p, r, max_tok)
        nh, nw = llm_grid(bh, bw, p, r)
    return bh, bw, nh, nw


def rms_norm(x, w, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def cos_sin(n_h, n_w, dim, theta):
    """2D rotary tables, row frequencies then column frequencies."""
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    h = torch.arange(n_h).unsqueeze(1).expand(n_h, n_w)
    w = torch.arange(n_w).unsqueeze(0).expand(n_h, n_w)
    freqs = torch.stack([h, w], -1).reshape(-1, 2, 1).float() * inv
    freqs = freqs.flatten(1)                       # [n, dim]
    return freqs.cos().unsqueeze(1), freqs.sin().unsqueeze(1)


def rope(x, cs, sn):
    """Split halves, NOT adjacent pairs."""
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat([x1 * cs - x2 * sn, x2 * cs + x1 * sn], -1)


class Tower:
    def __init__(self, c, vj):
        self.c = c
        self.t = c.t
        self.D = vj["hidden_size"]
        self.heads = vj["num_attention_heads"]
        self.inter = vj["intermediate_size"]
        self.layers = vj["num_hidden_layers"]
        self.patch = vj["patch_size"]
        self.r = vj["downsample_ratio"]
        self.eps = vj.get("rms_norm_eps", 1e-6)
        self.theta = vj.get("rope_theta", 10000.0)
        self.OD = vj["out_hidden_size"]
        self.pfx = c.prefix

    def W(self, name):
        return self.t[name]

    def run(self, px, n_h, n_w, stage=None):
        D, heads = self.D, self.heads
        hd = D // heads
        L = n_h * n_w
        x = px.reshape(L, -1) @ self.W("vision_tower.patch_embed.proj.weight").T
        x = x + self.W("vision_tower.patch_embed.proj.bias")
        if stage == "embed":
            return x
        cs, sn = cos_sin(n_h, n_w, hd // 2, self.theta)

        for b in range(self.layers):
            p = f"vision_tower.blocks.{b}."
            y = rms_norm(x, self.W(p + "norm1.weight"), self.eps)
            qkv = y @ self.W(p + "attn.wqkv.weight").T + self.W(p + "attn.wqkv.bias")
            q, k, v = (t.view(L, heads, hd) for t in qkv.chunk(3, -1))
            q, k = rope(q, cs, sn), rope(k, cs, sn)
            o = F.scaled_dot_product_attention(
                q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1))
            o = o.transpose(0, 1).reshape(L, D)
            x = x + o @ self.W(p + "attn.wo.weight").T + self.W(p + "attn.wo.bias")

            y = rms_norm(x, self.W(p + "norm2.weight"), self.eps)
            gate, up = (y @ self.W(p + "mlp.w1.weight").T).chunk(2, -1)
            x = x + (F.silu(gate) * up) @ self.W(p + "mlp.w2.weight").T
            if stage == f"block{b}":
                return x

        x = rms_norm(x, self.W("vision_tower.norm.weight"), self.eps)
        if stage == "post":
            return x

        r = self.r
        g = x.view(n_h, n_w, -1).permute(2, 0, 1)
        g = F.pad(g, (0, -n_w % r, 0, -n_h % r))
        g = F.unfold(g.unsqueeze(0), r, stride=r).squeeze(0).transpose(0, 1)
        g = g @ self.W("mm_projector.w1.weight").T + self.W("mm_projector.w1.bias")
        g = F.gelu(g)
        g = g @ self.W("mm_projector.w2.weight").T + self.W("mm_projector.w2.bias")

        nh, nw = math.ceil(n_h / r), math.ceil(n_w / r)
        start = self.t[self.pfx + "model.image_start"]
        end = self.t[self.pfx + "model.image_end"]
        newline = self.t[self.pfx + "model.image_newline"]
        rows = [start]
        for by in range(nh):
            for bx in range(nw):
                rows.append(g[by * nw + bx])
            rows.append(newline)
        rows.append(end)
        return torch.stack(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--container", required=True)
    ap.add_argument("--pixels", help="f32 [gh*gw][3*p*p]")
    ap.add_argument("--grid", help="HxW in patches")
    ap.add_argument("--engine", help="the engine's output, to diff against")
    ap.add_argument("--plan", default=None, metavar="WxH",
                    help="print the geometry for a source image of this "
                         "size and stop — the pure function the engine's "
                         "waste_image_plan_ds41 is, asked the same way")
    ap.add_argument("--stage", default=None,
                    help="embed | blockN | post (default: the whole span)")
    a = ap.parse_args()

    torch.set_grad_enabled(False)
    import json
    vj = json.load(open(os.path.join(a.container, "vision.json")))
    if vj.get("tower") != "ds41":
        raise SystemExit(f"{a.container} does not carry DeepSeek-V4.1's tower")
    if a.plan:
        w, h = (int(v) for v in a.plan.lower().split("x"))
        bh, bw, nh, nw = plan_image_grid(w, h, vj)
        print(f"box {bw}x{bh}  patches {bw // vj['patch_size']}x"
              f"{bh // vj['patch_size']}  tokens {nw}x{nh}  "
              f"span {nh * (nw + 1) + 2}")
        return 0
    if not (a.pixels and a.grid):
        raise SystemExit("--pixels and --grid are needed unless --plan is")
    c = Container(a.container)
    n_h, n_w = (int(v) for v in a.grid.lower().split("x"))
    tower = Tower(c, vj)
    npix = 3 * tower.patch * tower.patch
    raw = open(a.pixels, "rb").read()
    n = n_h * n_w * npix
    px = torch.tensor(struct.unpack(f"<{n}f", raw[:n * 4])).view(n_h * n_w, npix)

    out = tower.run(px, n_h, n_w, a.stage)
    print(f"tower -> {tuple(out.shape)}  mean {out.mean():+.6f}  "
          f"std {out.std():.6f}  absmax {out.abs().max():.6f}")
    if a.engine:
        eng = open(a.engine, "rb").read()
        m = out.numel()
        got = torch.tensor(struct.unpack(f"<{m}f", eng[:m * 4])).view_as(out)
        num = (out - got).norm()
        den = out.norm()
        print(f"vs the engine: rel L2 {100 * num / den:.6f}%  "
              f"maxabs {(out - got).abs().max():.3g}")
        return 0 if num / den < 1e-4 else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
