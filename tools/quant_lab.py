#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""
quant_lab.py — Gate 3: does sub-4-bit expert quantization actually hold up
on real Kimi MoE weights?

Loads a batch of real experts straight from safetensors shards (no HF
runtime needed) and compares quantization schemes at matched bit budgets:

  rtn4-row     per-row INT4, amax scale        (known-good int4 baseline)
  rtn4-g128    INT4, one scale per 128 inputs  (grouped int4 variant)
  rtn2-g64     INT2, one scale per 64 inputs   (the naive 2-bit strawman)
  vq2 / vq3    residual VQ, 8-dim vectors, N codebooks of 256 entries
  kbvq2/kbvq3  KBVQ-MoE: shared low-rank basis across experts (FP16,
               rank = in_features/128, ~0.125 bits/weight amortized)
               plus per-expert VQ on the residual

Metrics per scheme: effective bits/weight, weight-space relative Frobenius
error, and output-space relative error on Gaussian activations (a proxy —
true calibration needs real activations, which needs a CUDA box).

  uv run --with torch python tools/quant_lab.py --model /Volumes/WasteDisk/kimi-linear
"""

import argparse
import json
import os
import struct
import sys

import torch

GB = 1 << 30


# ---------------------------------------------------------------- loading --

def st_read(path, names):
    """Read tensors from one safetensors file without the safetensors lib."""
    out = {}
    with open(path, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        hdr = json.loads(f.read(hlen))
        base = 8 + hlen
        for name in names:
            if name not in hdr:
                continue
            meta = hdr[name]
            beg, end = meta["data_offsets"]
            f.seek(base + beg)
            raw = f.read(end - beg)
            dt = {"BF16": torch.bfloat16, "F16": torch.float16,
                  "F32": torch.float32}[meta["dtype"]]
            t = torch.frombuffer(bytearray(raw), dtype=dt).view(*meta["shape"])
            out[name] = t.float()
    return out


def load_experts(model_dir, layer, mat, n_experts):
    """Kimi-Linear stores experts as plain bf16; K3 ships them mxfp4-packed
    under a `language_model.` prefix. Handle both."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from mxfp4 import ST
    st = ST(model_dir)
    out = []
    for pref in ("", "language_model."):
        want = [f"{pref}model.layers.{layer}.block_sparse_moe.experts.{e}.{mat}.weight"
                for e in range(n_experts)]
        if not any(st.have(w) or st.have(w + "_packed") for w in want):
            continue
        for w in want:
            try:
                out.append(st.tensor(w))
            except KeyError:
                break
        break
    if not out:
        raise SystemExit(f"no experts available yet (layer {layer}, {mat})")
    return torch.stack(out)                              # [E, out, in]


# ----------------------------------------------------------- quantizers ----

def rtn_group(W, bits, group):
    """Round-to-nearest with a symmetric scale per group of `group` inputs."""
    E, M, N = W.shape
    g = group if group > 0 else N
    pad = (-N) % g
    X = torch.nn.functional.pad(W, (0, pad)).view(E, M, -1, g)
    qmax = 2 ** (bits - 1) - 1
    scale = X.abs().amax(-1, keepdim=True).clamp(min=1e-8) / qmax
    Q = torch.clamp(torch.round(X / scale), -qmax - 1, qmax)
    R = (Q * scale).view(E, M, -1)[:, :, :N]
    bpw = bits + 16.0 / g                      # payload + fp16 scale per group
    return R, bpw


def kmeans(X, k, iters=12, seed=0):
    """Small Lloyd's algorithm on [n, d] vectors; k-means++-ish init."""
    g = torch.Generator().manual_seed(seed)
    n = X.shape[0]
    C = X[torch.randperm(n, generator=g)[:k]].clone()
    for _ in range(iters):
        # assign in chunks to bound memory
        idx = torch.empty(n, dtype=torch.long)
        for s in range(0, n, 65536):
            d = torch.cdist(X[s:s + 65536], C)
            idx[s:s + 65536] = d.argmin(1)
        for j in range(k):
            m = idx == j
            if m.any():
                C[j] = X[m].mean(0)
    return C, idx


VQ_SAMPLE = 200000


def vq_residual(W, n_codebooks, vec_dim=8, cb_bits=8, sample=None, seed=0):
    sample = sample or VQ_SAMPLE
    """Multi-stage (residual) VQ. bits/weight = n_codebooks*cb_bits/vec_dim."""
    E, M, N = W.shape
    # per-output-channel scale first (cheap, big win, matches the format spec)
    scale = W.abs().amax(-1, keepdim=True).clamp(min=1e-8)
    Wn = W / scale

    pad = (-N) % vec_dim
    X = torch.nn.functional.pad(Wn, (0, pad)).view(-1, vec_dim)
    resid = X.clone()
    recon = torch.zeros_like(X)
    k = 2 ** cb_bits
    torch.manual_seed(seed)
    for s in range(n_codebooks):
        sub = resid[torch.randperm(resid.shape[0])[:sample]]
        C, _ = kmeans(sub, k, seed=seed + s)
        idx = torch.empty(resid.shape[0], dtype=torch.long)
        for a in range(0, resid.shape[0], 65536):
            idx[a:a + 65536] = torch.cdist(resid[a:a + 65536], C).argmin(1)
        q = C[idx]
        recon += q
        resid -= q
    R = recon.view(E, M, -1)[:, :, :N] * scale
    bpw = n_codebooks * cb_bits / vec_dim + 16.0 / N   # + per-channel scale
    return R, bpw


def kbvq(W, n_codebooks, rank_div=128, vec_dim=8, cb_bits=8, seed=0):
    """KBVQ-MoE: shared basis across experts (FP16 coefficients per expert)
    + residual VQ. Basis V comes from the right singular vectors of all
    experts' rows stacked — the directions every expert shares."""
    E, M, N = W.shape
    r = max(1, N // rank_div)
    rows = W.reshape(-1, N)
    # top-r right singular vectors of the stacked expert rows
    _, _, Vh = torch.svd_lowrank(rows, q=min(r + 8, N), niter=4)
    V = Vh[:, :r]                                   # [N, r], shared, FP16
    A = rows @ V                                    # [E*M, r], per expert, FP16
    shared = (A @ V.T).view(E, M, N)
    resid = W - shared
    Rq, bpw_resid = vq_residual(resid, n_codebooks, vec_dim, cb_bits, seed=seed)
    R = shared + Rq
    bpw = bpw_resid + 16.0 * r / N                  # A costs 16r/N bits/weight
    return R, bpw


# -------------------------------------------------------------- metrics ----

def errors(W, R, n_act=256, seed=0):
    fro = (W - R).norm() / W.norm()
    torch.manual_seed(seed)
    x = torch.randn(n_act, W.shape[-1])
    num = den = 0.0
    for e in range(W.shape[0]):
        y = x @ W[e].T
        yq = x @ R[e].T
        num += (y - yq).pow(2).sum().item()
        den += y.pow(2).sum().item()
    return fro.item(), (num / den) ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/Volumes/WasteDisk/kimi-linear")
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--mat", default="w1", choices=("w1", "w2", "w3"))
    ap.add_argument("--experts", type=int, default=32)
    ap.add_argument("--rank-div", type=int, default=128,
                    help="KBVQ shared-basis rank = in_features / this")
    ap.add_argument("--kmeans-sample", type=int, default=200000)
    ap.add_argument("--only", default="", help="comma-separated scheme names")
    args = ap.parse_args()

    global VQ_SAMPLE
    VQ_SAMPLE = args.kmeans_sample
    W = load_experts(args.model, args.layer, args.mat, args.experts)
    E, M, N = W.shape
    print(f"layer {args.layer} {args.mat}: {E} experts, {M}x{N} each "
          f"({E*M*N/1e6:.1f} M params)\n")

    schemes = [
        ("rtn4-row",  lambda w: rtn_group(w, 4, 0)),
        ("rtn4-g128", lambda w: rtn_group(w, 4, 128)),
        ("rtn3-g64",  lambda w: rtn_group(w, 3, 64)),
        ("rtn2-g64",  lambda w: rtn_group(w, 2, 64)),
        ("vq2",       lambda w: vq_residual(w, 2)),
        ("vq3",       lambda w: vq_residual(w, 3)),
        ("kbvq2",     lambda w: kbvq(w, 2, rank_div=args.rank_div)),
        ("kbvq3",     lambda w: kbvq(w, 3, rank_div=args.rank_div)),
    ]
    if args.only:
        keep = set(args.only.split(","))
        schemes = [s for s in schemes if s[0] in keep]

    print(f"{'scheme':<11} {'bits/w':>7} {'GB@2.3T':>8} {'weight err':>11} {'output err':>11}")
    for name, fn in schemes:
        R, bpw = fn(W)
        fe, oe = errors(W, R)
        disk = 2.33e12 * bpw / 8 / GB
        print(f"{name:<11} {bpw:>7.2f} {disk:>8.0f} {fe:>10.2%} {oe:>10.2%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
