#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""
bitalloc_lab.py — Gate 6: is per-expert bit allocation a real lever, or is
it just averaging?

docs/FORMAT.md design goal 5 says important experts get 3 bits and
unimportant ones get 2. Nothing was ever measured to justify it, and the
allocator was never built. This decides it before any of it is.

**It answered no, on 2026-07-29.** The value of the third bit spreads
1.06-1.15x between experts in a layer, 1.01x between layers and
1.09-1.30x between gate/up/down, on K3 and on Kimi-Linear alike, so the
optimal allocator writes the same container as a coin flip. Design goal 5
was dropped rather than implemented; docs/LEARNED.md §20 records the
numbers and the one measurement that would revive it. This file is kept
as the instrument that would run it again.

The arithmetic that makes the gate falsifiable. Total squared error over a
layer, when a set S of experts is dropped from 3 stages to 2:

    E(S) = sum_e err3_e + sum_{e in S} (err2_e - err3_e)
                          \\_______  delta_e  _______/

so for a fixed |S| the best possible choice is exactly "the experts with
the smallest delta". That much is arithmetic, not a result. What is NOT
known is whether delta has any *spread*: if every expert pays the same
price for the third stage, the optimal choice equals a random one and the
whole idea is dead. The gate is the gap between the greedy curve and the
random curve.

Weight error is the right currency for it. For activations x ~ N(0, I),
E||(W - R)x||^2 = ||W - R||_F^2 exactly, so the Gaussian-activation proxy
quant_lab.py reports IS the Frobenius error in expectation — which means
an importance metric the converter can compute for free (it has the
residuals already) is the same metric, and no calibration corpus is
needed to rank experts. Real activations are not isotropic, so this
remains a proxy; it is the same proxy Gate 3 used.

There are three places the spread could live, and the gate looks at all
three, because they cost very different amounts to exploit:

  within a layer   per-expert widths. Needs variable-size records, a
                   per-expert index, and a cache that can hold two record
                   sizes — the expensive one.
  across layers    per-layer widths. The engine already keys record size
                   by layer, so this is nearly free.
  across matrices  gate/up/down of the same expert at different widths.

  uv run --with torch python tools/bitalloc_lab.py \
      --model /Volumes/WasteDisk/k3 --layers 1,5,23,46,69,92 --experts 24

Reports, per allocation policy, the average bits per weight and the
resulting aggregate relative output error, so the greedy and random
curves can be read against each other at matched bits.
"""

import argparse
import ctypes
import json
import os
import sys
import time
from collections import Counter

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mxfp4 import ST                                            # noqa: E402

VEC_DIM = 8
CB_ENTRIES = 256
KINDS = (("gate", "w1"), ("up", "w3"), ("down", "w2"))
TRAIN_VECTORS = 300000        # what convert.py fits a codebook on
CB_SAMPLE = 12                # experts sampled for the fit, as convert.py


# ------------------------------------------------------------------ setup --

def load_vq():
    """The engine's own encoder, so the gate quantizes exactly the way the
    converter does rather than approximating it in torch."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for name in ("libwastevq.dylib", "libwastevq.so"):
        path = os.path.join(here, name)
        if os.path.exists(path):
            lib = ctypes.CDLL(path)
            lib.waste_vq_encode.argtypes = [
                ctypes.POINTER(ctypes.c_float), ctypes.c_int,
                ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.POINTER(ctypes.c_uint8), ctypes.c_int]
            lib.waste_vq_encode.restype = None
            return lib
    raise SystemExit("libwastevq not built — run `make libwastevq.dylib`")


def assign(X, C, chunk=1 << 20):
    cn = (C * C).sum(1)
    out = torch.empty(X.shape[0], dtype=torch.long, device=X.device)
    for s in range(0, X.shape[0], chunk):
        x = X[s:s + chunk]
        out[s:s + chunk] = (cn.unsqueeze(0) - 2.0 * (x @ C.T)).argmin(1)
    return out


def train_codebooks(X, n_stages, iters=10, seed=0):
    """Residual k-means, one codebook per stage — convert.py's own."""
    g = torch.Generator().manual_seed(seed)
    books, resid = [], X[torch.randperm(X.shape[0], generator=g)[:TRAIN_VECTORS]]
    for _ in range(n_stages):
        C = resid[torch.randperm(resid.shape[0], generator=g)[:CB_ENTRIES]].clone()
        for _ in range(iters):
            idx = assign(resid, C)
            for j in range(CB_ENTRIES):
                m = idx == j
                if m.any():
                    C[j] = resid[m].mean(0)
        books.append(C)
        resid = resid - C[assign(resid, C)]
    return torch.stack(books)                       # [stages, 256, 8]


def encode(lib, W, books):
    """-> indices [n_vec, stages] uint8, per-output-channel scale [M]."""
    M, N = W.shape
    scale = W.abs().amax(-1, keepdim=True).clamp(min=1e-8)
    X = (W / scale).reshape(-1, VEC_DIM).contiguous().float()
    B = books.contiguous().float()
    n, st = X.shape[0], books.shape[0]
    out = torch.empty(n * st, dtype=torch.uint8)
    fp = ctypes.POINTER(ctypes.c_float)
    lib.waste_vq_encode(ctypes.cast(X.data_ptr(), fp), n,
                        ctypes.cast(B.data_ptr(), fp), st, CB_ENTRIES, VEC_DIM,
                        ctypes.cast(out.data_ptr(), ctypes.POINTER(ctypes.c_uint8)), 0)
    return out.view(n, st), scale.flatten()


def sq_err(W, idx, scale, books, stages):
    """||W - dequant(first `stages` stages)||^2, in the original units."""
    M, N = W.shape
    recon = torch.zeros(idx.shape[0], VEC_DIM)
    for s in range(stages):
        recon += books[s][idx[:, s].long()]
    R = recon.view(M, N) * scale.unsqueeze(1)
    return (W - R).pow(2).sum().item()


# ------------------------------------------------------------ allocations --

def greedy(delta, n_demote):
    """The optimal set for a fixed count: smallest deltas first."""
    return set(torch.argsort(delta)[:n_demote].tolist())


def total_err(err_by_stage, demoted, lo, hi):
    """Aggregate squared error with `demoted` at `lo` stages, rest at `hi`."""
    return sum(err_by_stage[lo if e in demoted else hi][e]
               for e in range(len(err_by_stage[hi])))


def bits_of(k, n, lo, hi):
    """Average bits/weight when k of n experts sit at `lo` stages."""
    return (k * lo + (n - k) * hi) / n


# ------------------------------------------------------------------- main --

def measure_layer(lib, st, pref, layer, E, stages, t0):
    """[kind][stage_count][expert] -> squared error, plus the reference
    energy. Codebooks are fitted per (layer, kind) exactly as convert.py
    does, since a layer's codebooks are what its experts are quantized
    against and a borrowed set would flatter or damage every expert alike."""
    def ename(e, tag):
        return (f"{pref}model.layers.{layer}.block_sparse_moe.experts."
                f"{e}.{tag}.weight")

    err = {k: {s: [0.0] * E for s in range(stages + 1)} for k, _ in KINDS}
    energy = {k: [0.0] * E for k, _ in KINDS}

    for kind, tag in KINDS:
        sample = list(range(0, E, max(1, E // CB_SAMPLE)))[:CB_SAMPLE]
        per = max(1, TRAIN_VECTORS // len(sample))
        chunks = []
        for e in sample:
            W = st.tensor(ename(e, tag))
            sc = W.abs().amax(-1, keepdim=True).clamp(min=1e-8)
            V = (W / sc).reshape(-1, VEC_DIM)
            g = torch.Generator().manual_seed(1234 + e)
            chunks.append(V[torch.randperm(V.shape[0], generator=g)[:per]])
        books = train_codebooks(torch.cat(chunks), stages)
        del chunks

        for e in range(E):
            W = st.tensor(ename(e, tag))
            idx, scale = encode(lib, W, books)
            energy[kind][e] = W.pow(2).sum().item()
            for s in range(1, stages + 1):
                err[kind][s][e] = sq_err(W, idx, scale, books, s)
            err[kind][0][e] = energy[kind][e]
        print(f"  L{layer} {kind}: {E} experts ({time.time()-t0:.0f}s)",
              flush=True)
    return err, energy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/Volumes/WasteDisk/k3")
    ap.add_argument("--layers", default="5",
                    help="comma list; more than one enables the cross-layer "
                         "comparison")
    ap.add_argument("--experts", type=int, default=24)
    ap.add_argument("--stages", type=int, default=3, help="the high width")
    ap.add_argument("--seeds", type=int, default=8,
                    help="random allocations averaged for the control")
    ap.add_argument("--trace", default="",
                    help="routing trace JSONL; turns the flat delta into the "
                         "exchange rate between disk, I/O and error")
    ap.add_argument("--per-kind", action="store_true",
                    help="also allocate per (expert, matrix) instead of "
                         "per expert, to see if the finer lever pays")
    args = ap.parse_args()

    lib = load_vq()
    st = ST(args.model)
    E, hi = args.experts, args.stages
    layers = [int(x) for x in args.layers.split(",")]

    pref = ""
    for p in ("", "language_model."):
        n = (f"{p}model.layers.{layers[0]}.block_sparse_moe.experts.0"
             f".w1.weight")
        if st.have(n) or st.have(n + "_packed"):
            pref = p
            break
    else:
        raise SystemExit(f"no experts at layer {layers[0]} in {args.model}")

    t0 = time.time()
    per_layer = {}
    for L in layers:
        err, energy = measure_layer(lib, st, pref, L, E, hi, t0)
        # an expert's width applies to all three of its matrices
        tot = {s: [sum(err[k][s][e] for k, _ in KINDS) for e in range(E)]
               for s in range(hi + 1)}
        per_layer[L] = (tot, sum(sum(energy[k]) for k, _ in KINDS), err, energy)

    print(f"\n{args.model}, {E} experts/layer, "
          f"codebooks fitted per (layer, matrix)")

    # ---- 1. within a layer -------------------------------------------------
    for L in layers:
        tot, ref, err, energy = per_layer[L]

        def rel(sq, ref=ref):
            return (sq / ref) ** 0.5

        head = "  ".join(f"VQ{s}R {rel(sum(tot[s])):.2%}"
                         for s in range(1, hi + 1))
        print(f"\nlayer {L} uniform: {head}")
        for lo in range(1, hi):
            delta = torch.tensor([tot[lo][e] - tot[hi][e] for e in range(E)])
            spread = (delta.max() / delta.min()).item()
            print(f"  {hi}->{lo}: delta min {delta.min():.4g} max "
                  f"{delta.max():.4g} = {spread:.2f}x spread")
            print(f"  {'demoted':>8} {'bits':>5} {'greedy':>9} {'random':>9} "
                  f"{'gain':>8}")
            for frac in (0.25, 0.5, 0.75):
                k = int(round(frac * E))
                g_err = rel(total_err(tot, greedy(delta, k), lo, hi))
                r = 0.0
                for s in range(args.seeds):
                    gen = torch.Generator().manual_seed(s)
                    r += rel(total_err(
                        tot, set(torch.randperm(E, generator=gen)[:k].tolist()),
                        lo, hi))
                r /= args.seeds
                print(f"  {k:>8} {bits_of(k, E, lo, hi):>5.2f} "
                      f"{g_err:>8.2%} {r:>8.2%} "
                      f"{(r - g_err) / r if r else 0:>7.2%}")

    # ---- 2. across layers --------------------------------------------------
    if len(layers) > 1:
        print(f"\nacross layers — the same experts, layer by layer:")
        print(f"  {'layer':>6} " + " ".join(f"{'VQ%dR' % s:>8}"
                                            for s in range(1, hi + 1))
              + f" {'3->2 delta':>12}")
        deltas = {}
        for L in layers:
            tot, ref, _, _ = per_layer[L]
            row = " ".join(f"{(sum(tot[s])/ref)**0.5:>7.2%}"
                           for s in range(1, hi + 1))
            deltas[L] = (sum(tot[hi - 1]) - sum(tot[hi])) / ref
            print(f"  {L:>6} {row} {deltas[L]:>11.4g}")
        d = torch.tensor([deltas[L] for L in layers])
        print(f"  spread across layers: {(d.max()/d.min()).item():.2f}x "
              f"(within a layer it was ~1.1x)")

    # ---- 2b. weighted by how often an expert is actually used --------------
    # The one importance signal that is NOT flat. Delta being constant per
    # expert means the routing-weighted damage of demoting a set S is just
    # its share of activations: err^2 = err3^2 + mass(S) * delta. So the
    # whole policy space collapses to a choice of which end of the routing
    # distribution to demote, and this table is the exchange rate.
    if args.trace:
        freq, n_tok, n_layers, top = Counter(), 0, 0, 0
        with open(args.trace) as f:
            for line in f:
                if not line.strip():
                    continue
                n_tok += 1
                for li, layer in enumerate(json.loads(line)["layers"]):
                    freq.update((li, e) for e in layer)
                    n_layers = max(n_layers, li + 1)
                    top = max(top, max(layer))
        acts = sum(freq.values())
        # Slots never routed in the trace still exist in the container and
        # still occupy disk, so the denominator is every slot the model has,
        # not every slot the trace happened to touch.
        n_slots = n_layers * (top + 1)
        e3 = sum((sum(per_layer[L][0][hi]) / per_layer[L][1]) for L in layers) / len(layers)
        dl = sum((sum(per_layer[L][0][hi - 1]) - sum(per_layer[L][0][hi]))
                 / per_layer[L][1] for L in layers) / len(layers)
        print(f"\nrouting trace {args.trace}: {n_tok} tokens, {acts} activations "
              f"over ~{n_slots} slots; measured err{hi}^2 {e3:.4g}, "
              f"delta {dl:.4g}")
        ranked = sorted(freq.values(), reverse=True)
        ranked += [0] * max(0, n_slots - len(ranked))
        print(f"  {'demoted':>8} {'bits':>5} {'disk':>7} "
              f"{'coldest first':>21} {'hottest first':>21}")
        print(f"  {'':>8} {'':>5} {'':>7} {'mass':>7} {'I/O':>6} {'err':>6} "
              f"{'mass':>7} {'I/O':>6} {'err':>6}")
        for frac in (0.25, 0.5, 0.75):
            k = int(round(frac * n_slots))
            cold = sum(ranked[n_slots - k:]) / acts
            hot = sum(ranked[:k]) / acts
            bits = (k * (hi - 1) + (n_slots - k) * hi) / n_slots
            out = f"  {k:>8} {bits:>5.2f} {-frac/hi:>6.1%} "
            for mass in (cold, hot):
                out += (f"{mass:>7.1%} {-mass/hi:>6.1%} "
                        f"{(e3 + mass*dl)**0.5:>6.2%} ")
            print(out)
        print(f"  (uniform VQ{hi}R is {e3**0.5:.2%}; a short trace "
              f"overstates how cold the cold tail is)")

    # ---- 3. across matrices ------------------------------------------------
    if args.per_kind:
        # Each of gate/up/down chooses its own width: 3E decisions instead
        # of E, at the cost of a wider record header. Worth knowing before
        # the format is frozen.
        for L in layers:
            tot, ref, err, energy = per_layer[L]
            lo = hi - 1
            print(f"\nlayer {L} per-matrix, {hi}->{lo}: "
                  + "  ".join(
                      f"{k} {(err[k][lo][0]-err[k][hi][0])/energy[k][0]:.4g}"
                      for k, _ in KINDS))
            flat_lo = [err[k][lo][e] for k, _ in KINDS for e in range(E)]
            flat_hi = [err[k][hi][e] for k, _ in KINDS for e in range(E)]
            d = torch.tensor([a - b for a, b in zip(flat_lo, flat_hi)])
            print(f"  delta spread over the 3E matrices: "
                  f"{(d.max()/d.min()).item():.2f}x")
            print(f"  {'demoted':>8} {'bits':>5} {'greedy':>9}")
            for frac in (0.25, 0.5, 0.75):
                k = int(round(frac * 3 * E))
                sel = set(torch.argsort(d)[:k].tolist())
                sq = sum(flat_lo[i] if i in sel else flat_hi[i]
                         for i in range(3 * E))
                print(f"  {k:>8} {(k*lo + (3*E-k)*hi)/(3*E):>5.2f} "
                      f"{(sq/ref)**0.5:>8.2%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
