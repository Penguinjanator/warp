#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""
kda_ref.py — Gate 4: validate the C KDA kernel against the official
reference.

fla's `naive_recurrent_kda` is pure PyTorch (no Triton), so the reference
shipped with Kimi-Linear runs on any machine — including Apple Silicon,
where the production Triton kernels cannot.

  uv run --with torch --with fla-core python tools/kda_ref.py

Writes a random fixture, runs ./test_kda on it, and compares outputs and
final state. Non-zero exit if tolerances are exceeded.
"""

import os
import struct
import subprocess
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

T, H, K, V = (int(os.environ.get(k, d)) for k, d in
                (("KDA_T", 24), ("KDA_H", 4), ("KDA_K", 32), ("KDA_V", 32)))
TOL_O, TOL_S = 2e-4, 2e-4


def load_reference():
    """Import fla's naive_recurrent_kda without importing the fla package.

    fla/ops/__init__.py pulls in Triton-backed kernels, which don't exist on
    macOS — but naive.py is plain PyTorch, so load it straight from its file.
    """
    import importlib.util
    import glob

    cands = []
    for root in sys.path:
        if root:
            cands += glob.glob(os.path.join(root, "fla", "ops", "kda", "naive.py"))
    if not cands:
        raise ImportError("fla-core not found; run with: uv run --with fla-core")
    spec = importlib.util.spec_from_file_location("kda_naive", cands[0])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    print(f"reference: {cands[0]}")
    return mod.naive_recurrent_kda


def main():
    naive_recurrent_kda = load_reference()

    torch.manual_seed(0)
    # q, k get L2-normalized inside the C kernel (use_qk_l2norm_in_kernel),
    # so normalize here before handing them to the naive reference.
    q = torch.randn(1, T, H, K)
    k = torch.randn(1, T, H, K)
    v = torch.randn(1, T, H, V)
    # g is log-space decay: keep it negative so exp(g) is a decay factor
    g = -torch.rand(1, T, H, K) * 0.5
    beta = torch.rand(1, T, H)
    S0 = torch.randn(1, H, K, V) * 0.1

    qn = torch.nn.functional.normalize(q, dim=-1, p=2)
    kn = torch.nn.functional.normalize(k, dim=-1, p=2)

    o_ref, S_ref = naive_recurrent_kda(
        q=qn, k=kn, v=v, g=g, beta=beta,
        initial_state=S0, output_final_state=True,
    )

    tmp = tempfile.mkdtemp(prefix="kda_")
    fx, out = os.path.join(tmp, "fixture.bin"), os.path.join(tmp, "out.bin")
    with open(fx, "wb") as f:
        f.write(struct.pack("<4i", T, H, K, V))
        def raw(t):
            b = t[0].contiguous().float().flatten()
            return struct.pack(f"<{b.numel()}f", *b.tolist())
        for t in (q, k, v, g):                 # unnormalized q,k on purpose
            f.write(raw(t))
        f.write(raw(beta))
        f.write(raw(S0))

    exe = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "test_kda")
    if not os.path.exists(exe):
        print("build first:  cc -O2 -o test_kda tests/test_kda.c src/kda.c -lm")
        return 2
    r = subprocess.run([exe, fx, out], capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr.strip())
    if r.returncode != 0:
        return r.returncode

    buf = open(out, "rb").read()
    n_o = T * H * V
    o_c = torch.frombuffer(bytearray(buf[: n_o * 4]), dtype=torch.float32).view(T, H, V)
    S_c = torch.frombuffer(bytearray(buf[n_o * 4:]), dtype=torch.float32).view(H, K, V)

    do = (o_c - o_ref[0]).abs()
    ds = (S_c - S_ref[0]).abs()
    print(f"output   max|diff| = {do.max():.3e}   mean = {do.mean():.3e}")
    print(f"state    max|diff| = {ds.max():.3e}   mean = {ds.mean():.3e}")
    ok = do.max() < TOL_O and ds.max() < TOL_S
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
