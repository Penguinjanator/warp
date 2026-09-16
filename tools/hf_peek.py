#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""
hf_peek.py — read individual tensors out of a HuggingFace safetensors repo
over HTTP range requests, without downloading a shard.

Written for gate 8. DeepSeek-V4.1-Flash is 510 GB in 48 shards of ~7.4 GB,
one per layer, and the question the gate had to answer — does 3-bit VQ
survive weights that are already fp4? — needs eight experts, not a shard.
A safetensors file states every tensor's byte range in its own header, so
eight experts is 190 MB of range requests against a file nobody fetched.

It also dequantizes the two formats a modern release ships weights in, so
what comes back is f32 and not a puzzle:

  F8_E4M3 / I8-packed-E2M1  payload, with an F8_E8M0 or F32 scale stream
  laid out [rows, cols / group] — one scale per `group` inputs per row.

  # what gate 8 ran
  python3 tools/hf_peek.py --repo deepseek-ai/DeepSeek-V4.1-Flash \
      --shard model-00003-of-00048.safetensors \
      --tensor 'layers.0.ffn.experts.{e}.w1' --range 0:8 --out /tmp/w1.npy
  uv run --with torch python tools/quant_lab.py --npy /tmp/w1.npy

  # what is in a shard, without fetching it
  python3 tools/hf_peek.py --repo <id> --shard <file> --list

`--tensor` names a weight *without* its `.weight` / `.scale` suffix; both
are fetched and combined when a scale exists, and `{e}` is expanded over
`--range`. Reads the network and nothing else: no HF client, no token, no
cache. A gated repo will 401 and say so.
"""

import argparse
import json
import struct
import subprocess
import sys

# e2m1: three bits of magnitude, one of sign, no NaN, no inf.
E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def _curl(url, first=None, last=None):
    cmd = ["curl", "-sS", "-L", "--fail", "--retry", "3", "--retry-delay", "1"]
    if first is not None:
        cmd += ["-r", f"{first}-{last}"]
    p = subprocess.run(cmd + [url], capture_output=True)
    if p.returncode:
        raise SystemExit(f"fetch failed ({p.returncode}): {p.stderr.decode()[:400]}")
    return p.stdout


class Remote:
    """One safetensors file, read through its header."""

    def __init__(self, repo, shard, revision="main"):
        self.url = f"https://huggingface.co/{repo}/resolve/{revision}/{shard}"
        n = _curl(self.url, 0, 7)
        if len(n) != 8:
            raise SystemExit("server ignored the range request — not a range-capable host")
        (hlen,) = struct.unpack("<Q", n)
        self.hdr = json.loads(_curl(self.url, 8, 8 + hlen - 1))
        self.hdr.pop("__metadata__", None)
        self.base = 8 + hlen

    def raw(self, name):
        if name not in self.hdr:
            raise KeyError(name)
        m = self.hdr[name]
        a, b = m["data_offsets"]
        buf = _curl(self.url, self.base + a, self.base + b - 1)
        if len(buf) != b - a:
            raise SystemExit(f"short read on {name}: {len(buf)} of {b - a}")
        return m["dtype"], m["shape"], buf


def _payload(np, dtype, shape, buf):
    """The weight stream as f32, still unscaled. Returns (values, cols)."""
    if dtype == "I8":
        # E2M1 packed two per byte, low nibble first — the order
        # DeepSeek's own convert.py unpacks with.
        q = np.frombuffer(buf, dtype=np.uint8).reshape(shape)
        tbl = np.array(E2M1, dtype=np.float32)
        out = np.empty((shape[0], shape[1] * 2), dtype=np.float32)
        out[:, 0::2] = tbl[q & 0x0F]
        out[:, 1::2] = tbl[(q >> 4) & 0x0F]
        return out
    if dtype == "F8_E4M3":
        b = np.frombuffer(buf, dtype=np.uint8).astype(np.int32)
        sign = np.where(b & 0x80, -1.0, 1.0).astype(np.float32)
        exp = (b >> 3) & 0x0F
        man = (b & 0x07).astype(np.float32)
        # subnormals share exponent 1's scale with no implicit leading 1
        val = np.where(exp == 0, man / 8.0 * 2.0 ** -6,
                       (1.0 + man / 8.0) * np.ldexp(np.ones_like(man), exp - 7))
        # 0x7F / 0xFF are e4m3fn's only NaN codes. Decoding them as 480 would
        # be a plausible-looking number where the checkpoint said "not a
        # value", so say NaN and let whatever consumes this notice.
        val = np.where((exp == 0x0F) & (man == 7.0), np.float32("nan"), val)
        return (sign * val).astype(np.float32).reshape(shape)
    if dtype in ("BF16", "F16", "F32"):
        w = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32}[dtype]
        a = np.frombuffer(buf, dtype=w)
        if dtype == "BF16":
            a = (a.astype(np.uint32) << 16).view(np.float32)
        return a.astype(np.float32).reshape(shape)
    raise SystemExit(f"unhandled dtype {dtype}")


def _scales(np, dtype, shape, buf):
    if dtype == "F8_E8M0":
        e = np.frombuffer(buf, dtype=np.uint8).astype(np.int32).reshape(shape)
        return np.ldexp(np.ones(e.shape, dtype=np.float32), e - 127)
    return _payload(np, dtype, shape, buf)


def tensor(np, rem, stem):
    """`stem` without its suffix: fetch `stem.weight`, apply `stem.scale`."""
    name = stem if stem in rem.hdr else stem + ".weight"
    w = _payload(np, *rem.raw(name))
    try:
        s = _scales(np, *rem.raw(stem + ".scale"))
    except KeyError:
        return w
    if s.ndim != 2 or w.shape[0] != s.shape[0] or w.shape[1] % s.shape[1]:
        raise SystemExit(f"{stem}: scale {s.shape} does not tile weight {w.shape}")
    g = w.shape[1] // s.shape[1]
    return (w.reshape(w.shape[0], s.shape[1], g) * s[:, :, None]).reshape(w.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="e.g. deepseek-ai/DeepSeek-V4.1-Flash")
    ap.add_argument("--shard", required=True)
    ap.add_argument("--revision", default="main")
    ap.add_argument("--list", action="store_true", help="print the header and stop")
    ap.add_argument("--tensor", help="stem, with {e} expanded over --range")
    ap.add_argument("--range", default="0:1", help="BEGIN:END for {e}")
    ap.add_argument("--out", help="write the stack as .npy")
    a = ap.parse_args()

    try:
        import numpy as np
    except ImportError:
        raise SystemExit("needs numpy: uv run --with numpy python tools/hf_peek.py ...")

    rem = Remote(a.repo, a.shard, a.revision)
    if a.list or not a.tensor:
        for k, v in sorted(rem.hdr.items()):
            print(f"{k:<60} {v['dtype']:<8} {v['shape']}")
        print(f"\n{len(rem.hdr)} tensors", file=sys.stderr)
        return 0

    beg, end = (int(x) for x in a.range.split(":"))
    stack = []
    for e in range(beg, end):
        t = tensor(np, rem, a.tensor.replace("{e}", str(e)))
        print(f"{a.tensor.replace('{e}', str(e))}  {t.shape}  "
              f"absmax {abs(t).max():.6g}  zeros {(t == 0).mean():.2%}", file=sys.stderr)
        stack.append(t)
    out = np.stack(stack) if len(stack) > 1 else stack[0][None]
    if a.out:
        np.save(a.out, out)
        print(f"wrote {a.out} {out.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
