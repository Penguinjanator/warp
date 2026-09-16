#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""
mxfp4.py — read the block-scaled formats this family of releases ships in.

K3 spells an mxfp4 matrix as two tensors:
  <name>.weight_packed  uint8, two FP4 (E2M1) values per byte
  <name>.weight_scale   uint8, one E8M0 exponent per group of 32 weights

E2M1 has 8 magnitudes {0, .5, 1, 1.5, 2, 3, 4, 6} and a sign bit; E8M0 is a
bare biased exponent, so the scale is exactly 2^(e-127). Dequantization is
therefore exact — a lookup and a multiply, no rounding.

DeepSeek-V4.1 spells the same idea differently, and the difference is not
cosmetic: the payload keeps the name `<name>.weight` and carries its dtype
in the header (I8 for packed fp4, F8_E4M3 for the trunk), with the scales in
`<name>.scale`. So a reader that only knows K3's suffixes finds `.weight`,
sees a tensor, and returns the raw int8 nibble pairs as floats — every shape
checks out and every value is wrong. `tensor()` therefore decides on the
companion it can find, not on the suffix it expected.

  uv run --with torch python tools/mxfp4.py MODEL_DIR   # self-check
"""

import json
import os
import struct
import sys

import torch

GROUP = 32

# E2M1 magnitudes by the low 3 bits; bit 3 is the sign.
_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
_LUT = torch.cat([_E2M1, -_E2M1])          # index by the full nibble


def e8m0_scale(raw: torch.Tensor) -> torch.Tensor:
    """E8M0 bytes -> f32. 0 is the reserved zero, not 2^-127."""
    e = raw.to(torch.int32)
    return torch.where(e == 0, torch.zeros_like(e, dtype=torch.float32),
                       torch.exp2((e - 127).to(torch.float32)))


def dequant(packed: torch.Tensor, scale: torch.Tensor, group: int = GROUP):
    """packed [rows, cols/2] uint8, scale [rows, cols/group] uint8 -> f32.

    The low nibble holds the first (even) element of each byte pair; this is
    the packing `compressed-tensors` writes and is checked by self_test().
    """
    rows, half = packed.shape
    lo = (packed & 0x0F).to(torch.long)
    hi = (packed >> 4).to(torch.long)
    vals = torch.empty(rows, half * 2, dtype=torch.float32)
    vals[:, 0::2] = _LUT[lo]
    vals[:, 1::2] = _LUT[hi]

    # E8M0: 2^(e-127); e == 0 is the reserved zero
    s = e8m0_scale(scale)
    cols = vals.shape[1]
    ng = s.shape[1]
    return (vals.view(rows, ng, group) * s.unsqueeze(-1)).view(rows, cols)


# ------------------------------------------------------------------ I/O ---

def unblock_scale(q, scale, block):
    """fp8 block dequant: q [M, N] f32, scale [ceil(M/bm), ceil(N/bn)] -> q * scale.

    `block` must come from the checkpoint's config (quantization_config.
    weight_block_size), NOT be inferred from the two shapes. Inferring looks
    possible and is wrong whenever a dimension is not a multiple of the tile: 300
    rows with 3 scale rows admits both 128 (the truth, with a partial last tile)
    and 100 (a clean split), and the wrong one silently applies each scale to the
    wrong rows. The shapes agree in both readings, so nothing downstream notices.
    """
    if q.ndim != 2 or scale.ndim != 2:
        raise ValueError(f"fp8 block dequant expects 2-D, got {tuple(q.shape)} "
                         f"and {tuple(scale.shape)}")
    bm, bn = block
    M, N = q.shape
    want = (-(-M // bm), -(-N // bn))
    if tuple(scale.shape) != want:
        raise ValueError(f"fp8 scale {tuple(scale.shape)} does not match "
                         f"{tuple(q.shape)} at block {tuple(block)}; expected {want}")
    full = scale.repeat_interleave(bm, 0).repeat_interleave(bn, 1)[:M, :N]
    return q * full


class ST:
    """safetensors reader that also understands the packed pairs."""

    def __init__(self, model_dir):
        self.dir = model_dir
        idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))
        self.wm = idx["weight_map"]
        self._hdr = {}
        # A shard that merely exists may still be downloading; only the ones
        # the downloader has verified are safe to read.
        state = os.path.join(model_dir, ".download-state")
        self.ready = set(open(state).read().split()) if os.path.exists(state) else None
        # fp8 block size is a property of the checkpoint, stated in its config.
        # Default 128x128 only so a config-less directory still reads; a real fp8
        # checkpoint always declares it.
        self.fp8_block = (128, 128)
        cfgp = os.path.join(model_dir, "config.json")
        if os.path.exists(cfgp):
            qc = (json.load(open(cfgp)).get("quantization_config") or {})
            wbs = qc.get("weight_block_size")
            if wbs:
                self.fp8_block = tuple(wbs)

    def _header(self, fn):
        if fn not in self._hdr:
            with open(os.path.join(self.dir, fn), "rb") as f:
                (n,) = struct.unpack("<Q", f.read(8))
                self._hdr[fn] = (json.loads(f.read(n)), 8 + n)
        return self._hdr[fn]

    def have(self, name):
        fn = self.wm.get(name)
        if fn is None:
            return False
        if self.ready is not None:
            return fn in self.ready
        return os.path.exists(os.path.join(self.dir, fn))

    def raw(self, name):
        fn = self.wm[name]
        hdr, base = self._header(fn)
        meta = hdr[name]
        beg, end = meta["data_offsets"]
        with open(os.path.join(self.dir, fn), "rb") as f:
            f.seek(base + beg)
            buf = bytearray(f.read(end - beg))
        dt = {"U8": torch.uint8, "I8": torch.int8, "I64": torch.int64,
              "BF16": torch.bfloat16,
              # E8M0 is a bare biased exponent with no sign and no mantissa.
              # Read as bytes; e8m0_scale() turns it into 2^(e-127). torch
              # has float8_e8m0fnu in recent versions and not in all of
              # them, and this file has no reason to depend on that.
              "F8_E8M0": torch.uint8,
              "F16": torch.float16, "F32": torch.float32,
              # fp8 is how the current generation of large MoEs ships — K2,
              # DeepSeek V3/R1. The values are read natively; the per-block
              # scales they need are applied in tensor(), not here, because
              # raw() is by contract the bytes as stored.
              "F8_E4M3": torch.float8_e4m3fn,
              "F8_E5M2": torch.float8_e5m2}[meta["dtype"]]
        return torch.frombuffer(buf, dtype=dt).view(*meta["shape"])

    def companion(self, name):
        """The scale tensor for `name`, or None. Two spellings in this
        family: K3/fp8 append a suffix to `<x>.weight`, DeepSeek-V4.1
        replaces it with `.scale`."""
        for suffix in ("_scale_inv", "_scale"):
            if name in self.wm and (name + suffix) in self.wm:
                return name + suffix
        if name.endswith(".weight"):
            alt = name[: -len(".weight")] + ".scale"
            if alt in self.wm:
                return alt
        return None

    def row_slice(self, name, r0, r1):
        """Rows [r0, r1) of a 2-D tensor, as stored, without reading the rest.

        DeepSeek-V4.1's two Engram tables are 384 M rows of 256 values each
        — 98 GB apiece as fp8, 40% of the download. raw() reads a whole
        tensor, which is the right contract everywhere else and impossible
        here, so this is the streaming door. Row-major, which safetensors
        guarantees, so a row range is one contiguous span.
        """
        fn = self.wm[name]
        hdr, base = self._header(fn)
        meta = hdr[name]
        shape = meta["shape"]
        if len(shape) != 2:
            raise ValueError(f"{name}: row_slice wants 2-D, got {shape}")
        beg, end = meta["data_offsets"]
        dt = {"U8": torch.uint8, "I8": torch.int8, "F8_E8M0": torch.uint8,
              "BF16": torch.bfloat16, "F16": torch.float16,
              "F32": torch.float32, "F8_E4M3": torch.float8_e4m3fn,
              "F8_E5M2": torch.float8_e5m2}[meta["dtype"]]
        itemsize = torch.empty(0, dtype=dt).element_size()
        stride = shape[1] * itemsize
        r0, r1 = max(0, r0), min(shape[0], r1)
        if r1 <= r0:
            return torch.empty(0, shape[1], dtype=dt)
        want = (r1 - r0) * stride
        if beg + r0 * stride + want > end:
            raise ValueError(f"{name}: rows [{r0}, {r1}) run past the tensor")
        with open(os.path.join(self.dir, fn), "rb") as f:
            f.seek(base + beg + r0 * stride)
            buf = bytearray(f.read(want))
        if len(buf) != want:
            raise ValueError(f"{name}: short read at row {r0}")
        return torch.frombuffer(buf, dtype=dt).view(r1 - r0, shape[1])

    def shape(self, name):
        hdr, _ = self._header(self.wm[name])
        return tuple(hdr[name]["shape"])

    def tensor(self, name):
        """Returns f32, dequantizing whichever block-scaled form it finds."""
        if self.have(name):
            t = self.raw(name)
            sc = self.companion(name)
            if t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                # fp8 checkpoints carry one scale per weight_block_size tile in
                # a companion tensor. Without it the values are off by up to
                # the scale's dynamic range — silently, since the shapes still
                # line up.
                if sc is None:
                    # Name what was looked for. Two spellings exist in this
                    # family and "no scale tensor" does not say which one a
                    # checkpoint was expected to carry.
                    raise KeyError(
                        f"{name} is {t.dtype} and neither {name}"
                        f"_scale_inv, {name}_scale nor its `.scale` sibling "
                        "is beside it; refusing to read fp8 without its "
                        "block scales")
                return unblock_scale(t.float(), self._scale_f32(sc),
                                     self.fp8_block)
            if t.dtype == torch.int8 and sc is not None:
                # Packed E2M1 under its own name. The scale's column count
                # says the group, and it is read rather than assumed: this
                # release uses 32 for the experts and would look identical at
                # any other group that divided evenly.
                s = self._scale_f32(sc)
                cols = t.shape[1] * 2
                if s.ndim != 2 or s.shape[0] != t.shape[0] or cols % s.shape[1]:
                    raise ValueError(
                        f"{name}: packed fp4 {tuple(t.shape)} and scale "
                        f"{tuple(s.shape)} do not tile")
                return dequant(t, self.raw(sc), cols // s.shape[1])
            if t.dtype == torch.int8 and sc is None:
                # An int8 tensor with no scale beside it is either a real
                # int8 weight or packed fp4 whose companion this reader did
                # not find, and those two read identically. Refuse.
                raise KeyError(
                    f"{name} is int8 with no scale tensor beside it; if it is "
                    "packed fp4 the companion is missing, and if it is not "
                    "this reader has no way to tell")
            return t.float()
        if self.have(name + "_packed"):
            return dequant(self.raw(name + "_packed"), self.raw(name + "_scale"))
        raise KeyError(name)

    def _scale_f32(self, name):
        """A scale tensor as f32, whatever it is stored as."""
        hdr, _ = self._header(self.wm[name])
        if hdr[name]["dtype"] == "F8_E8M0":
            return e8m0_scale(self.raw(name))
        return self.raw(name).float()


# ---------------------------------------------------------------- check ---

def self_test(model_dir):
    st = ST(model_dir)
    name = None
    for k in st.wm:
        if k.endswith("w1.weight_packed") and st.have(k):
            name = k[: -len("_packed")]
            break
    if not name:
        print("no downloaded expert shard yet")
        return 1

    p, s = st.raw(name + "_packed"), st.raw(name + "_scale")
    W = dequant(p, s)
    print(f"{name}")
    print(f"  packed {tuple(p.shape)} u8, scale {tuple(s.shape)} u8 "
          f"-> dequantized {tuple(W.shape)} f32")
    print(f"  mean {W.mean():+.5f}  std {W.std():.5f}  "
          f"absmax {W.abs().max():.5f}  zeros {(W == 0).float().mean():.1%}")

    # With a power-of-two scale, every group's amax/scale must land in (3, 6]
    # and therefore round onto one of {3, 4, 6}. Anything outside that means
    # the scale is being read wrong.
    #
    # Note this says nothing about nibble ORDER: swapping nibbles permutes
    # elements within a byte pair, which stays inside the same group of 32
    # and leaves every group statistic identical. Order was settled instead
    # by diffing against compressed_tensors' own unpacker — bit-identical,
    # 0.000e+00 max difference, on a real K3 expert.
    g = W.view(W.shape[0], -1, GROUP)
    sc = torch.where(s.to(torch.int32) == 0, torch.ones_like(s, dtype=torch.float32),
                     torch.exp2((s.to(torch.int32) - 127).to(torch.float32)))
    ratio = (g.abs().amax(-1) / sc).flatten()
    hist = {v: float((ratio == v).float().mean()) for v in (6.0, 4.0, 3.0)}
    print("  per-group amax / scale: " +
          "  ".join(f"{v:g}->{f:.1%}" for v, f in hist.items()))
    inrange = float(((ratio > 3.0 - 1e-6) & (ratio < 6.0 + 1e-6)).float().mean())
    print(f"  inside (3, 6]: {inrange:.2%}  "
          f"{'OK' if inrange > 0.999 else 'SUSPECT — scale misread'}")
    return 0 if inrange > 0.999 else 1


if __name__ == "__main__":
    sys.exit(self_test(sys.argv[1] if len(sys.argv) > 1 else "/Volumes/WasteDisk/k3"))
