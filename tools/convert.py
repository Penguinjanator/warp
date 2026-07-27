#!/usr/bin/env python3
"""
convert.py — safetensors (Kimi family) -> WASTE container.

Writes the layout in docs/FORMAT.md with records binary-compatible with
src/waste_format.h: 4 KiB-aligned expert records holding gate/up/down
adjacently so one pread() yields a whole expert.

Experts use VQ3R by default: 3-stage residual VQ over 8-dim vectors,
256 entries per stage (3.0 bits/weight), plus one FP16 scale per output
channel. Codebooks are trained per (layer, matrix kind) on a sample and
stored once. Gate 3 measured this recipe on real Kimi experts.

  uv run --with torch python tools/convert.py \
      --src /Volumes/WasteDisk/kimi-linear \
      --out /Volumes/WasteDisk/kimi-linear.waste \
      --layers 1,2                       # subset for a fast first pass

Resumable: a layer whose bank file already exists is skipped. Never holds
more than one layer of experts in memory.
"""

import argparse
import json
import os
import struct
import sys
import time
import zlib

import torch

MAGIC_EXPERT = 0x50584557        # 'WEXP'
MAGIC_CODEBOOK = 0x4B424357      # 'WCBK'
ALIGN = 4096
FMT_F32, FMT_F16, FMT_Q8G, FMT_Q4G, FMT_VQ3R, FMT_VQ2R = 0, 1, 2, 3, 4, 5
VEC_DIM = 8
CB_ENTRIES = 256
KINDS = (("gate", "w1"), ("up", "w3"), ("down", "w2"))   # Mixtral naming


# ---------------------------------------------------------------- reading --

class ShardReader:
    """Lazy safetensors reader over a sharded model directory."""

    def __init__(self, model_dir):
        self.dir = model_dir
        idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))
        self.wm = idx["weight_map"]
        self._hdr = {}

    def _header(self, fn):
        if fn not in self._hdr:
            with open(os.path.join(self.dir, fn), "rb") as f:
                (hlen,) = struct.unpack("<Q", f.read(8))
                self._hdr[fn] = (json.loads(f.read(hlen)), 8 + hlen)
        return self._hdr[fn]

    def names(self):
        return self.wm.keys()

    def get(self, name):
        fn = self.wm[name]
        hdr, base = self._header(fn)
        meta = hdr[name]
        beg, end = meta["data_offsets"]
        with open(os.path.join(self.dir, fn), "rb") as f:
            f.seek(base + beg)
            raw = f.read(end - beg)
        dt = {"BF16": torch.bfloat16, "F16": torch.float16,
              "F32": torch.float32}[meta["dtype"]]
        return torch.frombuffer(bytearray(raw), dtype=dt).view(*meta["shape"]).float()


# ------------------------------------------------------------ quantizers --

def train_codebooks(X, n_stages, dev, iters=10, sample=300000, seed=0):
    """Residual k-means: one codebook per stage, fitted on the running
    residual. X is [n, VEC_DIM] on `dev`."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    books = []
    resid = X[torch.randperm(X.shape[0], generator=g)[:sample]].to(dev)
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
    return books


def assign(X, C, chunk=1 << 20):
    """Nearest centroid via the ||x||^2 - 2x.c + ||c||^2 expansion (a GEMM)."""
    cn = (C * C).sum(1)
    out = torch.empty(X.shape[0], dtype=torch.long, device=X.device)
    for s in range(0, X.shape[0], chunk):
        x = X[s:s + chunk]
        d = cn.unsqueeze(0) - 2.0 * (x @ C.T)
        out[s:s + chunk] = d.argmin(1)
    return out


def quantize_vq(W, books, dev):
    """W [out, in] -> (indices uint8 [stages, nvec], per-channel fp16 scale)."""
    M, N = W.shape
    scale = W.abs().amax(-1, keepdim=True).clamp(min=1e-8)
    X = (W / scale).to(dev).reshape(-1, VEC_DIM)
    idxs = []
    resid = X
    for C in books:
        i = assign(resid, C)
        idxs.append(i.to(torch.uint8).cpu())
        resid = resid - C[i]
    return torch.stack(idxs), scale.half().flatten().cpu()


def quantize_q8g(W, group=128):
    """int8 + fp16 scale per group of `group` inputs. Returns (bytes, meta)."""
    orig = W.shape
    X = W.reshape(-1, orig[-1])
    N = X.shape[-1]
    pad = (-N) % group
    if pad:
        X = torch.nn.functional.pad(X, (0, pad))
    Xg = X.view(X.shape[0], -1, group)
    scale = Xg.abs().amax(-1, keepdim=True).clamp(min=1e-8) / 127.0
    Q = torch.clamp(torch.round(Xg / scale), -127, 127).to(torch.int8)
    return Q.flatten(), scale.half().flatten(), list(orig)


# ---------------------------------------------------------------- writing --

def raw_bytes(t):
    """Contiguous little-endian bytes of a tensor (torch has no .tobytes())."""
    t = t.detach().cpu().contiguous()
    n = t.numel() * t.element_size()
    buf = torch.empty(n, dtype=torch.uint8)
    buf.view(t.dtype)[:t.numel()] = t.flatten()
    return bytes(memoryview(buf.numpy() if False else bytearray(buf.tolist())))

def write_expert_record(f, layer, eid, cb_base, payloads, scales):
    """One 4 KiB-aligned WEXP record: header, then gate|up|down indices,
    then the per-channel scales for all three."""
    hdr_size = 48
    off = hdr_size
    offsets = []
    body = bytearray()
    for p in payloads:                       # [stages, nvec] uint8
        offsets.append(off)
        # transpose so a vector's stage indices sit together (decode locality)
        b = raw_bytes(p.T.contiguous())
        body += b
        off += len(b)
    corr_off = off
    for s in scales:
        b = raw_bytes(s)
        body += b
        off += len(b)

    total = hdr_size + len(body)
    blocks = (total + ALIGN - 1) // ALIGN
    pad = blocks * ALIGN - total
    crc = zlib.crc32(bytes(body)) & 0xFFFFFFFF

    hdr = struct.pack("<IHHBBHHHIIIIIIII",
                      MAGIC_EXPERT, layer, eid, FMT_VQ3R, 0, cb_base, 0, 0,
                      blocks, offsets[0], offsets[1], offsets[2], corr_off,
                      crc, 0, 0)
    assert len(hdr) == hdr_size, len(hdr)
    f.write(hdr)
    f.write(body)
    f.write(b"\0" * pad)
    return blocks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/Volumes/WasteDisk/kimi-linear")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="", help="comma list; default = all MoE layers")
    ap.add_argument("--stages", type=int, default=3, help="3 = VQ3R, 2 = VQ2R")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--experts", type=int, default=0, help="limit experts (debug)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cfg = json.load(open(os.path.join(args.src, "config.json")))
    sr = ShardReader(args.src)
    dev = torch.device(args.device)
    print(f"device={dev}  stages={args.stages}")

    n_layers = cfg["num_hidden_layers"]
    n_exp = cfg.get("num_experts") or cfg.get("n_routed_experts")
    first_dense = cfg.get("first_k_dense_replace", 0)
    if args.experts:
        n_exp = min(n_exp, args.experts)
    layers = ([int(x) for x in args.layers.split(",")] if args.layers
              else list(range(first_dense, n_layers)))

    # ---- expert banks, one layer at a time ------------------------------
    cb_path = os.path.join(args.out, "codebooks.bin")
    cb_f = open(cb_path, "ab" if os.path.exists(cb_path) else "wb")
    cb_index, cb_next = {}, cb_f.tell() // (16 + CB_ENTRIES * VEC_DIM * 2)
    manifest_layers = {}

    for L in layers:
        bank = os.path.join(args.out, f"experts-L{L}.bin")
        if os.path.exists(bank):
            print(f"layer {L}: bank exists, skipping")
            continue
        t0 = time.time()
        prefix = f"model.layers.{L}.block_sparse_moe.experts"
        if f"{prefix}.0.w1.weight" not in sr.wm:
            print(f"layer {L}: no experts, skipping")
            continue

        # load this layer's experts (one layer resident at a time)
        W = {}
        for kind, tag in KINDS:
            W[kind] = torch.stack([sr.get(f"{prefix}.{e}.{tag}.weight")
                                   for e in range(n_exp)])
        print(f"layer {L}: loaded {n_exp} experts "
              f"({sum(w.numel() for w in W.values())/1e6:.0f} M params) "
              f"in {time.time()-t0:.1f}s", flush=True)

        # per-(layer, kind) codebooks, trained on normalized weights
        books = {}
        for kind, _ in KINDS:
            Wk = W[kind]
            sc = Wk.abs().amax(-1, keepdim=True).clamp(min=1e-8)
            X = (Wk / sc).reshape(-1, VEC_DIM)
            books[kind] = train_codebooks(X, args.stages, dev)
            cb_index[(L, kind)] = cb_next
            for C in books[kind]:
                cb_f.write(struct.pack("<IHBBII", MAGIC_CODEBOOK, cb_next & 0xFFFF,
                                       FMT_VQ3R, VEC_DIM, CB_ENTRIES, 0))
                cb_f.write(raw_bytes(C.cpu().half()))
                cb_next += 1
        print(f"layer {L}: codebooks trained ({time.time()-t0:.1f}s)", flush=True)

        with open(bank + ".tmp", "wb") as f:
            for e in range(n_exp):
                payloads, scales = [], []
                for kind, _ in KINDS:
                    idx, sc = quantize_vq(W[kind][e], books[kind], dev)
                    payloads.append(idx)
                    scales.append(sc)
                write_expert_record(f, L, e, cb_index[(L, "gate")], payloads, scales)
        os.replace(bank + ".tmp", bank)
        sz = os.path.getsize(bank)
        manifest_layers[str(L)] = {"file": os.path.basename(bank),
                                   "experts": n_exp, "bytes": sz,
                                   "codebook_base": cb_index[(L, "gate")]}
        print(f"layer {L}: wrote {sz/2**20:.0f} MB in {time.time()-t0:.1f}s\n",
              flush=True)
        del W

    cb_f.close()

    # ---- trunk ----------------------------------------------------------
    trunk_path = os.path.join(args.out, "trunk.bin")
    tindex = []
    with open(trunk_path, "wb") as tf:
        for name in sorted(sr.names()):
            if ".experts." in name:
                continue
            t = sr.get(name)
            off = tf.tell()
            if t.dim() == 1 or t.numel() < 1 << 16:
                tf.write(raw_bytes(t.float()))
                tindex.append({"name": name, "fmt": FMT_F32, "off": off,
                               "shape": list(t.shape),
                               "bytes": tf.tell() - off})
            else:
                q, sc, shape = quantize_q8g(t)
                tf.write(raw_bytes(q))
                sc_off = tf.tell()
                tf.write(raw_bytes(sc))
                tindex.append({"name": name, "fmt": FMT_Q8G, "off": off,
                               "shape": shape, "group": 128,
                               "scale_off": sc_off, "bytes": tf.tell() - off})
    print(f"trunk: {os.path.getsize(trunk_path)/2**20:.0f} MB, "
          f"{len(tindex)} tensors")

    manifest = {
        "format_version": 0,
        "arch": cfg.get("model_type", "kimi"),
        "config": cfg,
        "expert_quant": {"fmt": "VQ3R" if args.stages == 3 else "VQ2R",
                         "stages": args.stages, "vec_dim": VEC_DIM,
                         "entries": CB_ENTRIES,
                         "bits_per_weight": args.stages},
        "layers": manifest_layers,
        "trunk": tindex,
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"wrote {args.out}/manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
