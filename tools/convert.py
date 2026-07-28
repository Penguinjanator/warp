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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mxfp4 import ST                                            # noqa: E402

# --- native VQ encoder (optional; ~15x the torch path) --------------------
_VQ = None


def _load_vq():
    """libwastevq: the assign fused with the argmin, so the [n, 256] distance
    matrix never exists. torch has to materialize it — 1.4 GB per stage for
    one expert matrix — which is what made conversion memory-bound."""
    global _VQ
    if _VQ is not None:
        return _VQ or None
    import ctypes
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
            _VQ = (lib, ctypes)
            return _VQ
    _VQ = False
    return None

MAGIC_EXPERT = 0x50584557        # 'WEXP'
MAGIC_CODEBOOK = 0x4B424357      # 'WCBK'
ALIGN = 4096
FMT_F32, FMT_F16, FMT_Q8G, FMT_Q4G, FMT_VQ3R, FMT_VQ2R = 0, 1, 2, 3, 4, 5
VEC_DIM = 8
CB_ENTRIES = 256
TRAIN_VECTORS = 300000       # vectors k-means sees per (layer, matrix kind)
IDX_BLOCK = 64          # rows per index block; matches VQ_TILE in the engine
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


def block_indices(idx, M, N, stages):
    """Reorder [stages, nvec] -> [M/B][v][row_in_block][stage].

    The engine walks a tile of rows for one vector position at a time; in
    row-major order those rows sit N/8*stages bytes apart, so each one is a
    separate cache line. Blocked, a tile's indices for a given position are
    contiguous. Measured 1.44x on the gather loop.
    """
    nvr = N // VEC_DIM
    t = idx.view(stages, M, nvr).permute(1, 2, 0)            # [M, nvr, st]
    pad = (-M) % IDX_BLOCK
    if pad:
        t = torch.cat([t, torch.zeros(pad, nvr, stages, dtype=t.dtype)], 0)
    nb = t.shape[0] // IDX_BLOCK
    t = t.view(nb, IDX_BLOCK, nvr, stages).permute(0, 2, 1, 3)
    return t.contiguous()


def quantize_vq(W, books, dev):
    """W [out, in] -> (indices uint8 [stages, nvec], per-channel fp16 scale)."""
    M, N = W.shape
    scale = W.abs().amax(-1, keepdim=True).clamp(min=1e-8)

    vq = _load_vq()
    if vq is not None:
        lib, ctypes = vq
        X = (W / scale).reshape(-1, VEC_DIM).contiguous().float()
        B = torch.stack([C.detach().cpu().float() for C in books]).contiguous()
        n, st = X.shape[0], len(books)
        out = torch.empty(n * st, dtype=torch.uint8)
        fp = ctypes.POINTER(ctypes.c_float)
        lib.waste_vq_encode(
            ctypes.cast(X.data_ptr(), fp), n,
            ctypes.cast(B.data_ptr(), fp), st, CB_ENTRIES, VEC_DIM,
            ctypes.cast(out.data_ptr(), ctypes.POINTER(ctypes.c_uint8)), 0)
        return out.view(n, st).T.contiguous(), scale.half().flatten().cpu()

    X = (W / scale).to(dev).reshape(-1, VEC_DIM)
    idxs, resid = [], X
    for C in books:
        i = assign(resid, C)
        idxs.append(i.to(torch.uint8).cpu())
        resid = resid - C[i]
    return torch.stack(idxs), scale.half().flatten().cpu()


def quantize_q4g(W, group=128):
    """int4 packed two per byte (low nibble first), fp16 scale per group.

    The trunk is the RAM floor and K3's is 54 B parameters — at 8 bits that
    is 54 GB, over the budget of the machine this is meant to run on. At
    4 bits it is 27."""
    orig = W.shape
    X = W.reshape(-1, orig[-1])
    N = X.shape[-1]
    pad = (-N) % group
    if pad:
        X = torch.nn.functional.pad(X, (0, pad))
    Xg = X.view(X.shape[0], -1, group)
    scale = Xg.abs().amax(-1, keepdim=True).clamp(min=1e-8) / 7.0
    Q = torch.clamp(torch.round(Xg / scale), -8, 7).to(torch.int32).flatten()
    nib = (Q + 8).to(torch.uint8) & 0x0F                 # 0..15, biased
    packed = (nib[0::2] | (nib[1::2] << 4)).contiguous()
    return packed, scale.half().flatten(), list(orig)


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

def write_expert_record(f, layer, eid, cb_base, payloads, scales, shapes):
    """One 4 KiB-aligned WEXP record: header, then gate|up|down indices,
    then the per-channel scales for all three."""
    hdr_size = 48
    off = hdr_size
    offsets = []
    body = bytearray()
    for i, p in enumerate(payloads):          # [stages, nvec] uint8
        offsets.append(off)
        M, N = shapes[i]
        b = raw_bytes(block_indices(p, M, N, p.shape[0]))
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


# ------------------------------------------------------------- worker ----

def convert_layer(job):
    """One layer, in its own process. Layers share nothing: separate bank
    file, separate codebook file, codebook ids assigned by the parent from
    the layer's position, so the merge is a concatenation in layer order."""
    (L, src, out, prefix, n_exp, stages, device, cb_sample, cb_base) = job
    import time as _t
    bank = os.path.join(out, f"experts-L{L}.bin")
    cbf = os.path.join(out, f"codebooks-L{L}.bin")
    # A finished bank means the layer is done. The per-layer codebook file
    # is gone by then because the parent merged it into codebooks.bin — an
    # earlier version also required it here, which made every resume
    # re-convert the whole model.
    if os.path.exists(bank):
        return (L, os.path.getsize(bank), cb_base, "cached")

    st = ST(src)
    dev = torch.device(device)

    def ename(e, tag):
        return f"{prefix}model.layers.{L}.block_sparse_moe.experts.{e}.{tag}.weight"

    if not (st.have(ename(0, "w1")) or st.have(ename(0, "w1") + "_packed")):
        return (L, 0, cb_base, "missing")

    t0 = _t.time()
    shapes = [tuple(st.tensor(ename(0, tag)).shape) for _, tag in KINDS]

    books, sample_ids = {}, list(range(0, n_exp, max(1, n_exp // cb_sample)))[:cb_sample]
    per = max(1, TRAIN_VECTORS // len(sample_ids))
    with open(cbf + ".tmp", "wb") as cf:
        for ki, (kind, tag) in enumerate(KINDS):
            chunks = []
            for e in sample_ids:
                W = st.tensor(ename(e, tag))
                sc = W.abs().amax(-1, keepdim=True).clamp(min=1e-8)
                V = (W / sc).reshape(-1, VEC_DIM)
                g = torch.Generator().manual_seed(1234 + e)
                chunks.append(V[torch.randperm(V.shape[0], generator=g)[:per]])
                del W, V
            X = torch.cat(chunks); del chunks
            books[kind] = train_codebooks(X, stages, dev, sample=TRAIN_VECTORS)
            del X
            for si, C in enumerate(books[kind]):
                cid = cb_base + ki * stages + si
                cf.write(struct.pack("<IHBBII", MAGIC_CODEBOOK, cid & 0xFFFF,
                                     FMT_VQ3R, VEC_DIM, CB_ENTRIES, 0))
                cf.write(raw_bytes(C.cpu().half()))
    os.replace(cbf + ".tmp", cbf)

    with open(bank + ".tmp", "wb") as f:
        for e in range(n_exp):
            payloads, scales = [], []
            for kind, tag in KINDS:
                W = st.tensor(ename(e, tag))
                idx, sc = quantize_vq(W, books[kind], dev)
                payloads.append(idx); scales.append(sc)
                del W
            write_expert_record(f, L, e, cb_base, payloads, scales, shapes)
    os.replace(bank + ".tmp", bank)
    return (L, os.path.getsize(bank), cb_base, f"{_t.time()-t0:.0f}s")



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/Volumes/WasteDisk/kimi-linear")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="", help="comma list; default = all MoE layers")
    ap.add_argument("--stages", type=int, default=3, help="3 = VQ3R, 2 = VQ2R")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--experts", type=int, default=0, help="limit experts (debug)")
    ap.add_argument("--trunk8", action="store_true",
                    help="keep the whole trunk at 8 bits (needs the RAM)")
    ap.add_argument("--skip-trunk", action="store_true",
                    help="experts only (the trunk is unchanged between runs)")
    ap.add_argument("--jobs", type=int, default=3,
                    help="layers converted in parallel; measured sweet spot "
                         "is 3 — beyond that the native encoder is already "
                         "using every core")
    ap.add_argument("--cb-sample", type=int, default=12,
                    help="experts sampled per layer to fit the codebooks")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cfg = json.load(open(os.path.join(args.src, "config.json")))
    prefix = ""
    if "text_config" in cfg:                     # K3 nests the text model
        cfg = {**cfg["text_config"], "_outer": {k: v for k, v in cfg.items()
                                                if k != "text_config"}}
        prefix = "language_model."
    st = ST(args.src)
    sr = ShardReader(args.src)
    dev = torch.device(args.device)
    print(f"device={dev}  stages={args.stages}")

    n_layers = cfg["num_hidden_layers"]
    n_exp = cfg.get("num_experts") or cfg.get("n_routed_experts")
    print(f"prefix {prefix!r}  layers {n_layers}  experts {n_exp}")
    first_dense = cfg.get("first_k_dense_replace", 0)
    if args.experts:
        n_exp = min(n_exp, args.experts)
    layers = ([int(x) for x in args.layers.split(",")] if args.layers
              else list(range(first_dense, n_layers)))

    # ---- expert banks, one layer at a time ------------------------------
    manifest_layers = {}


    def ename(L, e, tag):
        return f"{prefix}model.layers.{L}.block_sparse_moe.experts.{e}.{tag}.weight"

    n_cb_per_layer = 3 * args.stages
    jobs = [(L, args.src, args.out, prefix, n_exp, args.stages, str(dev),
             args.cb_sample, i * n_cb_per_layer)
            for i, L in enumerate(layers)]

    if args.jobs > 1:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")     # torch/MPS is not fork-safe
        print(f"converting {len(jobs)} layers with {args.jobs} processes", flush=True)
        with ctx.Pool(args.jobs) as pool:
            results = []
            for res in pool.imap_unordered(convert_layer, jobs):
                results.append(res)
                Lr, sz, base, how = res
                print(f"  layer {Lr}: {sz/2**20:.0f} MB, cb base {base} [{how}] "
                      f"({len(results)}/{len(jobs)})", flush=True)
    else:
        results = []
        for j in jobs:
            res = convert_layer(j)
            results.append(res)
            print(f"  layer {res[0]}: {res[1]/2**20:.0f} MB [{res[3]}]", flush=True)

    for L, sz, base, how in sorted(results):
        if sz:
            manifest_layers[str(L)] = {"file": f"experts-L{L}.bin",
                                       "experts": n_exp, "bytes": sz,
                                       "codebook_base": base}

    # merge the per-layer codebook files in layer order; ids were handed out
    # from the same order, so the concatenation is already indexed correctly.
    # If none are present every layer came from cache, and codebooks.bin is
    # already correct — rewriting it would truncate it to nothing.
    parts = [os.path.join(args.out, f"codebooks-L{L}.bin") for L in layers]
    if any(os.path.exists(p) for p in parts):
        merged = os.path.join(args.out, "codebooks.bin")
        with open(merged + ".tmp", "wb") as cb_out:
            for part in parts:
                if os.path.exists(part):
                    with open(part, "rb") as pf:
                        cb_out.write(pf.read())
                    os.remove(part)
        os.replace(merged + ".tmp", merged)
    else:
        print("codebooks: all layers cached, keeping existing codebooks.bin")


    # ---- tokenizer: copy it in so the container is self-contained -------
    import shutil
    for name in ("tiktoken.model", "tokenizer.model"):
        src_tok = os.path.join(args.src, name)
        if os.path.exists(src_tok):
            shutil.copyfile(src_tok, os.path.join(args.out, "tokenizer.model"))
            print(f"tokenizer: copied {name}")
            break

    # ---- trunk ----------------------------------------------------------
    trunk_path = os.path.join(args.out, "trunk.bin")
    tindex = []
    if args.skip_trunk:
        print("skipping trunk")
        return 0
    with open(trunk_path, "wb") as tf:
        for name in sorted(sr.names()):
            if ".experts." in name or name.endswith(("_packed", "_scale")):
                continue
            if not st.have(name):
                continue                      # shard not downloaded yet
            t = st.tensor(name)
            off = tf.tell()
            if t.dim() == 1 or t.numel() < 1 << 16:
                tf.write(raw_bytes(t.float()))
                tindex.append({"name": name, "fmt": FMT_F32, "off": off,
                               "shape": list(t.shape),
                               "bytes": tf.tell() - off})
            else:
                # 4 bits for the bulk; the embedding table and the output
                # head keep 8, they are small and sit at both ends of the
                # network where error is least forgiving
                big = not (name.endswith("embed_tokens.weight")
                           or name.endswith("lm_head.weight"))
                use4 = big and not args.trunk8
                q, sc, shape = (quantize_q4g(t) if use4 else quantize_q8g(t))
                tf.write(raw_bytes(q))
                sc_off = tf.tell()
                tf.write(raw_bytes(sc))
                tindex.append({"name": name, "fmt": FMT_Q4G if use4 else FMT_Q8G,
                               "off": off, "shape": shape, "group": 128,
                               "scale_off": sc_off, "bytes": tf.tell() - off})
    print(f"trunk: {os.path.getsize(trunk_path)/2**20:.0f} MB, "
          f"{len(tindex)} tensors")

    manifest = {
        "format_version": 0,
        "arch": cfg.get("model_type", "kimi"),
        "tensor_prefix": prefix,
        "config": cfg,
        "expert_quant": {"fmt": "VQ3R" if args.stages == 3 else "VQ2R",
                         "stages": args.stages, "vec_dim": VEC_DIM,
                         "entries": CB_ENTRIES, "index_block": IDX_BLOCK,
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
