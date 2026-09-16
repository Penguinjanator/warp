#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""
ds41_ref.py — pure-PyTorch DeepSeek-V4.1-Flash, running off a WASTE container.

The oracle `src/model.c`'s CSA2 path is diffed against. Companion to
kimi_ref.py, which cannot serve this model: it has no CSA2, no Engram, and
its mHC collapses with the `pre` each site computes rather than with the
previous site's.

Same contract as the others, and it is the whole point: **weights come FROM
THE CONTAINER**, trunk dequantized on demand and experts dequantized per
use, so a diff against the C engine measures ARITHMETIC and not
quantization. Both sides see the same 3-bit experts and the same 4-bit
trunk.

Transcribed from the release's `inference/model.py`. Where that file batches
a prefill, this runs one token at a time, because that is what the engine
does — and the two are not trivially the same: the compressor publishes a
latent only when its group completes, so a decode step is stateful in a way
the batched form hides.

  uv run --with torch python tools/ds41_ref.py --container model.waste \\
      --ids 3,7,11,5 --top 10
  ... --dump out.bin          # f32 logits, for a byte-level diff
  ... --hidden h.bin          # the residual stream after every layer

`--no-engram` and `--no-sink` are bisection switches: they turn off one
mechanism on BOTH sides of a diff only if the engine is asked to as well,
so on their own they answer "is this term where the difference is".
"""

import argparse
import array
import json
import os
import struct
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kimi_ref import Container                                  # noqa: E402


def rms_norm(x, w, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def rope_tables(inv, pos):
    a = pos * inv
    return torch.cos(a), torch.sin(a)


def rope_apply(x, cs, sn, inverse=False):
    """Adjacent element pairs as complex numbers — the release's
    view_as_complex, which is NOT the split-halves convention."""
    e, o = x[..., 0::2], x[..., 1::2]
    if inverse:
        sn = -sn
    out = torch.empty_like(x)
    out[..., 0::2] = e * cs - o * sn
    out[..., 1::2] = e * sn + o * cs
    return out


def yarn_inv(dim, base, factor, orig, beta_fast, beta_slow, yarn):
    import math
    inv = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
    if yarn and factor > 1:
        def corrected(rot):
            return dim * math.log(orig / (rot * 2 * math.pi)) / (2 * math.log(base))
        low = max(math.floor(corrected(beta_fast)), 0)
        high = min(math.ceil(corrected(beta_slow)), dim - 1)
        if low == high:
            high += 0.001
        ramp = ((torch.arange(dim // 2, dtype=torch.float64) - low) /
                (high - low)).clamp(0, 1)
        smooth = 1 - ramp
        inv = inv / factor * (1 - smooth) + inv * smooth
    return inv.float()


class Engram:
    """The tables and the hashing that addresses them, read from the
    container: engram.json says which row, engram-L{n}.bin holds it."""

    def __init__(self, path, cfg, layer_ids, man):
        self.doc = json.load(open(os.path.join(path, "engram.json")))
        self.dim = cfg["engram_head_dim"]
        self.ng = cfg["engram_max_ngram_size"]
        self.nh = cfg["engram_n_heads"]
        self.pad = self.doc["pad_id"]
        n = cfg["vocab_size"]
        a = array.array("i")
        with open(os.path.join(path, "engram-tokmap.bin"), "rb") as f:
            a.fromfile(f, n)
        self.map = list(a)
        self.layers = {}
        for i, L in enumerate(layer_ids):
            e = man["engram"][str(L)]
            self.layers[L] = (i, e, open(os.path.join(path, f"engram-L{L}.bin"), "rb"))

    def hashes(self, i, hist, pos):
        """(ngram-1) * n_heads row ids for the position."""
        d = self.doc["layers"][i]
        out, rolling, blocked = [], 0, False
        for shift in range(self.ng):
            at = pos - shift
            src = hist[at] if at >= 0 else -1
            if at < 0 or src < 0:
                blocked = True
            tok = self.pad if blocked else src
            prod = tok * d["multipliers"][shift]
            if shift == 0:
                rolling = prod
                continue
            rolling ^= prod
            for h in range(self.nh):
                col = (shift - 1) * self.nh + h
                flat = [p for per in d["primes"] for p in per]
                out.append(rolling % flat[col] + d["offsets"][col])
        return out

    def rows(self, L, ids):
        i, meta, f = self.layers[L]
        bits, group, rb = meta["bits"], meta["group"], meta["row_bytes"]
        dim = meta["dim"]
        ngr = dim // group
        vals = []
        for r in ids:
            f.seek(r * rb)
            buf = f.read(rb)
            if bits == 4:
                b = torch.frombuffer(bytearray(buf[:dim // 2]),
                                     dtype=torch.uint8).int()
                q = torch.stack([b & 0x0F, b >> 4], -1).view(-1) - 8
            else:
                q = torch.frombuffer(bytearray(buf[:dim]), dtype=torch.int8).int()
            sc = torch.frombuffer(bytearray(buf[dim * bits // 8:]),
                                  dtype=torch.float16).float()
            vals.append((q.view(ngr, group).float() * sc.view(ngr, 1)).view(-1))
        return torch.cat(vals)


class DS41Ref:
    def __init__(self, c, engram=True, sink=True):
        self.c = c
        self.cfg = c.cfg
        self.t = c.t
        self.pfx = c.prefix
        self.eps = self.cfg["rms_norm_eps"]
        self.hid = self.cfg["hidden_size"]
        self.hd = self.cfg["head_dim"]
        self.nh = self.cfg["num_attention_heads"]
        self.rd = self.cfg["qk_rope_head_dim"]
        self.win = self.cfg["sliding_window"]
        self.H = self.cfg["hc_mult"]
        self.ratios = self.cfg["compress_ratios"]
        self.kv_src = set(self.cfg["kv_source_layer_ids"])
        self.ix_src = set(self.cfg["index_source_layer_ids"])
        self.want_engram = engram
        self.want_sink = sink
        rs = self.cfg.get("rope_scaling") or {}
        self.inv_plain = yarn_inv(self.rd, self.cfg.get("rope_theta", 10000.0),
                                  0, 0, 0, 0, False)
        self.inv_comp = yarn_inv(
            self.rd, self.cfg["compress_rope_theta"],
            rs.get("factor", 1.0),
            rs.get("original_max_position_embeddings", 4096),
            rs.get("beta_fast", 32), rs.get("beta_slow", 1), True)
        self.eg = None
        ids = self.cfg.get("engram_layer_ids") or []
        if ids and engram:
            self.eg = Engram(c.path, self.cfg, ids, c.man)
            self.eg_index = {L: i for i, L in enumerate(ids)}
        self.reset()

    def reset(self):
        n = self.cfg["num_hidden_layers"]
        self.winkv = [None] * n
        self.ckv = [None] * n
        self.ikey = [None] * n
        self.cpool = [[] for _ in range(n)]
        self.hist = []
        # What a source publishes for the layers after it. Upstream keeps
        # this in a module-level singleton and says so — "layers run in
        # order and every source writes before its consumers read, so one
        # slot each is enough and nothing needs resetting between forwards".
        # Rebuilding it per step instead makes a layer attend over nothing
        # on any step whose compressor did not complete a group: at ratio 2
        # that is every other token, and the engine was right where this
        # was wrong.
        self.shared = {}

    def inv(self, L):
        return self.inv_comp if self.ratios[L] else self.inv_plain

    def W(self, fmt, *a):
        return self.t[(self.pfx + fmt) % a]

    # ---------------------------------------------------------------- CSA2
    def attn(self, L, x, pos, shared):
        cfg, hd, nh, rd = self.cfg, self.hd, self.nh, self.rd
        inv = self.inv(L)
        cs, sn = rope_tables(inv, pos)

        qr = rms_norm(self.W("model.layers.%d.self_attn.q_a_proj.weight", L) @ x,
                      self.W("model.layers.%d.self_attn.q_a_layernorm.weight", L),
                      self.eps)
        q = (self.W("model.layers.%d.self_attn.q_b_proj.weight", L) @ qr).view(nh, hd)
        q = torch.cat([q[:, :hd - rd], rope_apply(q[:, hd - rd:], cs, sn)], -1)

        kv = rms_norm(self.W("model.layers.%d.self_attn.kv_proj.weight", L) @ x,
                      self.W("model.layers.%d.self_attn.kv_layernorm.weight", L),
                      self.eps)
        kv = torch.cat([kv[:hd - rd], rope_apply(kv[hd - rd:], cs, sn)], -1)
        if self.winkv[L] is None:
            self.winkv[L] = torch.zeros(self.win, hd)
        self.winkv[L][pos % self.win] = kv
        wvalid = min(pos + 1, self.win)
        kvs = [self.winkv[L][s] for s in range(wvalid)]

        r = self.ratios[L]
        if r:
            nlat = (pos + 1) // r
            lat = self.compress(L, x, pos) if L in self.kv_src else None
            if L in self.kv_src:
                shared["ckv"] = L
            sel = shared.get("sel", [])
            if L in self.ix_src:
                sel = self.index(L, x, qr, lat, pos, nlat, shared)
                shared["sel"] = sel
            if lat is not None:
                cs2, sn2 = rope_tables(inv, pos + 1 - r)
                lat = torch.cat([lat[:hd - rd], rope_apply(lat[hd - rd:], cs2, sn2)])
                if self.ckv[L] is None:
                    self.ckv[L] = {}
                self.ckv[L][pos // r] = lat
            src = shared.get("ckv")
            if src is not None:
                for t in sel:
                    if t < nlat and t in self.ckv[src]:
                        kvs.append(self.ckv[src][t])

        K = torch.stack(kvs)                                   # [n, hd]
        scores = (q @ K.T) / (hd ** 0.5)                       # [nh, n]
        mx = scores.max(-1, keepdim=True).values
        ex = torch.exp(scores - mx)
        sink = self.W("model.layers.%d.self_attn.attn_sink", L)
        den = ex.sum(-1, keepdim=True)
        if self.want_sink:
            den = den + torch.exp(sink.view(-1, 1) - mx)
        o = (ex / den) @ K                                     # [nh, hd]
        o = torch.cat([o[:, :hd - rd],
                       rope_apply(o[:, hd - rd:], cs, sn, inverse=True)], -1)

        G, R = cfg["o_groups"], cfg["o_lora_rank"]
        W = nh * hd // G
        wa = self.W("model.layers.%d.self_attn.o_a_proj.weight", L).view(G, R, W)
        og = o.reshape(G, W)
        acc = torch.einsum("gw,grw->gr", og, wa).reshape(-1)
        return self.W("model.layers.%d.self_attn.o_b_proj.weight", L) @ acc

    def compress(self, L, x, pos):
        r = self.ratios[L]
        kv = self.W("model.layers.%d.self_attn.compress.kv_proj.weight", L) @ x
        if r == 1:
            return rms_norm(kv, self.W(
                "model.layers.%d.self_attn.compress.norm.weight", L), self.eps)
        sc = self.W("model.layers.%d.self_attn.compress.gate_proj.weight", L) @ x
        slot = pos % r
        while len(self.cpool[L]) <= slot:
            self.cpool[L].append(None)
        self.cpool[L][slot] = (kv, sc)
        if (pos + 1) % r:
            return None
        kvs = torch.stack([p[0] for p in self.cpool[L]])        # [r, hd]
        scs = torch.stack([p[1] for p in self.cpool[L]])
        w = torch.softmax(scs, dim=0)                          # per channel
        return rms_norm((kvs * w).sum(0), self.W(
            "model.layers.%d.self_attn.compress.norm.weight", L), self.eps)

    def index(self, L, x, qr, lat, pos, nlat, shared):
        cfg = self.cfg
        D, IH, rd, r = cfg["index_head_dim"], cfg["index_n_heads"], self.rd, self.ratios[L]
        inv = self.inv(L)
        if L in self.kv_src and lat is not None:
            k = rms_norm(self.W("model.layers.%d.self_attn.indexer.k_proj.weight", L) @ lat,
                         self.W("model.layers.%d.self_attn.indexer.k_layernorm.weight", L),
                         self.eps)
            cs2, sn2 = rope_tables(inv, pos + 1 - r)
            k = torch.cat([k[:D - rd], rope_apply(k[D - rd:], cs2, sn2)])
            if self.ikey[L] is None:
                self.ikey[L] = {}
            self.ikey[L][pos // r] = k
            shared["ikey"] = L
        src = shared.get("ikey")
        if src is None or nlat == 0:
            return []
        q = (self.W("model.layers.%d.self_attn.indexer.q_b_proj.weight", L) @ qr).view(IH, D)
        cs, sn = rope_tables(inv, pos)
        q = torch.cat([q[:, :D - rd], rope_apply(q[:, D - rd:], cs, sn)], -1)
        w = self.W("model.layers.%d.self_attn.indexer.weights_proj.weight", L) @ x
        w = w * ((D ** -0.5) * (IH ** -0.5))
        K = torch.stack([self.ikey[src][t] for t in range(nlat)])
        score = (F.relu(q @ K.T) * w.view(-1, 1)).sum(0)        # [nlat]

        cand_src = cfg.get("candidate_source_layer_id", -1)
        if cand_src == L:
            shared["cand"] = self.candidates(score, nlat)
        elif 0 <= cand_src < L and "cand" in shared:
            score = score.masked_fill(~shared["cand"][:nlat], float("-inf"))
        keep = min(cfg["index_topk"], nlat)
        idx = torch.topk(score, keep).indices
        return sorted(int(i) for i in idx)

    def candidates(self, score, n):
        bs = self.cfg["candidate_block_size"]
        nb = (n + bs - 1) // bs
        pad = torch.full((nb * bs,), float("-inf"))
        pad[:n] = score
        blk = pad.view(nb, bs).amax(-1)
        blk[(n - 1) // bs] = float("inf")
        keep = min(self.cfg["candidate_topk_blocks"], nb)
        top = torch.topk(blk, keep)
        mask = torch.zeros(nb, dtype=torch.bool)
        mask[top.indices[top.values > float("-inf")]] = True
        return mask.repeat_interleave(bs)[:n]

    # -------------------------------------------------------------- Engram
    def engram(self, L, h, pos):
        i = self.eg_index[L]
        ids = self.eg.hashes(i, self.hist, pos)
        row = self.eg.rows(L, ids)
        kv = self.W("model.layers.%d.engram.wkv.weight", L) @ row
        key = kv[:self.H * self.hid].view(self.H, self.hid)
        value = kv[self.H * self.hid:]
        w = (self.W("model.layers.%d.engram.q_weight", L) *
             self.W("model.layers.%d.engram.k_weight", L))
        rstd = (torch.rsqrt(h.pow(2).mean(-1) + self.eps) *
                torch.rsqrt(key.pow(2).mean(-1) + self.eps))
        dot = (h * w * key).sum(-1) * rstd * self.hid ** -0.5
        gate = torch.sigmoid(torch.copysign(
            dot.abs().clamp_min(1e-6).sqrt(), dot))
        return h + gate.view(-1, 1) * value.view(1, -1)

    # ----------------------------------------------------------------- mHC
    def mixes(self, L, site, x):
        H, hid = self.H, self.hid
        nmix = (2 + H) * H
        flat = x.reshape(-1)
        rstd = torch.rsqrt(flat.pow(2).mean() + self.eps)
        w = (self.W("model.layers.%d.hc_%s_fn", L, site) @ flat) * rstd
        base = self.W("model.layers.%d.hc_%s_base", L, site)
        sc = self.W("model.layers.%d.hc_%s_scale", L, site)
        eps = self.cfg["hc_eps"]
        pre = torch.sigmoid(w[:H] * sc[0] + base[:H]) + eps
        post = 2 * torch.sigmoid(w[H:2 * H] * sc[1] + base[H:2 * H])
        comb = (w[2 * H:nmix] * sc[2] + base[2 * H:nmix]).view(H, H)
        comb = torch.softmax(comb, -1) + eps
        comb = comb / (comb.sum(0, keepdim=True) + eps)
        for _ in range(self.cfg["hc_sinkhorn_iters"] - 1):
            comb = comb / (comb.sum(-1, keepdim=True) + eps)
            comb = comb / (comb.sum(0, keepdim=True) + eps)
        return pre, post, comb

    def moe(self, L, x):
        cfg, c = self.cfg, self.c
        E, K = cfg["num_experts"], cfg["num_experts_per_token"]
        g = self.W("model.layers.%d.block_sparse_moe.gate.weight", L) @ x
        fn = cfg.get("scoring_func", "sigmoid")
        if fn == "sqrtsoftplus":
            score = F.softplus(g).sqrt()
        elif fn == "softmax":
            score = torch.softmax(g, -1)
        else:
            score = torch.sigmoid(g)
        bias = self.t.get(self.pfx +
            "model.layers.%d.block_sparse_moe.gate.e_score_correction_bias" % L)
        sel = torch.topk(score + (bias if bias is not None else 0), K).indices
        w = score[sel]
        if cfg.get("moe_renormalize") and K > 1:
            w = w / (w.sum() + 1e-20)
        w = w * cfg.get("routed_scaling_factor", 1.0)
        lim = cfg.get("swiglu_limit", 0.0)

        def ffn(gate_w, up_w, down_w, inp):
            a, b = gate_w @ inp, up_w @ inp
            if lim > 0:
                a = a.clamp(max=lim)
                b = b.clamp(-lim, lim)
            return down_w @ (F.silu(a) * b)

        out = torch.zeros_like(x)
        for j in range(K):
            e = c.expert(L, int(sel[j]))
            out = out + w[j] * ffn(e["gate"], e["up"], e["down"], x)
        sh = "model.layers.%d.block_sparse_moe.shared_experts.%s_proj.weight"
        return out + ffn(self.W(sh, L, "gate"), self.W(sh, L, "up"),
                         self.W(sh, L, "down"), x)

    # --------------------------------------------------------------- step
    def step(self, token, pos, hidden_out=None):
        cfg = self.cfg
        H, hid = self.H, self.hid
        while len(self.hist) <= pos:
            self.hist.append(-1)
        self.hist[pos] = (self.eg.map[token] if self.eg else -1)

        emb = self.t[self.pfx + "model.embed_tokens.weight"][token]
        x = emb.view(1, -1).repeat(H, 1)
        pre = torch.zeros(H)
        pre[0] = 1.0
        shared = self.shared
        for L in range(cfg["num_hidden_layers"]):
            if self.eg and L in self.eg_index:
                x = self.engram(L, x, pos)
            apre, apost, acomb = self.mixes(L, "attn", x)
            col = (pre.view(-1, 1) * x).sum(0)
            y = self.attn(L, rms_norm(
                col, self.W("model.layers.%d.input_layernorm.weight", L),
                self.eps), pos, shared)
            x = apost.view(-1, 1) * y.view(1, -1) + torch.einsum("ki,kd->id", acomb, x)

            fpre, fpost, fcomb = self.mixes(L, "ffn", x)
            col = (apre.view(-1, 1) * x).sum(0)
            y = self.moe(L, rms_norm(
                col, self.W("model.layers.%d.post_attention_layernorm.weight", L),
                self.eps))
            x = fpost.view(-1, 1) * y.view(1, -1) + torch.einsum("ki,kd->id", fcomb, x)
            pre = fpre
            if hidden_out is not None:
                hidden_out.append(x.reshape(-1).clone())
        col = (pre.view(-1, 1) * x).sum(0)
        h = rms_norm(col, self.W("model.norm.weight"), self.eps)
        return self.t[self.pfx + "lm_head.weight"] @ h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--container", required=True)
    ap.add_argument("--ids", required=True, help="comma-separated token ids")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--dump", help="write the final f32 logits here")
    ap.add_argument("--hidden", help="write the residual stream after every layer")
    ap.add_argument("--no-engram", action="store_true")
    ap.add_argument("--no-sink", action="store_true")
    a = ap.parse_args()

    torch.set_grad_enabled(False)
    c = Container(a.container)
    ref = DS41Ref(c, engram=not a.no_engram, sink=not a.no_sink)
    ids = [int(x) for x in a.ids.split(",")]
    hid = [] if a.hidden else None
    logits = None
    for p, t in enumerate(ids):
        logits = ref.step(t, p, hid)
    v, i = torch.topk(logits, a.top)
    print(f"{len(ids)} tokens, vocab {logits.numel()}")
    for k in range(a.top):
        print(f"  {int(i[k]):6d}  {float(v[k]):+.6f}")
    if a.dump:
        v = logits.float().tolist()
        open(a.dump, "wb").write(struct.pack(f"<{len(v)}f", *v))
        print(f"wrote {a.dump}")
    if a.hidden:
        with open(a.hidden, "wb") as f:
            for h in hid:
                v = h.float().tolist()
                f.write(struct.pack(f"<{len(v)}f", *v))
        print(f"wrote {a.hidden} ({len(hid)} layers)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
