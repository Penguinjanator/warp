#!/usr/bin/env python3
"""
k3parts_ref.py — recompute the K3 components from the released reference
source and diff against the C engine.

K3's weights cannot be run yet, so the parts whose maths is new are checked
in isolation on synthetic inputs. Each reference below is transcribed from
the shipped `modeling_kimi_linear.py` / `fla` sources, cited inline.

  ./test_k3parts /tmp/k3parts.bin
  uv run --with torch python tools/k3parts_ref.py /tmp/k3parts.bin
"""

import struct
import sys

import torch
import torch.nn.functional as F

TOL = 2e-5


class R:
    def __init__(self, path):
        self.b = open(path, "rb").read()
        self.o = 0

    def i(self):
        v = struct.unpack_from("<i", self.b, self.o)[0]; self.o += 4; return v

    def f(self):
        v = struct.unpack_from("<f", self.b, self.o)[0]; self.o += 4; return v

    def a(self, n):
        v = torch.tensor(struct.unpack_from(f"<{n}f", self.b, self.o)); self.o += 4 * n
        return v


def check(name, got, ref, tol=TOL):
    d = (got - ref).abs()
    rel = (got - ref).norm() / ref.norm().clamp(min=1e-12)
    ok = d.max().item() < tol
    print(f"  {name:<28} max|diff| {d.max():.3e}  rel {rel:.3e}   "
          f"{'OK' if ok else 'FAIL'}")
    return ok


def main(path):
    r = R(path)
    ok = True
    print("K3 components vs the released reference, synthetic inputs\n")

    # --- 1. SiTU -----------------------------------------------------------
    # modeling_kimi_linear.py, class SituAndMul:
    #   situ_a = beta * tanh(gate/beta) * sigmoid(gate)
    #   up     = linear_beta * tanh(up/linear_beta)      (when set)
    #   out    = situ_a * up
    n = r.i(); beta = r.f(); lbeta = r.f()
    g, u, out_c = r.a(n), r.a(n), r.a(n)
    situ_a = beta * torch.tanh(g / beta) * torch.sigmoid(g)
    ref = situ_a * (lbeta * torch.tanh(u / lbeta))
    ok &= check(f"SiTU (beta={beta:g}, lb={lbeta:g})", out_c, ref)

    # --- 2. decay gate -----------------------------------------------------
    # fla/ops/kda/gate.py:
    #   naive_kda_lowerbound_gate: g = lower_bound * sigmoid(A_log.exp() * z)
    #   naive_kda_gate:            g = -A_log.exp() * softplus(z)
    #   with z = g_in + dt_bias
    H = r.i(); D = r.i(); lb = r.f()
    gin = r.a(H * D).view(H, D); A = r.a(H).view(H, 1); dt = r.a(H * D).view(H, D)
    g_k3_c = r.a(H * D).view(H, D); g_lin_c = r.a(H * D).view(H, D)
    z = gin + dt
    ok &= check(f"decay gate, K3 (lb={lb:g})", g_k3_c, lb * torch.sigmoid(A.exp() * z))
    ok &= check("decay gate, Kimi-Linear", g_lin_c, -A.exp() * F.softplus(z))

    # --- 3. attention residuals -------------------------------------------
    # modeling_kimi_linear.py, _apply_attn_res:
    #   v      = cat(block_residual, prefix_sum.unsqueeze(1))
    #   k      = v * rsqrt(v.pow(2).mean(-1) + eps)
    #   scores = (k * (norm.weight * proj.weight)).sum(-1)
    #   out    = softmax(scores) @ v
    nb = r.i(); hid = r.i(); eps = r.f()
    blocks = r.a(nb * hid).view(nb, hid)
    ps = r.a(hid); nw = r.a(hid); pw = r.a(hid); out_c = r.a(hid)
    v = torch.cat([blocks, ps.unsqueeze(0)], 0)                  # [nb+1, hid]
    k = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)
    scores = (k * (nw * pw)).sum(-1)
    ref = (scores.softmax(-1).unsqueeze(0) @ v).squeeze(0)
    ok &= check(f"AttnRes ({nb} blocks, hid {hid})", out_c, ref)

    print("\n" + ("PASS — the C port matches the reference"
                  if ok else "FAIL — see above"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "k3parts.bin"))
