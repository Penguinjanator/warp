#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""test_convert_resume.py — tools/convert.py must resume, and must never
renumber a bank that already exists.

Written after a resume fix removed the one thing resume depended on: the
cache check needed a manifest, and the manifest is published last, so a
first conversion interrupted at layer 40 of 92 re-did all 40. Six
scenarios, because the interesting ones are the partial states.

No torch, no source weights, no minutes of quantization: the expensive
half is stubbed and the decision half is convert.py's own.

Only the quantizer is stubbed: convert_layer is replaced by a fake that
writes the bank and codebook part a real one would, using the base the
parent handed it. Everything that decides *which* base that is, and how the
parts are merged, is convert.py's own code.

  python3 tests/test_convert_resume.py
"""
import json
import os
import shutil
import struct
import sys
import tempfile
import types

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

N_LAYERS, N_EXPERTS, FIRST_DENSE, STAGES = 6, 4, 1, 3
MOE_LAYERS = list(range(FIRST_DENSE, N_LAYERS))


TRUNK_TENSOR = "model.norm.weight"      # one small f32 trunk tensor is enough


class FakeTensor:
    """Just enough tensor for the trunk loop's f32 branch."""
    shape = (8,)

    def dim(self): return 1
    def numel(self): return 8
    def float(self): return self


def install_stubs():
    torch = types.ModuleType("torch")
    torch.device = lambda s: s
    torch.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: False))
    sys.modules["torch"] = torch
    mx = types.ModuleType("mxfp4")

    class ST:
        def __init__(self, src): pass

        def have(self, name):
            if name == TRUNK_TENSOR:
                return True
            return ".experts." in name and "_packed" not in name

        def tensor(self, name): return FakeTensor()
        def names(self): return [TRUNK_TENSOR]
    mx.ST = ST
    sys.modules["mxfp4"] = mx
    sys.path.insert(0, os.path.join(REPO, "tools"))
    import convert
    convert.ShardReader = lambda src: types.SimpleNamespace(
        names=lambda: [TRUNK_TENSOR])
    # Serialization is not what this checks, and the real one needs torch.
    convert.raw_bytes = lambda t: b"\0" * (t.numel() * 4)
    return convert


CONV = install_stubs()
REC = struct.calcsize("<IHBBII") + CONV.CB_ENTRIES * CONV.VEC_DIM * 2
PER_LAYER = 3 * STAGES

converted = []           # layers the fake quantizer actually ran on


def fake_convert_layer(job):
    (L, src, out, prefix, n_exp, stages, device, cb_sample, cb_base,
     cached_ok) = job
    bank = os.path.join(out, f"experts-L{L}.bin")
    if cached_ok and os.path.exists(bank):
        return (L, os.path.getsize(bank), cb_base, "cached")
    converted.append(L)
    # A bank whose every record names cb_base, as write_expert_record does.
    with open(bank, "wb") as f:
        for eid in range(n_exp):
            f.write(struct.pack("<IHHBBHHH", CONV.MAGIC_EXPERT, L, eid,
                                6, 0, cb_base, 0, 0))
            f.write(b"\0" * (CONV.ALIGN - 16))
    with open(os.path.join(out, f"codebooks-L{L}.bin"), "wb") as f:
        for i in range(PER_LAYER):
            # Tag each record with its absolute id so the merge can be checked.
            f.write(struct.pack("<I", cb_base + i) + b"\0" * (REC - 4))
    return (L, os.path.getsize(bank), cb_base, "converted")


CONV.convert_layer = fake_convert_layer


def run(out, src, layers=None, jobs=1, skip_trunk=False, want_rc=0):
    del converted[:]
    argv = ["convert.py", "--src", src, "--out", out, "--jobs", str(jobs),
            "--stages", str(STAGES)]
    if layers:
        argv += ["--layers", ",".join(str(x) for x in layers)]
    if skip_trunk:
        argv += ["--skip-trunk"]
    old = sys.argv
    sys.argv = argv
    try:
        rc = CONV.main()
    finally:
        sys.argv = old
    assert rc == want_rc, f"convert.main() returned {rc}, wanted {want_rc}"
    return list(converted)


def manifest(out):
    p = os.path.join(out, "manifest.json")
    return json.load(open(p)) if os.path.exists(p) else None


def books(out):
    p = os.path.join(out, "codebooks.bin")
    if not os.path.exists(p):
        return []
    raw = open(p, "rb").read()
    assert len(raw) % REC == 0, "codebooks.bin is not a whole number of records"
    return [struct.unpack_from("<I", raw, i * REC)[0]
            for i in range(len(raw) // REC)]


fails = 0


def ck(cond, what):
    global fails
    print(f"  {'ok  ' if cond else 'FAIL'}  {what}")
    if not cond:
        fails += 1


def check_consistency(out, expect_layers):
    """Every bank's own base must index its own records in codebooks.bin."""
    b = books(out)
    for L in expect_layers:
        base = CONV.bank_codebook_base(os.path.join(out, f"experts-L{L}.bin"))
        if base < 0 or base + PER_LAYER > len(b):
            return False, f"L{L} base {base} outside {len(b)} records"
        got = b[base:base + PER_LAYER]
        if got != list(range(base, base + PER_LAYER)):
            return False, f"L{L} base {base} points at {got[:3]}..."
    return True, f"{len(expect_layers)} banks, {len(b)} records"


def main():
    tmp = tempfile.mkdtemp(prefix="resume-")
    src = os.path.join(tmp, "src")
    os.makedirs(src)
    json.dump({"num_hidden_layers": N_LAYERS, "num_experts": N_EXPERTS,
               "first_k_dense_replace": FIRST_DENSE,
               "num_experts_per_token": 2, "hidden_size": 8,
               "moe_intermediate_size": 8, "intermediate_size": 8,
               "vocab_size": 32},
              open(os.path.join(src, "config.json"), "w"))
    try:
        print("a fresh conversion")
        out = os.path.join(tmp, "fresh.waste")
        did = run(out, src)
        ck(did == MOE_LAYERS, f"converts every MoE layer {did}")
        ck(books(out) == list(range(len(MOE_LAYERS) * PER_LAYER)),
           "codebooks.bin is dense and in order")
        ok, why = check_consistency(out, MOE_LAYERS)
        ck(ok, f"each bank indexes its own records ({why})")

        print("resume of a run interrupted before it published anything")
        out = os.path.join(tmp, "interrupted.waste")
        os.makedirs(out)
        # Emulate a crash: layers 1,2,3 finished (bank + unmerged part),
        # nothing merged, no manifest — exactly what is on disk mid-run.
        for i, L in enumerate(MOE_LAYERS[:3]):
            fake_convert_layer((L, src, out, "", N_EXPERTS, STAGES, "cpu", 1,
                                i * PER_LAYER, False))
        ck(not os.path.exists(os.path.join(out, "manifest.json")),
           "no manifest exists, as after a real crash")
        did = run(out, src)
        ck(did == MOE_LAYERS[3:],
           f"re-converts only the unfinished layers {did}")
        ck(books(out) == list(range(len(MOE_LAYERS) * PER_LAYER)),
           "and the merged codebooks are still dense")
        ok, why = check_consistency(out, MOE_LAYERS)
        ck(ok, f"each bank indexes its own records ({why})")

        print("resume of a parallel run that finished layers out of order")
        out = os.path.join(tmp, "gap.waste")
        os.makedirs(out)
        for L in (1, 2, 3, 5):                 # layer 4 never finished
            fake_convert_layer((L, src, out, "", N_EXPERTS, STAGES, "cpu", 1,
                                (L - FIRST_DENSE) * PER_LAYER, False))
        did = run(out, src)
        ck(did == [4], f"re-converts only the hole {did}")
        ok, why = check_consistency(out, MOE_LAYERS)
        ck(ok, f"each bank still indexes its own records ({why})")

        print("resume of a completed run")
        out = os.path.join(tmp, "done.waste")
        run(out, src)
        before = books(out)
        did = run(out, src)
        ck(did == [], f"converts nothing {did}")
        ck(books(out) == before, "and leaves codebooks.bin byte-identical")

        print("resume with a different --layers, the case that used to renumber")
        out = os.path.join(tmp, "subset.waste")
        os.makedirs(out)
        for L in (1, 2, 3):
            fake_convert_layer((L, src, out, "", N_EXPERTS, STAGES, "cpu", 1,
                                (L - FIRST_DENSE) * PER_LAYER, False))
        did = run(out, src, layers=[3])
        ck(did == [], f"layer 3 is recognised as already done {did}")
        base3 = CONV.bank_codebook_base(os.path.join(out, "experts-L3.bin"))
        ck(base3 == 2 * PER_LAYER,
           f"and keeps the base its records name ({base3}), not 0")
        b = books(out)
        ck(b[base3:base3 + PER_LAYER] ==
           list(range(base3, base3 + PER_LAYER)),
           "its records sit at that base in codebooks.bin")

        print("--skip-trunk keeps the trunk and still publishes")
        out = os.path.join(tmp, "skiptrunk.waste")
        run(out, src, layers=[1])
        keep = manifest(out)["trunk"]
        trunk_before = open(os.path.join(out, "trunk.bin"), "rb").read()
        did = run(out, src, layers=[2], skip_trunk=True)
        ck(did == [2], f"converts the layer it was asked for {did}")
        m = manifest(out)
        ck("2" in m["layers"],
           "and the manifest it publishes lists that layer")
        ck(m["trunk"] == keep,
           "carrying the trunk index forward instead of emptying it")
        ck(open(os.path.join(out, "trunk.bin"), "rb").read() == trunk_before,
           "and leaving trunk.bin itself untouched")
        ok, why = check_consistency(out, [1, 2])
        ck(ok, f"both banks index their own records ({why})")
        ck(not os.path.exists(os.path.join(out, "trunk.bin.tmp")),
           "no staged trunk left behind")

        print("--skip-trunk with no published trunk to keep")
        out = os.path.join(tmp, "notrunk.waste")
        os.makedirs(out)
        did = run(out, src, layers=[1], skip_trunk=True, want_rc=1)
        ck(manifest(out) is None,
           "refuses rather than publish a manifest with an empty trunk")

        print("a bank whose codebook part was lost cannot be trusted")
        out = os.path.join(tmp, "lostpart.waste")
        os.makedirs(out)
        fake_convert_layer((1, src, out, "", N_EXPERTS, STAGES, "cpu", 1,
                            0, False))
        os.remove(os.path.join(out, "codebooks-L1.bin"))
        did = run(out, src, layers=[1])
        ck(did == [1], f"it is re-converted rather than reused {did}")
        ok, why = check_consistency(out, [1])
        ck(ok, f"and comes out consistent ({why})")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("RESUME FAILED" if fails else "RESUME OK")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
