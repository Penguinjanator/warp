#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""split_banks.py — split a WASTE container's expert banks into N shard sets.

Round-robin placement matching src/model.c bank_fetch: expert e lives on
shard e % N at slot e // N. Byte-exact by construction — the same record
bytes land in the same logical order; only device placement changes.

Two modes:
  --mode split   write N shard directories, each holding shard files
  --mode verify  check an existing shard set against the source bank
                 (reads both, compares every record byte-for-byte)

Usage:
  python3 tools/split_banks.py <container.waste> --dirs /mnt/a,/mnt/b [--mode split]
  python3 tools/split_banks.py <container.waste> --dirs /mnt/a,/mnt/b --mode verify

The container's manifest.json lists per-layer bank files under "layers".
This tool reads the manifest, never the trunk. Shards are plain files named
after the bank's basename, placed one per directory. The engine opens them
via WASTE_BANK_SHARDS="/mnt/a,/mnt/b".
"""
import argparse
import json
import os
import sys

def bank_manifest(container):
    mf = os.path.join(container, "manifest.json")
    if not os.path.exists(mf):
        # some containers nest it
        for cand in ("waste.json", "index.json"):
            p = os.path.join(container, cand)
            if os.path.exists(p):
                mf = p
                break
        else:
            sys.exit(f"no manifest.json under {container}")
    with open(mf) as f:
        return json.load(f), mf

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("container")
    ap.add_argument("--dirs", required=True, help="comma-separated shard dirs")
    ap.add_argument("--mode", choices=("split", "verify"), default="split")
    a = ap.parse_args()
    dirs = [d for d in a.dirs.split(",") if d]
    if len(dirs) < 2:
        sys.exit("need at least 2 dirs")
    man, mf = bank_manifest(a.container)
    root = os.path.dirname(mf)
    layers = man.get("layers", {})
    n_done = 0
    for key, ent in sorted(layers.items(), key=lambda kv: int(kv[0])):
        fn = ent["file"]
        src = os.path.join(root, fn)
        n_exp = int(ent["experts"])
        if not os.path.exists(src):
            sys.exit(f"bank missing: {src}")
        rec = os.path.getsize(src) // n_exp
        assert os.path.getsize(src) == rec * n_exp, f"bank not divisible: {src}"
        if a.mode == "split":
            outs = []
            for d in dirs:
                os.makedirs(d, exist_ok=True)
                outs.append(open(os.path.join(d, os.path.basename(fn)), "wb"))
            with open(src, "rb") as f:
                for e in range(n_exp):
                    blob = f.read(rec)
                    assert len(blob) == rec
                    outs[e % len(dirs)].write(blob)
            for o in outs:
                o.close()
            print(f"layer {key}: {n_exp} experts x {rec}B -> {len(dirs)} shards")
        else:
            fhs = [open(os.path.join(d, os.path.basename(fn)), "rb") for d in dirs]
            with open(src, "rb") as f:
                for e in range(n_exp):
                    want = f.read(rec)
                    got = fhs[e % len(dirs)].read(rec)
                    if want != got:
                        sys.exit(f"MISMATCH layer {key} expert {e}")
            for fh in fhs:
                fh.close()
            print(f"layer {key}: VERIFY OK ({n_exp} records byte-identical)")
        n_done += 1
    print(f"{a.mode} complete: {n_done} layers, {len(dirs)} shard dirs")

if __name__ == "__main__":
    main()
