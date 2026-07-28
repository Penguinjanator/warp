#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""
requant_vision.py — rewrite the vision tower at 8 bits, in place.

The converter quantizes everything outside the expert banks the same way,
which for the text trunk is a measured 11.8% weight error the model was
trained to survive. The vision tower was not: it is 27 layers deep, its
position embedding is a learned spatial prior, and at 4 bits the damage
measures 17.9% on the packed QKV and 12.6% on the position embedding.
Implementing the ViT on those weights would produce garbage indis-
tinguishable from a bug in the implementation.

Rather than reconvert a 982 GB container to fix 223 MB, this appends
requantized payloads to trunk.bin and repoints the manifest at them. The
old bytes stay where they are — dead weight in the file, not in RAM, and
the file is append-only so an interrupted run leaves the container intact
until the manifest is replaced atomically at the end.

  uv run --with torch --with safetensors python tools/requant_vision.py \
      --src /path/to/hf-model --container model.waste
"""

import argparse
import io
import json
import os
import shutil
import sys

import torch
from safetensors import safe_open

FMT_F32, FMT_Q8G = 0, 2
GROUP = 128
PREFIXES = ("vision_tower.", "mm_projector.")


def quantize_q8g(W, group=GROUP):
    """int8 with one fp16 scale per group of `group` inputs — the format the
    engine already reads, and 0.64% error against the source."""
    rows, N = W.shape[0], W.shape[-1]
    W = W.reshape(rows, N).float()
    ng = (N + group - 1) // group
    pad = ng * group - N
    if pad:
        W = torch.cat([W, torch.zeros(rows, pad)], 1)
    g = W.view(rows, ng, group)
    amax = g.abs().amax(-1, keepdim=True)
    scale = (amax / 127.0).clamp(min=1e-12)
    q = torch.round(g / scale).clamp(-127, 127).to(torch.int8)
    return q.reshape(rows, ng * group), scale.reshape(rows, ng).half()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="the HF model the container came from")
    ap.add_argument("--container", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    man_path = os.path.join(args.container, "manifest.json")
    man = json.load(open(man_path))
    index = json.load(open(os.path.join(args.src, "model.safetensors.index.json")))
    wm = index["weight_map"]

    targets = [e for e in man["trunk"]
               if e["name"].startswith(PREFIXES) and e["fmt"] not in (FMT_F32, FMT_Q8G)]
    if not targets:
        print("nothing to do: vision weights are already f32 or int8")
        return 0
    print(f"{len(targets)} tensors to requantize "
          f"({sum(e['bytes'] for e in targets) / 2**20:.0f} MB at 4 bits)")
    if args.dry_run:
        for e in targets[:8]:
            print(f"  {e['name']}  {e['shape']}")
        return 0

    trunk = os.path.join(args.container, "trunk.bin")
    added = 0
    with open(trunk, "r+b") as tf:
        tf.seek(0, os.SEEK_END)
        for e in targets:
            name = e["name"]
            with safe_open(os.path.join(args.src, wm[name]), framework="pt") as sf:
                W = sf.get_tensor(name).float()
            flat = W.reshape(-1, W.shape[-1]) if W.dim() > 1 else W.reshape(1, -1)
            q, sc = quantize_q8g(flat)

            off = tf.tell()
            # torch's own byte view: numpy is not a dependency of this tool
            tf.write(q.contiguous().view(torch.uint8).untyped_storage().nbytes() and
                     bytes(q.contiguous().view(torch.int8).untyped_storage()))
            soff = tf.tell()
            tf.write(bytes(sc.contiguous().view(torch.int16).untyped_storage()))
            added += tf.tell() - off

            e["fmt"] = FMT_Q8G
            e["off"] = off
            e["scale_off"] = soff
            e["group"] = GROUP
            e["bytes"] = tf.tell() - off
        tf.flush()
        os.fsync(tf.fileno())

    tmp = man_path + ".tmp"
    with io.open(tmp, "w", encoding="utf-8") as f:
        json.dump(man, f, indent=1)
    os.replace(tmp, man_path)          # the container flips in one step
    print(f"appended {added / 2**20:.0f} MB, manifest updated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
