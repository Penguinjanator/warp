#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""dsa_diff.py — did two runs of a DSA model attend over the same tokens?

The sparse-attention counterpart of tests/route_diff.py, and it exists for
the same reason. DSA ranks pools of `index_kpool` tokens and keeps the top
`index_topk / index_kpool` of them plus the tail; a different set of pools
is a different function, so the distance between the two runs' logits stops
measuring arithmetic the moment the selection moves. Reading a logit
difference as "the kernel is off by X" when the two implementations attended
to different tokens is how a divergence gets investigated for hours in the
wrong place.

Both sides already write the trace, in the same format and under the same
environment variable: `WASTE_DUMP_DSA` in `src/model.c` and again in
`tools/kimi_ref.py`, whose own comment says it is there "so the two
selections can be diffed directly rather than inferred from a logit
difference". This is the thing that does the diffing.

  tests/dsa_diff.py --a engine.dsa --b oracle.dsa

Exit 0 when every selection agrees, 1 when any differs or the traces cannot
be compared. Trace format, one line per (layer, token):

  L  pos  nvis  keep  id0,id1,..,  :  s0 s1 .. s(nvis-1)

Only the ids are compared. The scores are what a near-tie would be argued
from, and they are printed either side of the colon precisely so that a
difference in them can be told from a difference in the choice they drove.
"""

import argparse
import sys


def load(path):
    """{(layer, token): the pool ids kept}, ignoring the scores."""
    out = {}
    with open(path) as f:
        for line in f:
            head = line.split(":")[0].split()
            if len(head) < 5:
                continue
            out[(int(head[0]), int(head[1]))] = head[4]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="reference trace")
    ap.add_argument("--b", required=True, help="the run under test")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    try:
        a, b = load(args.a), load(args.b)
    except OSError as e:
        print(f"dsa_diff: {e}", file=sys.stderr)
        return 1
    if not a or not b:
        print("dsa_diff: a trace is empty, so which half moved is unanswered",
              file=sys.stderr)
        return 1

    # A trace covering different (layer, token) pairs is itself the finding:
    # the two did not attend over the same shape, and comparing the overlap
    # would report agreement on the part that happens to line up.
    if a.keys() != b.keys():
        print(f"dsa_diff: the traces cover different (layer, token) pairs "
              f"({len(a.keys() ^ b.keys())} on one side only) — the two did "
              f"not attend over the same shape", file=sys.stderr)
        return 1

    diff = sorted(k for k in a if a[k] != b[k])
    if diff:
        k = diff[0]
        print(f"dsa_diff: selections differ in {len(diff)} of {len(a)} "
              f"places, first at layer {k[0]} token {k[1]}: {a[k]} vs "
              f"{b[k]} — a different set of pools is a different function, "
              f"not an arithmetic error", file=sys.stderr)
        return 1
    if not args.quiet:
        print(f"DSA selections agree in all {len(a)} places, so the sparse "
              f"attention chose the same tokens and the difference is "
              f"upstream or downstream of it")
    return 0


if __name__ == "__main__":
    sys.exit(main())
