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

Ties are the reason the scores share the line. A selection is a ranking, and
a ranking of equal numbers has no answer -- whoever implements it picks one.
Measured on the synthetic GLM container the engine and the PyTorch oracle
disagree at exactly one place, layer 2 token 15, where the visible pools
score 0, 0, 0 and 0.00164 with keep=2: pool 3 wins outright and the second
slot is an *exact* three-way tie. The engine takes pool 1 and torch.topk,
called with sorted=False and so free to answer in any order, takes pool 2 on
Linux and pool 1 on macOS. That is worth rel L2 0.0078 in the logits, and it
is not a defect in either.

So this reports the same three answers tests/route_diff.py does:

  identical  every selection agrees.
  tie        the first disagreement is between pools the reference scored
             within --eps of each other, relative.
  diverged   the first disagreement is between pools the reference ranked
             with room to spare, which is a defect in one of the two.

Only the first disagreement is judged: once one layer attends to different
tokens its output differs, and every later selection is computed on a hidden
state that has already moved.

  tests/dsa_diff.py --a engine.dsa --b oracle.dsa

Exit 0 identical, 2 tie, 1 diverged or not comparable. Trace format, one
line per (layer, token):

  L  pos  nvis  keep  id0,id1,..,  :  s0 s1 .. s(nvis-1)
"""

import argparse
import sys


def load(path):
    """{(layer, token): (the pool ids kept, the score of every visible pool)}"""
    out = {}
    with open(path) as f:
        for line in f:
            head, _, tail = line.partition(":")
            head = head.split()
            if len(head) < 5:
                continue
            ids = [int(i) for i in head[4].split(",") if i != ""]
            out[(int(head[0]), int(head[1]))] = (ids,
                                                 [float(v) for v in tail.split()])
    return out


def margin(scores, only_a, only_b):
    """How far apart the reference held the two pools that swapped, relative
    to their size. None when there is no single swap to weigh, or a pool id
    the score list does not cover."""
    if len(only_a) != 1 or len(only_b) != 1:
        return None
    if max(only_a[0], only_b[0]) >= len(scores):
        return None
    x, y = scores[only_a[0]], scores[only_b[0]]
    scale = max(abs(x), abs(y))
    return abs(x - y) / scale if scale else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="reference trace")
    ap.add_argument("--b", required=True, help="the run under test")
    ap.add_argument("--eps", type=float, default=1e-5,
                    help="relative margin at or under which two pools count "
                         "as tied (default 1e-5)")
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

    diff = sorted(k for k in a if set(a[k][0]) != set(b[k][0]))
    if not diff:
        if not args.quiet:
            print(f"DSA selections agree in all {len(a)} places, so the "
                  f"sparse attention chose the same tokens and the "
                  f"difference is upstream or downstream of it")
        return 0

    k = diff[0]
    only_a = sorted(set(a[k][0]) - set(b[k][0]))
    only_b = sorted(set(b[k][0]) - set(a[k][0]))
    m = margin(a[k][1], only_a, only_b)
    where = (f"selections differ in {len(diff)} of {len(a)} places, first at "
             f"layer {k[0]} token {k[1]}: pools {only_a} -> {only_b}")
    if m is None:
        print(f"dsa_diff: {where}; more than one pool moved at once, so "
              f"there is no single margin to weigh it by", file=sys.stderr)
        return 1
    if m > args.eps:
        print(f"dsa_diff: {where}, at relative margin {m:.3e} — the "
              f"reference ranked these pools apart by more than "
              f"{args.eps:g}, so this is a disagreement and not a tie",
              file=sys.stderr)
        return 1
    if not args.quiet:
        print(f"{where}, at relative margin {m:.3e} — a tie in the pool "
              f"ranking, which nothing can resolve; the other "
              f"{len(diff) - 1} are downstream of it")
    return 2


if __name__ == "__main__":
    sys.exit(main())
