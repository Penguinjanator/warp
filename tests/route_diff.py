#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""route_diff.py — did two runs of the same model route to the same experts,
and if not, was the first disagreement a tie?

On a MoE with a top-K router, comparing two arithmetically-equivalent paths
by the distance between their final logits measures the wrong thing. The
router ranks E scores and keeps K; two paths that agree on every score to
float precision can still order two *adjacent* scores differently, and the
moment they do they are running different experts and computing genuinely
different functions. The logit distance that follows is the model's
sensitivity to that swap, not the size of the arithmetic difference — so a
logit threshold either fails on a tie or is loose enough to miss a real
defect. It cannot be both.

This asks the question the threshold was standing in for. Given the
reference run's route trace, the other path's trace, and the reference's own
score vectors:

  identical  every decision agrees.
  tie        the *first* disagreement is between two experts the reference
             itself could not separate, and the rest are downstream of it.
  diverged   the first disagreement is a decision the reference made with
             room to spare, which is a defect in one of the two paths.

Only the first disagreement is judged, and that is the point rather than a
concession. Once one layer routes differently its output differs materially,
so every later score in that token and every score in every later token is
computed on a hidden state that already moved; their margins are
consequences and say nothing about arithmetic. Measured on K3 (92 layers,
384 experts, top-16, a 16-token prefill: 1472 decisions) the first
disagreement between the default path and either the CPU baseline or
chunked prefill sits at a relative margin of 7.3e-07 — *the minimum over
all 1472 decisions* — while the 47 that follow it average 5e-03, three
orders of magnitude wider and squarely typical. A near-tie flipping and a
kernel being wrong do not look alike.

--eps is measured, not chosen. Over that same run the tightest boundary gap
the router did *not* trip on is 4.4e-05, at the 1st percentile, and the
median is 7.4e-03. The default 1e-5 sits in the empty decade between the
tie that flipped and the tightest call that held: 14x above the former, 4x
below the latter.

  tests/route_diff.py --ref seq.route --other cpu.route --scores seq.scores

Trace formats, both written by `src/model.c` (WASTE_DUMP_ROUTE,
WASTE_DUMP_SCORES), one line per (token, layer):

  route    pos  L  id0..id(K-1)  [w0..w(K-1)]
  scores   pos  L  v0..v(E-1)          the ranked quantity, score+bias
"""

import argparse
import sys


def _leading_ints(fields):
    """The ids, which are the run of integers before the weights start."""
    out = []
    for f in fields:
        try:
            out.append(int(f))
        except ValueError:
            break
    return out


def load_routes(path):
    routes = {}
    with open(path) as f:
        for line in f:
            g = line.split()
            if len(g) < 3:
                continue
            routes[(int(g[0]), int(g[1]))] = _leading_ints(g[2:])
    return routes


def load_scores(path):
    scores = {}
    with open(path) as f:
        for line in f:
            g = line.split()
            if len(g) < 3:
                continue
            scores[(int(g[0]), int(g[1]))] = [float(x) for x in g[2:]]
    return scores


def margin(scores, key, ref_only, other_only):
    """How far apart the reference held the two experts that swapped,
    relative to their size — a router score has no natural unit, and the
    absolute gap between two scores of 0.11 says nothing on its own.

    None when the trace cannot answer: no score line for this decision, or
    more than one expert entering and leaving at once, which is not a
    boundary swap and has no single margin to report."""
    v = scores.get(key)
    if v is None or len(ref_only) != 1 or len(other_only) != 1:
        return None
    a, b = v[ref_only[0]], v[other_only[0]]
    scale = max(abs(a), abs(b))
    return abs(a - b) / scale if scale else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="reference route trace")
    ap.add_argument("--other", required=True, help="the path under test")
    ap.add_argument("--scores", default="",
                    help="the reference's score trace; without it a "
                         "disagreement cannot be told from a tie and is "
                         "reported as a divergence")
    ap.add_argument("--eps", type=float, default=1e-5,
                    help="relative margin at or under which two experts "
                         "count as tied (default 1e-5)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    ref, other = load_routes(args.ref), load_routes(args.other)
    if not ref or not other:
        print("route_diff: a trace is empty", file=sys.stderr)
        return 1
    # A trace missing decisions the other made is itself the finding: the
    # two paths did not run the same shape, and comparing the overlap would
    # report agreement on the part that happens to line up.
    if ref.keys() != other.keys():
        only = len(set(ref) ^ set(other))
        print(f"route_diff: the traces cover different decisions "
              f"({only} on one side only)", file=sys.stderr)
        return 1

    diffs = [k for k in sorted(ref) if set(ref[k]) != set(other[k])]
    total = len(ref)
    if not diffs:
        if not args.quiet:
            print(f"routes identical over {total} decisions")
        return 0

    first = diffs[0]
    scores = load_scores(args.scores) if args.scores else {}
    ref_only = sorted(set(ref[first]) - set(other[first]))
    other_only = sorted(set(other[first]) - set(ref[first]))
    m = margin(scores, first, ref_only, other_only)

    where = (f"{len(diffs)} of {total} decisions differ, first at "
             f"token {first[0]} layer {first[1]}: "
             f"{ref_only} -> {other_only}")
    if m is None:
        print(f"route_diff: {where}; no score trace to weigh it against",
              file=sys.stderr)
        return 1
    if m > args.eps:
        print(f"route_diff: {where}, at relative margin {m:.3e} — the "
              f"reference separated these two experts by more than "
              f"{args.eps:g}, so this is a disagreement and not a tie",
              file=sys.stderr)
        return 1
    if not args.quiet:
        print(f"{where}, at relative margin {m:.3e} — a tie the reference "
              f"itself could not resolve; the other {len(diffs) - 1} are "
              f"downstream of it")
    return 2


if __name__ == "__main__":
    sys.exit(main())
