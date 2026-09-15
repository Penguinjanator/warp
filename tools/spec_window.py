#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""
spec_window.py — what a speculative batch of K tokens costs THIS engine.

Speculative decoding verifies K draft tokens in one backbone pass. On a GPU
that is nearly free: the pass is compute-bound, the K tokens ride along in
the same matmuls, and the only question is how many of them get accepted.

Here the pass IS the expert reads. A decode token on K3 pulls 17 GB off
disk; verifying five of them pulls whatever the union of their five routes
comes to, and a top-k router gives consecutive tokens only a fraction of
their experts in common. So the question that decides whether speculation
helps at all is not the acceptance rate on its own — it is the acceptance
rate against **how much the union grows**.

That is what this measures, from a real WASTE_DUMP_ROUTE trace:

    WASTE_DUMP_ROUTE=/tmp/m.route ./test_forward MODEL IDS out.bin 48
    python3 tools/spec_window.py /tmp/m.route --decode-from 28

`--decode-from` is the first generated position; the prompt's routes are
prefill and are not the regime speculation runs in.

The number to read is the last column: how many of the K drafts must be
accepted for the batch to read no more PER ACCEPTED TOKEN than plain
decoding does. Below it, speculation costs bytes rather than saving them.

See docs/GATES.md gate 9 for the three models this was run on and what
their agreement means.
"""

import argparse
import sys
from collections import defaultdict


def load(path):
    """pos -> {(layer, expert)}, from a WASTE_DUMP_ROUTE trace.

    The line is `pos layer idx[K] w[K] look[K]`, so K is a third of what is
    left after the two leading fields — read rather than assumed, because
    the same script has to serve a top-6 model and a top-16 one.
    """
    rows = defaultdict(set)
    topk = 0
    for line in open(path):
        f = line.split()
        if len(f) < 5:
            continue
        pos, layer = int(f[0]), int(f[1])
        topk = (len(f) - 2) // 3
        for j in range(topk):
            rows[pos].add((layer, int(f[2 + j])))
    return rows, topk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--decode-from", type=int, default=0, metavar="POS",
                    help="first generated position; earlier ones are prefill")
    ap.add_argument("--max-k", type=int, default=8)
    a = ap.parse_args()

    rows, topk = load(a.trace)
    if not rows:
        print(f"{a.trace}: no routes", file=sys.stderr)
        return 1
    pos = sorted(p for p in rows if p >= a.decode_from)
    if len(pos) < a.max_k:
        print(f"only {len(pos)} decode positions; generate more",
              file=sys.stderr)
        return 1
    one = sum(len(rows[p]) for p in pos) / len(pos)
    print(f"{len(rows)} positions, {len(pos)} decode, top-{topk}")
    print(f"one token touches {one:.1f} distinct expert records\n")
    print(f"{'K':>2} {'distinct':>9} {'vs K x 1':>9} {'best case':>10} "
          f"{'break-even accept':>18}")
    for K in range(1, a.max_k + 1):
        tot = 0
        for i in range(len(pos) - K + 1):
            s = set()
            for p in pos[i:i + K]:
                s |= rows[p]
            tot += len(s)
        d = tot / (len(pos) - K + 1)
        print(f"{K:>2} {d:>9.1f} {d / (K * one):>8.1%} "
              f"{d / K / one:>9.2f}x {d / one:>17.2f}")
    print("\n'best case' is bytes per token with every draft accepted.")
    print("'break-even accept' is how many of the K must survive for the")
    print("batch to read no more per accepted token than plain decoding.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
