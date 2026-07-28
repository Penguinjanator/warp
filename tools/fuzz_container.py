#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""
fuzz_container.py — feed the engine broken containers and insist it says no.

A container is an untrusted input: docs/FORMAT.md says so, and the manifest
is JSON parsed by hand while the banks are binary records full of offsets
the engine follows. The failure that matters is not a wrong answer, it is
a read outside a buffer, so this is worth most when run against an
instrumented build — `make fuzz-asan`.

Structure-aware rather than random: it starts from a valid container
(tools/make_test_container.py) and breaks one thing at a time — a JSON
field retyped or moved out of range, an offset pointed past the end, a
file truncated mid-record, bytes flipped in a header. Purely random bytes
would be rejected by the JSON parser before reaching anything interesting.

A clean rejection is a pass. A nonzero exit is a pass. Being killed by a
signal is a failure, and so is hanging.

  python3 tools/fuzz_container.py --runs 200
  make fuzz-asan FUZZ_RUNS=1000
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TIMEOUT = 60


def build_seed(path):
    subprocess.run([sys.executable, os.path.join(HERE, "tools",
                                                 "make_test_container.py"), path],
                   check=True, stdout=subprocess.DEVNULL)


# ---- mutations ------------------------------------------------------------
# Each takes a container directory and breaks it in one specific way.

def m_json_retype(d, rng):
    """A field becomes the wrong type — the parser must not assume."""
    p = os.path.join(d, "manifest.json")
    man = json.load(open(p))
    keys = [k for k in man if k != "format_version"]
    k = rng.choice(keys)
    man[k] = rng.choice([None, [], {}, "x", -1, 1e30])
    json.dump(man, open(p, "w"))
    return f"manifest[{k}] retyped"


def m_json_shape(d, rng):
    """A tensor's declared shape or offset stops matching the file."""
    p = os.path.join(d, "manifest.json")
    man = json.load(open(p))
    if not man.get("trunk"):
        return None
    e = rng.choice(man["trunk"])
    field = rng.choice(["off", "bytes", "shape", "group", "scale_off"])
    if field == "shape":
        e["shape"] = rng.choice([[0], [-1, 8], [1 << 20, 1 << 20], []])
    elif field in e:
        e[field] = rng.choice([-1, 0, 1 << 40, e.get(field, 0) + 7])
    else:
        e[field] = 1 << 40
    json.dump(man, open(p, "w"))
    return f"trunk entry {field} corrupted"


def m_json_layers(d, rng):
    """The expert bank index lies about record count or size."""
    p = os.path.join(d, "manifest.json")
    man = json.load(open(p))
    if not man.get("layers"):
        return None
    L = rng.choice(list(man["layers"]))
    f = rng.choice(["experts", "bytes", "codebook_base", "file"])
    man["layers"][L][f] = rng.choice([-1, 0, 1 << 30, "../../etc/passwd"])
    json.dump(man, open(p, "w"))
    return f"layers[{L}].{f} corrupted"


def m_truncate(d, rng):
    files = [f for f in os.listdir(d) if f.endswith(".bin")]
    if not files:
        return None
    f = rng.choice(files)
    p = os.path.join(d, f)
    n = os.path.getsize(p)
    keep = rng.randrange(0, n) if n else 0
    with open(p, "r+b") as fh:
        fh.truncate(keep)
    return f"{f} truncated to {keep}/{n}"


def m_flip(d, rng):
    files = [f for f in os.listdir(d) if f.endswith(".bin")]
    if not files:
        return None
    f = rng.choice(files)
    p = os.path.join(d, f)
    n = os.path.getsize(p)
    if not n:
        return None
    # bias toward the first pages, where the headers and offsets live
    where = rng.randrange(0, min(n, 4096)) if rng.random() < 0.7 else rng.randrange(0, n)
    with open(p, "r+b") as fh:
        fh.seek(where)
        b = fh.read(1)
        fh.seek(where)
        fh.write(bytes([b[0] ^ (1 << rng.randrange(8))]))
    return f"{f} bit flipped at {where}"


def m_json_garbage(d, rng):
    p = os.path.join(d, "manifest.json")
    s = open(p).read()
    cut = rng.randrange(0, len(s))
    open(p, "w").write(s[:cut] + rng.choice(["", "}", "\0", "[[[", '"']))
    return "manifest truncated"


def m_remove(d, rng):
    files = os.listdir(d)
    f = rng.choice(files)
    os.remove(os.path.join(d, f))
    return f"{f} removed"


MUTATIONS = [m_json_retype, m_json_shape, m_json_layers, m_truncate,
             m_flip, m_json_garbage, m_remove]


def run_engine(container):
    """`waste info` parses the manifest and opens every bank — the whole
    untrusted surface, without needing a forward pass."""
    return subprocess.run([os.path.join(HERE, "waste"), "info", container],
                          stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                          timeout=TIMEOUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    tmp = tempfile.mkdtemp(prefix="wastefuzz")
    seed_dir = os.path.join(tmp, "seed")
    try:
        build_seed(seed_dir)
        if run_engine(seed_dir).returncode != 0:
            print("FAIL: the unmutated seed container does not load")
            return 1

        work = os.path.join(tmp, "case")
        crashes, hangs, rejected, accepted = [], [], 0, 0
        for i in range(args.runs):
            shutil.rmtree(work, ignore_errors=True)
            shutil.copytree(seed_dir, work)
            what = MUTATIONS[i % len(MUTATIONS)](work, rng)
            if what is None:
                continue
            try:
                r = run_engine(work)
            except subprocess.TimeoutExpired:
                hangs.append(what)
                continue
            if r.returncode < 0:                    # killed by a signal
                # Keep the whole report and the container that produced it:
                # a sanitizer names the bug in its first lines, and the tail
                # is a legend nobody reads.
                err = r.stderr.decode(errors="replace")
                keep = os.path.join(os.getcwd(),
                                    "fuzz-case-%d" % len(crashes))
                shutil.rmtree(keep, ignore_errors=True)
                shutil.copytree(work, keep)
                crashes.append((what, -r.returncode, err[:1200], keep))
            elif r.returncode:
                rejected += 1
            else:
                accepted += 1                       # survived: not a bug per se

        print(f"{args.runs} cases: {rejected} rejected, {accepted} still "
              f"loaded, {len(crashes)} crashed, {len(hangs)} hung")
        for what, sig, err, keep in crashes[:10]:
            print(f"\n  CRASH (signal {sig}) after: {what}")
            print(f"    reproduce: ./waste info {keep}")
            for line in err.strip().split("\n")[:8]:
                print(f"    {line}")
        for what in hangs[:5]:
            print(f"  HANG after: {what}")
        if crashes or hangs:
            print("\nFUZZ FAIL")
            return 1
        print("FUZZ OK")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
