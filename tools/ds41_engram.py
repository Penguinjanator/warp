#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""
ds41_engram.py — everything DeepSeek-V4.1's Engram needs that is not weights.

Engram adds an n-gram lookup into the residual stream at two layers. Which
row it looks up is not in the checkpoint: it is derived, at model build
time, from the tokenizer and a fixed RNG, and the release's `engram.py` is
the only statement of how. Three derived things, all of which the engine
needs and none of which it can recompute:

  the compressed token map   129,280 ids collapsed onto 99,092 by a
                             normalizer chain, so " The", "the" and "THE"
                             hash alike. Needs the `tokenizers` normalizers,
                             which is a Python dependency and not one this
                             engine is going to grow.
  the bucket primes          24 per layer, drawn in order from
                             engram_vocab_size - 1 and never reused, so
                             every (n-gram size, head) pair owns a disjoint
                             range of the table.
  the hash multipliers       one per (layer, lookback), from numpy's PCG64
                             seeded 10007 * layer_id.

Every one of them is derived from `engram_compressed_vocab_size`, so a map
that comes out a different size rehashes the whole 197 B table into noise.
The release asserts that size; so does this, and it refuses rather than
writes.

Written at conversion into `engram.json` and `engram-tokmap.bin`. Both are
small — the map is 517 KB of int32 — and neither changes with the weights.

  uv run --with torch --with tokenizers --with numpy python \\
      tools/ds41_engram.py --src /path/to/DeepSeek-V4.1-Flash --out /tmp/eg
"""

import argparse
import array
import json
import os
import sys


def is_prime(n):
    """Trial division. The primes wanted here sit just above 16 M, so this
    is 4,000 divisions each and there are 48 of them — sympy for that would
    be a dependency bought with nothing."""
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    f = 3
    while f * f <= n:
        if n % f == 0:
            return False
        f += 2
    return True


def next_prime(start, seen):
    """The smallest prime above `start` that has not been handed out yet."""
    c = start + 1
    while not is_prime(c) or c in seen:
        c += 1
    return c


def bucket_layout(layer_ids, max_ngram_size, n_heads, engram_vocab_size):
    """Per layer, per n-gram size, per head: the bucket modulus and the
    offset of its range in the table.

    Drawn in one sequence across every layer and never reused, which is what
    makes the ranges disjoint — so the order of `layer_ids` is part of the
    answer, not a presentation choice.
    """
    primes, offsets, seen = [], [], set()
    for _ in layer_ids:
        per_layer = []
        for _ in range(max_ngram_size - 1):
            sizes, cur = [], engram_vocab_size - 1
            for _ in range(n_heads):
                cur = next_prime(cur, seen)
                seen.add(cur)
                sizes.append(cur)
            per_layer.append(sizes)
        primes.append(per_layer)
        flat, run = [], 0
        for per_ngram in per_layer:
            for p in per_ngram:
                flat.append(run)
                run += p
        offsets.append(flat)
    return primes, offsets


def multipliers(layer_ids, max_ngram_size, compressed_vocab_size):
    """One odd multiplier per (layer, lookback), from numpy's PCG64.

    numpy rather than a hand-rolled PRNG because the values have to be the
    ones training used, and "a PCG64 stream" is a specification only if it
    is the same implementation. Bounded so `token_id * multiplier` cannot
    overflow int64, which is what the release's bound means.
    """
    import numpy as np
    max_long = np.iinfo(np.int64).max
    bound = max(1, (max_long // compressed_vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        gen = np.random.default_rng(10007 * layer_id)
        v = gen.integers(low=0, high=bound, size=(max_ngram_size,),
                         dtype=np.int64)
        rows.append([int(x) * 2 + 1 for x in v])
    return rows


def compressed_token_map(src):
    """Every token id onto the smaller id space n-grams are hashed over.

    The normalizer chain is the release's, character for character. The
    sentinel matters: a token that is exactly one space would otherwise
    collapse to the empty string under Strip() and merge with unrelated
    tokens.
    """
    from tokenizers import Tokenizer, Regex, normalizers
    tok = Tokenizer.from_file(os.path.join(src, "tokenizer.json"))
    sentinel = ""
    norm = normalizers.Sequence([
        normalizers.NFKC(),
        normalizers.NFD(),
        normalizers.StripAccents(),
        normalizers.Lowercase(),
        normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
        normalizers.Replace(Regex(r"^ $"), sentinel),
        normalizers.Strip(),
        normalizers.Replace(sentinel, " "),
    ])
    n = tok.get_vocab_size(with_added_tokens=True)
    key_to_new, lookup = {}, [0] * n
    for tid in range(n):
        text = tok.decode([tid], skip_special_tokens=False)
        if "�" in text:
            # A partial UTF-8 byte token: nothing to normalize, so key it by
            # its raw form rather than by a replacement character every such
            # token would share.
            key = tok.id_to_token(tid)
        else:
            normalized = norm.normalize_str(text)
            key = normalized if normalized else text
        new = key_to_new.get(key)
        if new is None:
            new = len(key_to_new)
            key_to_new[key] = new
        lookup[tid] = new
    return lookup, len(key_to_new)


def build(src, cfg, out_dir):
    """Writes engram.json and engram-tokmap.bin. Returns the json, or None
    when this config has no Engram."""
    layer_ids = list(cfg.get("engram_layer_ids") or [])
    if not layer_ids:
        return None
    max_ngram = cfg["engram_max_ngram_size"]
    n_heads = cfg["engram_n_heads"]
    stated = cfg["engram_compressed_vocab_size"]

    lookup, size = compressed_token_map(src)
    if size != stated:
        raise SystemExit(
            f"the compressed token map came out {size} ids and this release "
            f"states {stated}. Every hash multiplier is derived from that "
            f"number, so a container written now would address the table "
            f"with the wrong hashes — 197 B parameters of noise, and nothing "
            f"downstream would report it. Check the `tokenizers` version "
            f"against the release's normalizer chain.")

    primes, offsets = bucket_layout(layer_ids, max_ngram, n_heads,
                                    cfg["engram_vocab_size"])
    mults = multipliers(layer_ids, max_ngram, size)
    # The table rows a layer holds must cover its own buckets exactly; the
    # release states both and they are two statements of one number.
    for i, L in enumerate(layer_ids):
        want = sum(p for per_ngram in primes[i] for p in per_ngram)
        have = (cfg.get("engram_num_embeddings") or [0] * len(layer_ids))[i]
        if have and have != want:
            raise SystemExit(
                f"engram layer {L}: the primes sum to {want} rows and the "
                f"config states {have}. One of the two is not this model.")

    pad_raw = cfg.get("engram_pad_token_id", 2)
    doc = {
        "compressed_vocab_size": size,
        "pad_id": lookup[pad_raw],
        "max_ngram_size": max_ngram,
        "n_heads": n_heads,
        "head_dim": cfg["engram_head_dim"],
        "layers": [{"layer": L, "multipliers": mults[i],
                    "primes": primes[i], "offsets": offsets[i]}
                   for i, L in enumerate(layer_ids)],
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "engram.json"), "w", newline="\n") as f:
        json.dump(doc, f, indent=1)
    # int32 and not JSON: 129,280 numbers is 517 KB either way as bytes and
    # a megabyte as text, and the engine reads it once at open.
    array.array("i", lookup).tofile(
        open(os.path.join(out_dir, "engram-tokmap.bin"), "wb"))
    return doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="the release directory")
    ap.add_argument("--out", required=True, help="container directory")
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(a.src, "config.json")))
    cfg = cfg.get("text_config", cfg)
    doc = build(a.src, cfg, a.out)
    if doc is None:
        print("this config has no Engram")
        return 0
    print(f"engram: {len(doc['layers'])} layer(s), "
          f"{doc['compressed_vocab_size']} compressed ids, "
          f"pad -> {doc['pad_id']}")
    for L in doc["layers"]:
        rows = sum(p for per in L["primes"] for p in per)
        print(f"  layer {L['layer']}: {rows} rows over "
              f"{len(L['offsets'])} buckets")
    return 0


if __name__ == "__main__":
    sys.exit(main())
