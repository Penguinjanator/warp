#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""
hf_tokenizer.py — HuggingFace tokenizer.json -> the tiktoken rank file the
container carries.

Both Kimi releases ship `tiktoken.model` and convert.py copies it. GLM-5.3
ships `tokenizer.json` instead: the same byte-level BPE, written in the
`tokenizers` library's own JSON, with the token bytes escaped through GPT-2's
bytes-to-unicode map and the ids where a tiktoken file puts ranks.

So this is a re-encoding, not a re-training — undo the escape, emit
`base64(bytes) rank`, and src/tokenizer.c reads it exactly as it reads
Kimi's. What is *not* a re-encoding, and what this checks rather than
assumes:

  - the pre-tokenization pattern. tokenizer.c implements the cl100k-style
    pattern directly (no regex engine), so a release whose Split pattern is
    a different one would be mis-split with no error anywhere. The pattern
    is compared against the one that file implements and the conversion
    stops if it differs.

  - merge order. tiktoken picks the adjacent pair whose *result* has the
    lowest id; a `tokenizers` BPE picks the pair earliest in the merge list.
    The two agree exactly when the merge list is ordered by result id, which
    is how byte-level BPE is trained and is checked here rather than
    believed. (What is not checked is a redundant rule: this vocabulary
    lists 321649 merges for 154564 distinct results, because a token can be
    reachable several ways — 'Ġ'+'th' as well as 'Ġt'+'h'. Rules that share
    a result are adjacent in a list ordered by result and produce the same
    token, so which one fires does not change the output. The agreement
    against the reference tokenizer is measured, not deduced: tools/
    tokdiff.py --hf.)

  - the Han branch. Kimi's pattern has `[\\p{Han}]+` and GLM's does not, so
    the container is told which — see `tokenizer_han_split` in the manifest
    and waste_tok_set_han_split.

  python3 tools/hf_tokenizer.py --src /path/to/glm --out model.waste
  python3 tools/hf_tokenizer.py --src /path/to/glm --out /tmp/probe --force
"""

import argparse
import base64
import io
import json
import os
import sys

# The pattern src/tokenizer.c implements, minus the Han branch that only the
# Kimi models carry. Compared literally: a release that reorders one
# alternative is a release this splits differently, and the difference does
# not show up as an error.
PAT_NO_HAN = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|"
              r"\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")
PAT_HAN = r"[\p{Han}]+|" + PAT_NO_HAN


def bytes_to_unicode():
    """GPT-2's map from byte to a printable codepoint, so a vocabulary can be
    JSON text. `tokenizers`' ByteLevel pre-tokenizer applies it; this undoes
    it."""
    bs = (list(range(ord("!"), ord("~") + 1)) +
          list(range(ord("¡"), ord("¬") + 1)) +
          list(range(ord("®"), ord("ÿ") + 1)))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def split_pattern(tok):
    """The Split regex out of the pre_tokenizer, whatever it is wrapped in."""
    pre = tok.get("pre_tokenizer") or {}
    stack = [pre]
    while stack:
        node = stack.pop()
        if not isinstance(node, dict):
            continue
        if node.get("type") == "Split":
            pat = node.get("pattern") or {}
            if "Regex" in pat:
                return pat["Regex"]
        stack.extend(node.get("pretokenizers") or [])
    return None


def convert(src, quiet=False):
    """Returns (rank-file text, han_split, specials list)."""
    path = os.path.join(src, "tokenizer.json")
    with io.open(path, encoding="utf-8") as f:
        tok = json.load(f)

    model = tok.get("model") or {}
    if model.get("type") != "BPE":
        raise SystemExit(f"{path}: model.type is {model.get('type')!r}, and "
                         "src/tokenizer.c is a BPE")
    if model.get("byte_fallback"):
        raise SystemExit(f"{path}: byte_fallback is set, which means tokens "
                         "spelled <0xNN> rather than byte-level escapes — "
                         "src/tokenizer.c reads the byte-level form")

    pat = split_pattern(tok)
    if pat == PAT_HAN:
        han = True
    elif pat == PAT_NO_HAN:
        han = False
    else:
        raise SystemExit(
            "this release pre-tokenizes with a pattern src/tokenizer.c does "
            "not implement, and the difference would be silent:\n"
            f"  release: {pat}\n  engine : {PAT_NO_HAN}\n"
            "(optionally preceded by [\\p{Han}]+). See tools/hf_tokenizer.py.")

    dec = bytes_to_unicode()
    vocab = model["vocab"]

    def raw(text):
        try:
            return bytes(dec[ch] for ch in text)
        except KeyError as e:
            raise SystemExit(f"token {text!r} has {e} outside the byte-level "
                             f"escape map; this is not a byte-level BPE")

    # The merge list has to be ordered by the id of what it produces, or
    # merge-by-rank and merge-by-list-position are two different encoders
    # sharing one vocabulary — and the difference shows up as a shifted
    # token stream, never as an error.
    merges = model.get("merges") or []
    prev, descents, missing = -1, 0, 0
    for m in merges:
        a, b = m if isinstance(m, (list, tuple)) else m.split(" ", 1)
        j = vocab.get(a + b)
        if j is None:
            missing += 1
            continue
        if j < prev:
            descents += 1
        prev = j
    if descents or missing:
        raise SystemExit(
            f"this merge list is not ordered by the id of what it produces "
            f"({descents} descents, {missing} results outside the "
            f"vocabulary), so rank-ordered BPE would encode it differently "
            f"from the release. Refusing to write a tokenizer that silently "
            f"disagrees.")

    lines = []
    for text, rank in sorted(vocab.items(), key=lambda kv: kv[1]):
        lines.append(base64.b64encode(raw(text)).decode("ascii") + " " +
                     str(rank))

    specials = sorted(
        ({"id": int(a["id"]), "text": a["content"]}
         for a in (tok.get("added_tokens") or [])),
        key=lambda e: e["id"])
    if not quiet:
        print(f"tokenizer: {len(lines)} merges, {len(specials)} specials, "
              f"pattern {'with' if han else 'without'} a Han branch")
    return "\n".join(lines) + "\n", han, specials


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="HF checkpoint directory")
    ap.add_argument("--out", required=True, help="container directory")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing tokenizer.model")
    args = ap.parse_args()

    dst = os.path.join(args.out, "tokenizer.model")
    if os.path.exists(dst) and not args.force:
        print(f"{dst} exists; --force to replace it", file=sys.stderr)
        return 1
    text, han, specials = convert(args.src)
    os.makedirs(args.out, exist_ok=True)
    with io.open(dst, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    if specials:
        with io.open(os.path.join(args.out, "specials.json"), "w",
                     encoding="utf-8", newline="\n") as f:
            json.dump(specials, f, indent=1)
    print(f"wrote {dst}"
          + ("" if han else "  (tokenizer_han_split must be false)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
