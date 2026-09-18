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
    and waste_tok_set_han_split. DeepSeek-V4.1 needs more than a flag: its
    pre_tokenizer is three isolating Splits rather than one pattern, with
    punctuation and symbols as classes of their own and no contraction
    branch, so it gets a mode of its own and `tokenizer_pattern` says so.

  python3 tools/hf_tokenizer.py --src /path/to/glm --out model.waste
  python3 tools/hf_tokenizer.py --src /path/to/glm --out /tmp/probe --force
"""

import argparse
import base64
import io
import json
import os
import sys

# The patterns src/tokenizer.c implements, spelled out rather than parsed.
# Three things vary across this family and nothing else does, so the whole
# set is enumerated and compared literally: a release that reorders one
# alternative is a release this splits differently, and the difference does
# not show up as an error.
#
#   han     — `[\p{Han}]+` as its own leading branch (both Kimi releases)
#             or Han left to the letter branch (GLM, Qwen).
#   marks   — `[\p{L}\p{M}]` (Qwen) or `\p{L}` (Kimi, GLM). Descriptive
#             only: tokenizer.c's letter class is the union either way, so
#             the two spellings are the same engine behaviour.
#   digits  — `\p{N}{1,3}` (Kimi, GLM) or `\p{N}` (Qwen). NOT cosmetic;
#             it is carried to the engine as `tokenizer_digit_run`.
def _pattern(han, marks, digit_run):
    letter = r"[\p{L}\p{M}]" if marks else r"\p{L}"
    other = r"[^\s\p{L}\p{M}\p{N}]" if marks else r"[^\s\p{L}\p{N}]"
    return ((r"[\p{Han}]+|" if han else "") +
            r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?" + letter +
            r"+|\p{N}" + (r"{1,3}" if digit_run == 3 else "") +
            r"| ?" + other + r"+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")


# pattern -> (han_split, digit_run)
KNOWN_PATTERNS = {_pattern(h, m, d): (h, d)
                  for h in (0, 1) for m in (0, 1) for d in (1, 3)}
PAT_NO_HAN = _pattern(0, 0, 3)
PAT_HAN = _pattern(1, 0, 3)

# DeepSeek-V4.1 splits with three isolating Splits in sequence rather than
# one pattern, and src/tokenizer.c implements the composition as a mode of
# its own (WASTE_TOKPAT_DEEPSEEK). Compared literally, all three, in order:
# a release that reorders them or widens one class splits differently, and
# the difference is a shifted token stream rather than an error.
PAT_DS41 = [
    '\\p{N}{1,3}',
    '[一-龥\u3040-ゟ゠-ヿ]+',
    '[!"#$%&\'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+',
]

# Values of `tokenizer_pattern` in the manifest; mirrors WASTE_TOKPAT_* in
# src/tokenizer.h.
TOKPAT_CL100K, TOKPAT_DEEPSEEK = 0, 1


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


def split_patterns(tok):
    """Every Split regex in the pre_tokenizer, in the order they apply.

    Order matters and a set would lose it: DeepSeek's three Splits isolate
    numbers, then CJK, then everything else, and running them in any other
    order is a different tokenizer.
    """
    out = []

    def walk(node):
        if not isinstance(node, dict):
            return
        if node.get("type") == "Split":
            pat = node.get("pattern") or {}
            if "Regex" in pat:
                out.append(pat["Regex"])
        for child in node.get("pretokenizers") or []:
            walk(child)

    walk(tok.get("pre_tokenizer") or {})
    return out


def convert(src, quiet=False):
    """Returns (rank text, han split, specials, digit run, pattern mode)."""
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

    pats = split_patterns(tok)
    pat = pats[0] if pats else None
    if pats == PAT_DS41:
        han, digit_run, pattern = True, 3, TOKPAT_DEEPSEEK
    elif len(pats) == 1 and pat in KNOWN_PATTERNS:
        han, digit_run = KNOWN_PATTERNS[pat]
        han, pattern = bool(han), TOKPAT_CL100K
    else:
        raise SystemExit(
            "this release pre-tokenizes with a pattern src/tokenizer.c does "
            "not implement, and the difference would be silent:\n"
            + "".join(f"  release: {p}\n" for p in pats or [None]) +
            + "".join(f"  known  : {k}\n" for k in KNOWN_PATTERNS) +
            "or DeepSeek-V4.1's three "
            "Splits. See tools/hf_tokenizer.py.")

    dec = bytes_to_unicode()
    vocab = model["vocab"]

    # Kimi and GLM append their control tokens after the BPE table; DeepSeek
    # puts three of them (BOS, EOS, PAD) at ids 0-2 *inside* it. Those rows
    # are not byte-level escapes and have no byte sequence that reaches them,
    # so they belong in specials.json and not in the rank file — but only
    # when the release itself declares them as added tokens. Anything else
    # outside the escape map is still a refusal.
    declared = {a["content"] for a in (tok.get("added_tokens") or [])}

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

    lines, inline_specials = [], 0
    for text, rank in sorted(vocab.items(), key=lambda kv: kv[1]):
        if text in declared:
            inline_specials += 1
            continue
        lines.append(base64.b64encode(raw(text)).decode("ascii") + " " +
                     str(rank))

    specials = sorted(
        ({"id": int(a["id"]), "text": a["content"]}
         for a in (tok.get("added_tokens") or [])),
        key=lambda e: e["id"])
    if not quiet:
        which = ("DeepSeek-V4.1's three Splits" if pattern == TOKPAT_DEEPSEEK
                 else f"cl100k {'with' if han else 'without'} a Han branch")
        print(f"tokenizer: {len(lines)} merges, {len(specials)} specials"
              + (f" ({inline_specials} of them inside the BPE table)"
                 if inline_specials else "") + f", pattern {which}, "
              f"up to {digit_run} digit(s) per piece")
    return "\n".join(lines) + "\n", han, specials, digit_run, pattern


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
    text, han, specials, digit_run, pattern = convert(args.src)
    os.makedirs(args.out, exist_ok=True)
    with io.open(dst, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    if specials:
        with io.open(os.path.join(args.out, "specials.json"), "w",
                     encoding="utf-8", newline="\n") as f:
            # ensure_ascii=False for the reason convert.py's atomic_json
            # gives: a control token that is not ASCII should be in the file
            # as itself, not as \uXXXX.
            json.dump(specials, f, indent=1, ensure_ascii=False)
    notes = []
    if not han:
        notes.append("tokenizer_han_split must be false")
    if pattern != TOKPAT_CL100K:
        notes.append(f"tokenizer_pattern must be {pattern}")
    if digit_run != 3:
        notes.append(f"tokenizer_digit_run must be {digit_run}")
    print(f"wrote {dst}" + (f"  ({'; '.join(notes)})" if notes else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
