#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""
kimi_tok.py — minimal Kimi tokenizer (tiktoken BPE), enough to drive the
reference and the engine end to end.

The HF tokenizer class pulls in transformers; all it really does is build a
tiktoken.Encoding from tiktoken.model plus the pat_str and the reserved
special-token block. That is reproduced here verbatim from
tokenization_kimi.py.
"""

import base64
import os

import tiktoken

PAT_STR = "|".join([
    r"""[\p{Han}]+""",
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
    r"""\p{N}{1,3}""",
    r""" ?[^\s\p{L}\p{N}]+[\r\n]*""",
    r"""\s*[\r\n]+""",
    r"""\s+(?!\S)""",
    r"""\s+""",
])

SPECIAL_NAMES = {
    "<|im_end|>", "<|im_user|>", "<|im_assistant|>", "<|start_header_id|>",
    "<|end_header_id|>", "[EOT]", "<|im_system|>", "<|im_middle|>",
    "[BOS]", "[EOS]",
}
N_RESERVED = 256


def load(model_dir):
    path = os.path.join(model_dir, "tiktoken.model")
    ranks = {}
    with open(path, "rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tok, rank = line.split()
            ranks[base64.b64decode(tok)] = int(rank)
    base = len(ranks)
    # tokenizer_config.json names some of the reserved slots; the rest keep
    # their placeholder names. Only the ids matter for us.
    import json
    cfg_path = os.path.join(model_dir, "tokenizer_config.json")
    named = {}
    if os.path.exists(cfg_path):
        cfg = json.load(open(cfg_path))
        for i, ent in (cfg.get("added_tokens_decoder") or {}).items():
            named[int(i)] = ent["content"]
    special = {named.get(i, f"<|reserved_token_{i}|>"): i
               for i in range(base, base + N_RESERVED + 2)}
    enc = tiktoken.Encoding(name="kimi", pat_str=PAT_STR,
                            mergeable_ranks=ranks, special_tokens=special)
    return enc, special


if __name__ == "__main__":
    import sys
    enc, sp = load(sys.argv[1] if len(sys.argv) > 1 else "/Volumes/WasteDisk/kimi-linear")
    print(f"vocab {enc.n_vocab}")
    for name in ("[BOS]", "[EOS]", "<|im_user|>", "<|im_assistant|>",
                 "<|im_middle|>", "<|im_end|>"):
        print(f"  {name:<22} {sp.get(name)}")
    s = "The capital of France is"
    ids = enc.encode(s)
    print(f"\n{s!r} -> {ids} -> {enc.decode(ids)!r}")
