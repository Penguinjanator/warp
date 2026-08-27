#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""tokdiff.py — the C tokenizer against the release's own, on a fixed corpus.

Which reference depends on what the release ships: a tiktoken rank file
(both Kimi models) is read with tiktoken, and a `tokenizers` tokenizer.json
(GLM-5.3-Flash) with `tokenizers`. The second is the one that needs
checking hardest — the container's rank file was *re-encoded* from that
JSON by tools/hf_tokenizer.py, so this is the only place the re-encoding is
compared against the thing it was derived from.

  uv run --with tiktoken python tools/tokdiff.py CONTAINER SRC_WEIGHTS
  uv run --with tokenizers python tools/tokdiff.py CONTAINER GLM_SRC
"""
import subprocess, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CONT = sys.argv[1] if len(sys.argv) > 1 else sys.exit("usage: tokdiff.py CONTAINER [text]")
SRC = sys.argv[2] if len(sys.argv) > 2 else "/Volumes/WasteDisk/kimi-linear"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The Han branch is the one pre-tokenization difference in the family, and
# the C side takes it from the container config at run time — which
# test_tokenizer, opening a directory rather than a manifest, cannot see.
env = dict(os.environ)
if any(os.path.exists(os.path.join(SRC, n))
       for n in ("tiktoken.model", "tokenizer.model")):
    import kimi_tok
    _enc, _ = kimi_tok.load(SRC)
    encode = _enc.encode
    which = "tiktoken"
else:
    from tokenizers import Tokenizer
    import hf_tokenizer
    _tk = Tokenizer.from_file(os.path.join(SRC, "tokenizer.json"))
    if not hf_tokenizer.convert(SRC, quiet=True)[1]:
        env["WASTE_TOK_NOHAN"] = "1"
    def encode(t):
        return _tk.encode(t, add_special_tokens=False).ids
    which = "tokenizers"
tests = [
 "The capital of France is",
 "Hello, world!",
 "Write a C function that parses a JSON array of integers.",
 "int main(void) { return 0; }",
 "La capitale d'Italia e' Roma, e la popolazione e' 60.000.000.",
 "  indented\n\ttabbed\n\nblank line",
 "numbers 1234567 and 42 and 007",
 "snake_case CamelCase SCREAMING_CASE kebab-case",
 "don't can't we've I'll you're it's",
 "emoji: cafe naive Zurich",
 'x=1;y=2;/*comment*/ printf("%d\\n", x+y);',
 "Perche' non funziona? Perche' si'.",
 # Long single pre-tokens. The pre-tokenizer gives Han its own branch and
 # consumes a whole unpunctuated run, so one of these is one piece — and
 # the C side used to truncate a piece at 256 bytes while advancing past
 # all of it, dropping the rest of the prompt without a word. The cliff
 # was 85 Chinese characters. Nothing above was long enough to find it:
 # twelve short ASCII strings is not a tokenizer corpus.
 "好" * 86,
 "好" * 200,
 "=" * 400,
 "-" * 300 + "\n" + "=" * 300,
 "\n" * 300,
 "a" * 400,
 "好" * 90 + "! " + "=" * 300 + " fine",
 # Han running straight into Latin. Kimi's pattern gives Han its own branch
 # and GLM's does not, so "A股" is two pieces there and one token here —
 # the whole reason tokenizer_han_split exists, and invisible without a
 # case that crosses the boundary.
 "A股 维生素C 和C罗聊QQ音乐",
 "中文abc mixed 汉字test",
 # Everything here stays under the 1024-byte BPE window, where the C
 # tokenizer is exact. Past it a piece is encoded in windows and can
 # differ from tiktoken by a token per seam — deliberately, and
 # documented in encode_piece. All bytes survive either way.
]
out = subprocess.run([os.path.join(HERE, "test_tokenizer"), CONT, *tests],
                     capture_output=True, text=True,
                     env=env).stdout.strip().split("\n")
ok = 0
for t, line in zip(tests, out):
    c = [int(x) for x in line.split()][1:]
    p = list(encode(t))
    if c == p:
        ok += 1
    else:
        print(f"DIFF {t[:44]!r}\n  C      {c}\n  Python {p}")
print(f"{ok}/{len(tests)} identical (against {which})")
sys.exit(0 if ok == len(tests) else 1)
