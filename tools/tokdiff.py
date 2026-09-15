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
  ... --wide 20000       # add a randomized corpus over the whole BMP

The curated list below is twenty-one strings and its own comment says that
is not a tokenizer corpus. `--wide N` is the rest of the sentence: N random
strings drawn from every codepoint, plus a block of whitespace runs, which
is what it took to find that the character classes in tokenizer.c covered
the scripts they had been tried on and nothing else. On `--wide 20000`
(24021 strings), before the classes were generated: Kimi-Linear
22937, GLM-5.3-Flash 22914. After: 24017 and 24020, the remainder being
codepoints assigned after the Unicode revision the tables were generated
from. Twenty-one curated strings scored 21/21 throughout, both times.
"""
import random, subprocess, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

argv = sys.argv[1:]
WIDE = 0
if "--wide" in argv:
    i = argv.index("--wide")
    WIDE = int(argv[i + 1])
    del argv[i:i + 2]
CONT = argv[0] if argv else sys.exit("usage: tokdiff.py [--wide N] CONTAINER [src]")
SRC = argv[1] if len(argv) > 1 else "/Volumes/WasteDisk/kimi-linear"
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
    _han, _pat = hf_tokenizer.convert(SRC, quiet=True)[1:4:2]
    if not _han:
        env["WASTE_TOK_NOHAN"] = "1"
    env["WASTE_TOK_PATTERN"] = str(_pat)
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

if WIDE:
    # Every codepoint, thinned by a stride so the corpus stays a corpus and
    # not an enumeration, and the surrogates left out because they are not
    # characters. Seeded, so a failure is reproducible.
    random.seed(11)
    wide = [chr(c) for c in range(0x20, 0x2FFFF, 7)
            if not (0xD800 <= c <= 0xDFFF)]
    for _ in range(WIDE):
        tests.append("".join(random.choice(wide)
                             for _ in range(random.randint(1, 24))))
    # \s+(?!\S) has to back off one CHARACTER, and every space here but the
    # first two is more than one byte.
    spaces = [" ", "\t", "\u00a0", "\u3000", "\u2009", "\u2002"]
    for _ in range(max(200, WIDE // 5)):
        tests.append(random.choice(["", "a", "1", "\u4e2d"]) +
                     "".join(random.choice(spaces)
                             for _ in range(random.randint(1, 4))) +
                     random.choice(["", "b", "2", "\u3002", "\n"]))

tests = [t for t in tests if "\0" not in t]
ok, shown = 0, 0
for i in range(0, len(tests), 200):
    chunk = tests[i:i + 200]
    out = subprocess.run([os.path.join(HERE, "test_tokenizer"), CONT, *chunk],
                         capture_output=True, text=True,
                         env=env).stdout.strip().split("\n")
    for t, line in zip(chunk, out):
        c = [int(x) for x in line.split()][1:]
        p = list(encode(t))
        if c == p:
            ok += 1
        elif shown < 10:
            shown += 1
            print(f"DIFF {t[:44]!r}\n  C      {c[:24]}\n  Python {p[:24]}")
print(f"{ok}/{len(tests)} identical (against {which})")
sys.exit(0 if ok == len(tests) else 1)
