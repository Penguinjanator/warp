#!/usr/bin/env python3
"""tokdiff.py — C tokenizer vs Python tiktoken on a fixed corpus.

  uv run --with tiktoken python tools/tokdiff.py CONTAINER SRC_WEIGHTS
"""
import subprocess, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kimi_tok

CONT = sys.argv[1] if len(sys.argv) > 1 else "/Users/marco/models/kimi-linear.waste"
SRC = sys.argv[2] if len(sys.argv) > 2 else "/Volumes/WasteDisk/kimi-linear"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

enc, _ = kimi_tok.load(SRC)
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
]
out = subprocess.run([os.path.join(HERE, "test_tokenizer"), CONT, *tests],
                     capture_output=True, text=True).stdout.strip().split("\n")
ok = 0
for t, line in zip(tests, out):
    c = [int(x) for x in line.split()][1:]
    p = enc.encode(t)
    if c == p:
        ok += 1
    else:
        print(f"DIFF {t[:44]!r}\n  C      {c}\n  Python {p}")
print(f"{ok}/{len(tests)} identical")
sys.exit(0 if ok == len(tests) else 1)
