#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
# pipeline.sh — download -> convert -> verify -> first K3 run, unattended.
#
# Every stage is resumable and refuses to start on a failed predecessor, so
# this can be killed and restarted at any point. It writes a running report
# to $RUN/pipeline.log and a final summary to $RUN/REPORT.md, where
# $RUN defaults to $OUT.runs — beside the container, not inside it, so
# the container holds only what the engine loads.
#
#   tools/pipeline.sh
#   SRC=... OUT=... JOBS=3 tools/pipeline.sh
#
# Stages
#   1 download   loops tools/fetch_k3.sh until all shards verify
#   2 convert    tools/convert.py --jobs N (skips layers already written)
#   3 verify     container round-trip against the source weights
#   4 run        the C engine generates from a prompt
#   5 oracle     PyTorch reference on the same prompt, logits diffed
#
# Stage 5 is the one that matters: it is the first end-to-end check that
# K3's latent MoE, Attention Residuals, SiTU and full-rank gate are wired
# up correctly, not merely implemented correctly in isolation.

set -uo pipefail
cd "$(dirname "$0")/.."

SRC="${SRC:-/Volumes/WasteDisk/k3}"
OUT="${OUT:-/Users/marco/models/k3.waste}"
JOBS="${JOBS:-3}"
PROMPT="${PROMPT:-The capital of France is}"
NTOK="${NTOK:-24}"
RUN="${RUN_DIR:-$OUT.runs}"          # reports and logs, never inside $OUT
LOG="$RUN/pipeline.log"
UV="uv run --quiet --with torch --with fla-core --with einops --no-project python"

mkdir -p "$OUT" "$RUN"
say() { printf '%s  %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$LOG"; }
die() { say "FAILED at $*"; echo "$*" > "$RUN/.failed"; exit 1; }

rm -f "$RUN/.failed"
say "=== pipeline start ==="
say "src $SRC -> out $OUT, $JOBS conversion processes"

# ---- 1. download ---------------------------------------------------------
NEED=$(python3 -c "
import json;print(len(set(json.load(open('$SRC/model.safetensors.index.json'))['weight_map'].values())))" 2>/dev/null || echo 0)
[ "$NEED" -gt 0 ] || die "download (no index at $SRC)"

have() { wc -l < "$SRC/.download-state" 2>/dev/null | tr -d ' '; }
say "stage 1: download — $(have)/$NEED shards complete"
tries=0
while [ "$(have)" -lt "$NEED" ]; do
    tries=$((tries + 1))
    [ "$tries" -gt 200 ] && die "download (gave up after $tries passes)"
    ./tools/fetch_k3.sh >/dev/null 2>&1
    say "  pass $tries: $(have)/$NEED"
done
say "stage 1 done: all $NEED shards verified"

# ---- 2. convert ----------------------------------------------------------
FREE=$(df -g "$(dirname "$OUT")" | awk 'NR==2 {print $4}')
say "stage 2: convert — ${FREE} GB free on the target volume"
[ "$FREE" -lt 1100 ] && die "convert (need ~1.0 TB, ${FREE} GB free)"

T0=$(date +%s)
$UV tools/convert.py --src "$SRC" --out "$OUT" --jobs "$JOBS" >>"$LOG" 2>&1 \
    || die "convert"
say "stage 2 done in $(( ($(date +%s) - T0) / 60 )) min, $(du -sh "$OUT" | cut -f1)"

# ---- 3. verify the container --------------------------------------------
say "stage 3: container round-trip"
$UV tools/verify_container.py --container "$OUT" --src "$SRC" --experts 1 \
    > "$RUN/verify.txt" 2>&1
grep -q "^PASS" "$RUN/verify.txt" || die "verify (see $RUN/verify.txt)"
ERR=$(grep -oE "rel err +[0-9.]+%" "$RUN/verify.txt" | head -1)
say "stage 3 done: round-trip PASS, $ERR"

# ---- 4. the engine actually runs -----------------------------------------
say "stage 4: engine"
make -s >>"$LOG" 2>&1 || die "build"
./waste run "$OUT" "$PROMPT" -n "$NTOK" --budget 46G > "$RUN/generated.txt" 2>&1 \
    || die "engine run (see $RUN/generated.txt)"
say "stage 4 done: $(head -c 200 "$RUN/generated.txt")"

# ---- 5. against the oracle ----------------------------------------------
say "stage 5: PyTorch oracle (slow — 93 layers in Python)"
IDS=$(./test_tokenizer "$OUT" "$PROMPT" | head -1 | cut -d' ' -f2- | tr ' ' ',')
[ -n "$IDS" ] || die "tokenize"
say "  prompt ids: $IDS"

./test_forward "$OUT" "$IDS" "$RUN/logits_c.bin" 0 >>"$LOG" 2>&1 || die "engine logits"
$UV tools/kimi_ref.py --container "$OUT" --tokens 0 --prompt-ids "$IDS" \
    --dump "$RUN/logits_ref.bin" >>"$LOG" 2>&1 || die "oracle"

python3 - "$RUN/logits_c.bin" "$RUN/logits_ref.bin" > "$RUN/diff.txt" <<'PY'
import struct, sys
def L(p):
    b = open(p, "rb").read()
    return struct.unpack(f"<{len(b)//4}f", b)
a, b = L(sys.argv[1]), L(sys.argv[2])
md = max(abs(x - y) for x, y in zip(a, b))
na = sum(x * x for x in b) ** 0.5
rel = (sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5) / (na or 1)
ta = sorted(range(len(a)), key=lambda i: -a[i])[:10]
tb = sorted(range(len(b)), key=lambda i: -b[i])[:10]
print(f"max|diff| {md:.3e}   rel {rel:.3e}")
print(f"argmax  engine {ta[0]}  oracle {tb[0]}  {'MATCH' if ta[0]==tb[0] else 'MISMATCH'}")
print(f"top-10  {'identical' if ta==tb else 'differ'}")
print("ORACLE OK" if md < 1e-2 and ta[0] == tb[0] else "ORACLE MISMATCH")
PY
cat "$RUN/diff.txt" | tee -a "$LOG"
grep -q "^ORACLE OK" "$RUN/diff.txt" || die "oracle diff (see $RUN/diff.txt)"

# ---- report --------------------------------------------------------------
{
    echo "# K3 pipeline report"
    echo
    echo "Finished $(date '+%Y-%m-%d %H:%M')."
    echo
    echo '## Container'
    echo
    echo "- source: \`$SRC\` ($(du -sh "$SRC" | cut -f1))"
    echo "- container: \`$OUT\` ($(du -sh "$OUT" | cut -f1))"
    echo "- round-trip error: $ERR"
    echo
    echo '## Generated'
    echo
    echo '```'
    cat "$RUN/generated.txt"
    echo '```'
    echo
    echo '## Against the PyTorch oracle'
    echo
    echo '```'
    cat "$RUN/diff.txt"
    echo '```'
} > "$RUN/REPORT.md"

say "=== pipeline complete — see $RUN/REPORT.md ==="
