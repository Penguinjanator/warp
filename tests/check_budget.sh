#!/usr/bin/env bash
# Does the engine actually stay inside waste_cfg.ram_budget_bytes?
#
# waste.h calls the budget a hard ceiling on everything the engine
# allocates. That is a claim about peak RSS, so measure peak RSS.
set -uo pipefail
cd "$(dirname "$0")/.."
MODEL="${1:-/Users/marco/models/kimi-linear.waste}"
BUDGET_GB="${2:-6}"
# A short prompt never allocates the chunked-prefill scratch, which is the
# largest single thing the plan has to size. Pass "long" to force a chunk.
case "${3:-}" in
    long) PROMPT=$(python3 -c "print(' '.join(['token'] * 90))");;
    *)    PROMPT="hello";;
esac

# /usr/bin/time -l reports peak RSS in bytes on macOS, KB on Linux
OUT=$(/usr/bin/time -l ./waste run "$MODEL" "$PROMPT" -n 2 --budget "${BUDGET_GB}G" -q 2>&1)
RSS=$(echo "$OUT" | grep -E "maximum resident set size" | tr -s ' ' | cut -d' ' -f2)
[ -z "$RSS" ] && { echo "could not read peak RSS"; exit 1; }
case "$(uname)" in Linux) RSS=$((RSS * 1024));; esac

python3 - "$RSS" "$BUDGET_GB" <<'PY'
import sys
rss = int(sys.argv[1]); budget = float(sys.argv[2]) * (1 << 30)
# Allow the process image itself (binary, libc, thread stacks) on top of
# what the engine allocates: 512 MB is generous and still catches a real
# overrun, which would be gigabytes.
slack = 512 << 20
print(f"peak RSS {rss/2**30:.2f} GB, budget {budget/2**30:.2f} GB, "
      f"slack {slack/2**20:.0f} MB")
print("BUDGET OK" if rss <= budget + slack else "BUDGET EXCEEDED")
sys.exit(0 if rss <= budget + slack else 1)
PY
