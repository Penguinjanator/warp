#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
# Does the engine actually stay inside waste_cfg.ram_budget_bytes?
#
# waste.h calls the budget a hard ceiling on everything the engine
# allocates. That is a claim about peak RSS, so measure peak RSS.
set -uo pipefail
cd "$(dirname "$0")/.."
MODEL="${1:-$HOME/models/kimi-linear.waste}"
BUDGET_GB="${2:-6}"
# A short prompt never allocates the chunked-prefill scratch, which is the
# largest single thing the plan has to size. Pass "long" to force a chunk.
case "${3:-}" in
    long) PROMPT=$(python3 -c "print(' '.join(['token'] * 90))");;
    *)    PROMPT="hello";;
esac

# Peak RSS of the child, through getrusage. /usr/bin/time was the obvious
# tool and the wrong one: its flags differ between BSD and GNU, and it is
# not installed in a plain debian image at all — which is how this check
# first "failed" on Linux. ru_maxrss is bytes on macOS, kilobytes elsewhere.
RSS=$(python3 -c '
import resource, subprocess, sys
subprocess.run(["./waste", "run", sys.argv[1], sys.argv[2], "-n", "2",
                "--budget", sys.argv[3] + "G", "-q"],
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
r = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
print(r if sys.platform == "darwin" else r * 1024)
' "$MODEL" "$PROMPT" "$BUDGET_GB")
[ -z "$RSS" ] && { echo "could not read peak RSS"; exit 1; }

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
