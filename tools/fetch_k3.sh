#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
# fetch_k3.sh — long-haul download of a very large model, built to survive.
#
# A 1.4 TB pull over hours will hit dropped connections, 5xx from the CDN,
# and at least one interrupted run. So:
#   - every shard is resumed with `curl -C -`, never restarted;
#   - each shard is retried with exponential backoff and jitter;
#   - a shard counts as done only when its size matches Content-Length,
#     and completed shards are recorded in a state file so re-runs skip
#     them without even a HEAD request;
#   - free space is checked before starting and again every shard, so the
#     run stops cleanly instead of filling the disk;
#   - progress, failures and the resume point are logged with timestamps.
#
#   tools/fetch_k3.sh                    # start or resume
#   tools/fetch_k3.sh --check            # verify what is on disk, no download
#   REPO=... DEST=... JOBS=3 tools/fetch_k3.sh
#
# Safe to run repeatedly and safe to kill: the next run picks up where it
# stopped, mid-shard.

set -uo pipefail

REPO="${REPO:-moonshotai/Kimi-K3}"
DEST="${DEST:-/Volumes/WasteDisk/k3}"
JOBS="${JOBS:-3}"
MAX_RETRY="${MAX_RETRY:-8}"
MIN_FREE_GB="${MIN_FREE_GB:-40}"
CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1

RAW="https://huggingface.co/${REPO}/resolve/main"
STATE="$DEST/.download-state"
LOG="$DEST/download.log"

mkdir -p "$DEST"
touch "$STATE"

log() { printf '%s  %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$LOG"; }

free_gb() { df -g "$DEST" | awk 'NR==2 {print $4}'; }

# --- small files ----------------------------------------------------------
for f in config.json model.safetensors.index.json generation_config.json \
         tokenizer_config.json tiktoken.model configuration_kimi_k3.py \
         modeling_kimi_k3.py modeling_kimi_linear.py chat_template.jinja; do
    [ -s "$DEST/$f" ] && continue
    curl -sfL --max-time 300 -o "$DEST/$f.part" "$RAW/$f" 2>/dev/null \
        && mv "$DEST/$f.part" "$DEST/$f" && log "got $f" \
        || rm -f "$DEST/$f.part"
done

[ -s "$DEST/model.safetensors.index.json" ] || { log "FATAL: no index"; exit 1; }

# --- plan -----------------------------------------------------------------
python3 - "$DEST/model.safetensors.index.json" > "$DEST/.shards" <<'PY'
import json, sys
idx = json.load(open(sys.argv[1]))
for s in sorted(set(idx["weight_map"].values())):
    print(s)
PY
TOTAL=$(wc -l < "$DEST/.shards" | tr -d ' ')
SIZE_GB=$(python3 -c "
import json;print(f'{json.load(open(\"$DEST/model.safetensors.index.json\"))[\"metadata\"][\"total_size\"]/2**30:.0f}')")

have=0
while read -r f; do grep -qxF "$f" "$STATE" && have=$((have+1)); done < "$DEST/.shards"
log "repo $REPO -> $DEST"
log "$TOTAL shards, ${SIZE_GB} GB total; $have already complete; $(free_gb) GB free"

if [ "$CHECK_ONLY" = 1 ]; then
    log "--check: verifying sizes on disk against the remote"
    bad=0
    while read -r f; do
        [ -f "$DEST/$f" ] || continue
        want=$(curl -sIL --max-time 60 "$RAW/$f" | awk -F': ' '/^[Cc]ontent-[Ll]ength/{print $2}' | tr -d '\r' | tail -1)
        got=$(stat -f%z "$DEST/$f" 2>/dev/null || echo 0)
        if [ "$want" != "$got" ]; then log "  INCOMPLETE $f ($got / ${want:-?})"; bad=$((bad+1)); fi
    done < "$DEST/.shards"
    log "--check done: $bad incomplete"
    exit 0
fi

REMAIN_GB=$(( SIZE_GB - have * SIZE_GB / TOTAL ))
if [ "$(free_gb)" -lt "$REMAIN_GB" ]; then
    log "FATAL: need ~${REMAIN_GB} GB, only $(free_gb) GB free"
    exit 1
fi

# --- worker ---------------------------------------------------------------
cat > "$DEST/.worker.sh" <<WORKER
#!/bin/bash
f="\$1"
dest="$DEST"; raw="$RAW"; state="$STATE"; log="$LOG"
max_retry=$MAX_RETRY; min_free=$MIN_FREE_GB

say() { printf '%s  %s\n' "\$(date '+%H:%M:%S')" "\$*" >> "\$log"; }

grep -qxF "\$f" "\$state" && exit 0

for try in \$(seq 1 \$max_retry); do
    free=\$(df -g "\$dest" | awk 'NR==2 {print \$4}')
    if [ "\$free" -lt "\$min_free" ]; then say "STOP \$f: only \${free} GB free"; exit 2; fi

    want=\$(curl -sIL --max-time 90 "\$raw/\$f" | awk -F': ' '/^[Cc]ontent-[Ll]ength/{print \$2}' | tr -d '\r' | tail -1)
    got=\$(stat -f%z "\$dest/\$f" 2>/dev/null || echo 0)
    if [ -n "\$want" ] && [ "\$got" = "\$want" ]; then
        # atomic append: flock is unavailable on macOS sh, but a single
        # write under the pipe buffer size is atomic enough here
        echo "\$f" >> "\$state"
        say "ok   \$f (\$((got/1048576)) MB)"
        exit 0
    fi

    [ "\$got" -gt 0 ] && say "resume \$f at \$((got/1048576)) MB (try \$try)" \\
                      || say "pull \$f (try \$try)"
    # -C - resumes; --speed-limit kills a connection that has stalled
    curl -fL -C - --retry 3 --retry-delay 5 --speed-limit 1024 --speed-time 120 \\
         -o "\$dest/\$f" "\$raw/\$f" 2>/dev/null
    rc=\$?
    [ \$rc -eq 0 ] && continue          # loop verifies the size next pass

    # exponential backoff with jitter, capped
    wait=\$(( (1 << (try > 6 ? 6 : try)) * 5 + RANDOM % 10 ))
    say "fail \$f rc=\$rc, retry in \${wait}s"
    sleep \$wait
done
say "GIVE UP \$f after \$max_retry tries"
exit 1
WORKER
chmod +x "$DEST/.worker.sh"

log "downloading with $JOBS parallel streams (resumable, ${MAX_RETRY} retries each)"
grep -vxF -f "$STATE" "$DEST/.shards" 2>/dev/null | xargs -P "$JOBS" -n1 "$DEST/.worker.sh"
rc=$?

done_now=$(wc -l < "$STATE" | tr -d ' ')
log "pass finished (rc=$rc): $done_now / $TOTAL shards complete, $(free_gb) GB free"
if [ "$done_now" -lt "$TOTAL" ]; then
    log "re-run this script to continue; nothing already downloaded is refetched"
    exit 1
fi
log "ALL SHARDS COMPLETE"
rm -f "$DEST/.worker.sh" "$DEST/.shards"
