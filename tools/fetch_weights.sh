#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
# fetch_weights.sh — long-haul download of a very large model, built to survive.
#
# Storage split (docs/GATES.md, Gate H): raw shards land on the external
# staging disk; the converted WASTE container goes to internal NVMe. That is
# why --dest defaults to a volume rather than to the repo.
#
# A 1.4 TB pull over hours will hit dropped connections, 5xx from the CDN,
# and at least one interrupted run. So:
#   - every shard is resumed with `curl -C -`, never restarted;
#   - each shard is retried with exponential backoff and jitter;
#   - a shard counts as done only when its size matches Content-Length,
#     and completed shards are recorded in a state file so re-runs skip
#     them without even a HEAD request;
#   - free space is checked before starting and again before every shard,
#     so the run stops cleanly instead of filling the disk;
#   - progress, failures and the resume point are logged with timestamps.
#
#   tools/fetch_weights.sh --dry-run          # preflight only, no download
#   tools/fetch_weights.sh                    # start or resume
#   tools/fetch_weights.sh --check            # verify what is on disk, no fetch
#   tools/fetch_weights.sh --repo moonshotai/Kimi-Linear --dest /data/kl
#
# Set HF_TOKEN for a gated repo. Safe to run repeatedly and safe to kill:
# the next run picks up where it stopped, mid-shard.

set -uo pipefail

REPO="${REPO:-moonshotai/Kimi-K3}"
DEST="${DEST:-/Volumes/WasteDisk/k3}"
JOBS="${JOBS:-3}"
MAX_RETRY="${MAX_RETRY:-8}"
MIN_FREE_GB="${MIN_FREE_GB:-40}"
DRY=0
CHECK_ONLY=0

while [ $# -gt 0 ]; do
    case "$1" in
        --repo) REPO="$2"; shift 2 ;;
        --dest) DEST="$2"; shift 2 ;;
        --jobs) JOBS="$2"; shift 2 ;;
        --dry-run) DRY=1; shift ;;
        --check) CHECK_ONLY=1; shift ;;
        -h|--help) sed -n '4,27p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

API="https://huggingface.co/api/models/${REPO}"
RAW="https://huggingface.co/${REPO}/resolve/main"
STATE="$DEST/.download-state"
LOG="$DEST/download.log"

# GNU and BSD disagree on how to ask a file its size, and this script has to
# run on the laptop that converts and on a Linux box that only fetches.
if stat --version >/dev/null 2>&1; then
    fsize() { stat -c%s "$1" 2>/dev/null || echo 0; }
    STAT_MODE=gnu
else
    fsize() { stat -f%z "$1" 2>/dev/null || echo 0; }
    STAT_MODE=bsd
fi

# -P forces POSIX single-line output: without it a long device name wraps
# and `NR==2` picks up half a row.
free_kb() { df -kP "$DEST" | awk 'NR==2 {print $4}'; }
free_gb() { echo $(( $(free_kb) / 1048576 )); }

# curl, with the auth header when HF_TOKEN is set. A function rather than an
# array of extra arguments because "${arr[@]}" on an *empty* array trips
# `set -u` in bash 3.2 — which is exactly what macOS ships, so the array
# version failed on the first machine it ran on. The token never reaches the
# log or the argument list of anything but curl itself.
hcurl() {
    if [ -n "${HF_TOKEN:-}" ]; then
        curl -H "Authorization: Bearer $HF_TOKEN" "$@"
    else
        curl "$@"
    fi
}

echo "repo:  $REPO"
echo "dest:  $DEST"

code=$(hcurl -s -o /dev/null -w '%{http_code}' --max-time 30 "$API")
if [ "$code" != "200" ]; then
    echo "!! repo not reachable (HTTP $code)."
    echo "   HuggingFace answers 401 both for a missing repo and for a gated"
    echo "   one, so this does not tell them apart. If the release just"
    echo "   happened, try the -Base / -Instruct variants; if the repo is"
    echo "   gated, set HF_TOKEN."
    exit 1
fi

mkdir -p "$DEST" || exit 1
touch "$STATE"

log() { printf '%s  %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$LOG"; }

# --- small files ----------------------------------------------------------
# The index is needed by every mode, because it is what says which shards
# exist and how big they are. Everything else is only needed by a real
# download and by convert.py afterwards, so --check and --dry-run stay
# inspection modes: they fetch nothing but the index, and nothing at all
# once it is on disk.
get_small() {
    for f in "$@"; do
        [ -s "$DEST/$f" ] && continue
        if hcurl -sfL --max-time 300 -o "$DEST/$f.part" "$RAW/$f" 2>/dev/null; then
            mv "$DEST/$f.part" "$DEST/$f"
            log "got $f"
        else
            rm -f "$DEST/$f.part"          # a 404 here is normal: not every
        fi                                  # repo ships every one of these
    done
}

get_small model.safetensors.index.json
if [ "$DRY" = 0 ] && [ "$CHECK_ONLY" = 0 ]; then
    # Ask the repo what it contains instead of guessing filenames. The
    # hardcoded list this replaces cost real money: it did not know about
    # encoding_k3.py, so the chat template looked absent and had to be
    # reconstructed from a figure in the technical report; and it silently
    # missed preprocessor_config.json, so the image normalization was the
    # CLIP convention for a day when the release states mean = std = 0.5.
    # A whitelist cannot report what it never knew to ask for.
    SMALL=$(hcurl -sfL --max-time 60 "$API" 2>/dev/null | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    sys.exit(1)
for s in d.get("siblings", []):
    f = s.get("rfilename", "")
    # shards come down the resumable path; skip directories and the index
    if f.endswith(".safetensors") or "/" in f or f == "model.safetensors.index.json":
        continue
    print(f)
' 2>/dev/null)
    if [ -n "$SMALL" ]; then
        log "repo lists $(printf '%s\n' "$SMALL" | wc -l | tr -d ' ') small files"
        # shellcheck disable=SC2086
        get_small $SMALL
    else
        log "WARNING: could not list the repo; falling back to known names."
        log "         Check $API by hand for files this misses."
        get_small config.json generation_config.json tokenizer.json \
                  tokenizer_config.json tiktoken.model preprocessor_config.json \
                  chat_template.jinja configuration_kimi_k3.py \
                  modeling_kimi_k3.py modeling_kimi_linear.py
    fi
fi

[ -s "$DEST/model.safetensors.index.json" ] || {
    log "FATAL: no safetensors index — single-file model or a different"
    log "       layout. Inspect $API and adapt."
    exit 1; }

# --- plan ------------------------------------------------------------------
python3 - "$DEST/model.safetensors.index.json" > "$DEST/.shards" <<'PY'
import json, sys
idx = json.load(open(sys.argv[1]))
for s in sorted(set(idx["weight_map"].values())):
    print(s)
PY
TOTAL=$(wc -l < "$DEST/.shards" | tr -d ' ')
TOTAL_BYTES=$(python3 - "$DEST/model.safetensors.index.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1])).get("metadata", {}).get("total_size", 0))
PY
)

# What is already here, in bytes rather than by shard count: a proportional
# estimate is wrong whenever shards differ in size, which they do.
have=0
have_bytes=0
while read -r f; do
    if grep -qxF "$f" "$STATE"; then
        have=$((have + 1))
        have_bytes=$((have_bytes + $(fsize "$DEST/$f")))
    fi
done < "$DEST/.shards"

log "repo $REPO -> $DEST  (stat: $STAT_MODE)"
python3 - "$TOTAL_BYTES" "$have_bytes" "$(free_kb)" "$TOTAL" "$have" <<'PY'
import sys
tot, got, availkb, n, nhave = (int(x) for x in sys.argv[1:6])
g = 1 << 30
avail = availkb * 1024
# The index's total_size counts tensor bytes; the files on disk carry their
# safetensors headers too, so a complete download measures slightly *more*
# than the advertised total. Without the clamp the last line of a finished
# run reads "still to fetch: -0.1 GB".
todo = max(0, tot - got)
print(f"\nshards        : {nhave} / {n} complete")
print(f"download size : {tot/g:8.1f} GB")
print(f"already here  : {got/g:8.1f} GB")
print(f"still to fetch: {todo/g:8.1f} GB")
print(f"free on dest  : {avail/g:8.1f} GB")
print(f"after download: {(avail-todo)/g:8.1f} GB free\n")
PY

# --- verify-only ----------------------------------------------------------
if [ "$CHECK_ONLY" = 1 ]; then
    log "--check: verifying sizes on disk against the remote"
    bad=0
    while read -r f; do
        [ -f "$DEST/$f" ] || continue
        want=$(hcurl -sIL --max-time 60 "$RAW/$f" \
               | awk -F': ' '/^[Cc]ontent-[Ll]ength/{print $2}' | tr -d '\r' | tail -1)
        got=$(fsize "$DEST/$f")
        if [ "$want" != "$got" ]; then
            log "  INCOMPLETE $f ($got / ${want:-?})"
            bad=$((bad + 1))
        fi
    done < "$DEST/.shards"
    log "--check done: $bad incomplete"
    rm -f "$DEST/.shards"
    exit $(( bad > 0 ))
fi

# --- preflight ------------------------------------------------------------
TODO_BYTES=$(( TOTAL_BYTES - have_bytes ))
[ "$TODO_BYTES" -lt 0 ] && TODO_BYTES=0
NEED_KB=$(( TODO_BYTES / 1024 * 105 / 100 ))                    # 5% margin
if [ "$(free_kb)" -lt "$NEED_KB" ]; then
    log "FATAL: need ~$(( NEED_KB / 1048576 )) GB with margin, only $(free_gb) GB free"
    exit 1
fi

if [ "$DRY" = 1 ]; then
    echo "--dry-run: stopping before download."
    rm -f "$DEST/.shards"
    exit 0
fi

# --- worker ---------------------------------------------------------------
# In its own file rather than inline: passing this to `xargs -I` blows
# macOS's command-line assembly limit.
cat > "$DEST/.worker.sh" <<WORKER
#!/bin/bash
f="\$1"
dest="$DEST"; raw="$RAW"; state="$STATE"; log="$LOG"
max_retry=$MAX_RETRY; min_free=$MIN_FREE_GB
hcurl() {
    if [ -n "\${HF_TOKEN:-}" ]; then
        curl -H "Authorization: Bearer \$HF_TOKEN" "\$@"
    else
        curl "\$@"
    fi
}
if [ "$STAT_MODE" = gnu ]; then
    fsize() { stat -c%s "\$1" 2>/dev/null || echo 0; }
else
    fsize() { stat -f%z "\$1" 2>/dev/null || echo 0; }
fi

say() { printf '%s  %s\n' "\$(date '+%H:%M:%S')" "\$*" >> "\$log"; }

grep -qxF "\$f" "\$state" && exit 0

for try in \$(seq 1 \$max_retry); do
    free=\$(( \$(df -kP "\$dest" | awk 'NR==2 {print \$4}') / 1048576 ))
    if [ "\$free" -lt "\$min_free" ]; then say "STOP \$f: only \${free} GB free"; exit 2; fi

    want=\$(hcurl -sIL --max-time 90 "\$raw/\$f" \\
           | awk -F': ' '/^[Cc]ontent-[Ll]ength/{print \$2}' | tr -d '\r' | tail -1)
    got=\$(fsize "\$dest/\$f")
    if [ -n "\$want" ] && [ "\$got" = "\$want" ]; then
        # A single line well under the pipe buffer, so concurrent appends
        # from the other workers do not interleave. flock is not portable
        # enough to rely on here.
        echo "\$f" >> "\$state"
        say "ok   \$f (\$((got/1048576)) MB)"
        exit 0
    fi

    [ "\$got" -gt 0 ] && say "resume \$f at \$((got/1048576)) MB (try \$try)" \\
                      || say "pull \$f (try \$try)"
    # -C - resumes; --speed-limit kills a connection that has stalled rather
    # than waiting out a TCP timeout that may never come.
    hcurl -fL -C - --retry 3 --retry-delay 5 --speed-limit 1024 --speed-time 120 \\
          -o "\$dest/\$f" "\$raw/\$f" 2>/dev/null
    rc=\$?
    [ \$rc -eq 0 ] && continue          # the next pass verifies the size

    # 33 is "server does not support byte ranges". Retrying the resume can
    # only fail the same way, so drop the partial file and let the next try
    # start clean — otherwise a mirror without Range support burns every
    # retry and gives up on a file it could have fetched whole.
    if [ \$rc -eq 33 ]; then
        say "no range support for \$f, restarting from zero"
        rm -f "\$dest/\$f"
        continue
    fi

    wait=\$(( (1 << (try > 6 ? 6 : try)) * 5 + RANDOM % 10 ))
    say "fail \$f rc=\$rc, retry in \${wait}s"
    sleep \$wait
done
say "GIVE UP \$f after \$max_retry tries"
exit 1
WORKER
chmod +x "$DEST/.worker.sh"

log "downloading with $JOBS parallel streams (resumable, $MAX_RETRY retries each)"
grep -vxF -f "$STATE" "$DEST/.shards" 2>/dev/null \
    | xargs -P "$JOBS" -n1 "$DEST/.worker.sh"
rc=$?

done_now=$(wc -l < "$STATE" | tr -d ' ')
log "pass finished (rc=$rc): $done_now / $TOTAL shards complete, $(free_gb) GB free"
if [ "$done_now" -lt "$TOTAL" ]; then
    log "re-run to continue; nothing already downloaded is refetched"
    exit 1
fi
log "ALL SHARDS COMPLETE"
rm -f "$DEST/.worker.sh" "$DEST/.shards"
echo "Next: tools/convert.py --src $DEST --out model.waste"
