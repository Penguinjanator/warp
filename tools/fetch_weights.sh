#!/usr/bin/env bash
# fetch_weights.sh — download K3 weight shards to the staging disk.
#
# Storage split (docs/GATES.md, Gate H): raw shards land on the external
# staging disk; the converted WASTE container goes to internal NVMe.
#
#   ./tools/fetch_weights.sh --dry-run                 # preflight only
#   ./tools/fetch_weights.sh                           # download
#   ./tools/fetch_weights.sh --repo moonshotai/Kimi-K3-Instruct
#
# Preflight (always runs first, and --dry-run stops there):
#   - repo reachable and public?
#   - total download size from the safetensors index
#   - free space on the staging disk >= size + 5% margin
# Downloads are resumable (curl -C -); re-running skips complete files.

set -euo pipefail

REPO="${REPO:-moonshotai/Kimi-K3}"
DEST="${DEST:-/Volumes/WasteDisk/k3}"
DRY=0
JOBS="${JOBS:-4}"

while [ $# -gt 0 ]; do
  case "$1" in
    --repo) REPO="$2"; shift 2 ;;
    --dest) DEST="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

API="https://huggingface.co/api/models/${REPO}"
RAW="https://huggingface.co/${REPO}/resolve/main"

echo "repo:  $REPO"
echo "dest:  $DEST"

code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 30 "$API")
if [ "$code" != "200" ]; then
  echo "!! repo not publicly available (HTTP $code)."
  echo "   HuggingFace returns 401 both for missing and for gated/private repos."
  echo "   If the release just happened, try the -Base / -Instruct variants."
  exit 1
fi

mkdir -p "$DEST"

# --- small files first: they drive everything downstream -------------------
for f in config.json model.safetensors.index.json generation_config.json \
         tokenizer.json tokenizer_config.json; do
  if curl -sfL --max-time 120 -o "$DEST/$f.tmp" "$RAW/$f" 2>/dev/null; then
    mv "$DEST/$f.tmp" "$DEST/$f"
    echo "  got $f"
  else
    rm -f "$DEST/$f.tmp"
    echo "  (no $f)"
  fi
done

[ -f "$DEST/model.safetensors.index.json" ] || {
  echo "!! no safetensors index — single-file model or different layout;"
  echo "   inspect $API and adapt."; exit 1; }

# --- preflight: size vs free space ----------------------------------------
read -r TOTAL NSHARDS < <(python3 - "$DEST/model.safetensors.index.json" <<'PY'
import json, sys
idx = json.load(open(sys.argv[1]))
shards = sorted(set(idx["weight_map"].values()))
total = idx.get("metadata", {}).get("total_size", 0)
print(total, len(shards))
PY
)

AVAIL=$(df -k "$DEST" | awk 'NR==2 {print $4 * 1024}')
NEED=$(( TOTAL * 105 / 100 ))

python3 - "$TOTAL" "$AVAIL" "$NSHARDS" <<'PY'
import sys
tot, avail, n = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
g = 1 << 30
print(f"\nshards: {n}")
print(f"download size : {tot/g:8.1f} GB")
print(f"free on dest  : {avail/g:8.1f} GB")
print(f"after download: {(avail-tot)/g:8.1f} GB free")
PY

if [ "$AVAIL" -lt "$NEED" ]; then
  echo "!! not enough free space on $DEST (need ~5% margin). Aborting."
  exit 1
fi

if [ "$DRY" = "1" ]; then
  echo "--dry-run: stopping before download."
  echo "Next: tools/routing_stats.py math --data $DEST"
  exit 0
fi

# --- shards, resumable, N in parallel -------------------------------------
python3 - "$DEST/model.safetensors.index.json" > "$DEST/.shards" <<'PY'
import json, sys
idx = json.load(open(sys.argv[1]))
for s in sorted(set(idx["weight_map"].values())):
    print(s)
PY

echo
echo "downloading $NSHARDS shards with $JOBS parallel streams (resumable)..."

# The per-shard worker lives in its own file: passing it inline to xargs -I
# blows macOS's command-line assembly limit.
WORKER="$DEST/.worker.sh"
cat > "$WORKER" <<WORKEREOF
#!/bin/sh
f="\$1"
dest="$DEST"
raw="$RAW"
want=\$(curl -sIL "\$raw/\$f" | awk -F': ' '/^[Cc]ontent-[Ll]ength/ {print \$2}' | tr -d '\r' | tail -1)
have=\$(stat -f%z "\$dest/\$f" 2>/dev/null || echo 0)
if [ -n "\$want" ] && [ "\$have" = "\$want" ]; then echo "  ok   \$f (\$((have/1024/1024)) MB)"; exit 0; fi
echo "  pull \$f"
if curl -sfL -C - --retry 5 --retry-delay 5 -o "\$dest/\$f" "\$raw/\$f"; then
  echo "  done \$f"
else
  echo "  FAIL \$f" >&2
  exit 1
fi
WORKEREOF
chmod +x "$WORKER"

xargs -P "$JOBS" -n1 "$WORKER" < "$DEST/.shards"

rm -f "$DEST/.shards" "$WORKER"
echo
echo "download complete: $DEST"
echo "Next: tools/routing_stats.py math --data $DEST   (freeze the format TBDs)"
