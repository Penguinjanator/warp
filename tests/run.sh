#!/usr/bin/env bash
# tests/run.sh — every check we have, in one place, exiting non-zero on the
# first real failure.
#
# Written after losing time twice to checks that silently did not run: once
# to objects compiled against a stale header, once to a stale test binary.
# So this rebuilds first, states what it is about to do, and never treats a
# missing prerequisite as a pass — it says SKIP, loudly.
#
#   tests/run.sh [model.waste]
#
# Env: WASTE_REF_MODEL  container to use for the end-to-end checks
#      WASTE_REF_SRC    source weights, for the container round-trip
#      WASTE_ORACLE     logits dumped by tools/kimi_ref.py for THE SAME
#                       token ids this script uses (see IDS below) — a dump
#                       from a different prompt will look like an engine bug

set -uo pipefail
cd "$(dirname "$0")/.."

MODEL="${1:-${WASTE_REF_MODEL:-/Users/marco/models/kimi-linear.waste}}"
SRC="${WASTE_REF_SRC:-/Volumes/WasteDisk/kimi-linear}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

pass=0; fail=0; skip=0
ok()   { printf "  \033[32mPASS\033[0m  %s\n" "$1"; pass=$((pass+1)); }
no()   { printf "  \033[31mFAIL\033[0m  %s\n" "$1"; fail=$((fail+1)); }
sk()   { printf "  \033[33mSKIP\033[0m  %s — %s\n" "$1" "$2"; skip=$((skip+1)); }
head_() { printf "\n\033[1m%s\033[0m\n" "$1"; }

head_ "build"
if make -s test >/dev/null 2>&1 && make -s >/dev/null 2>&1; then
    ok "make && make test"
else
    no "build failed"
    make test 2>&1 | grep -E "error" | head -5
    exit 1
fi

# ---------------------------------------------------------------- unit ----
head_ "kernels vs the reference implementations"

if command -v uv >/dev/null 2>&1; then
    ./test_k3parts "$TMP/k3parts.bin" >/dev/null 2>&1
    if uv run --quiet --with torch --no-project python tools/k3parts_ref.py \
           "$TMP/k3parts.bin" 2>/dev/null | grep -q "^PASS"; then
        ok "K3 components (SiTU, both decay gates, AttnRes)"
    else
        no "K3 components"
    fi

    if KDA_T=24 KDA_H=4 KDA_K=32 KDA_V=32 uv run --quiet --with torch \
           --with fla-core --with einops --no-project python tools/kda_ref.py \
           2>/dev/null | grep -q "^PASS"; then
        ok "KDA kernel vs fla's naive_recurrent_kda"
    else
        no "KDA kernel"
    fi
else
    sk "kernel checks" "uv not installed"
fi

# ------------------------------------------------------------ container ----
head_ "container"

if [ -d "$MODEL" ]; then
    if ./test_container "$MODEL"/experts-L*.bin 2 2>/dev/null | grep -q "0 problems"; then
        ok "expert records read through the C structs (magic, offsets, 4 KiB)"
    else
        no "expert record layout"
    fi

    if [ -d "$SRC" ] && command -v uv >/dev/null 2>&1; then
        if uv run --quiet --with torch --no-project python tools/verify_container.py \
               --container "$MODEL" --src "$SRC" --experts 1 2>/dev/null \
               | grep -q "^PASS"; then
            ok "dequantized weights match the source"
        else
            no "container round-trip"
        fi
    else
        sk "container round-trip" "source weights not at $SRC"
    fi
else
    sk "container checks" "no container at $MODEL"
fi

# ----------------------------------------------------------------- e2e ----
head_ "engine"

if [ -d "$MODEL" ]; then
    IDS=1008,10484,318,15383,387,11,316,276,10484,318,19509,387,31082,13,646,10484

    ./test_forward "$MODEL" "$IDS" "$TMP/seq.bin" 0 >/dev/null 2>&1
    WASTE_CHUNK=1 ./test_forward "$MODEL" "$IDS" "$TMP/chunk.bin" 0 >/dev/null 2>&1
    if python3 - "$TMP/seq.bin" "$TMP/chunk.bin" <<'PY'
import struct, sys
def L(p):
    b = open(p, "rb").read()
    return struct.unpack(f"<{len(b)//4}f", b)
a, b = L(sys.argv[1]), L(sys.argv[2])
d = max(abs(x - y) for x, y in zip(a, b))
sys.exit(0 if d < 1e-3 and a.index(max(a)) == b.index(max(b)) else 1)
PY
    then ok "chunked prefill == token-at-a-time"
    else no "chunked prefill diverges"
    fi

    WASTE_Q8=0 ./test_forward "$MODEL" "$IDS" "$TMP/f32.bin" 0 >/dev/null 2>&1
    if python3 - "$TMP/seq.bin" "$TMP/f32.bin" <<'PY'
import struct, sys
def L(p):
    b = open(p, "rb").read()
    return struct.unpack(f"<{len(b)//4}f", b)
a, b = L(sys.argv[1]), L(sys.argv[2])
sys.exit(0 if max(abs(x - y) for x, y in zip(a, b)) < 1e-3 else 1)
PY
    then ok "int8 trunk storage == f32 weights"
    else no "int8 storage changes results"
    fi

    WASTE_BACKEND=cpu ./test_forward "$MODEL" "$IDS" "$TMP/cpu.bin" 0 >/dev/null 2>&1
    if cmp -s "$TMP/seq.bin" "$TMP/cpu.bin"; then
        ok "SIMD backend bit-identical to the CPU baseline"
    else
        # a difference here is allowed to be tiny, but must be tiny
        if python3 - "$TMP/seq.bin" "$TMP/cpu.bin" <<'PY'
import struct, sys
def L(p):
    b = open(p, "rb").read()
    return struct.unpack(f"<{len(b)//4}f", b)
a, b = L(sys.argv[1]), L(sys.argv[2])
sys.exit(0 if max(abs(x - y) for x, y in zip(a, b)) < 1e-3 else 1)
PY
        then ok "SIMD backend matches the CPU baseline (within fp noise)"
        else no "SIMD backend diverges from the CPU baseline"
        fi
    fi

    WASTE_CACHE_MB=512 ./test_forward "$MODEL" "$IDS" "$TMP/cache.bin" 0 >/dev/null 2>&1
    if cmp -s "$TMP/seq.bin" "$TMP/cache.bin"; then
        ok "expert cache is bit-identical to no cache"
    else
        no "expert cache changes results"
    fi

    ORACLE="${WASTE_ORACLE:-tests/fixtures/oracle_kimilinear_16tok.bin}"
    if [ -f "$ORACLE" ]; then
        if python3 - "$TMP/seq.bin" "$ORACLE" <<'PY'
import struct, sys
def L(p):
    b = open(p, "rb").read()
    return struct.unpack(f"<{len(b)//4}f", b)
a, b = L(sys.argv[1]), L(sys.argv[2])
sys.exit(0 if max(abs(x - y) for x, y in zip(a, b)) < 1e-3 else 1)
PY
        then ok "engine matches the PyTorch oracle"
        else no "engine diverges from the oracle"
        fi
    else
        sk "oracle diff" "no fixture; regenerate with tools/kimi_ref.py --dump"
    fi
else
    sk "engine checks" "no container at $MODEL"
fi

    if ./test_state "$MODEL" 2>/dev/null | grep -q "^STATE OK"; then
        ok "saved session resumes identically"
    else
        no "session state does not round-trip"
    fi

    # learned hotlist: a second run should start warmer than the first
    rm -f "$MODEL/usage.waste"
    cold=$(./waste run "$MODEL" "The capital of France is" -n 12 --budget 5G --learn 2>&1 \
           | grep -oE "[0-9]+ miss" | grep -oE "[0-9]+" || echo 0)
    warm=$(./waste run "$MODEL" "The capital of France is" -n 12 --budget 5G 2>&1 \
           | grep -oE "[0-9]+ miss" | grep -oE "[0-9]+" || echo 0)
    rm -f "$MODEL/usage.waste"
    if [ "${warm:-0}" -gt 0 ] && [ "${warm:-0}" -lt "${cold:-0}" ]; then
        ok "learned hotlist warms the cache ($cold -> $warm misses)"
    else
        no "hotlist did not reduce misses ($cold -> $warm)"
    fi

# --------------------------------------------------------------- budget ----
head_ "RAM budget"

if [ -d "$MODEL" ]; then
    if ./waste plan "$MODEL" >/dev/null 2>&1; then ok "waste plan"; else no "waste plan"; fi
    # capture first: `set -o pipefail` would otherwise propagate the
    # deliberate non-zero exit of the command under test
    refusal=$(./waste run "$MODEL" x --budget 1M 2>&1 || true)
    if printf '%s' "$refusal" | grep -q "below the model's floor"; then
        ok "a budget under the floor is refused, not swapped into"
    else
        no "under-floor budget not refused"
    fi
    if tests/check_budget.sh "$MODEL" 2>/dev/null | grep -q "^BUDGET OK"; then
        ok "peak RSS stays inside the configured budget"
    else
        no "peak RSS exceeded the budget"
    fi
else
    sk "budget checks" "no container at $MODEL"
fi

# The small model cannot catch budget accounting that is wrong in
# proportion to the model: K3 overran by 2-3 GB on scratch that Kimi-Linear
# sizes in single megabytes. Run the same check against K3 when it is here.
BIG="${BIG_MODEL:-/Users/marco/models/k3.waste}"
if [ -f "$BIG/manifest.json" ]; then
    if tests/check_budget.sh "$BIG" 32 long 2>/dev/null | grep -q "^BUDGET OK"; then
        ok "peak RSS stays inside the budget on K3 too"
    else
        no "peak RSS exceeded the budget on K3"
    fi
else
    sk "K3 budget check" "no container at $BIG"
fi

# ------------------------------------------------------------ tokenizer ----
head_ "tokenizer"

if [ -d "$MODEL" ] && command -v uv >/dev/null 2>&1 && [ -d "$SRC" ]; then
    if uv run --quiet --with tiktoken --no-project python tools/tokdiff.py \
           "$MODEL" "$SRC" 2>/dev/null | tail -1 | grep -q "identical"; then
        ok "C tokenizer matches Python tiktoken"
    else
        no "tokenizer differs from tiktoken"
    fi
else
    sk "tokenizer diff" "needs uv, a container and source weights"
fi

printf "\n\033[1m%d passed, %d failed, %d skipped\033[0m\n" "$pass" "$fail" "$skip"
[ "$fail" -eq 0 ]
